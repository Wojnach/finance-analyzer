# Independent Adversarial Pass — 2026-05-27

Reviewer: main thread (Opus 4.7 with 1M context), focused on cross-cutting
concerns not bound to a single subsystem. The 8 subsystem reviewers
spawned in parallel cover scoped findings; this pass looks at
coupling/concurrency and the prior-incident-class issues.

## Summary

- **Total findings: 6** (1 P1, 4 P2, 1 P3)
- **Top themes:** persistence-of-monotonic-state across restarts, NoneType
  propagation through gates, gate scope coverage gaps, dispatcher
  shadow-write races.
- **Biggest risk one-liner:** persisted `time.monotonic()` value of
  `sustained` debounce state lives across process restarts; on a fresh
  process the comparison can go negative (boot-aware monotonic) or huge
  (continuous QPC), causing the sustained-flip duration gate to either
  always fire or never fire — a single restart can poison the trigger
  state until the next "value changed" event resets the entry.

---

### [P1] Persisted monotonic clock across process restarts breaks sustained-flip duration gate

**File:** `portfolio/trigger.py:130-161` (`_update_sustained`) read with
`portfolio/trigger.py:521-522` (state["sustained_counts"] = sustained;
save).

**Issue:** `_update_sustained` stores `time.monotonic()` into
`state_dict[key]["_mono_start"]`. That dict is the in-memory
`sustained = state.get("sustained_counts", {})` loaded from
`data/trigger_state.json` at the top of `check_triggers`. On next
process start, `_load_state()` returns the same persisted dict
including the stale `_mono_start` value. Inside `_update_sustained`:

```python
if prev.get("value") == value:
    state_dict[key] = {
        "value": value,
        "count": prev["count"] + 1,
        "_mono_start": prev.get("_mono_start", mono_now),  # stale!
    }
...
duration_ok = (mono_now - entry["_mono_start"]) >= SUSTAINED_DURATION_S
```

`time.monotonic()`'s reference point is platform-defined and process-
dependent in CPython docs. On Windows the value is QPC-based and
typically large (≈boot uptime in seconds); restart-to-restart it can
drift by hours. On Linux it is seconds-since-boot; restart leaves the
old value massively in the future of the new process's clock domain.

Outcomes after a restart, when the persisted action is unchanged:
- New `mono_now` < persisted `_mono_start` → `mono_now - _mono_start`
  negative → `duration_ok = False` permanently for that key, until the
  ticker's action changes (which resets `_mono_start`).
- New `mono_now` >> persisted `_mono_start` → `duration_ok = True`
  immediately on the first cycle, even though the count gate
  legitimately requires 3 cycles.

The docstring says "On process restart, monotonic origin resets and the
duration gate conservatively starts fresh" — this is wrong. The state
DOES persist across restarts via the JSON dict. The reset only happens
when `prev.get("value") != value`.

**Impact:** Trigger fidelity degraded after every loop restart for any
ticker whose signal action hasn't changed across the restart. Either a
storm of premature triggers (positive elapsed-overshoot) or an outright
gap in duration-gated triggers (negative elapsed). Since the loop
restarts daily via PF-DataLoop's auto-restart and Windows
WakeUp/ForceSleep schedule, this is a regular failure mode, not a
corner case.

**Fix:** Store wall-clock ISO timestamps instead of monotonic offsets
for any state persisted to disk. Compute elapsed as
`(datetime.now(UTC) - datetime.fromisoformat(prev_iso)).total_seconds()`.
Wall-clock NTP jumps are the original concern, but for a 900-second
debounce window NTP drift is negligible and the persistence semantics
become correct. Alternative: detect process restart (compare PID like
the startup grace logic does) and wipe `_mono_start` on first call per
PID.

**Confidence:** high

---

### [P2] `None` confidence raises TypeError in ranging dampening branch

**File:** `portfolio/trigger.py:301-313`

**Issue:**
```python
conf = sig.get("confidence", 0)
ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
if (
    ticker_regime == "ranging"
    and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
    and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
):
```

`sig.get("confidence", 0)` returns the default `0` ONLY when the key is
absent. If a signal returns `{"action": "BUY", "confidence": None}` —
which several signal helpers do on edge-case branches (insufficient
data, divide-by-zero guards) — `conf` is `None` and
`None < RANGING_CONSENSUS_MIN_CONFIDENCE` raises `TypeError`. That
exception isn't caught here; it propagates up through `check_triggers`,
which propagates into the main loop body where it's caught by the
main-loop crash handler — the cycle is lost.

Same pattern in the claude-budget consensus floor block at line 322-330:
`(conf * 100) < min_pct` fails the same way when `conf is None`.

The `_flip_conf = sig.get("confidence", 0) or 0` pattern at line 398
DOES coerce None to 0 (the `or 0` part). The dampening branch is
missing that coercion.

**Impact:** Single cycle loss when any signal returns explicit
`confidence=None` AND the regime is "ranging" AND action is BUY/SELL.
Probability per cycle is low (most signals return floats), but the
crash drains the cycle and the loop falls back to exponential backoff
recovery — observable as `[CRASH] check_triggers` in agent.log.

**Fix:**
```python
conf = sig.get("confidence", 0) or 0
```
Apply to both the ranging dampening read (line 301) and the
`claude_budget` consensus floor read (line 322-324).

**Confidence:** high (pattern verified to TypeError on `None < float`)

---

### [P2] `_no_position_skip` ignores warrant holdings — could skip Layer 2 while warrants are open

**File:** `portfolio/agent_invocation.py:357-423`

**Issue:** The gate loads `load_state()` and `load_bold_state()` and
checks `state.get("holdings", {}).get(tk, {}).get("shares", 0)`. It
does NOT consult `data/portfolio_state_warrants.json` or the
`warrant_portfolio.py` accessors. The system trades XBT-TRACKER,
ETH-TRACKER, MINI-SILVER as Tier-3 instruments whose signals derive
from the underlying — yet the gate decides "no holdings → no
Claude" using only spot/equity portfolios.

Practical scenario: user holds 100 oz of MINI-SILVER 5x warrant. A
new BUY consensus fires on XAG-USD with `weighted_confidence` just
below the entry threshold. The gate sees `_shares(patient, "XAG-USD")
= 0` and `_shares(bold, "XAG-USD") = 0`, returns skip. Claude is
never invoked, so the warrant position is never re-evaluated against
the new signal posture — exactly the "manage open position" path
that Layer 2 exists to handle.

The gate is currently behind `claude_budget.no_position_skip_enabled`
(default False per the code comment), so impact today is zero. But
the default is documented to flip after rollout validation; without
fixing this first, a future enable will silence warrant management
invocations.

**Impact:** Latent — only manifests if/when
`no_position_skip_enabled` is turned on. Warrant positions could go
unreviewed through significant signal regime changes.

**Fix:** Before returning skip, also consult warrant holdings. Map
warrant ticker → underlying (XBT-TRACKER→BTC-USD, ETH-TRACKER→ETH-USD,
MINI-SILVER→XAG-USD) and treat a warrant on the underlying as a
position on the underlying for skip-gate purposes.

```python
from portfolio.warrant_portfolio import load_warrant_state
warrants = (load_warrant_state() or {}).get("holdings", {}) or {}
WARRANT_UNDERLYING = {
    "XBT-TRACKER": "BTC-USD", "ETH-TRACKER": "ETH-USD",
    "MINI-SILVER": "XAG-USD",
}
for tk in tickers:
    if _shares(patient, tk) > 0 or _shares(bold, tk) > 0:
        return (False, "")
    for w_name, underlying in WARRANT_UNDERLYING.items():
        if underlying == tk and float(warrants.get(w_name, {}).get("units", 0) or 0) > 0:
            return (False, "")
```

**Confidence:** medium (depends on whether warrant management was
explicitly out of scope for the gate; CLAUDE.md does not say)

---

### [P2] `_PER_TICKER_CONSENSUS_GATE = 0.38` hardcoded floor diverges from the global 0.47 gate

**File:** `portfolio/signal_engine.py:788-789`

**Issue:** Two related accuracy floors live in this module:

- Global signal accuracy gate: 0.47 (with 0.50 for 10K+ sample
  signals) — disables individual signals.
- Per-ticker overall-consensus gate: 0.38 — disables NON-HOLD
  consensus for tickers where the system-wide consensus accuracy is
  below 38%.

These have different rationale (one signal-vs-ticker, one whole-
consensus-vs-ticker), but the 9-percentage-point gap between them
means a ticker can have all signals individually clearing the 0.47
floor while the COMPOSITE consensus accuracy sits at 0.40 — and the
0.40 case still gets through (above 0.38) even though it's
demonstrably below the individual-signal floor.

Comment at line 786 cites AMD 24.8%, GOOGL 31.3%, META 34.2% as the
motivation. All three were force-removed from Tier-1 per CLAUDE.md
(removed Mar 15 + Apr 09). The gate justification is for tickers no
longer in the universe; the current 5 Tier-1 instruments (BTC, ETH,
MSTR, XAU, XAG) all sit well above 0.38, so the gate is a no-op
today — BUT the threshold is also too low to catch a new MSTR-class
regression (e.g., MSTR's 1d accuracy crashed to 21.9% in W15/W16
per the comment at line 826). A 0.40 → 0.39 → 0.37 slide would
trigger the gate only at the third step.

**Impact:** A new ticker degradation has to fall past 0.38 before
the consensus gate fires, which means real money at risk through the
0.38–0.47 band where individual-signal gates are already firing.

**Fix:** Either raise to 0.45 (close to the individual gate), or
make the threshold a function of the individual gate so the two
move together:
```python
_PER_TICKER_CONSENSUS_GATE = ACCURACY_GATE_FLOOR - 0.05  # always 5pp below
```
Pair with a code comment explaining why the consensus floor is below
the individual floor (composite has more samples = more confidence
in the metric).

**Confidence:** medium (depends on whether the current threshold was
chosen deliberately as "evidence-only" or as the right safety level)

---

### [P2] Fix-agent auto-dispatcher backoff defaults to permanent-disable; no resurrection path documented in CLAUDE.md

**File:** `scripts/process_pending_pickups.py` (implied — referenced in
CLAUDE.md "Auto-spawn fix agent" section but the cooldown table is
described as `30m → 2h → 12h → effectively disabled`)

**Issue:** CLAUDE.md says "if you see repeated `fix_agent_failed`
entries in the journal, the dispatcher has given up on that category
— manual investigation is required." The "effectively disabled"
state is achieved by ever-growing cooldown, with no automatic
recovery. This means:

1. A genuine transient failure in fix-agent execution (e.g., Claude
   CLI auth blip, network hiccup during the agent's tool calls) can
   ratchet the category cooldown up to 12h+ — even though the
   underlying critical_errors.jsonl entry is real and still wants
   resolution.
2. The 12h+ cooldown for that category means subsequent NEW
   critical_errors of the SAME category also wait through the
   backoff, not just the original entry.
3. The only documented reset path is `touch data/fix_agent.disabled
   && rm` — that disables ALL categories globally, not just one.

**Impact:** A small number of transient fix-agent failures can
silently disable the auto-remediation pipeline for whole categories
of critical errors (e.g., "auth_failure" category), reverting the
system to the pre-2026-04-13 manual-intervention regime. The user
won't notice until they actively read `data/fix_agent.disabled` or
re-read CLAUDE.md.

**Fix:** Add a daily reset of the per-category cooldown at the
PF-PendingPickups 08:00 CET cron (or as a separate
`scripts/reset_fix_agent_backoffs.py`). Document in CLAUDE.md that
the reset exists and runs nightly so an operator knows the
12h-effective-disable is not permanent.

**Confidence:** medium (would need to read the dispatcher itself to
confirm the backoff is per-category not global — comment-based read)

---

### [P3] Sidecar lock for JSONL append is per-file but rotation interplay is undocumented at call sites

**File:** `portfolio/file_utils.py:210-292` plus all ~20 callers of
`atomic_append_jsonl`.

**Issue:** The docstring of `atomic_append_jsonl` explains why the
sidecar lock exists and references `log_rotation.rotate_jsonl` as
the partner — both must use the same lock or rotation can swallow
appends that landed mid-rotate. This is correctly implemented.

But the ~20 callers across the codebase (signal_log, claude_invocations,
critical_errors, telegram_messages, accuracy_snapshots, layer2_journal,
metals_trades, etc.) get no co-located reminder that any OTHER code
writing to those same files MUST also use `atomic_append_jsonl` or
go through `jsonl_sidecar_lock`. A new contributor adding a one-off
`with open(JSONL_FILE, "ab") as f: f.write(...)` would bypass the
lock and reintroduce the torn-line risk the docstring warns about.

Same risk for code that does a transactional read-modify-write on a
JSONL file (e.g., `prune_jsonl` is correctly inside the lock — but
the pattern is implicit, not enforced).

**Impact:** Latent. No current bug found, but the architectural
invariant is undocumented at the call site. Next torn-line incident
likely to come from someone bypassing the helper.

**Fix:** Add a one-line header comment at the top of every
JSONL-write call site referencing the lock contract:
```python
# Append-only — use atomic_append_jsonl(), never raw open(..., "ab")
# Reading + rewriting? Use jsonl_sidecar_lock(path).
```
Alternative: add a `_warn_if_raw_write` linter rule.

**Confidence:** low (defensive-coding concern, no concrete
near-term bug)

---

## Out-of-scope observations (not findings)

- The auth-error cooldown in `agent_invocation.py:762-789` correctly
  walks back `recent[-50:]` and breaks on first non-skipped entry —
  this means 50+ consecutive skips never find an older auth_error;
  in practice the cooldown only ratchets back from 30 min, so the
  worst case is one extra invocation per long skip run. Acceptable.
- `_drawdown_block_pct = 50.0` uses strict `>` (not `>=`) — exactly
  50% drawdown does NOT block. Aligns with memory/feedback_risk_tolerance
  which says user accepts up to 50%.
- `file_utils.atomic_write_json` resolves symlinks before write — this
  is essential because `config.json` is a symlink to an external file.
  Verified safe.
