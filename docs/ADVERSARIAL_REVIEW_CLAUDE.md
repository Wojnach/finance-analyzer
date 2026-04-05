# Adversarial Review — Claude

**Reviewer:** Claude (Opus 4.6 1M)
**Date:** 2026-04-05
**Scope:** finance-analyzer codebase, partitioned into 8 subsystems
**Baseline:** HEAD = `6c3f154` (main branch, 2026-04-04)

## Framing

This is an independent adversarial review written BEFORE reading any Codex output.
The goal is to challenge design choices and identify where the system is most likely
to fail under real trading conditions, not to lint the code for style.

Severity legend:
- **CRITICAL**: operational trading impact, money at risk, or data corruption path
- **HIGH**: systemic reliability issue, silently wrong output, or recovery-blocking bug
- **MEDIUM**: real design weakness but bounded blast radius
- **LOW**: worth noting but not urgent

Confidence legend (against a finding being real and matters):
- **H** = high confidence (verified by reading code and tracing the failure path)
- **M** = medium (plausible but requires runtime verification)
- **L** = low (speculative — flagging for debate)

---

## Subsystem 1 — signals-core

Files reviewed: `signal_engine.py` (1637 LOC), `accuracy_stats.py`, `outcome_tracker.py`,
`signal_weights.py`, `train_signal_weights.py`.

### Finding 1.1 — CRITICAL [H]: Regime-gated signals cannot recover through data (dead-signal trap)
**Where:** `signal_engine.py:1339-1341` + `outcome_tracker.py:123-129` + `accuracy_stats.py:103-104`
**What:** When a signal is on the `REGIME_GATED_SIGNALS` list for the current regime
(e.g., `trend` in `ranging`), its vote in the `votes` dict is force-set to HOLD **before**
the dict is passed to `log_signal_snapshot` via `extra._votes`. The outcome tracker
records these HOLD votes; the accuracy computation skips HOLDs (`if vote == "HOLD":
continue`). The consequence is that as long as a signal is on the regime gate list,
its accuracy data in that regime cannot accumulate — it has zero recorded non-HOLD
votes for that regime and therefore no way to prove it works.

**Why it matters:** The regime gate lists (lines 169-221) were built from dated
audit snapshots (`"2026-04-02 audit"`, `"2026-04-04"`). The comment explicitly
acknowledges each entry is a point-in-time observation. But the design offers
no mechanism to update those lists from fresh data, because the very act of
gating destroys the data needed to re-evaluate. Over time, the gate list drifts
into stale pessimism and the system becomes progressively LESS adaptive, not more.

**Argument:** This violates a core principle the system otherwise follows —
evidence-based adaptation. Accuracy-based gating at line 606 has a recovery path
(votes are still logged, only the consensus contribution is skipped). Regime
gating does NOT; votes are rewritten to HOLD before logging.

**Fix shape (not prescriptive):** Log the **raw** pre-gate vote alongside the
gated vote. Gate the consensus contribution, not the logged vote. Then a weekly
"shadow accuracy" job can identify gate-list entries that would now flip back.

### Finding 1.2 — HIGH [H]: "Never invert sub-50% signals" contradicts the data for biased signals
**Where:** `signal_engine.py:32-36, 606-607`
**What:** The accuracy gate at 45% force-HOLDs signals below threshold. The
comment on line 33-35 argues that inverting noise "just produces different noise
with whiplash as accuracy oscillates around 50%." This is correct *only* for
signals whose errors are genuinely random, i.e., no directional bias.

The codebase itself documents signals that violate this assumption. Line 291-293:
```
# Activity rate cap: targets volume_flow (83.1% activity, 49.2% accuracy).
```
A signal that activates 83% of the time with 49% accuracy cannot be random noise —
a random 50/50 predictor cannot hit 83% activation without directional bias. That
signal has real information, just in the wrong direction. Force-HOLD discards it
entirely; inversion would capture it.

**Why it matters:** The design decision is hard-coded as policy, but the codebase
already identifies at least one signal (volume_flow) where the policy is strictly
worse in expectation than inversion. Others in the `CORRELATION_GROUPS`
trend_direction {ema, trend, heikin_ashi, volume_flow} share "permanent SELL lean"
per the comment on line 480 — these are biased, not noisy.

**Note on prior guidance:** The user's auto-memory includes a rule "No signal
inversion — never invert sub-50% signals, it's noise not alpha, gate them
instead." I respect that as a deliberate decision but surface the contradiction
here because the codebase has since introduced signals where the noise-vs-bias
distinction is empirically clear. A per-signal classification (noise/biased)
with gate for noise and **conditional** invert for biased would be more
principled than a blanket policy either way.

### Finding 1.3 — HIGH [H]: Market-health penalty creates a structural SELL bias at market bottoms
**Where:** `signal_engine.py:1564-1578`
**What:** The market-health penalty is explicitly asymmetric: "Only affects BUY;
SELL and HOLD pass through." When market health is "unhealthy" (distribution days,
broken FTD, bad breadth), BUY confidence is discounted but SELL confidence is not.

**Why it matters:** The exact moment "market health" is most unhealthy is near
capitulation lows — which is also when contrarian BUY alpha is highest. By
discounting BUYs asymmetrically in that regime, the system systematically reduces
its participation in mean-reversion bounces and leaves SELL bias intact at the
worst possible time. Over a long bear market, this compounds into structural
underperformance on recovery.

**Fix shape:** Either symmetric (penalize both directions in bad health) or
regime-conditional (invert the penalty in the "capitulation" sub-regime of
unhealthy markets).

### Finding 1.4 — HIGH [H]: `id(df)` ADX cache can return stale values after GC
**Where:** `signal_engine.py:23-27, 774-809`
**What:** The cache keys on `id(df)`. Python re-uses object IDs after GC. Comment
at line 24 claims "naturally expire when DataFrames are garbage-collected between
cycles" — this is only safe if every DataFrame id is used **exactly once** in
the cache's lifetime, which the 200-entry LRU-like clear does not guarantee.

If any code path retains a reference to a DataFrame across cycles (cache,
closure, thread-local), the old id may return cached ADX for a different
DataFrame with the same id. This feeds the volume/ADX gate at line 853-869 which
can force HOLD on the wrong basis.

**Why it matters:** Silent wrong-ADX doesn't crash anything — it quietly
mis-classifies trend strength and forces HOLDs on signals that should fire, or
passes weak trends through gates they shouldn't pass. No alert, no diagnostic.

**Fix shape:** Hash on `(ticker, df.index[-1], len(df))` tuple instead. Or use
`weakref.WeakValueDictionary` keyed on id so entries auto-purge on GC.

### Finding 1.5 — MEDIUM [H]: Unanimity penalty reduces confidence but never flips to HOLD
**Where:** `signal_engine.py:919-933`
**What:** Comment states "90%+ confidence has 28-32% actual accuracy across all
horizons." The code responds by multiplying confidence by 0.6x at 90%+ agreement,
0.75x at 80-90%. This takes a 1.0 consensus down to 0.6 — **still above the 0.5
execution threshold**. The trade still fires, at a cosmetic lower confidence.

If the empirical observation (90%+ → 28-32% accuracy) is correct, unanimity is a
**contrarian** signal, not a confidence reduction. The correct response is
HOLD (it's too good to be true) or INVERT (contrarian bounce plays). A
proportional discount preserves the same decision at lower confidence, which is
exactly wrong if the data says unanimity = edge reversal.

### Finding 1.6 — MEDIUM [H]: Global confidence cap at 0.80 creates pile-up at the boundary
**Where:** `signal_engine.py:1627-1630`
**What:** `conf = min(conf, 0.80)`. Comment: ">80% confidence is anti-correlated
with accuracy at every horizon (70-80% bucket is the best performing at 57-59%
actual accuracy)."

This fix is wrong-shaped. If 80%+ is anti-correlated with accuracy, the cap
doesn't fix the problem — it just compresses many high-confidence signals into
exactly 0.80, creating an artificial pile-up at the cap that operators will then
mis-interpret as "very confident" when it's actually the worst-calibrated bucket.

**Fix shape:** Either non-linear decay (e.g., `conf * (1 - max(0, conf-0.70) * 2)`
bending the curve) or inversion at the bucket boundary. A hard cap is the
weakest response.

### Finding 1.7 — MEDIUM [H]: Correlation-group leader gating silences the whole group when leader is borderline
**Where:** `signal_engine.py:569-582`
**What:** `_GROUP_LEADER_GATE_THRESHOLD = 0.47`. When the highest-accuracy signal
in a correlation group falls below 47%, the **entire group** is silenced. For
the `macro_external` group {fear_greed, macro_regime, structure, sentiment,
news_event}, that's 5 signals.

**Why it matters:** The comment justifies this with current accuracy stats
(fear_greed 25.9%, sentiment 46.8%, news_event 29.5%) and says "even the leader
is near noise." But:
1. The leader is determined among **actively voting** signals in the cycle
   (line 554-556). In any cycle where the leader happens to HOLD, a weaker
   signal in the group becomes the new leader — potentially flipping the gate on/off
   cycle-to-cycle based on vote presence rather than accuracy.
2. If the leader's samples fluctuate around `ACCURACY_GATE_MIN_SAMPLES = 30`,
   the gate toggles unstably.

Silencing 5 signals based on 1 signal's stats is high-blast-radius for a
threshold-sensitive decision.

### Finding 1.8 — MEDIUM [M]: `HORIZON_SIGNAL_WEIGHTS` static dict is drift-prone
**Where:** `signal_engine.py:233-287`
**What:** Hard-coded per-signal horizon multipliers with explicit dates in
comments ("2026-03-29", "2026-04-02"). The dynamic path
`_compute_dynamic_horizon_weights` exists but returns `{}` when samples are
below `_DYNAMIC_HORIZON_MIN_SAMPLES = 50` or ratios fall within the ±10%
deadband — in which case it falls back to the static dict.

For rarely-voting signals that rarely hit 50 samples on a given horizon, the
static dict IS the permanent weight. Those signals never get updated from
fresh data.

**Fix shape:** Lower the min-sample bar for static-overwrite (e.g., 20) and
age-out the static dict entries after N days.

### Finding 1.9 — MEDIUM [M]: Dynamic MIN_VOTERS (stage 4) can force HOLD after stage 3 already cleared conviction
**Where:** `signal_engine.py:901-917`
**What:** Stage 4 in `apply_confidence_penalties` requires 3/4/5 active voters
by regime. In ranging regime the minimum is 5. The **earlier** min-voters check
at line 1358-1368 used `MIN_VOTERS_CRYPTO = 3` or `MIN_VOTERS_STOCK = 3`. So a
crypto signal in `ranging` regime passes the top-level MIN_VOTERS=3 check, goes
through weighted consensus, gets a non-HOLD action, then hits stage 4 min=5 and
reverts to HOLD with conf=0.

This is not strictly wrong but it's **opaque** — the decision is made by two
different thresholds in two different places. Operators reading the decision
journal see "reached weighted consensus then forced HOLD by stage 4" with no
explanation of why the threshold jumped from 3 to 5.

**Fix shape:** Collapse into one authoritative `required_voters(regime)` function
called once, before consensus computation.

### Finding 1.10 — MEDIUM [H]: Per-ticker accuracy override is sample-threshold whiplash
**Where:** `signal_engine.py:1450-1473`
**What:** Per-ticker accuracy replaces global when `total >= 30`. For a signal
that's accumulating exactly around 30 samples on a ticker, whether per-ticker
or global is used flips cycle-by-cycle as old samples age out of windows and
new samples arrive. This changes the effective signal weight instant-by-instant
with no smoothing.

**Fix shape:** Use a blend `w_per_ticker = min(1.0, n/60)` instead of a hard
threshold. Or require 30+ samples **and** stability for 3 consecutive cycles
before switching.

### Finding 1.11 — MEDIUM [H]: `_weighted_consensus` produces a decision the outer code may discard
**Where:** `signal_engine.py:1515-1525`
**What:** `_weighted_consensus` computes a full weighted decision (BUY/SELL +
confidence) based on accuracy, regime multipliers, horizon weights, correlation
penalties, gated signals log, etc. Then at line 1523-1525 the outer code applies
the MIN_VOTERS and core-gate checks AGAIN and may overwrite everything to HOLD.
The work in `_weighted_consensus` is thrown away.

This is not a bug, but it is a source of confusion: the journal records both
`_weighted_action` and the final `action`, and operators debugging HOLDs have
seen plenty of reports where `_weighted_action=BUY, _weighted_confidence=0.67`
but `action=HOLD`. The answer is the min-voters re-check. This is underdocumented.

### Finding 1.12 — MEDIUM [H]: Earnings gate silently disables on any exception
**Where:** `signal_engine.py:1582-1593`
**What:** The earnings gate is wrapped in `try/except Exception: pass`. If
`earnings_calendar.should_gate_earnings()` raises for any reason (network,
file corruption, schema mismatch), the gate silently does nothing and the
signal proceeds as if no earnings were scheduled. A stock with earnings tomorrow
could be traded.

**Why it matters:** Earnings gates exist because post-earnings moves are
frequently catastrophic for signal-based systems (gap-up/gap-down that ignore
technicals). The gate silently failing **exactly when the earnings calendar is
degraded** is the worst possible failure mode.

**Fix shape:** On exception, either (a) default to HOLD as a safe default, or
(b) alert via health system and still force HOLD until recovered. Silent
pass-through is not acceptable here.

### Finding 1.13 — LOW [H]: Sentiment hysteresis cache is global (module-level) — one ticker can read another's state
**Where:** `signal_engine.py:58-100`
**What:** `_prev_sentiment` is a module-level dict shared across all tickers.
Access is protected by `_sentiment_lock`. Thread safety is fine. But the
semantics: `_get_prev_sentiment(ticker)` reads per-ticker state, so the global
shape is correct. Low-risk finding, but: if a new ticker is added and
`sentiment_state.json` doesn't have an entry for it, there's no issue — the
`.get(ticker)` returns None and hysteresis defaults to the lower threshold. Fine.

### Finding 1.14 — LOW [H]: `_load_prev_sentiments` early-returns under lock without loading when flag is set
**Where:** `signal_engine.py:67-84`
**What:** The `if _prev_sentiment_loaded: return` check sits inside the lock,
so it's race-safe. But there's a subtle issue: if the first load fails (exception
at line 82), `_prev_sentiment_loaded` is still set to True at line 84, and
subsequent calls never retry. An operator who fixes the JSON file on disk
mid-run would need to restart the process to pick up the data.

### Finding 1.15 — MEDIUM [M]: `train_signal_weights.py` walk-forward windows shrink silently on small data
**Where:** `train_signal_weights.py:134-138`
**What:** `train_window=min(720, len/3)`, `test_window=min(168, len/6)`. For
`len(signals_df) = 100`, you get train=33, test=16. That's statistically
meaningless walk-forward validation — the windows are too small to estimate
anything, and the reported `avg_oos_corr` will be dominated by noise.

No minimum-sample guard, no warning in the log when windows collapse. A user
reading the "walk-forward OOS corr" output cannot tell whether it's a real
validation or a noise artifact.

**Fix shape:** Guard: `if len(signals_df) < 500: return {"model": ..., "walk_forward": {"error": "insufficient_data"}}`.

### Architectural critique — signals-core

The overall design is **sophisticated but fragile**: many interlocking gates
(regime, accuracy, correlation, horizon, activity, unanimity, market health,
earnings, dynamic min-voters) each driven by recent audit snapshots hardcoded
as constants. The aggregate effect is that the system is tuned for a very
specific moment in market history (late March to early April 2026). Every gate
has a dated comment justifying it.

When the market regime shifts — which is inevitable — most of these gates will
become wrong simultaneously. The architecture does not have a coherent story for
how gates update themselves, who audits them, or when the system re-evaluates
whether a gate is still needed. The regime-gated-signal trap (Finding 1.1) is
the most critical instance of this: once a signal is gated in a regime, the
data needed to un-gate it can never exist.

The **single most important improvement** this subsystem could make is: maintain
a "shadow" raw-vote log alongside the gated-vote log, so that gate recovery
paths exist automatically.

---

## Subsystem 2 — orchestration

Files reviewed: `main.py` (1041 LOC), `agent_invocation.py`, `trigger.py`,
`market_timing.py`, `autonomous.py`, `reporting.py`, `claude_gate.py`,
`loop_contract.py`, `health.py`.

### Finding 2.1 — CRITICAL [H]: Singleton lock silently no-ops on non-Windows
**Where:** `main.py:39-55`
```python
try:
    import msvcrt
except ImportError:
    msvcrt = None
...
def _acquire_singleton_lock():
    if msvcrt is None:
        return True  # <-- silent pass-through
```
**What:** On any non-Windows host, `msvcrt` is None and the lock function
**returns True unconditionally** — meaning zero singleton enforcement.

**Why it matters:** The user's environment is WSL on Windows. If the loop is
launched in WSL (which uses Linux Python, no msvcrt) **alongside** the Windows
scheduled task (PF-DataLoop, which does have msvcrt), you get two loops writing
to the same `portfolio_state.json`, `signal_log.jsonl`, `trigger_state.json`,
`sentiment_state.json`, and every other data file, racing each other with
atomic writes. The atomic writes prevent corruption within each write, but
they do not prevent **interleaved** state — each loop reads, modifies, and
writes, clobbering the other's update.

This is especially dangerous during development when a dev iterates from WSL
and forgets that the Windows scheduled task is still running. The failure mode
is silent: no alert, no error, just slowly corrupted state.

**Fix shape:** Use `fcntl.flock` as a cross-platform alternative when msvcrt
is unavailable. Or, at minimum, fail loudly: `raise RuntimeError("Singleton
lock unsupported on this platform — risk of concurrent instances")`.

### Finding 2.2 — HIGH [H]: `_startup_grace_active` is a module-level mutable global with test-suite leakage
**Where:** `trigger.py:40, 98, 134, 138`
**What:** `_startup_grace_active = True` is set at module import, flipped to
`False` after the first `check_triggers` call. This has two issues:

1. **Test isolation**: in the pytest suite, many tests import `trigger.py` via
   `from portfolio.main import check_triggers` (re-exported) or directly. The
   first test to invoke `check_triggers` consumes the grace period; subsequent
   tests never hit it. Any test that EXPECTS a trigger to fire on first call
   relies on other test ordering to have consumed the grace. A test suite
   re-ordering (xdist, `-k` selection) could silently change behavior.

2. **Restart semantics**: the grace period relies on PID comparison
   (`saved_pid != current_pid`). On PID reuse after a crash, the comparison
   could match by coincidence and silently skip grace — or mismatch on a
   process that wasn't actually restarted (e.g., a hot-reload) and silently
   activate grace. The logic is **presence/absence** when it should be
   **monotonic restart-count**.

**Fix shape:** Track a monotonically-increasing `boot_id` in the state file,
written at startup via `uuid.uuid4()`. Grace is active when the persisted
boot_id differs from the in-memory one.

### Finding 2.3 — HIGH [H]: Trade-detection catches only `(KeyError, AttributeError)` — ValueError crashes propagate
**Where:** `trigger.py:76-93`
```python
except (KeyError, AttributeError) as exc:
    logger.warning("Failed to parse portfolio file %s: %s", pf_file, exc)
```
**What:** If `portfolio_state.json` is corrupted (truncated write during a
crash, partial overlay, bad JSON), `load_json` may raise `json.JSONDecodeError`
(which is a subclass of `ValueError`). That exception propagates, crashes
`check_triggers`, which crashes `run()`, which triggers `_crash_alert` and
`_crash_sleep`. The loop then backs off 10s→5min. Over N consecutive failed
reads (e.g., if the file is permanently corrupt until human repair), the loop
effectively stops for the full backoff window.

**Why it matters:** The crash-recovery path is triggered by a **recoverable**
file-level issue that could have been handled in place (treat as "no trade
detected" and move on).

**Fix shape:** `except Exception: continue` with a logged warning, and raise
a health-check alert if the same file fails to parse for >5 cycles.

### Finding 2.4 — HIGH [H]: Multi-agent specialists block the main loop for up to 150s
**Where:** `agent_invocation.py:249-260`
```python
procs = launch_specialists(ticker, reasons)
if procs:
    results = wait_for_specialists(procs, timeout=150)
```
**What:** When multi-agent mode is enabled, the main loop **blocks** for up
to 150 seconds waiting for 3 subprocess specialists. Inside a 60-second cycle,
this means 2.5 cycles are missed. The critical cycle-level ThreadPoolExecutor
work (signal generation for 20 tickers) doesn't start until specialists return.

**Why it matters:** During the exact moments when multi-agent is most useful
(volatile market, new consensus), blocking means missing the next 2-3 cycles
of fresh signal data. A 3% price move could happen during the block and the
system wouldn't detect it until specialists return.

**Fix shape:** Move the wait to a background thread/future. The synthesis
prompt is built from completed specialist outputs anyway; the main loop can
continue collecting signals and check specialist completion on the next
cycle, just like the single-agent case.

### Finding 2.5 — HIGH [H]: Stack-overflow detection hardcodes Windows exit code
**Where:** `agent_invocation.py:35, 551`
```python
_STACK_OVERFLOW_EXIT_CODE = 3221225794  # 0xC00000FD on Windows
```
**What:** This is the Windows STATUS_STACK_OVERFLOW code. On Linux/WSL,
Node.js stack overflows produce either exit 134 (SIGABRT) or exit 139 (SIGSEGV).
The auto-disable-after-N-stack-overflows logic at line 142-148 will never
trigger on Linux, even though stack overflows still happen.

**Why it matters:** The reason stack-overflow detection exists at all is
because of a real, recurrent crash pattern. Falling back to Linux development
environments (WSL) loses the protection.

**Fix shape:** Detect cross-platform: `exit_code in (3221225794, 134, 139)`
plus a stderr match on "stack overflow" or "out of memory" to be safe.

### Finding 2.6 — HIGH [H]: Agent subprocess uses `--bare` but parent env strips `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT`
**Where:** `agent_invocation.py:292-296`
**What:** The comment says "Strip Claude Code session markers to avoid nested
session error". This works but the coupling is brittle: the Claude CLI can
change these env var names in a future version, breaking the isolation
silently. There's no version pinning on `claude`, no test that validates the
subprocess actually started without nesting.

**Why it matters:** If Claude CLI ever changes its env-var names, every
Layer 2 invocation will fail with "nested session" and the subprocess will
exit with a non-zero code. The system will log "L2 FAILED" and eventually
auto-disable after stack-overflow accumulation (even though the issue isn't
stack overflow). Recovery path is obscure.

**Fix shape:** Call `claude --version` during config validation; pin
compatible versions in config. Add an e2e test that launches a `claude -p
"echo hi" --bare` and checks for success.

### Finding 2.7 — HIGH [H]: `check_triggers` modifies `state["last_checked_tx_count"]` even when trade detection partial-fails
**Where:** `trigger.py:82-92`
**What:** The loop updates `new_tx_counts[label] = current_count` for each
portfolio file successfully read. Files that fail to parse are skipped and
NOT added to `new_tx_counts`. Then at line 91: `if new_tx_counts: state[...] =
new_tx_counts` — this **replaces** the old dict, discarding any labels that
weren't re-read this cycle.

**Why it matters:** If `portfolio_state.json` parses but `portfolio_state_bold.json`
doesn't, the bold state's old tx_count is lost from `last_checked_tx_count`.
On the next successful read, `prev_count = last_checked_tx.get("bold",
current_count)` falls back to `current_count`, and any trade between the old
known-good read and now is MISSED (not detected as a new trade).

**Fix shape:** Merge dict instead of replacing: `state["last_checked_tx_count"].update(new_tx_counts)`.

### Finding 2.8 — HIGH [H]: Autonomous mode always records HOLD regardless of consensus
**Where:** `autonomous.py:119-122`
```python
decisions = {
    "patient": {"action": "HOLD", "reasoning": patient_reasoning},
    "bold": {"action": "HOLD", "reasoning": bold_reasoning},
}
```
**What:** Autonomous fallback (Layer 2 disabled) always writes HOLD in the
journal for both strategies, regardless of the upstream consensus. The
`_ticker_prediction` function computes BUY/SELL outlook and conviction, but
the decisions dict discards that and commits to HOLD.

**Why it matters:** The journal is used by `reflection.py`, by the dashboard,
by `trading_insights.md` generation, and by future Layer 2 invocations as
"previous context". If autonomous mode runs for hours and records all-HOLD
journal entries, the reflection analysis will observe "we predicted HOLD
correctly X% of the time" — which is trivially near-100% (HOLD has no
directional edge to be wrong about) and poisons any learning signal.

**Design question:** is autonomous mode a "passive observer" or a
"decision engine without execution"? The code says both. If passive, the
journal should mark these entries as `"source": "autonomous_observer"` and
exclude them from accuracy/reflection stats. If decision-engine, the actions
should reflect the actual recommended direction (BUY/SELL/HOLD) and be
tracked separately from executed trades.

### Finding 2.9 — HIGH [M]: EU market open hardcoded to 07:00 UTC, no CET/CEST DST adjustment
**Where:** `market_timing.py:8`
```python
MARKET_OPEN_HOUR = 7  # ~Frankfurt/London open
```
**What:** Frankfurt/London open at 08:00 local time, which is:
- 07:00 UTC in winter (CET, UTC+1) ✓
- **06:00 UTC** in summer (CEST, UTC+2) ✗

The constant is never adjusted for EU DST. In summer, the EU market opens
one hour before the loop switches to "market open" state (line 140-141). The
first hour of EU market trading is treated as "market closed" → 120s loop
interval, no US stocks fetched, no T3 reviews.

**Why it matters:** The system underweights the EU morning session in summer
months. Silver and gold have meaningful EU session moves; they'd be observed
on 2-minute cadence instead of 1-minute for the first hour.

**Note on US DST:** The US DST logic is correct (`_is_us_dst` computes
second Sunday of March / first Sunday of November). The bug is specifically
the EU side being hardcoded.

**Fix shape:** Compute `_eu_market_open_hour_utc(dt)` similar to
`_market_close_hour_utc`. EU DST: last Sunday of March through last Sunday of
October.

### Finding 2.10 — HIGH [M]: Agent completion detection is timestamp-diffing, not subprocess return
**Where:** `agent_invocation.py:460-495`
**What:** `check_agent_completion` determines success/failure via:
1. `exit_code` (exit status) — reliable
2. `journal_written` = new timestamp in `layer2_journal.jsonl` vs baseline
3. `telegram_sent` = new timestamp in `telegram_messages.jsonl` vs baseline

The journal/telegram checks are **best-effort** comparisons. If the agent
writes the journal but the file read races with a concurrent truncation or
rotation, the timestamp may appear unchanged and journal_written = False.
Status then becomes "incomplete" even though the agent actually succeeded.

**Why it matters:** The status feeds `get_completion_stats` which powers the
dashboard's "Layer 2 success rate" metric. False "incomplete" readings drag
down the reported success rate and trigger operator worry. The remediation
(investigating "why did the agent not write a journal") leads to a ghost chase.

**Fix shape:** Have the agent write a sentinel file on exit (`.agent_done`
with success/fail + timestamp). The parent reads the sentinel atomically —
no JSONL tailing needed.

### Finding 2.11 — MEDIUM [H]: `_run_post_cycle` swallows per-task failures without rate-limiting alerts
**Where:** `main.py:241-350`
**What:** Every post-cycle task (market_health, daily_digest, message_throttle,
alpha_vantage, local_llm_report, metals_precompute, oil_precompute, jsonl_prune,
fin_evolve, crypto_scheduler, signal_postmortem) is wrapped in `_track()` with
a try/except logger.warning. On failure, the task is skipped and a warning is
logged. There's no cumulative health signal: if the same task fails 100 cycles
in a row, there's no alert, just 100 warning log lines.

**Fix shape:** Track consecutive failure count per task in a health dict.
Alert Telegram after 10 consecutive failures of any specific post-cycle task.

### Finding 2.12 — MEDIUM [H]: `_sleep_for_next_cycle` silently drops cycles when overrun
**Where:** `main.py:780-791`
**What:** If cycle N takes longer than the interval, cycle N+1 starts
immediately (remaining < 0). The loop logs a warning but continues. If cycles
keep overrunning, they run back-to-back without any cooldown, potentially
starving rate-limited APIs (Binance, Alpaca, Alpha Vantage).

**Why it matters:** Under heavy load (restart after outage + backfill work +
ML retraining post-cycle), the loop can enter a sustained overrun state where
the rate limiters start throttling inbound calls, causing more cycles to
over-run — a positive feedback loop into degraded operation.

**Fix shape:** When cycle overruns, skip the next N-1 scheduled starts and
recover cadence, logging how many cycles were dropped.

### Finding 2.13 — MEDIUM [H]: `classify_tier` can fire multiple T3 invocations on clock skew
**Where:** `trigger.py:293-305`
**What:** `hours_since = (time.time() - last_full) / 3600`. If system clock
jumps forward (NTP sync after a suspend/resume, manual change), `hours_since`
could jump from 2.5 → 20.5 in a single call. Any tick during the jumpy window
fires T3 again until `last_full_review_time` is updated. Conversely, a backward
clock jump makes T3 impossible for hours.

**Fix shape:** Use `time.monotonic()` for intervals and `time.time()` only for
wall-clock display. Alternatively, cap `hours_since` at a sanity upper bound
(e.g., 48h).

### Finding 2.14 — MEDIUM [M]: Price threshold and F&G triggers have no per-ticker cooldown
**Where:** `trigger.py:193-215`
**What:** Price moves ≥2% fire on every cycle that still satisfies the
threshold. There's no "already triggered for this move" state — if the
baseline is updated at line 252-263 each trigger, the NEXT cycle's baseline
is the post-move price, so the 2% check is relative to the new baseline. That's
OK for preventing repeat triggers on the SAME move, but a whipsaw ticker
oscillating across the 2% line will trigger twice per cycle (once up, once
down). SUSTAINED_CHECKS only applies to signal flips, not price triggers.

### Finding 2.15 — MEDIUM [H]: Health safeguard checks are on a 100-cycle schedule — that's ~100 minutes
**Where:** `main.py:685-714`
**What:** Dead signal detection and outcome staleness check fire every 100
cycles. At 60s cadence that's 100 minutes. A stuck outcome backfill could run
for 99 minutes before detection. Worse, at "market closed" cadence (120s), it
becomes 200 minutes (~3.3h).

**Fix shape:** Run these checks on a **wall-clock** schedule (every 1 hour)
independent of cycle cadence.

### Architectural critique — orchestration

The orchestration layer has **accumulated** protection mechanisms (singleton
lock, crash backoff, tier classification, perception gate, grace period,
stack-overflow auto-disable, multi-agent mode, timeout handling) without a
unified error budget model. Each mechanism is locally correct but globally
hard to reason about. When an operator asks "why didn't Layer 2 fire on that
obvious trigger?", the answer can be any of:

1. `_is_agent_window` returned False (off-hours)
2. `_agent_proc` is still running from a previous invocation
3. `_consecutive_stack_overflows >= 5` (auto-disabled)
4. `perception_gate.should_invoke` returned False
5. Startup grace period was active
6. Layer 2 multi-agent launch failed silently
7. Crash backoff sleep is active after an earlier crash
8. Market timing says "closed" even though EU market is open (DST bug)

There is no **single place** that answers "why did Layer 2 skip this cycle?"
with all possible reasons enumerated. A debug endpoint that returns this
decision tree would dramatically reduce operator confusion.

The design principle violated here is: **observability of silent decisions**.
The system makes many silent decisions to skip/defer/gate/fall-through, each
logged at INFO level in different module loggers. Consolidating those into a
single `last_skip_reason` state field (and a Telegram alert when the skip
pattern is unusual) would make the system self-explaining.

---

## Subsystem 3 — portfolio-risk

Files reviewed: `portfolio_mgr.py`, `trade_guards.py`, `risk_management.py`,
`equity_curve.py`, `monte_carlo_risk.py`, `kelly_sizing.py`.

### Finding 3.1 — CRITICAL [H]: `load_state` silently regenerates defaults on corrupt JSON — wiping transaction history
**Where:** `portfolio_mgr.py:39-44, 26-36`
**What:** `load_state` calls `load_json(default=None)`. If the file is missing
or the JSON is corrupted, `load_json` returns None. `load_state` then returns
`{**_DEFAULT_STATE, "start_date": ...}` — a **fresh blank state**. The next
`save_state` call atomically writes this blank state over the (potentially
only-partially-corrupted) file, permanently destroying transaction history.

**Why it matters:** `atomic_write_json` guarantees we don't leave a half-written
file, but it does NOT guarantee the file is readable after an OS crash or
disk-full event. A partial write that `atomic_write_json` catches (exception
during write) is safe, but a successful write of corrupted content (rare but
possible on NTFS with antivirus scanning, shadow copies, or backup software
mid-write) is not. The first successful `load_state → save_state` round after
the corruption is the moment of permanent data loss.

**Fix shape:** On load failure, raise a loud exception and refuse to `save_state`
until explicitly overridden. Keep a rolling backup: `portfolio_state.json.bak`
renamed to `.bak2` on each save. Recovery = copy `.bak` back.

### Finding 3.2 — HIGH [H]: `portfolio_value` returns cash-only on invalid fx_rate — reported P&L discontinuity
**Where:** `portfolio_mgr.py:64-67`
**What:** `if not isinstance(fx_rate, (int, float)) or fx_rate <= 0: return
state.get("cash_sek", 0)`. When `fetch_usd_sek` fails transiently (caught and
returns None or 0), portfolio value reporting drops to cash-only for that
cycle. For a portfolio with 400K SEK in holdings and 100K cash, this reports
a 75% portfolio value drop — which then feeds the drawdown check, triggers
Telegram alerts, and may activate the circuit breaker.

**Why it matters:** A transient network failure to fx_rates.py produces a
"catastrophic loss" display that's entirely spurious. Operators on alert may
take unnecessary action.

**Fix shape:** On fx_rate failure, use the **last known good** fx_rate from
cache (with a freshness note) instead of silently zeroing out holdings.

### Finding 3.3 — HIGH [H]: 2×ATR stop-loss does not account for leverage (contradicts user rule)
**Where:** `risk_management.py:184`
```python
stop_price = entry_price * (1 - 2 * atr_pct / 100)
```
**What:** For a 5x warrant where `atr_pct` is measured on the warrant itself
(not the underlying), 2×ATR is equivalent to -10% on the warrant, which is
-2% on the underlying. That's a **tight** stop that will be triggered by
ordinary noise during the trading day.

The user's auto-memory explicitly says: `"5x certs need -15%+ stops, not -8%,
to survive intraday wicks"`. The current 2×ATR rule is closer to -8% on
medium-volatility warrants, which is **exactly what the user said does NOT work**.

**Why it matters:** The user has already learned this lesson through losses.
The code does not encode the learning. Any future warrant position recorded
in portfolio_state_warrants.json is subject to the tight-stop whipsaw.

**Fix shape:** Detect leverage from `instrument_profile.py` (if it records
leverage per warrant) and widen the stop proportionally:
`stop_mult = max(2, leverage * 1.5)`.

### Finding 3.4 — HIGH [H]: Drawdown check scans full unbounded JSONL on every call
**Where:** `risk_management.py:97-110`
**What:** `portfolio_value_history.jsonl` is not pruned by the `jsonl_prune`
post-cycle step (only invocations/layer2_journal/telegram are pruned — see
`main.py:313`). Drawdown check opens the file and reads every line to find
the max. At 1 entry per cycle × 60s cadence × 30 days = 43,200 lines. Not
catastrophic today, but growing without bound. Also: the linear scan runs on
every `check_drawdown` call, which runs every cycle when drawdown checks fire.

**Fix shape:** Add `portfolio_value_history.jsonl` to the prune set. Or cache
the running peak in `trigger_state.json` and update it on each log entry.

### Finding 3.5 — HIGH [H]: Trade guards emit "warning" severity universally — no "block" enforcement
**Where:** `trade_guards.py:93-166`
**What:** Every guard result returns `"severity": "warning"`. The interface
supports `"block"` per the docstring at line 64, but no code path ever emits
`"block"`. The caller (`get_all_guard_warnings`) returns a list of warnings
with summary — presumably displayed to the Layer 2 agent or written to
`agent_summary`. Whether the trade proceeds or is blocked is delegated to the
Layer 2 agent's discretion (or the user reading a Telegram message).

**Why it matters:** Cooldown "enforcement" is actually "advisory" — Layer 2
can override every guard. A runaway Layer 2 decision loop (stuck on a
strongly-BUY'd ticker) will generate warnings but still place trades. The
multiplicative loss escalation `{0:1, 1:1, 2:2, 3:4, 4:8}` is also advisory —
the multiplier just affects the warning message, not an enforced cooldown.

**Fix shape:** Introduce hard limits for extreme cases:
- `4+ consecutive losses` on same strategy → "block" severity, enforced
- More than `2×` the position rate limit in the window → "block"
Keep warnings for softer signals.

### Finding 3.6 — HIGH [H]: `record_trade` resets loss streak on pnl_pct==0
**Where:** `trade_guards.py:194-202`
**What:** The branch structure is `if pnl_pct < 0: increment; else: reset`.
A breakeven trade (pnl_pct == 0) takes the else branch and resets the
consecutive-loss counter to zero, undermining the escalation system. A loss
sequence of -3%, -2%, 0%, -4% is recorded as "1 consecutive loss" at the end.

**Fix shape:** `if pnl_pct < 0: increment; elif pnl_pct > 0: reset`. Leave the
counter unchanged on exactly-zero trades.

### Finding 3.7 — HIGH [H]: `record_trade` prune loop has unguarded `fromisoformat`
**Where:** `trade_guards.py:213-217`
```python
cutoff = now - timedelta(hours=24)
state["new_position_timestamps"][strategy] = [
    ts for ts in state["new_position_timestamps"][strategy]
    if datetime.fromisoformat(ts) >= cutoff
]
```
**What:** The list comprehension calls `fromisoformat(ts)` with no exception
handling. If any timestamp in the persisted list is malformed, ValueError
propagates and the prune fails, which aborts `record_trade`, which may be
called from a critical trade-recording path in Layer 2. The trade may be
applied to portfolio state but not recorded in the guards state.

**Fix shape:** Wrap in try/except and drop malformed entries on the floor (or
log + drop). Prune logic must be defensive.

### Finding 3.8 — MEDIUM [H]: `drawdown.breached` is computed but never enforced by the system
**Where:** `risk_management.py:53-128`, search for callers
**What:** `check_drawdown` returns `{"breached": bool, ...}`. There's no code
that reads this and **halts trading**. The value is presumably displayed in
reports and used by Layer 2 for context, but the "circuit breaker" described
in the module docstring is a computation, not an enforcement mechanism.

Compare to the name "circuit breaker" — which implies that hitting the
breach automatically stops trades. In this codebase, the breach is advisory.

**Fix shape:** Either rename to `drawdown_metric` (honest) or wire the
breached flag into the guard chain: if `breached`, `check_overtrading_guards`
returns a `"block"` for all trades.

### Finding 3.9 — MEDIUM [H]: `monte_carlo_risk.CORRELATION_PRIORS` omits BTC-MSTR and AMD-NVDA correlations
**Where:** `monte_carlo_risk.py:112-127`
**What:** CLAUDE.md describes MSTR as a "leveraged BTC proxy" — implying very
high BTC-MSTR correlation. Not in the priors dict. `("AMD", "NVDA")` is also
absent despite being a commonly-traded pair in the universe and both being AI
semiconductor names. When historical data is absent, these default to zero
correlation → portfolio VaR is understated (diversification assumed where
none exists).

**Fix shape:** Add `("BTC-USD", "MSTR"): 0.80, ("AMD", "NVDA"): 0.70`. Also
gold/silver is already there.

### Finding 3.10 — MEDIUM [M]: `_nearest_psd` clips eigenvalues to 1e-8 — near-singular correlation matrices
**Where:** `monte_carlo_risk.py:86-104`
**What:** When many assets are near-perfectly correlated (e.g., all tech
stocks during a sector selloff), the correlation matrix has eigenvalues
approaching zero. Clipping to 1e-8 produces a near-singular matrix; subsequent
Cholesky decomposition for correlated sampling is numerically unstable.
Generated samples have spurious noise instead of reflecting the true
correlation structure.

**Fix shape:** Clip eigenvalues to a larger floor (e.g., 1e-4 or 1% of max
eigenvalue) and log when clipping is heavy (e.g., >20% of eigenvalues hit
the floor) — that's a signal that VaR numbers should be treated with caution.

### Finding 3.11 — MEDIUM [H]: `compute_probabilistic_stops` returns {} on ImportError without telemetry
**Where:** `risk_management.py:230-234`
**What:** `except ImportError: logger.warning(...); return {}`. An ImportError
here means the probabilistic-stop feature is entirely disabled. The caller
(`reporting.py`) cannot distinguish "no positions" from "feature broken". No
health signal is raised.

**Fix shape:** Set a health flag on ImportError, alert via Telegram on first
occurrence, and return a sentinel like `{"_import_error": str(e)}` that the
caller can detect.

### Architectural critique — portfolio-risk

The portfolio-risk layer is **advisory, not enforcing**. Every risk check
computes a metric but does not halt action. The "circuit breaker" name is
aspirational; the actual system relies on the Layer 2 agent to honor
warnings. For a system that trades real money based on these decisions
elsewhere (per CLAUDE.md: "The user trades real money elsewhere based on
your signals"), this is a weak line of defense.

The design assumption is that a thinking agent (Claude Layer 2) will read
the warnings and make good decisions. But the same warnings are not enforced
in autonomous mode (where there's no LLM to think). The fallback path
has weaker risk controls than the primary path — the inverse of what safety
engineering would prescribe.

The **single most important improvement**: define a small set of hard-block
rules ("4+ consecutive losses", "drawdown > 25%", "concentration > 40% in
one ticker") that are enforced by code, not by advisory warnings. The rest
can remain advisory for nuanced decisions.

---

## Subsystem 4 — metals-core

Files reviewed: `data/metals_loop.py` (5261 LOC), `microstructure.py`,
`microstructure_state.py`, `metals_orderbook.py`, `metals_cross_assets.py`,
`exit_optimizer.py`, `price_targets.py`, `fin_snipe.py`, `orb_predictor.py`.

### Finding 4.1 — CRITICAL [H]: Silver fast-tick silently no-ops on any error path
**Where:** `data/metals_loop.py:812-834`
**What:** `_silver_fast_tick` has multiple silent-return paths:
- `silver_key is None` → return (line 823-824)
- `price is None or <= 0` → return (line 827-828)
- `ref is None or <= 0` → return (line 833-834)

Plus `_silver_fetch_xag` wraps `requests.get` in a bare `except Exception: pass`
and returns cached data (line 746-748). A multi-hour Binance FAPI outage would
return the last-known price for every fast-tick, firing NO alerts even as the
real silver market crashed.

**Why it matters:** Silver fast-tick is the **safety net** that catches rapid
XAG drops (-3% to -12.5% thresholds per SILVER_ALERT_LEVELS). The whole point
is to detect flash crashes when the 60-second main cycle would be too slow.
Silent degradation of the safety net is worse than not having it — operators
assume protection that isn't there.

**Fix shape:** Each silent-return path should increment a health counter.
When 5+ consecutive ticks fail to fetch a valid XAG, send a Telegram alert.
Log each degradation reason with exc_info=True.

### Finding 4.2 — CRITICAL [H]: `place_stop_loss` has no volume constraint enforcement
**Where:** `portfolio/avanza_session.py:476-538`
**What:** The function accepts `volume` blindly. It does not check against:
- Current position size for the orderbook
- Existing stop-loss volume (would be a double-stop)
- Existing open SELL orders that would trigger before the stop

The user's memory explicitly says: *"sell + stop-loss volume must not exceed
position size"*. This rule is not encoded in the API wrapper. A caller that
computes the wrong volume (e.g., after a partial fill that wasn't re-checked)
will pass an over-limit stop-loss, which Avanza may reject (warrant refunded)
or accept to create a short position — a real-money mistake.

**Why it matters:** The March 3, 2026 incident (documented in
`docs/TRADING_PLAYBOOK.md` and CLAUDE.md critical rules) happened **because**
the wrong endpoint was used. The current fix documents the endpoint
(`/_api/trading/stoploss/new` ✓ line 528) but does NOT add the pre-call
volume check. The next incident will be the same category.

**Fix shape:** Before the `api_post` call, fetch current positions and open
orders for the orderbook, compute `free_volume = position - sum(open_sells) -
sum(existing_stops)`, and raise ValueError if `volume > free_volume`.

### Finding 4.3 — HIGH [H]: Metals loop is singleton-locked via msvcrt (same bug as main.py)
**Where:** `data/metals_loop.py:583-611`
**What:** Same issue as Finding 2.1 — `msvcrt is None: return True` silently
disables the singleton lock on Linux/WSL. Metals loop writes positions state,
trade state, swing state, fish state, etc. Two concurrent metals loops would
race on all of them.

**Why it matters:** The metals loop has a richer state surface than main.py
(positions, trades, orders, contract state). Interleaved writes are more
damaging.

### Finding 4.4 — HIGH [H]: `_silver_init_ref` persists ref price on first tick without confirming position entry
**Where:** `metals_loop.py:751-785`
**What:** If `_silver_underlying_ref` is None (first run after restart or
session reset), the function uses the **current XAG price** as the reference
(line 779-784). The comment says "for first run". But: a restart **during
market hours** with an active position recorded days ago would set the ref
to the CURRENT XAG, not the REAL entry price. Subsequent alert thresholds
(e.g., -5% from ref) would measure from wrong baseline.

The code tries to read `underlying_entry` from persisted state first (line
772), which is the correct baseline. But if the state file was written
without that field (older versions or corruption), the code fallbacks to
current XAG as ref — silently misaligned.

**Fix shape:** If `underlying_entry` is missing from state AND a position is
active, DO NOT proceed with a current-price fallback. Log an ERROR and
refuse to arm fast-tick until `underlying_entry` is populated (either by
re-recording the position or by manual config).

### Finding 4.5 — HIGH [H]: `_sleep_for_cycle` overrun handling has different semantics than main.py
**Where:** `metals_loop.py:678-709`
**What:** On overrun (`remaining <= 0`), metals_loop logs and returns
immediately (line 692-693). If the cycle consistently runs long (e.g., heavy
Telegram queue + LLM inference + Playwright warmup), the loop has no
recovery strategy — it runs back-to-back cycles with no fast-tick (because
`min_remaining` check at line 700 skips the sub-loop when too little time
remains). Silver protection degrades to zero during sustained overrun.

### Finding 4.6 — HIGH [H]: Optional module imports at module scope mean silent feature loss
**Where:** `metals_loop.py:84-198`
**What:** 10+ try/except ImportError blocks at module scope, each setting a
flag (`LLM_AVAILABLE`, `RISK_AVAILABLE`, `TRACKER_AVAILABLE`,
`SWING_TRADER_AVAILABLE`, `EXECUTION_ENGINE_AVAILABLE`, `CATALOG_AVAILABLE`,
`AVANZA_CONTROL_AVAILABLE`, `CRYPTO_DATA_AVAILABLE`, `NEWS_KEYWORDS_AVAILABLE`, ...).
When an import fails, the warning prints ONCE at startup and the flag is False.
There's no runtime telemetry of which features are off. A developer who
breaks an import silently kills a whole feature; the loop keeps running with
degraded capability and no alert.

**Fix shape:** At startup, consolidate feature flags into a single dict and
send it to the health channel. Any disabled feature should trigger a
Telegram alert for the operator to investigate.

### Finding 4.7 — HIGH [M]: Metals loop and main loop share `portfolio_state.json` family but may write concurrently
**Where:** `metals_loop.py` writes to state files in `data/` that main.py may
also touch (transitively via `alpha_vantage.py`, `market_health.py` etc.).
**What:** Two separate processes, each with their own msvcrt singleton lock.
Each uses `atomic_write_json`. `atomic_write_json` is atomic per write, but
two processes racing on the same file can lose updates (P1 reads → P2 writes
→ P1 writes its older version, overwriting P2's). Neither lock prevents this
cross-process race because they lock DIFFERENT files (`main_loop.singleton.lock`
vs `metals_loop.singleton.lock`).

**Fix shape:** For shared state files (like health, fundamentals cache),
introduce file-level locking (fcntl on Linux, msvcrt region lock on Windows).
Or, make each loop responsible for its own set of files with no overlap.

### Finding 4.8 — MEDIUM [H]: `get_cet_time` depends on timeapi.io with no rate-limit tracking
**Where:** `metals_loop.py:897-928`
**What:** Every call to `cet_hour()` or `cet_time_str()` hits timeapi.io.
These functions are called dozens of times per cycle (in helpers, logs,
spike-check, market-hours). At 60s cadence, that's ~1000 external requests
per hour to a third-party time API. No rate limit, no batching, no short-TTL
cache of the result.

**Why it matters:** timeapi.io is a public free API with rate limits. The
fallback (`zoneinfo`) is always available and correct, but the code **always**
tries the network first. This is wasteful and makes the loop's accuracy
depend on an external service.

**Fix shape:** Default to `zoneinfo` (local DST-aware). Only hit timeapi.io
for an initial sanity check at startup and hourly drift correction.

### Finding 4.9 — MEDIUM [H]: Metals signal tracker and main loop signal log are independent
**Where:** `metals_loop.py:121-132` imports `metals_signal_tracker`, which has
its own `log_snapshot` and accuracy tracking separate from the main
`portfolio.outcome_tracker`.
**What:** Two parallel accuracy systems: one in `portfolio/` for the main
loop's 30 signals, one in `data/metals_signal_tracker.py` for metals-specific
signals. If they share ticker data (XAG-USD, XAU-USD), they track accuracy
separately and may report different numbers. Downstream consumers (dashboard,
reports) may pick whichever they happen to read.

**Fix shape:** Consolidate into the main accuracy system with a `source=metals_loop`
tag. One source of truth per ticker.

---

## Subsystem 5 — avanza-api

Files reviewed: `avanza_session.py`, `avanza_orders.py`, `avanza_control.py`,
`avanza_tracker.py`, `avanza_client.py`, `portfolio/avanza/*`.

### Finding 5.1 — CRITICAL [H]: Dual implementation (legacy + new package) can double-fire orders
**Where:** `portfolio/avanza_session.py` (legacy Playwright-based) AND
`portfolio/avanza/trading.py` (new TOTP-based package).
**What:** The user has migrated to the new `portfolio/avanza/` package with
TOTP auth (per memory: `reference_avanza_new_package.md`), but the legacy
`avanza_session.py`, `avanza_orders.py`, `avanza_control.py`,
`avanza_tracker.py`, `avanza_client.py` still exist and are imported by
`metals_loop.py`, `portfolio/iskbets.py`, `portfolio/fin_snipe.py`, and others.

The two implementations are **not coordinated**. A metal trade could fire via
`portfolio.avanza_control.place_order` while a simultaneous operation via
`portfolio.avanza.trading.place_order` executes the same decision — two
orders for the same intent. Both implementations talk to the same Avanza
account (DEFAULT_ACCOUNT_ID = "1625505").

**Why it matters:** No idempotency key, no order deduplication. The nonce for
orders is server-side (orderId returned on success), so a client-side caller
cannot detect duplication before both orders execute.

**Fix shape:** Introduce an order-intent table with `(orderbook_id, side,
price_rounded, volume, 5-minute window)` as the dedup key. Any caller must
acquire a lock on this intent before submitting an order.

### Finding 5.2 — HIGH [H]: Playwright context is module-level and never recreated after 401
**Where:** `avanza_session.py:115-134, 195-200`
**What:** The context is created once via `_get_playwright_context()` and
cached in `_pw_context`. On 401, `close_playwright()` is called which clears
the context. The **next** call to `_get_playwright_context` creates a new
context. Between the 401 and the next call, **any other thread holding a
reference** to the closed context will fail on use.

The `_pw_lock` at line 35 protects creation/cleanup, but the **usage** of the
context (e.g., `ctx.request.get`) is OUTSIDE the lock (line 194). So: thread
A acquires ctx, thread B hits 401 and closes ctx, thread A tries to use its
closed ctx → exception.

**Fix shape:** Add a reference count or a generation counter. Or, only use
the context inside the lock (serializes all API calls, lowers concurrency
but eliminates the race).

### Finding 5.3 — HIGH [H]: Session expiry returns `True` (expiring) on parse failure
**Where:** `avanza_session.py:104-112`
```python
def is_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
    remaining = session_remaining_minutes()
    if remaining is None:
        return True  # treat as expiring
```
**What:** If the session file is unreadable or has no `expires_at`, this
returns True, treating the session as expiring. Callers that use this to
decide whether to pre-emptively refresh auth will refresh unnecessarily.

That's not dangerous by itself, but: the refresh path requires user BankID
interaction (`scripts/avanza_login.py`). A spurious "expiring" indicator
asks the user to BankID every N minutes, which causes workflow friction.

**Fix shape:** Only return True if we actively know the session is expired.
Return False if we can't tell — caller can then verify via live API call.

### Finding 5.4 — HIGH [H]: `_get_csrf` only searches cookies, doesn't fall back to localStorage
**Where:** `avanza_session.py:206-212`
**What:** The function iterates `ctx.cookies()` looking for `AZACSRF`. If
Avanza's auth is in a flow where the CSRF is in localStorage (newer web flow)
or in a response header (API-first flow), this fails with
`AvanzaSessionError("No AZACSRF cookie found")`. The session file may be
valid, but CSRF extraction fails, which looks like a session expiry.

**Fix shape:** Fallback chain: cookies → localStorage → make a GET to `/_cqr`
endpoint and parse the response for a CSRF refresh.

### Finding 5.5 — HIGH [H]: `api_post` throws AvanzaSessionError on 401/403 but does NOT auto-retry with refreshed session
**Where:** `avanza_session.py:238-256`
**What:** On 401/403, the function calls `close_playwright()` and raises.
The caller is responsible for catching and retrying. But the caller of
`place_order`, `place_stop_loss` etc. sees an exception mid-trade, with no
indication that a simple reauth would fix it.

For time-sensitive operations (trading at market open, emergency stop), the
round trip of (exception → alert → human runs login → retry) is 5+ minutes.
A position could move significantly in that time.

**Fix shape:** On 401, automatically trigger a re-auth flow if possible
(TOTP path from the new package) and retry once. Alert the user only if
auto-retry also fails.

### Finding 5.6 — HIGH [M]: `get_instrument_price` tries 4 instrument types sequentially, logging a warning per miss
**Where:** `avanza_session.py:586-607`
**What:** The function tries `stock → certificate → fund → exchange_traded_fund`.
For a warrant, `stock` and `fund` will fail first (2 network round-trips
wasted) before `certificate` succeeds. Each failure logs a WARNING at line 603.

**Why it matters:** A warrant price lookup triggers 3 warning log lines in
production. Operators reading logs see many warnings, learn to ignore them,
and miss real warnings. Also: 3 failed requests = 3x latency + rate-limit
usage.

**Fix shape:** Maintain a `instrument_type_cache` per orderbook_id populated
on first successful lookup. Subsequent calls hit the right endpoint directly.

### Finding 5.7 — MEDIUM [H]: `place_order` has no idempotency token — retries create duplicate orders
**Where:** `avanza_session.py:357-389`
**What:** The payload has no client-side unique ID. If the caller retries
(e.g., after a timeout), the same order payload is sent again. Avanza's API
may de-duplicate server-side but there's no guarantee. Even if it does, the
client has no way to know if the "retry" was a new order or a re-confirmation.

**Fix shape:** Generate a client UUID, include it in the payload (if Avanza
supports it) or store it locally and compare with `orderId` responses.

### Finding 5.8 — MEDIUM [H]: `get_open_orders` endpoint fallback chain swallows ambiguity
**Where:** `avanza_session.py:404-420`
**What:** The function tries `trading/rest/order/account/{aid}`, on
`RuntimeError` (any API error) falls back to `trading/rest/deals-and-orders`,
on another `RuntimeError` returns `[]`. The fallback chain makes it impossible
to distinguish "no open orders" from "API is broken and we don't know your
orders". Trades may be placed thinking there are no open orders when in fact
the API is degraded.

**Fix shape:** Treat "all endpoints failed" as an explicit error state. Do
NOT silently return `[]` in that case — raise a RuntimeError so callers must
handle the unknown state.

---

## Subsystem 6 — signals-modules

Files reviewed: selected from `portfolio/signals/` — `trend.py`, `momentum.py`,
`volume_flow.py`, `volatility.py`, `mean_reversion.py`, `smart_money.py`,
`fibonacci.py`, `econ_calendar.py`, `news_event.py`, `forecast.py`,
`claude_fundamental.py`, `calendar_seasonal.py`, `macro_regime.py`,
`heikin_ashi.py`.

### Finding 6.1 — CRITICAL [H]: Econ calendar dates hardcoded for 2026-2027 only
**Where:** `portfolio/econ_dates.py:23,38,127`; `portfolio/fomc_dates.py:13,25,37`
**What:**
```python
CPI_DATES_2026 = [...]
CPI_DATES_2027 = [...]
FOMC_DATES_2026 = [...]
FOMC_DATES_2027 = [...]
```
These are static lists. After 2027-12-31, `next_event` returns None and
`_event_proximity` / `_pre_event_risk` return HOLD for everything. The
econ_calendar signal silently stops contributing. No alert, no visible
failure — the signal just becomes dead weight.

**Why it matters:** Today is 2026-04-05, so ~21 months until 2028. This is
a ticking bomb. An operator in 2028 will see `econ_calendar` accuracy drop
to trivial levels and won't know why unless they read the comments.

**Fix shape:** Fetch from a remote source (trading-calendars Python library,
Fed website, or a simple `data/calendar_2028.json` that the operator updates).
At minimum: on first call when `max(calendar_dates) - now < 90 days`, raise
a Telegram alert asking for calendar refresh.

### Finding 6.2 — HIGH [H]: Trend signal and Heikin-Ashi signal measure the same underlying feature
**Where:** `portfolio/signals/trend.py` (7 sub-signals), `portfolio/signals/heikin_ashi.py` (HA trend + Hull MA + Alligator + Elder Impulse + TTM Squeeze).
**What:** Both modules include multiple trend-following indicators that are
mathematically correlated. Trend.py has Golden Cross, MA Ribbon, Price vs MA200,
Supertrend (ATR-based trend), Parabolic SAR, Ichimoku Cloud, ADX. Heikin-Ashi
has HA Trend (moving average of HA candles), Hull MA (smoothed MA), Alligator
(three MAs), Elder Impulse (EMA + MACD histogram), TTM Squeeze (BB inside KC
breakout — partly mean-reversion).

At least 60% of these are "is price above long MA" in different disguises.
When the market trends, they all vote the same way. The correlation group
`trend_direction = {ema, trend, heikin_ashi, volume_flow}` in signal_engine
already applies a 0.3x penalty to secondary signals — but this is an ad-hoc
patch on a structural problem.

**Why it matters:** The system believes it has 30 independent signals. It
actually has ~8 independent axes (trend, momentum-reversion, volatility,
volume, structure, sentiment, macro, LLM). Each axis has 3-5 correlated
signals stuffed in. Effective sample size for consensus is much smaller than
nominal, so "3 signals agree" carries less information than the MIN_VOTERS
threshold assumes.

**Fix shape:** Build a **correlation clustering** report monthly. Run a
correlation matrix on signal votes over last 30d; identify clusters with
>0.7 correlation; within each cluster, keep only the best accuracy signal as
the primary voter; others get the penalty automatically (rather than manual
`CORRELATION_GROUPS` maintenance).

### Finding 6.3 — HIGH [H]: `news_event._persist_headlines` writes to a single shared file clobbered by ThreadPoolExecutor
**Where:** `news_event.py:45-97`
**What:** `_HEADLINES_PATH = data/headlines_latest.json`. Every call to
`_persist_headlines(ticker, headlines)` **overwrites** this file with one
ticker's headlines. In `signal_engine.generate_signal`, news_event runs inside
the per-ticker ThreadPoolExecutor (8 workers). The 8 tickers race; the
last-to-complete wins. Earlier-completing tickers' headlines are lost.

**Why it matters:** `headlines_latest.json` is consumed by the fish monitor
(per the persistence comment at line 46). The fish monitor sees whichever
ticker happened to complete last — not necessarily XAG-USD even if XAG news
is most relevant. The "latest" file is effectively a random ticker's
headlines.

**Fix shape:** Per-ticker files: `data/headlines/{ticker}.json`. Or keyed
single file: `data/headlines_latest.json` with shape `{ticker: [...], ...}`
and append semantics per ticker.

### Finding 6.4 — HIGH [H]: Forecast circuit breakers are module-level floats
**Where:** `signals/forecast.py:91-96`
**What:** `_kronos_tripped_until` and `_chronos_tripped_until` are module-level
floats protected by `_forecast_lock`. The check-and-set is atomic, but the
downstream usage is not: after `_forecast_lock.acquire(); check; release`,
another thread can trip the breaker before this thread calls the model, so
the current thread's call goes to a broken model anyway.

More practically: 90s Kronos timeout + 120s Chronos timeout = up to 210s
blocked on a single ticker before the breaker trips. At 20 tickers, worst
case is 20 × 210s = 70 minutes — one "failed" cycle can eat more than an
hour.

**Fix shape:** Trip the breaker on the FIRST timeout and reset only after
5+ minutes. Short-circuit the forecast call entirely when breaker is tripped
(no retry within the cycle).

### Finding 6.5 — HIGH [M]: `_init_kronos_enabled` reads config at module import
**Where:** `signals/forecast.py:58-74`
**What:** Config.json is read once at import time. Changing `forecast.kronos_enabled`
requires a process restart. For a live trading system where the operator
might want to flip a signal on/off in response to a crisis, this is
inflexible.

Many other signal modules have the same pattern (import-time config reads).
The system as a whole is not reconfigurable at runtime.

**Fix shape:** Move config reads inside the `compute` function (once per
cycle is cheap) or cache with a 60-second TTL.

### Finding 6.6 — MEDIUM [H]: Mean-reversion RSI(2) thresholds are extreme (<10, >90)
**Where:** `signals/mean_reversion.py:56-59`
**What:** RSI(2) below 10 or above 90 is very rare on daily data — maybe a
handful of times per year per ticker. The signal fires rarely, which means
its accuracy stats accumulate slowly. 30-sample minimum for the accuracy gate
may take years to hit per ticker. Meanwhile the signal is effectively a
constant HOLD.

**Fix shape:** Relax to <15/>85 or add RSI(5) as a secondary with looser
thresholds. Document the expected firing frequency.

### Finding 6.7 — MEDIUM [H]: `context` dict is shared and mutable across signals in the same cycle
**Where:** `signal_engine.py:1246` + `signals/*.py` compute_* functions
**What:** `context_data = {"ticker": ..., "config": ..., "macro": ..., "regime": ...}`.
This dict is passed by reference to every signal with `requires_context`. If any
signal modifies `context_data["regime"]` or adds a key (e.g., for inter-signal
communication), the change leaks to all subsequent signals in the same cycle.
No signal appears to do this today, but there's no guard preventing it.

**Fix shape:** Deep-copy the dict on each invocation or use a frozen dataclass.

### Finding 6.8 — MEDIUM [H]: `_golden_cross` compares iloc[-1] and iloc[-2] without verifying no NaN gap
**Where:** `signals/trend.py:37-62`
**What:** The check at line 52-53 verifies `valid.sum() >= 2` (at least 2 bars
where both SMAs are valid) but then compares `iloc[-1]` and `iloc[-2]`, which
might be separated by NaN bars if the series has gaps. A real cross on a
non-adjacent pair is detected as "cross at this bar" even when the
intermediate bars could have had different behavior.

**Fix shape:** Find the last 2 **adjacent** valid bars, not just the last 2
positions.

### Finding 6.9 — MEDIUM [M]: Confidence cap of 0.7 is pervasive across signal modules but unenforced
**Where:** `signals/news_event.py:43`, `signals/forecast.py:40`,
`signals/econ_calendar.py:27`
**What:** Each module defines its own `_MAX_CONFIDENCE = 0.7`. This is a
per-module cap. The signal_engine.py global cap (0.8 at line 1630) is less
strict. There's inconsistency: some modules enforce 0.7, some enforce 1.0, the
engine enforces 0.8. A chart of "per-signal confidence distribution" would
show odd bunching at 0.7, 0.8, 1.0 — artifacts of multiple cap stages.

**Fix shape:** Single source of truth: max_confidence defined in `signal_registry`
entry per signal and applied by `_validate_signal_result` (line 431). Remove
the local `_MAX_CONFIDENCE` constants.

### Finding 6.10 — MEDIUM [H]: `next_event` falls back to "no events" when past 2027, silently
**Where:** `portfolio/econ_dates.py` — `next_event(ref_date)` iterates a static list
**What:** When all dates in the list are before `ref_date`, the function
returns None. The caller in `econ_calendar._event_proximity` returns HOLD
on None. The signal becomes a silent no-op. No telemetry, no alert.

**Fix shape:** When no next event is found but the caller expects one, log
a WARNING "Econ calendar exhausted — update CPI_DATES_{year}" and record it
to health state.

---

## Subsystem 7 — data-external

Files reviewed: `data_collector.py`, `sentiment.py`, `fear_greed.py`,
`alpha_vantage.py` (scanned), `futures_data.py` (scanned), `onchain_data.py`
(scanned), `fx_rates.py` (scanned).

### Finding 7.1 — HIGH [H]: Empty API response treated as circuit-breaker failure
**Where:** `data_collector.py:87-91`
```python
if not data:
    logger.warning("Binance %s returned empty data for %s %s", ...)
    cb.record_failure()
    return pd.DataFrame()
```
**What:** A legitimate empty response (e.g., weekend data for a currency pair
that Binance doesn't quote) is counted as a circuit-breaker failure. After 5
such "failures" across any symbol, the circuit opens and ALL symbols on that
exchange are blocked for 60 seconds.

**Why it matters:** A cross-symbol false-positive: 5 empty-data events on
obscure pairs can shut down real data collection for BTC, ETH, XAG, XAU
simultaneously. The circuit-breaker was designed to protect against exchange
outages, not against "legitimate no-data-for-this-query".

**Fix shape:** Distinguish "API error" from "query returned zero rows".
Empty response on a valid symbol at a valid interval is an API quality
issue, not an outage — log and return empty df without tripping the breaker.

### Finding 7.2 — HIGH [M]: Sentiment module subprocesses LLM inference scripts at module-level paths
**Where:** `sentiment.py:32-45`
**What:** Paths to inference scripts (`Q:\models\cryptobert_infer.py`, etc.)
are module-level constants. If the models directory is moved or the venv is
rebuilt, the paths become invalid and every subprocess call fails with a
FileNotFoundError. There's no validation at startup.

**Fix shape:** At import, verify that the configured scripts exist and the
model Python binary is executable. Fail loudly (or degrade to "sentiment
disabled" with health alert) rather than failing silently on first use.

### Finding 7.3 — HIGH [H]: Subprocess-based sentiment and forecast models have no concurrency limit
**Where:** Every call to a local LLM signal spawns a Python subprocess.
**What:** ThreadPoolExecutor(8) fires 8 concurrent ticker computes. Each
ticker may fire 2-3 subprocess-based signals (forecast, sentiment shadow
models, etc.). That's 16-24 concurrent Python subprocesses mid-cycle, each
consuming ~500MB-2GB RAM for GPU model loading. On a 16GB workstation with
a 10GB GPU, this is the exact path to OOM crashes or GPU memory exhaustion.

**Fix shape:** Global semaphore limiting concurrent model subprocesses to
2-4 (per GPU memory budget). Model-loaded subprocesses should share a
long-running worker process (as `llm_batch.py` already does for
Ministral/Qwen3 — extend the pattern to Kronos/Chronos/FinGPT).

### Finding 7.4 — HIGH [M]: NewsAPI daily budget tracked in-memory, lost on restart
**Where:** `shared_state.py:193-196`
**What:** `_newsapi_daily_count` and `_newsapi_daily_reset` are module-level
integers. A loop restart resets both to 0/0.0 → the budget is re-used from
scratch. A loop that crashes and restarts 10 times in a day could burn
10× the quota (900+ calls / day limit 100).

**Fix shape:** Persist counter to `data/api_quota_state.json`; load at
startup; reset only on crossing UTC midnight.

### Finding 7.5 — HIGH [H]: `_cached` dogpile prevention has a TOCTOU window
**Where:** `shared_state.py:67-77`
**What:** The check `if key in _loading_keys` happens inside the lock, and
the `_loading_keys.add(key)` is also inside the lock — atomic. BUT the
subsequent `func(*args)` runs **outside** the lock. If `func` takes longer
than 120s (the timeout mentioned at line 29), and the cache entry expires
during that time, a second thread checking may see `key in _loading_keys`
AND stale-beyond-max-stale data, returning None (line 75-76). The loop
continues the call, eventually completes; the first thread updates the
cache at line 82; the second thread's `None` return has already been used.

**Fix shape:** Add a load-started timestamp to `_loading_keys` (change to
dict `{key: start_time}`). When another thread hits the loading key, check
if the load has exceeded LOADING_TIMEOUT and, if so, allow itself to
proceed.

### Finding 7.6 — HIGH [H]: Sentiment category map silently lumps all US stocks as TECHNOLOGY
**Where:** `sentiment.py:50-71`
**What:** Every stock in the universe (including LMT which is **defense**,
not tech) is mapped to TECHNOLOGY for sentiment categorization. The
CryptoCompare/Yahoo Finance sentiment request uses this category. LMT news
is fetched as "technology" category news, pulling in NVDA/AMD headlines and
missing actual Lockheed Martin / defense news.

**Fix shape:** Accurate category mapping. LMT → DEFENSE, TTWO → GAMING, etc.
At minimum, use the `TICKER_SECTORS` from `news_keywords.py` which already
has this mapping.

### Finding 7.7 — MEDIUM [H]: `_RateLimiter` uses `time.time()` for interval tracking
**Where:** `shared_state.py:167-174`
**What:** `time.time()` can jump backward (NTP sync) or forward (suspend/
resume). The rate limiter would then fire incorrectly — either too many
calls in quick succession (backward jump) or unnecessary waits (forward
jump).

**Fix shape:** Use `time.monotonic()` for intervals (standard fix).

---

## Subsystem 8 — infrastructure

Files reviewed: `file_utils.py`, `http_retry.py`, `shared_state.py`,
`telegram_notifications.py`, `health.py` (scanned), `process_lock.py`
(scanned), `journal.py` (scanned).

### Finding 8.1 — HIGH [H]: `atomic_write_json` does not fsync the parent directory
**Where:** `file_utils.py:13-28`
**What:** `os.replace(tmp, str(path))` is an atomic rename. On POSIX, the
file's contents are durable after fsync on the file (which happens when the
`with` block exits and closes the FD... though even close may not fsync —
only the `os.fdopen` close implicit flush). But the DIRECTORY ENTRY for the
rename is not durable until the parent directory is fsync'd.

On a power loss after `os.replace` returns but before the parent directory
sync, the filesystem may lose the rename and leave the OLD file in place
(or nothing at all, depending on FS). The write looks atomic from a process
crash but not from a system crash.

**Why it matters:** On Windows NTFS this is less of an issue (NTFS tends
to be more conservative about metadata durability), but on WSL ext4 or any
Linux setup, power loss can silently lose the rename. The trade state, the
portfolio state, the signal log — any of them could revert to old content.

**Fix shape:** After `os.replace`, fsync the parent directory:
```python
dirfd = os.open(str(path.parent), os.O_RDONLY)
try:
    os.fsync(dirfd)
finally:
    os.close(dirfd)
```
(Windows: skip this; Windows does metadata durability differently.)

### Finding 8.2 — HIGH [H]: `atomic_write_json` doesn't fsync the file before `os.replace`
**Where:** `file_utils.py:22-24`
**What:** The `with os.fdopen(fd, "w", ...)` closes the file on exit but
does NOT fsync it. Depending on the filesystem buffer flushing, the data
may be in OS cache when `os.replace` renames the temp to the final name.
On a crash during this window, the renamed file exists but is empty or
partial.

**Fix shape:** Before `os.replace`, call `f.flush()` and `os.fsync(fd)`
inside the `with` block.

### Finding 8.3 — HIGH [H]: `load_json` returns default on ALL errors without differentiation
**Where:** `file_utils.py:31-48`
**What:** FileNotFoundError → default. OSError (permission, lock) → default.
JSONDecodeError → default. ValueError → default. The caller cannot
distinguish "file doesn't exist (new install)" from "file corrupted (data
loss)" from "file locked (retry later)". Downstream code (portfolio_mgr,
trigger, trade_guards) uses the default as if nothing was wrong and may
silently overwrite the corrupted file with default state.

**Fix shape:** Two variants: `load_json` (current behavior for
true-optional reads) and `require_json` (raises on corruption / permission
errors). Callers dealing with state should use `require_json`.

### Finding 8.4 — HIGH [H]: `atomic_append_jsonl` is not atomic under concurrent writers
**Where:** `file_utils.py:134-146`
**What:** `open(path, "a")` uses O_APPEND on POSIX which is atomic for
writes **smaller than PIPE_BUF** (4096 on Linux). The function writes a
single line via `f.write(line)` where `line` is the JSON-serialized entry
plus `\n`. For JSON entries **larger than 4096 bytes** (agent summary logs,
large signal snapshots, journal entries with nested dicts), the POSIX O_APPEND
atomicity guarantee does NOT apply, and two concurrent writers may
interleave — producing corrupted JSONL.

On Windows, `O_APPEND`-style behavior is not guaranteed by Python's `open`
mode `"a"`. Even 100-byte entries can interleave.

**Why it matters:** `signal_log.jsonl` gets 20-30 tickers per entry. Each
entry can easily exceed 4KB. `journal.jsonl`, `layer2_journal.jsonl`,
`invocations.jsonl` — all at risk.

**Fix shape:** Use a per-file `threading.Lock` OR OS-level file locking
(`fcntl.flock` on POSIX, msvcrt.locking on Windows) for the append.

### Finding 8.5 — HIGH [H]: Telegram HTTP 400 fallback strips markdown but doesn't log the original failure
**Where:** `telegram_notifications.py:66-81`
**What:** On HTTP 400 with parse error, the function retries without
`parse_mode`. The retry may succeed, delivering an unformatted message. The
operator sees the unformatted message but doesn't know that the formatted
version failed or why. Over time, the system sends many unformatted
messages and nobody notices the formatting bug until someone looks closely.

Also: the Markdown-parse retry does not log the ORIGINAL message or the
specific parse error — only a generic "parse failed" warning. Debugging
"why did this specific message lose its formatting?" is hard.

**Fix shape:** On Markdown failure, log the FIRST 200 chars of the problematic
message at WARNING level with the error description. Also increment a
`telegram_markdown_failures` counter in health state.

### Finding 8.6 — MEDIUM [H]: `load_jsonl_tail` may skip the last entry on mid-line seek
**Where:** `file_utils.py:112-115`
**What:** The code seeks to `max(0, file_size - tail_bytes)` and reads to
EOF. If the seek lands mid-line, the first line is corrupt and `lines[1:]`
skips it. That's correct. BUT: if the LAST line is incomplete (writer
crashed mid-append), the reader silently drops it and reports the second-
to-last as "last". The caller has no way to know the last entry was truncated.

**Fix shape:** Validate that the last returned entry starts with `{` and
ends with `}` (or otherwise has balanced brackets). If not, log and skip.

### Finding 8.7 — MEDIUM [H]: `fetch_with_retry` does not honor `Retry-After` header on 429
**Where:** `http_retry.py:14, 39-45`
**What:** RETRYABLE_STATUS includes 429 (Too Many Requests). On 429, the
function sleeps `backoff * backoff_factor**attempt + jitter`. But most APIs
(Alpha Vantage, NewsAPI, Binance) return a `Retry-After` header specifying
the mandatory wait. The code ignores it and waits the exponential default.

If the header says "wait 60s" and the code waits 2s, the retry will hit
another 429, and so on, burning both quota and latency.

**Fix shape:** Parse `Retry-After` (seconds or HTTP date) and use it as the
sleep time when present, clamped to a reasonable maximum.

### Finding 8.8 — MEDIUM [H]: `_cached` error path updates `time` to `now - ttl + RETRY_COOLDOWN`
**Where:** `shared_state.py:103-104`
**What:** On exception, the code sets `_tool_cache[key]["time"] = now - ttl
+ _RETRY_COOLDOWN`. This effectively makes the cache entry expire in
`_RETRY_COOLDOWN` (60s) from now. But the cached `data` is still stale.
The intent is to prevent immediate retry but still serve stale. This works,
but has a subtle bug: the `time` field is used for stale-age calculations
elsewhere (`now - _tool_cache[key]["time"]`). By artificially setting the
time to the past, the stale-age calculation reports wrong ages.

**Fix shape:** Separate `refresh_after` and `written_at` fields. Stale-age
uses `written_at`; retry-cooldown uses `refresh_after`.

### Finding 8.9 — MEDIUM [M]: Rate limiter granularity is per-instance, not per-endpoint
**Where:** `shared_state.py:159-186`
**What:** Single `_binance_limiter` at 600/min is shared between all
Binance calls (spot klines, FAPI klines, FAPI ticker, FAPI orderbook,
futures_data endpoints). Binance actually has separate rate limits per
endpoint family. A spike in orderbook calls (from metals_loop fast-tick)
will throttle kline fetches unnecessarily and vice versa.

**Fix shape:** Separate limiters for `binance_spot_klines`, `binance_fapi_klines`,
`binance_fapi_ticker`, `binance_fapi_orderbook`. Or at minimum, one limiter
per endpoint family based on Binance's public rate limit doc.

### Finding 8.10 — LOW [H]: `_tool_cache` eviction can oscillate between two sizes
**Where:** `shared_state.py:51-63`
**What:** When the cache exceeds `_CACHE_MAX_SIZE=256`, the eviction loop
drops expired entries first, then drops 25% of remaining if still over
limit. This is cheap but oscillatory: every insertion after crossing 256
triggers a full eviction pass. With a 256-entry cache at steady state and
many per-cycle key insertions, eviction runs O(cycle_keys × 256) per cycle.

**Fix shape:** Use `OrderedDict` with `.popitem(last=False)` for O(1) LRU
eviction, or raise the high-water mark to avoid thrashing at the boundary.

### Architectural critique — data/infra

The file_utils and shared_state layers are **careful but not paranoid
enough** given the trading context. Atomic writes don't quite achieve the
guarantee their name implies (no parent-dir fsync). `load_json` returning
default on all errors is a convenience that hides data loss. The cache layer
has correct locking but time/error interactions that leak into stale-age
calculations.

The rate limiter tracking being in-memory means restarts re-use daily
quotas from scratch, which is the opposite of what the quota-protecting
limiters should do.

The **single most important improvement** for this subsystem: split
`load_json` into two variants (optional vs required) and audit every
caller to use the strict version for critical state files. The current
silent-default behavior is a corruption risk masquerading as robustness.

---

## Cross-subsystem themes

### Theme A — silent failures dominate the failure mode catalog
Every subsystem has multiple findings of the form "X fails silently on Y
exception" (1.12 earnings gate, 2.7 trade detection, 4.1 silver fast-tick,
6.10 econ calendar exhaustion, 7.1 empty API response, 8.3 load_json).
The system consistently prefers "degrade gracefully" over "fail loudly" —
which is the right call for non-critical paths but the WRONG call for
safety-critical paths (earnings gate, stop-loss volume check, session
expiry, silver fast-tick).

**Cross-cutting recommendation:** Add a "degradation severity" metadata
tag to each error handler (e.g., `@degrades(severity="critical")`) and
force severity=critical handlers to also emit a health event. Operators
could then see "X has degraded N times today" at a glance.

### Theme B — hardcoded dated audit findings embedded throughout the code
Signal engine gating lists, horizon weight dicts, correlation groups,
regime penalties, FOMC/CPI calendars, correlation priors — all are
snapshots from specific dates. When the market shifts, these all become
wrong simultaneously. There's no "audit on a schedule" mechanism, just
"find it when it breaks".

**Cross-cutting recommendation:** Every hardcoded audit-derived constant
should be in a single `data/audit_constants.json` file with a `"audited_on":
date` field per entry. A monthly job should re-audit and emit a Telegram
summary of constants that have drifted from current data.

### Theme C — multi-process / multi-file state fragmentation
Main loop, metals loop, and future bots (GoldDigger, Elongir) each have
their own state files, their own accuracy systems, their own singleton
locks. Cross-process coordination is weak. Shared files (health, fundamentals
cache) are race-prone.

**Cross-cutting recommendation:** Declare a "state ownership contract" per
file (which process owns writes, who can read). Files with shared ownership
need cross-process file locking or a single-writer daemon.

### Theme D — the Layer 2 LLM is trusted to catch issues that the code should enforce
Trade guards emit "warning" but never "block". Autonomous mode does not
enforce them at all. Drawdown check is computed but not enforced. The
design pattern is "the LLM will read these and do the right thing", which
only works when Layer 2 is available.

**Cross-cutting recommendation:** Define the minimum set of HARD limits
that must be enforced by code regardless of Layer 2 state (e.g., "never
open a new position while drawdown > 25%", "never trade a ticker with
4+ consecutive losses"). Make them explicit, tested, and un-overridable
by LLM decisions.

---

## Summary table (top critical findings across all subsystems)

| ID | Severity | Finding | Subsystem |
|---|---|---|---|
| 1.1 | CRITICAL | Regime-gated signals can't recover through data | signals-core |
| 2.1 | CRITICAL | Singleton lock silently no-ops on non-Windows | orchestration |
| 3.1 | CRITICAL | `load_state` wipes tx history on corrupt JSON | portfolio-risk |
| 4.1 | CRITICAL | Silver fast-tick silently no-ops on errors | metals-core |
| 4.2 | CRITICAL | `place_stop_loss` no volume constraint enforcement | metals-core |
| 5.1 | CRITICAL | Dual avanza implementations can double-fire orders | avanza-api |
| 6.1 | CRITICAL | Econ dates hardcoded only for 2026-2027 | signals-modules |
| 1.2 | HIGH | Force-HOLD discards information for biased sub-50% signals | signals-core |
| 1.3 | HIGH | Market-health penalty creates structural SELL bias at bottoms | signals-core |
| 2.2 | HIGH | `_startup_grace_active` test-isolation leakage | orchestration |
| 3.2 | HIGH | `portfolio_value` cash-only on invalid fx_rate | portfolio-risk |
| 3.3 | HIGH | 2×ATR stop does not account for leverage (contradicts user rule) | portfolio-risk |
| 8.1/8.2 | HIGH | `atomic_write_json` doesn't fsync file or parent dir | infrastructure |

---

