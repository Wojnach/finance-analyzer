# FGL Adversarial Review — SYNTHESIS (2026-05-29)

8 subsystems, each reviewed by a fresh independent Claude Code subagent
(6× pr-review-toolkit:code-reviewer, 2× caveman:cavecrew-reviewer) + one independent
orchestrator pass focused on cross-subsystem seams and the live incident. Findings
cross-critiqued and deduped below. **Review-only — no code was changed.**

## Tally

| Subsystem        | P0 | P1 | P2 | P3 | Reviewer |
|------------------|----|----|----|----|----------|
| signals-core     | 0  | 4  | 8  | 5  | code-reviewer |
| orchestration    | 1  | 4  | 8  | 6  | code-reviewer |
| portfolio-risk   | 1  | 6  | 8  | 5  | code-reviewer |
| metals-core      | 0  | 4  | 5  | 3  | code-reviewer |
| avanza-api       | 2  | 8  | 0  | 0  | cavecrew-reviewer |
| signals-modules  | 0  | 3  | 8  | 5  | code-reviewer |
| data-external    | 0  | 3  | 7  | 5  | code-reviewer |
| infrastructure   | 3  | 4  | 3  | 0  | cavecrew-reviewer |
| orchestrator pass| 1  | 4  | 2  | 0  | (me) |
| **TOTAL**        | **8** | **40** | **49** | **29** | 126 findings |

> Counts are reviewers' self-assigned severities. **Post-synthesis verification downgraded
> 2 of the 8 P0s** (`file_utils.py:269`, `telegram_poller.py:361` — see §2) to P1, giving
> ~6 verified P0 + 2 over-rated. The cavecrew (infrastructure/avanza) reviewer ran hotter on
> severity than the code-reviewer agents; weight its P0 labels accordingly.

Headline: **the consensus hot path, metals execution invariants, and atomic-write core
are well hardened** (years of incident-driven patching shows). The risk has migrated to
(a) the *detection* layer that is supposed to catch silent failures — now chronically
red and ignored, and (b) gate/lock *bypasses* and *success-status-but-broken* paths.

---

## 1. The headline — silent-failure DETECTION is itself failing (cross-confirmed P0)

Two independent passes (orchestrator + orchestration subagent) converged on the same
root cause for the **39 unresolved `critical_errors.jsonl` entries** flagged at session
start (28× `layer2_journal_activity`, 9× `accuracy_degradation`, 4× `avanza` session-expired):

- **`main.py:890` + `autonomous.py:83`** (orchestration subagent, P1): in autonomous-first
  mode, if `autonomous_decision` raises *before* its journal write, the swallow-all
  wrapper hides it and main logs `autonomous_{why}` — a status NOT in the contract's
  legitimate-skip set — so `check_layer2_journal_activity` (`loop_contract.py:277`) fires
  a CRITICAL **indistinguishable from a real silent Layer 2 failure.**
- **Orchestrator pass (P0, meta):** `critical_errors.jsonl` is append-only and resolved
  only by a manually-appended resolution line. Nobody appends them, so the system's #1
  silent-failure tripwire stands at 39 alerts. When the next real 3-week-outage-class
  failure arrives it is entry #40 in a list the operator has trained themselves to ignore.
  **Alert fatigue by design** — the detector built to catch the March–April auth outage
  is now in the exact desensitized state that let that outage run 3 weeks.

**Why this is the top priority:** every other P0 below is a single bug. This one silently
disarms the mechanism that is supposed to surface single bugs. Fix the detector's
trust-value first (auto-resolution + root-cause tagging + a failure-stub on the
autonomous-raise path), or the rest of this review's value decays as alerts pile up.

---

## 2. Confirmed P0 findings (8, deduped)

Spot-checked by orchestrator (✓ = independently verified at HEAD this session):

1. **orchestration — `multi_agent_layer2.py:181` + `agent_invocation.py:1071` ✓**:
   specialist + synthesis Claude subprocesses spawn via raw `subprocess.Popen`, bypassing
   `claude_gate.invoke_claude` entirely — no `CLAUDE_ENABLED` kill switch, no daily rate
   limiter, no `_invoke_lock` serialization. The master kill switch cannot stop specialist
   spawns; 3 specialists + main L2 + bigbet/iskbets can hit Claude concurrently. (Live only
   when `multi_agent` config flag is enabled — confirm before prioritizing.)
2. **infrastructure — `file_utils.py:269` — ⚠ DOWNGRADED to P1 after verification**:
   the subagent claimed the `msvcrt` sidecar lock is "never released on crash → held
   forever." **False premise** — the `finally` (lines 259-265) releases on normal/exception
   exit, and Windows releases `msvcrt` byte-range locks when the OS closes the handle on
   *process termination*. A crashed process does NOT leave a permanent lock. The REAL
   (lesser) risk: `LK_LOCK` is blocking with no caller timeout, so a **live-but-hung**
   holder can stall other JSONL writers. → add a bounded acquire + stale detection for the
   live-hang case, not a crash-reap. *(Orchestrator verified `atomic_write_json`/
   `atomic_write_text`/`atomic_append_jsonl` are otherwise correct.)*
3. **infrastructure — `telegram_poller.py:361` — ⚠ DOWNGRADED to P1 after verification**:
   the subagent claimed a read→write race "destroys API keys." **Already guarded** — the
   BUG-210 check at `:350` refuses the write when the loaded config has `<5` keys (catches
   the transient-unreadable `{}` case), and `atomic_write_json` resolves the symlink + writes
   atomically. Residual real risk is *architectural*: the poller still writes through the
   `config.json`-to-external-secrets symlink at runtime (rewrites the whole secrets file to
   change one notification flag), violating "never write config.json." → use a transient
   state file for notification mode. Real, but not the catastrophic-key-loss P0 as filed.
4. **infrastructure — `http_retry.py:76` — likely P1, not verified**: returns `None`
   without distinguishing fatal (401/400) from transient (503/timeout) → a bad token retries
   forever as transient. Design-robustness gap; severity P1. → return `(response, error_type)`.
5. **portfolio-risk — `warrant_portfolio.py:257`**: SELL on a `config_key` not in holdings
   (or over-sell) appends the transaction but never reduces units / records the over-sell →
   phantom holdings, ledger fails reconciliation. → refuse + log CRITICAL.
6. **avanza-api — `avanza_orders.py:379`**: order confirm saves `orderId="?"` placeholder when
   missing → unreferenceable order, later cancel/track fails. → reject missing orderId.
7. **avanza-api — `avanza_session.py:542,590`**: `place_buy_order` doesn't validate
   `orderbook_id` non-empty/numeric before POST. → validate before send.
   *(avanza-api line numbers reconstructed from the subagent's summary after its file-write
   failed — re-confirm lines before acting.)*
8. **orchestrator pass (meta) — standing-violation alert-fatigue trap** (see §1).

---

## 3. Recurring themes (the real value — patterns across subsystems)

### Theme A — "success status but broken" (the silent-failure family) — DOMINANT
Same shape in 4 subsystems: a layer reports OK while actually broken.
- orchestration: exit 0 + no journal; `loop_health.py:215` hardcodes `"status":"ok"`
  regardless of `ok=False` arg ✓ (any consumer keying on `status` never sees a failed cycle).
- avanza-api: `verify_session()`/`api_get`/`api_post` treat HTTP 200-with-empty-or-error-body
  as success; `orderRequestStatus="REJECTION"` returned as a placed order.
- data-external: `macro_context.py:297` fabricates `change_5d=0.0` on ^TNX outage;
  `crypto_precompute.py:185` substitutes `0.0` for missing funding (can't tell "0" from "absent").
- http_retry: exhausted-retry `None` indistinguishable from a config-fatal error.
**Shared fix:** a `require_valid_response()` / explicit error-typing convention so "broken"
can never wear "OK"'s clothes. This is the same class as the 3-week auth outage.

### Theme B — regime gates leaking directional votes
signals-modules found gates that should only SUPPRESS instead emitting directional votes
on noise: `amihud_illiquidity_regime.py:107` (RVOL→BUY/SELL), `choppiness_regime_gate.py:123`
(force `direction@0.35` when composite abstains), and the LLM regex-fallback at
`finance_llama.py:204`/`cryptotrader_lm.py:150` returning a directional vote at conf=0.50
from a malformed completion (biased toward BUY by the few-shot example). Orchestrator seam B3:
nothing at the engine level *enforces* that a registered regime-gate can only contribute HOLD
or a multiplier — it trusts each module to self-classify. **Engine-level guard recommended.**

### Theme C — leverage-blind risk math (real-money, metals)
portfolio-risk: `risk_management.py:373` ATR stop `max(2*atr,3%)` on the *underlying* = 15%
on a 5x cert (barrier-proximity class the metals rules explicitly warn against);
`kelly_metals.py:215` allows up to 95% of buying power into one 5x cert;
`kelly_sizing.py:270` uses gross (fee-excluded) pnl → overstated edge → over-betting.
metals-core: `EOD_EXIT_MINUTES_BEFORE=0` (`metals_swing_config.py:323`) means swing EOD-flat
can never fire in-hours (comment literally says "REVERT to 25") + hardcoded 21:55 CET close
ignores DST → overnight-warrant bleed, against the user's stated no-overnight preference.

### Theme D — cancel-before-replace leaves positions naked
metals-core grid fisher: `rotate_on_buy_fill` (`grid_fisher.py:1538`) and `eod_market_flat`
(`:1939`) cancel the protective stop/ladder *before* placing the replacement; on placement
failure the position is unprotected for ~60s (or through the illiquid close auction).
→ place-before-cancel / sell-before-cancel.

### Theme E — read-modify-write races on shared JSON under 8 workers
infrastructure: `health.py:64` heartbeat RMW without `_health_lock`; `shared_state.py:89,208`
dogpile-eviction without max-retry bounds (stuck key wedges for 120s, all 8 workers pile on);
`process_lock.py:39` append-mode open can let two fresh-file acquirers both win.
orchestrator seam B2: `_record_new_trades()` tied to in-process invocation lifecycle, so a
loop restart mid-trade hides the trade from overtrading guards.

---

## 4. Cross-critique (agreements, contradictions, false-positive risk)

- **Strong agreement (two independent passes):** the journal-contract root cause (§1) —
  orchestrator traced the symptom; orchestration subagent independently pinpointed the
  `autonomous.py:83` raise-before-journal line. High confidence.
- **Internal contradiction flagged:** signals-modules' headline ("no module silently injects
  a wrong-direction vote") is contradicted by its OWN P1 #1 (amihud directional) and #3
  (LLM regex→0.50 directional). The headline overstates safety; the per-finding detail is
  correct. Treat Theme B as real, not reassured-away.
- **Refuted / downgraded:** orchestrator seam B4 ("stale cache served as fresh = P0") is
  REFUTED for the live Tier-1 price path — data-external confirmed `_cached()` caps staleness
  at `ttl*_MAX_STALE_FACTOR` then returns `None`, FX has explicit staleness alerts, and there
  is no Binance `10m` interval. The stale-data theme survives only on the **4h precompute /
  shadow path** (`mstr_precompute` yfinance bypass, fred fabricated 0.0) → P1, lower blast radius.
- **Confidence caveats / re-confirm before acting:**
  - avanza-api line numbers are reconstructed from the subagent's summary (it failed to write
    its file); the *findings-class* is sound but exact lines need re-checking.
  - infrastructure `process_lock.py:39` "two processes both lock same byte" and the `msvcrt`
    double-lock claim are plausible but partly speculative — verify with a concurrency test
    before treating as P1.
  - `multi_agent_layer2` P0 is only live if the `multi_agent` config flag is on — gate the
    priority on that.
- **Spot-checks that PASSED (not hallucinated):** `loop_health.py:215` status-hardcode ✓;
  `multi_agent_layer2.py:181` raw Popen bypass ✓; `file_utils.atomic_write_json` correctness
  (no finding) ✓; `warrant_portfolio.py:257` over-sell, `autonomous.py:83` raise-before-journal
  (both corroborated by their subagents' detail).
- **Spot-checks that FAILED verification (severity corrected here):** `file_utils.py:269`
  "lock held forever on crash" — wrong premise (OS releases on process death; `finally`
  releases on exit) → P1. `telegram_poller.py:361` "race destroys API keys" — BUG-210 `<5
  keys` guard already prevents it → P1 (residual = architectural coupling only). Both were the
  cavecrew reviewer's P0s. This is the premortem's "hallucinated/over-rated finding" mode
  caught by the synthesis layer doing its job.

---

## 5. Seam risks (coverage gaps a per-subsystem reviewer can't see)
- **B1 (orchestration↔infra):** `agent_invocation._scan_agent_log_for_auth_failure` byte-offset
  scan can miss a "Not logged in" line if hourly `log_rotation.rotate_text()` rotates+regrows
  agent.log past the offset mid-run. Scan by invocation-id, not byte offset.
- **B2 (orchestration↔portfolio-risk):** trade-recording tied to invocation lifecycle → loop
  restart mid-trade bypasses overtrading guards.
- **B3 (signals-core↔signals-modules):** no engine-level enforcement that regime gates stay
  non-directional (see Theme B).

---

## 6. Prioritized fix queue (recommendation — NOT executed; review-only deliverable)

**Tier 0 — restore the detector (do before anything else):**
1. Auto-resolve `critical_errors.jsonl` entries when a later cycle observes the invariant
   PASS; tag each `layer2_journal_activity` violation with its root-cause split. (§1)
2. Failure-stub on the `autonomous_decision` raise-before-journal path. (`autonomous.py:83`)
3. Re-establish the Avanza session (expired since 05-23) + make the daily mismatch alert escalate.

**Tier 1 — P0 bugs:**
4. Stop runtime writes to the config.json symlink (`telegram_poller.py:361`).
5. Stale-lock reap for the JSONL sidecar lock (`file_utils.py:269`).
6. Route multi-agent specialists through `claude_gate` (or gate the feature off until they are).
7. Refuse phantom/over-sell in `warrant_portfolio.py:257`; reject `orderId="?"` + validate
   `orderbook_id` in avanza order flow.
8. `http_retry` fatal-vs-transient error typing.

**Tier 2 — themed hardening:** shared `require_valid_response()` (Theme A); engine-level
regime-gate direction guard (Theme B); leverage-aware stops + Kelly caps + restore
`EOD_EXIT_MINUTES_BEFORE` + DST-aware close (Theme C); place-before-cancel in grid fisher
(Theme D); lock the health RMW + dogpile retry bound (Theme E).

**Tier 3:** P2/P3 robustness + repo hygiene (10+ stale worktrees under `.worktrees/` and
`.claude/worktrees/` — GUIDELINES rule 9 cleanup not happening).

---

## 7. Methodology & limitations
- **Mechanic:** empty-baseline diff (`git diff fgl-empty-baseline HEAD -- <files>`) instead of
  8 full worktrees — data/ is 1.5G, 8 checkouts ≈ 12GB. GUIDELINES permits "worktree **or**
  branch." Reviewers read source directly + emitted findings as their result; no collision.
- **Coverage:** signals-modules covered the 26 active/LLM modules, not all 73 (44 are
  DISABLED/force-HOLD). data-external covered live path + key precompute, not every collector.
  Untouched: golddigger/, elongir/, mstr_loop/, strategies/, dashboard export/blueprint,
  most scripts/. A follow-up session should sweep those.
- **Confidence:** line numbers from the avanza-api reviewer are reconstructed (file-write
  failed); all P0/P1 should be re-confirmed at the cited `file:line` before code changes.
- Per-subsystem detail lives in the sibling `*.md` files; orchestrator-only findings in
  `_my-independent-pass.md`.
