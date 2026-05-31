# Adversarial Codebase Review — SYNTHESIS — 2026-05-31

Authoritative roll-up of the 8-subsystem adversarial review. Per-subsystem
detail lives in `docs/reviews/00..08-*.md`; this doc consolidates, de-dups,
cross-references, and prioritizes.

## Method (so a future reader can trust the confidence levels)
- Isolated worktree `Q:/fa-rev-0531` off `main@9ecdeecc`; empty-baseline branch
  `review/baseline-empty` for full-content diffs. Live loops untouched.
- 8 fresh review subagents in background (5× `pr-review-toolkit:code-reviewer`
  on the money/reliability-critical subsystems, 3× `caveman:cavecrew-reviewer`
  on the high-volume sweeps), each briefed with this project's real failure
  history (the 3-week auth outage, the Mar-3 stop-loss instant-fill, atomic-I/O
  and ISK-only invariants) rather than generic smells.
- Independent main-thread pass on the 3 riskiest subsystems + a cross-critique
  step that marked every headline finding **confirmed / plausible / needs-
  verification / refuted** (`00-independent-pass-and-crosscritique.md`).
- Seeded the reviewers with the 3 live unresolved critical-error categories so
  they'd root-cause them, not just pattern-match.

## Headline numbers
~81 substantive findings (1 refuted on cross-critique). **9 P0**, ~30 P1, the
rest P2/P3. Confidence is mixed and labeled per finding — this is a *map of where
to look*, not a to-fix list to apply blindly.

## What is VERIFIED HOLDING (the good news — do not "fix" these)
- **Stop-loss API** routes through `/_api/trading/stoploss/new` in **every** live
  path (avanza-api + metals-core both verified). The Mar-3 instant-fill class is
  absent.
- **`file_utils` atomic writes** are correct (tmp + fsync + `os.replace` same-fs).
- **Grid caps** (6500 SEK global / 1200 SEK leg) enforced per-tier live, fail-
  closed, single-threaded tick — no concurrent-fill breach.
- **Dashboard auth**: constant-time `hmac.compare_digest`, CF JWT verified.
- **HTTP retry**: 4xx are fatal (not retried) → no duplicate-order risk.
- **DST math** in `market_timing.py`: aware-UTC throughout, no offset bug.
- **`--bare` flag** correctly absent from all Layer 2 spawn sites.
- Order-result checking requires `orderRequestStatus=="SUCCESS"` + real orderId —
  closes the silent-HTTP-200-error-body class.

---

## SYSTEMIC THEMES (the real value — emergent only across subsystems)

### THEME 1 — The critical-error / alerting layer is the least trustworthy part of the system  ★ top priority
**All three seeded critical-error categories are false-positive or mislabeled:**
| Category | Reality | Source |
|---|---|---|
| `contract_violation` (139 rows) | timestamp-provenance race (journal stamped with signal-snapshot time vs trigger `now()`). **Already fixed 2026-05-30 (Precondition 6); ZERO fires since 2026-05-29.** Backlog is stale. | 02 + 00/A1 |
| `accuracy_degradation` (×4) | regime shift (rally→pullback moves many signals 15-30pp at once), not signal degradation; non-overlapping-window diff + ≥3-drop CRITICAL classifier | 01 P1 |
| `avanza_account_mismatch` (daily 06:01) | session-expiry mislabeled — verifier downgrades `AvanzaSessionError` to `fetch_failed` under an account-named critical | 05 root cause |

Consequence: `scripts/check_critical_errors.py` shows 31 "unresolved" that are
mostly noise → an operator cannot triage real failures, and `PF-FixAgentDispatcher`
spawns Claude fix-agents on already-fixed phantom problems (budget burn).
**Cheapest highest-value fixes:** (a) append resolution lines for pre-2026-05-30
`contract_violation` entries; (b) auto-resolve a category when a fix post-dates
all its unresolved entries or it hasn't re-fired in N days; (c) rename
session-expiry critical accurately; (d) regime-gate the accuracy alert.

### THEME 2 — Atomic writes are solid; cross-process read-modify-write is NOT
`atomic_write_json` prevents torn files but does nothing about lost updates. Every
money-state mutation does load→mutate→write with only an in-process
`threading.Lock`, while CLAUDE.md documents 3+ OS processes (main loop, metals
loop, Layer 2 subprocess, dashboard) writing the same files. Confirmed in:
warrant_portfolio (03 P0-1), portfolio_mgr (03 P0-2), trade_guards TOCTOU
(03 P2-4), signal_engine `_persistence_state` (01 P2). **One fix serves all:**
wrap money-state RMW in the existing `file_utils.jsonl_sidecar_lock` cross-process
primitive.

### THEME 3 — Silent-failure / fail-open is the dominant bug shape
auth_error writes no journal stub (02 P0-#1); grid_fisher swallows
AvanzaSessionError into a generic placement failure (05 P1-2); data caches serve
stale silently with no freshness flag (07, theme); risk gates compute but don't
enforce (03 P0-3); `get_positions()` returns `[]`/merged on error,
indistinguishable from flat (05 P2-4, 04 P0-1). The system prefers "look healthy"
over "fail loud" — exactly what hid the 3-week 2026 auth outage.

### THEME 4 — `avanza_session.get_positions()` is account-blind — one root, two P0/P1 consequences
The single function returns positions merged across ISK **and** pension with no
filter. (a) metals-core P0-1: grid_fisher sees pension holdings of the same
orderbook id as phantom ISK inventory → real SELL of unheld units + EOD
short-sell (violates `feedback_isk_only`). (b) avanza-api P2-4: TOTP path returns
`[]` on error = "flat" ambiguity. **Fix once:** add an `account_id` filter to
`get_positions()` and thread `self.account_id` through every reconcile call site.

---

## CONSOLIDATED P0 TABLE

| # | Subsystem | Finding | Confidence | Note |
|---|---|---|---|---|
| P0-A | signals-core | `outcome_tracker._fetch_historical_price` returns close of the 1h bar *containing/after* `target_ts` while entry is the exact `entry_ts` price → realized horizon overstated up to ~1h (negligible 1d, ~+17-33% at 3h, up to +100% at 1h). Biases every short-horizon accuracy/IC/weight + Mode-B probability the user trades. | **VERIFIED** (caller convention checked) | Fix: `interval="1m"`, bar **open** at `target_ts`. Single most important correctness bug. |
| P0-B | metals-core | `grid_fisher` reconciles against ALL accounts (`get_positions()` no filter) → pension holding of same orderbook id = phantom ISK inventory → real SELL of unheld units + EOD short-sell. Violates `feedback_isk_only`. | **CONFIRMED + RECURRING** (same as 2026-05-25 review, still unfixed) | Top execution priority. Fix = THEME 4. Until fixed, don't run grid on accounts sharing orderbook ids. |
| P0-C | portfolio-risk | `warrant_portfolio.record_warrant_transaction` unlocked cross-process read-modify-write on the LEVERAGED book → lost average-in corrupts position size + stop/knockout reference. | **CONFIRMED (class)** | Fix = THEME 2. Highest-leverage book = highest impact. |
| P0-D | portfolio-risk | `portfolio_mgr.update_state` in-process `threading.Lock` only; docstring oversells cross-process safety → lost updates on Patient/Bold cash+holdings. | **CONFIRMED (class)** | Fix = THEME 2. `main.py:796 save_state` gating limits exposure today. |
| P0-E | portfolio-risk | Drawdown circuit breaker computed (`check_drawdown`) but no consumer in the risk subsystem enforces it on entries → fails open on >20% DD. | **NEEDS VERIFICATION** | Agent found no consumer in 13 files; enforcement may live in an execution module. **Verify before fixing.** |
| P0-F | orchestration | `auth_error` completion writes no journal stub (timeout/failed/incomplete all do) → a real auth outage leaves zero journal record, relying on fragile status-suppression. | **CONFIRMED-plausible** | Explains 23/139 auth_error contract_violations. Fix: write a stub like the other terminal statuses. |
| P0-G | orchestration | Layer 2 auth detection regex-scans a rotation-fragile 16-line slice of the shared, mid-run-rotated `agent.log` (no `--output-format json`) → CLI banner/spinner/rotation can evade → reopens the 3-week-outage class. | **CONFIRMED-plausible** | Fix: parse structured JSON envelope, or a dedicated per-invocation capture file; treat scan exception as "unknown, don't downgrade to success". |
| P0-H | signals-core | MSTR cross-ticker consensus cache keyed by `ticker` only and overwritten by all 7 horizons each cycle → `btc_proxy` may read a stale/wrong-horizon BTC vote. | **PLAUSIBLE — verify** | Matches CLAUDE.md MSTR btc_proxy design. Key cache by `(ticker, horizon)`, write only the 1d decision horizon. |
| P0-I | data-external | `fx_rates` out-of-range sanity check falls through without updating cache; subsequent calls serve a stale USD/SEK rate silently → scales every SEK position. | **ACCEPT (borderline P0/P1)** | Fix: flag/raise on bad rate; never silently serve stale FX. |

## P1 highlights (full lists in per-subsystem docs)
- **signals-core**: `accuracy_degradation` regime false-alarm (THEME 1); utility-boost applied to raw accuracy not final weight (no-op for mature directional signals, over-weights immature); directional rescue bypasses circuit-breaker high-sample floor; yfinance stock backfill returns first-available bar (wrong `change_pct`); ICIR computed on entry-iteration-order (not time-order) series → IC stability gate runs on noise.
- **orchestration**: `check_layer2_journal_activity` documented "never raises" but unguarded → naive-vs-aware journal `ts` throws TypeError → `main.py` blanket-except silently drops the **entire cycle's** contracts; deferred-trigger lost between detection and dispatch when `batch_window_s>0`; stack-overflow auto-disable is effectively permanent with no recurring alert; crash-alert suppression can hide ~8h of crash-looping.
- **portfolio-risk**: `kelly_metals` leverage math inverted (can recommend 95% of buying power into one 5x cert) — **note: ADVISORY only; live grid uses fixed 1200 SEK/leg, swing uses leverage-scaled sizing — not a live-execution P0, but fix the recommendation math**; Kelly consensus-accuracy path has no sample-size gate (5-sample 100% → full Kelly); VaR/CVaR silently drops/ inverts shorts; concentration uses stale entry cost when live price missing.
- **metals-core**: swing trader SELLs full volume before cancelling its full-volume hardware stop → broker rejects → exit silently fails (position can't exit while stop is live); hardware-trailing-stop failure at fill leaves a naked leveraged position with no retry; grid rotation cancels old stop before confirming the new one (multi-cycle naked window).
- **avanza-api**: no proactive session-expiry check before order POST (relies on reactive 401); cached Playwright context skips the only expiry check; metals/golddigger page path accepts arbitrary `account_id` with no whitelist (account isolation held only by a hard-coded constant).
- **data-external**: pervasive stale-but-silent caches/fallbacks (price_source, onchain, metals_cross_assets, funding_rate, sentiment VIX) with no freshness marker; AlphaVantage 429 burns daily budget untracked; Binance error-dict parsed as kline array → TypeError.
- **infrastructure**: `heartbeat()` keepalive masks a 59-299s mid-cycle stall (process-alive ≠ working).

---

## PRIORITIZED REMEDIATION BACKLOG

### Tier 0 — verify first (don't fix blind)
1. **P0-E** drawdown-breaker enforcement: grep the execution path for a `breached` consumer. If none → it's a live P0; if found → downgrade to a doc fix.
2. **P0-H** MSTR consensus cache: runtime-check what `btc_proxy` actually reads.

### Tier 1 — highest value, low blast radius (additive, testable)
3. **THEME 1**: resolve the stale `contract_violation` backlog + add auto-resolve logic to `check_critical_errors.py` + rename the session-expiry critical accurately. (Stops fix-agent budget burn + restores triageability.)
4. **P0-A** outcome_tracker forward-shift fix (1m bar open). Re-backfill short-horizon outcomes after. Correctness foundation for every accuracy-gated decision.
5. **P0-F / P0-G** orchestration: auth_error journal stub + structured-JSON auth detection.

### Tier 2 — concurrency hardening (one primitive, several call sites)
6. **THEME 2**: cross-process `jsonl_sidecar_lock` around money-state RMW (warrant_portfolio P0-C, portfolio_mgr P0-D, trade_guards P2-4).
7. **THEME 4 / P0-B**: `account_id` filter on `get_positions()` + thread through grid_fisher reconcile. Recurring P0 — prioritize within concurrency work.

### Tier 3 — risk/exec correctness
8. metals swing: cancel stop before SELL; rearm-retry on stop-placement failure; place-new-before-cancel-old on rotation.
9. kelly_metals leverage math (advisory) + Kelly sample-size gate; VaR short handling.
10. data-external freshness-flag pattern (uniform: every cache/fallback carries `_age_seconds`/`_source`, decision code rejects stale).

### Tier 4 — coverage gaps / re-review
11. **Dedicated signals-modules re-review** — the cavecrew pass under-delivered (1 finding for 58 files, that one REFUTED). Split into ~3 batches of ~20 modules, focus on lookahead/shift-direction + NaN/empty guards + vote sign. Confidence on signals-modules is currently LOW.

## Confidence / caveats
- `pr-review-toolkit` subsystems (01-05) are high-confidence: file:line + causal
  chains, cross-checked.
- `00` cross-critique flags which P0s are verified vs plausible vs needs-verify.
- signals-modules (06) is explicitly LOW confidence (see Tier 4).
- This review shipped **zero production code** by design (/fgl docs-only review).
  Every fix above is a hypothesis to be implemented test-first in a later session.
