# Dual Adversarial Review — Synthesis (2026-05-10)

Two independent reviewers (Claude Opus 4.7, Codex GPT-5.4 xhigh) audited the
finance-analyzer codebase across 8 subsystems, then critiqued each other's
findings. This doc consolidates the outcome.

Method, partition, and artefact layout: `docs/ADVERSARIAL_REVIEW_2026-05-10.md`.
Raw evidence: `data/adv-2026-05-08/` (independent reviews) and
`data/adv-2026-05-10/` (cross-critiques).

## Headline Result

| | Codex findings | Claude findings |
|---|---|---|
| Total raised | ~62 | ~38 |
| Confirmed by counter-reviewer | **30** | **5** |
| Partial (real signal, wrong cause/line/sev) | 4 | 32 |
| False-positive | 28 | 83 |
| New on inspection | — | 21 |

Codex held up far better under cross-critique than Claude. Claude's reviews
were terse (50-200 lines per subsystem) and made enough wrong claims about
specific line numbers, control flow, and gate semantics that Codex was able to
falsify most of them with literal code citations. Codex's reviews were dense
(5-12k lines per subsystem incl. trace logs) and grounded findings in concrete
file citations that survived re-inspection. **Treat the 30 bilateral-confirmed
items as the high-conviction queue; treat lone-Claude findings with skepticism
unless re-validated.**

This asymmetry is worth recording. It is not a generic claim about either
model — it reflects the depth budget each reviewer was given. The lesson for
future review sessions: cap the codex pass at the same depth/length as the
claude pass, or grant claude an equivalent budget; otherwise the cross-critique
collapses asymmetrically and you cannot tell whether the model or the budget
won.

## High-Conviction Remediation Queue (bilateral-confirmed)

Ordered by severity then by blast radius. Severity reflects conviction after
cross-critique, which may differ from either reviewer's original tag.

### P0 / P1 — fix soon, can affect live trading

1. **`portfolio/avanza_orders.py:348` — order placement is non-atomic**
   The order is sent to Avanza before the local journal write. A crash between
   the two leaves the broker holding a position the system does not know about.
   Fix: write a "pending" journal entry with idempotency key before the API
   call, finalize after.

2. **`portfolio/kelly_sizing.py:90-104` — sells matched to wrong entry lots**
   When computing per-trade win/loss for Kelly, sells are paired against a
   single blended entry price rather than the lot they exit. Mis-attributed
   PnL distorts the win-rate input and therefore the Kelly fraction.
   Fix: FIFO-match sell quantity to entry lots before computing per-trade PnL.

3. **`portfolio/monte_carlo.py:205-219, 328` — stop-hit probability uses
   terminal price, not path crossings**
   `drawdown_probability()` checks the simulation endpoint instead of any
   barrier touch along the path. Real stops trigger on first crossing, so the
   reported stop-hit prob systematically understates risk for time-extended
   positions. Fix: replace terminal `<=` check with along-path min crossing.

4. **`portfolio/trade_guards.py:290-296, 189-229` — position-rate guard counts
   scale-ins as new positions**
   Every BUY appends to the rate counter, including adds to existing
   positions. Guard 3 ("max N new positions per window") therefore over-counts
   and fires on legitimate scale-ins. Fix: only count BUYs that open a fresh
   position in a previously-flat ticker.

5. **`portfolio/warrant_portfolio.py:100-103` — no floor on warrant value**
   A 5x warrant on a 30% adverse underlying move yields a negative computed
   value because the leverage formula has no `max(0, ...)` guard. Equity-curve
   and risk computations downstream see negative SEK. Fix: floor at zero;
   knockout is the loss cap.

6. **`portfolio/fin_snipe.py:160-166` — BEAR ladder built from long-side
   defaults**
   `build_intraday_ladder()` is invoked without a direction sign, so BEAR MINI
   ladders use LONG entry/exit math. The system can recommend short-product
   trades using long-product price logic. Fix: thread `direction_sign` through
   to the ladder builder and invert.

7. **`portfolio/fin_snipe_manager.py:492-497` — exit translation no direction
   sign for BEAR holdings**
   `translate_underlying_target()` invocation lacks the BEAR sign flip. Exit
   prices for short products come out inverted. Fix: pass `direction_sign` and
   negate appropriately.

8. **`portfolio/fin_fish.py:732-735` — bare `pass` lets knocked-out BEAR
   MINIs through**
   The barrier-distance check uses `pass` (no-op) instead of `continue` for
   the BEAR-after-barrier branch. `evaluate_warrants()` can return an
   already-knocked-out instrument as a tradeable candidate. Fix: replace
   `pass` with `continue`.

9. **`portfolio/agent_invocation.py:236-240` — stock-trigger ticker parser
   misses delimiter cases**
   `_extract_ticker()` does not handle reason strings used by stock triggers,
   so the agent gets invoked with the wrong ticker tag and Layer 2 reads
   compact context for a different instrument. Fix: extend the parser regex
   to cover the actual reason-string vocabulary.

10. **`portfolio/trigger.py:230-231` — suppressed ranging consensus consumes
    the baseline**
    When a ranging-regime consensus is suppressed, the trigger baseline still
    advances, so the next material change is silently swallowed. Fix: treat
    suppressed rows as if no event happened (don't consume baseline).

11. **`portfolio/forecast_accuracy.py:322-327` — silent JSONL row drop at
    `max_entries`**
    The loop breaks at `max_entries` updated, then rewrites the file with only
    the prefix processed. All later rows (>500 by default) are deleted.
    Permanent forecast-history loss the first time the file grows past the
    cap. Fix: append the unprocessed tail to `modified_entries` before
    rewrite, or stream-update in place.

12. **`portfolio/accuracy_stats.py:150-153` — return stale SQLite when JSONL is
    ahead**
    `load_entries()` returns DB rows as soon as the DB has any data. The
    code's own warning ("SQLite may lag") acknowledges JSONL can be ahead, but
    callers (`signal_accuracy()`, accuracy gates in `signal_engine`) silently
    use the stale source. Fix: when JSONL is fresher (compare row counts or
    last-ts), prefer JSONL.

13. **`data/metals_loop.py:212-217` — missing support modules on fresh
    checkout**
    `metals_loop.py` imports `metals_shared` and other helpers that are not
    part of the committed tree on fresh worktrees, breaking import. Fix:
    commit the helpers or restructure imports to be self-contained.

14. **`portfolio/avanza_session.py:84` — expiry boundary off-by-one**
    `session_remaining_minutes()` returns 0 (expired) at the exact expiry
    instant; auth check elsewhere uses `> 0` which falsely flips to "not
    expired" for the same instant due to rounding. Fix: standardise on either
    `>= 0` or strict `>` across the entire auth-gate code path.

15. **`portfolio/avanza_session.py:82-86` — session expiry not re-checked on
    every API call**
    Auth state is cached at session start but not re-validated before each
    HTTP call. Long-running sessions can issue requests minutes after expiry.
    Fix: insert a cheap `_check_expiry()` at the top of every `_request()`.

16. **`portfolio/avanza_session.py:280-286` — `_get_csrf()` two-path race**
    Two lock-acquisition paths can both observe the same stale CSRF cookie
    and re-fetch concurrently, with the second writer overwriting the first.
    Fix: collapse to a single double-checked-locking pattern under a single
    lock.

17. **`portfolio/avanza_orders.py:115-142` — confirm token logged at INFO**
    The Avanza confirm token (single-use auth credential) is included in INFO
    logs that go to disk and Telegram fan-out. Fix: redact before logging
    (`token[:4] + "***"`).

18. **`portfolio/avanza_orders.py:186` — unguarded `datetime.fromisoformat()`
    on `order["expires"]`**
    Bad/missing field crashes the order-tracking loop on a non-essential
    parse. Fix: wrap in try/except and treat parse failure as "expired".

19. **`portfolio/avanza_orders.py:268-269` — Telegram offset omitted at 0**
    The first Telegram poll passes no offset because the code stringifies
    `offset` only when truthy; offset 0 (a valid value after a deletion) is
    therefore not sent. Fix: always pass the offset, including 0.

20. **`portfolio/data_collector.py:333-338` — timed-out timeframe pool**
    `ThreadPoolExecutor.shutdown(wait=True)` is called after `cancel()`, but
    cancel does not stop already-running futures. The shutdown blocks on the
    slow worker, defeating the per-timeframe timeout. Fix: track futures and
    skip waiting on those that did not return within the budget.

21. **`portfolio/market_health.py:446-450` — FTD age stored as window offset**
    FTD age is recorded as an index into the lookback window, so when the
    window slides one bar the age silently drifts. Fix: persist the FTD
    timestamp in calendar terms and recompute age each evaluation.

22. **`portfolio/telegram_poller.py:361` — `/mode` clobbers `config.json`
    symlink**
    `config.json` is a symlink to an external file (with API keys); the
    `/mode` handler calls `atomic_write_json` on the symlink path, which
    replaces the link with a regular file and severs the external link.
    Fix: detect symlink, write to the link target instead.

### P2 — fix when convenient

23. **`portfolio/outcome_tracker.py:269-276` — daily close used for sub-day
    horizons**
    For YF-mapped tickers, intraday outcome backfill uses the daily-interval
    yfinance call and picks last-close-on-or-before target. The 1h/3h/4h/12h
    horizon outcomes are therefore EOD-close, not the intraday target. Stock
    accuracy stats are systematically wrong on short horizons. Fix: switch to
    1h/15m interval for sub-day horizons.

24. **`portfolio/trade_risk_classifier.py:105-111` — concentration scoring
    pre-trade only**
    Concentration gate evaluates current exposure, not post-trade. Lets
    concentrated entries through and penalises de-risking trades. Fix:
    evaluate the simulated post-fill book.

25. **`portfolio/monte_carlo_risk.py:368-388, 395-419` — drawdown probability
    vs holdings, not portfolio**
    Threshold is `5% of invested capital`, not `5% of total portfolio incl.
    cash`. A 20%-invested book trips `drawdown_5pct_prob` on a 5% holdings
    move that is only a 1% portfolio drawdown. Fix: divide by total portfolio
    value.

26. **`portfolio/market_health.py:206-210` — FTD cold-start defaults to
    `correcting`**
    On fresh deploy, with no snapshot, the state machine seeds at
    `STATE_CORRECTING` and re-seeds `recent_high` from the lookback. In a
    sustained uptrend the drawdown is ~0 so the state never leaves
    `correcting`. Fix: reconstruct state from the historical series instead
    of one-bar update.

27. **`portfolio/avanza_session.py:82` — naive `expires_at` parse**
    `session_remaining_minutes()` does not handle naive datetimes, so a
    timezone-stripped `expires_at` mis-computes against a TZ-aware now().
    Fix: assume UTC for naive, log a warning.

28. **`portfolio/avanza_orders.py:126` — `total_sek` as float**
    Order ledger stores `round(volume * price, 2)` as float; ledger reads do
    `==` comparisons elsewhere. Float bait. Fix: store as `Decimal` or as
    integer öre (×100).

29. **`portfolio/signals/forecast.py:107-111` — host-specific Kronos
    subprocess paths**
    Hard-coded `Q:\models\...` paths in the subprocess invocation prevent the
    forecast signal from running on any host that does not match the dev
    box. Fix: read from config or fall back to a discoverable model dir.

30. **`portfolio/signals/realized_skewness.py:25` — undeclared scipy import**
    The signal does `import scipy` at module top but `scipy` is not in the
    project's declared dependencies. On a clean install the registry import
    fails and the signal stays HOLD forever. Fix: add scipy to requirements
    or drop the signal.

31. **`data/metals_loop.py:209` — `os.chdir(BASE_DIR)` at import**
    Side-effect at import violates pytest contract (other tests load this
    module and inherit the cwd change). Fix: replace with absolute paths.

### Single-reviewer surviving findings (lower conviction)

The following findings were raised by exactly one reviewer and survived (or
partially survived) cross-critique. They should be triaged before
implementation, not implemented blindly.

- **`portfolio/agent_invocation.py:932`** — broad `except Exception` on
  agent spawn masks `ImportError` (Claude only). Real, but hardly P1.
- **`portfolio/avanza_session.py:110`** — `fromisoformat` ValueError
  swallowed by generic warn (Claude only). Same class as #18.
- **`portfolio/signal_engine.py:2230-2233`** — Claude P1 on consensus
  tie-break to BUY at `>=`. Codex falsified: with `total = buy + sell`,
  `>` already implies `> 0.5`; ties already return HOLD. Withdraw.
- **`portfolio/signal_engine.py:2247`** — Claude P1 on `_confluence_score`
  using `buy_count >= sell_count`. Codex partial: real but score is
  diagnostic-only, not a trade-affecting flaw. Demote to P3.
- **`portfolio/signal_engine.py:264-310`** — Claude P1 on
  `_apply_persistence_filter()` cold-start bypass. Codex partial: bypass
  is intentional and documented; fix sketch was wrong. Re-write the
  finding before acting.
- **`portfolio/accuracy_stats.py:920`** — Claude P2 on NaN propagation in
  `blend_accuracy_data()`. Codex partial: missing local sanitation real,
  but downstream gate already sanitises so there is no live exposure.
  Demote to P3 hygiene.
- **`portfolio/shared_state.py:100`** — Codex P2 on `_cached()` capturing
  `now` before the fetch. Claude partial: the capture timing is a real
  performance cliff for slow fetches but is also what protects against
  caching transient None results; fix needs to preserve both behaviours.

## Per-Subsystem Card

Each card lists the bilateral-confirmed items for that subsystem and points
to the raw evidence files.

### signals-core
- 3 bilateral-confirmed (forecast_accuracy, accuracy_stats, outcome_tracker).
  Items #11, #12, #23 in the queue above.
- Evidence: `data/adv-2026-05-08/{claude,codex}-signals-core.md`,
  `data/adv-2026-05-10/cross-{claude-on-codex,codex-on-claude}-signals-core.md`.

### orchestration
- 2 bilateral-confirmed (`agent_invocation` ticker parser, `trigger`
  ranging suppression). Items #9, #10.
- 9 Claude-original new findings on inspection (broad except masking
  imports, weekday TZ mismatch in tier classification, journal full-file
  scan O(N), thread-pool resource leak on cancel, etc.). Triage required.
- Evidence: `data/adv-2026-05-08/{claude,codex}-orchestration.md`,
  `data/adv-2026-05-10/cross-{claude-on-codex,codex-on-claude}-orchestration.md`.

### portfolio-risk
- 6 bilateral-confirmed: kelly lot-matching, MC stop-hit prob, position-rate
  scale-in, warrant floor, concentration scoring, drawdown vs holdings.
  Items #2, #3, #4, #5, #24, #25.
- This subsystem had the strongest convergence; treat all 6 as same-day
  fixes when prioritised.

### metals-core
- 5 bilateral-confirmed: missing module imports, `os.chdir` on import,
  BEAR ladder direction sign, BEAR exit translation, BEAR MINI past
  barrier. Items #6, #7, #8, #13, #31.
- Direction-sign defect cluster — fix together to avoid partial pass.

### avanza-api
- 8+ bilateral-confirmed: order atomicity, expiry off-by-one, expiry
  re-check, CSRF race, token in logs, fromisoformat, offset 0, naive
  expires_at, total_sek as float. Items #1, #14-19, #27, #28.
- Several touch broker safety — fix order matters: atomicity first.

### signals-modules
- 2 bilateral-confirmed: forecast Kronos paths, realized_skewness scipy.
  Items #29, #30.
- Plus one Claude-original orphan finding (`realized_skewness` registered
  in `APPLICABLE_SIGNALS` while marked KILLED in `DISABLED_SIGNALS`).

### data-external
- 3 bilateral-confirmed: timed-out timeframe pool, FTD age window-offset,
  FTD cold-start. Items #20, #21, #26.

### infrastructure
- 1 bilateral-confirmed: telegram_poller `/mode` clobbering symlink. Item
  #22.
- 1 partial: `_cached()` TTL capture (in single-reviewer list above).
- Most codex P0/P1 here were "missing module" worktree-staleness false
  positives (claude critique caught them).

## Severity Distribution (post-cross-critique)

| Severity | Count | Examples |
|---|---|---|
| P0 | 1 | order atomicity |
| P1 | 21 | direction-sign cluster, kelly lot-match, MC stop prob, csrf race, etc. |
| P2 | 9 | concentration scoring, naive datetime, scipy declaration, etc. |
| P3 | 3 | dead constant, blend NaN hygiene, confluence diagnostic tie |

## Implementation Notes (for the remediation session)

This is review work; nothing was modified in this session. When the
remediation session opens:

1. **Don't bundle.** Each remediation item should land as a separate
   worktree branch, separate codex re-review, separate test pass. A
   30-fix mega-PR is going to hide regressions.

2. **Test gaps.** Several confirmed items (kelly lot-matching, MC
   stop-hit prob, position-rate guard, BEAR direction sign cluster,
   FTD state machine) need new tests written **first** to lock in
   current behaviour before fixing — the bug is the test invariant
   today.

3. **Live-trading exposure.** Items #1, #14-19 touch the broker session
   and order placement. Run end-to-end against the Avanza paper account
   before merging to main and restarting `PF-MetalsLoop`.

4. **Direction-sign cluster.** Items #6, #7, #8 are correlated. Fixing
   one without the others leaves BEAR MINI logic half-wired. Per the
   GUIDELINES P1 rule for half-wired features: gate the BEAR-side
   pipeline off behind a config flag while the cluster is in flight,
   rather than ship a half-working direction-sign system.

5. **Forecast history loss (#11) is recoverable but not retroactive.**
   If forecast_accuracy.py has already silently truncated history
   (check `data/forecast_predictions.jsonl` row count vs the snapshot
   threshold), the discarded rows are gone. Fix the bug, then accept
   the lost samples and re-baseline.

## Process Lessons

- **Asymmetric depth → asymmetric findings.** The codex pass produced 5-12k
  lines per subsystem; the claude pass produced 50-200. Convergence
  collapsed in codex's favour. For future dual-review sessions: equalise
  the budget, or run multiple claude passes per subsystem.

- **Cross-critique is essential.** A single-reviewer P1 list at this depth
  is dangerously polluted with false positives (Claude's pass would have
  shipped 83 false-positives unfiltered). The cross-critique pass cut
  noise by ~10x.

- **Worktree git issues confused codex.** Several codex P0 findings about
  "missing modules in worktree" were artefacts of the empty-baseline diff
  setup, not real problems on main. Future passes: verify `git status`
  inside each worktree before review starts.

## Status

- Independent reviews: complete (16 docs).
- Cross-critique: complete (16 docs).
- Synthesis: this doc + `docs/ADVERSARIAL_REVIEW_2026-05-10.md`.
- Implementation: not in scope. Remediation queue handed off above.
- Worktrees: cleaned up at end of this session (see commit log).

## Files Produced This Session

```
docs/ADVERSARIAL_REVIEW_2026-05-10.md           # index doc (committed earlier)
docs/ADVERSARIAL_SYNTHESIS_2026-05-10.md        # this doc
data/adv-2026-05-10/cross-claude-on-codex-*.md  # 8 files (Claude critiques codex)
data/adv-2026-05-10/cross-codex-on-claude-*.md  # 8 files (Codex critiques Claude)
data/adv-2026-05-10/_critique_prompt_template.txt
```
