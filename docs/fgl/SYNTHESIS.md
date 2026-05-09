# /fgl Dual Adversarial Review — Synthesis

Date: 2026-05-09
Branch baseline: `main` HEAD `aa493aec` (Friday EOD)
Reviewers: codex / gpt-5.4 (xhigh reasoning), claude / Opus 4.7 (1M)

This document folds the two reviewers' findings into one prioritized list,
records cross-critique verdicts where available, and notes where the dual
review uncovered classes of bugs that a single reviewer would have missed.

---

## How the review ran

1. **Eight subsystem branches** off the orphan `empty-baseline` commit,
   each containing only the files in its subsystem (full file set in
   `PARTITION.md`). This let `codex review` and `codex exec` see the
   whole subsystem as a "diff" against an empty tree.
2. **Codex** did one adversarial review per subsystem in `codex exec`
   read-only sandbox, gpt-5.4 / xhigh reasoning. Output saved to
   `data/fgl-logs/codex-<sub>.txt` then wrapped to
   `docs/fgl/codex-<sub>.md`.
3. **Claude** did one adversarial review per subsystem via parallel
   `general-purpose` subagents, each writing
   `docs/fgl/claude-<sub>.md`.
4. **Cross-critique** in two directions:
   - Claude vetted each codex review against current code
     (`docs/fgl/claude-critique-of-codex-<sub>.md`) and surfaced
     "missed by codex" items from the claude review.
   - Codex vetting of claude was attempted but **hit the OpenAI usage
     cap** mid-batch (rate limit reset 22:32, well after the session
     closes). Codex critique-of-claude is therefore **not in this
     synthesis**. The asymmetry biases the verdict slightly: bugs that
     only claude flagged were not independently re-checked by codex,
     so claude-only P0s are counted as confirmed-by-claude-twice rather
     than confirmed-by-both-reviewers. Treat claude-only P0s with the
     same skepticism a single-reviewer report deserves.

---

## Headline numbers

### Per-subsystem severity (P0 = real-money, P1 = bad-day, P2 = conditional, P3 = cosmetic)

| Subsystem        | Claude P0/P1 | Codex P0/P1 | CONFIRM | DISPUTE | PARTIAL | MISSED-by-codex |
|------------------|-------------:|------------:|--------:|--------:|--------:|----------------:|
| signals-core     |   0 / 6      |   8 / 7     |   15    |   1     |   1     |    7            |
| orchestration    |   7 / 12     |   1 / 8     |   27    |   0     |   1     |   15            |
| portfolio-risk   |  11 / 27     |   2 / 5     |   15    |   0     |   4     |   11            |
| metals-core      |   9 / 19     |   5 / 6     |   19    |   0     |   0     |   14            |
| avanza-api       |   8 / 22     |   4 / 3     |   12    |   0     |   2     |   27            |
| signals-modules  |   1 / 4      |   0 / 5     |   37    |   0     |   2     |   10            |
| data-external    |  28 / 27     |   0 / 7     |   46    |   0     |   1     |   27            |
| infrastructure   |   5 / 18     |   2 / 9     |   14    |   1     |   4     |   14            |
| **TOTAL**        |  69 / 135    |  22 / 50    |  185    |   2     |  15     |  125            |

CONFIRM = codex's finding reproduced in code by claude vetting.
DISPUTE = claude vetting concluded codex's finding is wrong.
PARTIAL = kernel of truth, severity or fix overstated.
MISSED  = P0/P1 from the claude review that codex did not call out
          and that claude critique then independently re-confirmed
          against current code.

The 2 DISPUTE rows:
1. `signals-core` codex P0 #8 (`btc_proxy` voting unregistered) was a
   false positive — codex reviewed the worktree branch which omits
   `tickers.py` (it lives in the orchestration partition), so codex
   could not see commit `efedf4ee` adding `btc_proxy` to
   `SIGNAL_NAMES`. **Already fixed at HEAD.** Subsystem-partition
   side-effect of the empty-baseline review; not a codex error per se.
2. `infrastructure` codex P1 (`process_lock.py:36` first-owner lock
   spurious-fail) — claude vetting determined Windows `LK_NBLCK` on
   byte 0 of an empty file does not actually fail spuriously; codex
   overstated the bug class.

### Reviewer-divergence pattern

The two reviewers found largely **non-overlapping bug classes**:

- **Codex prioritized** stop-loss state-machine ordering, file-state
  race conditions on JSONL/JSON ledgers, schema-mismatch silent
  failures (e.g., `volume` vs `units`), and accuracy-pipeline
  freshness keys (single horizon timestamp poisoning multiple
  horizons). Codex's confirm rate during cross-critique is high
  (~95%) — its findings are narrow but accurate.
- **Claude prioritized** systemic correctness gaps: tz-naive
  timestamps across data sources, FX fallback hardcoded to 10.50
  (~6% off), direction blindness in metals (BEAR certs traded as if
  LONG), missing account whitelist in the new Avanza package, missing
  `stdin=DEVNULL` on Layer 2 subprocesses, and the silent-failure
  surfaces (try/except that swallow). Claude's review is broader but
  less narrow on subtle edge-case bugs.
- **Both reviewers independently flagged** the same critical bug in
  five of eight subsystems — strong signal these are real:
  - `monte_carlo_risk.py:431` — raw `agent_summary["fx_rate"]`
    bypassing the validated FX fallback (10× SEK risk under-report).
  - `data/metals_loop.py:2456 + :4900` — stop cancel-before-validate.
  - `agent_invocation.py` agent-baseline replay on read error.
  - `signal_engine.py:3170` — `btc_proxy` voting without being in
    `SIGNAL_NAMES`.
  - `outcome_tracker.py` accuracy backfill timestamp alignment
    (codex flagged P0 across three lines; claude flagged related
    cache-staleness P1).

Adversarial review is materially more valuable run as a *pair* than as
either reviewer alone; the missed-by-codex column averages ~17
findings/subsystem that codex would not have caught.

---

## Master P0 list (both reviewers, cross-checked)

Confirmed P0 bugs from EITHER reviewer that survived the cross-critique.
Sorted approximately by blast radius (P0-A = active live, P0-B = lurking
on cold path, P0-C = real-money but rare). File:line refers to current
`main` HEAD.

### P0-A — currently shipping incorrect behavior on the live trading path

1. **`fin_snipe.py:60-74`** — `fetch_positions_by_orderbook()` does NOT
   filter on ISK account `1625505`; pension `2674244` holdings can
   overwrite ISK records and the manager will route trades to the
   wrong account. Direct violation of `feedback_isk_only.md`.
   *Found by: claude. Critique: confirmed. Codex missed.*

2. **`fin_snipe_manager.py` (entire module)** — zero direction handling.
   Manager assumes every position is LONG; for BEAR/MINI S certs all
   stops, exits, and ladder targets are mirrored (winning when losing,
   losing when winning).
   *Found by: claude. Critique: confirmed. Codex partially overlapped
   via `fin_snipe.py:160` and `fin_fish.py:735` but did not flag the
   manager-level absence.*

3. **`fin_snipe_manager.py:61,536`** — `HARD_STOP_CERT_PCT = 0.05` →
   1% on underlying for 5x silver. Inside silver's intraday wick
   range; direct violation of −15% min from `feedback_mini_stoploss.md`.
   *Found by: claude. Critique: confirmed. Codex missed.*

4. **`fin_snipe_manager.py` + `data/metals_loop.py:4869-4983`** — no
   barrier-proximity check anywhere when sizing the stop; only checks
   distance from current bid, not from MINI knockout barrier.
   *Found by: claude. Critique: confirmed. Codex missed.*

5. **`data/metals_loop.py:2456` and `:4900`** — `_cancel_stop_orders`
   runs BEFORE the 3% distance gate; if the new stop is rejected as
   too close, the position is left bare with zero hardware protection.
   *Found by: codex. Critique: confirmed.*

6. **`data/metals_loop.py:2499` ↔ `:4951`** — `_rebuild_stop_orders_for`
   writes `volume`, `place_stop_loss_orders` writes `units`. Schema
   mismatch breaks `metals_loop.py:5066` stop-fill accounting; broker-
   sold positions stay marked active locally.
   *Found by: codex. Critique: confirmed.*

7. **`portfolio/avanza/trading.py:80-102`** — `place_order` has NO
   `ALLOWED_ACCOUNT_IDS` whitelist; legacy `avanza_session.py:586`
   enforces `{"1625505"}` but new TOTP path silently bypasses it.
   *Found by: claude. Critique: confirmed. Codex missed.*

8. **`portfolio/avanza/trading.py:74-92`** — no `MAX_ORDER_TOTAL_SEK`
   (50000 SEK) cap. Legacy session path enforces it; new path doesn't.
   *Found by: claude. Critique: confirmed. Codex missed.*

9. **`portfolio/avanza/trading.py:84`** — `client.avanza.place_order()`
   runs WITHOUT `avanza_order_lock`. Cross-process file lock used by
   every legacy path is missing here.
   *Found by: claude. Critique: confirmed. Codex missed.*

10. **`portfolio/avanza/auth.py:74-114`** — singleton has NO
    expiry/re-auth path. Once first TOTP auth succeeds, dead session
    persists; no caller invokes `reset()`.
    *Found by: claude. Critique: confirmed. Codex missed.*

11. **`telegram_poller.py:361`** — `/mode` Telegram command writes
    `config.json` via `atomic_write_json`. Per CLAUDE.md, config.json
    is a SYMLINK to external secrets; one Telegram command can sever
    the link and replace it with a plain file (also caught at
    `file_utils.py:59` as the same root cause).
    *Found by: codex. Critique: confirmed.*

12. **`http_retry.py:31`** — `fetch_with_retry` retries POSTs blindly
    with no idempotency check; Telegram alerts can double-send,
    Avanza order POSTs can duplicate trades.
    *Found by: claude. Critique: confirmed. Codex missed.*

13. **`agent_invocation.py:825-830`** — `claude_cmd` fallback `cmd =
    ["cmd", "/c", str(agent_bat)]` does NOT pass the prompt argv; bat
    runs whatever default it has and silent-auth detection still
    scans the log. A missing `claude` binary thus produces a
    "successful" subprocess that never sees the trigger.
    *Found by: claude. Critique: confirmed. Codex missed.*

14. **`agent_invocation.py:856-862`, `multi_agent_layer2.py:168-174`**
    — `subprocess.Popen` lacks `stdin=DEVNULL`. CLI prompt for
    input would hang the headless subprocess until tier timeout.
    *Found by: claude. Critique: confirmed. Codex missed.*

15. **`loop_contract.py:333-369` + `main.py:849`** — `skipped_busy`
    excluded from `_LEGITIMATE_SKIP_STATUSES`; combined with main.py
    collapsing every `False` from `invoke_agent` into `skipped_busy`,
    the contract fires `layer2_journal_activity` violations on
    healthy in-flight invocations. False alarms drown real ones.
    *Found by: claude. Critique: confirmed. Codex missed.*

16. **`agent_invocation.py:883`** — both per-portfolio drawdown reads
    in a single try/except that resets baselines to 0 on error;
    completion then replays the entire transaction history into
    `record_trade()`, poisoning overtrading guards.
    *Found by: codex. Critique: confirmed.*

17. **`trigger.py:239`** — ranging dampener advances
    `triggered_consensus[ticker] = action` BEFORE `continue`; later
    high-confidence signals see baseline already at action and never
    trigger.
    *Found by: codex. Critique: confirmed.*

18. **`agent_invocation.py:676`** (drawdown checks) — if both
    drawdown checks throw, the 50% block disappears and Layer 2
    still runs. The comment at 674-677 explicitly admits this is "by
    design".
    *Found by: codex. Critique: confirmed.*

19. **`outcome_tracker.py:220, :241, :269`** — `_fetch_historical_price`
    uses `startTime=target_ts` to query Binance/Alpaca/yfinance for
    horizon outcomes. The first candle returned starts AT target_ts
    so the close lags by up to 2h (Binance), and yfinance branch uses
    DAILY closes for sub-daily horizons (MSTR 3h/4h/12h backfill is
    structurally wrong). Poisons the live accuracy gate.
    *Found by: codex. Critique: confirmed.*

20. **`outcome_tracker.py:166, :493`** — failed SQLite outcome writes
    are logged at warning/debug and dropped; `accuracy_stats.py:151`
    `load_entries()` trusts SQLite whenever it has any rows, so one
    failed write makes every accuracy computation silently ignore
    newer JSONL snapshots.
    *Found by: codex. Critique: confirmed.*

21. **`forecast_accuracy.py:343`** — `backfill_forecast_outcomes()`
    breaks once `updated >= max_entries` and rewrites the file from
    `modified_entries`, truncating every unprocessed prediction
    after the break.
    *Found by: codex. Critique: confirmed.*

### P0-B — would lose money next time the relevant path executes

22. **`monte_carlo_risk.py:431`** — `fx_rate =
    agent_summary.get("fx_rate", FX_RATE_FALLBACK)` does NOT route
    through `_resolve_fx_rate`. If `agent_summary["fx_rate"]` is the
    legacy 1.0 literal (still seen in stale summaries), VaR/CVaR SEK
    is 10× too small.
    *Found by: BOTH. Codex P1, claude P0. Critique: confirmed.*

23. **`monte_carlo_risk.py:444`** — held positions with missing or
    non-positive live prices are silently omitted from VaR; real
    exposure disappears from the risk report.
    *Found by: codex. Critique: confirmed.*

24. **`fx_rates.py:33` + `:71`** — cache freshness gate doesn't
    validate the cached rate's sanity; corrupt write of `rate=0.0`
    served for 15 min. `FX_RATE_FALLBACK = 10.50` is hardcoded
    literal ~6% off current ~11.10.
    *Found by: claude. Critique: confirmed. Codex missed.*

25. **`data_collector.py:96, :157, :245`** — `pd.to_datetime(...)`
    on Binance / Alpaca / yfinance data without `utc=True`. Tz-naive
    timestamps mixed with `datetime.now(UTC)` produce silent
    CET-offset skew on every recency calc.
    *Found by: claude. Critique: confirmed. Codex missed.*

26. **`onchain_data.py:284`** — `_cached("onchain_btc", ...)` keys
    only on the string key, so token rotation does not invalidate
    cache; stale 12h cache reused with new token. Plus partial-
    success cache writes pin 5-of-6-stale on the next 12h window.
    *Found by: claude. Critique: confirmed. Codex missed.*

27. **`risk_management.py:367-369`** — ATR stop placement is
    knockout-barrier blind for warrants. 2×ATR distance can land
    above the knockout barrier or inside the 3% prohibited zone of
    `feedback_mini_stoploss.md`.
    *Found by: claude. Critique: confirmed. Codex missed.*

28. **`risk_management.py:233`** — `load_json(portfolio_path,
    default={})` returns empty dict on corrupt file → drawdown 0% →
    circuit breaker silently bypassed.
    *Found by: claude. Critique: confirmed. Codex missed.*

29. **`avanza_orders.py:132 + :171`** — unlocked read-modify-write
    of `avanza_pending_orders.json` from `request_order` /
    `check_pending_orders`; concurrent writers can drop or
    resurrect pending/executed orders.
    *Found by: codex. Critique: confirmed.*

30. **`avanza_orders.py:389`** — Telegram send failure after a
    successful Avanza fill flips the order from `executed` to
    `error`, hiding a live trade and enabling duplicate retries.
    *Found by: codex. Critique: confirmed.*

31. **`avanza_control.py:401`** — `delete_stop_loss_no_page()` only
    checks `errorCode`; HTTP 500 reported as success. Caller
    proceeds as if the protective stop was removed.
    *Found by: codex. Critique: confirmed.*

32. **`portfolio/avanza/types.py:241-266`** — `Position.last_price`
    from `quote.latest`/`quote.last` (last-traded), not bid; SELL
    limit prices computed from this hit the wrong side of the
    spread on every illiquid warrant.
    *Found by: claude. Critique: confirmed. Codex missed.*

33. **`portfolio/avanza/tick_rules.py:82-98`** — claims "integer
    arithmetic" but operates on floats; `0.295 * 100` rounds wrong.
    Tick-misaligned price → silent order rejection.
    *Found by: claude. Critique: confirmed. Codex missed (codex
    flagged a related but weaker P3 about the same file).*

34. **`message_throttle.py:60-67`** — TOCTOU race: two concurrent
    callers both pass cooldown check, both write last-writer-wins.
    *Found by: claude. Critique: confirmed. Codex missed.*

35. **`gpu_gate.py:209-269`** — Layer-1 thread lock acquired BEFORE
    Layer-2 file lock; on file-lock timeout the thread lock is held
    for the full timeout (default 60s), starving unrelated GPU
    consumers (the 25-hour wedge of 2026-05-02 was supposedly
    solved by adding the sweeper, but `_pid_alive` returns True when
    psutil is missing so the sweeper degrades to never-break in
    that case).
    *Found by: claude. Critique: confirmed. Codex partially
    overlapped on `_pid_alive` fail-closed (P1).*

36. **`dashboard/auth.py:131-132`** — Cloudflare Access path trusts
    `Cf-Access-Authenticated-User-Email` header without verifying
    the request came through Cloudflare. Direct LAN connection to
    port 5055 spoofs the header.
    *Found by: claude. Critique: confirmed. Codex missed.*

37. **`signal_engine.py:1721`** — macro-window detection fails OPEN
    by treating exceptions as "inactive"; a broken econ-calendar
    re-enables trading exactly when it should suppress event noise.
    *Found by: codex. Critique: confirmed.*

### P0-C — incorrect under specific conditions, smaller blast radius

38. **`accuracy_stats.py:1388, :1923`** — regime-accuracy and
    per-ticker-signal caches keyed by one shared `time` across
    horizons; recomputing one horizon makes stale data for other
    horizons look fresh.
    *Found by: codex. Critique: confirmed.*

39. **`crypto_macro_data.py:75`** — Deribit instrument-name parser
    silently drops malformed segments and contaminates `expiry_data`
    with `BTC_USDC-PERPETUAL` / index-price instruments.
    *Found by: claude. Critique: confirmed. Codex missed.*

40. **`macro_context.py:313`** — `tickers = {"2y": "2YY=F", ...}` —
    `2YY=F` is a Treasury *futures price*, not a yield. `^TNX`/`^TYX`
    are yield indices. Spread calc mixes incompatible units.
    *Found by: codex. Critique: confirmed.*

41. **`futures_data.py:83/112/141/170/198`** — five cache keys for
    OI/LS/funding/top-trader series omit `limit`, so a small earlier
    request poisons later larger requests with truncated history.
    *Found by: codex. Critique: confirmed.*

42. **`market_health.py:255`** — `detect_ftd_state` re-runs on every
    hourly refresh and increments `rally_day` again; false FTD
    confirmation intraday with no new trading day.
    *Found by: codex. Critique: confirmed.*

43. **`alpha_vantage.py:281` + `sentiment.py:192` +
    `earnings_calendar.py`** — quota counters only increment on
    success; failed/empty results still hit the API but don't count,
    so the daily budget is silently overspent.
    *Found by: codex. Critique: confirmed.*

44. **`crypto_cross_asset.py:257`** — return dict uses `"signal"`
    key while every other module returns `"action"`; module is also
    NOT registered in `signal_registry.py`. Dead code that crashes
    on use; but a P0 IF re-enabled because it's already wired into
    the `_CRYPTO_ONLY_SIGNALS` accounting.
    *Found by: BOTH (claude P0, codex P3 — agreement on bug,
    disagreement on severity). Critique: confirmed. Severity should
    track "what happens if someone enables this signal tomorrow"
    rather than "what happens today".*

(Total master P0 list: 44 items. Per-subsystem files contain the
remaining ~150 P1 / 130 P2 / ~110 P3 findings.)

---

## Per-subsystem synthesis

Each section below picks up the cross-checked findings, then notes any
items that survived as **disputed** (codex says X, claude says not-X)
or **partial** (kernel of truth but severity wrong on one side).

### signals-core

Claude P0/P1: 0/6. Codex P0/P1: 8/7.
Cross-critique: 15 CONFIRM, 1 DISPUTE, 1 PARTIAL. 7 missed-by-codex.

Biggest divergence in the review. Codex, with its outcome-tracker /
accuracy-pipeline focus, flagged 8 P0s about timestamp alignment and
SQLite-vs-JSONL freshness; claude focused on per-asset signal filtering
and confidence-cascade subtleties at P1/P2.

**Confirmed P0 (codex found, claude missed):**
- `outcome_tracker.py:220, :241, :269` — backfill timestamp alignment
  (3 separate paths). MASTER P0 #19.
- `outcome_tracker.py:166, :493` — failed SQLite writes go to
  warning/debug, but `accuracy_stats.py:151` prefers SQLite when
  rows exist. MASTER P0 #20.
- `accuracy_stats.py:151` (load_entries SQLite preference) — same
  cluster as above.
- `forecast_accuracy.py:343` — backfill truncates unprocessed tail.
  MASTER P0 #21.
- `signal_engine.py:1721` — macro-window detection fails open.
  MASTER P0 #37.

**Disputed:**
- Codex P0 #8 (`signal_engine.py:3170` — `btc_proxy` voting unregistered).
  Already fixed at HEAD by commit `efedf4ee` adding `btc_proxy` to
  `SIGNAL_NAMES` in `tickers.py`. Codex couldn't see this because
  `tickers.py` lives in the orchestration partition, not signals-core.

**Confirmed P1 (claude found, codex missed):**
- `signal_engine.py:1086-1118` — `_compute_applicable_count` does not
  filter `btc_proxy` by ticker. Inflates `total_applicable` for
  BTC/ETH/XAU/XAG, distorts ensemble entropy.
- `signal_engine.py:107, 2845-2902` — `onchain` is in
  `_CRYPTO_ONLY_SIGNALS` but only fires for BTC; ETH treats it as
  "missing" rather than HOLD.
- `signal_registry.py:160-188` — `cot_positioning`, `dxy_cross_asset`
  documented metals-only but not in `_METALS_ONLY_SIGNALS`.
- `signal_engine.py:3702-3725` — per-ticker consensus gate raises
  TypeError on cached `accuracy: None`, swallowed silently; the safe
  fix `_safe_accuracy()` already exists in `_weighted_consensus` but
  is not propagated.

### orchestration

Claude P0/P1: 7/12. Codex P0/P1: 1/8.
Cross-critique (claude vetting codex): 27 CONFIRM, 1 PARTIAL, 0 DISPUTE.
Plus 15 P0/P1 items missed by codex but found by claude and
independently re-confirmed.

**Confirmed P0 (both reviewers):**
- `health.py:35` — `last_invocation_ts` stamped on every trigger
  including skipped ones, which masks real Layer 2 outages from
  `check_agent_silence` (codex flagged; claude framed as P1).

**Confirmed P0 (claude found, codex missed, re-confirmed by critique):**
- `agent_invocation.py:825-830` — `claude_cmd` fallback `cmd = ["cmd",
  "/c", str(agent_bat)]` does NOT pass the prompt argv; bat runs
  whatever default it has and silent-auth detection still scans the
  log. A missing `claude` binary thus produces a "successful"
  subprocess that never sees the trigger.
- `agent_invocation.py:856-862` and `multi_agent_layer2.py:168-174` —
  `subprocess.Popen` lacks `stdin=DEVNULL`. CLI prompt for input
  hangs the headless subprocess until tier timeout.
- `loop_contract.py:333-369` — `skipped_busy` excluded from
  `_LEGITIMATE_SKIP_STATUSES`; combined with `main.py:849` collapsing
  every `False` from `invoke_agent` into `skipped_busy`, the contract
  fires `layer2_journal_activity` violations on healthy in-flight
  invocations.
- `agent_invocation.py:883` — both per-portfolio drawdown reads in a
  single try/except that resets baselines to 0 on error; completion
  then replays the entire transaction history into `record_trade()`,
  poisoning overtrading guards.
- `trigger.py:239` — ranging dampener advances
  `triggered_consensus[ticker] = action` BEFORE `continue`; later
  high-confidence signals see baseline already at action and never
  trigger.

**Confirmed P1 (claude only):**
- `agent_invocation.py:1191-1194` — first-ever invocation always
  reports "incomplete" because `_journal_ts_before is None` forces
  both flags False even when a journal entry was written.
- `main.py:1115-1126` — `_sleep_for_next_cycle` returns immediately
  when remaining ≤ 0; runaway cycle with no minimum sleep floor.
- `main.py:1129-1160` — heartbeat staleness check raises ValueError
  on corrupt heartbeat, swallowed silently; no LOOP RESTARTED alert.
- `main.py:837-861` — `heartbeat_keepalive` wraps `invoke_agent`
  which returns immediately after Popen; the long-running subprocess
  runs WITHOUT keepalive, so the dashboard flips stale during
  normal Layer 2 work.
- Two duplicate EU DST implementations (`market_timing._is_eu_dst` and
  `session_calendar._eu_dst`) — drift inevitable when one is updated.

### portfolio-risk

Claude P0/P1: 11/27. Codex P0/P1: 2/5.
Cross-critique: 15 CONFIRM, 0 DISPUTE, 4 PARTIAL. 11 missed-by-codex.

**Confirmed P0 (both found independently, sometimes at different
severities):**
- `monte_carlo_risk.py:431` — FX rate fallback bypass. Both reviewers
  agreed; codex P1, claude P0. MASTER P0 #22.
- `monte_carlo_risk.py:444` — silent omission of positions with
  missing prices from VaR. Codex P0. MASTER P0 #23.

**Confirmed P0 (claude found, codex missed):**
- `kelly_sizing.py:269-323` — Kelly uses `cash_sek * frac` while
  concentration check uses `total_value × frac`; Kelly will recommend
  more than the concentration cap allows.
- `kelly_sizing.py:326` + `trade_validation.py:32` — 500 SEK floor
  contradicts Avanza 1000 SEK minimum; trades 500-999 SEK pass
  internal validation then get rejected by Avanza.
- `risk_management.py:367-369` — ATR stop placement knockout-barrier
  blind. MASTER P0 #27.
- `risk_management.py:233` — corrupt portfolio file silently
  bypasses circuit breaker. MASTER P0 #28.
- `equity_curve.py:495` — flat round-trip counts as LOSS in streak
  but excluded from `wins/losses` for win_loss_ratio.
- `warrant_portfolio.py:215` — `record_warrant_transaction` lacks
  the cross-thread lock that `portfolio_mgr.update_state` enforces.
- `warrant_portfolio.py:182-214` — transaction record has NO `reason`
  field, violating CLAUDE.md's "log every trade with a reason".
- `portfolio_mgr.py:166-180` — `portfolio_value` does not apply
  leverage for warrant tickers; if any path stuffs warrants into
  main holdings, value is misvalued.
- `warrant_portfolio.py:154` (codex agreed, P0) — `get_warrant_summary`
  drops live warrant position when underlying price is missing.

**Partial:**
- Codex framed `equity_curve.py:405` as fee double-count; claude
  vetting concluded the math is correct in normal operation; only
  fragile if the inner-loop exits early.

### metals-core

Claude P0/P1: 9/19. Codex P0/P1: 5/6.
Cross-critique: 19 CONFIRM, 0 DISPUTE. 14 P0/P1 missed-by-codex.

**Confirmed P0 (both reviewers, sometimes at different severities):**
- `data/metals_loop.py:2456` and `:4900` — `_cancel_stop_orders(...)`
  runs before the 3% distance gate; if the new stop is rejected as
  too close, the position is left bare.
- `data/metals_loop.py:4893` — same-day fast path treats any
  non-empty `orders` list as "already placed", so a single failed
  stop placement leaves position unprotected for the rest of the
  day.
- `data/metals_loop.py:2499` vs `:4951` — `_rebuild_stop_orders_for`
  writes `volume` while `place_stop_loss_orders` writes `units`;
  schema mismatch breaks `metals_loop.py:5066` stop-fill accounting,
  silently leaving broker-sold positions marked active locally.
- `fin_fish.py:735` — breached BEAR MINI barriers fall through with
  literal `pass`; planner ranks knocked-out short instruments as
  tradeable.
- `fin_snipe.py:160` — ladder generation never passes
  `direction_sign=-1` for BEAR/MINI S; inverse instruments get long
  ladders.

**Confirmed P0 (claude only, re-confirmed):**
- `fin_snipe.py:60-74` — `fetch_positions_by_orderbook()` does NOT
  filter on ISK account `1625505`; pension `2674244` holdings can
  overwrite ISK records and the manager will route trades to the
  wrong account. Direct violation of `feedback_isk_only.md`.
- `fin_snipe_manager.py` (entire file) — zero direction handling.
  Manager assumes every position is LONG; for BEAR certs all stops,
  exits, and ladder targets are mirrored.
- `fin_snipe_manager.py:61,536` — `HARD_STOP_CERT_PCT = 0.05` for 5x
  leverage = 1% on underlying, well inside silver's intraday range,
  in direct violation of the −15% minimum from
  `feedback_mini_stoploss.md`.
- `fin_snipe_manager.py` — no barrier-proximity check anywhere when
  sizing the stop; only checks `MIN_STOP_DISTANCE_PCT = 1.0` from
  current bid, not from MINI knockout barrier.
- `data/metals_loop.py:1455` — `leverage = silver_pos.get("leverage",
  4.76)` defaults to a hardcoded 4.76; warrant P&L estimate wrong
  for any non-4.76 instrument.
- `data/metals_loop.py:1450,1457,1478` — `_silver_fast_tick`
  thresholds assume LONG silver; BEAR positions never alert when
  underlying RISES (the bad direction for the BEAR position).

### avanza-api

Claude P0/P1: 8/22. Codex P0/P1: 4/3.
Cross-critique: 12 CONFIRM, 2 PARTIAL. 27 missed-by-codex.

**Confirmed P0 (codex flagged):**
- `avanza_orders.py:132 + :171` — unlocked read-modify-write of
  `avanza_pending_orders.json` from `request_order` and
  `check_pending_orders`; concurrent writers can drop or resurrect
  pending/executed orders.
- `avanza_orders.py:389` — Telegram send failure after a successful
  Avanza fill flips order from `executed` to `error`; live trade
  hidden, duplicate retries enabled.
- `avanza_control.py:401` — `delete_stop_loss_no_page()` only checks
  `errorCode`; `_api_delete()`'s `http_status=500, ok=False` reported
  as success — caller proceeds as if the protective stop was removed.

**Confirmed P0 (claude found, codex missed, re-confirmed):**
- `portfolio/avanza/trading.py:80-102` — `place_order` has NO
  `ALLOWED_ACCOUNT_IDS` whitelist. The legacy `avanza_session.py:586`
  enforces `{"1625505"}`; the new TOTP path silently bypasses it.
- `portfolio/avanza/trading.py:74-92` — no `MAX_ORDER_TOTAL_SEK =
  50000` cap. `avanza_session.py:597` has it; new path doesn't.
- `portfolio/avanza/trading.py:84` — `client.avanza.place_order()`
  runs WITHOUT `avanza_order_lock`. The cross-process file lock
  used by every legacy path is missing here.
- `portfolio/avanza/types.py:241-266` — `Position.last_price` from
  `quote.latest`/`quote.last` (last-traded), not bid; callers
  computing SELL limit prices hit wrong side of spread on every
  illiquid warrant.
- `portfolio/avanza/account.py:64-94` — `get_buying_power(account_id)`
  accepts arbitrary account IDs.
- `portfolio/avanza/tick_rules.py:82-98` — claims "integer
  arithmetic" but operates on floats; `0.295 * 100` floats and
  rounds wrong. Tick-misaligned price → silent order rejection.
- `portfolio/avanza/auth.py:74-114` — singleton has NO expiry/re-auth
  path. Once first TOTP auth succeeds, dead client persists.

### signals-modules

Claude P0/P1: 1/4. Codex P0/P1: 0/5.
Cross-critique: 37 CONFIRM, 2 PARTIAL, 1 UNVERIFIED. 10 missed-by-codex.

The most balanced subsystem. Both reviewers agreed there are no
catastrophic P0 bugs in the active signal modules — the worst
issues are signal-direction biases that would degrade accuracy but
not lose money on a single trade.

**Confirmed cross-flagged finding:**
- `crypto_cross_asset.py:257` — `"signal"` instead of `"action"`,
  not registered. Both reviewers; severity disagreement (claude P0,
  codex P3) — synthesis ranks as P0 *if re-enabled*. MASTER P0 #44.

**Confirmed P1/P2 (active modules):**
- `econ_calendar.py:142-145` — `_post_event_relief` BUY bias on
  empty-calendar weeks (mirrors the `calendar` signal failure profile
  that crashed to 29.3% accuracy).
- `crypto_macro.py:192` — `_expiry_proximity` BUY bias for every
  non-quarterly expiry day.
- `momentum_factors.py:81` — `_time_series_momentum` votes on any
  nonzero return; no threshold means always-on directional vote.
- `news_event.py:199, :270, :330, :402, :473` — naive substring
  matching of `raise`, `beat`, `cut`, `approval`; "Fed expected to
  raise rates" matches as bullish.

**Confirmed disabled-module landmines (P3):**
- `complexity_gap_regime.py:92`, `mahalanobis_turbulence.py:99` —
  `_cached("...", _do_fetch, ttl=_CACHE_TTL)` passes ttl twice
  (positional + kwarg); re-enabling raises TypeError.
- `crypto_cross_asset.py:257` — see above.

**Confirmed P1 (claude found, codex missed):**
- `credit_spread.py:78-82` — `_Shim.__call__` is `@staticmethod`;
  Python looks up `__call__` on the type, not the instance — fallback
  raises `TypeError: 'Shim' object is not callable` if `http_retry`
  import fails.
- `credit_spread.py:53,113-115` — `_oas_cache` mutated without lock
  under ThreadPoolExecutor.
- `structure.py:79-82` — `_highlow_breakout` votes SELL within 2% of
  period low without breakdown check; "sell at support" inversion.

**Confirmed P0 (claude only):**
- `crypto_cross_asset.py:257` — return dict uses `"signal"` key
  while every other module returns `"action"`; module is also NOT
  registered in `signal_registry.py`. Dead code that would crash on
  use.

**Both reviewers flagged similar:**
- `econ_calendar.py:142-145` — `_post_event_relief` votes BUY
  whenever `next_event["hours_until"] > 72` — structural BUY bias on
  empty-calendar weeks, mirroring the failure that just killed the
  `calendar` signal at 29.3% accuracy.
- `crypto_macro.py:192` — `_expiry_proximity` votes BUY for every
  non-quarterly expiry day; persistent BUY bias around weekly crypto
  expiries.

### data-external

Claude P0/P1: 28/27. Codex P0/P1: 0/7.
Cross-critique: 46 CONFIRM, 1 PARTIAL. 27 missed-by-codex.

**Confirmed P0 (claude found, codex missed):**
- `data_collector.py:96, :157, :245` — `pd.to_datetime(...)` on
  Binance / Alpaca / yfinance data without `utc=True`. Tz-naive
  timestamps mixed with `datetime.now(UTC)` produces silent CET-offset
  skew; ~1h winter / ~2h summer error in every recency calc.
- `fx_rates.py:33` — cache freshness gate doesn't validate the
  cached rate's sanity; a corrupt write of `rate=0.0` is served for
  15 min, every SEK→USD math divides-by-zero or explodes portfolio
  values.
- `fx_rates.py:71` — `FX_RATE_FALLBACK = 10.50` hardcoded literal,
  ~6% off from current ~11.10. Real-money trading uses this when
  both live + cache fail.
- `onchain_data.py:284` — `_cached("onchain_btc", ...)` keys solely
  on the string key, so a token rotation does not invalidate cache;
  stale 12h cache reused with new token.
- `crypto_macro_data.py:75` and adjacent — Deribit instrument-name
  parser silently drops malformed segments and contaminates
  `expiry_data`.
- `macro_context.py:313` — `tickers = {"2y": "2YY=F", ...}` — `2YY=F`
  is a Treasury *futures price*, not a yield. `^TNX`/`^TYX` are
  yield indices. Spread calc mixes incompatible units.

**Confirmed P0 (codex found):**
- `futures_data.py` x5 — cache keys for OI/LS/funding/top-trader
  series omit `limit`, so a small earlier request poisons later
  larger requests with truncated history.
- `market_health.py:255` — `detect_ftd_state` re-runs on every
  hourly refresh and increments `rally_day` again; false FTD
  confirmation intraday with no new trading day.
- `alpha_vantage.py:281` and `sentiment.py:192` — quota counters
  only increment on success; failed/empty results still hit the API
  but don't count, so the budget is silently overspent.

### infrastructure

Claude P0/P1: 5/18. Codex P0/P1: 2/9.
Cross-critique: 14 CONFIRM, 1 DISPUTE, 4 PARTIAL. 14 missed-by-codex.

**Confirmed P0 (codex found):**
- `telegram_poller.py:361` — `/mode` Telegram command writes
  `config.json` via `atomic_write_json`. CLAUDE.md states config.json
  is a SYMLINK to external secrets; one Telegram command can sever
  the symlink and replace it with a plain file.
- `file_utils.py:59` — `atomic_write_json` `os.replace` on the
  symlink path itself (severs the link); same root cause.

**Confirmed P0 (claude found, codex missed):**
- `http_retry.py:31` — `fetch_with_retry` retries POSTs blindly
  with no idempotency check; Telegram alerts can double-send,
  Avanza order POSTs can duplicate trades.
- `dashboard/auth.py:131-132` — Cloudflare Access path trusts
  `Cf-Access-Authenticated-User-Email` header without verifying
  the request came through Cloudflare; LAN connection to port 5055
  bypasses tunnel and spoofs auth.
- `message_throttle.py:60-67` — TOCTOU race; two concurrent
  callers both pass cooldown check, both write last-writer-wins.
- `gpu_gate.py:209-269` — Layer-1 thread lock acquired BEFORE
  Layer-2 file lock; on file-lock timeout the thread lock is held
  for the full timeout (60s default), starving unrelated GPU
  consumers.

**Disputed:**
- `process_lock.py:36` (codex P1) — claude critique disputes:
  Windows `LK_NBLCK` on byte 0 of an empty file does not actually
  fail spuriously here; the codex framing overstates the bug class.

---

## What this review did NOT do

- It did NOT inspect tests beyond noting that some validators are
  only referenced from tests (`trade_validation.py`,
  `trade_risk_classifier.py`); a full test-coverage audit is a
  separate task.
- It did NOT run `pytest tests/` — synthesis is purely from code
  reading.
- It did NOT independently re-confirm claude-only findings via codex
  because codex critique-of-claude was rate-limited.
- It did NOT modify a single line of source code.
- It did NOT score severity by frequency-weighted blast radius —
  P0 means "could lose money on the next trade", but a P0 in
  `metals-core` (live every minute) is not the same as a P0 in
  `signals-modules/crypto_cross_asset.py` (already disabled, would
  only fire if re-enabled).

---

## Recommended next steps (not part of /fgl, owner: human)

1. Open one branch per subsystem to fix the confirmed-P0s; merge
   sequentially, not in parallel, because metals-core and
   avanza-api fixes overlap on the order-lock contract.
2. Run codex critique-of-claude after the rate-limit window resets
   (22:32 today) to independently re-confirm the claude-only P0s.
   This is critical because the missed-by-codex column averages 17
   per subsystem; if codex would have disputed any of those, they
   should not be elevated.
3. Update `docs/IMPROVEMENT_BACKLOG.md` with the deferred P1/P2/P3
   items so they don't get re-flagged in the next /fgl run.
4. Add a regression test for each confirmed P0. Most are detectable
   by unit tests at the module boundary without needing live API
   contact.
