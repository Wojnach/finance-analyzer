# Dual Adversarial Review — Synthesis (2026-05-04)

**Method:** 8 subsystems × 2 reviewers (Claude + Codex), each working independently.
24 review docs total: 8 Claude reviews (`claude-N-*.md`), 8 Codex reviews
(`codex-N-*.md`), 8 cross-critiques (`cross-N-*.md`). This file is the ranked
punch list.

**Review confidence:** When both reviewers independently flag the same line,
confidence is very high. When only one flags it, the cross-critique
documents why the other missed it (e.g., requires runtime probe, requires
project memory context, requires specific OS behavior knowledge). Findings
without explanation of "why the other missed" got deprecated to P2.

**Scope:** Findings only. Implementation is a separate session — `/fgl`
explicitly says "no code changes during review."

---

## Summary by subsystem

| Subsystem | P0 | P1 | P2 | Notes |
|-----------|----|----|----|-------|
| 1. signals-core | 4 | 4 | 2 | Forecast accuracy path silently dead end-to-end (Codex). SQL accuracy methods skip neutral filter (Claude). Both compound. |
| 2. orchestration | 4 | 5 | 4 | Journal read/write race on Windows (Claude). XAG-USD silver-bias on stock triggers (Codex). |
| 3. portfolio-risk | 5 | 8 | 4 | Warrant marks go negative on knockout (Codex). Volatility annualization wrong for non-daily candles (Claude). Both money-math. |
| 4. metals-core | 6 | 4 | 2 | Hardware trailing stop has never worked (Claude). Live warrant catalog refresh broken (Codex). Two distinct silent feature deaths. |
| 5. avanza-api | 5 | 5 | 4 | `_api_delete()` 403/500 reported as success (Codex). Pension account leak (Claude). |
| 6. signals-modules | 3 | 7 | 3 | Calendar signal multiply broken (both). Metals seasonality silently dead (Codex). |
| 7. data-external | 3 | 8 | 3 | Microstructure OFI double-counted (Claude). Both reviewers agree on metals_precompute None-on-cache and econ_dates 14:00 UTC. |
| 8. infrastructure | 3 | 7 | 5 | `/mode` breaks config.json symlink (Codex). `log_rotation` is broken three independent ways. |
| **Total** | **33** | **48** | **27** | **108 distinct findings across 142 modules.** |

---

## P0 — Must fix before next live deploy

These are bugs that lose money, corrupt unrecoverable state, or silently dead-end
critical features. Each entry: `(reviewer)` `path:line` — title — *why it matters*.

### Money-losing math (portfolio-risk)
1. **(Codex)** `portfolio/warrant_portfolio.py:100-103` — implied warrant mark goes
   negative when underlying drops > 100/leverage% (e.g. 5x cert on >20% drop).
   *Negative position values flow into total_value_sek and portfolio summaries.
   Confidence 95%.*
2. **(Claude)** `portfolio/monte_carlo.py:63` (and `risk_management.py:461-462`) —
   `volatility_from_atr` hardcodes `sqrt(252/period)`. For 1h candles the correct
   factor is ~25, not ~4 — **5.9x vol underestimation**. *All MC stop-hit
   probabilities and VaR/CVaR understate tail risk.*
3. **(Codex)** `portfolio/monte_carlo.py:328` — `p_stop_hit_*` measures terminal
   `P(S_T < stop)` not path-minimum `P(min S_t < stop)`. *Understates intra-horizon
   stop-out risk on volatile multi-day horizons. Compounds with #2.*
4. **(Claude)** `portfolio/monte_carlo_risk.py:426` — raw
   `agent_summary.get("fx_rate", 10.0)` instead of `_resolve_fx_rate()`. *VaR in
   SEK off by ~8% when agent_summary is empty.*

### Silent feature death (metals-core, signals-core, signals-modules)
5. **(Claude)** `data/metals_loop.py:4793-4802` — hardware trailing stop calls
   `place_stop_loss` with kwargs the function does not accept. *The "hardware
   trailing stop" advertised in config has never worked. Every queue-path fill
   is left without broker-side trailing protection. Same silent-feature-dead
   class as the March-April outage.*
6. **(Claude)** `data/metals_swing_trader.py:142, 2738` — wrong import:
   `avanza_control.place_stop_loss` returns `(ok, dict)` but unpacked as
   `(success, stop_id)`. `pos["stop_order_id"]` stores a dict. *Hardware stops
   on swing-trader positions are never cancelled on exit.*
7. **(Codex)** `data/metals_warrant_refresh.py:171-173` — Avanza market-guide
   value objects not unwrapped; `bid <= 0` raises TypeError; abort →
   `load_catalog_or_fetch()` falls back to **stale cache forever**. *Live warrant
   discovery is broken.*
8. **(Codex)** `portfolio/forecast_signal.py:365-372` — forecast votes written
   in `chronos`/`prophet` nested payload but `compute_forecast_accuracy()` only
   scores `entry["sub_signals"]`. *Every backfilled row contributes zero scored
   votes — forecast accuracy reports are blank.*
9. **(Codex)** `portfolio/orb_predictor.py:384-388` — MINI long intrinsic value
   not floored at zero. *`format_prediction()` produces losses worse than -100%
   and negative `warrant_price_factor`.*
10. **(Codex)** `data/metals_execution_engine.py:137-141` — `chronos_24h_pct`
    return scaled with `sqrt(252)` (volatility scaling). *1% becomes 0.16
    instead of 2.52 — Chronos confidence 16x too small in execution decisions.*

### Project-rule violations (infrastructure, avanza-api)
11. **(Codex)** `portfolio/telegram_poller.py:361` — `/mode` writes through
    `atomic_write_json(config_path)` which `os.replace`s the **symlink** with a
    regular file. *First `/mode` command silently severs the external config
    sync. CLAUDE.md "NEVER commit config.json" rule violated by side effect.*
12. **(Claude)** `portfolio/avanza_session.py:671-712` — `get_positions()`
    returns positions from ALL accounts including pension (2674244). Project
    memory `feedback_isk_only.md` says ISK only. `fin_fish.py:1359-1360` calls
    without post-filtering.
13. **(Claude)** `portfolio/avanza/trading.py:80-81` — unified-package
    `place_order()` has no account whitelist guard. *Caller can pass
    `account_id="2674244"` and trade pension.*
14. **(Codex)** `portfolio/avanza_control.py:397-404` — `_api_delete()` 403/500
    return reported as `(True, result)`. *Caller treats stop-loss as cancelled
    while still live → trade attempt with reserved volume → reject loop or
    duplicate.*

### Data corruption (data-external, signals-core)
15. **(Claude)** `portfolio/microstructure_state.py:205-213` — `persist_state()`
    calls `get_microstructure_state(ticker)` which calls `record_ofi()`. **Every
    persist appends an OFI value with no new snapshot data.** *Inflates
    `_ofi_history`, corrupts z-score distribution.*
16. **(Claude)** `data/crypto_data.py:184-185` vs `portfolio/mstr_precompute.py:35,37` —
    Hardcoded MSTR BTC holdings differ by 6%; shares outstanding differ by 25%.
    Both claim "early 2026". *NAV premium signal can flip buy/sell.*
17. **(Codex+Claude joint, P0)** `portfolio/metals_precompute.py:137-256` —
    un-refreshed sources yield None instead of carrying forward last successful
    values. *Deep-context files oscillate between complete and empty every
    cycle.*
18. **(Claude)** `portfolio/signal_decay_alert.py:35-36` — raw `open() + json.load()`
    on actively-written `accuracy_cache.json`. Torn read → JSONDecodeError
    caught → silent `[]` return. *Signal decay alerts are silenced on the exact
    failure mode they're supposed to catch.*
19. **(Claude)** `portfolio/signal_db.py:271, 302-303, 330-331, 370` — SQL
    accuracy methods skip the `_MIN_CHANGE_PCT` neutral-outcome filter that the
    Python path uses. *Per-ticker confidence penalty silently bypassed for
    sub-52% tickers (MSTR at 47.8%).*

### Orchestration / loop reliability
20. **(Claude)** `portfolio/journal.py:28-40` — `load_recent` reads JSONL with
    bare `open()` while `atomic_append_jsonl` renames over the same file. *Windows
    PermissionError blocks the appender → journal entry NOT written →
    `check_agent_completion` sees `journal_written=False` → loop contract fires
    CRITICAL → fix-agent dispatcher spawns.*
21. **(Claude)** `portfolio/multi_agent_layer2.py:166-178` — file handle leak
    when `Popen()` raises between `open()` and `proc._log_fh = log_fh`.
22. **(Claude)** `portfolio/trigger.py:130-148` — `_check_recent_trade()` only
    catches `KeyError, AttributeError`. *JSONDecodeError/OSError crashes
    `check_triggers()`. trade_detected=False default suppresses post-trade
    reassessment trigger permanently until file is repaired.*
23. **(Claude)** `portfolio/agent_invocation.py:739-741` — auth-scan offset
    captured BEFORE `open()`. Log rotation between stat+open makes scan cover
    zero bytes. *Same class as the March-April outage — "Not logged in" in the
    new session would be invisible.*

### Codex P0 (signals-core, infrastructure)
24. **(Codex)** `portfolio/forecast_accuracy.py:341-348` — backfill `break` on
    max_entries cap exits before copying remaining unprocessed entries.
    `_write_predictions()` rewrites with only the prefix. *Deletes the rest of
    `forecast_predictions.jsonl` on any backlog at cap.*
25. **(Codex)** `portfolio/accuracy_stats.py:150-153` — SQLite-first reader
    becomes authoritative whenever DB has any rows; writes are best-effort. *One
    transient SQLite write error → permanent stale data for accuracy gating and
    decay checks.*
26. **(Claude)** `portfolio/log_rotation.py:320-327` — `rotate_jsonl` writes
    temp file without fsync, no .tmp cleanup on os.replace failure.
    *signal_log.jsonl 68MB rotation can be unrecoverable on power loss.*
27. **(Claude)** `portfolio/local_llm_report.py:37` — `path.read_text()` on
    actively-written `forecast_predictions.jsonl`. *Truncated lines silently
    dropped → corrupted accuracy stats.*

### Signals-modules
28. **(Codex)** `portfolio/signals/mean_reversion.py:460` (and
    `momentum_factors._apply_seasonality()`) — `hasattr(df.index, "hour")` is
    False for the standard OHLCV shape. *Metals seasonality profile silently
    never applied in production.*
29. **(Claude)** `portfolio/signals/econ_calendar.py:137` — returns BUY when
    `next_event() is None`. *Stale calendar (post 2026 dates) emits standing
    BUY on every ticker.*
30. **(Claude)** `portfolio/signals/credit_spread.py:285` — relative
    `config.json` path → silent HOLD when scheduled task CWD wrong. *Same fix
    pattern as `cot_positioning.py`.*

### Avanza-api (continued)
31. **(Codex)** `portfolio/avanza/market_data.py:55-57` — `Quote.from_api()`
    doesn't unwrap nested `quote` object → bid/ask/last/high/low all 0.0.
    *`portfolio.avanza.get_quote()` returns unusable quotes.*
32. **(Codex)** `portfolio/avanza_session.py:1158-1164` — re-arm reads only
    `sl["order"]` shape, but snapshot can be `orderEvent`. *Failed
    cancel-before-sell can leave position naked.*
33. **(Codex)** `portfolio/portfolio_mgr.py:68-69` — `_DEFAULT_STATE` shallow-copied;
    `holdings` and `transactions` are shared with the module-level default. *Mutating one
    fresh state contaminates later `load_state()` calls with ghost holdings.*

---

## P1 — Should fix soon (high-confidence bugs, non-immediate impact)

### signals-core
- (Claude) `signal_decay_alert.py:27` relative path default fails under PF-OutcomeCheck CWD.
- (Claude) `signal_engine.py:3475-3484` utility boost can cross 47% accuracy gate.
- (Claude) `accuracy_stats.py:1918-1929` ticker accuracy cache shared "time" key.
- (Claude) `signal_engine.py:3132-3139` `btc_proxy` bypasses accuracy gate.

### orchestration
- (Codex) `agent_invocation.py:147-149` `_extract_ticker()` defaults to XAG-USD for stock triggers.
- (Codex) `market_timing.py:334-336` Swedish-only holidays (Ascension Day) not consulted.
- (Codex) `agent_invocation.py:724-731` pf-agent.bat fallback uses originally-requested tier timeout instead of T3 900s.
- (Claude) `multi_agent_layer2.py:193-210` sequential `proc.wait()` consumes total budget on first specialist.
- (Claude) `bigbet.py:173-181` `CLAUDECODE` not stripped → silent nested-session in dev.

### portfolio-risk
- (Codex) `monte_carlo.py:304-307` 1d-calibrated drift reused for 3d run.
- (Claude) `kelly_sizing.py:91-95` blended-avg not FIFO → wrong Kelly inputs.
- (Claude) `risk_management.py:374` ATR stop anchored to entry, not current price.
- (Claude) `equity_curve.py:492-493` mixed gross/net basis between win_rate and profit_factor.
- (Codex) `trade_guards.py:286-291` over-counts BUY adds toward new-position quota.
- (Codex) `cumulative_tracker.py:129-130` rolling windows anchored to wall-clock not snapshot timestamp.
- (Codex) `portfolio_mgr.py:21-26` `total_fees_sek` missing from `_DEFAULT_STATE`.
- (Claude) `trade_validation.py:76` TOCTOU on cash check between concurrent L2.

### metals-core
- (Claude) `metals_swing_trader.py:3151` no cancel-before-sell → stop reservation conflict loop possible.
- (Codex) `exit_optimizer.py:617-621` `hold_ev` averages percentiles not mean (skew-sensitive on warrants).
- (Claude) `metals_swing_trader.py:537-538` UTC+1 fallback wrong half the year.
- (Claude) `metals_swing_trader.py:2426, 2758` hardcoded 21:55 EOD across DST gap.

### avanza-api
- (Codex) `avanza_client.py:102-106` `get_client()` BankID-only mode broken (requires TOTP).
- (Claude) `avanza/tick_rules.py:87` float multiply causes off-by-one tick.
- (Codex) `avanza_orders.py:389-393` Telegram failure flips order status to error → duplicate retry risk.
- (Codex) `avanza_client.py:151-152` session-backed `get_price()` returns wrong shape (zeros).
- (Claude) `avanza/trading.py:105-147` `modify_order()` no min-size guard.

### signals-modules
- (Codex) `calendar_seasonal.py:157-163` January Sell-in-May + January Effect double-vote BUY bias.
- (Codex) `calendar_seasonal.py:222-223` pre-holiday BUY misses Saturday gap (MLK/Presidents'/Memorial/Labor Day).
- (Claude) `calendar_seasonal.py` no asset-class guard (active at 3h for crypto/metals).
- (Codex) `futures_flow.py:286-287` `d["oi"]` KeyError aborts entire signal.
- (Claude) `futures_flow.py:118` `ls_ratio[-1]["longShortRatio"]` similar KeyError.
- (Claude) `volume_flow.py:323-324` NaN → BUY bias.
- (Claude) `volatility.py:160, 264` inconsistent `sqrt(365)` vs `sqrt(252)`.

### data-external
- (Codex+Claude joint) `econ_dates.py:155, 180, 224, 273` all events at 14:00 UTC — wrong by 1.5-5h on FOMC/CPI/NFP.
- (Codex) `data_collector.py:334-339` BUG-179 fix doesn't actually prevent yfinance hangs.
- (Claude) `crypto_precompute.py:185` funding rate fallback to 0.0 silently masks missing field.
- (Claude) `earnings_calendar.py:48-53` AV calls bypass budget counter.
- (Claude) `metals_precompute.py:407-409, 458-460` COT fetch no retry → 7-day silent failure.
- (Claude) `crypto_macro_data.py:208-218` reads stale `agent_summary_compact.json` for prices.
- (Codex) `data_refresh.py:30-31` futures backfill uses spot endpoint.
- (Codex) `crypto_macro_data.py:137` max_pain_value=-1 init bug.

### infrastructure
- (Claude) `log_rotation.py:298-309` gz archive decompress + truncate in-place.
- (Codex) `log_rotation.py:319-327` race with appenders → live records dropped.
- (Codex) `gpu_gate.py:33-34` `Path("Q:/models")` Linux portability.
- (Claude) `claude_gate.py:271-279` full-file scan + outside lock.
- (Claude) `vector_memory.py:272-281` raw `open()` on journal.
- (Codex) `llama_server.py:179-180` `UnboundLocalError` on malformed PID file.
- (Claude) `gpu_gate.py:98-102` dead `_write_lock` — maintenance trap.

---

## P2 — Worth addressing (correctness smells, edge cases)

Listed in cross-critique docs per subsystem; not duplicated here.

---

## Themes across subsystems

Patterns that appeared repeatedly suggest systemic issues:

### 1. Schema drift between writer and reader
- `forecast_signal.py` writes `chronos`/`prophet` payload; `forecast_accuracy.py` reads
  `sub_signals`.
- `avanza/market_data.py` parses raw response while `scanner.fetch_detail` correctly
  unwraps nested `quote`.
- `avanza_client.py` session-backed `get_price()` returns market-guide shape;
  `avanza_tracker.fetch_avanza_prices` expects `lastPrice`/`changePercent`.
- `data/crypto_data.py` and `mstr_precompute.py` have diverged hardcoded constants.

**Fix posture:** Introduce `pydantic` (or simple dataclasses) for inter-module
contracts. Or pick one canonical write path per data file.

### 2. Atomic I/O rule violations (raw `open()` / `read_text()`)
Every reviewed subsystem found at least one. CLAUDE.md says "Never raw
`json.loads(open(...).read())`" — but the rule is followed inconsistently. Specific
violations: `signal_decay_alert.py`, `local_llm_report.py`, `vector_memory.py`,
`config_validator.py`, `macro_context.py`, `journal.py` (load_recent),
`crypto_macro_data.py`.

**Fix posture:** Add a CI lint rule banning raw `open(...)` on `*.json`/`*.jsonl`
in `portfolio/` and `data/`.

### 3. Silent-failure-as-feature
Recurring pattern: `try: do_thing() except Exception: pass` with no log, no
metric, no alert. Examples: `fish_engine.py:654-655`, `_safe_fetch` paths,
`signal_decay_alert.py` returning `[]`, `econ_calendar.py:137` returning BUY
on data exhaustion, `_check_recent_trade()` narrow except clause that crashes
upward instead.

**Fix posture:** Replace bare `except: pass` with `except Exception as e:
logger.warning(...)` everywhere in `portfolio/` and `data/`. Add per-call
metric counters where the exception path matters.

### 4. Hardcoded path / CWD assumptions
`data/metals_llm.py` chdirs the entire process. `signal_decay_alert.py` default
`"data/accuracy_cache.json"`. `credit_spread.py` `load_json("config.json")`.
`gpu_gate.py` `Path("Q:/models")`. All silently fail under non-canonical CWD or
non-Windows.

**Fix posture:** Standard pattern `Path(__file__).resolve().parent.parent / "data" / ...`.
Audit every occurrence.

### 5. Time-related bugs
`econ_dates.py` 14:00 UTC for all events. `metals_swing_trader.py` UTC+1
fallback. `metals_swing_trader.py` hardcoded 21:55 EOD. `agent_invocation.py`
hour-only market-open check. `market_timing.py` ignores Swedish holidays.

**Fix posture:** All datetime work through one canonical helper. All "event time"
through one calendar lookup that knows release-time per event type.

### 6. Annualization conventions
Three independent volatility/return scaling bugs:
- `monte_carlo.py:63` `sqrt(252)` for any candle period.
- `volatility.py:160 vs 264` `sqrt(365) vs sqrt(252)` in same composite.
- `metals_execution_engine.py:137-141` Chronos return scaled as vol.

**Fix posture:** Centralize annualization in `indicators.py` with explicit
`candles_per_year` parameter.

---

## What both reviewers missed (suspected blind spots)

Aggregating "what both missed" sections from each cross-critique:

- **`telegram_notifications.py` Markdown escaping** — user text with `_*[]` breaks
  Telegram.
- **`telegram_poller.py` command authorization** — `/mode` lacks sender ID validation.
- **`process_lock.py` cross-worktree semantics** — two worktrees compete for same lock path.
- **Knockout barrier handling at exactly-zero implied price** — Codex caught negative;
  neither audited the post-clamp downstream behavior.
- **`fx_rates.py`** — neither flagged. USD/SEK staleness silently mis-prices.
- **`fomc_dates.py` calendar exhaustion** — same risk as `econ_calendar.py:137`.
- **Multi-strategy Patient/Bold concurrent edits** — both passed but no actual
  test of the per-strategy lock semantics.
- **Round-trip fee accounting with intra-position FX changes**.
- **`smart_money.py` (DISABLED)** — both skimmed; if re-enabled, BOS/CHoCH
  state machine deserves its own review.
- **`silver_fomc_loop.py` race against `metals_loop`** — neither audited.

---

## Implementation roadmap (suggested ordering)

The user said no code changes during review — this is for the follow-up session.

### Phase A: silent feature deaths + project-rule violations (1-2 days)
1. Fix `metals_loop.py:4793-4802` hardware trailing stop kwargs (#5).
2. Fix `metals_swing_trader.py:142, 2738` wrong place_stop_loss import (#6).
3. Fix `telegram_poller.py:361` config.json symlink overwrite (#11).
4. Fix `avanza_session.py:671-712` and `avanza/trading.py:80-81` account whitelist (#12, #13).
5. Fix `metals_warrant_refresh.py:171-173` value-object unwrap (#7).
6. Fix `metals_llm.py:27-28` import-time chdir (#4 in metals).

### Phase B: money-math (1-2 days)
7. Fix `warrant_portfolio.py:100-103` clamp at zero (#1).
8. Fix `monte_carlo.py:63` and `risk_management.py:461-462` annualization (#2).
9. Fix `monte_carlo.py:328` path-min not terminal (#3).
10. Fix `monte_carlo_risk.py:426` `_resolve_fx_rate` (#4).
11. Fix `orb_predictor.py:384-388` floor at zero (#9).
12. Fix `metals_execution_engine.py:137-141` Chronos return scaling (#10).

### Phase C: data corruption + atomic I/O (1-2 days)
13. Fix `microstructure_state.py:205-213` double-count (#15).
14. Reconcile `crypto_data.py` and `mstr_precompute.py` constants (#16).
15. Fix `metals_precompute.py:137-256` carry-forward (#17).
16. Fix `signal_decay_alert.py:35-36` atomic load (#18).
17. Fix `signal_db.py:271+` neutral filter (#19).
18. Fix `local_llm_report.py:37` atomic load (#27).

### Phase D: orchestration reliability (1 day)
19. Fix `journal.py:28-40` (#20).
20. Fix `multi_agent_layer2.py:166-178` (#21).
21. Fix `trigger.py:130-148` (#22).
22. Fix `agent_invocation.py:739-741` (#23).

### Phase E: forecast accuracy path (1 day)
23. Fix `forecast_signal.py:365-372` schema (#8).
24. Fix `forecast_accuracy.py:341-348` data loss (#24).
25. Fix `accuracy_stats.py:150-153` SQLite-vs-JSONL (#25).

### Phase F: log rotation rewrite (0.5 day)
26. Rewrite `log_rotation.py` using atomic_write_json pattern + append lock (#26 + Codex+Claude P1s).

### Phase G: signals-modules cleanup (1 day)
27. Fix `mean_reversion.py:460` and `momentum_factors._apply_seasonality()` (#28).
28. Fix `econ_calendar.py:137` (#29).
29. Fix `credit_spread.py:285` absolute path (#30).
30. Fix `_api_delete()` ok-flag (#14).
31. Calendar signal fixes (P1s).
32. KeyError guards (P1s).

Then P1s and P2s.

**Total estimate:** 7-10 focused days for all P0+P1. P2 is a steady backlog.

---

## Files

| Doc | Purpose |
|---|---|
| `PLAN.md` | Execution plan for this dual review |
| `subsystems.txt` | Concrete file partition into 8 subsystems |
| `adversarial-prompt.md` | Prompt template for codex (unused — codex review uses default prompt) |
| `branch-shas.txt` | The 8 review-branch SHAs (cleaned up after this commit) |
| `claude-N-*.md` | 8 Claude independent reviews |
| `codex-N-*.md` | 8 Codex independent reviews |
| `cross-N-*.md` | 8 cross-critique reconciliations |
| `SYNTHESIS.md` | This file — single ranked punch list |

---

## Appendix: review confidence calibration

- **Both reviewers flagged same line / very close lines:** Highest confidence.
  Examples: `econ_dates.py 14:00 UTC` (P1, joint), `metals_precompute.py None
  on cache miss` (P0, joint).
- **One reviewer found, other reviewer's "Did NOT find" explicitly addressed
  the same area but reached opposite conclusion:** Mark as Disagreement in cross-critique.
  *None occurred in this run.*
- **Only one reviewer found, other didn't audit that path:** Confidence as
  stated by the finder; cross-critique documents the miss to inform future
  reviewer prompts.

The independence of the two reviews is preserved by ordering: Claude reviewers
finished writing their docs **before** any Codex output was read. The first peek
at codex output happened in `cross-1-signals-core.md`. Cross-contamination is
limited to the cross-critique step itself, which is the intended interaction.
