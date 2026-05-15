# Real Codex CLI Findings — All 8 Subsystems

Generated 2026-05-15T15:47:23Z from `codex-raw/*.lastmsg.txt`.

Each subsystem section is the verbatim 'last message' from
`codex exec review --commit <SHA>`. Findings here SUPPLEMENT the
Claude reviews and Codex-substitute reviews. See `SYNTHESIS.md`
for the prioritized punch-list.

## 1. signals-core

The patch introduces data loss in forecast backfill and several accuracy/calibration paths that can silently use or produce incorrect data. These affect runtime decision metrics and persisted logs, so the patch is not correct.

Full review comments:

- [P1] Preserve forecast rows after hitting max_entries — Q:\fa-review-0514\portfolio\forecast_accuracy.py:322-327
  When `updated` reaches `max_entries`, this `break` leaves the rest of `entries` out of `modified_entries`, and the subsequent `_write_predictions(modified_entries, path)` rewrites the JSONL with only the processed prefix. A routine backfill with more than `max_entries` eligible outcomes will therefore delete every later forecast prediction; preserve the unprocessed tail before writing or stop updating without truncating the file.

- [P2] Emit forecast rows in the format accuracy reads — Q:\fa-review-0514\portfolio\forecast_signal.py:355-359
  `compute_forecast_accuracy()` only looks at `sub_signals`/`raw_sub_signals`, but `run_forecasts()` creates rows with just `chronos` and `prophet` blocks. If this function is the producer for `forecast_predictions.jsonl`, those rows can get outcomes backfilled but will never be counted in forecast accuracy, so the forecast accuracy metrics remain empty for these predictions.

- [P2] Log raw LLM actions before vote gating — Q:\fa-review-0514\portfolio\signal_engine.py:3655-3656
  For local LLMs this reads from `votes` after `_gate_local_model_vote()` may have replaced a raw BUY/SELL with HOLD, while the confidence still comes from the model output. When qwen3 or ministral are held by the accuracy gate, the probability log records a confident HOLD instead of the model's raw prediction, corrupting calibration and Brier analysis; log the raw action or record this before gating.

- [P2] Expire ticker accuracy cache per horizon — Q:\fa-review-0514\portfolio\accuracy_stats.py:1923-1926
  Because `write_ticker_accuracy_cache()` refreshes the same `time` key for every horizon, a fresh write for `1d` makes any old `3h` or `4h` block pass this TTL check. Callers such as `signal_engine` then use stale per-ticker accuracy for horizon-specific gates and exemptions, so keep per-horizon timestamps or expire each horizon independently.

- [P2] Expire regime accuracy cache per horizon — Q:\fa-review-0514\portfolio\accuracy_stats.py:1388-1390
  Because `write_regime_accuracy_cache()` updates a single shared `time`, computing one horizon refreshes the TTL for all other cached regime blocks. A stale `3h` regime accuracy block can be served as fresh after a `1d` write and then overlay signal weights and gates for the wrong horizon in `signal_engine`; use per-horizon timestamps here as well.

## 2. orchestration

Several orchestration paths can use wrong or stale context, and core health/reflection metrics are incorrect under normal operating scenarios. These issues can mislead Layer 2 decisions or suppress expected triggers, so the patch should not be considered correct as-is.

Full review comments:

- [P2] Match stock consensus/move trigger tickers — Q:/fa-review-0514/portfolio/agent_invocation.py:274-274
  Stock trigger reasons emitted by `check_triggers` are `"{ticker} consensus ..."` and `"{ticker} moved ..."`; this regex only accepts stock symbols followed by `flipped`/`crossed`/`broke`, so those triggers fall through to the `XAG-USD` default. In those cases trade-guard blocking, decision feedback, and multi-agent specialist prompts run against the wrong instrument.

- [P2] Avoid stale specialist reports after failures — Q:/fa-review-0514/portfolio/agent_invocation.py:849-849
  When `layer2.multi_agent` is enabled and any specialist times out or auth-fails before writing its report, `wait_for_specialists` records `False` but this still builds a synthesis prompt over the fixed report paths. Because the report files are never cleaned before launch and `cleanup_reports()` is unused, the synthesis agent can read a previous run's report for another ticker as if it were current.

- [P2] Persist FTD confirmation age by date — Q:/fa-review-0514/portfolio/market_health.py:244-246
  With the normal fixed-length 90-day data fetch, `ftd_day_offset` is persisted as an array index (`n - 1` when detected). On later refreshes `n` remains the same while the saved index is unchanged, so this expression never exceeds the failure window and `FTD_CONFIRMED` never promotes to `confirmed_uptrend` after 10 days; market health stays understated until another state change.

- [P2] Do not treat open positions as losses — Q:/fa-review-0514/portfolio/reflection.py:80-80
  After an open BUY, cash drops by the cost of the position while the holding still has value, so computing `total_pnl_pct` from cash alone reports a large loss even if the portfolio is flat. These reflection metrics feed Layer 2 and can generate false "down X% — reduce size" insights whenever positions are open.

- [P3] Convert probability fractions before formatting — Q:/fa-review-0514/portfolio/autonomous.py:710-710
  `focus_probabilities` stores probabilities as fractions (the daily digest and focus analysis multiply by 100), but Mode B appends the raw value with a percent sign. When `notification.mode` is `probability`, a `0.62` probability is displayed as `0.62%` instead of `62%`, making the Telegram summary misleading.

- [P3] Handle backward clock jumps in flip cooldown — Q:/fa-review-0514/portfolio/trigger.py:271-271
  Because `flip_cooldowns` are persisted wall-clock timestamps, a backward NTP/manual clock adjustment after a flip makes `_flip_now_ts - last_flip_ts` negative. This branch then suppresses every sustained flip for that ticker until the clock catches up, potentially much longer than the 30-minute cooldown.

## 3-8. portfolio-risk / metals-core / avanza-api / signals-modules / data-external / infrastructure

Real Codex quota exhausted before reviewing subs 3-8 (reset 22:27 local).
Codex-substitute reviews (Claude subagent with isolated context) covered
these subsystems — see `codex-N-{subsystem}.md` and `cross-N-{subsystem}.md`.

**Re-run instructions** when quota resets:

```bash
cd Q:/fa-review-0514
bash run_codex_all.sh   # script skips subs that already have lastmsg.txt — re-edit to start at sub 3
```

The `branch-shas.txt` records the commit SHAs to feed `codex exec review --commit`.

