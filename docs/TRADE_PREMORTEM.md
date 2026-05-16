# Pre-decision trade premortem

Apply this **before** issuing a BUY/SELL/HOLD verdict or recommending an order ladder. Mirrors the plan-time premortem in `/fgl` (commit `41702363`) and the diff-time premortem in `/fin-prereview` (commit `87d264e4`), at the third critical decision point: **trade-time**.

## Mechanism

Imagine you've already placed the order this verdict recommends. Fast-forward to the close of the relevant horizon (minutes-to-hours for warrants, hours-to-days for spot crypto, days-to-weeks for swing positions). The trade closed at a **loss** — or at breakeven after sitting through a drawdown you couldn't have tolerated. Why?

Write a `## Premortem` block **above** the verdict line, enumerating **≥3 distinct failure narratives** (≥5 for warrants and leveraged products). Each narrative = concrete causal chain (`X happened because Y assumption was wrong, manifested as Z`), not vague "could go wrong".

## Required coverage

Each premortem block MUST include at least one narrative from each category that applies to the trade:

1. **Direction wrong (regime mismatch).** Signal consensus disagrees with primary trend at one timeframe higher (`feedback_fishing_direction.md`). Signal generated in different volatility regime than current.
2. **Sizing wrong.** Position size too big for instrument's typical intraday range (ATR violation); too small to clear minimum courtage (`feedback_min_order_size_1000_sek.md`); over-leverages session given account `1625505` cash limit (`feedback_check_buying_power.md`).
3. **Stop too tight.** Stop inside typical intraday volatility band of instrument; wicked out by noise (`feedback_stops_outside_volatility.md`, `feedback_wider_stop_losses.md`).
4. **Stop too wide / no stop.** Loss exceeds what the thesis can absorb; no automatic invalidation level (`feedback_stops_outside_volatility.md`).
5. **Entry timing.** Move already happened — RSI extreme, > 1 ATR from entry trigger, signal computed > 5 min ago (`feedback_live_price_every_query.md`, `feedback_repull_before_orders.md`).
6. **Liquidity / execution.** Spread too wide for the leg; can't exit at planned target; hit ask instead of placing limit (`feedback_fishing_patience.md`, `feedback_no_panic_sell.md`).
7. **Signal accuracy.** Voter mix dominated by sub-50% signals that should have been gated, not inverted (`feedback_no_signal_inversion.md`); confidence below floor (`feedback_signal_confidence_threshold.md`).
8. **Calibration warnings ignored.** `system_lessons.json` flagged this asset/direction as previously underconfident — leaning against the bias amplifies the prior loss pattern (`feedback_weight_calibration_warnings.md`).
9. **Macro window.** FOMC/CPI/NFP within trade horizon; macro-window gate should have force-HOLD'd or downweighted (`project_macro_window_gating_20260428.md`).
10. **Stale position state.** Position / open orders / stops re-pulled > 5 min ago; conversation-start data is stale (`feedback_repull_before_orders.md`).

For each narrative: one-line mitigation (tighter gate, wider stop, smaller size, defer entry, etc.) or mark `ACCEPT` with reasoning.

## Output position

The premortem goes **immediately above the verdict line**, not in an appendix. Reader sees the failure modes before the recommendation. Skim-readers must encounter the risk surface before the directional call.

```
## Premortem
1. **Direction wrong**: ... → mitigation: ...
2. **Stop too tight**: ... → mitigation: ...
3. **Entry timing**: ... → mitigation: ...

## Verdict
P(direction) = ..., confidence = ..., historical accuracy on similar calls = ...
Recommendation: ...
```

## When to skip

- Pure information requests ("show me the current price") — no verdict, no premortem.
- Read-only diagnostics (health, status, accuracy reports) — no recommendation, no premortem.
- Already-open positions where the question is "should I exit now?" — the exit decision itself gets a premortem ("exit now turns out wrong because…"), even if you skip the entry-style coverage list.

## Why a premortem at trade-time, not just plan-time

`/fgl` premortem catches code-level failure modes. `/fin-prereview` premortem catches diff-level failure modes. **Neither catches trade-level failure modes** — those depend on the specific entry/stop/size at this moment, not on the code that generated the signal. The pattern is identical (prospective hindsight), but the failure taxonomy is different. See `feedback_premortem_pattern.md` for the broader two-shot pattern (now three-shot with this).

## History

- 2026-03-30 fishing session — direction conviction ignored, lost on bull-fish during downtrend (`feedback_fishing_lessons_20260330.md`).
- 2026-03-31 fishing session — regime detection failure, no pre-flight check (`feedback_fishing_20260331.md`).
- 2026-05-05 fin-gold — calibration warning ignored, trim recommended against 5x prior wrong-direction calls (`feedback_weight_calibration_warnings.md`).
- 2026-05-05 stop mismatch — 853→428 stop, stale conversation-start data, no re-pull before order (`feedback_repull_before_orders.md`).

Each of these would have been caught by a 30-second pre-trade premortem.
