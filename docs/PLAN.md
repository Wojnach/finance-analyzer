# Plan — Macro-event regime gating (auto-adapt signal weights during event-heavy windows)

## Context

The accuracy audit shipped 2026-04-28 PM (merge `47b4d474`) confirmed that 19
of 21 flagged per-ticker signals are statistically REAL degradation —
sentiment, momentum_factors, structure, and claude_fundamental dropping
hardest across BTC/ETH/XAG/MSTR. Investigation traced the root cause to
the past 7 days being the densest macro-event window of 2026 (FOMC, CPI,
NFP, four central banks, Mag 7 earnings). Technical/sentiment signals
trained on price-pattern continuity get systematically wrong when macro
news drives prices.

User asked: *"sounds like we need to identify when these events happen
and adapt the config to this timeline and then change back when events
subside"*. Confirmed (`yes`). This plan scopes that work.

The system has partial infra: `econ_dates.recent_high_impact_events()`
exists; `accuracy_degradation` already short-circuits during a 24h ±
FOMC/CPI/NFP window. **The gap is that signal *weights* don't adapt — only
the alert layer does.** During a macro week we keep voting with the same
technical signals that the news is overriding, then go quiet about how
badly they're doing.

## Goal

When high-impact macro events are within 24h past or 72h future:
1. **Auto-detect** the macro window from existing `econ_dates` data
2. **Down-weight or force-HOLD** the four signals known to fail in news-driven regimes
3. **Auto-revert** when the window passes (no manual config flip)
4. **Stay observable** — log enter/exit transitions; surface in journal so we can backtest the gate's value

## Non-goals (explicitly out of scope)

- **Expanding the macro calendar.** `econ_dates.py` only tracks FOMC, CPI,
  NFP, GDP. International central banks (ECB, BoE, BoJ) and earnings
  weeks are NOT in the calendar today. The plan ships against the
  current calendar; expanding it is a separate work item documented in
  "Roadmap" below.
- **Per-sector gating.** Existing `EVENT_SECTOR_MAP` could differentiate
  metals vs crypto, but for v1 we apply uniformly across all tickers.
- **Replacing the failing signals.** This is a temporary down-weight, not
  a signal redesign. The signals are regime-inappropriate, not broken.
- **Backtest harness.** Validation will be live-monitored over the next
  macro window; no synthetic backtest in v1.

## Approach

### 1. Add `is_macro_window()` to `portfolio/econ_dates.py`

```python
def is_macro_window(
    now=None,
    lookback_hours: float = 24.0,
    lookahead_hours: float = 72.0,
    impact_filter: tuple[str, ...] = ("high",),
) -> bool:
    """True iff a high-impact event is within ``lookback_hours`` past
    OR ``lookahead_hours`` future."""
```

Reuses existing `events_within_hours()` (forward) and
`recent_high_impact_events()` (backward). No new data sources.

### 2. Add macro-window weight overlay to `portfolio/signal_engine.py`

```python
MACRO_WINDOW_DOWNWEIGHT_SIGNALS = frozenset({
    "sentiment", "momentum_factors", "structure",
})
MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER = 0.5
MACRO_WINDOW_FORCE_HOLD_SIGNALS = frozenset({"claude_fundamental"})
_MACRO_WINDOW_CACHE_TTL_S = 300
```

### 3. Wire into `_weighted_consensus`

- Force-HOLD pre-pass after `_get_horizon_disabled_signals` (mutates votes
  dict to set MACRO_WINDOW_FORCE_HOLD_SIGNALS to "HOLD").
- Multiplier branch inside the weight loop, after `ic_mult`, applies
  `MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER` for downweight signals.

Composes cleanly with existing regime/horizon multipliers.

### 4. Observability

- Log macro-window state changes (entered/exited) at INFO once per transition.
- Mark journal entries with `regime_macro_window: True`.
- Optional: include `macro_window` block in `agent_summary` (`reporting.py`).

### 5. Tests — `tests/test_macro_window_gating.py`

- TestIsMacroWindow: empty calendar, future-12h, future-60h (in window),
  future-96h (out), past-12h (in 24h lookback), past-36h (out),
  medium-impact filtered, lookback=0.
- TestSignalEngineMacroWindowOverlay: macro=False unchanged,
  claude_fundamental forced HOLD, downweight signals × 0.5,
  multiplier compounds with regime, other signals untouched, cache TTL.

## Files to modify

| File | Change |
|---|---|
| `portfolio/econ_dates.py` | Add `is_macro_window()` |
| `portfolio/signal_engine.py` | Constants + cache helper + force-HOLD pre-pass + multiplier in `_weighted_consensus` |
| `portfolio/reporting.py` | Add `macro_window` block to `agent_summary` |
| `tests/test_macro_window_gating.py` | New |
| `docs/CHANGELOG.md` | New entry |
| `docs/SESSION_PROGRESS.md` | Append session summary |

## Existing functions to reuse

- `portfolio/econ_dates.events_within_hours(hours, ref_date)` — forward
- `portfolio/econ_dates.recent_high_impact_events(hours, impact_filter, ref_time)` — backward
- `portfolio/signal_engine._get_horizon_disabled_signals(ticker, horizon)` — pre-pass pattern

## Risks

- **False positives:** detector triggers in normal week — survivable
  (downweight is reduction, not suppression). Force-HOLD on
  claude_fundamental loses 1 voter for ~96h per FOMC; consensus ≥3
  voters still reachable.
- **False negatives:** events not in `econ_dates.py` (international CBs,
  earnings) don't trigger. Roadmap addresses.
- **Compounding multipliers:** macro 0.5× × ranging 0.75× × horizon 1.3×
  = 0.49× — strong reduction. Fine for v1.
- **Cache TTL 5min** — acceptable; events have hourly cadence at fastest.
- **No backtest in v1** — journal flag enables post-hoc A/B comparison.

## Roadmap (not v1)

1. Expand calendar with ECB, BoE, BoJ, earnings windows.
2. Per-sector gating via `EVENT_SECTOR_MAP`.
3. Severity scoring (FOMC > CPI > NFP > GDP).
4. Backtest harness replaying signal_log against the gate.

## Verification

1. Force `is_macro_window=True` via stub OR wait for next high-impact event.
2. Journal entries inside window have `regime_macro_window: True`.
3. `claude_fundamental` votes are HOLD.
4. sentiment/momentum_factors/structure weights halved (visible in
   agent_summary debug).
5. Window closes 24h post-event; auto-revert verified.
6. Re-run `scripts/audit_accuracy_drops.py` after window passes — expect
   fewer flagged signals.

## Effort

- ~80 LOC across 3 files; ~250 LOC tests
- 1 batch (single worktree commit)
- Codex review: 1-2 rounds expected
