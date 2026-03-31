# Smart Fishing Architecture Plan

## Problem

`/fin-fish` is a dumb price table that computes ATR-based levels and shows EV.
When monitoring a position, the agent polls a price number without signal awareness.
Meanwhile `/fin-silver` has 30+ signals, Monte Carlo, cross-asset correlations,
prophecy beliefs, and learned lessons -- none of which feed into fishing.

The user wants `/fin-fish` to be as intelligent as `/fin-silver` -- knowing which
signals matter for the instrument, continuously tracking them, and using them
to drive entry/exit decisions.

## Architecture

### 1. Instrument Profile (`portfolio/instrument_profile.py`)

Each metal gets a "personality" defining what signals matter and how to interpret them.

```python
PROFILES = {
    "XAG-USD": {
        "name": "Silver",
        "trusted_signals": ["econ_calendar", "fear_greed", "claude_fundamental",
                            "momentum_factors", "structure"],  # >70% accuracy
        "ignored_signals": ["sentiment", "ministral", "oscillators"],  # <45% accuracy
        "cross_asset_drivers": {
            "DXY": {"correlation": -0.65, "lead_minutes": 15},
            "copper": {"correlation": 0.55, "lead_minutes": 30},
            "gold": {"correlation": 0.85, "lead_minutes": 5},
            "gold_silver_ratio": {"mean": 62, "revert_threshold": 5},
        },
        "regime_behaviors": {
            "trending-up": {"fish_direction": "LONG", "tp_mult": 1.5},
            "trending-down": {"fish_direction": "SHORT", "tp_mult": 1.5},
            "ranging": {"fish_direction": "BOTH", "tp_mult": 1.0},
        },
        "precompute_file": "data/silver_deep_context.json",
        "prophecy_key": "silver_bull_2026",
        "typical_daily_range_pct": 5.0,
        "typical_hourly_vol_pct": 0.4,
    },
    "XAU-USD": {
        "name": "Gold",
        "trusted_signals": ["econ_calendar", "macro_regime", "claude_fundamental",
                            "fibonacci", "structure"],
        "ignored_signals": ["sentiment", "oscillators"],
        "cross_asset_drivers": {
            "DXY": {"correlation": -0.70, "lead_minutes": 10},
            "real_yields": {"correlation": -0.60, "lead_minutes": 60},
            "VIX": {"correlation": 0.40, "lead_minutes": 15},
        },
        "precompute_file": "data/gold_deep_context.json",
        "prophecy_key": None,
        "typical_daily_range_pct": 2.9,
        "typical_hourly_vol_pct": 0.25,
    },
}
```

### 2. Smart Analysis Integration (`portfolio/fin_fish.py` enhancement)

Before computing fishing levels, load the same intelligence `/fin-silver` uses:

1. Load `agent_summary_compact.json` -- full signal data for the ticker
2. Load `silver_deep_context.json` / `gold_deep_context.json` -- precomputed context
3. Load `system_lessons.json` -- calibration data
4. Load signal reliability for this ticker -- trust ranking
5. Load prophecy belief -- long-term conviction context

Use this to:
- **Filter fishing levels** by trusted signals (not just ATR)
- **Set direction conviction** from signal-weighted probabilities (not just RSI)
- **Compute realistic targets** using Monte Carlo bands + structural levels
- **Show a /fin-silver-style briefing** alongside the fishing table

### 3. Signal-Aware Monitoring Loop (`portfolio/fish_monitor_smart.py`)

New module. Core loop (60s cycle):

```
while position_active and within_session:
    1. Fetch live price (Binance FAPI)
    2. Re-run preflight scoring -> detect conviction shifts
    3. Check cross-asset leading indicators:
       - DXY movement (15-min lead on silver)
       - Copper movement (30-min lead)
       - Gold movement (5-min lead)
       - Gold/silver ratio shift
    4. Re-compute key metrics:
       - RSI (current vs entry)
       - Z-score (mean reversion progress)
       - Half-life countdown
       - GARCH vol regime (expanding/compressing)
    5. Check exit triggers:
       - TP0 hit (+5% -> sell 30%)
       - Conviction dropped >20 pts from entry
       - Regime shifted (overnight move reconciliation)
       - Time decay (>3h -> tighten stops)
       - Cross-asset divergence (gold up, silver down)
    6. Log state to fish_monitor_log.jsonl
    7. Alert on significant changes (console + optional Telegram)
```

### 4. Exit Intelligence (`portfolio/fish_exit_signals.py`)

Instead of static TP/SL cascades, compute exits from signals:

- **Z-score normalized exits**: sell when z-score returns to +-0.5 from extreme
- **RSI crossover exits**: RSI crosses 50 from overbought/oversold
- **Half-life timeout**: if position held > 2x half-life, close (mean reversion failed)
- **Cross-asset divergence**: gold rallying but silver flat -> close short silver
- **Conviction-based**: if preflight score drops below 30 from >60, close
- **Time-based**: cascade tighten at 3h/5h/session-end (existing, keep)

## Batch Plan

### Batch 1: Instrument Profile + Data Loading
Files: `portfolio/instrument_profile.py` (NEW), `portfolio/fin_fish.py` (modify)
- Create instrument profile module with signal trust rankings
- Integrate deep context loading into fin_fish analysis
- Add profile-aware direction selection (replace pure RSI logic)

### Batch 2: Smart Monitoring Loop
Files: `portfolio/fish_monitor_smart.py` (NEW), `scripts/fin_fish.py` (modify)
- Create signal-aware monitoring loop
- Add `--monitor` flag to fin_fish CLI
- Cross-asset leading indicator checks
- Conviction tracking and shift detection

### Batch 3: Exit Intelligence
Files: `portfolio/fish_exit_signals.py` (NEW), `portfolio/fish_monitor_smart.py` (modify)
- Z-score and half-life exit logic
- RSI crossover detection
- Cross-asset divergence exits
- Conviction-based exits
- Integration with monitoring loop

### Batch 4: Enhanced Analysis Output
Files: `portfolio/fin_fish.py` (modify)
- /fin-silver-style briefing (signal reliability, cross-asset context, prophecy)
- Signal credibility ranking display
- Monte Carlo-informed fishing levels (not just ATR)
- Timeframe-routed signal display (3h vs 1d vs 3d)

### Batch 5: Tests + Integration
Files: `tests/test_instrument_profile.py`, `tests/test_fish_monitor_smart.py`,
       `tests/test_fish_exit_signals.py` (all NEW)
- Unit tests for instrument profile
- Unit tests for monitoring state machine
- Unit tests for exit signal logic
- Integration test: full fishing session simulation

## What Could Break

1. **Import cycles** -- fin_fish.py importing from instrument_profile which imports from signal modules.
   Mitigation: keep instrument_profile as pure data (no signal module imports).

2. **Performance** -- monitoring loop re-running signals every 60s could be heavy.
   Mitigation: only re-compute lightweight signals (RSI, z-score, cross-asset).
   Full signal re-evaluation every 15 min.

3. **Rate limits** -- yfinance for cross-assets has 5-min cache.
   Mitigation: use existing `metals_cross_assets.py` cache, don't bypass it.

4. **State persistence** -- monitoring state (conviction history, alerts) needs to survive restarts.
   Mitigation: use `file_utils.atomic_write_json()` to persist state.

## Not in Scope

- Automated order placement (user controls this)
- Warrant price prediction (depends on Avanza API)
- New signal modules (GARCH/half-life already added)
- Metals loop integration (separate process, different lifecycle)
