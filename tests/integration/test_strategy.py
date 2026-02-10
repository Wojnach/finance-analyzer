"""
Integration tests for TABaseStrategy — runs inside the Freqtrade Podman container.

Execute with: ./scripts/ft-test.sh tests/integration/ -v
"""

import importlib
import sys

import pytest
import numpy as np
import pandas as pd


def _have_talib():
    try:
        import talib  # noqa: F401

        return True
    except ImportError:
        return False


def _have_freqtrade():
    try:
        import freqtrade  # noqa: F401

        return True
    except ImportError:
        return False


requires_talib = pytest.mark.skipif(
    not _have_talib(), reason="TA-Lib not available (run inside container)"
)
requires_freqtrade = pytest.mark.skipif(
    not _have_freqtrade(), reason="Freqtrade not available (run inside container)"
)


def _make_ohlcv(rows=200):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "open": close + np.random.randn(rows) * 0.1,
            "high": close + abs(np.random.randn(rows) * 0.3),
            "low": close - abs(np.random.randn(rows) * 0.3),
            "close": close,
            "volume": np.random.randint(100, 10000, rows).astype(float),
        }
    )


def _make_config():
    """Minimal config that satisfies @informative decorator requirements."""
    from freqtrade.enums import CandleType

    return {
        "strategy": "TABaseStrategy",
        "stake_currency": "USDT",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "candle_type_def": CandleType.FUTURES,
        "exchange": {"name": "binance", "pair_whitelist": ["BTC/USDT:USDT"]},
    }


def _add_1h_columns(df):
    """Add fake 1h informative columns that the @informative decorator would provide."""
    import talib

    df["close_1h"] = df["close"]
    df["ema_50_1h"] = talib.EMA(df["close"], timeperiod=50)
    df["adx_1h"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    return df


@requires_talib
def test_talib_available():
    import talib

    result = talib.RSI(np.random.randn(50), timeperiod=14)
    assert len(result) == 50


@requires_freqtrade
@requires_talib
def test_strategy_loads():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    mod = importlib.import_module("ta_base_strategy")
    assert hasattr(mod, "TABaseStrategy")
    strategy = mod.TABaseStrategy
    assert strategy.INTERFACE_VERSION == 3
    assert strategy.timeframe == "5m"


@requires_freqtrade
@requires_talib
def test_populate_indicators():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    result = s.populate_indicators(df, {"pair": "BTC/USDT"})

    for col in [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "ema_fast",
        "ema_slow",
        "volume_sma",
        "atr",
        "bb_lower",
        "bb_middle",
        "bb_upper",
        "cdl_bullish",
        "cdl_bearish",
    ]:
        assert col in result.columns, f"Missing column: {col}"

    valid_rsi = result["rsi"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


@requires_freqtrade
@requires_talib
def test_candlestick_pattern_scores():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    result = s.populate_indicators(df, {"pair": "BTC/USDT"})

    assert (result["cdl_bullish"] >= 0).all()
    assert (result["cdl_bearish"] >= 0).all()
    assert result["cdl_bullish"].max() >= 0
    assert result["cdl_bearish"].max() >= 0


@requires_freqtrade
@requires_talib
def test_populate_entry_trend():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})
    df = _add_1h_columns(df)
    result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})

    assert "enter_long" in result.columns


@requires_freqtrade
@requires_talib
def test_populate_exit_trend():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})
    df = _add_1h_columns(df)
    df = s.populate_entry_trend(df, {"pair": "BTC/USDT"})
    result = s.populate_exit_trend(df, {"pair": "BTC/USDT"})

    assert "exit_long" in result.columns


@requires_freqtrade
@requires_talib
def test_trend_filter_blocks_entries_in_downtrend():
    """When 1h trend is down, no entries should fire regardless of confidence."""
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})

    # Simulate downtrend: close below EMA, low ADX
    df["close_1h"] = df["close"] * 0.9
    df["ema_50_1h"] = df["close"]
    df["adx_1h"] = 15.0

    result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})
    enter_count = result["enter_long"].fillna(0).sum()
    assert enter_count == 0, f"Expected 0 entries in downtrend, got {enter_count}"


@requires_freqtrade
@requires_talib
def test_trend_filter_blocks_entries_low_adx():
    """When ADX is below threshold (choppy market), no entries should fire."""
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})

    # Simulate uptrend but weak (low ADX)
    df["close_1h"] = df["close"] * 1.1
    df["ema_50_1h"] = df["close"]
    df["adx_1h"] = 10.0

    result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})
    enter_count = result["enter_long"].fillna(0).sum()
    assert enter_count == 0, f"Expected 0 entries with low ADX, got {enter_count}"


@requires_freqtrade
@requires_talib
def test_trigger_guard_params():
    """Strategy should have trigger+guard params but not confidence params."""
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)

    assert hasattr(s, "rsi_oversold")
    assert hasattr(s, "volume_spike_mult")
    assert hasattr(s, "adx_threshold")
    assert not hasattr(s, "min_confidence")
    assert not hasattr(s, "pattern_weight")


@requires_freqtrade
@requires_talib
def test_custom_stoploss_exists():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    assert s.use_custom_stoploss is True
    assert hasattr(s, "custom_stoploss")
    assert callable(s.custom_stoploss)


@requires_freqtrade
@requires_talib
def test_custom_exit_exists():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    assert hasattr(s, "custom_exit")
    assert callable(s.custom_exit)


@requires_freqtrade
@requires_talib
def test_custom_stake_amount_exists():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    assert hasattr(s, "custom_stake_amount")
    assert callable(s.custom_stake_amount)


@requires_freqtrade
@requires_talib
def test_confirm_trade_entry_exists():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    assert hasattr(s, "confirm_trade_entry")
    assert callable(s.confirm_trade_entry)


@requires_freqtrade
@requires_talib
def test_atr_indicator_values():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    result = s.populate_indicators(df, {"pair": "BTC/USDT"})

    valid_atr = result["atr"].dropna()
    assert len(valid_atr) > 0
    assert (valid_atr > 0).all()


@requires_freqtrade
@requires_talib
def test_1h_informative_method():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    result = s.populate_indicators_1h(df, {"pair": "BTC/USDT"})

    assert "ema_50" in result.columns
    assert "adx" in result.columns
    valid_adx = result["adx"].dropna()
    assert (valid_adx >= 0).all() and (valid_adx <= 100).all()


@requires_freqtrade
@requires_talib
def test_hyperoptable_params():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = _make_config()
    s = TABaseStrategy(config)

    assert hasattr(s, "rsi_oversold")
    assert hasattr(s, "volume_spike_mult")
    assert hasattr(s, "adx_threshold")
    assert hasattr(s, "max_daily_loss_pct")
    assert hasattr(s, "max_drawdown_pct")

    assert s.rsi_oversold.space == "buy"
    assert s.volume_spike_mult.space == "buy"
    assert s.adx_threshold.space == "buy"
    assert s.max_daily_loss_pct.space == "sell"
    assert s.max_drawdown_pct.space == "sell"
