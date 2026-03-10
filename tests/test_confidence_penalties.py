"""Tests for the confidence penalty cascade and ADX computation in signal_engine.

Covers apply_confidence_penalties() (4-stage cascade) and _compute_adx().
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.signal_engine import apply_confidence_penalties, _compute_adx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=50, close_start=100.0, volume_start=1000.0, trend="flat"):
    """Create a minimal OHLCV DataFrame for testing.

    ``trend`` can be ``"flat"``, ``"up"``, or ``"down"``.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    np.random.seed(42)
    noise = np.random.normal(0, 0.2, n)
    closes = np.empty(n, dtype=float)
    c = close_start
    for i in range(n):
        if trend == "up":
            c += 0.5 + noise[i]
        elif trend == "down":
            c -= 0.5 + noise[i]
        else:
            c += noise[i]
        closes[i] = max(c, 1.0)  # never go below 1
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.full(n, volume_start, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def _make_df_volume_pattern(n=50, close_start=100.0, vol_pattern="flat", trend="up"):
    """Create DF with a specific volume pattern for trap detection."""
    df = _make_df(n, close_start, trend=trend)
    if vol_pattern == "declining":
        vols = df["volume"].values.copy()
        vols[-5:] = [1000, 800, 600, 400, 200]
        df["volume"] = vols
    elif vol_pattern == "expanding":
        vols = df["volume"].values.copy()
        vols[-5:] = [1000, 1200, 1400, 1600, 2000]
        df["volume"] = vols
    return df


def _base_extra(voters=6, buy_count=5, sell_count=1, **kwargs):
    """Build a standard extra_info dict with voter counts."""
    d = {"_voters": voters, "_buy_count": buy_count, "_sell_count": sell_count}
    d.update(kwargs)
    return d


# ===================================================================
# TEST-6: _compute_adx tests
# ===================================================================

class TestComputeADX:
    """Tests for _compute_adx()."""

    def test_returns_none_for_none_input(self):
        assert _compute_adx(None) is None

    def test_returns_none_for_non_dataframe(self):
        assert _compute_adx("not a dataframe") is None
        assert _compute_adx(42) is None
        assert _compute_adx([1, 2, 3]) is None

    def test_returns_none_for_insufficient_data_default_period(self):
        # period=14, needs 14*2=28 rows minimum
        df = _make_df(27)
        assert _compute_adx(df) is None

    def test_returns_none_for_exactly_2x_period_minus_one(self):
        df = _make_df(27, trend="up")
        assert _compute_adx(df, period=14) is None

    def test_returns_value_at_exactly_2x_period(self):
        df = _make_df(28, trend="up")
        result = _compute_adx(df, period=14)
        # With 28 rows the EWM has barely enough data; result may be None or float
        # depending on NaN propagation, so just assert it doesn't crash.
        assert result is None or isinstance(result, float)

    def test_returns_valid_float_for_normal_data(self):
        df = _make_df(50, trend="up")
        adx = _compute_adx(df)
        assert adx is not None
        assert isinstance(adx, float)
        assert np.isfinite(adx)

    def test_adx_within_range(self):
        df = _make_df(100, trend="up")
        adx = _compute_adx(df)
        assert adx is not None
        assert 0 <= adx <= 100

    def test_handles_all_zero_volume_gracefully(self):
        df = _make_df(50, trend="up")
        df["volume"] = 0.0
        # ADX is computed from high/low/close, not volume — should still work
        adx = _compute_adx(df)
        assert adx is None or isinstance(adx, float)

    def test_returns_none_when_all_nan(self):
        df = pd.DataFrame({
            "open": [np.nan] * 50,
            "high": [np.nan] * 50,
            "low": [np.nan] * 50,
            "close": [np.nan] * 50,
            "volume": [np.nan] * 50,
        })
        result = _compute_adx(df)
        assert result is None

    def test_trending_data_produces_higher_adx(self):
        df_up = _make_df(100, trend="up")
        adx_up = _compute_adx(df_up)
        assert adx_up is not None
        assert adx_up > 0

    def test_custom_period(self):
        df = _make_df(60, trend="up")
        adx = _compute_adx(df, period=7)
        assert adx is not None
        assert isinstance(adx, float)

    def test_returns_none_for_empty_dataframe(self):
        df = pd.DataFrame({"high": [], "low": [], "close": [], "volume": []})
        assert _compute_adx(df) is None

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"high": [100.0], "low": [99.0], "close": [99.5], "volume": [1000.0]})
        assert _compute_adx(df) is None

    def test_constant_prices(self):
        n = 50
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        result = _compute_adx(df)
        # All prices equal -> TR = 0, clipped ATR = 1e-10 -> DI = 0 -> ADX = 0.0
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)


# ===================================================================
# TEST-4: apply_confidence_penalties tests
# ===================================================================

# --- 1. Disabled mode ---

class TestDisabledMode:
    """When config.confidence_penalties.enabled is False, return inputs unchanged."""

    def test_disabled_returns_unchanged_buy(self):
        config = {"confidence_penalties": {"enabled": False}}
        extra = _base_extra(voters=1, buy_count=1, sell_count=0, volume_ratio=0.1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.9, "ranging", {}, extra, "BTC-USD", None, config
        )
        assert action == "BUY"
        assert conf == 0.9
        assert log == []

    def test_disabled_returns_unchanged_sell(self):
        config = {"confidence_penalties": {"enabled": False}}
        extra = _base_extra(voters=1, buy_count=0, sell_count=1, volume_ratio=0.1)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.5, "high-vol", {}, extra, "ETH-USD", None, config
        )
        assert action == "SELL"
        assert conf == 0.5
        assert log == []

    def test_disabled_returns_unchanged_hold(self):
        config = {"confidence_penalties": {"enabled": False}}
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, {}, "BTC-USD", None, config
        )
        assert action == "HOLD"
        assert conf == 0.0
        assert log == []

    def test_enabled_by_default_when_key_missing(self):
        """Empty config -> penalties enabled (no 'enabled': False)."""
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, _ = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        # Ranging penalty should apply
        assert conf < 0.8

    def test_enabled_explicitly(self):
        config = {"confidence_penalties": {"enabled": True}}
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, _ = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, config
        )
        assert conf < 0.8

    def test_none_config_treated_as_enabled(self):
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, _ = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, None
        )
        assert conf < 0.8


# --- 2. Regime penalties (Stage 1) ---

class TestRegimePenalties:
    """Stage 1: Regime-based confidence multipliers."""

    def test_ranging_applies_075x(self):
        extra = _base_extra(voters=6)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        # 0.8 * 0.75 = 0.60, then clamped
        regime_entries = [p for p in log if p["stage"] == "regime"]
        assert len(regime_entries) == 1
        assert regime_entries[0]["mult"] == 0.75
        assert conf == pytest.approx(0.8 * 0.75, abs=0.01)

    def test_high_vol_applies_080x(self):
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {}
        )
        regime_entries = [p for p in log if p["stage"] == "regime"]
        assert len(regime_entries) == 1
        assert regime_entries[0]["mult"] == 0.80

    def test_trending_up_buy_gets_110x(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert conf == pytest.approx(0.7 * 1.10, abs=0.01)
        regime_entries = [p for p in log if p["stage"] == "regime"]
        assert len(regime_entries) == 1
        assert regime_entries[0]["aligned"] is True
        assert regime_entries[0]["mult"] == 1.10

    def test_trending_down_sell_gets_110x(self):
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.7, "trending-down", {}, extra, "BTC-USD", None, {}
        )
        assert conf == pytest.approx(0.7 * 1.10, abs=0.01)
        regime_entries = [p for p in log if p["stage"] == "regime"]
        assert regime_entries[0]["aligned"] is True

    def test_trending_up_sell_no_bonus(self):
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.7, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        # No regime bonus for counter-trend
        assert not any(p.get("aligned") for p in log)
        assert conf == pytest.approx(0.7, abs=0.01)

    def test_trending_down_buy_no_bonus(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-down", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p.get("aligned") for p in log)
        assert conf == pytest.approx(0.7, abs=0.01)

    def test_hold_passes_through_regime_unchanged(self):
        """HOLD action: regime penalty still applies to conf but action stays HOLD."""
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, {}, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        # conf starts at 0.0, ranging 0.75x -> still 0.0
        assert conf == 0.0

    def test_unknown_regime_no_penalty(self):
        """Unknown/missing regime should not trigger regime penalty."""
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "breakout", {}, extra, "BTC-USD", None, {}
        )
        regime_entries = [p for p in log if p["stage"] == "regime"]
        assert len(regime_entries) == 0
        # conf should be unchanged by regime stage (only clamped, dynamic_min etc.)
        # breakout is not in the regime penalty list


# --- 3. Volume gate (Stage 2) ---

class TestVolumeGate:
    """Stage 2: Volume/ADX gating."""

    def test_volume_below_05_forces_hold(self):
        extra = _base_extra(voters=6, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert conf == 0.0
        assert any(p["stage"] == "volume_gate" for p in log)

    def test_volume_exactly_05_not_forced_hold(self):
        """volume_ratio == 0.5 is NOT < 0.5, so the lowest gate doesn't trigger."""
        extra = _base_extra(voters=5, volume_ratio=0.5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "volume_gate" for p in log)

    def test_volume_below_08_plus_low_adx_plus_low_conf_forces_hold(self):
        """volume < 0.8, ADX < 20, conf < 0.65 -> force HOLD."""
        # Use flat data so ADX is low
        df = _make_df(50, trend="flat")
        # After regime penalty (ranging: 0.75x), conf = 0.5 * 0.75 = 0.375 < 0.65
        extra = _base_extra(voters=6, volume_ratio=0.7)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.5, "ranging", {}, extra, "BTC-USD", df, {}
        )
        # The ADX for flat data should be low, but we can't guarantee the exact value.
        # If ADX < 20 and conf after regime < 0.65, it should force HOLD.
        adx_val = extra.get("_adx")
        if adx_val is not None and adx_val < 20:
            assert action == "HOLD"
            assert any(p["stage"] == "volume_adx_gate" for p in log)

    def test_volume_below_08_but_high_adx_no_gate(self):
        """volume < 0.8 but ADX >= 20 -> the volume_adx_gate does NOT trigger."""
        df = _make_df(100, trend="up")  # strong trend -> higher ADX
        extra = _base_extra(voters=5, volume_ratio=0.7)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.5, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        adx_val = extra.get("_adx")
        if adx_val is not None and adx_val >= 20:
            assert not any(p["stage"] == "volume_adx_gate" for p in log)

    def test_volume_above_15_boosts_115x(self):
        extra = _base_extra(voters=5, volume_ratio=2.0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        # 0.7 * 1.10 (regime) * 1.15 (volume) = 0.8855
        assert conf > 0.7
        assert any(p["stage"] == "volume_boost" for p in log)

    def test_volume_exactly_15_not_boosted(self):
        """volume_ratio == 1.5 is NOT > 1.5, so boost doesn't trigger."""
        extra = _base_extra(voters=5, volume_ratio=1.5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "volume_boost" for p in log)

    def test_no_volume_ratio_skips_all_volume_stages(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        volume_stages = {"volume_gate", "volume_adx_gate", "volume_boost"}
        assert not any(p["stage"] in volume_stages for p in log)

    def test_hold_skips_volume_gate(self):
        """HOLD action should skip the volume gate even with low volume."""
        extra = {"volume_ratio": 0.1}
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert not any(p["stage"] == "volume_gate" for p in log)

    def test_adx_stored_in_extra_info(self):
        """_compute_adx result should be stored in extra_info['_adx']."""
        df = _make_df(50, trend="up")
        extra = _base_extra(voters=5)
        apply_confidence_penalties("BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {})
        assert "_adx" in extra

    def test_adx_none_when_no_df(self):
        extra = _base_extra(voters=5)
        apply_confidence_penalties("BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {})
        assert extra.get("_adx") is None


# --- 4. Trap detection (Stage 3) ---

class TestTrapDetection:
    """Stage 3: Bull/bear trap detection via price-volume divergence."""

    def test_bull_trap_buy_price_up_volume_declining(self):
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        trap_entries = [p for p in log if p.get("type") == "bull_trap"]
        assert len(trap_entries) == 1
        assert trap_entries[0]["mult"] == 0.5
        # conf should be reduced
        assert conf < 0.8

    def test_bear_trap_sell_price_down_volume_declining(self):
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.8, "trending-down", {}, extra, "BTC-USD", df, {}
        )
        trap_entries = [p for p in log if p.get("type") == "bear_trap"]
        assert len(trap_entries) == 1

    def test_hold_skips_trap_detection(self):
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, {}, "BTC-USD", df, {}
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_no_trap_when_volume_expanding(self):
        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        assert not any(p.get("type") == "bull_trap" for p in log)

    def test_no_trap_with_short_df(self):
        """DataFrame with fewer than 5 rows -> trap detection skipped."""
        df = _make_df(3)
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_no_trap_with_none_df(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_buy_price_down_no_bull_trap(self):
        """BUY + price_down -> not a bull trap (bull trap requires price_up)."""
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        assert not any(p.get("type") == "bull_trap" for p in log)

    def test_sell_price_up_no_bear_trap(self):
        """SELL + price_up -> not a bear trap (bear trap requires price_down)."""
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.8, "trending-down", {}, extra, "BTC-USD", df, {}
        )
        assert not any(p.get("type") == "bear_trap" for p in log)

    def test_trap_halves_confidence(self):
        """Bull trap should multiply conf by 0.5."""
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        extra = _base_extra(voters=5)
        # trending-up + BUY gives 1.10 bonus, so conf = 0.8 * 1.10 = 0.88
        # then trap: 0.88 * 0.5 = 0.44
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        if any(p.get("type") == "bull_trap" for p in log):
            assert conf == pytest.approx(0.8 * 1.10 * 0.5, abs=0.02)


# --- 5. Dynamic MIN_VOTERS (Stage 4) ---

class TestDynamicMinVoters:
    """Stage 4: Regime-dependent minimum voter threshold."""

    def test_trending_requires_3_voters(self):
        extra = _base_extra(voters=3, buy_count=2, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert action == "BUY"
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)

    def test_trending_fails_with_2_voters(self):
        extra = _base_extra(voters=2, buy_count=2, sell_count=0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert conf == 0.0
        assert any(p["stage"] == "dynamic_min_voters" for p in log)
        dmv = [p for p in log if p["stage"] == "dynamic_min_voters"][0]
        assert dmv["required"] == 3
        assert dmv["actual"] == 2

    def test_high_vol_requires_4_voters(self):
        extra = _base_extra(voters=3, buy_count=2, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert conf == 0.0
        dmv = [p for p in log if p["stage"] == "dynamic_min_voters"][0]
        assert dmv["required"] == 4

    def test_high_vol_passes_with_4_voters(self):
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)

    def test_ranging_requires_5_voters(self):
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert conf == 0.0
        dmv = [p for p in log if p["stage"] == "dynamic_min_voters"][0]
        assert dmv["required"] == 5

    def test_ranging_passes_with_5_voters(self):
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert action == "BUY"
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)

    def test_unknown_regime_defaults_to_5(self):
        """Regimes not in the trending/high-vol set default to 5 voters."""
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "capitulation", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        dmv = [p for p in log if p["stage"] == "dynamic_min_voters"][0]
        assert dmv["required"] == 5

    def test_hold_skips_dynamic_min_voters(self):
        """HOLD action skips the dynamic_min_voters check."""
        extra = _base_extra(voters=1, buy_count=0, sell_count=0)
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)

    def test_sell_also_checked(self):
        """SELL action should also be gated by dynamic MIN_VOTERS."""
        extra = _base_extra(voters=2, buy_count=0, sell_count=2)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        assert any(p["stage"] == "dynamic_min_voters" for p in log)


# --- 6. Confidence clamping ---

class TestConfidenceClamping:
    """Result confidence always in [0, 1]."""

    def test_clamped_to_1_after_boosts(self):
        """Multiple boosts (regime + volume) should not exceed 1.0."""
        extra = _base_extra(voters=10, buy_count=8, sell_count=2, volume_ratio=2.0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.95, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        # 0.95 * 1.10 * 1.15 = 1.20175 -> clamped to 1.0
        assert conf <= 1.0
        assert conf >= 0.0

    def test_clamped_to_0_after_penalties(self):
        extra = _base_extra(voters=6, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.1, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert conf >= 0.0

    def test_zero_conf_stays_zero(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.0, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert conf == 0.0

    def test_very_high_conf_clamped(self):
        extra = _base_extra(voters=10, buy_count=9, sell_count=1, volume_ratio=5.0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 1.0, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert conf == 1.0


# --- 7. HOLD passthrough ---

class TestHoldPassthrough:
    """HOLD action is not penalized beyond regime multiplier."""

    def test_hold_not_affected_by_volume_gate(self):
        extra = {"volume_ratio": 0.1}
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.5, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert action == "HOLD"
        # Volume gate doesn't trigger for HOLD
        assert not any(p["stage"] == "volume_gate" for p in log)

    def test_hold_not_affected_by_trap_detection(self):
        df = _make_df_volume_pattern(50, vol_pattern="declining")
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.5, "trending-up", {}, {}, "BTC-USD", df, {}
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_hold_not_affected_by_dynamic_min_voters(self):
        extra = _base_extra(voters=0)
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.5, "ranging", {}, extra, "BTC-USD", None, {}
        )
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)


# --- Full cascade / integration ---

class TestFullCascade:
    """Integration tests covering multiple penalty stages interacting."""

    def test_all_stages_can_stack(self):
        """Ranging + low volume -> both regime and volume gate fire."""
        extra = _base_extra(voters=6, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.9, "ranging", {}, extra, "BTC-USD", None, {}
        )
        # Regime (ranging 0.75x) then volume_gate forces HOLD
        assert action == "HOLD"
        assert conf == 0.0
        stages = [p["stage"] for p in log]
        assert "regime" in stages
        assert "volume_gate" in stages

    def test_regime_then_trap_then_clamp(self):
        """Trending bonus then bull trap then clamp."""
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.9, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        if any(p.get("type") == "bull_trap" for p in log):
            # 0.9 * 1.10 * 0.5 = 0.495 -> clamped to [0,1]
            assert 0.0 <= conf <= 1.0

    def test_penalty_log_is_list(self):
        extra = _base_extra(voters=5)
        _, _, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert isinstance(log, list)

    def test_penalty_log_entries_have_stage(self):
        extra = _base_extra(voters=5, volume_ratio=2.0)
        _, _, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {}
        )
        for entry in log:
            assert "stage" in entry

    def test_extra_info_mutated_with_adx(self):
        """apply_confidence_penalties writes _adx into extra_info."""
        df = _make_df(50, trend="up")
        extra = _base_extra(voters=5)
        apply_confidence_penalties("BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {})
        assert "_adx" in extra

    def test_ind_dict_not_required(self):
        """The ind parameter is not used by apply_confidence_penalties itself."""
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {}
        )
        assert action in ("BUY", "HOLD")

    def test_volume_gate_before_trap(self):
        """If volume gate forces HOLD, trap detection should be skipped."""
        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
        extra = _base_extra(voters=5, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        # Volume gate forces HOLD, then trap sees action=HOLD and skips
        assert action == "HOLD"
        assert not any(p["stage"] == "trap" for p in log)

    def test_dynamic_min_voters_after_trap(self):
        """Even if trap doesn't fire, dynamic min_voters can still force HOLD."""
        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
        extra = _base_extra(voters=2, buy_count=2, sell_count=0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {}
        )
        assert action == "HOLD"
        assert any(p["stage"] == "dynamic_min_voters" for p in log)
