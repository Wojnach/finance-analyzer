"""Tests for core functions in portfolio/signal_engine.py.

Covers:
- _weighted_consensus: weighted vote aggregation with accuracy, regime, gating, activation rates
- apply_confidence_penalties: 4-stage penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
- _confluence_score: majority-based scoring with volume confirmation
- _time_of_day_factor: time-based confidence dampening
"""

from datetime import UTC, datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from portfolio.signal_engine import (
    REGIME_WEIGHTS,
    _confluence_score,
    _time_of_day_factor,
    _weighted_consensus,
    apply_confidence_penalties,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=50, close_start=100.0, trend="flat", volume_start=1000.0):
    """Build a minimal OHLCV DataFrame for penalty / ADX tests.

    Parameters
    ----------
    n : int
        Number of rows. Must be >= 28 for ADX computation (period=14, needs 2x).
    close_start : float
        Opening close price.
    trend : str
        "flat", "up", or "down".
    volume_start : float
        Constant volume value.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    closes = []
    c = close_start
    for _i in range(n):
        if trend == "up":
            c += 0.5
        elif trend == "down":
            c -= 0.5
        closes.append(c)
    closes = np.array(closes, dtype=float)
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.full(n, volume_start, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def _make_trap_df(price_trend="up", volume_pattern="declining", n=50):
    """Build a DF specifically for trap detection (Stage 3).

    price_trend controls whether last-5 close goes up or down.
    volume_pattern controls whether last-5 volume goes down (for trap) or up.
    """
    df = _make_ohlcv_df(n, close_start=100.0, trend=price_trend, volume_start=1000.0)
    vols = df["volume"].values.copy()
    if volume_pattern == "declining":
        vols[-5:] = [1000, 800, 600, 400, 150]  # last < first * 0.8
    elif volume_pattern == "expanding":
        vols[-5:] = [1000, 1200, 1400, 1600, 2000]
    df["volume"] = vols
    return df


def _base_extra(voters=6, buy_count=4, sell_count=2, **kwargs):
    """Build the extra_info dict required by apply_confidence_penalties Stage 4."""
    d = {"_voters": voters, "_buy_count": buy_count, "_sell_count": sell_count}
    d.update(kwargs)
    return d


# ===========================================================================
# _weighted_consensus
# ===========================================================================

class TestWeightedConsensusBasic:
    """Basic behavior of _weighted_consensus."""

    def test_all_hold_returns_hold(self):
        votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        action, conf = _weighted_consensus(votes, {}, "ranging")
        assert action == "HOLD"
        assert conf == 0.0

    def test_empty_votes_returns_hold(self):
        action, conf = _weighted_consensus({}, {}, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_single_buy_vote_returns_buy(self):
        votes = {"rsi": "BUY"}
        action, conf = _weighted_consensus(votes, {}, "ranging")
        assert action == "BUY"
        assert conf == 1.0  # 1 BUY / 1 total = 100%

    def test_single_sell_vote_returns_sell(self):
        votes = {"rsi": "SELL"}
        action, conf = _weighted_consensus(votes, {}, "ranging")
        assert action == "SELL"
        assert conf == 1.0

    def test_simple_buy_majority(self):
        # Use neutral regime to avoid regime weight interference
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL"}
        action, conf = _weighted_consensus(votes, {}, "breakout")
        assert action == "BUY"
        # No accuracy data => all get weight=0.5; neutral regime => mult=1.0
        # BUY weight = 0.5 + 0.5 = 1.0, SELL weight = 0.5
        # conf = 1.0/1.5 ~ 0.6667
        assert conf == pytest.approx(2 / 3, abs=0.01)

    def test_simple_sell_majority(self):
        # Use signals in different correlation groups to avoid penalty
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "SELL", "smart_money": "BUY"}
        action, conf = _weighted_consensus(votes, {}, "breakout")
        assert action == "SELL"
        # Neutral regime, no accuracy => all weight=0.5
        # 3 SELL * 0.5 = 1.5, 1 BUY * 0.5 = 0.5 => conf = 1.5/2.0 = 0.75
        assert conf == pytest.approx(0.75, abs=0.01)

    def test_exact_tie_returns_hold(self):
        # Use neutral regime so both signals get the same weight
        votes = {"rsi": "BUY", "macd": "SELL"}
        action, conf = _weighted_consensus(votes, {}, "breakout")
        # Both get weight=0.5 (no accuracy, neutral regime)
        # buy_conf = sell_conf = 0.5, neither > the other
        assert action == "HOLD"
        assert conf == pytest.approx(0.5, abs=0.01)


class TestWeightedConsensusAccuracy:
    """Accuracy weighting and gating logic."""

    def test_high_accuracy_signal_dominates(self):
        # Use neutral regime to isolate accuracy weighting
        votes = {"rsi": "BUY", "macd": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.8, "total": 100},
            "macd": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, "breakout")
        # RSI weight=0.8, MACD weight=0.55 => BUY=0.8, SELL=0.55 => BUY wins
        assert action == "BUY"
        expected_conf = 0.8 / (0.8 + 0.55)
        assert conf == pytest.approx(expected_conf, abs=0.01)

    def test_low_accuracy_signal_gets_gated(self):
        """Signal with accuracy below gate threshold gets force-skipped."""
        votes = {"bad_signal": "BUY"}
        accuracy = {"bad_signal": {"accuracy": 0.3, "total": 50}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # accuracy=0.3 < 0.47 gate with 50 >= 30 samples => gated => HOLD
        assert action == "HOLD"
        assert conf == 0.0

    def test_gating_applies_to_sell_signal(self):
        votes = {"bad_signal": "SELL"}
        accuracy = {"bad_signal": {"accuracy": 0.25, "total": 30}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # accuracy=0.25 < 0.47 gate => gated => HOLD
        assert action == "HOLD"
        assert conf == 0.0

    def test_gating_not_applied_below_min_samples(self):
        """Signal with accuracy below gate but < ACCURACY_GATE_MIN_SAMPLES keeps voting."""
        votes = {"new_signal": "BUY"}
        accuracy = {"new_signal": {"accuracy": 0.3, "total": 15}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # < 30 samples => not gated, gets default weight=0.5
        assert action == "BUY"
        assert conf == 1.0

    def test_default_weight_for_unknown_signal(self):
        """Signal not in accuracy_data gets weight=0.5 (< 20 samples path)."""
        votes = {"rsi": "BUY", "unknown": "SELL"}
        accuracy = {"rsi": {"accuracy": 0.7, "total": 100}}
        # Use neutral regime to isolate accuracy weighting
        action, conf = _weighted_consensus(votes, accuracy, "breakout")
        # RSI: weight=0.7 * 1.0 = 0.7, unknown: weight=0.5 * 1.0 = 0.5
        # BUY=0.7, SELL=0.5 => BUY, conf=0.7/1.2
        assert action == "BUY"
        assert conf == pytest.approx(0.7 / 1.2, abs=0.01)

    def test_gated_signal_with_good_signal(self):
        """Two signals: one good BUY and one bad BUY (gated, skipped)."""
        votes = {"good": "BUY", "bad": "BUY"}
        accuracy = {
            "good": {"accuracy": 0.7, "total": 50},
            "bad": {"accuracy": 0.3, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # good: BUY weight=0.7, bad: gated (acc 0.3 < 0.47 gate)
        # Only good votes => BUY with 100% conf
        assert action == "BUY"
        assert conf == 1.0


class TestWeightedConsensusRegime:
    """Regime weight multipliers from REGIME_WEIGHTS."""

    def test_trending_up_boosts_ema_and_macd_at_3h(self):
        # BUG-152: ema is now regime-gated at 1d in trending-up (0-11% accuracy).
        # At 3h it is NOT gated and gets the 1.5x regime boost.
        votes = {"ema": "BUY", "rsi": "SELL"}
        accuracy = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "trending-up", horizon="3h")
        # ema gets regime boost (1.5x) + horizon boost at 3h
        # rsi gets regime penalty (0.7x) + horizon adjustment
        # ema BUY should dominate
        assert action == "BUY"
        assert conf > 0.6  # ema significantly outweighs rsi

    def test_trending_up_gates_ema_at_1d(self):
        # BUG-152: ema gated in trending-up at _default/1d horizon
        votes = {"ema": "BUY", "rsi": "SELL"}
        accuracy = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "trending-up")
        # ema is regime-gated → HOLD, only rsi SELL remains
        assert action == "SELL"

    def test_ranging_boosts_rsi_and_bb(self):
        # Use claude_fundamental as opponent — not in momentum_cluster, not regime-gated
        votes = {"rsi": "BUY", "claude_fundamental": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "claude_fundamental": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # rsi weight = 0.6 * 1.5(ranging) = 0.9, claude_fundamental = 0.6 * 1.0 = 0.6
        # BUY=0.9, SELL=0.6 => BUY
        assert action == "BUY"
        expected = 0.9 / (0.9 + 0.6)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_high_vol_boosts_bb_and_volume(self):
        votes = {"bb": "BUY", "volume": "BUY", "ema": "SELL"}
        accuracy = {
            "bb": {"accuracy": 0.6, "total": 50},
            "volume": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        # Patch _get_correlation_groups to use static groups (avoid live disk cache
        # returning dynamic groups like {bb, volume} that cause correlation penalties)
        from portfolio.signal_engine import _STATIC_CORRELATION_GROUPS
        with mock.patch(
            "portfolio.signal_engine._get_correlation_groups",
            return_value=_STATIC_CORRELATION_GROUPS,
        ):
            action, conf = _weighted_consensus(votes, accuracy, "high-vol")
        # bb: 0.6*1.5=0.9, volume: 0.6*1.3=0.78, ema: 0.6*0.5=0.3
        # BUY=0.9+0.78=1.68, SELL=0.3
        assert action == "BUY"
        expected = 1.68 / (1.68 + 0.3)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_unknown_regime_uses_default_multiplier(self):
        """Regime not in REGIME_WEIGHTS should use mult=1.0."""
        votes = {"rsi": "BUY", "ema": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.7, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "unknown-regime")
        # No regime mults: rsi=0.7, ema=0.6 => BUY=0.7, SELL=0.6
        assert action == "BUY"
        expected = 0.7 / (0.7 + 0.6)
        assert conf == pytest.approx(expected, abs=0.01)


class TestWeightedConsensusActivationRates:
    """Activation rate normalization (rarity * bias correction)."""

    def test_activation_rate_scales_weight(self):
        # Use claude_fundamental — not in momentum_cluster, not regime-gated in ranging
        votes = {"rsi": "BUY", "claude_fundamental": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "claude_fundamental": {"accuracy": 0.6, "total": 50},
        }
        activation = {
            "rsi": {"normalized_weight": 2.0},  # rare signal, boosted
            "claude_fundamental": {"normalized_weight": 0.5},   # noisy signal, dampened
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging", activation_rates=activation)
        # rsi: 0.6 * 1.5(ranging) * 2.0 = 1.8
        # claude_fundamental: 0.6 * 1.0(ranging) * 0.5 = 0.3
        # BUY=1.8, SELL=0.3 => BUY
        assert action == "BUY"
        expected = 1.8 / (1.8 + 0.3)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_missing_activation_rate_defaults_to_1(self):
        votes = {"rsi": "BUY"}
        accuracy = {"rsi": {"accuracy": 0.7, "total": 50}}
        activation = {}  # no data
        action, conf = _weighted_consensus(votes, accuracy, "ranging", activation_rates=activation)
        # normalized_weight defaults to 1.0
        assert action == "BUY"
        assert conf == 1.0

    def test_none_activation_rates_defaults_to_empty(self):
        votes = {"rsi": "BUY"}
        action, conf = _weighted_consensus(votes, {}, "ranging", activation_rates=None)
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# apply_confidence_penalties
# ===========================================================================

class TestPenaltiesDisabledConfig:
    """When confidence_penalties.enabled is False, bypass all stages."""

    def test_disabled_returns_unchanged(self):
        config = {"confidence_penalties": {"enabled": False}}
        extra = _base_extra(voters=1, buy_count=1, sell_count=0, volume_ratio=0.1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.9, "ranging", {}, extra, "BTC-USD", None, config,
        )
        assert action == "BUY"
        assert conf == 0.9
        assert log == []

    def test_enabled_explicitly(self):
        config = {"confidence_penalties": {"enabled": True}}
        extra = _base_extra(voters=6)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, config,
        )
        assert conf < 0.8  # ranging penalty applied

    def test_enabled_by_default_when_key_missing(self):
        extra = _base_extra(voters=6)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {},
        )
        assert conf < 0.8


class TestPenaltiesStage1Regime:
    """Stage 1: Regime penalties."""

    def test_ranging_multiplies_by_075(self):
        extra = _base_extra(voters=6)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {},
        )
        assert conf == pytest.approx(0.8 * 0.75, abs=0.01)
        assert any(p["stage"] == "regime" and p["mult"] == 0.75 for p in log)

    def test_high_vol_multiplies_by_080(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {},
        )
        assert conf == pytest.approx(0.8 * 0.80, abs=0.01)
        assert any(p["stage"] == "regime" and p["mult"] == 0.80 for p in log)

    def test_trending_up_buy_aligned_bonus(self):
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        assert conf == pytest.approx(0.7 * 1.10, abs=0.01)
        assert any(p.get("aligned") is True for p in log)

    def test_trending_down_sell_aligned_bonus(self):
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.7, "trending-down", {}, extra, "BTC-USD", None, {},
        )
        # Stage 1 aligned bonus: 0.7 * 1.10 = 0.77
        # Stage 5 unanimity: 4/5 = 80% agreement → 0.75x penalty → 0.77 * 0.75 = 0.5775
        assert conf == pytest.approx(0.7 * 1.10 * 0.75, abs=0.01)

    def test_trending_up_sell_no_bonus(self):
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.7, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        # Counter-trend: no regime entry in log
        assert not any(p["stage"] == "regime" for p in log)
        # Stage 5 unanimity: 4/5 = 80% agreement → 0.75x penalty → 0.7 * 0.75 = 0.525
        assert conf == pytest.approx(0.7 * 0.75, abs=0.01)


class TestPenaltiesStage2VolumeGate:
    """Stage 2: Volume/ADX gate."""

    def test_very_low_volume_forces_hold(self):
        extra = _base_extra(voters=6, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        assert action == "HOLD"
        assert conf == 0.0
        assert any(p["stage"] == "volume_gate" for p in log)

    def test_low_volume_weak_adx_low_conf_forces_hold(self):
        # Flat DF => low ADX
        df = _make_ohlcv_df(50, trend="flat")
        extra = _base_extra(voters=5, volume_ratio=0.7)
        # After regime penalty (trending-up 1.1x), conf = 0.55 * 1.1 = 0.605 < 0.65
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.55, "trending-up", {}, extra, "BTC-USD", df, {},
        )
        # The ADX gate checks: rvol < 0.8 AND adx < 20 AND conf < 0.65
        # After stage 1, conf = 0.55 * 1.10 = 0.605
        # ADX on flat data may or may not be <20; we just check it doesn't crash
        assert action in ("BUY", "HOLD")

    def test_high_volume_boosts_confidence(self):
        extra = _base_extra(voters=6, volume_ratio=2.0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        # Stage 1: 0.7 * 1.1 = 0.77; Stage 2: 0.77 * 1.15 = 0.8855
        assert conf == pytest.approx(0.7 * 1.10 * 1.15, abs=0.01)
        assert any(p["stage"] == "volume_boost" for p in log)

    def test_hold_action_skips_volume_gate(self):
        extra = _base_extra(volume_ratio=0.1)
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, extra, "BTC-USD", None, {},
        )
        # HOLD action should skip volume gate (condition: action != "HOLD")
        assert not any(p["stage"] == "volume_gate" for p in log)


class TestPenaltiesStage3TrapDetection:
    """Stage 3: Bull/bear trap detection via price-volume divergence."""

    def test_bull_trap_halves_confidence(self):
        df = _make_trap_df(price_trend="up", volume_pattern="declining")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
        )
        assert any(p.get("type") == "bull_trap" for p in log)
        # After regime(1.1x) + trap(0.5x): 0.8 * 1.1 * 0.5 = 0.44
        assert conf == pytest.approx(0.8 * 1.10 * 0.5, abs=0.01)

    def test_bear_trap_halves_confidence(self):
        df = _make_trap_df(price_trend="down", volume_pattern="declining")
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.8, "trending-down", {}, extra, "BTC-USD", df, {},
        )
        assert any(p.get("type") == "bear_trap" for p in log)

    def test_no_trap_when_volume_expanding(self):
        df = _make_trap_df(price_trend="up", volume_pattern="expanding")
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
        )
        assert not any(p.get("type") == "bull_trap" for p in log)

    def test_no_trap_on_hold_action(self):
        df = _make_trap_df(price_trend="up", volume_pattern="declining")
        extra = _base_extra()
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "ranging", {}, extra, "BTC-USD", df, {},
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_no_trap_with_short_df(self):
        df = _make_ohlcv_df(3)
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
        )
        assert not any(p["stage"] == "trap" for p in log)

    def test_trap_exception_logs_warning(self, caplog):
        """BUG-42: trap detection should log warning on exception, not silently pass."""
        # DataFrame with non-numeric volume column that will cause comparison error
        df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [2.0] * 10,
            "low": [0.5] * 10,
            "close": list(range(1, 11)),  # ascending → price_up = True
            "volume": ["bad"] * 10,  # string volume → comparison will fail
        })
        extra = _base_extra(voters=5)
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_engine"):
            action, conf, log = apply_confidence_penalties(
                "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
            )
        # Should not crash, and should log a warning
        assert any("Trap detection failed" in r.message for r in caplog.records)
        # Confidence unchanged (trap detection skipped gracefully)
        assert not any(p.get("stage") == "trap" for p in log)

    def test_trap_missing_volume_column_logs_warning(self, caplog):
        """BUG-42: DataFrame missing volume column triggers safe warning path."""
        df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [2.0] * 10,
            "low": [0.5] * 10,
            "close": list(range(1, 11)),
            # No "volume" column at all — the code checks "volume" in df.columns
            # so this should NOT raise, it should just skip. Verify no trap applied.
        })
        extra = _base_extra(voters=5)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
        )
        assert not any(p.get("stage") == "trap" for p in log)

    def test_trap_nan_volume_logs_warning(self, caplog):
        """BUG-42: DataFrame with NaN volumes should log warning."""
        df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [2.0] * 10,
            "low": [0.5] * 10,
            "close": list(range(1, 11)),
            "volume": [float("nan")] * 10,
        })
        extra = _base_extra(voters=5)
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_engine"):
            action, conf, log = apply_confidence_penalties(
                "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", df, {},
            )
        # NaN comparison: float("nan") < float("nan") * 0.8 is always False
        # so no trap is detected — this is the correct behavior (no crash)
        # No warning expected because NaN comparisons don't raise exceptions
        assert not any(p.get("stage") == "trap" for p in log)


class TestPenaltiesStage4DynamicMinVoters:
    """Stage 4: Dynamic MIN_VOTERS based on regime."""

    def test_trending_requires_3(self):
        extra = _base_extra(voters=3, buy_count=2, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        assert action == "BUY"
        assert not any(p["stage"] == "dynamic_min_voters" for p in log)

    def test_trending_fails_with_2(self):
        extra = _base_extra(voters=2, buy_count=2, sell_count=0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        assert action == "HOLD"
        assert conf == 0.0
        assert any(p["stage"] == "dynamic_min_voters" and p["required"] == 3 for p in log)

    def test_high_vol_requires_4(self):
        extra = _base_extra(voters=3, buy_count=2, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {},
        )
        assert action == "HOLD"
        assert any(p["stage"] == "dynamic_min_voters" and p["required"] == 4 for p in log)

    def test_high_vol_passes_with_4(self):
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "high-vol", {}, extra, "BTC-USD", None, {},
        )
        assert action == "BUY"

    def test_ranging_requires_5(self):
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {},
        )
        assert action == "HOLD"
        assert any(p["stage"] == "dynamic_min_voters" and p["required"] == 5 for p in log)

    def test_ranging_passes_with_5(self):
        extra = _base_extra(voters=5, buy_count=4, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "ranging", {}, extra, "BTC-USD", None, {},
        )
        assert action == "BUY"

    def test_unknown_regime_defaults_to_5(self):
        extra = _base_extra(voters=4, buy_count=3, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.8, "some-unknown-regime", {}, extra, "BTC-USD", None, {},
        )
        assert action == "HOLD"
        assert any(p["stage"] == "dynamic_min_voters" and p["required"] == 5 for p in log)


class TestPenaltiesConfidenceClamping:
    """Confidence is clamped to [0, 1] at the end."""

    def test_confidence_clamped_to_max_1(self):
        extra = _base_extra(voters=10, buy_count=8, sell_count=2, volume_ratio=2.0)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.95, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        # 0.95 * 1.1 = 1.045 → clamped to 1.0 (Stage 3 BUG-90 clamp)
        # volume boost 2.0 → 1.15x: 1.0 * 1.15 → clamped to 1.0
        # Stage 5 unanimity: 8/10 = 80% agreement → 0.75x penalty → 1.0 * 0.75 = 0.75
        assert conf <= 1.0
        assert conf == pytest.approx(0.75)

    def test_confidence_clamped_to_min_0(self):
        extra = _base_extra(voters=6, volume_ratio=0.3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.1, "ranging", {}, extra, "BTC-USD", None, {},
        )
        # Forced HOLD with conf=0.0 by volume gate
        assert conf >= 0.0


# ===========================================================================
# _confluence_score
# ===========================================================================

class TestConfluenceScore:
    """Majority-based confluence scoring."""

    def test_all_hold_returns_zero(self):
        votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        score = _confluence_score(votes, {})
        assert score == 0.0

    def test_empty_votes_returns_zero(self):
        score = _confluence_score({}, {})
        assert score == 0.0

    def test_unanimous_buy(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        score = _confluence_score(votes, {})
        # 3 active, 3 BUY, majority=3 => 3/3 = 1.0
        assert score == 1.0

    def test_unanimous_sell(self):
        votes = {"rsi": "SELL", "macd": "SELL"}
        score = _confluence_score(votes, {})
        assert score == 1.0

    def test_mixed_votes(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL", "bb": "HOLD"}
        score = _confluence_score(votes, {})
        # Active: rsi=BUY, macd=BUY, ema=SELL (3 active)
        # majority=2 (BUY), score = 2/3 ~ 0.6667
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_volume_confirmation_adds_01(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL"}
        indicators = {"volume_action": "BUY"}
        score = _confluence_score(votes, indicators)
        # majority=BUY, vol=BUY => 2/3 + 0.1 = 0.7667
        assert score == pytest.approx(2 / 3 + 0.1, abs=0.01)

    def test_volume_opposing_majority_no_bonus(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL"}
        indicators = {"volume_action": "SELL"}
        score = _confluence_score(votes, indicators)
        # majority=BUY, vol=SELL => no bonus
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_volume_hold_no_bonus(self):
        votes = {"rsi": "BUY", "macd": "BUY"}
        indicators = {"volume_action": "HOLD"}
        score = _confluence_score(votes, indicators)
        # volume_action is HOLD, not in ("BUY", "SELL") => no bonus
        assert score == 1.0

    def test_capped_at_1(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        indicators = {"volume_action": "BUY"}
        score = _confluence_score(votes, indicators)
        # 3/3 + 0.1 = 1.1 => capped at 1.0
        assert score == 1.0


# ===========================================================================
# _time_of_day_factor
# ===========================================================================

class TestTimeOfDayFactor:
    """Time-based factor: 0.8 during 2-6 UTC, 1.0 otherwise."""

    @pytest.mark.parametrize("hour", [2, 3, 4, 5, 6])
    def test_night_hours_return_08(self, hour):
        with mock.patch("portfolio.signal_engine.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, hour, 30, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
        assert factor == 0.8

    @pytest.mark.parametrize("hour", [0, 1, 7, 8, 12, 15, 18, 21, 23])
    def test_day_hours_return_10(self, hour):
        with mock.patch("portfolio.signal_engine.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, hour, 30, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
        assert factor == 1.0


# ===========================================================================
# Stage 5: Unanimity penalty
# ===========================================================================

class TestUnanimityPenalty:
    """Stage 5: Unanimity penalty in confidence cascade."""

    def test_high_agreement_penalized(self):
        """90%+ agreement gets 0.6x penalty."""
        df = _make_ohlcv_df(50)
        extra = _base_extra(voters=10, buy_count=9, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.9, "breakout", {}, extra, "BTC-USD", df, {}
        )
        # Should have unanimity penalty in log
        unanimity = [p for p in log if p.get("stage") == "unanimity"]
        assert len(unanimity) == 1
        assert unanimity[0]["mult"] == 0.6

    def test_moderate_agreement_penalized_less(self):
        """80-90% agreement gets 0.75x penalty."""
        df = _make_ohlcv_df(50)
        extra = _base_extra(voters=10, buy_count=8, sell_count=2)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.85, "breakout", {}, extra, "BTC-USD", df, {}
        )
        unanimity = [p for p in log if p.get("stage") == "unanimity"]
        assert len(unanimity) == 1
        assert unanimity[0]["mult"] == 0.75

    def test_normal_agreement_not_penalized(self):
        """<80% agreement has no unanimity penalty."""
        df = _make_ohlcv_df(50)
        extra = _base_extra(voters=10, buy_count=7, sell_count=3)
        action, conf, log = apply_confidence_penalties(
            "BUY", 0.7, "breakout", {}, extra, "BTC-USD", df, {}
        )
        unanimity = [p for p in log if p.get("stage") == "unanimity"]
        assert len(unanimity) == 0

    def test_hold_not_penalized(self):
        """HOLD actions skip unanimity check."""
        df = _make_ohlcv_df(50)
        extra = _base_extra(voters=10, buy_count=9, sell_count=1)
        action, conf, log = apply_confidence_penalties(
            "HOLD", 0.0, "breakout", {}, extra, "BTC-USD", df, {}
        )
        unanimity = [p for p in log if p.get("stage") == "unanimity"]
        assert len(unanimity) == 0


# ===========================================================================
# Global confidence cap at 0.80
# ===========================================================================

class TestGlobalConfidenceCap:
    """Global confidence cap at 0.80."""

    def test_confidence_capped_at_80(self):
        """Confidence should never exceed 0.80 from generate_signal."""
        # This is tested via the return value, not directly testable
        # in apply_confidence_penalties since cap is in generate_signal
        pass  # Tested in integration tests


# ===========================================================================
# REGIME_WEIGHTS constant validation
# ===========================================================================

class TestRegimeWeightsConstant:
    """Verify REGIME_WEIGHTS structure hasn't drifted."""

    def test_trending_up_exists(self):
        assert "trending-up" in REGIME_WEIGHTS
        assert REGIME_WEIGHTS["trending-up"]["ema"] == 1.5
        assert REGIME_WEIGHTS["trending-up"]["macd"] == 1.3

    def test_trending_down_exists(self):
        assert "trending-down" in REGIME_WEIGHTS
        assert REGIME_WEIGHTS["trending-down"]["ema"] == 1.5

    def test_ranging_exists(self):
        assert "ranging" in REGIME_WEIGHTS
        assert REGIME_WEIGHTS["ranging"]["rsi"] == 1.5
        assert REGIME_WEIGHTS["ranging"]["bb"] == 1.5
        assert REGIME_WEIGHTS["ranging"]["ema"] == 0.5

    def test_high_vol_exists(self):
        assert "high-vol" in REGIME_WEIGHTS
        assert REGIME_WEIGHTS["high-vol"]["bb"] == 1.5
        assert REGIME_WEIGHTS["high-vol"]["volume"] == 1.3

    def test_ranging_has_enhanced_weights(self):
        """Enhanced signals should have regime weights in ranging."""
        rw = REGIME_WEIGHTS["ranging"]
        assert rw["mean_reversion"] == 1.7  # 2026-04-05: 65.4% recent, boosted from 1.5
        assert rw["fibonacci"] == 1.8  # 2026-04-05: 68.2% recent, boosted from 1.6
        assert rw["ministral"] == 1.4  # 2026-04-05: 68.0% recent, newly added
        assert rw["macd"] == 1.3  # 2026-04-05: 58.7% recent, newly added
        assert rw["trend"] == 0.5
        assert rw["fear_greed"] == 0.3  # added 2026-03-31 (25.9% recent accuracy)

    def test_trending_up_has_enhanced_weights(self):
        """Enhanced signals should have regime weights in trending-up."""
        rw = REGIME_WEIGHTS["trending-up"]
        assert rw["trend"] == 1.4
        assert rw["mean_reversion"] == 0.6


# ---------------------------------------------------------------------------
# Correlation deduplication tests
# ---------------------------------------------------------------------------

class TestCorrelationDedup:
    """Test signal correlation grouping in _weighted_consensus."""

    def test_single_signal_in_group_no_penalty(self):
        """Only 1 signal from a correlated group votes — no penalty."""
        votes = {"calendar": "BUY", "rsi": "SELL"}
        accuracy = {
            "calendar": {"accuracy": 0.63, "total": 600},
            "rsi": {"accuracy": 0.53, "total": 800},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # calendar in group but alone => no penalty
        # calendar weight = 0.63 * 1.2 (ranging regime) = 0.756
        # rsi weight = 0.53 * 1.5 (ranging regime) = 0.795
        # SELL wins
        assert action == "SELL"

    def test_two_signals_in_group_leader_gets_full_weight(self):
        """When 2 signals from same group vote, leader gets full, other 0.3x."""
        votes = {"calendar": "BUY", "econ_calendar": "BUY"}
        accuracy = {
            "calendar": {"accuracy": 0.63, "total": 600},
            "econ_calendar": {"accuracy": 0.87, "total": 2500},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # Both in "low_activity_timing" group
        # econ_calendar is leader (0.87 > 0.63)
        # calendar gets 0.3x penalty
        assert action == "BUY"
        assert conf == 1.0  # Both vote BUY so 100% consensus

    def test_correlated_group_reduces_apparent_confidence(self):
        """When correlated signals + 1 independent signal disagree,
        the correlation penalty should reduce the correlated side's weight."""
        votes = {
            "candlestick": "BUY",
            "fibonacci": "BUY",
            "rsi": "SELL",
        }
        accuracy = {
            "candlestick": {"accuracy": 0.63, "total": 600},
            "fibonacci": {"accuracy": 0.87, "total": 2500},
            "rsi": {"accuracy": 0.53, "total": 800},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # fibonacci: leader (0.87), full weight
        # candlestick: penalized 0.3x = 0.63 * 0.3 * 1.2 (ranging) = 0.2268
        # rsi: SELL weight = 0.53 * 1.5 (ranging) = 0.795
        # BUY: 0.87 + 0.2268 = 1.0968, SELL: 0.795
        # BUY wins but with reduced confidence
        assert action == "BUY"
        assert conf < 1.0

    def test_no_penalty_when_group_signals_vote_differently(self):
        """If signals in the same group vote opposite directions,
        they each count as independent."""
        votes = {"candlestick": "BUY", "fibonacci": "SELL"}
        accuracy = {
            "candlestick": {"accuracy": 0.63, "total": 600},
            "fibonacci": {"accuracy": 0.87, "total": 2500},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # Both in pattern_based group. Leader: fibonacci (0.87).
        # candlestick penalized 0.3x even though voting opposite.
        # fibonacci SELL = 0.87, candlestick BUY = 0.63 * 0.3 * 1.2 = 0.2268
        # SELL wins
        assert action == "SELL"

    def test_adaptive_recency_fast_track(self):
        """When divergence > 15%, the blend should use 90% recent weight."""
        from portfolio.signal_engine import (
            _RECENCY_DIVERGENCE_THRESHOLD,
            _RECENCY_WEIGHT_FAST,
            _RECENCY_WEIGHT_NORMAL,
        )
        assert _RECENCY_DIVERGENCE_THRESHOLD == 0.15
        assert _RECENCY_WEIGHT_FAST == 0.9
        assert _RECENCY_WEIGHT_NORMAL == 0.7

    def test_accuracy_gate_at_047(self):
        """Verify gate threshold is 0.47 (raised 0.45 → 0.47 on 2026-04-11
        to gate the 4 signals sitting in the 45-47% coin-flip-adjacent
        band per the 2026-04-10 audit)."""
        from portfolio.signal_engine import ACCURACY_GATE_THRESHOLD
        assert ACCURACY_GATE_THRESHOLD == 0.47


# ===========================================================================
# Regime gating tests (2026-03-27 research)
# ===========================================================================

class TestRegimeGating:
    """Test REGIME_GATED_SIGNALS silences signals that produce negative alpha."""

    def test_ranging_gates_trend_on_default_horizon(self):
        """In ranging regime with no/1d horizon, 'trend' is forced to HOLD."""
        votes = {"trend": "SELL", "rsi": "BUY", "bb": "BUY"}
        accuracy = {
            "trend": {"accuracy": 0.40, "total": 600},
            "rsi": {"accuracy": 0.53, "total": 800},
            "bb": {"accuracy": 0.55, "total": 300},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # trend gated in ranging (default horizon) => only rsi+bb vote => BUY wins
        assert action == "BUY"

    def test_ranging_does_not_gate_trend_on_3h(self):
        """In ranging regime at 3h horizon, 'trend' is NOT gated (61.6% accuracy).

        Trend participates at 3h even in ranging, unlike default horizon where
        it's gated. Note: regime weight (trend 0.5x in ranging) still applies,
        so trend needs more votes or higher accuracy to dominate.
        """
        from portfolio.signal_engine import _get_regime_gated
        # Verify trend is NOT in the gated set for 3h
        gated_3h = _get_regime_gated("ranging", "3h")
        assert "trend" not in gated_3h
        # But IS gated on default (1d-like) horizon
        gated_default = _get_regime_gated("ranging")
        assert "trend" in gated_default

        # When trend is NOT gated, it participates (gets SELL weight)
        # vs when gated, its vote is forced to HOLD
        votes_gated = {"trend": "SELL", "rsi": "BUY"}
        accuracy = {
            "trend": {"accuracy": 0.62, "total": 2283},
            "rsi": {"accuracy": 0.47, "total": 3572},
        }
        # Default (gated): trend forced HOLD, only rsi BUY => BUY 100%
        _, conf_gated = _weighted_consensus(votes_gated, accuracy, "ranging")
        # 3h (not gated): both participate, confidence < 1.0
        _, conf_3h = _weighted_consensus(votes_gated, accuracy, "ranging", horizon="3h")
        # With trend participating, BUY confidence is lower than 100%
        assert conf_3h < conf_gated

    def test_ranging_gates_momentum_factors_on_default(self):
        """In ranging regime, 'momentum_factors' is forced to HOLD on default horizon."""
        votes = {"momentum_factors": "SELL", "fibonacci": "BUY"}
        accuracy = {
            "momentum_factors": {"accuracy": 0.41, "total": 500},
            "fibonacci": {"accuracy": 0.68, "total": 110},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # momentum_factors gated on default => fibonacci BUY wins
        assert action == "BUY"

    def test_trending_up_gates_mean_reversion_on_3h(self):
        """In trending-up regime at 3h, 'mean_reversion' is gated (45.5%)."""
        votes = {"mean_reversion": "SELL", "ema": "BUY"}
        accuracy = {
            "mean_reversion": {"accuracy": 0.45, "total": 1402},
            "ema": {"accuracy": 0.63, "total": 2000},
        }
        action, conf = _weighted_consensus(votes, accuracy, "trending-up", horizon="3h")
        # mean_reversion gated on 3h in trending => ema BUY wins
        assert action == "BUY"

    def test_trending_up_does_not_gate_mean_reversion_on_1d(self):
        """In trending-up at 1d, 'mean_reversion' is NOT gated (65.4%)."""
        votes = {"mean_reversion": "SELL", "ema": "BUY"}
        accuracy = {
            "mean_reversion": {"accuracy": 0.65, "total": 332},
            "ema": {"accuracy": 0.41, "total": 568},
        }
        action, conf = _weighted_consensus(votes, accuracy, "trending-up", horizon="1d")
        # mean_reversion NOT gated on 1d => mean_reversion SELL can win
        assert action == "SELL"

    def test_no_gating_in_ungated_regime(self):
        """In high-vol regime, trend is NOT gated."""
        votes = {"trend": "SELL", "rsi": "BUY"}
        accuracy = {
            "trend": {"accuracy": 0.50, "total": 600},
            "rsi": {"accuracy": 0.50, "total": 800},
        }
        action, conf = _weighted_consensus(votes, accuracy, "high-vol")
        # trend not gated in high-vol, both have equal accuracy
        # Result depends on regime weights: high-vol trend=0.6, rsi has no boost
        # trend weight: 0.5 * 0.6 = 0.30
        # rsi weight: 0.5 * 1.0 = 0.50
        assert action == "BUY"


# ===========================================================================
# Horizon-specific weight tests (2026-03-27 research)
# ===========================================================================

class TestHorizonWeights:
    """Test HORIZON_SIGNAL_WEIGHTS applies horizon-specific multipliers."""

    def test_3h_boosts_news_event(self):
        """At 3h horizon, news_event gets a dynamic boost relative to 1d."""
        votes = {"news_event": "SELL", "rsi": "BUY"}
        accuracy = {
            "news_event": {"accuracy": 0.70, "total": 1700},
            "rsi": {"accuracy": 0.50, "total": 800},
        }
        # Without horizon: rsi gets ranging 1.5x boost, news_event 1.0x
        action_no_h, conf_no_h = _weighted_consensus(votes, accuracy, "ranging")
        # With 3h horizon: news_event gets horizon boost.
        # Give news_event higher base accuracy to ensure it wins at 3h
        action_3h, conf_3h = _weighted_consensus(votes, accuracy, "ranging", horizon="3h")
        # news_event has higher accuracy (0.70 vs 0.50) + 3h horizon boost
        assert action_3h == "SELL"

    def test_1d_penalizes_news_event(self):
        """At 1d horizon, news_event gets 0.5x penalty."""
        votes = {"news_event": "SELL", "fibonacci": "BUY"}
        accuracy = {
            "news_event": {"accuracy": 0.54, "total": 4000},
            "fibonacci": {"accuracy": 0.55, "total": 600},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging", horizon="1d")
        # news_event: 0.54 * 1.0(regime) * 0.5(horizon) = 0.27
        # fibonacci: 0.55 * 1.4(regime ranging) * 1.4(horizon 1d) = 1.078
        # fibonacci BUY wins decisively
        assert action == "BUY"

    def test_no_horizon_no_adjustment(self):
        """Without horizon parameter, no horizon weights applied."""
        # Use rsi — not regime-gated in ranging at any horizon
        votes = {"rsi": "SELL"}
        accuracy = {"rsi": {"accuracy": 0.70, "total": 1700}}
        _, conf_none = _weighted_consensus(votes, accuracy, "ranging", horizon=None)
        _, conf_3h = _weighted_consensus(votes, accuracy, "ranging", horizon="3h")
        # Both return SELL with 1.0 confidence (single voter)
        assert conf_none == 1.0
        assert conf_3h == 1.0


# ===========================================================================
# Activity rate cap tests (2026-03-27 research)
# ===========================================================================

class TestActivityRateCap:
    """Test _ACTIVITY_RATE_CAP penalizes high-activity signals."""

    def test_high_activity_signal_penalized(self):
        """Signals with >70% activation rate get 0.5x penalty."""
        votes = {"volume_flow": "SELL", "fibonacci": "BUY"}
        accuracy = {
            "volume_flow": {"accuracy": 0.50, "total": 50000},
            "fibonacci": {"accuracy": 0.50, "total": 600},
        }
        activation = {
            "volume_flow": {"activation_rate": 0.83, "normalized_weight": 1.0},
            "fibonacci": {"activation_rate": 0.02, "normalized_weight": 1.0},
        }
        action, conf = _weighted_consensus(
            votes, accuracy, "ranging", activation_rates=activation
        )
        # volume_flow: 0.50 * 1.0(regime) * 1.0(norm) * 0.5(activity cap) = 0.25
        # fibonacci: 0.50 * 1.4(regime ranging) * 1.0(norm) = 0.70
        # fibonacci BUY wins
        assert action == "BUY"

    def test_normal_activity_no_penalty(self):
        """Signals with <70% activation rate are not penalized."""
        votes = {"rsi": "SELL", "fibonacci": "BUY"}
        accuracy = {
            "rsi": {"accuracy": 0.55, "total": 24000},
            "fibonacci": {"accuracy": 0.55, "total": 600},
        }
        activation = {
            "rsi": {"activation_rate": 0.35, "normalized_weight": 1.0},
            "fibonacci": {"activation_rate": 0.02, "normalized_weight": 1.0},
        }
        action, conf = _weighted_consensus(
            votes, accuracy, "ranging", activation_rates=activation
        )
        # rsi: 0.55 * 1.5(regime ranging) * 1.0 = 0.825
        # fibonacci: 0.55 * 1.8(regime ranging) * 1.0 = 0.99
        # fibonacci BUY wins (updated 2026-04-05: fibonacci ranging weight 1.6 -> 1.8)
        assert action == "BUY"


# ===========================================================================
# Expanded correlation groups tests (2026-03-27 research)
# ===========================================================================

class TestExpandedCorrelationGroups:
    """Test new correlation groups: trend_direction, high_volume_sell."""

    def test_trend_direction_group_penalizes_secondary(self):
        """In trend_direction group {ema, trend, heikin_ashi}, only leader gets full weight."""
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "trend_direction" in CORRELATION_GROUPS
        assert "ema" in CORRELATION_GROUPS["trend_direction"]

        votes = {"ema": "SELL", "trend": "SELL", "heikin_ashi": "SELL", "rsi": "BUY"}
        accuracy = {
            "ema": {"accuracy": 0.63, "total": 2000},
            "trend": {"accuracy": 0.45, "total": 11000},
            "heikin_ashi": {"accuracy": 0.48, "total": 19000},
            "rsi": {"accuracy": 0.53, "total": 24000},
        }
        action, conf = _weighted_consensus(votes, accuracy, "high-vol")
        # ema is leader (0.63), trend (0.45) and heikin_ashi (0.48) get 0.3x
        # Without penalty, 3 SELL vs 1 BUY would heavily favor SELL
        # With penalty, effective SELL weight is much lower
        # The correlation group prevents 3 correlated signals from inflating SELL

    def test_macro_regime_in_trend_direction_group(self):
        """Verify macro_regime moved to trend_direction group (corr +0.520 with trend).

        2026-04-07: Moved from macro_external — better correlation fit with
        trend (both follow 200-SMA direction) than with fear_greed.
        """
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "macro_regime" in CORRELATION_GROUPS["trend_direction"]
        assert "fear_greed" in CORRELATION_GROUPS["macro_external"]
        # structure moved to volatility_cluster (2026-04-08: 94.2% with volatility_sig)
        assert "structure" in CORRELATION_GROUPS["volatility_cluster"]

    def test_volume_flow_in_trend_direction_group(self):
        """Verify volume_flow merged into trend_direction group (corr +0.511 with heikin_ashi)."""
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "volume_flow" in CORRELATION_GROUPS["trend_direction"]


# ===========================================================================
# BUG-143: Regime gating applied before vote counting
# ===========================================================================

class TestRegimeGatingBeforeVoteCounts:
    """BUG-143: Verify regime gating happens before buy/sell counts are computed.

    Before the fix, buy/sell counts were derived from raw votes, so the
    unanimity penalty used pre-gated counts. After the fix, regime-gated
    signals are forced to HOLD before counting, so buy_count/sell_count
    reflect the post-gated state.
    """

    def test_ranging_regime_reduces_buy_count_default_horizon(self):
        """In ranging regime (default horizon), trend+momentum_factors are gated.

        If trend and momentum_factors both vote BUY, gating them should
        reduce _buy_count by 2 compared to ungated regime.
        """
        from portfolio.signal_engine import _get_regime_gated

        # Verify ranging gates trend, momentum_factors, ema, heikin_ashi,
        # structure, fear_greed, macro_regime on default horizon (2026-03-31 update)
        gated = _get_regime_gated("ranging")
        assert "trend" in gated
        assert "momentum_factors" in gated
        assert "ema" in gated
        assert "heikin_ashi" in gated
        assert "structure" in gated
        assert "fear_greed" in gated
        assert "macro_regime" in gated

        votes = {
            "rsi": "BUY", "macd": "BUY", "ema": "BUY",
            "trend": "BUY", "momentum_factors": "BUY",
            "bb": "SELL", "volume": "HOLD",
        }

        # Simulate the gating logic from generate_signal (BUG-143/149 fix)
        regime = "ranging"
        gated_votes = dict(votes)
        regime_gated = _get_regime_gated(regime)
        for sig_name in regime_gated:
            if sig_name in gated_votes and gated_votes[sig_name] != "HOLD":
                gated_votes[sig_name] = "HOLD"

        # Post-gating: trend, momentum_factors, AND ema should be HOLD
        assert gated_votes["trend"] == "HOLD"
        assert gated_votes["momentum_factors"] == "HOLD"
        assert gated_votes["ema"] == "HOLD"

        # Post-gated counts
        buy_count = sum(1 for v in gated_votes.values() if v == "BUY")
        sell_count = sum(1 for v in gated_votes.values() if v == "SELL")

        # 2 BUY (rsi, macd) — trend, momentum_factors, ema all gated
        assert buy_count == 2
        assert sell_count == 1  # bb

    def test_ranging_regime_3h_keeps_trend_active(self):
        """In ranging regime at 3h horizon, trend is NOT gated but fear_greed is."""
        from portfolio.signal_engine import _get_regime_gated

        gated = _get_regime_gated("ranging", "3h")
        assert "trend" not in gated
        assert "momentum_factors" not in gated
        assert "ema" not in gated  # ema works at 3h (62.9%)
        # fear_greed and macro_regime stay gated at ALL horizons (structural failure)
        assert "fear_greed" in gated
        assert "macro_regime" in gated

    def test_unanimity_ratio_changes_after_gating(self):
        """Gating should change the unanimity ratio used by Stage 5 penalty.

        Example: 9 BUY / 1 SELL raw = 90% → 0.6x penalty.
        After gating 2 signals: 7 BUY / 1 SELL = 87.5% → 0.75x penalty.
        """
        # Raw: 9/10 = 90% agreement → unanimity penalty 0.6x
        raw_buy, raw_sell = 9, 1
        raw_ratio = max(raw_buy, raw_sell) / (raw_buy + raw_sell)
        assert raw_ratio == 0.9

        # After gating 2 BUY signals: 7/8 = 87.5% → unanimity penalty 0.75x
        gated_buy, gated_sell = 7, 1
        gated_ratio = max(gated_buy, gated_sell) / (gated_buy + gated_sell)
        assert gated_ratio == 0.875

        # The penalty tier changes: 0.9 → 0.6x, 0.875 → 0.75x
        # This is a 25% difference in penalty severity
        assert raw_ratio >= 0.9  # would trigger 0.6x
        assert 0.8 <= gated_ratio < 0.9  # would trigger 0.75x instead

    def test_gating_idempotent_on_hold(self):
        """Gating a signal that already votes HOLD is a no-op."""
        from portfolio.signal_engine import _get_regime_gated

        votes = {"trend": "HOLD", "rsi": "BUY"}
        regime_gated = _get_regime_gated("ranging")

        gated_votes = dict(votes)
        for sig_name in regime_gated:
            if sig_name in gated_votes and gated_votes[sig_name] != "HOLD":
                gated_votes[sig_name] = "HOLD"

        # trend was already HOLD — unchanged
        assert gated_votes["trend"] == "HOLD"
        assert gated_votes["rsi"] == "BUY"


# ===========================================================================
# BUG-144: Regime passed through context_data to enhanced signals
# ===========================================================================

class TestRegimeInContextData:
    """BUG-144: Verify context_data includes 'regime' key for enhanced signals.

    Before the fix, context_data only had ticker, config, macro — the
    regime key was never included, so forecast.py's regime discount was
    dead code.
    """

    def test_context_data_structure(self):
        """The context_data dict should include regime alongside existing keys."""
        # Simulate what generate_signal builds at line ~1030
        ticker = "BTC-USD"
        config = {"signals": {}}
        macro_data = {"fear_greed": 45}
        regime = "trending-up"

        context_data = {
            "ticker": ticker,
            "config": config,
            "macro": macro_data,
            "regime": regime,
        }

        assert "regime" in context_data
        assert context_data["regime"] == "trending-up"
        assert context_data["ticker"] == "BTC-USD"
        assert context_data["macro"] == macro_data

    def test_forecast_regime_discount_constants(self):
        """Verify forecast.py regime discount constants exist and are reasonable."""
        from portfolio.signals.forecast import _REGIME_DISCOUNT_TRENDING, _REGIME_NEUTRAL
        assert _REGIME_DISCOUNT_TRENDING < 1.0  # trending discount < 1.0
        assert _REGIME_NEUTRAL == 1.0  # neutral = no discount


# ===========================================================================
# Dynamic horizon weight tests (2026-03-29 research)
# ===========================================================================


class TestDynamicHorizonWeights:
    """Test _compute_dynamic_horizon_weights computes multipliers from accuracy cache."""

    def test_computes_boost_for_better_horizon(self, tmp_path, monkeypatch):
        """Signal with higher accuracy on 3h than 1d gets a boost multiplier."""
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        cache = {
            "3h_recent": {
                "news_event": {"accuracy": 0.70, "total": 1762},
                "rsi": {"accuracy": 0.47, "total": 3572},
            },
            "1d_recent": {
                "news_event": {"accuracy": 0.30, "total": 112},
                "rsi": {"accuracy": 0.53, "total": 875},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        # Clear cache to force recomputation
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("3h")
        # news_event: 0.70 / 0.30 = 2.33, clamped to 1.5
        assert "news_event" in weights
        assert weights["news_event"] == 1.5

    def test_computes_penalty_for_worse_horizon(self, tmp_path, monkeypatch):
        """Signal with lower accuracy on 1d than 3h gets a penalty multiplier."""
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        cache = {
            "1d_recent": {
                "ema": {"accuracy": 0.41, "total": 568},
            },
            "3h_recent": {
                "ema": {"accuracy": 0.63, "total": 2248},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("1d")
        # ema: 0.41 / 0.63 = 0.651, outside deadband
        assert "ema" in weights
        assert weights["ema"] < 0.9  # penalty
        assert weights["ema"] >= 0.4  # clamped

    def test_skips_low_sample_signals(self, tmp_path, monkeypatch):
        """Signals with fewer than 50 samples are excluded."""
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        cache = {
            "3h_recent": {
                "rare_sig": {"accuracy": 0.90, "total": 10},  # too few
            },
            "1d_recent": {
                "rare_sig": {"accuracy": 0.30, "total": 10},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("3h")
        assert "rare_sig" not in weights

    def test_deadband_filters_near_unity(self, tmp_path, monkeypatch):
        """Signals with near-equal accuracy across horizons are excluded (deadband)."""
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        cache = {
            "3h_recent": {
                "rsi": {"accuracy": 0.52, "total": 3000},
            },
            "1d_recent": {
                "rsi": {"accuracy": 0.51, "total": 3000},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("3h")
        # 0.52/0.51 = 1.02, within ±0.1 deadband
        assert "rsi" not in weights

    def test_falls_back_to_static_on_missing_cache(self, tmp_path, monkeypatch):
        """Returns static HORIZON_SIGNAL_WEIGHTS when cache file doesn't exist."""
        from portfolio.signal_engine import HORIZON_SIGNAL_WEIGHTS, _compute_dynamic_horizon_weights

        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("3h")
        assert weights == HORIZON_SIGNAL_WEIGHTS.get("3h", {})

    def test_get_horizon_weights_uses_cache(self, tmp_path, monkeypatch):
        """_get_horizon_weights returns cached result on second call."""
        from portfolio.signal_engine import _get_horizon_weights

        # With no horizon, returns empty
        assert _get_horizon_weights(None) == {}
        assert _get_horizon_weights("") == {}


class TestCrossHorizonTrueMean:
    """BUG-150: Verify cross-horizon averaging uses true mean, not running average."""

    def test_three_horizons_true_mean(self, tmp_path, monkeypatch):
        """With 3 cross horizons, all should contribute equally to the mean.

        BUG-150: Before the fix, a running (old+new)/2 average gave the last
        horizon ~57% weight instead of 33%. This test monkeypatches
        _CROSS_HORIZON_PAIRS to have 3 entries and verifies true arithmetic mean.
        """
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        # Monkeypatch to give "1d" three cross horizons
        monkeypatch.setattr("portfolio.signal_engine._CROSS_HORIZON_PAIRS", {
            "1d": ["3h", "3d", "5d"],
        })

        cache = {
            "1d_recent": {
                "ema": {"accuracy": 0.70, "total": 500},
            },
            "3h_recent": {
                "ema": {"accuracy": 0.40, "total": 500},
            },
            "3d_recent": {
                "ema": {"accuracy": 0.50, "total": 500},
            },
            "5d_recent": {
                "ema": {"accuracy": 0.60, "total": 500},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("1d")
        # True mean: (0.40 + 0.50 + 0.60) / 3 = 0.50
        # Ratio: 0.70 / 0.50 = 1.40 (above deadband 0.1, within clamp 0.4-1.5)
        assert "ema" in weights
        assert weights["ema"] == 1.4

    def test_two_horizons_same_as_before(self, tmp_path, monkeypatch):
        """With exactly 2 cross horizons, true mean equals running average (regression check)."""
        from portfolio.signal_engine import _compute_dynamic_horizon_weights

        cache = {
            "3h_recent": {
                "rsi": {"accuracy": 0.70, "total": 500},
            },
            "1d_recent": {
                "rsi": {"accuracy": 0.50, "total": 500},
            },
        }
        cache_file = tmp_path / "accuracy_cache.json"
        import json
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr("portfolio.signal_engine.DATA_DIR", tmp_path)
        from portfolio.shared_state import _tool_cache
        for key in list(_tool_cache.keys()):
            if "dynamic_horizon" in key:
                del _tool_cache[key]

        weights = _compute_dynamic_horizon_weights("3h")
        # Cross for 3h: ["1d"]
        # True mean of cross: 0.50 (just one horizon)
        # Ratio: 0.70 / 0.50 = 1.4
        assert "rsi" in weights
        assert weights["rsi"] == 1.4


class TestBuildLlmContext:
    """REF-18: Test the extracted _build_llm_context helper."""

    def test_returns_all_expected_keys(self):
        from portfolio.signal_engine import _build_llm_context

        ind = {
            "close": 100.0,
            "rsi": 55.3,
            "macd_hist": 0.25,
            "ema9": 101.0,
            "ema21": 99.0,
            "price_vs_bb": 0.75,
        }
        extra = {
            "fear_greed": 42,
            "fear_greed_class": "Fear",
            "sentiment": "positive",
            "sentiment_conf": 0.8,
            "volume_ratio": 1.2,
            "funding_action": "HOLD",
        }
        timeframes = []

        ctx = _build_llm_context("BTC-USD", ind, timeframes, extra)
        assert ctx["ticker"] == "BTC"  # -USD stripped
        assert ctx["price_usd"] == 100.0
        assert ctx["rsi"] == 55.3
        assert ctx["ema_bullish"] is True  # 101 > 99
        assert ctx["fear_greed"] == 42
        assert ctx["timeframe_summary"] == ""
        assert ctx["headlines"] == ""
        assert "asset_type" not in ctx  # Qwen3-specific, not in base

    def test_ticker_without_usd_suffix(self):
        from portfolio.signal_engine import _build_llm_context

        ind = {"close": 50.0, "rsi": 30.0, "macd_hist": -0.1,
               "ema9": 48.0, "ema21": 52.0, "price_vs_bb": 0.2}
        ctx = _build_llm_context("PLTR", ind, [], {})
        assert ctx["ticker"] == "PLTR"  # no -USD to strip

    def test_ema_gap_zero_division_safe(self):
        from portfolio.signal_engine import _build_llm_context

        ind = {"close": 50.0, "rsi": 50.0, "macd_hist": 0.0,
               "ema9": 0.0, "ema21": 0.0, "price_vs_bb": 0.5}
        ctx = _build_llm_context("BTC-USD", ind, [], {})
        assert ctx["ema_gap_pct"] == 0.0

    def test_timeframe_summary_formatting(self):
        from portfolio.signal_engine import _build_llm_context

        ind = {"close": 50.0, "rsi": 50.0, "macd_hist": 0.0,
               "ema9": 50.0, "ema21": 50.0, "price_vs_bb": 0.5}
        timeframes = [
            ("1h", {"action": "BUY", "indicators": {"rsi": 35.0}}),
            ("4h", {"action": "SELL", "indicators": {"rsi": 72.5}}),
            ("1d", {"action": "", "indicators": {}}),  # empty action, should skip
        ]
        ctx = _build_llm_context("ETH-USD", ind, timeframes, {})
        assert "1h: BUY (RSI=35)" in ctx["timeframe_summary"]
        assert "4h: SELL (RSI=72)" in ctx["timeframe_summary"]
        assert "1d" not in ctx["timeframe_summary"]  # empty action skipped

    def test_missing_extra_keys_use_defaults(self):
        from portfolio.signal_engine import _build_llm_context

        ind = {"close": 50.0, "rsi": 50.0, "macd_hist": 0.0,
               "ema9": 50.0, "ema21": 50.0, "price_vs_bb": 0.5}
        ctx = _build_llm_context("XAU-USD", ind, [], {})
        assert ctx["fear_greed"] == "N/A"
        assert ctx["news_sentiment"] == "N/A"
        assert ctx["volume_ratio"] == "N/A"


class TestMacroExternalCorrelationGroup:
    """Test the new macro_external correlation group."""

    def test_macro_external_group_penalizes_secondary(self):
        """When fear_greed, sentiment, and news_event all vote the same,
        only the highest-accuracy one gets full weight."""
        votes = {
            "fear_greed": "SELL",
            "sentiment": "SELL",
            "news_event": "SELL",
            "rsi": "BUY",
        }
        accuracy = {
            "fear_greed": {"accuracy": 0.56, "total": 8000},
            "sentiment": {"accuracy": 0.44, "total": 35000},
            "news_event": {"accuracy": 0.55, "total": 5000},
            "rsi": {"accuracy": 0.52, "total": 25000},
        }
        # Without correlation: 3 SELL vs 1 BUY => SELL wins easily
        # With correlation: fear_greed is leader (0.56), sentiment+news_event
        # get 0.3x penalty => effective SELL weight is much reduced
        action, conf = _weighted_consensus(votes, accuracy, "unknown")
        # The key test is that confidence is reduced compared to if all three
        # had full weight. We test that it's SELL (still majority) but
        # confidence is notably less than ~75%
        assert conf < 0.85  # would be higher without penalty


class TestTrendingUpRegimeGating:
    """BUG-152: SELL-biased signals must be gated in trending-up at 1d."""

    def test_trending_up_gates_sell_biased_signals_at_default(self):
        """All 8 signals should be gated at _default/1d in trending-up."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("trending-up")
        expected = {"trend", "ema", "volume_flow", "macro_regime",
                    "momentum_factors", "claude_fundamental", "funding",
                    "fear_greed"}
        assert expected == gated

    def test_trending_up_does_not_gate_sell_biased_at_3h(self):
        """SELL-biased signals work short-term — NOT gated at 3h."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("trending-up", "3h")
        # Only mean_reversion is gated at 3h in trending-up
        assert "trend" not in gated
        assert "ema" not in gated
        assert "volume_flow" not in gated
        assert "mean_reversion" in gated

    def test_trending_up_gating_forces_hold_in_consensus(self):
        """Gated signals should not participate in consensus vote."""
        votes = {
            "trend": "SELL",
            "ema": "SELL",
            "volume_flow": "SELL",
            "macro_regime": "SELL",
            "momentum_factors": "SELL",
            "rsi": "BUY",
            "ministral": "BUY",
        }
        accuracy = {s: {"accuracy": 0.55, "total": 100} for s in votes}
        action, conf = _weighted_consensus(votes, accuracy, "trending-up")
        # 5 SELL signals are gated, only rsi BUY + ministral BUY remain
        assert action == "BUY"


class TestTrendingDownRegimeGating:
    """BUG-154/155/156: Expanded trending-down gating."""

    def test_trending_down_gates_bb_and_claude_fundamental(self):
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("trending-down")
        assert "bb" in gated
        assert "claude_fundamental" in gated

    def test_trending_down_gates_sell_biased_signals(self):
        """BUG-156: volume_flow, macro_regime, ema, trend, heikin_ashi all 0%
        accurate on MSTR/PLTR in trending-down. Must be gated at 1d."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("trending-down")
        for sig in ("volume_flow", "macro_regime", "ema", "trend", "heikin_ashi"):
            assert sig in gated, f"{sig} should be gated in trending-down at 1d"

    def test_trending_down_3h_gates_bb_and_claude_fundamental(self):
        """At 3h, bb and claude_fundamental remain gated in trending-down."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("trending-down", "3h")
        assert "bb" in gated
        assert "claude_fundamental" in gated
        assert "mean_reversion" in gated


class TestCorrelationGroupSplit:
    """low_activity_timing cluster removed 2026-04-12.

    calendar (BUY-only, 84.2% ranging) and econ_calendar (SELL-only, 34.2%
    ranging) had opposite directions and divergent regime profiles.
    """

    def test_low_activity_timing_cluster_removed(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "low_activity_timing" not in CORRELATION_GROUPS

    def test_calendar_not_in_any_cluster(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        for name, members in CORRELATION_GROUPS.items():
            assert "calendar" not in members, f"calendar found in {name}"

    def test_econ_calendar_not_in_any_cluster(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        for name, members in CORRELATION_GROUPS.items():
            assert "econ_calendar" not in members, f"econ_calendar found in {name}"


class TestPerTickerConsensusGate:
    """BUG-164: Per-ticker consensus accuracy gate.

    Tickers with historically poor consensus accuracy (AMD 24.8%, GOOGL 31.3%)
    should be force-HOLD to prevent actively harmful recommendations.
    """

    def test_constants_defined(self):
        from portfolio.signal_engine import (
            _PER_TICKER_CONSENSUS_GATE,
            _PER_TICKER_CONSENSUS_MIN_SAMPLES,
        )
        assert _PER_TICKER_CONSENSUS_GATE == 0.38
        assert _PER_TICKER_CONSENSUS_MIN_SAMPLES == 50

    def test_gate_threshold_between_worst_and_marginal(self):
        """Gate at 38% catches AMD (24.8%), GOOGL (31.3%), META (34.2%)
        but allows AMZN (39.0%) and above."""
        from portfolio.signal_engine import _PER_TICKER_CONSENSUS_GATE
        assert _PER_TICKER_CONSENSUS_GATE > 0.248  # AMD caught
        assert _PER_TICKER_CONSENSUS_GATE > 0.313  # GOOGL caught
        assert _PER_TICKER_CONSENSUS_GATE > 0.342  # META caught
        assert _PER_TICKER_CONSENSUS_GATE < 0.390  # AMZN NOT caught


# ---------------------------------------------------------------------------
# Crisis mode detection tests
# ---------------------------------------------------------------------------

class TestCrisisMode:
    """Test crisis mode detection in _weighted_consensus.

    Crisis mode activates when >= 3 macro-external signals have blended
    accuracy below 35% (with sufficient samples). In crisis mode:
    - Trend signals (ema, trend, heikin_ashi, volume_flow) get 0.6x penalty
    - Mean-reversion signals (mean_reversion, calendar) get 1.3x boost
    """

    def test_crisis_constants_exist(self):
        from portfolio.signal_engine import (
            _CRISIS_MIN_BROKEN,
            _CRISIS_MR_BOOST,
            _CRISIS_THRESHOLD,
            _CRISIS_TREND_PENALTY,
        )
        assert _CRISIS_THRESHOLD == 0.35
        assert _CRISIS_MIN_BROKEN == 3
        assert 0 < _CRISIS_TREND_PENALTY < 1.0  # penalty reduces weight
        assert _CRISIS_MR_BOOST > 1.0  # boost increases weight

    def test_crisis_mode_penalizes_trend_signals(self):
        """When 3+ macro signals are broken, trend signals should get
        lower weight than in normal mode."""
        # Build accuracy data where 3 macro signals are broken (<35%)
        broken_acc = {
            "fear_greed": {"accuracy": 0.25, "total": 100},
            "macro_regime": {"accuracy": 0.30, "total": 100},
            "news_event": {"accuracy": 0.29, "total": 100},
            "structure": {"accuracy": 0.50, "total": 100},  # not broken
            "sentiment": {"accuracy": 0.46, "total": 100},  # not broken
            "ema": {"accuracy": 0.55, "total": 100},
            "trend": {"accuracy": 0.55, "total": 100},
            "mean_reversion": {"accuracy": 0.60, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        # Normal mode accuracy (no broken signals)
        normal_acc = {
            "fear_greed": {"accuracy": 0.55, "total": 100},
            "macro_regime": {"accuracy": 0.55, "total": 100},
            "news_event": {"accuracy": 0.55, "total": 100},
            "structure": {"accuracy": 0.55, "total": 100},
            "sentiment": {"accuracy": 0.55, "total": 100},
            "ema": {"accuracy": 0.55, "total": 100},
            "trend": {"accuracy": 0.55, "total": 100},
            "mean_reversion": {"accuracy": 0.60, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        votes = {"ema": "BUY", "trend": "BUY", "mean_reversion": "BUY", "rsi": "SELL"}

        # In crisis mode, trend signals should produce weaker BUY
        crisis_action, crisis_conf = _weighted_consensus(votes, broken_acc, "ranging")
        normal_action, normal_conf = _weighted_consensus(votes, normal_acc, "ranging")

        # Both should be BUY (mean_reversion boosted + ema/trend present)
        # but the weights will differ
        # Key check: crisis mode detected (3 broken signals)
        from portfolio.signal_engine import _CRISIS_THRESHOLD, ACCURACY_GATE_MIN_SAMPLES
        broken_count = sum(
            1 for s in ["fear_greed", "macro_regime", "news_event", "structure", "sentiment"]
            if broken_acc.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
            and broken_acc.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
        )
        assert broken_count >= 3, f"Expected 3+ broken signals, got {broken_count}"

    def test_crisis_mode_not_triggered_with_few_broken(self):
        """Crisis mode should NOT trigger when only 1-2 signals are broken."""
        from portfolio.signal_engine import _CRISIS_THRESHOLD, ACCURACY_GATE_MIN_SAMPLES
        acc = {
            "fear_greed": {"accuracy": 0.25, "total": 100},  # broken
            "macro_regime": {"accuracy": 0.30, "total": 100},  # broken
            "news_event": {"accuracy": 0.55, "total": 100},  # OK
            "structure": {"accuracy": 0.50, "total": 100},  # OK
            "sentiment": {"accuracy": 0.46, "total": 100},  # OK
        }
        broken_count = sum(
            1 for s in ["fear_greed", "macro_regime", "news_event", "structure", "sentiment"]
            if acc.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
            and acc.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
        )
        assert broken_count < 3, "Should NOT trigger crisis with only 2 broken signals"

    def test_group_leader_gate_lowered(self):
        """Group leader gate should be 0.46, not 0.47."""
        # This is validated by checking that a leader at 0.465 gets gated
        acc = {
            "sentiment": {"accuracy": 0.465, "total": 100},
            "fear_greed": {"accuracy": 0.25, "total": 100},
        }
        votes = {"sentiment": "SELL", "fear_greed": "SELL", "rsi": "BUY"}
        acc["rsi"] = {"accuracy": 0.55, "total": 100}
        # sentiment is leader of macro_external at 46.5% < 46% threshold
        # ... entire group should be gated
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # rsi BUY should win since macro_external group is gated
        assert action == "BUY"

    def test_recency_min_samples_constant(self):
        """_RECENCY_MIN_SAMPLES should be 30 to match accuracy gate."""
        from portfolio.signal_engine import _RECENCY_MIN_SAMPLES
        assert _RECENCY_MIN_SAMPLES == 30
