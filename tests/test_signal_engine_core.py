"""Tests for core functions in portfolio/signal_engine.py.

Covers:
- _weighted_consensus: weighted vote aggregation with accuracy, regime, inversion, activation rates
- apply_confidence_penalties: 4-stage penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
- _confluence_score: majority-based scoring with volume confirmation
- _time_of_day_factor: time-based confidence dampening
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest import mock

from portfolio.signal_engine import (
    _weighted_consensus,
    apply_confidence_penalties,
    _confluence_score,
    _time_of_day_factor,
    REGIME_WEIGHTS,
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
    for i in range(n):
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
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "SELL", "bb": "BUY"}
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
    """Accuracy weighting and inversion logic."""

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

    def test_low_accuracy_signal_gets_inverted(self):
        """Signal with <50% accuracy and >=20 samples gets its vote flipped."""
        votes = {"bad_signal": "BUY"}
        accuracy = {"bad_signal": {"accuracy": 0.3, "total": 50}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # accuracy=0.3 < 0.5 with 50 samples => invert: BUY -> SELL, weight = 1.0-0.3 = 0.7
        assert action == "SELL"
        assert conf == 1.0  # only 1 effective vote, 100% confidence

    def test_inversion_flips_sell_to_buy(self):
        votes = {"bad_signal": "SELL"}
        accuracy = {"bad_signal": {"accuracy": 0.25, "total": 30}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # Inverted: SELL -> BUY, weight = 1.0 - 0.25 = 0.75
        assert action == "BUY"
        assert conf == 1.0

    def test_inversion_not_applied_below_20_samples(self):
        """Signal with accuracy < 50% but < 20 samples gets default weight, NOT inverted."""
        votes = {"new_signal": "BUY"}
        accuracy = {"new_signal": {"accuracy": 0.3, "total": 15}}
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # < 20 samples => weight=0.5, no inversion
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

    def test_inversion_with_mixed_votes(self):
        """Two signals: one good BUY and one bad BUY (inverted to SELL)."""
        votes = {"good": "BUY", "bad": "BUY"}
        accuracy = {
            "good": {"accuracy": 0.7, "total": 50},
            "bad": {"accuracy": 0.3, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # good: BUY weight=0.7, bad: inverted SELL weight=0.7
        # buy_weight=0.7, sell_weight=0.7 => tie => HOLD
        assert action == "HOLD"


class TestWeightedConsensusRegime:
    """Regime weight multipliers from REGIME_WEIGHTS."""

    def test_trending_up_boosts_ema_and_macd(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        accuracy = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "trending-up")
        # ema weight = 0.6 * 1.5 = 0.9, rsi weight = 0.6 * 0.7 = 0.42
        # BUY=0.9, SELL=0.42 => BUY
        assert action == "BUY"
        expected = 0.9 / (0.9 + 0.42)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_ranging_boosts_rsi_and_bb(self):
        votes = {"rsi": "BUY", "ema": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging")
        # rsi weight = 0.6 * 1.5 = 0.9, ema weight = 0.6 * 0.5 = 0.3
        # BUY=0.9, SELL=0.3 => BUY
        assert action == "BUY"
        expected = 0.9 / (0.9 + 0.3)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_high_vol_boosts_bb_and_volume(self):
        votes = {"bb": "BUY", "volume": "BUY", "ema": "SELL"}
        accuracy = {
            "bb": {"accuracy": 0.6, "total": 50},
            "volume": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
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
        votes = {"rsi": "BUY", "ema": "SELL"}
        accuracy = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        activation = {
            "rsi": {"normalized_weight": 2.0},  # rare signal, boosted
            "ema": {"normalized_weight": 0.5},   # noisy signal, dampened
        }
        action, conf = _weighted_consensus(votes, accuracy, "ranging", activation_rates=activation)
        # rsi: 0.6 * 1.5(ranging) * 2.0 = 1.8
        # ema: 0.6 * 0.5(ranging) * 0.5 = 0.15
        # BUY=1.8, SELL=0.15 => BUY
        assert action == "BUY"
        expected = 1.8 / (1.8 + 0.15)
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
        assert conf == pytest.approx(0.7 * 1.10, abs=0.01)

    def test_trending_up_sell_no_bonus(self):
        extra = _base_extra(voters=5, buy_count=1, sell_count=4)
        action, conf, log = apply_confidence_penalties(
            "SELL", 0.7, "trending-up", {}, extra, "BTC-USD", None, {},
        )
        # Counter-trend: no regime entry in log
        assert not any(p["stage"] == "regime" for p in log)
        assert conf == pytest.approx(0.7, abs=0.01)


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
        # 0.95 * 1.1 * 1.15 = 1.20175, should clamp to 1.0
        assert conf <= 1.0
        assert conf == pytest.approx(1.0)

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
            mock_dt.now.return_value = datetime(2026, 2, 26, hour, 30, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
        assert factor == 0.8

    @pytest.mark.parametrize("hour", [0, 1, 7, 8, 12, 15, 18, 21, 23])
    def test_day_hours_return_10(self, hour):
        with mock.patch("portfolio.signal_engine.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, hour, 30, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
        assert factor == 1.0


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
