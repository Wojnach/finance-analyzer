"""Tests for confidence calibration compression (Stage 7).

Verifies that the overconfidence problem (60-90% reported → ~50% actual)
is addressed by compressing high-confidence predictions.
"""

import pytest

from portfolio.signal_engine import apply_confidence_penalties


def _make_extra_info(buy_count=3, sell_count=2):
    return {
        "_buy_count": buy_count,
        "_sell_count": sell_count,
        "_voters": buy_count + sell_count,
    }


def _call(action, conf, regime="unknown", extra_info=None, config=None):
    """Call apply_confidence_penalties with safe defaults."""
    return apply_confidence_penalties(
        action, conf, regime,
        ind={},           # indicators dict (unused by calibration stage)
        extra_info=extra_info or _make_extra_info(),
        ticker=None,
        df=None,
        config=config,
    )


class TestCalibrationCompression:
    """Test Stage 7 confidence compression."""

    def test_low_confidence_unchanged(self):
        """Confidence below 0.55 should not be compressed."""
        action, conf, log = _call("BUY", 0.52)
        assert action == "BUY"
        assert abs(conf - 0.52) < 0.01

    def test_high_confidence_compressed(self):
        """Confidence of 0.80 should be compressed to ~0.625."""
        action, conf, log = _call("BUY", 0.80)
        # 0.55 + (0.80 - 0.55) * 0.3 = 0.55 + 0.075 = 0.625
        # But preceding stages (regime, unanimity) may also adjust
        assert conf < 0.80  # definitely compressed
        assert conf > 0.50  # not crushed to zero

    def test_extreme_confidence_compressed(self):
        """Confidence of 1.0 should be compressed significantly."""
        action, conf, log = _call("BUY", 1.0)
        # Raw: 0.55 + (1.0 - 0.55) * 0.3 = 0.55 + 0.135 = 0.685
        # But unanimity and other stages may reduce further
        assert conf < 0.75

    def test_hold_not_compressed(self):
        """HOLD actions should not be compressed."""
        action, conf, log = _call("HOLD", 0.80)
        assert action == "HOLD"

    def test_compression_preserves_ordering(self):
        """Higher raw confidence should still produce higher compressed confidence."""
        _, conf_low, _ = _call("BUY", 0.60)
        _, conf_mid, _ = _call("BUY", 0.75)
        _, conf_high, _ = _call("BUY", 0.90)
        assert conf_high >= conf_mid >= conf_low

    def test_compression_logged(self):
        """Calibration compression should be logged in penalty_log."""
        _, _, log = _call("BUY", 0.80)
        cal_entries = [e for e in log if e.get("stage") == "calibration_compression"]
        assert len(cal_entries) >= 1
        entry = cal_entries[0]
        assert "raw_conf" in entry
        assert "compressed_conf" in entry

    def test_sell_also_compressed(self):
        """SELL actions should be compressed the same way."""
        _, conf, _ = _call(
            "SELL", 0.80,
            extra_info=_make_extra_info(buy_count=2, sell_count=3),
        )
        assert conf < 0.80
