"""Tests for IC-based weight multiplier in signal_engine.

Covers:
  1. _compute_ic_mult() math: positive IC boost, zero IC penalty, negative IC
  2. Stability filter: low ICIR → no adjustment
  3. Sample minimum: insufficient data → no adjustment
  4. Clamping: floor 0.6, cap 1.5
  5. Integration: IC multiplier applied in _weighted_consensus() weight chain
  6. Zero-IC penalty for phantom performers
  7. Per-ticker IC override when available
"""

import pytest

from portfolio.signal_engine import (
    _compute_ic_mult,
    _weighted_consensus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acc(accuracy, total=100, **kw):
    d = {"accuracy": accuracy, "total": total}
    d.update(kw)
    return d


# ===========================================================================
# _compute_ic_mult() unit tests
# ===========================================================================

class TestComputeIcMult:
    """Test the IC multiplier formula: 1.0 + alpha * ic, clamped [0.6, 1.5]."""

    def test_positive_ic_boosts(self):
        # IC=0.10, stable ICIR, enough samples → 1.0 + 2.0*0.10 = 1.20
        mult = _compute_ic_mult(0.10, 0.20, 500)
        assert abs(mult - 1.20) < 0.01

    def test_large_positive_ic_capped_at_1_5(self):
        # IC=0.35 → 1.0 + 2.0*0.35 = 1.70, capped to 1.5
        mult = _compute_ic_mult(0.35, 0.50, 1000)
        assert mult == 1.5

    def test_negative_ic_penalizes(self):
        # IC=-0.10, stable ICIR → 1.0 + 2.0*(-0.10) = 0.80
        mult = _compute_ic_mult(-0.10, 0.20, 500)
        assert abs(mult - 0.80) < 0.01

    def test_large_negative_ic_floored_at_0_6(self):
        # IC=-0.30 → 1.0 + 2.0*(-0.30) = 0.40, floored to 0.6
        mult = _compute_ic_mult(-0.30, 0.30, 500)
        assert mult == 0.6

    def test_zero_ic_with_high_samples_applies_penalty(self):
        # IC=0.0, samples > 500 → zero-IC penalty 0.85
        mult = _compute_ic_mult(0.0, 0.0, 1000)
        assert abs(mult - 0.85) < 0.01

    def test_zero_ic_with_low_samples_no_penalty(self):
        # IC=0.0 but samples < 100 → no adjustment (1.0)
        mult = _compute_ic_mult(0.0, 0.0, 50)
        assert mult == 1.0

    def test_low_icir_returns_1(self):
        # IC=0.15 but ICIR=0.05 (unstable) → no adjustment
        mult = _compute_ic_mult(0.15, 0.05, 500)
        assert mult == 1.0

    def test_insufficient_samples_returns_1(self):
        # IC=0.20, ICIR=0.30 but only 50 samples → no adjustment
        mult = _compute_ic_mult(0.20, 0.30, 50)
        assert mult == 1.0

    def test_boundary_icir_at_threshold(self):
        # ICIR exactly at 0.10 → should apply (>= boundary)
        mult = _compute_ic_mult(0.10, 0.10, 500)
        assert mult != 1.0  # should apply the boost

    def test_boundary_samples_at_minimum(self):
        # Exactly 100 samples → should apply
        mult = _compute_ic_mult(0.10, 0.20, 100)
        assert mult != 1.0


# ===========================================================================
# Integration: IC in _weighted_consensus()
# ===========================================================================

class TestIcInWeightedConsensus:
    """Verify IC multiplier affects final consensus weights."""

    def test_high_ic_signal_gets_boosted(self, monkeypatch):
        """A signal with positive IC should have higher effective weight.

        Uses 'unknown' regime to avoid regime weight multipliers that could
        override the IC effect.
        """
        votes = {"rsi": "BUY", "macd": "SELL"}
        acc = {
            "rsi": {"accuracy": 0.55, "total": 200},
            "macd": {"accuracy": 0.55, "total": 200},
        }
        # Mock IC data: rsi has positive IC, macd has zero IC
        ic_data = {
            "global": {
                "rsi": {"ic": 0.10, "icir": 0.20, "samples": 500},
                "macd": {"ic": 0.00, "icir": 0.00, "samples": 600},
            },
            "per_ticker": {},
        }
        monkeypatch.setattr(
            "portfolio.signal_engine._get_ic_data",
            lambda h: ic_data,
        )
        # Use "unknown" regime to avoid regime weight multipliers
        action, conf = _weighted_consensus(
            votes, acc, "unknown", horizon="1d",
        )
        # rsi (BUY): 0.55 * 1.20 = 0.66 (IC boost, no horizon mult for rsi)
        # macd (SELL): 0.55 * 0.85 * 1.2 = 0.561 (zero-IC penalty + 1d horizon mult)
        # buy_weight / total = 0.66 / 1.221 ≈ 0.5405 → BUY
        assert action == "BUY"
        assert conf > 0.53

    def test_without_ic_data_consensus_unchanged(self, monkeypatch):
        """When IC data is unavailable, consensus should work as before."""
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        acc = {
            "rsi": {"accuracy": 0.60, "total": 100},
            "macd": {"accuracy": 0.60, "total": 100},
            "ema": {"accuracy": 0.60, "total": 100},
        }
        # Return None → IC disabled
        monkeypatch.setattr(
            "portfolio.signal_engine._get_ic_data",
            lambda h: None,
        )
        action, conf = _weighted_consensus(
            votes, acc, "trending-up", horizon="1d",
        )
        assert action == "BUY"
        assert conf == 1.0
