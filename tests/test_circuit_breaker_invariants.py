"""P3 defensive-hardening tests for circuit-breaker invariants (2026-04-17).

These pin relationships between module-level constants so that a future
refactor bumping one without the other fails at import time rather than
silently producing wrong behavior.
"""

from __future__ import annotations

import pytest


class TestModuleLoadAssertions:
    """The assert statements at module load must catch bad configurations."""

    def test_constants_import_cleanly(self):
        """Sanity: the module imports without assertion errors."""
        import portfolio.signal_engine  # noqa: F401

    def test_min_voters_positive(self):
        from portfolio.signal_engine import MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK
        assert MIN_VOTERS_CRYPTO > 0
        assert MIN_VOTERS_STOCK > 0

    def test_post_exclusion_min_at_most_soft_floor(self):
        from portfolio.signal_engine import _MIN_ACTIVE_VOTERS_SOFT, _POST_EXCLUSION_MIN
        assert _POST_EXCLUSION_MIN <= _MIN_ACTIVE_VOTERS_SOFT

    def test_relaxation_step_positive(self):
        from portfolio.signal_engine import _GATE_RELAXATION_STEP
        assert _GATE_RELAXATION_STEP > 0

    def test_relaxed_gate_above_directional_gate(self):
        """The accuracy gate after max relaxation must stay above the
        directional gate - otherwise 'directional gate is never relaxed'
        becomes meaningless."""
        from portfolio.signal_engine import (
            ACCURACY_GATE_THRESHOLD,
            _DIRECTIONAL_GATE_THRESHOLD,
            _GATE_RELAXATION_MAX,
        )
        relaxed_floor = ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX
        assert relaxed_floor > _DIRECTIONAL_GATE_THRESHOLD, (
            f"Relaxed accuracy floor ({relaxed_floor:.2f}) must stay above "
            f"directional gate ({_DIRECTIONAL_GATE_THRESHOLD})"
        )

    def test_relaxation_step_divides_max_cleanly(self):
        from portfolio.signal_engine import _GATE_RELAXATION_MAX, _GATE_RELAXATION_STEP
        ratio = _GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP
        assert abs(ratio - round(ratio)) < 1e-9

    def test_lone_signal_floor_matches_min_voters_base(self):
        from portfolio.signal_engine import (
            MIN_VOTERS_CRYPTO,
            MIN_VOTERS_STOCK,
            _LONE_SIGNAL_FLOOR,
        )
        assert _LONE_SIGNAL_FLOOR == max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)


class TestTickerDisabledByHorizonShape:
    """_TICKER_DISABLED_BY_HORIZON must stay well-formed."""

    def test_default_key_present(self):
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON
        assert "_default" in _TICKER_DISABLED_BY_HORIZON

    def test_all_horizon_keys_valid(self):
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON, _VALID_HORIZON_KEYS
        for key in _TICKER_DISABLED_BY_HORIZON:
            assert key in _VALID_HORIZON_KEYS

    def test_all_inner_values_are_dicts(self):
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON
        for inner in _TICKER_DISABLED_BY_HORIZON.values():
            assert isinstance(inner, dict)

    def test_all_ticker_entries_are_frozensets(self):
        """Mutable set would allow accidental mutation via
        _TICKER_DISABLED_SIGNALS['MSTR'].add(...)."""
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON
        for horizon_map in _TICKER_DISABLED_BY_HORIZON.values():
            for ticker, sigs in horizon_map.items():
                assert isinstance(sigs, frozenset), (
                    f"{ticker} signals must be frozenset, got {type(sigs).__name__}"
                )


class TestRegimeNormalization:
    """P2-D: _normalize_regime handles case/typo variants."""

    def test_canonical_passes_through(self):
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime("trending-up") == "trending-up"
        assert _normalize_regime("high-vol") == "high-vol"
        assert _normalize_regime("ranging") == "ranging"

    def test_uppercase_normalized(self):
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime("TRENDING-UP") == "trending-up"
        assert _normalize_regime("High-Vol") == "high-vol"

    def test_whitespace_stripped(self):
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime("  trending-up  ") == "trending-up"

    def test_underscore_to_dash(self):
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime("trending_up") == "trending-up"
        assert _normalize_regime("high_vol") == "high-vol"

    def test_none_passes_through(self):
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime(None) is None

    def test_unknown_preserved(self):
        """Unknown regime strings are normalized (case/whitespace) but not
        mapped - downstream _dynamic_min_voters_for_regime will default them."""
        from portfolio.signal_engine import _normalize_regime
        assert _normalize_regime("UNKNOWN") == "unknown"
        assert _normalize_regime("foobar") == "foobar"


class TestDynamicMinVotersRegimeTolerance:
    """P3-6 + P2-D: _dynamic_min_voters_for_regime accepts variant regime strings."""

    @pytest.mark.parametrize("variant", [
        "trending-up", "TRENDING-UP", "trending_up", "  trending-up  ",
        "Trending-Up",
    ])
    def test_trending_up_variants(self, variant):
        from portfolio.signal_engine import _dynamic_min_voters_for_regime
        assert _dynamic_min_voters_for_regime(variant) == 3

    @pytest.mark.parametrize("variant", [
        "high-vol", "HIGH-VOL", "high_vol", "High-Vol",
    ])
    def test_high_vol_variants(self, variant):
        from portfolio.signal_engine import _dynamic_min_voters_for_regime
        assert _dynamic_min_voters_for_regime(variant) == 4

    def test_none_and_unknown_equivalent(self):
        from portfolio.signal_engine import _dynamic_min_voters_for_regime
        assert _dynamic_min_voters_for_regime(None) == _dynamic_min_voters_for_regime("unknown")
        assert _dynamic_min_voters_for_regime(None) == _dynamic_min_voters_for_regime("ranging")
        assert _dynamic_min_voters_for_regime(None) == 5


class TestNaNInfInputsNoCrash:
    """P3-5 variants: malformed accuracy_data values must not crash the hot path."""

    def test_nan_accuracy_treated_as_default(self):
        from portfolio.signal_engine import _count_active_voters_at_gate
        votes = {"s1": "BUY"}
        accuracy = {
            "s1": {"accuracy": float("nan"), "total": 100,
                   "buy_accuracy": 0.55, "total_buy": 50},
        }
        # Should not raise; NaN -> default 0.5, samples=100, 0.5 >= 0.47 -> passes.
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_inf_accuracy_treated_as_default(self):
        from portfolio.signal_engine import _count_active_voters_at_gate
        votes = {"s1": "BUY"}
        accuracy = {"s1": {"accuracy": float("inf"), "total": 100}}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_negative_samples_treated_as_zero(self):
        """A negative sample count (corrupted cache) must be treated as 0,
        not crash. Since samples=0 < MIN_SAMPLES, the signal bypasses gate."""
        from portfolio.signal_engine import _count_active_voters_at_gate
        votes = {"s1": "BUY"}
        accuracy = {"s1": {"accuracy": 0.30, "total": -50}}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1


class TestGetHorizonDisabledDefensive:
    """P3-1: _get_horizon_disabled_signals uses .get('_default', {})."""

    def test_missing_default_returns_empty(self, monkeypatch):
        from portfolio.signal_engine import (
            _TICKER_DISABLED_BY_HORIZON,
            _get_horizon_disabled_signals,
        )
        # Remove _default temporarily and verify no crash.
        original_default = _TICKER_DISABLED_BY_HORIZON.pop("_default")
        try:
            result = _get_horizon_disabled_signals("MSTR", "1d")
            # Should not crash. Result may have horizon-specific entries only.
            assert isinstance(result, frozenset)
        finally:
            _TICKER_DISABLED_BY_HORIZON["_default"] = original_default
