"""Tests for horizon-specific per-ticker blacklist (Batch 4 of 2026-04-16
accuracy gating reconfiguration).

The W15/W16 MSTR consensus collapse was caused by a blacklist built from
3h accuracy data applied at all horizons; at 1d the blacklisted signals
were 66-81% accurate. Batch 4 adds the per-horizon mechanism (entries empty
by default — populated by future audits).
"""

import pytest

from portfolio.signal_engine import (
    _TICKER_DISABLED_BY_HORIZON,
    _TICKER_DISABLED_SIGNALS,
    _get_horizon_disabled_signals,
    _weighted_consensus,
)


class TestHorizonDisabledStructure:
    """Guardrail: the dict has the expected shape."""

    def test_has_default_key(self):
        assert "_default" in _TICKER_DISABLED_BY_HORIZON

    def test_has_all_supported_horizons(self):
        for h in ("3h", "4h", "12h", "1d", "3d", "5d", "10d"):
            assert h in _TICKER_DISABLED_BY_HORIZON, f"missing horizon {h}"

    def test_legacy_alias_points_to_default(self):
        """_TICKER_DISABLED_SIGNALS must remain as a view of _default for
        backward compatibility with the signal dispatch loop."""
        assert _TICKER_DISABLED_SIGNALS is _TICKER_DISABLED_BY_HORIZON["_default"]

    def test_default_includes_mstr_trimmed_to_two(self):
        """Batch 1 trim: MSTR default has claude_fundamental + credit_spread_risk."""
        mstr = _TICKER_DISABLED_BY_HORIZON["_default"].get("MSTR", frozenset())
        assert mstr == frozenset({"claude_fundamental", "credit_spread_risk"})


class TestGetHorizonDisabledSignals:
    """Helper function: returns union of _default + horizon-specific."""

    def test_returns_default_set_when_no_horizon(self):
        result = _get_horizon_disabled_signals("MSTR", horizon=None)
        assert result == frozenset({"claude_fundamental", "credit_spread_risk"})

    def test_returns_default_only_when_horizon_empty(self):
        # No entries in "1d" map for MSTR today -> returns _default.
        result = _get_horizon_disabled_signals("MSTR", horizon="1d")
        assert result == frozenset({"claude_fundamental", "credit_spread_risk"})

    def test_unions_horizon_specific_with_default(self):
        """Temporarily add a 3h-specific entry for MSTR and verify union."""
        try:
            _TICKER_DISABLED_BY_HORIZON["3h"]["MSTR"] = frozenset({"volatility_sig"})
            result = _get_horizon_disabled_signals("MSTR", horizon="3h")
            assert result == frozenset(
                {"claude_fundamental", "credit_spread_risk", "volatility_sig"}
            )
            # 1d unaffected
            result_1d = _get_horizon_disabled_signals("MSTR", horizon="1d")
            assert result_1d == frozenset({"claude_fundamental", "credit_spread_risk"})
        finally:
            _TICKER_DISABLED_BY_HORIZON["3h"].pop("MSTR", None)

    def test_empty_frozenset_for_unknown_ticker(self):
        result = _get_horizon_disabled_signals("UNKNOWN-TICKER", horizon="1d")
        assert result == frozenset()

    def test_empty_frozenset_for_none_ticker(self):
        result = _get_horizon_disabled_signals(None, horizon="1d")
        assert result == frozenset()


class TestWeightedConsensusAppliesHorizonBlacklist:
    """_weighted_consensus must force-HOLD horizon-blacklisted signals."""

    def _make_stats(self, acc, total=100):
        return {
            "accuracy": acc, "total": total,
            "buy_accuracy": acc, "sell_accuracy": acc,
            "total_buy": total // 2, "total_sell": total // 2,
        }

    def test_consensus_ignores_horizon_blacklisted_signal(self):
        """If a ticker+horizon combo blacklists a signal, consensus drops its vote."""
        try:
            # Temporarily blacklist "rsi" on TEST-TICKER at 3h only.
            _TICKER_DISABLED_BY_HORIZON["3h"]["TEST-TICKER"] = frozenset({"rsi"})

            votes = {"rsi": "BUY", "ema": "SELL", "macd": "SELL",
                     "bb": "SELL", "volume": "SELL", "mean_reversion": "SELL"}
            accuracy = {k: self._make_stats(0.60) for k in votes}

            # At 3h: rsi is blacklisted, so BUY vote dropped. Consensus SELL.
            action_3h, _ = _weighted_consensus(
                votes, accuracy, regime="unknown", horizon="3h", ticker="TEST-TICKER",
            )
            assert action_3h == "SELL"

            # At 1d: rsi is NOT blacklisted, vote counts. Still SELL (5 vs 1)
            # but the rsi BUY reduces SELL confidence.
            action_1d, conf_1d = _weighted_consensus(
                votes, accuracy, regime="unknown", horizon="1d", ticker="TEST-TICKER",
            )
            assert action_1d == "SELL"
        finally:
            _TICKER_DISABLED_BY_HORIZON["3h"].pop("TEST-TICKER", None)

    def test_consensus_applies_default_blacklist_without_horizon_entry(self):
        """Default (MSTR claude_fundamental) applies at every horizon."""
        votes = {
            "claude_fundamental": "BUY",
            "rsi": "SELL", "ema": "SELL", "macd": "SELL",
            "bb": "SELL", "volume": "SELL",
        }
        accuracy = {k: self._make_stats(0.60) for k in votes}

        for horizon in ("3h", "1d", "3d"):
            action, _ = _weighted_consensus(
                votes, accuracy, regime="unknown", horizon=horizon, ticker="MSTR",
            )
            # claude_fundamental should be HOLD'd regardless of horizon (in _default)
            assert action == "SELL", f"Expected SELL at {horizon}, got {action}"

    def test_consensus_unchanged_without_ticker(self):
        """No ticker -> no horizon-specific gating applied (matches pre-Batch-4 behavior)."""
        votes = {
            "claude_fundamental": "BUY",
            "rsi": "BUY", "ema": "BUY", "macd": "BUY",
        }
        accuracy = {k: self._make_stats(0.60) for k in votes}
        action, _ = _weighted_consensus(
            votes, accuracy, regime="unknown", horizon="1d", ticker=None,
        )
        # claude_fundamental NOT gated (no ticker -> no blacklist lookup) -> BUY
        assert action == "BUY"
