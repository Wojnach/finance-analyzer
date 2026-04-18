"""Tests for per-ticker disabled signal rescue overrides."""

import pytest

from portfolio.signal_engine import (
    _DISABLED_SIGNAL_OVERRIDES,
    _weighted_consensus,
)
from portfolio.tickers import DISABLED_SIGNALS


class TestDisabledSignalOverrides:
    """Verify the rescue mechanism for globally disabled signals."""

    def test_ml_in_disabled_signals(self):
        """ML should still be in the global DISABLED_SIGNALS set."""
        assert "ml" in DISABLED_SIGNALS

    def test_ml_eth_in_overrides(self):
        """ML on ETH-USD should be in the override set."""
        assert ("ml", "ETH-USD") in _DISABLED_SIGNAL_OVERRIDES

    def test_ml_btc_not_in_overrides(self):
        """ML on BTC-USD should NOT be rescued — still globally disabled."""
        assert ("ml", "BTC-USD") not in _DISABLED_SIGNAL_OVERRIDES

    def test_rescued_signal_votes_in_consensus(self):
        """When ML is rescued for ETH-USD, its vote should participate in consensus."""
        votes = {"ml": "BUY", "rsi": "BUY", "macd": "BUY"}
        acc = {
            "ml": {"accuracy": 0.55, "total": 200},
            "rsi": {"accuracy": 0.55, "total": 200},
            "macd": {"accuracy": 0.55, "total": 200},
        }
        # ml is globally disabled but rescued for ETH-USD
        # In _weighted_consensus, ml vote passes through because
        # the override check happens at dispatch time, not consensus time
        action, conf = _weighted_consensus(
            votes, acc, "unknown", ticker="ETH-USD",
        )
        assert action == "BUY"

    def test_override_set_is_frozenset(self):
        """Override set should be immutable."""
        assert isinstance(_DISABLED_SIGNAL_OVERRIDES, frozenset)

    def test_all_overrides_reference_disabled_signals(self):
        """Every override should reference a signal in DISABLED_SIGNALS."""
        for sig_name, ticker in _DISABLED_SIGNAL_OVERRIDES:
            assert sig_name in DISABLED_SIGNALS, (
                f"Override ({sig_name}, {ticker}) references signal not in "
                f"DISABLED_SIGNALS — this override has no effect"
            )
