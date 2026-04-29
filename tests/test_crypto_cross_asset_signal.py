"""Tests for portfolio/signals/crypto_cross_asset.py."""
from __future__ import annotations

import pandas as pd
import pytest

from portfolio.signals.crypto_cross_asset import compute_crypto_cross_asset_signal


def _df(closes):
    return pd.DataFrame({"Close": closes})


class TestNonCryptoTickers:
    @pytest.mark.parametrize("ticker", ["XAU-USD", "XAG-USD", "MSTR", "AAPL", ""])
    def test_returns_hold(self, ticker):
        out = compute_crypto_cross_asset_signal(_df([100, 101]),
                                                {"ticker": ticker})
        assert out["signal"] == "HOLD"
        assert "non-crypto" in out["reason"]


class TestNoData:
    def test_btc_no_cross_asset_returns_hold(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]),
                                                {"ticker": "BTC-USD"})
        assert out["signal"] == "HOLD"
        assert "no cross-asset" in out["reason"]


class TestFearGreed:
    def test_extreme_fear_buys(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 15,
                                            "classification": "Extreme Fear"}},
        })
        assert out["signal"] == "BUY"
        assert out["sub_signals"]["fear_greed"]["decision"] == "BUY"

    def test_extreme_greed_sells(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 90,
                                            "classification": "Extreme Greed"}},
        })
        assert out["signal"] == "SELL"

    def test_neutral_holds(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 50,
                                            "classification": "Neutral"}},
        })
        assert out["signal"] == "HOLD"


class TestEthBtcRatio:
    def test_eth_outperformance_bullish_for_eth(self):
        # ETH +5%, BTC flat -> ratio rises -> BUY for ETH
        eth = _df([3500.0, 3675.0])
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([3500, 3675]), {
            "ticker": "ETH-USD",
            "cross_asset": {"eth_history": eth, "btc_history": btc},
        })
        assert out["sub_signals"]["eth_btc_ratio"]["decision"] == "BUY"

    def test_eth_underperformance_sells_for_eth(self):
        eth = _df([3500.0, 3325.0])  # -5%
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([3500, 3325]), {
            "ticker": "ETH-USD",
            "cross_asset": {"eth_history": eth, "btc_history": btc},
        })
        assert out["sub_signals"]["eth_btc_ratio"]["decision"] == "SELL"


class TestDxy:
    def test_dxy_up_bearish_for_crypto(self):
        dxy = _df([100.0, 101.0])  # +1% — crypto headwind
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"dxy_history": dxy},
        })
        assert out["sub_signals"]["dxy"]["decision"] == "SELL"

    def test_dxy_down_bullish(self):
        dxy = _df([100.0, 99.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"dxy_history": dxy},
        })
        assert out["sub_signals"]["dxy"]["decision"] == "BUY"


class TestSpy:
    def test_spy_strong_up_bullish(self):
        spy = _df([580.0, 585.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"spy_history": spy},
        })
        assert out["sub_signals"]["spy"]["decision"] == "BUY"


class TestGoldBtcRatio:
    def test_gold_strong_btc_weak_bearish(self):
        # Gold rises, BTC flat -> ratio rises -> SELL crypto
        gold = _df([4500.0, 4600.0])
        btc = _df([100000.0, 100000.0])
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"gold_history": gold, "btc_history": btc},
        })
        assert out["sub_signals"]["gold_btc_ratio"]["decision"] == "SELL"


class TestConfidenceBounds:
    def test_confidence_capped_at_max(self):
        # All five sub-indicators voting BUY
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {
                "fear_greed": {"value": 10, "classification": "Extreme Fear"},
                "dxy_history": _df([100.0, 99.0]),  # DXY down -> BUY
                "spy_history": _df([580.0, 585.0]),  # SPY up -> BUY
                "eth_history": _df([3500.0, 3500.0]),
                "btc_history": _df([100000.0, 100000.0]),
                "gold_history": _df([4500.0, 4400.0]),  # gold down -> BUY (gold/btc ratio falls)
            },
        })
        assert out["signal"] == "BUY"
        assert 0 < out["confidence"] <= 0.7  # _MAX_CONFIDENCE

    def test_confidence_zero_on_hold(self):
        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
            "ticker": "BTC-USD",
            "cross_asset": {"fear_greed": {"value": 50,
                                            "classification": "Neutral"}},
        })
        assert out["signal"] == "HOLD"
        assert out["confidence"] == 0.0
