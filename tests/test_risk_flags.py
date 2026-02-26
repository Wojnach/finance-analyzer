"""Tests for risk audit flags in risk_management.py."""

import pytest

from portfolio.risk_management import (
    check_concentration_risk,
    check_regime_mismatch,
    check_correlation_risk,
    check_atr_stop_proximity,
    compute_all_risk_flags,
    CORRELATED_PAIRS,
)


# --- Helper fixtures ---

def _make_agent_summary(tickers_data=None, fx_rate=10.0):
    """Build a minimal agent_summary dict."""
    signals = {}
    for ticker, data in (tickers_data or {}).items():
        signals[ticker] = {
            "price_usd": data.get("price", 100),
            "regime": data.get("regime", "ranging"),
            "atr_pct": data.get("atr_pct", 2.0),
            "action": data.get("action", "HOLD"),
            "extra": data.get("extra", {}),
        }
    return {"signals": signals, "fx_rate": fx_rate}


def _make_portfolio(cash=400000, holdings=None):
    """Build a minimal portfolio state dict."""
    return {
        "cash_sek": cash,
        "initial_value_sek": 500000,
        "holdings": holdings or {},
        "transactions": [],
    }


# --- Concentration risk ---

class TestConcentrationRisk:
    def test_no_flag_for_sell(self):
        pf = _make_portfolio()
        summary = _make_agent_summary({"BTC-USD": {"price": 60000}})
        result = check_concentration_risk("BTC-USD", "SELL", pf, summary, "bold")
        assert result is None

    def test_no_flag_when_below_threshold(self):
        pf = _make_portfolio(cash=400000)
        summary = _make_agent_summary({"BTC-USD": {"price": 60000}})
        # Bold: 30% of 400K = 120K. Total portfolio ~400K. 120K/400K = 30% < 40%
        result = check_concentration_risk("BTC-USD", "BUY", pf, summary, "bold")
        assert result is None

    def test_flag_when_above_threshold_with_existing(self):
        # Already holds a large position
        pf = _make_portfolio(cash=200000, holdings={
            "BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}
        })
        summary = _make_agent_summary({"BTC-USD": {"price": 60000}}, fx_rate=10.0)
        # Existing: 1.0 * 60K * 10 = 600K. Cash: 200K. Total: 800K.
        # Bold alloc: 30% of 200K = 60K. New position: 660K. Concentration: 660K/800K = 82.5%
        result = check_concentration_risk("BTC-USD", "BUY", pf, summary, "bold")
        assert result is not None
        assert result["flag"] == "concentration"
        assert result["concentration_pct"] > 40

    def test_patient_smaller_allocation(self):
        pf = _make_portfolio(cash=200000, holdings={
            "BTC-USD": {"shares": 0.5, "avg_cost_usd": 60000}
        })
        summary = _make_agent_summary({"BTC-USD": {"price": 60000}}, fx_rate=10.0)
        # Existing: 0.5 * 60K * 10 = 300K. Cash: 200K. Total: 500K.
        # Patient alloc: 15% of 200K = 30K. New position: 330K. Concentration: 66% > 40%
        result = check_concentration_risk("BTC-USD", "BUY", pf, summary, "patient")
        assert result is not None
        assert result["flag"] == "concentration"


# --- Regime mismatch ---

class TestRegimeMismatch:
    def test_no_mismatch_on_hold(self):
        summary = _make_agent_summary({"BTC-USD": {"regime": "trending-down"}})
        result = check_regime_mismatch("BTC-USD", "HOLD", summary)
        assert result is None

    def test_buy_in_downtrend_flags(self):
        summary = _make_agent_summary({"BTC-USD": {
            "regime": "trending-down",
            "extra": {"volume_ratio": 0.8},
        }})
        result = check_regime_mismatch("BTC-USD", "BUY", summary)
        assert result is not None
        assert result["flag"] == "regime_mismatch"

    def test_buy_in_downtrend_with_high_volume_ok(self):
        summary = _make_agent_summary({"BTC-USD": {
            "regime": "trending-down",
            "extra": {"volume_ratio": 2.0},
        }})
        result = check_regime_mismatch("BTC-USD", "BUY", summary)
        assert result is None  # High volume = potential reversal, no flag

    def test_sell_in_uptrend_flags(self):
        summary = _make_agent_summary({"BTC-USD": {
            "regime": "trending-up",
            "extra": {"volume_ratio": 0.5},
        }})
        result = check_regime_mismatch("BTC-USD", "SELL", summary)
        assert result is not None
        assert result["flag"] == "regime_mismatch"

    def test_sell_in_uptrend_with_high_volume_ok(self):
        summary = _make_agent_summary({"BTC-USD": {
            "regime": "trending-up",
            "extra": {"volume_ratio": 1.8},
        }})
        result = check_regime_mismatch("BTC-USD", "SELL", summary)
        assert result is None

    def test_buy_in_ranging_no_flag(self):
        summary = _make_agent_summary({"BTC-USD": {"regime": "ranging"}})
        result = check_regime_mismatch("BTC-USD", "BUY", summary)
        assert result is None

    def test_no_volume_data_flags(self):
        summary = _make_agent_summary({"BTC-USD": {
            "regime": "trending-down",
            "extra": {},
        }})
        result = check_regime_mismatch("BTC-USD", "BUY", summary)
        assert result is not None


# --- Correlation risk ---

class TestCorrelationRisk:
    def test_no_flag_for_sell(self):
        pf = _make_portfolio(holdings={"BTC-USD": {"shares": 1.0}})
        result = check_correlation_risk("ETH-USD", "SELL", pf)
        assert result is None

    def test_flag_when_correlated_held(self):
        pf = _make_portfolio(holdings={"BTC-USD": {"shares": 1.0}})
        result = check_correlation_risk("ETH-USD", "BUY", pf)
        assert result is not None
        assert result["flag"] == "correlation"
        assert "BTC-USD" in result["correlated_held"]

    def test_no_flag_when_uncorrelated(self):
        pf = _make_portfolio(holdings={"XAU-USD": {"shares": 10.0}})
        result = check_correlation_risk("NVDA", "BUY", pf)
        assert result is None

    def test_no_flag_when_correlated_not_held(self):
        pf = _make_portfolio(holdings={})
        result = check_correlation_risk("ETH-USD", "BUY", pf)
        assert result is None

    def test_multiple_correlated_held(self):
        pf = _make_portfolio(holdings={
            "AMD": {"shares": 100},
            "TSM": {"shares": 50},
        })
        result = check_correlation_risk("NVDA", "BUY", pf)
        assert result is not None
        assert len(result["correlated_held"]) >= 2

    def test_zero_shares_not_held(self):
        pf = _make_portfolio(holdings={"BTC-USD": {"shares": 0}})
        result = check_correlation_risk("ETH-USD", "BUY", pf)
        assert result is None

    def test_correlated_pairs_symmetric(self):
        """Verify key relationships are defined bidirectionally."""
        assert "BTC-USD" in CORRELATED_PAIRS.get("ETH-USD", [])
        assert "ETH-USD" in CORRELATED_PAIRS.get("BTC-USD", [])
        assert "NVDA" in CORRELATED_PAIRS.get("AMD", [])
        assert "AMD" in CORRELATED_PAIRS.get("NVDA", [])


# --- ATR stop proximity ---

class TestATRStopProximity:
    def test_no_flag_for_hold_no_position(self):
        pf = _make_portfolio()
        summary = _make_agent_summary({"BTC-USD": {"price": 60000, "atr_pct": 3.0}})
        result = check_atr_stop_proximity("BTC-USD", "BUY", pf, summary)
        assert result is None

    def test_flag_when_near_stop(self):
        # Entry at 60000, ATR 3%. Stop = 60000 * (1 - 2*0.03) = 56400.
        # Current price = 57000. Distance = 57000 - 56400 = 600.
        # ATR value = 57000 * 0.03 = 1710. Distance in ATR = 600/1710 = 0.35
        pf = _make_portfolio(holdings={
            "BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}
        })
        summary = _make_agent_summary({"BTC-USD": {"price": 57000, "atr_pct": 3.0}})
        result = check_atr_stop_proximity("BTC-USD", "BUY", pf, summary)
        assert result is not None
        assert result["flag"] == "atr_stop_proximity"
        assert result["distance_atr"] < 1.0

    def test_no_flag_when_far_from_stop(self):
        # Entry at 60000, ATR 2%. Stop = 60000 * (1 - 0.04) = 57600.
        # Current price = 65000. Distance = 65000 - 57600 = 7400.
        # ATR value = 65000 * 0.02 = 1300. Distance = 7400/1300 = 5.7 ATR
        pf = _make_portfolio(holdings={
            "BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}
        })
        summary = _make_agent_summary({"BTC-USD": {"price": 65000, "atr_pct": 2.0}})
        result = check_atr_stop_proximity("BTC-USD", "BUY", pf, summary)
        assert result is None

    def test_no_flag_when_no_atr_data(self):
        pf = _make_portfolio(holdings={
            "BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}
        })
        summary = _make_agent_summary({"BTC-USD": {"price": 60000, "atr_pct": 0}})
        result = check_atr_stop_proximity("BTC-USD", "BUY", pf, summary)
        assert result is None


# --- compute_all_risk_flags ---

class TestComputeAllRiskFlags:
    def test_returns_flags_for_non_hold(self):
        patient_pf = _make_portfolio(holdings={"BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}})
        bold_pf = _make_portfolio()
        summary = _make_agent_summary({
            "ETH-USD": {"price": 2000, "action": "BUY", "regime": "trending-down"},
        })
        result = compute_all_risk_flags(
            summary["signals"], patient_pf, bold_pf, summary
        )
        # Should get correlation (ETH correlated with held BTC) + regime mismatch
        assert len(result["flags"]) >= 1

    def test_disabled_returns_empty(self):
        config = {"risk_audit": {"enabled": False}}
        result = compute_all_risk_flags({}, {}, {}, {}, config)
        assert result["flags"] == []
        assert "disabled" in result["summary"].lower()

    def test_all_hold_minimal_flags(self):
        patient_pf = _make_portfolio()
        bold_pf = _make_portfolio()
        summary = _make_agent_summary({
            "BTC-USD": {"price": 60000, "action": "HOLD"},
        })
        result = compute_all_risk_flags(
            summary["signals"], patient_pf, bold_pf, summary
        )
        # No non-HOLD signals, only ATR proximity checks for held positions
        assert isinstance(result["flags"], list)

    def test_summary_string_format(self):
        patient_pf = _make_portfolio(holdings={"BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}})
        bold_pf = _make_portfolio()
        summary = _make_agent_summary({
            "ETH-USD": {"price": 2000, "action": "BUY", "regime": "trending-down"},
        })
        result = compute_all_risk_flags(
            summary["signals"], patient_pf, bold_pf, summary
        )
        assert isinstance(result["summary"], str)

    def test_empty_signals(self):
        result = compute_all_risk_flags({}, {}, {}, {})
        assert result["flags"] == []
        assert result["summary"] == "All clear"
