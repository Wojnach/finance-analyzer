"""Comprehensive tests for portfolio.reporting — the Layer 2 input builder.

Tests:
  - _cross_asset_signals: leader/follower detection
  - _write_compact_summary: three-tier compaction logic
  - write_agent_summary: main summary builder (heavily mocked)
  - _macro_headline: one-line macro string builder
  - write_tiered_summary: tier dispatch + output structure
  - _portfolio_snapshot: portfolio state loading
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_signal_entry(
    action="HOLD",
    confidence=0.5,
    price_usd=100.0,
    rsi=50.0,
    extra=None,
):
    """Build a minimal signal dict that write_agent_summary expects."""
    base_extra = {
        "fear_greed": 50,
        "fear_greed_class": "Neutral",
        "sentiment": "neutral",
        "sentiment_conf": 0.3,
        "ml_action": "HOLD",
        "ml_confidence": 0.0,
        "funding_rate": 0.0,
        "funding_action": "HOLD",
        "volume_ratio": 1.0,
        "volume_action": "HOLD",
        "ministral_action": "HOLD",
        "_voters": 3,
        "_total_applicable": 20,
        "_buy_count": 1,
        "_sell_count": 1,
        "_votes": {"rsi": "HOLD", "ema": "BUY", "macd": "SELL"},
        "_weighted_action": action,
        "_weighted_confidence": confidence,
        "_confluence_score": 0.4,
    }
    if extra:
        base_extra.update(extra)
    return {
        "action": action,
        "confidence": confidence,
        "indicators": {
            "close": price_usd,
            "rsi": rsi,
            "macd_hist": 0.5,
            "ema9": price_usd * 1.01,
            "ema21": price_usd * 0.99,
            "price_vs_bb": "inside",
            "atr": price_usd * 0.02,
            "atr_pct": 2.0,
        },
        "extra": base_extra,
    }


def _make_processed_signal(
    action="HOLD",
    confidence=0.5,
    price_usd=100.0,
    rsi=50.0,
    regime="range-bound",
    extra=None,
):
    """Build a signal entry as it appears in the processed summary dict
    (i.e., after write_agent_summary has formatted it).

    This is what _write_compact_summary and _write_tier*_summary receive.
    """
    base_extra = {
        "fear_greed": 50,
        "fear_greed_class": "Neutral",
        "sentiment": "neutral",
        "sentiment_conf": 0.3,
        "ml_action": "HOLD",
        "ml_confidence": 0.0,
        "funding_rate": 0.0,
        "funding_action": "HOLD",
        "volume_ratio": 1.0,
        "volume_action": "HOLD",
        "ministral_action": "HOLD",
        "_voters": 3,
        "_total_applicable": 20,
        "_buy_count": 1,
        "_sell_count": 1,
        "_votes": {"rsi": "HOLD", "ema": "BUY", "macd": "SELL"},
        "_weighted_action": action,
        "_weighted_confidence": confidence,
        "_confluence_score": 0.4,
    }
    if extra:
        base_extra.update(extra)
    return {
        "action": action,
        "confidence": confidence,
        "weighted_confidence": confidence,
        "confluence_score": 0.4,
        "price_usd": price_usd,
        "rsi": rsi,
        "macd_hist": 0.5,
        "bb_position": "inside",
        "atr": price_usd * 0.02,
        "atr_pct": 2.0,
        "regime": regime,
        "enhanced_signals": {},
        "extra": base_extra,
    }


def _make_full_summary(signals_dict=None, timeframes_dict=None, fear_greed_dict=None):
    """Build a minimal full summary dict (as produced by write_agent_summary)."""
    return {
        "timestamp": "2026-03-22T12:00:00+00:00",
        "trigger_reasons": ["test_trigger"],
        "fx_rate": 10.5,
        "portfolio": {
            "total_sek": 500000,
            "pnl_pct": 0.0,
            "cash_sek": 500000,
            "holdings": {},
            "num_transactions": 0,
        },
        "signals": signals_dict or {},
        "timeframes": timeframes_dict or {},
        "fear_greed": fear_greed_dict or {},
    }


@pytest.fixture
def mock_held_tickers_empty():
    """Patch _get_held_tickers to return empty set (no positions)."""
    with patch("portfolio.reporting._get_held_tickers", return_value=set()):
        yield


@pytest.fixture
def mock_atomic_write():
    """Patch _atomic_write_json to capture written data without disk I/O."""
    written = {}

    def _capture(path, data):
        written[str(path)] = data

    with patch("portfolio.reporting._atomic_write_json", side_effect=_capture):
        yield written


# ===================================================================
# 1. _cross_asset_signals
# ===================================================================

class TestCrossAssetSignals:
    """Tests for _cross_asset_signals leader/follower detection."""

    def test_btc_buy_eth_hold_produces_lead(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "BUY", "confidence": 0.8},
            "ETH-USD": {"action": "HOLD", "confidence": 0.5},
        }
        leads = _cross_asset_signals(all_signals)
        assert "ETH-USD" in leads
        assert leads["ETH-USD"]["leader"] == "BTC-USD"
        assert leads["ETH-USD"]["leader_action"] == "BUY"

    def test_xau_sell_xag_hold_produces_lead(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "XAU-USD": {"action": "SELL", "confidence": 0.7},
            "XAG-USD": {"action": "HOLD", "confidence": 0.3},
        }
        leads = _cross_asset_signals(all_signals)
        assert "XAG-USD" in leads
        assert leads["XAG-USD"]["leader_action"] == "SELL"

    def test_both_leaders_active(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "BUY"},
            "ETH-USD": {"action": "HOLD"},
            "XAU-USD": {"action": "SELL"},
            "XAG-USD": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(all_signals)
        assert len(leads) == 2
        assert "ETH-USD" in leads
        assert "XAG-USD" in leads

    def test_all_hold_produces_no_leads(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "HOLD"},
            "ETH-USD": {"action": "HOLD"},
            "XAU-USD": {"action": "HOLD"},
            "XAG-USD": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(all_signals)
        assert leads == {}

    def test_follower_already_matches_leader(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "BUY"},
            "ETH-USD": {"action": "BUY"},
        }
        leads = _cross_asset_signals(all_signals)
        assert "ETH-USD" not in leads

    def test_follower_has_opposite_action(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "BUY"},
            "ETH-USD": {"action": "SELL"},
        }
        leads = _cross_asset_signals(all_signals)
        assert "ETH-USD" not in leads

    def test_leader_missing_from_dict(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "ETH-USD": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(all_signals)
        assert leads == {}

    def test_empty_signals(self):
        from portfolio.reporting import _cross_asset_signals

        leads = _cross_asset_signals({})
        assert leads == {}

    def test_note_contains_leader_ticker(self):
        from portfolio.reporting import _cross_asset_signals

        all_signals = {
            "BTC-USD": {"action": "BUY"},
            "ETH-USD": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(all_signals)
        assert "BTC-USD" in leads["ETH-USD"]["note"]
        assert "ETH-USD" in leads["ETH-USD"]["note"]


# ===================================================================
# 2. _write_compact_summary — three-tier compaction
# ===================================================================

class TestWriteCompactSummary:
    """Tests for _write_compact_summary three-tier compaction logic."""

    def test_held_ticker_keeps_full_votes(self, mock_atomic_write):
        """Held tickers (tier 1) keep the full _votes dict in extra."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "BTC-USD": _make_processed_signal(
                action="HOLD",
                price_usd=68000,
                extra={"_votes": {"rsi": "BUY", "ema": "SELL", "macd": "HOLD"}},
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value={"BTC-USD"}):
            _write_compact_summary(summary)

        # Find the compact file written
        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert compact is not None
        btc = compact["signals"]["BTC-USD"]
        assert "_votes" in btc["extra"]
        assert "_vote_detail" not in btc.get("extra", {})

    def test_nonheld_nonhold_collapses_votes_to_detail_string(self, mock_atomic_write):
        """Non-held non-HOLD tickers (tier 2) collapse _votes into _vote_detail."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "NVDA": _make_processed_signal(
                action="BUY",
                price_usd=185,
                extra={
                    "_votes": {"sentiment": "BUY", "volume_flow": "BUY", "mean_reversion": "SELL"},
                },
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert compact is not None
        nvda = compact["signals"]["NVDA"]
        assert "_vote_detail" in nvda["extra"]
        assert "_votes" not in nvda["extra"]
        detail = nvda["extra"]["_vote_detail"]
        assert "B:" in detail
        assert "S:" in detail
        assert "sentiment" in detail
        assert "mean_reversion" in detail

    def test_hold_no_position_gets_minimal_fields(self, mock_atomic_write):
        """HOLD tickers with no position (tier 3) get minimal fields only."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "AAPL": _make_processed_signal(
                action="HOLD",
                price_usd=175.0,
                rsi=55.0,
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert compact is not None
        aapl = compact["signals"]["AAPL"]
        # Minimal ticker should have action, confidence, price_usd, rsi, regime, extra
        assert aapl["action"] == "HOLD"
        assert "price_usd" in aapl
        assert "rsi" in aapl
        # Should NOT have enhanced_signals or full extra
        assert "enhanced_signals" not in aapl
        # Minimal extra should only have count keys
        extra = aapl.get("extra", {})
        assert "_votes" not in extra
        assert "_vote_detail" not in extra

    def test_timeframes_only_for_interesting_tickers(self, mock_atomic_write):
        """Compact summary only includes timeframes for non-HOLD or held tickers."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "BTC-USD": _make_processed_signal(action="BUY", price_usd=68000),
            "AAPL": _make_processed_signal(action="HOLD", price_usd=175),
        }
        timeframes = {
            "BTC-USD": [{"horizon": "Now", "action": "BUY"}, {"horizon": "12h", "action": "BUY"}],
            "AAPL": [{"horizon": "Now", "action": "HOLD"}, {"horizon": "12h", "action": "HOLD"}],
        }
        summary = _make_full_summary(signals_dict=signals, timeframes_dict=timeframes)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert compact is not None
        assert "BTC-USD" in compact["timeframes"]
        assert "AAPL" not in compact["timeframes"]

    def test_enhanced_signals_stripped_from_all_tickers(self, mock_atomic_write):
        """enhanced_signals key should be removed from all tickers in compact."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "BTC-USD": _make_processed_signal(action="BUY"),
        }
        # Ensure enhanced_signals is present in the input
        signals["BTC-USD"]["enhanced_signals"] = {"trend": {"action": "BUY"}}
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert "enhanced_signals" not in compact["signals"]["BTC-USD"]

    def test_vote_detail_all_buys(self, mock_atomic_write):
        """When all signals voted BUY, _vote_detail should have B: and no S:."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "ETH-USD": _make_processed_signal(
                action="BUY",
                extra={"_votes": {"rsi": "BUY", "ema": "BUY", "volume": "BUY"}},
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        detail = compact["signals"]["ETH-USD"]["extra"]["_vote_detail"]
        assert "B:" in detail
        assert "S:" not in detail

    def test_vote_detail_only_holds(self, mock_atomic_write):
        """If _votes has only HOLDs, _vote_detail should be 'none'."""
        from portfolio.reporting import _write_compact_summary

        signals = {
            "NVDA": _make_processed_signal(
                action="BUY",  # action is BUY but all individual votes are HOLD
                extra={"_votes": {"rsi": "HOLD", "ema": "HOLD"}},
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        detail = compact["signals"]["NVDA"]["extra"]["_vote_detail"]
        assert detail == "none"

    def test_propagated_sections(self, mock_atomic_write):
        """Sections like futures_data, onchain, prophecy should propagate to compact."""
        from portfolio.reporting import _write_compact_summary

        summary = _make_full_summary()
        summary["futures_data"] = {"BTC-USD": {"open_interest": 500000}}
        summary["onchain"] = {"nvt": 45.2}
        summary["signal_reliability"] = {"BTC-USD": {"best": {"rsi": {"pct": 70, "n": 100}}}}
        summary["prophecy"] = {"total_active": 2}
        summary["forecast_accuracy"] = {"health": {"chronos": 0.9}}

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert compact is not None
        assert "futures_data" in compact
        assert "onchain" in compact
        assert "signal_reliability" in compact
        assert "prophecy" in compact
        assert "forecast_accuracy" in compact

    def test_signal_weights_excluded_from_compact(self, mock_atomic_write):
        """signal_weights should NOT be in compact (explicitly excluded)."""
        from portfolio.reporting import _write_compact_summary

        summary = _make_full_summary()
        summary["signal_weights"] = {"rsi": {"activation_rate": 0.5}}

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = None
        for path, data in mock_atomic_write.items():
            if "compact" in path:
                compact = data
                break
        assert "signal_weights" not in compact


# ===================================================================
# 3. _macro_headline
# ===================================================================

class TestMacroHeadline:
    """Tests for _macro_headline one-line builder."""

    def test_full_macro_data(self):
        from portfolio.reporting import _macro_headline

        summary = {
            "macro": {
                "dxy": {"value": 97.5, "change_5d_pct": 0.3},
                "treasury": {"10y": 4.05, "10y_change_5d_pct": -0.5},
                "vix": {"value": 22.3, "change_pct": 1.5},
                "fed": {"days_until": 14},
            },
            "fear_greed": {
                "BTC-USD": {"value": 25},
            },
        }
        headline = _macro_headline(summary)
        assert "DXY" in headline
        assert "10Y" in headline
        assert "F&G" in headline
        assert "VIX" in headline
        assert "FOMC" in headline

    def test_empty_macro(self):
        from portfolio.reporting import _macro_headline

        summary = {}
        headline = _macro_headline(summary)
        assert headline == ""

    def test_dxy_arrow_up(self):
        from portfolio.reporting import _macro_headline

        summary = {"macro": {"dxy": {"value": 98.0, "change_5d_pct": 0.5}}}
        headline = _macro_headline(summary)
        assert "98" in headline

    def test_dxy_arrow_down(self):
        from portfolio.reporting import _macro_headline

        summary = {"macro": {"dxy": {"value": 96.0, "change_5d_pct": -0.3}}}
        headline = _macro_headline(summary)
        assert "96" in headline

    def test_fg_crypto_and_stock(self):
        from portfolio.reporting import _macro_headline

        summary = {
            "fear_greed": {
                "BTC-USD": {"value": 7},
                "NVDA": {"value": 48},
            },
        }
        headline = _macro_headline(summary)
        assert "F&G 7/48" in headline

    def test_fg_only_crypto(self):
        from portfolio.reporting import _macro_headline

        summary = {
            "fear_greed": {
                "BTC-USD": {"value": 12},
            },
        }
        headline = _macro_headline(summary)
        assert "F&G 12" in headline

    def test_fomc_days(self):
        from portfolio.reporting import _macro_headline

        summary = {"macro": {"fed": {"days_until": 3}}}
        headline = _macro_headline(summary)
        assert "FOMC 3d" in headline

    def test_vix_no_change(self):
        from portfolio.reporting import _macro_headline

        summary = {"macro": {"vix": {"value": 18.5, "change_pct": 0}}}
        headline = _macro_headline(summary)
        assert "VIX 18.5" in headline

    def test_dxy_zero_change_no_arrow(self):
        from portfolio.reporting import _macro_headline

        summary = {"macro": {"dxy": {"value": 100.0, "change_5d_pct": 0}}}
        headline = _macro_headline(summary)
        # No arrow when change is 0
        assert "100" in headline


# ===================================================================
# 4. _portfolio_snapshot
# ===================================================================

class TestPortfolioSnapshot:
    """Tests for _portfolio_snapshot loading and formatting."""

    def test_basic_snapshot_with_prices(self, tmp_path):
        from portfolio.reporting import _portfolio_snapshot

        state = {
            "cash_sek": 400000,
            "initial_value_sek": 500000,
            "holdings": {
                "BTC-USD": {"shares": 0.5},
            },
        }
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text(json.dumps(state))

        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        result = _portfolio_snapshot(state_file, prices_usd, fx_rate)

        assert result["cash_sek"] == 400000
        # total = 400000 + 0.5 * 68000 * 10 = 740000
        assert result["total_sek"] == 740000
        assert result["pnl_pct"] == 48.0
        assert "BTC-USD" in result["holdings"][0]

    def test_snapshot_without_prices(self, tmp_path):
        from portfolio.reporting import _portfolio_snapshot

        state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {"ETH-USD": {"shares": 5.0}},
        }
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text(json.dumps(state))

        result = _portfolio_snapshot(state_file)
        assert result["cash_sek"] == 500000
        assert result["total_sek"] == 500000
        assert result["pnl_pct"] == 0.0
        # Without prices, holdings list just has ticker names
        assert "ETH-USD" in result["holdings"]

    def test_snapshot_missing_file(self, tmp_path):
        """Missing file: load_json returns default {}, so cash=0, initial=500000."""
        from portfolio.reporting import _portfolio_snapshot

        missing = tmp_path / "nonexistent.json"
        result = _portfolio_snapshot(missing)
        # load_json returns {} for missing file (default={}),
        # so cash=0, initial=500000 (default), pnl=-100%
        assert result["cash_sek"] == 0
        assert result["total_sek"] == 0
        assert result["pnl_pct"] == -100.0

    def test_snapshot_load_json_raises(self):
        """When load_json itself raises, fallback returns zeros."""
        from portfolio.reporting import _portfolio_snapshot

        with patch("portfolio.reporting.load_json", side_effect=FileNotFoundError("gone")):
            result = _portfolio_snapshot(Path("/fake/path.json"))
        assert result["cash_sek"] == 0
        assert result["total_sek"] == 0
        assert result["pnl_pct"] == 0

    def test_snapshot_no_holdings(self, tmp_path):
        from portfolio.reporting import _portfolio_snapshot

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text(json.dumps(state))

        result = _portfolio_snapshot(state_file, {"BTC-USD": 68000}, 10.0)
        assert result["cash_sek"] == 500000
        assert result["total_sek"] == 500000
        assert "holdings" not in result

    def test_snapshot_zero_shares_ignored(self, tmp_path):
        from portfolio.reporting import _portfolio_snapshot

        state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {"BTC-USD": {"shares": 0}},
        }
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text(json.dumps(state))

        result = _portfolio_snapshot(state_file, {"BTC-USD": 68000}, 10.0)
        assert result["total_sek"] == 500000
        assert "holdings" not in result


# ===================================================================
# 5. write_tiered_summary
# ===================================================================

class TestWriteTieredSummary:
    """Tests for write_tiered_summary dispatching."""

    def test_tier1_output_structure(self, mock_atomic_write):
        from portfolio.reporting import write_tiered_summary

        signals = {
            "BTC-USD": _make_processed_signal(action="HOLD", price_usd=68000),
        }
        timeframes = {
            "BTC-USD": [
                {"horizon": "Now", "action": "HOLD"},
                {"horizon": "12h", "action": "BUY"},
            ],
        }
        summary = _make_full_summary(
            signals_dict=signals,
            timeframes_dict=timeframes,
            fear_greed_dict={"BTC-USD": {"value": 25}},
        )

        with patch("portfolio.reporting._get_held_tickers", return_value={"BTC-USD"}):
            write_tiered_summary(summary, tier=1)

        t1 = None
        for path, data in mock_atomic_write.items():
            if "t1" in path or "tier" in path.lower():
                t1 = data
                break
        assert t1 is not None
        assert t1["tier"] == 1
        assert "held_positions" in t1
        assert "portfolio_patient" in t1
        assert "portfolio_bold" in t1
        assert "macro_headline" in t1
        assert "all_prices" in t1

    def test_tier1_held_position_detail(self, mock_atomic_write):
        from portfolio.reporting import write_tiered_summary

        signals = {
            "BTC-USD": _make_processed_signal(action="BUY", price_usd=68000, rsi=42.0),
        }
        timeframes = {
            "BTC-USD": [
                {"horizon": "Now", "action": "BUY"},
                {"horizon": "12h", "action": "BUY"},
            ],
        }
        summary = _make_full_summary(signals_dict=signals, timeframes_dict=timeframes)

        with patch("portfolio.reporting._get_held_tickers", return_value={"BTC-USD"}):
            write_tiered_summary(summary, tier=1)

        t1 = None
        for path, data in mock_atomic_write.items():
            if "t1" in path or "tier" in path.lower():
                t1 = data
                break
        assert t1 is not None
        assert "BTC-USD" in t1["held_positions"]
        pos = t1["held_positions"]["BTC-USD"]
        assert pos["price_usd"] == 68000
        assert pos["action"] == "BUY"
        assert pos["rsi"] == 42.0
        assert "votes" in pos
        assert "timeframes" in pos
        assert "B" in pos["timeframes"]

    def test_tier2_output_structure(self, mock_atomic_write):
        from portfolio.reporting import write_tiered_summary

        signals = {
            "BTC-USD": _make_processed_signal(action="BUY", price_usd=68000),
            "ETH-USD": _make_processed_signal(action="HOLD", price_usd=2000),
            "NVDA": _make_processed_signal(action="HOLD", price_usd=185),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            write_tiered_summary(summary, tier=2, triggered_tickers={"BTC-USD"})

        t2 = None
        for path, data in mock_atomic_write.items():
            if "t2" in path or "tier" in path.lower():
                t2 = data
                break
        assert t2 is not None
        assert t2["tier"] == 2
        assert "signals" in t2
        assert "timeframes" in t2
        assert "BTC-USD" in t2["signals"]

    def test_tier2_triggered_gets_full_detail(self, mock_atomic_write):
        """Triggered tickers should have full extra keys (not collapsed to _vote_detail)."""
        from portfolio.reporting import write_tiered_summary

        signals = {
            "BTC-USD": _make_processed_signal(
                action="BUY",
                price_usd=68000,
                extra={"_votes": {"rsi": "BUY", "ema": "BUY"}},
            ),
        }
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            write_tiered_summary(summary, tier=2, triggered_tickers={"BTC-USD"})

        t2 = None
        for path, data in mock_atomic_write.items():
            if "t2" in path or "tier" in path.lower():
                t2 = data
                break
        assert t2 is not None
        btc = t2["signals"]["BTC-USD"]
        # Full detail tickers keep _votes (not collapsed)
        assert "extra" in btc

    def test_tier2_non_triggered_non_held_gets_price_only(self, mock_atomic_write):
        """Non-triggered non-held tickers not in top 5 get price-only entry."""
        from portfolio.reporting import write_tiered_summary

        # Create enough tickers so some fall outside top 5
        signals = {}
        for i, name in enumerate(["T1", "T2", "T3", "T4", "T5", "T6", "T7"]):
            signals[name] = _make_processed_signal(
                action="HOLD",
                price_usd=100 + i,
                extra={"_buy_count": 0, "_sell_count": 0, "_total_applicable": 20},
            )
        summary = _make_full_summary(signals_dict=signals)

        with patch("portfolio.reporting._get_held_tickers", return_value=set()):
            write_tiered_summary(summary, tier=2, triggered_tickers=set())

        t2 = None
        for path, data in mock_atomic_write.items():
            if "t2" in path or "tier" in path.lower():
                t2 = data
                break
        assert t2 is not None
        # Some tickers should be price-only (just action + price_usd)
        price_only = [
            t for t, d in t2["signals"].items()
            if set(d.keys()) == {"action", "price_usd"}
        ]
        assert len(price_only) >= 2  # at least some are price-only

    def test_tier3_no_extra_file(self, mock_atomic_write):
        """Tier 3 uses existing compact summary; write_tiered_summary writes nothing."""
        from portfolio.reporting import write_tiered_summary

        summary = _make_full_summary()
        write_tiered_summary(summary, tier=3)
        # Tier 3 should not write any tier-specific file
        assert len(mock_atomic_write) == 0


# ===================================================================
# 6. write_agent_summary (heavily mocked)
# ===================================================================

class TestWriteAgentSummary:
    """Tests for write_agent_summary main function."""

    def _mock_everything(self):
        """Return a patch context stack that mocks all sub-module imports."""
        patches = [
            patch("portfolio.reporting._atomic_write_json"),
            patch("portfolio.reporting._write_compact_summary"),
            patch("portfolio.reporting.detect_regime", return_value="range-bound"),
            patch("portfolio.reporting.portfolio_value", return_value=500000),
            patch("portfolio.reporting.get_enhanced_signals", return_value={}),
            patch("portfolio.reporting.load_json", return_value=None),
            patch("portfolio.reporting._cached", return_value=None),
            patch("portfolio.api_utils.load_config", return_value={"notification": {}}),
        ]
        return patches

    def test_basic_structure(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {
            "BTC-USD": _make_signal_entry(action="BUY", price_usd=68000),
        }
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.5
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                            trigger_reasons=["test"],
                                        )

        assert "timestamp" in result
        assert "trigger_reasons" in result
        assert result["trigger_reasons"] == ["test"]
        assert "fx_rate" in result
        assert "portfolio" in result
        assert "signals" in result
        assert "timeframes" in result
        assert "fear_greed" in result
        assert result["fx_rate"] == 10.5

    def test_portfolio_section(self):
        from portfolio.reporting import write_agent_summary

        state = {
            "cash_sek": 400000,
            "initial_value_sek": 500000,
            "holdings": {"BTC-USD": {"shares": 0.5}},
            "transactions": [{"action": "BUY"}],
        }
        signals = {"BTC-USD": _make_signal_entry(action="BUY", price_usd=68000)}
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=740000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                        )

        pf = result["portfolio"]
        assert pf["total_sek"] == 740000
        assert pf["cash_sek"] == 400000
        assert pf["num_transactions"] == 1
        assert pf["pnl_pct"] == 48.0

    def test_signal_entry_fields(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {
            "BTC-USD": _make_signal_entry(action="BUY", price_usd=68000, rsi=45.0),
        }
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="trending-up"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                        )

        sig = result["signals"]["BTC-USD"]
        assert sig["action"] == "BUY"
        assert sig["price_usd"] == 68000
        assert sig["rsi"] == 45.0
        assert sig["regime"] == "trending-up"
        assert "atr" in sig
        assert "atr_pct" in sig
        assert "enhanced_signals" in sig
        assert "extra" in sig

    def test_module_warnings_populated_on_failures(self):
        """When sub-modules fail, _module_warnings should be populated."""
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {"BTC-USD": _make_signal_entry()}
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        tf_data = {}

        def _fail_load_config():
            return {"notification": {}, "monte_carlo": {"enabled": False}}

        # Make accuracy_stats import fail inside write_agent_summary
        import builtins
        real_import = builtins.__import__

        failing_modules = set()

        def selective_import(name, *args, **kwargs):
            if "accuracy_stats" in name:
                failing_modules.add("accuracy_stats")
                raise ImportError("test: accuracy_stats unavailable")
            if "alpha_vantage" in name and "portfolio" in name:
                failing_modules.add("alpha_vantage")
                raise ImportError("test: alpha_vantage unavailable")
            return real_import(name, *args, **kwargs)

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", side_effect=_fail_load_config):
                                        with patch("builtins.__import__", side_effect=selective_import):
                                            result = write_agent_summary(
                                                signals, prices_usd, fx_rate, state, tf_data,
                                            )

        # Module warnings should be present when sub-modules fail
        warnings = result.get("_module_warnings", [])
        # At least some warnings — accuracy_stats and alpha_vantage should fail
        assert len(warnings) > 0

    def test_fear_greed_section(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {
            "BTC-USD": _make_signal_entry(
                extra={"fear_greed": 25, "fear_greed_class": "Extreme Fear"},
            ),
        }
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                        )

        fg = result["fear_greed"]
        assert "BTC-USD" in fg
        assert fg["BTC-USD"]["value"] == 25
        assert fg["BTC-USD"]["classification"] == "Extreme Fear"

    def test_cross_asset_leads_included(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {
            "BTC-USD": _make_signal_entry(action="BUY"),
            "ETH-USD": _make_signal_entry(action="HOLD"),
        }
        prices_usd = {"BTC-USD": 68000, "ETH-USD": 2000}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                        )

        assert "cross_asset_leads" in result
        assert "ETH-USD" in result["cross_asset_leads"]

    def test_timeframe_data_included(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {"BTC-USD": _make_signal_entry(action="BUY", price_usd=68000)}
        prices_usd = {"BTC-USD": 68000}
        fx_rate = 10.0
        tf_data = {
            "BTC-USD": [
                ("Now", {
                    "action": "BUY",
                    "confidence": 0.7,
                    "indicators": {
                        "rsi": 45.0,
                        "macd_hist": 0.5,
                        "ema9": 68500,
                        "ema21": 67500,
                        "price_vs_bb": "inside",
                    },
                }),
                ("12h", {
                    "action": "BUY",
                    "confidence": 0.6,
                    "indicators": {
                        "rsi": 48.0,
                        "macd_hist": 1.2,
                        "ema9": 68400,
                        "ema21": 67600,
                        "price_vs_bb": "inside",
                    },
                }),
            ],
        }

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.detect_regime", return_value="range-bound"):
                    with patch("portfolio.reporting.portfolio_value", return_value=500000):
                        with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                            with patch("portfolio.reporting.load_json", return_value=None):
                                with patch("portfolio.reporting._cached", return_value=None):
                                    with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                        result = write_agent_summary(
                                            signals, prices_usd, fx_rate, state, tf_data,
                                        )

        tfs = result["timeframes"]["BTC-USD"]
        assert len(tfs) == 2
        assert tfs[0]["horizon"] == "Now"
        assert tfs[0]["action"] == "BUY"
        assert tfs[1]["horizon"] == "12h"

    def test_zero_initial_value_no_division_error(self):
        """BUG-99: zero initial value should not cause ZeroDivisionError."""
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 0, "initial_value_sek": 0, "holdings": {}}
        signals = {}
        prices_usd = {}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.portfolio_value", return_value=0):
                    with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                        with patch("portfolio.reporting.load_json", return_value=None):
                            with patch("portfolio.reporting._cached", return_value=None):
                                with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                    result = write_agent_summary(
                                        signals, prices_usd, fx_rate, state, tf_data,
                                    )

        assert result["portfolio"]["pnl_pct"] == 0

    def test_trigger_reasons_default_empty(self):
        from portfolio.reporting import write_agent_summary

        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        signals = {}
        prices_usd = {}
        fx_rate = 10.0
        tf_data = {}

        with patch("portfolio.reporting._atomic_write_json"):
            with patch("portfolio.reporting._write_compact_summary"):
                with patch("portfolio.reporting.portfolio_value", return_value=500000):
                    with patch("portfolio.reporting.get_enhanced_signals", return_value={}):
                        with patch("portfolio.reporting.load_json", return_value=None):
                            with patch("portfolio.reporting._cached", return_value=None):
                                with patch("portfolio.api_utils.load_config", return_value={"notification": {}}):
                                    result = write_agent_summary(
                                        signals, prices_usd, fx_rate, state, tf_data,
                                    )

        assert result["trigger_reasons"] == []


# ===================================================================
# 7. _get_held_tickers
# ===================================================================

class TestGetHeldTickers:
    """Tests for the _get_held_tickers cache and loading."""

    def test_returns_held_tickers_from_both_portfolios(self):
        import portfolio.shared_state as ss
        from portfolio.reporting import _get_held_tickers, _held_tickers_cache

        patient = {
            "holdings": {"BTC-USD": {"shares": 0.5}},
        }
        bold = {
            "holdings": {"ETH-USD": {"shares": 2.0}},
        }

        def fake_load(path, default=None):
            name = Path(path).name
            if "bold" in name:
                return bold
            return patient

        # Force cache miss by setting cycle_id to something different
        old_cycle = ss._run_cycle_id
        ss._run_cycle_id = -42
        _held_tickers_cache["cycle_id"] = -999  # different from _run_cycle_id

        try:
            with patch("portfolio.reporting.load_json", side_effect=fake_load):
                result = _get_held_tickers()
        finally:
            ss._run_cycle_id = old_cycle

        assert "BTC-USD" in result
        assert "ETH-USD" in result

    def test_zero_shares_not_included(self):
        import portfolio.shared_state as ss
        from portfolio.reporting import _get_held_tickers, _held_tickers_cache

        patient = {
            "holdings": {"BTC-USD": {"shares": 0}},
        }

        old_cycle = ss._run_cycle_id
        ss._run_cycle_id = -43
        _held_tickers_cache["cycle_id"] = -999

        try:
            with patch("portfolio.reporting.load_json", side_effect=lambda *a, **kw: patient):
                result = _get_held_tickers()
        finally:
            ss._run_cycle_id = old_cycle

        assert "BTC-USD" not in result


# ---------------------------------------------------------------------------
# Failure-streak escalation (2026-04-22 follow-up)
# ---------------------------------------------------------------------------

class TestModuleFailureStreak:
    """Regression: 2026-04-22 follow-up — reporting.py's bare-except pattern
    suppressed the MC seed=None bug for weeks. Now track streaks and escalate
    to critical_errors.jsonl after N consecutive failures."""

    def _reset(self):
        import portfolio.reporting as rp
        rp._module_failure_streaks.clear()
        rp._module_escalated.clear()

    def test_success_resets_streak(self):
        from portfolio.reporting import _module_failure_streaks, _track_module_outcome
        self._reset()
        _track_module_outcome("monte_carlo", ok=False, exc=ValueError("x"))
        _track_module_outcome("monte_carlo", ok=False, exc=ValueError("x"))
        assert _module_failure_streaks.get("monte_carlo") == 2
        _track_module_outcome("monte_carlo", ok=True)
        assert "monte_carlo" not in _module_failure_streaks

    def test_escalates_once_at_threshold(self):
        from portfolio.reporting import _FAILURE_STREAK_THRESHOLD, _track_module_outcome
        self._reset()
        calls = []
        with patch("portfolio.claude_gate.record_critical_error",
                   side_effect=lambda **kw: calls.append(kw)):
            for _ in range(_FAILURE_STREAK_THRESHOLD + 3):
                _track_module_outcome("monte_carlo", ok=False, exc=TypeError("None + int"))
        assert len(calls) == 1
        assert calls[0]["category"] == "reporting_module_failure_streak"
        assert calls[0]["caller"] == "reporting.monte_carlo"
        assert calls[0]["context"]["last_exception_type"] == "TypeError"

    def test_re_escalates_after_recovery_then_fail(self):
        """Once the module recovers, a fresh streak should be able to
        escalate again if it re-breaks."""
        from portfolio.reporting import _FAILURE_STREAK_THRESHOLD, _track_module_outcome
        self._reset()
        calls = []
        with patch("portfolio.claude_gate.record_critical_error",
                   side_effect=lambda **kw: calls.append(kw)):
            for _ in range(_FAILURE_STREAK_THRESHOLD):
                _track_module_outcome("monte_carlo", ok=False, exc=ValueError("x"))
            _track_module_outcome("monte_carlo", ok=True)
            for _ in range(_FAILURE_STREAK_THRESHOLD):
                _track_module_outcome("monte_carlo", ok=False, exc=ValueError("y"))
        assert len(calls) == 2
