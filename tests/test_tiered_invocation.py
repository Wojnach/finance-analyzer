"""Tests for the tiered Layer 2 invocation architecture.

Covers:
- classify_tier() with various reason combinations
- Tier 1/2 JSON generation (structure and content)
- _extract_triggered_tickers() parsing
- Tier state tracking (last_full_review_time, today_date)
- _build_tier_prompt() for each tier
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from portfolio.trigger import (
    STATE_FILE,
    classify_tier,
    update_tier_state,
    _load_state,
    _save_state,
    _today_str,
    _FULL_REVIEW_MARKET_HOURS,
    _FULL_REVIEW_OFF_HOURS,
)
from portfolio.reporting import (
    write_tiered_summary,
    _write_tier1_summary,
    _write_tier2_summary,
    _macro_headline,
    _portfolio_snapshot,
    TIER1_FILE,
    TIER2_FILE,
)
from portfolio.agent_invocation import (
    _build_tier_prompt,
    TIER_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TriggerTestBase:
    """Base class that backs up and restores trigger state between tests."""

    def setup_method(self):
        self._backup = None
        if STATE_FILE.exists():
            self._backup = STATE_FILE.read_text(encoding="utf-8")
            STATE_FILE.unlink()

    def teardown_method(self):
        if self._backup is not None:
            STATE_FILE.write_text(self._backup, encoding="utf-8")
        elif STATE_FILE.exists():
            STATE_FILE.unlink()


# ---------------------------------------------------------------------------
# classify_tier tests
# ---------------------------------------------------------------------------

class TestClassifyTier(TriggerTestBase):

    def _make_state(self, **overrides):
        """Create a trigger state dict with sensible defaults."""
        state = {
            "last_full_review_time": time.time(),  # recent full review
            "today_date": _today_str(),  # already invoked today
        }
        state.update(overrides)
        return state

    def test_cooldown_returns_tier1(self):
        """Cooldown triggers should classify as Tier 1."""
        state = self._make_state()
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 1

    def test_crypto_checkin_returns_tier1(self):
        """Off-hours crypto check-in should be Tier 1."""
        state = self._make_state()
        reasons = ["crypto check-in (2h)"]
        assert classify_tier(reasons, state=state) == 1

    def test_sentiment_returns_tier1(self):
        """Sentiment triggers should be Tier 1."""
        state = self._make_state()
        reasons = ["RXRX sentiment positive->negative (sustained)"]
        assert classify_tier(reasons, state=state) == 1

    def test_consensus_returns_tier2(self):
        """New consensus triggers should be Tier 2."""
        state = self._make_state()
        reasons = ["MU consensus BUY (79%)"]
        assert classify_tier(reasons, state=state) == 2

    def test_price_move_returns_tier2(self):
        """Price move triggers should be Tier 2."""
        state = self._make_state()
        reasons = ["BTC-USD moved 3.1% up"]
        assert classify_tier(reasons, state=state) == 2

    def test_post_trade_returns_tier2(self):
        """Post-trade reassessment should be Tier 2."""
        state = self._make_state()
        reasons = ["post-trade reassessment"]
        assert classify_tier(reasons, state=state) == 2

    def test_flipped_returns_tier2(self):
        """Sustained flip should be Tier 2."""
        state = self._make_state()
        reasons = ["BTC-USD flipped BUY->SELL (sustained)"]
        assert classify_tier(reasons, state=state) == 2

    def test_fg_crossed_returns_tier3(self):
        """F&G extreme crossing should be Tier 3."""
        state = self._make_state()
        reasons = ["F&G crossed 20 (25->18)"]
        assert classify_tier(reasons, state=state) == 3

    def test_first_of_day_returns_tier3(self):
        """First invocation of the day should be Tier 3."""
        state = self._make_state(today_date="2026-01-01")  # yesterday
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 3

    @mock.patch("portfolio.trigger.datetime")
    def test_periodic_market_hours_returns_tier3(self, mock_dt):
        """2h+ since last full review during market hours -> Tier 3."""
        # Simulate Tuesday 10:00 UTC (market hours)
        fake_now = datetime(2026, 2, 17, 10, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state = self._make_state(
            last_full_review_time=time.time() - 3 * 3600,  # 3h ago
        )
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 3

    @mock.patch("portfolio.trigger.datetime")
    def test_periodic_offhours_returns_tier3(self, mock_dt):
        """4h+ since last full review during off-hours -> Tier 3."""
        # Simulate Saturday 10:00 UTC (weekend)
        fake_now = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state = self._make_state(
            last_full_review_time=time.time() - 5 * 3600,  # 5h ago
        )
        reasons = ["crypto check-in (2h)"]
        assert classify_tier(reasons, state=state) == 3

    def test_no_last_full_review_returns_tier3(self):
        """Missing last_full_review_time should trigger Tier 3."""
        state = {"today_date": _today_str()}  # no last_full_review_time
        reasons = ["cooldown (10min)"]
        # time.time() - 0 will be huge, so this should be T3
        assert classify_tier(reasons, state=state) == 3

    def test_tier2_takes_precedence_over_tier1(self):
        """Mixed reasons: consensus + cooldown -> Tier 2 (highest wins)."""
        state = self._make_state()
        reasons = ["cooldown (10min)", "MU consensus BUY (79%)"]
        assert classify_tier(reasons, state=state) == 2

    def test_tier3_takes_precedence_over_tier2(self):
        """F&G extreme + consensus -> Tier 3 (highest wins)."""
        state = self._make_state()
        reasons = ["MU consensus BUY (79%)", "F&G crossed 20 (25->18)"]
        assert classify_tier(reasons, state=state) == 3


class TestUpdateTierState(TriggerTestBase):

    def test_tier3_updates_last_full_review_time(self):
        """update_tier_state(3) should set last_full_review_time."""
        _save_state({"today_date": _today_str()})
        update_tier_state(3)
        state = _load_state()
        assert "last_full_review_time" in state
        assert time.time() - state["last_full_review_time"] < 5

    def test_tier1_does_not_update_full_review(self):
        """update_tier_state(1) should NOT change last_full_review_time."""
        _save_state({"today_date": _today_str(), "last_full_review_time": 12345})
        update_tier_state(1)
        state = _load_state()
        assert state["last_full_review_time"] == 12345


# ---------------------------------------------------------------------------
# _extract_triggered_tickers tests
# ---------------------------------------------------------------------------

class TestExtractTriggeredTickers:

    def test_consensus_reason(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = ["MU consensus BUY (79%)"]
        assert _extract_triggered_tickers(reasons) == {"MU"}

    def test_price_move_reason(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = ["BTC-USD moved 3.1% up"]
        assert _extract_triggered_tickers(reasons) == {"BTC-USD"}

    def test_flipped_reason(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = ["ETH-USD flipped SELL->BUY (sustained)"]
        assert _extract_triggered_tickers(reasons) == {"ETH-USD"}

    def test_multiple_reasons(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = [
            "MU consensus BUY (79%)",
            "BTC-USD moved 3.1% up",
            "cooldown (10min)",
            "post-trade reassessment",
        ]
        result = _extract_triggered_tickers(reasons)
        assert result == {"MU", "BTC-USD"}

    def test_no_tickers_in_cooldown(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = ["cooldown (10min)", "post-trade reassessment"]
        assert _extract_triggered_tickers(reasons) == set()

    def test_fg_reason_has_no_ticker(self):
        from portfolio.main import _extract_triggered_tickers
        reasons = ["F&G crossed 20 (25->18)"]
        assert _extract_triggered_tickers(reasons) == set()


# ---------------------------------------------------------------------------
# Tiered summary generation tests
# ---------------------------------------------------------------------------

def _make_summary(held_tickers=None, triggered_tickers=None):
    """Build a minimal agent_summary dict for testing."""
    held_tickers = held_tickers or set()
    triggered_tickers = triggered_tickers or set()
    all_tickers = list(held_tickers | triggered_tickers | {"AAPL", "GOOGL", "AMZN", "META", "NVDA"})

    signals = {}
    timeframes = {}
    for i, t in enumerate(all_tickers):
        is_triggered = t in triggered_tickers
        action = "BUY" if is_triggered else "HOLD"
        buy_count = 5 if is_triggered else i % 3
        sell_count = 1 if is_triggered else 0
        signals[t] = {
            "action": action,
            "confidence": 0.7 if is_triggered else 0.5,
            "price_usd": 100 + i * 50,
            "rsi": 50 + i,
            "regime": "trending-up" if is_triggered else "ranging",
            "atr_pct": 1.5,
            "extra": {
                "_buy_count": buy_count,
                "_sell_count": sell_count,
                "_voters": buy_count + sell_count,
                "_total_applicable": 23,
                "_weighted_action": action,
                "_weighted_confidence": 0.6,
                "_votes": {f"signal_{j}": "BUY" if j < buy_count else "SELL"
                           for j in range(buy_count + sell_count)},
            },
        }
        tf_labels = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
        timeframes[t] = [
            {"horizon": label, "action": "BUY" if is_triggered else "HOLD"}
            for label in tf_labels
        ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_reasons": ["test"],
        "fx_rate": 9.07,
        "signals": signals,
        "timeframes": timeframes,
        "fear_greed": {},
        "macro": {
            "dxy": {"value": 97.8, "change_5d_pct": 0.32},
            "treasury": {"10y": 4.09, "10y_change_5d_pct": 0.15},
            "fed": {"days_until": 22},
        },
        "portfolio": {
            "total_sek": 500000,
            "pnl_pct": 0,
            "cash_sek": 500000,
            "holdings": {},
        },
    }


class TestTier1Summary:

    def test_tier1_has_correct_structure(self, tmp_path):
        """Tier 1 JSON should have the expected top-level keys."""
        summary = _make_summary(held_tickers={"MU"})
        with mock.patch("portfolio.reporting.TIER1_FILE", tmp_path / "t1.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}), \
             mock.patch("portfolio.reporting._portfolio_snapshot", return_value={"cash_sek": 500000, "total_sek": 500000, "pnl_pct": 0}):
            _write_tier1_summary(summary)
            t1 = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))

        assert t1["tier"] == 1
        assert "held_positions" in t1
        assert "all_prices" in t1
        assert "macro_headline" in t1
        assert "portfolio_patient" in t1
        assert "portfolio_bold" in t1

    def test_tier1_only_has_held_tickers(self, tmp_path):
        """Tier 1 held_positions should only contain held tickers."""
        summary = _make_summary(held_tickers={"MU"})
        with mock.patch("portfolio.reporting.TIER1_FILE", tmp_path / "t1.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}), \
             mock.patch("portfolio.reporting._portfolio_snapshot", return_value={"cash_sek": 500000, "total_sek": 500000, "pnl_pct": 0}):
            _write_tier1_summary(summary)
            t1 = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))

        assert "MU" in t1["held_positions"]
        # Other tickers should NOT be in held_positions
        assert len(t1["held_positions"]) == 1

    def test_tier1_all_prices_has_all_tickers(self, tmp_path):
        """Tier 1 all_prices should include every ticker."""
        summary = _make_summary(held_tickers={"MU"})
        with mock.patch("portfolio.reporting.TIER1_FILE", tmp_path / "t1.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}), \
             mock.patch("portfolio.reporting._portfolio_snapshot", return_value={"cash_sek": 500000, "total_sek": 500000, "pnl_pct": 0}):
            _write_tier1_summary(summary)
            t1 = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))

        assert len(t1["all_prices"]) == len(summary["signals"])

    def test_tier1_no_held_positions(self, tmp_path):
        """Tier 1 with no held positions should have empty held_positions."""
        summary = _make_summary()
        with mock.patch("portfolio.reporting.TIER1_FILE", tmp_path / "t1.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value=set()), \
             mock.patch("portfolio.reporting._portfolio_snapshot", return_value={"cash_sek": 500000, "total_sek": 500000, "pnl_pct": 0}):
            _write_tier1_summary(summary)
            t1 = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))

        assert t1["held_positions"] == {}


class TestTier2Summary:

    def test_tier2_has_correct_structure(self, tmp_path):
        """Tier 2 JSON should have signals, timeframes, and macro."""
        summary = _make_summary(held_tickers={"MU"}, triggered_tickers={"NVDA"})
        with mock.patch("portfolio.reporting.TIER2_FILE", tmp_path / "t2.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}):
            _write_tier2_summary(summary, triggered_tickers={"NVDA"})
            t2 = json.loads((tmp_path / "t2.json").read_text(encoding="utf-8"))

        assert t2["tier"] == 2
        assert "signals" in t2
        assert "timeframes" in t2
        assert "macro" in t2

    def test_tier2_full_detail_for_held_and_triggered(self, tmp_path):
        """Held and triggered tickers should have full _votes dict."""
        summary = _make_summary(held_tickers={"MU"}, triggered_tickers={"NVDA"})
        with mock.patch("portfolio.reporting.TIER2_FILE", tmp_path / "t2.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}):
            _write_tier2_summary(summary, triggered_tickers={"NVDA"})
            t2 = json.loads((tmp_path / "t2.json").read_text(encoding="utf-8"))

        # MU (held) and NVDA (triggered) should have _votes in extra
        for ticker in ("MU", "NVDA"):
            assert ticker in t2["signals"]
            assert "_votes" in t2["signals"][ticker].get("extra", {})

    def test_tier2_medium_detail_for_top5(self, tmp_path):
        """Top 5 non-held non-triggered tickers should have _vote_detail string."""
        summary = _make_summary(held_tickers={"MU"}, triggered_tickers={"NVDA"})
        # Ensure enough remaining tickers with active voters
        for t in ("AAPL", "GOOGL", "AMZN", "META"):
            summary["signals"][t]["extra"]["_buy_count"] = 3
            summary["signals"][t]["extra"]["_sell_count"] = 1
            summary["signals"][t]["extra"]["_voters"] = 4

        with mock.patch("portfolio.reporting.TIER2_FILE", tmp_path / "t2.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}):
            _write_tier2_summary(summary, triggered_tickers={"NVDA"})
            t2 = json.loads((tmp_path / "t2.json").read_text(encoding="utf-8"))

        # Check that at least one medium-detail ticker has _vote_detail
        medium_found = False
        for ticker, sig in t2["signals"].items():
            if ticker in ("MU", "NVDA"):
                continue
            extra = sig.get("extra", {})
            if "_vote_detail" in extra:
                medium_found = True
                assert "_votes" not in extra  # should be collapsed
                break
        assert medium_found, "Expected at least one medium-detail ticker with _vote_detail"

    def test_tier2_priceonly_for_remaining(self, tmp_path):
        """Remaining tickers should have only action and price_usd."""
        summary = _make_summary(held_tickers={"MU"}, triggered_tickers={"NVDA"})
        with mock.patch("portfolio.reporting.TIER2_FILE", tmp_path / "t2.json"), \
             mock.patch("portfolio.reporting._get_held_tickers", return_value={"MU"}):
            _write_tier2_summary(summary, triggered_tickers={"NVDA"})
            t2 = json.loads((tmp_path / "t2.json").read_text(encoding="utf-8"))

        # Find a price-only ticker (not held, not triggered, not top 5)
        full_detail = {"MU", "NVDA"}
        for ticker, sig in t2["signals"].items():
            if ticker in full_detail:
                continue
            if "extra" not in sig:
                # Price-only: should just have action and price_usd
                assert "action" in sig
                assert "price_usd" in sig
                assert len(sig) == 2
                break


class TestWriteTieredSummary:

    def test_tier1_calls_write_tier1(self, tmp_path):
        summary = _make_summary()
        with mock.patch("portfolio.reporting._write_tier1_summary") as m1:
            write_tiered_summary(summary, tier=1)
            m1.assert_called_once_with(summary)

    def test_tier2_calls_write_tier2(self, tmp_path):
        summary = _make_summary()
        with mock.patch("portfolio.reporting._write_tier2_summary") as m2:
            write_tiered_summary(summary, tier=2, triggered_tickers={"MU"})
            m2.assert_called_once_with(summary, {"MU"})

    def test_tier3_does_not_write_extra_file(self, tmp_path):
        """Tier 3 should not create any extra files (uses compact summary)."""
        summary = _make_summary()
        with mock.patch("portfolio.reporting._write_tier1_summary") as m1, \
             mock.patch("portfolio.reporting._write_tier2_summary") as m2:
            write_tiered_summary(summary, tier=3)
            m1.assert_not_called()
            m2.assert_not_called()


# ---------------------------------------------------------------------------
# _macro_headline tests
# ---------------------------------------------------------------------------

class TestMacroHeadline:

    def test_full_macro(self):
        summary = {
            "macro": {
                "dxy": {"value": 97.8, "change_5d_pct": 0.32},
                "treasury": {"10y": 4.09, "10y_change_5d_pct": -0.15},
                "fed": {"days_until": 22},
            },
            "fear_greed": {"BTC-USD": {"value": 5}},
        }
        headline = _macro_headline(summary)
        assert "DXY" in headline
        assert "10Y" in headline
        assert "FOMC" in headline
        assert "F&G" in headline

    def test_empty_macro(self):
        headline = _macro_headline({})
        assert headline == ""


# ---------------------------------------------------------------------------
# _build_tier_prompt tests
# ---------------------------------------------------------------------------

class TestBuildTierPrompt:

    def test_tier1_prompt_mentions_quick_check(self):
        prompt = _build_tier_prompt(1, ["cooldown (10min)"])
        assert "QUICK CHECK" in prompt
        assert "agent_context_t1.json" in prompt
        assert "Do NOT analyze all tickers" in prompt

    def test_tier2_prompt_mentions_signal_analysis(self):
        prompt = _build_tier_prompt(2, ["MU consensus BUY (79%)"])
        assert "SIGNAL ANALYSIS" in prompt
        assert "agent_context_t2.json" in prompt
        assert "BOTH strategies" in prompt

    def test_tier3_prompt_is_full_review(self):
        prompt = _build_tier_prompt(3, ["periodic review"])
        assert "agent_summary_compact.json" in prompt
        assert "CLAUDE.md" in prompt

    def test_prompt_includes_reasons(self):
        reasons = ["MU consensus BUY (79%)", "cooldown (10min)"]
        prompt = _build_tier_prompt(2, reasons)
        assert "MU consensus BUY" in prompt


# ---------------------------------------------------------------------------
# TIER_CONFIG tests
# ---------------------------------------------------------------------------

class TestTierConfig:

    def test_tier1_config(self):
        cfg = TIER_CONFIG[1]
        assert cfg["max_turns"] == 15
        assert cfg["timeout"] == 120
        assert cfg["label"] == "QUICK CHECK"

    def test_tier2_config(self):
        cfg = TIER_CONFIG[2]
        assert cfg["max_turns"] == 25
        assert cfg["timeout"] == 300
        assert cfg["label"] == "SIGNAL ANALYSIS"

    def test_tier3_config(self):
        cfg = TIER_CONFIG[3]
        assert cfg["max_turns"] == 40
        assert cfg["timeout"] == 900
        assert cfg["label"] == "FULL REVIEW"
