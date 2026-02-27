"""Thorough tests for classify_tier(), update_tier_state(), and _check_recent_trade().

Complements test_trigger_core.py (basic behavior) and test_trigger_edge_cases.py (edge cases
for check_triggers). This file focuses on deeper coverage of the tier classification and
trade detection functions added Feb 23.

Tests organized by function:
  classify_tier (17 tests):
    - Tier 3: F&G crossed reasons, periodic review market/off-hours, first-of-day,
      no last_full_review_time in state, boundary thresholds (exactly 2h/4h),
      just under thresholds, missing today_date in state, empty state dict
    - Tier 2: "flipped" pattern, mixed tier2+tier1 reasons, all four tier2 patterns
    - Tier 1: pure sentiment, empty reasons list, cooldown only
    - Priority: tier 3 beats tier 2, tier 2 beats tier 1

  update_tier_state (5 tests):
    - Tier 3 sets last_full_review_time, tier 1/2 do not, preserves other keys,
      works when state file does not exist, tier 3 called twice updates timestamp

  _check_recent_trade (6 tests):
    - No portfolio files, no new transactions, new patient transaction, new bold
      transaction, both portfolios have new transactions, empty transactions array
"""

import json
import time
from datetime import datetime, timezone
from unittest import mock

import pytest

import portfolio.trigger as trigger_mod
from portfolio.trigger import (
    classify_tier,
    update_tier_state,
    _check_recent_trade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_state_files(tmp_path, monkeypatch):
    """Redirect STATE_FILE, PORTFOLIO_FILE, PORTFOLIO_BOLD_FILE to tmp_path."""
    state_file = tmp_path / "trigger_state.json"
    portfolio_file = tmp_path / "portfolio_state.json"
    portfolio_bold_file = tmp_path / "portfolio_state_bold.json"

    monkeypatch.setattr(trigger_mod, "STATE_FILE", state_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_FILE", portfolio_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_BOLD_FILE", portfolio_bold_file)

    return {
        "state_file": state_file,
        "portfolio_file": portfolio_file,
        "portfolio_bold_file": portfolio_bold_file,
    }


def _seed_state(state_file, data):
    """Write trigger state JSON directly."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(data), encoding="utf-8")


def _write_portfolio(pf_file, transactions=None):
    """Write a minimal portfolio file with given transactions."""
    pf_file.write_text(
        json.dumps({"transactions": transactions or []}),
        encoding="utf-8",
    )


def _mock_market_hours():
    """Return a started mock that sets datetime.now to Tuesday 12:00 UTC (market hours)."""
    fake_now = datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc)
    m = mock.patch("portfolio.trigger.datetime")
    mock_dt = m.start()
    mock_dt.now.return_value = fake_now
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    return m, fake_now


def _mock_offhours():
    """Return a started mock that sets datetime.now to Saturday 03:00 UTC (off-hours)."""
    fake_now = datetime(2026, 2, 14, 3, 0, 0, tzinfo=timezone.utc)
    m = mock.patch("portfolio.trigger.datetime")
    mock_dt = m.start()
    mock_dt.now.return_value = fake_now
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    return m, fake_now


def _base_state(last_full_review_offset=0):
    """Build a state dict that does NOT trigger tier 3 by default.

    last_full_review_offset: seconds before now for last_full_review_time.
    Default 0 means "just now" (no periodic review trigger).
    """
    return {
        "last_full_review_time": time.time() - last_full_review_offset,
        "today_date": trigger_mod._today_str(),
    }


# ===========================================================================
# classify_tier tests
# ===========================================================================


class TestClassifyTierThree:
    """Tier 3 classification: periodic review, F&G, first-of-day."""

    def test_fg_crossed_20_returns_tier_3(self):
        """F&G crossed 20 threshold should return tier 3."""
        state = _base_state()
        reasons = ["F&G crossed 20 (25->18)"]
        assert classify_tier(reasons, state=state) == 3

    def test_fg_crossed_80_returns_tier_3(self):
        """F&G crossed 80 threshold should return tier 3."""
        state = _base_state()
        reasons = ["F&G crossed 80 (75->85)"]
        assert classify_tier(reasons, state=state) == 3

    def test_fg_among_other_reasons_still_tier_3(self):
        """F&G crossing mixed with other reasons should still return tier 3."""
        state = _base_state()
        reasons = [
            "cooldown (10min)",
            "F&G crossed 20 (30->15)",
            "BTC-USD sentiment positive->negative (sustained)",
        ]
        assert classify_tier(reasons, state=state) == 3

    def test_periodic_review_market_hours_at_exactly_2h(self):
        """During market hours, exactly 2h since last full review should return tier 3."""
        m, _ = _mock_market_hours()
        try:
            state = {
                "last_full_review_time": time.time() - 7200,  # exactly 2h ago
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["cooldown (10min)"]
            assert classify_tier(reasons, state=state) == 3
        finally:
            m.stop()

    def test_periodic_review_market_hours_just_under_2h(self):
        """During market hours, just under 2h should NOT return tier 3."""
        m, _ = _mock_market_hours()
        try:
            state = {
                "last_full_review_time": time.time() - 7190,  # 10s short of 2h
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["cooldown (10min)"]
            assert classify_tier(reasons, state=state) == 1
        finally:
            m.stop()

    def test_periodic_review_offhours_at_exactly_4h(self):
        """During off-hours, exactly 4h since last full review should return tier 3."""
        m, _ = _mock_offhours()
        try:
            state = {
                "last_full_review_time": time.time() - 14400,  # exactly 4h
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["crypto check-in (2h)"]
            assert classify_tier(reasons, state=state) == 3
        finally:
            m.stop()

    def test_periodic_review_offhours_just_under_4h(self):
        """During off-hours, just under 4h should NOT return tier 3 for periodic review."""
        m, _ = _mock_offhours()
        try:
            state = {
                "last_full_review_time": time.time() - 14390,  # 10s short of 4h
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["crypto check-in (2h)"]
            assert classify_tier(reasons, state=state) == 1
        finally:
            m.stop()

    def test_first_of_day_missing_today_date_in_state(self):
        """When state has no today_date key, it should be treated as first-of-day -> tier 3."""
        state = {"last_full_review_time": time.time()}
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 3

    def test_first_of_day_yesterday_date(self):
        """When today_date in state is yesterday, should be first-of-day -> tier 3."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": "2020-01-01",
        }
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 3

    def test_no_last_full_review_time_defaults_to_zero(self):
        """When state has no last_full_review_time, it defaults to 0 (epoch).

        This means hours_since is always huge, so tier 3 always triggers.
        """
        m, _ = _mock_market_hours()
        try:
            state = {"today_date": trigger_mod._today_str()}
            reasons = ["cooldown (10min)"]
            assert classify_tier(reasons, state=state) == 3
        finally:
            m.stop()

    def test_empty_state_dict_returns_tier_3(self):
        """Empty state dict triggers tier 3 (no last_full_review_time, no today_date)."""
        m, _ = _mock_market_hours()
        try:
            assert classify_tier(["cooldown (10min)"], state={}) == 3
        finally:
            m.stop()


class TestClassifyTierTwo:
    """Tier 2 classification: consensus, moved, post-trade, flipped."""

    def test_consensus_reason_returns_tier_2(self):
        state = _base_state()
        reasons = ["BTC-USD consensus BUY (80%)"]
        assert classify_tier(reasons, state=state) == 2

    def test_moved_reason_returns_tier_2(self):
        state = _base_state()
        reasons = ["ETH-USD moved 3.5% up"]
        assert classify_tier(reasons, state=state) == 2

    def test_post_trade_reason_returns_tier_2(self):
        state = _base_state()
        reasons = ["post-trade reassessment"]
        assert classify_tier(reasons, state=state) == 2

    def test_flipped_reason_returns_tier_2(self):
        """The 'flipped' keyword (sustained signal flip) should classify as tier 2."""
        state = _base_state()
        reasons = ["BTC-USD flipped BUY->HOLD (sustained)"]
        assert classify_tier(reasons, state=state) == 2

    def test_mixed_tier2_and_tier1_returns_tier_2(self):
        """When reasons contain both tier 2 and tier 1 patterns, tier 2 wins."""
        state = _base_state()
        reasons = [
            "cooldown (10min)",
            "BTC-USD consensus BUY (80%)",
            "ETH-USD sentiment positive->negative (sustained)",
        ]
        assert classify_tier(reasons, state=state) == 2

    def test_all_four_tier2_patterns_recognized(self):
        """All four tier 2 keywords are recognized in various reason strings."""
        state = _base_state()
        for keyword in ["consensus", "moved", "post-trade", "flipped"]:
            reasons = [f"some ticker {keyword} something"]
            assert classify_tier(reasons, state=state) == 2, (
                f"'{keyword}' should classify as tier 2"
            )


class TestClassifyTierOne:
    """Tier 1 classification: cooldown, sentiment, and other noise."""

    def test_cooldown_only_returns_tier_1(self):
        state = _base_state()
        reasons = ["cooldown (10min)"]
        assert classify_tier(reasons, state=state) == 1

    def test_crypto_checkin_returns_tier_1(self):
        state = _base_state()
        reasons = ["crypto check-in (2h)"]
        assert classify_tier(reasons, state=state) == 1

    def test_sentiment_reason_returns_tier_1(self):
        state = _base_state()
        reasons = ["SOUN sentiment negative->positive (sustained)"]
        assert classify_tier(reasons, state=state) == 1

    def test_empty_reasons_returns_tier_1(self):
        """Empty reasons list should return tier 1 (no patterns match)."""
        state = _base_state()
        assert classify_tier([], state=state) == 1

    def test_unrecognized_reason_returns_tier_1(self):
        """An unrecognized reason string falls through to tier 1."""
        state = _base_state()
        reasons = ["some unknown trigger reason"]
        assert classify_tier(reasons, state=state) == 1


class TestClassifyTierPriority:
    """Tier priority: 3 > 2 > 1."""

    def test_tier_3_beats_tier_2(self):
        """F&G (tier 3) combined with consensus (tier 2) should return tier 3."""
        state = _base_state()
        reasons = [
            "BTC-USD consensus BUY (80%)",
            "F&G crossed 20 (25->15)",
        ]
        assert classify_tier(reasons, state=state) == 3

    def test_tier_2_beats_tier_1(self):
        """Consensus (tier 2) combined with cooldown (tier 1) should return tier 2."""
        state = _base_state()
        reasons = [
            "cooldown (10min)",
            "BTC-USD consensus SELL (70%)",
        ]
        assert classify_tier(reasons, state=state) == 2

    def test_tier_3_periodic_beats_tier_2_reasons(self):
        """Periodic review (tier 3) should beat tier 2 consensus reasons."""
        m, _ = _mock_market_hours()
        try:
            state = {
                "last_full_review_time": time.time() - 8000,  # >2h ago
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["BTC-USD consensus BUY (80%)"]
            assert classify_tier(reasons, state=state) == 3
        finally:
            m.stop()


class TestClassifyTierStateFromFile:
    """Test classify_tier when state=None (reads from file)."""

    def test_state_none_loads_from_file(self, isolate_state_files):
        """When state=None, classify_tier should load state from STATE_FILE."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        })
        reasons = ["BTC-USD consensus BUY (80%)"]
        assert classify_tier(reasons, state=None) == 2

    def test_state_none_no_file_returns_tier_3(self, isolate_state_files):
        """When state=None and STATE_FILE does not exist, empty state -> tier 3."""
        m, _ = _mock_market_hours()
        try:
            # state_file does not exist -> _load_state() returns {}
            reasons = ["cooldown (10min)"]
            assert classify_tier(reasons, state=None) == 3
        finally:
            m.stop()


# ===========================================================================
# update_tier_state tests
# ===========================================================================


class TestUpdateTierState:
    """Tests for update_tier_state()."""

    def test_tier_3_sets_last_full_review_time(self, isolate_state_files):
        """update_tier_state(3) should set last_full_review_time to approximately now."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 0})

        before = time.time()
        update_tier_state(3)
        after = time.time()

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert before <= state["last_full_review_time"] <= after

    def test_tier_1_does_not_change_review_time(self, isolate_state_files):
        """update_tier_state(1) should NOT update last_full_review_time."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 42.0})

        update_tier_state(1)

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_full_review_time"] == 42.0

    def test_tier_2_does_not_change_review_time(self, isolate_state_files):
        """update_tier_state(2) should NOT update last_full_review_time."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 99.0})

        update_tier_state(2)

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_full_review_time"] == 99.0

    def test_preserves_other_state_keys(self, isolate_state_files):
        """update_tier_state should not destroy other keys in the state file."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {
            "last_full_review_time": 0,
            "last_trigger_time": 12345.0,
            "today_date": "2026-02-26",
            "triggered_consensus": {"BTC-USD": "BUY"},
            "sustained_counts": {"BTC-USD": {"action": "BUY", "count": 2}},
        })

        update_tier_state(3)

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_trigger_time"] == 12345.0
        assert state["today_date"] == "2026-02-26"
        assert state["triggered_consensus"]["BTC-USD"] == "BUY"
        assert state["sustained_counts"]["BTC-USD"]["count"] == 2
        # And the review time was updated
        assert state["last_full_review_time"] > 0

    def test_works_when_no_state_file(self, isolate_state_files):
        """update_tier_state should work when state file does not exist yet."""
        sf = isolate_state_files["state_file"]
        assert not sf.exists()

        before = time.time()
        update_tier_state(3)
        after = time.time()

        assert sf.exists()
        state = json.loads(sf.read_text(encoding="utf-8"))
        assert before <= state["last_full_review_time"] <= after

    def test_tier_3_called_twice_updates_timestamp(self, isolate_state_files):
        """Calling update_tier_state(3) twice should update the timestamp each time."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 0})

        update_tier_state(3)
        state1 = json.loads(sf.read_text(encoding="utf-8"))
        t1 = state1["last_full_review_time"]

        # Small delay to ensure different timestamp
        time.sleep(0.01)

        update_tier_state(3)
        state2 = json.loads(sf.read_text(encoding="utf-8"))
        t2 = state2["last_full_review_time"]

        assert t2 >= t1


# ===========================================================================
# _check_recent_trade tests
# ===========================================================================


class TestCheckRecentTrade:
    """Tests for _check_recent_trade()."""

    def test_no_portfolio_files_returns_false(self, isolate_state_files):
        """When neither portfolio file exists, should return False."""
        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
        assert _check_recent_trade(state) is False

    def test_no_new_transactions_returns_false(self, isolate_state_files):
        """When transaction count has not changed, should return False."""
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[{"action": "BUY"}])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {"patient": 1},
        }
        assert _check_recent_trade(state) is False

    def test_new_patient_transaction_returns_true(self, isolate_state_files):
        """When patient portfolio has a new transaction, should return True."""
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[
            {"action": "BUY"},
            {"action": "SELL"},
        ])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {"patient": 1},
        }
        result = _check_recent_trade(state)
        assert result is True
        assert state["last_checked_tx_count"]["patient"] == 2

    def test_new_bold_transaction_returns_true(self, isolate_state_files):
        """When bold portfolio has a new transaction, should return True."""
        pf_bold = isolate_state_files["portfolio_bold_file"]
        _write_portfolio(pf_bold, transactions=[
            {"action": "BUY"},
            {"action": "SELL"},
            {"action": "BUY"},
        ])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {"bold": 2},
        }
        result = _check_recent_trade(state)
        assert result is True
        assert state["last_checked_tx_count"]["bold"] == 3

    def test_both_portfolios_have_new_transactions(self, isolate_state_files):
        """When both portfolios have new transactions, should return True."""
        pf = isolate_state_files["portfolio_file"]
        pf_bold = isolate_state_files["portfolio_bold_file"]

        _write_portfolio(pf, transactions=[{"action": "BUY"}, {"action": "SELL"}])
        _write_portfolio(pf_bold, transactions=[{"action": "BUY"}])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {"patient": 1, "bold": 0},
        }
        result = _check_recent_trade(state)
        assert result is True
        assert state["last_checked_tx_count"]["patient"] == 2
        assert state["last_checked_tx_count"]["bold"] == 1

    def test_empty_transactions_baseline(self, isolate_state_files):
        """A portfolio with empty transactions should set baseline count to 0."""
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {},
        }
        # First check: no prev count -> defaults to current (0), so 0 == 0 -> False
        result = _check_recent_trade(state)
        assert result is False
        assert state["last_checked_tx_count"]["patient"] == 0

    def test_first_check_baselines_to_current_count(self, isolate_state_files):
        """First check with existing transactions should NOT detect them as new.

        The prev_count defaults to current_count when no prior entry exists.
        """
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[
            {"action": "BUY"},
            {"action": "SELL"},
            {"action": "BUY"},
        ])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {},
        }
        result = _check_recent_trade(state)
        assert result is False
        # Baseline is now set to 3
        assert state["last_checked_tx_count"]["patient"] == 3

    def test_malformed_json_returns_false(self, isolate_state_files):
        """Malformed JSON in portfolio file should not crash, returns False."""
        pf = isolate_state_files["portfolio_file"]
        pf.write_text("{{invalid json", encoding="utf-8")

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {},
        }
        result = _check_recent_trade(state)
        assert result is False

    def test_updates_state_dict_in_place(self, isolate_state_files):
        """_check_recent_trade should update the state dict passed to it."""
        pf = isolate_state_files["portfolio_file"]
        pf_bold = isolate_state_files["portfolio_bold_file"]

        _write_portfolio(pf, transactions=[{"action": "BUY"}])
        _write_portfolio(pf_bold, transactions=[{"action": "BUY"}, {"action": "SELL"}])

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {"patient": 0, "bold": 1},
        }
        _check_recent_trade(state)

        # Both should be updated
        assert state["last_checked_tx_count"]["patient"] == 1
        assert state["last_checked_tx_count"]["bold"] == 2

    def test_missing_transactions_key_treated_as_empty(self, isolate_state_files):
        """Portfolio file without 'transactions' key should be treated as 0 transactions."""
        pf = isolate_state_files["portfolio_file"]
        pf.write_text(json.dumps({"cash_sek": 500000}), encoding="utf-8")

        state = {
            "last_trigger_time": time.time(),
            "last_checked_tx_count": {},
        }
        result = _check_recent_trade(state)
        assert result is False
        assert state["last_checked_tx_count"]["patient"] == 0
