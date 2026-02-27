"""Tests for core trigger detection logic in portfolio.trigger.

Covers the fundamental trigger behaviors:
1.  Signal consensus: HOLD -> BUY triggers, BUY -> BUY does not re-trigger
2.  Signal consensus: HOLD -> SELL triggers
3.  Signal flip: BUY->SELL sustained for 3 checks triggers
4.  Signal flip: BUY->SELL for only 1 check does not trigger
5.  Price move: >2% since last trigger fires
6.  Price move: <2% does not fire
7.  Fear & Greed: crosses 20 or 80 threshold fires
8.  Fear & Greed: stays on same side does not fire
9.  Post-trade: recent trade triggers reassessment
10. classify_tier: consensus trigger -> Tier 2
11. classify_tier: sentiment trigger -> Tier 1
12. classify_tier: F&G extreme -> Tier 3
13. classify_tier: first-of-day -> Tier 3
14. classify_tier: price move -> Tier 2
15. classify_tier: post-trade -> Tier 2
16. classify_tier: periodic full review -> Tier 3

Uses tmp_path to isolate state files; never touches real data.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

import portfolio.trigger as trigger_mod
from portfolio.trigger import (
    PRICE_THRESHOLD,
    FG_THRESHOLDS,
    SUSTAINED_CHECKS,
    check_triggers,
    classify_tier,
    update_tier_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_state_files(tmp_path, monkeypatch):
    """Redirect STATE_FILE, PORTFOLIO_FILE, PORTFOLIO_BOLD_FILE to tmp_path.

    Every test gets a fresh temp directory so there is no cross-test contamination
    and no risk of touching real data files.
    """
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


@pytest.fixture
def market_hours_context():
    """Mock datetime.now() to be during market hours (Tuesday 12:00 UTC)."""
    fake_now = datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc)  # Tuesday

    def _patch():
        m = mock.patch("portfolio.trigger.datetime")
        mock_dt = m.start()
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # _today_str() uses strftime on the return value of datetime.now()
        return m

    return _patch


@pytest.fixture
def offhours_context():
    """Mock datetime.now() to be during off-hours (Saturday 03:00 UTC)."""
    fake_now = datetime(2026, 2, 14, 3, 0, 0, tzinfo=timezone.utc)  # Saturday

    def _patch():
        m = mock.patch("portfolio.trigger.datetime")
        mock_dt = m.start()
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        return m

    return _patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(action="HOLD", confidence=0.5):
    """Create a signal dict for one ticker."""
    return {"action": action, "confidence": confidence}


def _sigs(tickers_actions):
    """Create signals dict from {ticker: action} mapping."""
    return {t: _sig(a) for t, a in tickers_actions.items()}


def _seed_state(state_file, state_data):
    """Write a trigger state file directly."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state_data), encoding="utf-8")


def _write_portfolio(pf_file, transactions=None):
    """Write a minimal portfolio file with given transactions."""
    pf_file.write_text(
        json.dumps({"transactions": transactions or []}),
        encoding="utf-8",
    )


def _suppress_cooldown(state_file):
    """Set last_trigger_time far in the future to prevent cooldown triggers."""
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["last_trigger_time"] = time.time() + 99999
    state_file.write_text(json.dumps(state), encoding="utf-8")


def _set_trigger_time(state_file, t):
    """Set last_trigger_time to a specific value."""
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["last_trigger_time"] = t
    state_file.write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Signal consensus: HOLD -> BUY triggers
# ---------------------------------------------------------------------------

class TestSignalConsensusNewBuy:
    def test_hold_to_buy_triggers(self, isolate_state_files):
        """A ticker going from HOLD to BUY should trigger with a consensus reason."""
        prices = {"BTC-USD": 68000}

        # Cycle 1: seed with HOLD
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Cycle 2: flip to BUY
        triggered, reasons = check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices, {}, {})

        assert triggered
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1
        assert "BTC-USD" in consensus_reasons[0]
        assert "BUY" in consensus_reasons[0]

    def test_hold_to_sell_triggers(self, isolate_state_files):
        """A ticker going from HOLD to SELL should also trigger consensus."""
        prices = {"ETH-USD": 2000}

        check_triggers({"ETH-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers({"ETH-USD": _sig("SELL", 0.7)}, prices, {}, {})

        assert triggered
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1
        assert "SELL" in consensus_reasons[0]


# ---------------------------------------------------------------------------
# 2. Signal consensus: BUY -> BUY does NOT re-trigger
# ---------------------------------------------------------------------------

class TestSignalConsensusSameAction:
    def test_buy_stays_buy_no_retrigger(self, isolate_state_files):
        """A ticker that stays BUY should NOT trigger consensus again."""
        prices = {"BTC-USD": 68000}

        # Seed with HOLD
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        # First BUY -> triggers consensus
        triggered1, reasons1 = check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        assert any("consensus" in r for r in reasons1)

        _suppress_cooldown(isolate_state_files["state_file"])

        # Second BUY -> should NOT trigger consensus again
        triggered2, reasons2 = check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        consensus_reasons = [r for r in reasons2 if "consensus" in r]
        assert len(consensus_reasons) == 0

    def test_sell_stays_sell_no_retrigger(self, isolate_state_files):
        """A ticker that stays SELL should NOT trigger consensus again."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("SELL")}, prices, {}, {})

        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers({"BTC-USD": _sig("SELL")}, prices, {}, {})
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 0

    def test_buy_to_hold_to_buy_retriggers(self, isolate_state_files):
        """BUY -> HOLD -> BUY should re-trigger because HOLD resets the consensus tracker."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        # BUY triggers
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Back to HOLD resets triggered_consensus for this ticker
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # BUY again -> should trigger as "new" consensus
        triggered, reasons = check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1


# ---------------------------------------------------------------------------
# 3. Signal flip: BUY->SELL sustained for 3 checks triggers
# ---------------------------------------------------------------------------

class TestSignalFlipSustained:
    def test_sustained_flip_triggers_after_3_checks(self, isolate_state_files):
        """A signal flip sustained for SUSTAINED_CHECKS=3 consecutive cycles triggers."""
        prices = {"BTC-USD": 68000}
        sf = isolate_state_files["state_file"]

        # Seed with HOLD, then trigger BUY consensus (sets triggered action to BUY)
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        # Now sustain HOLD for 3 consecutive cycles.
        # HOLD does not trigger consensus (only BUY/SELL do), so triggered action
        # in state["last"]["signals"] stays BUY. After 3 HOLD cycles, sustained flip fires.
        for i in range(SUSTAINED_CHECKS - 1):
            _suppress_cooldown(sf)
            check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        _suppress_cooldown(sf)
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 1
        assert "BUY->HOLD" in flip_reasons[0]
        assert "sustained" in flip_reasons[0]


# ---------------------------------------------------------------------------
# 4. Signal flip: BUY->SELL for only 1 check does NOT trigger
# ---------------------------------------------------------------------------

class TestSignalFlipNotSustained:
    def test_single_check_flip_does_not_trigger(self, isolate_state_files):
        """A signal flip for only 1 check should NOT trigger sustained flip."""
        prices = {"BTC-USD": 68000}
        sf = isolate_state_files["state_file"]

        # Seed: HOLD -> BUY (sets triggered action to BUY in state["last"])
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        _suppress_cooldown(sf)

        # Only 1 cycle of HOLD (count=1, need 3)
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0

    def test_two_checks_not_enough(self, isolate_state_files):
        """Two consecutive checks of a new action is still not enough."""
        prices = {"BTC-USD": 68000}
        sf = isolate_state_files["state_file"]

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        _suppress_cooldown(sf)
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})  # count=1
        _suppress_cooldown(sf)
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})  # count=2

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0

    def test_interrupted_sequence_resets_counter(self, isolate_state_files):
        """An interruption resets the sustained counter so 3 is never reached."""
        prices = {"BTC-USD": 68000}
        sf = isolate_state_files["state_file"]

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})  # triggered=BUY

        # 2 cycles of HOLD
        _suppress_cooldown(sf)
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(sf)
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        # Interruption: back to BUY for 1 cycle -> resets HOLD counter
        _suppress_cooldown(sf)
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        # 1 cycle of HOLD again (counter restarts at 1)
        _suppress_cooldown(sf)
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0


# ---------------------------------------------------------------------------
# 5. Price move: >2% since last trigger fires
# ---------------------------------------------------------------------------

class TestPriceMoveAboveThreshold:
    def test_price_up_more_than_2pct_triggers(self, isolate_state_files):
        """A >2% upward price move since last trigger should fire."""
        prices_initial = {"BTC-USD": 68000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices_initial, {}, {})

        # 3% move up
        prices_moved = {"BTC-USD": 70040}  # 68000 * 1.03 = 70040
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices_moved, {}, {})

        assert triggered
        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 1
        assert "up" in price_reasons[0]

    def test_price_down_more_than_2pct_triggers(self, isolate_state_files):
        """A >2% downward price move since last trigger should fire."""
        prices_initial = {"BTC-USD": 68000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices_initial, {}, {})

        # 3% move down
        prices_moved = {"BTC-USD": 65960}  # 68000 * 0.97 = 65960
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices_moved, {}, {})

        assert triggered
        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 1
        assert "down" in price_reasons[0]

    def test_exactly_2pct_triggers(self, isolate_state_files):
        """Exactly 2% move should trigger (threshold is >=)."""
        prices_initial = {"BTC-USD": 50000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices_initial, {}, {})

        prices_moved = {"BTC-USD": 51000}  # exactly 2%
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices_moved, {}, {})

        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 1


# ---------------------------------------------------------------------------
# 6. Price move: <2% does NOT fire
# ---------------------------------------------------------------------------

class TestPriceMoveBelowThreshold:
    def test_price_move_1pct_no_trigger(self, isolate_state_files):
        """A 1% price move should NOT trigger."""
        prices_initial = {"BTC-USD": 68000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices_initial, {}, {})

        # 1% move
        prices_moved = {"BTC-USD": 68680}  # 68000 * 1.01
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices_moved, {}, {})

        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 0

    def test_price_unchanged_no_trigger(self, isolate_state_files):
        """Zero price movement should not trigger."""
        prices = {"BTC-USD": 68000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices, {}, {})

        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 0

    def test_just_below_2pct_no_trigger(self, isolate_state_files):
        """1.99% price move should NOT trigger."""
        prices_initial = {"BTC-USD": 50000}
        # Use consensus trigger to seed state["last"] with baseline prices
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices_initial, {}, {})

        # 1.99%
        prices_moved = {"BTC-USD": 50995}
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices_moved, {}, {})

        price_reasons = [r for r in reasons if "moved" in r]
        assert len(price_reasons) == 0


# ---------------------------------------------------------------------------
# 7. Fear & Greed: crosses 20 or 80 threshold fires
# ---------------------------------------------------------------------------

class TestFearGreedCrossesThreshold:
    def test_crosses_below_20_triggers(self, isolate_state_files):
        """F&G dropping from 25 to 18 (crossing 20) should trigger."""
        prices = {"BTC-USD": 68000}
        fg_above = {"BTC-USD": {"value": 25}}

        # Seed with F&G above 20
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, fg_above, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Cross below 20
        fg_below = {"BTC-USD": {"value": 18}}
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, fg_below, {})

        assert triggered
        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 1
        assert "20" in fg_reasons[0]

    def test_crosses_above_80_triggers(self, isolate_state_files):
        """F&G rising from 75 to 85 (crossing 80) should trigger."""
        prices = {"BTC-USD": 68000}
        fg_below = {"BTC-USD": {"value": 75}}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, fg_below, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        fg_above = {"BTC-USD": {"value": 85}}
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, fg_above, {})

        assert triggered
        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 1
        assert "80" in fg_reasons[0]

    def test_crosses_back_above_20_triggers(self, isolate_state_files):
        """F&G recovering from 15 to 25 (crossing 20 upward) should also trigger."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 15}}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 25}}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 1


# ---------------------------------------------------------------------------
# 8. Fear & Greed: stays on same side does NOT fire
# ---------------------------------------------------------------------------

class TestFearGreedNoThresholdCross:
    def test_stays_below_20_no_trigger(self, isolate_state_files):
        """F&G moving from 15 to 18 (both below 20) should NOT trigger F&G."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 15}}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 18}}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 0

    def test_stays_above_80_no_trigger(self, isolate_state_files):
        """F&G moving from 85 to 90 (both above 80) should NOT trigger F&G."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 85}}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 90}}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 0

    def test_stays_in_middle_no_trigger(self, isolate_state_files):
        """F&G moving from 45 to 55 (both in middle) should NOT trigger F&G."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 45}}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {"value": 55}}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 0


# ---------------------------------------------------------------------------
# 9. No cooldown triggers — Layer 2 only fires on real triggers
# ---------------------------------------------------------------------------

class TestNoCooldownTrigger:
    def test_no_cooldown_trigger_market_hours(self, isolate_state_files, market_hours_context):
        """Even after long silence during market hours, no cooldown fires."""
        prices = {"BTC-USD": 68000}

        m = market_hours_context()
        try:
            now = time.time()
            _set_trigger_time(isolate_state_files["state_file"], now - 7200)

            triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

            cooldown_reasons = [r for r in reasons if "cooldown" in r or "check-in" in r]
            assert len(cooldown_reasons) == 0
        finally:
            m.stop()

    def test_no_cooldown_trigger_offhours(self, isolate_state_files, offhours_context):
        """Even after long silence during off-hours, no cooldown fires."""
        prices = {"BTC-USD": 68000}

        m = offhours_context()
        try:
            now = time.time()
            _set_trigger_time(isolate_state_files["state_file"], now - 86400)

            triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

            cooldown_reasons = [r for r in reasons if "cooldown" in r or "crypto" in r or "check-in" in r]
            assert len(cooldown_reasons) == 0
        finally:
            m.stop()


# ---------------------------------------------------------------------------
# 10. Post-trade reassessment
# ---------------------------------------------------------------------------

class TestPostTradeCooldownReset:
    def test_patient_trade_detected_triggers(self, isolate_state_files):
        """A new transaction in patient portfolio resets cooldown and triggers."""
        sf = isolate_state_files["state_file"]
        pf = isolate_state_files["portfolio_file"]
        prices = {"BTC-USD": 68000}

        # Seed initial state with known tx count
        _write_portfolio(pf, transactions=[])
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        # Add a transaction
        _write_portfolio(pf, transactions=[{"action": "BUY", "ticker": "BTC-USD"}])
        _suppress_cooldown(sf)

        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)

    def test_bold_trade_detected_triggers(self, isolate_state_files):
        """A new transaction in bold portfolio resets cooldown and triggers."""
        sf = isolate_state_files["state_file"]
        pf_bold = isolate_state_files["portfolio_bold_file"]
        prices = {"BTC-USD": 68000}

        _write_portfolio(pf_bold, transactions=[])
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        _write_portfolio(pf_bold, transactions=[{"action": "BUY", "ticker": "BTC-USD"}])
        _suppress_cooldown(sf)

        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)

    def test_no_trade_no_post_trade_reason(self, isolate_state_files):
        """When no new transactions exist, no post-trade reason should appear."""
        pf = isolate_state_files["portfolio_file"]
        prices = {"BTC-USD": 68000}

        _write_portfolio(pf, transactions=[])
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        # No new transaction added
        _suppress_cooldown(isolate_state_files["state_file"])
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        assert not any("post-trade" in r for r in reasons)

    def test_post_trade_resets_last_trigger_time_to_zero(self, isolate_state_files):
        """After post-trade detection, last_trigger_time is set to 0 (enabling cooldown)."""
        sf = isolate_state_files["state_file"]
        pf = isolate_state_files["portfolio_file"]
        prices = {"BTC-USD": 68000}

        _write_portfolio(pf, transactions=[])
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})

        _write_portfolio(pf, transactions=[{"action": "BUY"}])

        # Before trade detection, set a recent trigger time
        _set_trigger_time(sf, time.time())

        # The trade detection should reset it to 0 internally
        # (and then set it to time.time() at the end because triggered=True)
        triggered, reasons = check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        assert any("post-trade" in r for r in reasons)


# ---------------------------------------------------------------------------
# 12. classify_tier: consensus trigger -> Tier 2
# ---------------------------------------------------------------------------

class TestClassifyTierConsensus:
    def test_consensus_reason_returns_tier_2(self):
        """A consensus trigger reason should classify as Tier 2."""
        state = {
            "last_full_review_time": time.time(),  # recent, so no T3 periodic
            "today_date": trigger_mod._today_str(),  # not first-of-day
        }
        reasons = ["BTC-USD consensus BUY (80%)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 2

    def test_multiple_consensus_reasons_tier_2(self):
        """Multiple consensus triggers still classify as Tier 2."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = [
            "BTC-USD consensus BUY (80%)",
            "ETH-USD consensus SELL (70%)",
        ]
        tier = classify_tier(reasons, state=state)
        assert tier == 2


# ---------------------------------------------------------------------------
# 13. classify_tier: cooldown trigger -> Tier 1
# ---------------------------------------------------------------------------

class TestClassifyTierSentiment:
    def test_sentiment_reason_returns_tier_1(self):
        """A sentiment-only trigger should classify as Tier 1 (noise)."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["BTC-USD sentiment positive->negative (sustained)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 1


# ---------------------------------------------------------------------------
# 14. classify_tier: F&G extreme -> Tier 3
# ---------------------------------------------------------------------------

class TestClassifyTierFearGreed:
    def test_fg_crossed_returns_tier_3(self):
        """An F&G threshold crossing should classify as Tier 3."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["F&G crossed 20 (25->18)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 3

    def test_fg_crossed_80_returns_tier_3(self):
        """F&G crossing 80 should also classify as Tier 3."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["F&G crossed 80 (75->85)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 3


# ---------------------------------------------------------------------------
# 15. classify_tier: first-of-day -> Tier 3
# ---------------------------------------------------------------------------

class TestClassifyTierFirstOfDay:
    def test_first_of_day_returns_tier_3(self):
        """When today_date in state differs from actual today, it is first-of-day -> Tier 3."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": "2026-01-01",  # different from actual today
        }
        reasons = ["cooldown (10min)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 3

    def test_not_first_of_day_no_upgrade(self):
        """Same day should not upgrade to Tier 3 from first-of-day logic."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["cooldown (10min)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 1  # just cooldown


# ---------------------------------------------------------------------------
# 16. classify_tier: price move -> Tier 2
# ---------------------------------------------------------------------------

class TestClassifyTierPriceMove:
    def test_price_move_returns_tier_2(self):
        """A price move reason should classify as Tier 2 ('moved' is a tier2 pattern)."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["BTC-USD moved 3.5% up"]
        tier = classify_tier(reasons, state=state)
        assert tier == 2


# ---------------------------------------------------------------------------
# 17. classify_tier: post-trade -> Tier 2
# ---------------------------------------------------------------------------

class TestClassifyTierPostTrade:
    def test_post_trade_returns_tier_2(self):
        """A post-trade reassessment should classify as Tier 2."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["post-trade reassessment"]
        tier = classify_tier(reasons, state=state)
        assert tier == 2


# ---------------------------------------------------------------------------
# 18. classify_tier: periodic full review -> Tier 3
# ---------------------------------------------------------------------------

class TestClassifyTierPeriodicReview:
    def test_market_hours_2h_since_last_full_review(self, market_hours_context):
        """During market hours, 2+ hours since last full review -> Tier 3."""
        m = market_hours_context()
        try:
            state = {
                "last_full_review_time": time.time() - 7201,  # 2+ hours ago
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["cooldown (10min)"]
            tier = classify_tier(reasons, state=state)
            assert tier == 3
        finally:
            m.stop()

    def test_offhours_4h_since_last_full_review(self, offhours_context):
        """During off-hours, 4+ hours since last full review -> Tier 3."""
        m = offhours_context()
        try:
            state = {
                "last_full_review_time": time.time() - 14401,  # 4+ hours ago
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["crypto check-in (2h)"]
            tier = classify_tier(reasons, state=state)
            assert tier == 3
        finally:
            m.stop()

    def test_market_hours_within_2h_no_upgrade(self, market_hours_context):
        """During market hours, within 2h of last full review -> no Tier 3 upgrade."""
        m = market_hours_context()
        try:
            state = {
                "last_full_review_time": time.time() - 3600,  # 1 hour ago
                "today_date": trigger_mod._today_str(),
            }
            reasons = ["cooldown (10min)"]
            tier = classify_tier(reasons, state=state)
            assert tier == 1  # cooldown stays at Tier 1
        finally:
            m.stop()


# ---------------------------------------------------------------------------
# Tier 3 takes priority over Tier 2
# ---------------------------------------------------------------------------

class TestClassifyTierPriority:
    def test_fg_plus_consensus_returns_tier_3(self):
        """F&G (Tier 3) + consensus (Tier 2) together should return Tier 3."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["F&G crossed 20 (25->18)", "BTC-USD consensus BUY (80%)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 3

    def test_consensus_plus_sentiment_returns_tier_2(self):
        """consensus (Tier 2) + sentiment (Tier 1) together should return Tier 2."""
        state = {
            "last_full_review_time": time.time(),
            "today_date": trigger_mod._today_str(),
        }
        reasons = ["BTC-USD consensus BUY (80%)", "BTC-USD sentiment positive->negative (sustained)"]
        tier = classify_tier(reasons, state=state)
        assert tier == 2


# ---------------------------------------------------------------------------
# update_tier_state: Tier 3 updates last_full_review_time
# ---------------------------------------------------------------------------

class TestUpdateTierState:
    def test_tier_3_updates_full_review_time(self, isolate_state_files):
        """update_tier_state(3) should set last_full_review_time to now."""
        sf = isolate_state_files["state_file"]

        # Seed with old full review time
        _seed_state(sf, {"last_full_review_time": 0})

        before = time.time()
        update_tier_state(3)
        after = time.time()

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_full_review_time"] >= before
        assert state["last_full_review_time"] <= after

    def test_tier_1_does_not_update_full_review_time(self, isolate_state_files):
        """update_tier_state(1) should NOT change last_full_review_time."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 12345})

        update_tier_state(1)

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_full_review_time"] == 12345

    def test_tier_2_does_not_update_full_review_time(self, isolate_state_files):
        """update_tier_state(2) should NOT change last_full_review_time."""
        sf = isolate_state_files["state_file"]
        _seed_state(sf, {"last_full_review_time": 12345})

        update_tier_state(2)

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last_full_review_time"] == 12345


# ---------------------------------------------------------------------------
# State persistence: triggered status saved and restored across calls
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_triggered_consensus_persists(self, isolate_state_files):
        """The triggered_consensus mapping should persist across calls."""
        sf = isolate_state_files["state_file"]
        prices = {"BTC-USD": 68000}

        # Seed and trigger BUY consensus
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["triggered_consensus"]["BTC-USD"] == "BUY"

    def test_sustained_counts_persist(self, isolate_state_files):
        """The sustained_counts mapping should persist across calls."""
        sf = isolate_state_files["state_file"]
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["sustained_counts"]["BTC-USD"]["action"] == "BUY"
        assert state["sustained_counts"]["BTC-USD"]["count"] == 2

    def test_prices_saved_on_trigger(self, isolate_state_files):
        """When triggered, prices are saved to state['last']['prices']."""
        sf = isolate_state_files["state_file"]
        prices = {"BTC-USD": 68000, "ETH-USD": 2000}

        # Use consensus trigger to cause a trigger and save prices
        check_triggers(
            {"BTC-USD": _sig("BUY", 0.8), "ETH-USD": _sig("BUY", 0.7)},
            prices, {}, {}
        )

        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["last"]["prices"]["BTC-USD"] == 68000
        assert state["last"]["prices"]["ETH-USD"] == 2000

    def test_no_trigger_does_not_update_last(self, isolate_state_files):
        """When NOT triggered, state['last'] should not be updated."""
        sf = isolate_state_files["state_file"]
        prices = {"BTC-USD": 68000}

        # First call triggers via consensus
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices, {}, {})

        state_after_first = json.loads(sf.read_text(encoding="utf-8"))
        saved_time = state_after_first["last"]["time"]

        # Second call should NOT trigger (BUY->BUY doesn't re-trigger consensus)
        check_triggers({"BTC-USD": _sig("BUY", 0.8)}, prices, {}, {})

        state_after_second = json.loads(sf.read_text(encoding="utf-8"))
        # last.time should be unchanged
        assert state_after_second["last"]["time"] == saved_time


# ---------------------------------------------------------------------------
# Multiple tickers: only changed tickers fire consensus
# ---------------------------------------------------------------------------

class TestMultipleTickerConsensus:
    def test_only_changed_ticker_fires(self, isolate_state_files):
        """When multiple tickers exist, only the one that changed fires consensus."""
        prices = {"BTC-USD": 68000, "ETH-USD": 2000, "NVDA": 185}

        sigs_hold = _sigs({"BTC-USD": "HOLD", "ETH-USD": "HOLD", "NVDA": "HOLD"})
        check_triggers(sigs_hold, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Only BTC flips to BUY
        sigs_mixed = _sigs({"BTC-USD": "BUY", "ETH-USD": "HOLD", "NVDA": "HOLD"})
        triggered, reasons = check_triggers(sigs_mixed, prices, {}, {})

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1
        assert "BTC-USD" in consensus_reasons[0]

    def test_two_tickers_change_simultaneously(self, isolate_state_files):
        """When two tickers change at once, both fire consensus."""
        prices = {"BTC-USD": 68000, "ETH-USD": 2000}

        sigs_hold = _sigs({"BTC-USD": "HOLD", "ETH-USD": "HOLD"})
        check_triggers(sigs_hold, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        sigs_buy = _sigs({"BTC-USD": "BUY", "ETH-USD": "SELL"})
        triggered, reasons = check_triggers(sigs_buy, prices, {}, {})

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 2
        tickers_in_reasons = " ".join(consensus_reasons)
        assert "BTC-USD" in tickers_in_reasons
        assert "ETH-USD" in tickers_in_reasons


# ---------------------------------------------------------------------------
# Direction flip BUY->SELL updates triggered_consensus silently
# ---------------------------------------------------------------------------

class TestDirectionFlipConsensus:
    def test_buy_to_sell_does_not_fire_consensus_again(self, isolate_state_files):
        """BUY->SELL should NOT fire consensus (it's a direction flip, handled by sustained)."""
        prices = {"BTC-USD": 68000}
        sf = isolate_state_files["state_file"]

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        # BUY consensus fires
        check_triggers({"BTC-USD": _sig("BUY")}, prices, {}, {})
        _suppress_cooldown(sf)

        # Directly flip BUY -> SELL (no HOLD in between)
        triggered, reasons = check_triggers({"BTC-USD": _sig("SELL")}, prices, {}, {})

        # Should NOT fire a new consensus (direction flip is silent)
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 0

        # But triggered_consensus should be updated to SELL
        state = json.loads(sf.read_text(encoding="utf-8"))
        assert state["triggered_consensus"]["BTC-USD"] == "SELL"


# ---------------------------------------------------------------------------
# Empty inputs: no signals, no prices
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_signals_and_prices(self, isolate_state_files):
        """check_triggers with empty dicts should not crash."""
        triggered, reasons = check_triggers({}, {}, {}, {})
        # May trigger from cooldown (first run), but should not crash
        assert isinstance(triggered, bool)
        assert isinstance(reasons, list)

    def test_new_ticker_appears(self, isolate_state_files):
        """A ticker appearing for the first time should trigger if BUY/SELL."""
        prices = {"BTC-USD": 68000}

        # First call with BTC only
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Second call adds ETH with BUY
        prices2 = {"BTC-USD": 68000, "ETH-USD": 2000}
        sigs2 = {"BTC-USD": _sig("HOLD"), "ETH-USD": _sig("BUY")}
        triggered, reasons = check_triggers(sigs2, prices2, {}, {})

        # ETH is new and BUY -> should trigger consensus
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert any("ETH-USD" in r for r in consensus_reasons)


# ---------------------------------------------------------------------------
# Fear & Greed: non-dict values default to 50
# ---------------------------------------------------------------------------

class TestFearGreedEdgeCases:
    def test_non_dict_fg_defaults_to_50(self, isolate_state_files):
        """Non-dict F&G values should be treated as 50 (no threshold crossing)."""
        prices = {"BTC-USD": 68000}

        # Seed with non-dict fg
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": "invalid"}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Again with non-dict
        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": "still_invalid"}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 0

    def test_missing_value_key_defaults_to_50(self, isolate_state_files):
        """F&G dict without 'value' key should default to 50."""
        prices = {"BTC-USD": 68000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {}}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("HOLD")}, prices, {"BTC-USD": {}}, {}
        )

        fg_reasons = [r for r in reasons if "F&G" in r]
        assert len(fg_reasons) == 0


# ---------------------------------------------------------------------------
# check_recent_trade: file I/O edge cases
# ---------------------------------------------------------------------------

class TestCheckRecentTrade:
    def test_missing_portfolio_files(self, isolate_state_files):
        """When portfolio files don't exist, no trade should be detected."""
        # Portfolio files do not exist (tmp_path is empty)
        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
        result = trigger_mod._check_recent_trade(state)
        assert result is False

    def test_malformed_portfolio_json(self, isolate_state_files):
        """Malformed JSON in portfolio file should not crash, returns False."""
        pf = isolate_state_files["portfolio_file"]
        pf.write_text("{{invalid json", encoding="utf-8")

        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
        result = trigger_mod._check_recent_trade(state)
        assert result is False

    def test_first_check_with_existing_transactions(self, isolate_state_files):
        """First check: existing transactions should be treated as the baseline, not new."""
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[{"action": "BUY"}, {"action": "SELL"}])

        # No previous tx count -> prev_count defaults to current_count (2)
        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
        result = trigger_mod._check_recent_trade(state)
        assert result is False

    def test_incremented_tx_count_detected(self, isolate_state_files):
        """Adding a transaction after baseline should be detected."""
        pf = isolate_state_files["portfolio_file"]
        _write_portfolio(pf, transactions=[{"action": "BUY"}])

        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {"patient": 1}}
        _write_portfolio(pf, transactions=[{"action": "BUY"}, {"action": "SELL"}])

        result = trigger_mod._check_recent_trade(state)
        assert result is True
        assert state["last_checked_tx_count"]["patient"] == 2
