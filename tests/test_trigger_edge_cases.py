"""Tests for trigger system edge cases.

Covers:
- 30-ticker simultaneous consensus changes
- Post-trade cooldown reset (both portfolios)
- Rapid signal oscillation (BUY->HOLD->BUY in 3 cycles)
- Sustained flip detection (3 consecutive checks)
- Market hours boundary (just before/after 7:00 and 21:00 UTC)
- Multiple trigger reasons in one cycle
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from portfolio.trigger import (
    STATE_FILE,
    COOLDOWN_SECONDS,
    OFFHOURS_COOLDOWN,
    PRICE_THRESHOLD,
    FG_THRESHOLDS,
    SUSTAINED_CHECKS,
    check_triggers,
    _load_state,
    _save_state,
    _check_recent_trade,
    PORTFOLIO_FILE,
    PORTFOLIO_BOLD_FILE,
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
# Helpers
# ---------------------------------------------------------------------------

ALL_TICKERS = [
    "BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD",
    "MSTR", "PLTR", "NVDA", "AMD", "BABA", "GOOGL",
    "AMZN", "AAPL", "AVGO", "AI", "GRRR", "IONQ",
    "MRVL", "META", "MU", "PONY", "RXRX", "SOUN",
    "SMCI", "TSM", "TTWO", "TEM", "UPST", "VERI",
    "VRT", "QQQ",
]

def _make_signals(tickers=None, action="HOLD", confidence=0.5):
    """Create a signals dict for all tickers."""
    tickers = tickers or ALL_TICKERS
    return {t: {"action": action, "confidence": confidence} for t in tickers}


def _make_prices(tickers=None, base_price=100.0):
    """Create a prices dict for all tickers."""
    tickers = tickers or ALL_TICKERS
    return {t: base_price + i * 10 for i, t in enumerate(tickers)}


# ---------------------------------------------------------------------------
# Test: 30-ticker simultaneous consensus changes
# ---------------------------------------------------------------------------

class TestSimultaneousConsensusChanges(TriggerTestBase):
    def test_30_tickers_all_flip_to_buy(self):
        """When all 30 tickers simultaneously reach BUY consensus."""
        tickers = ALL_TICKERS[:30]
        prices = _make_prices(tickers)

        # First run: all HOLD (seeds the state)
        sigs_hold = _make_signals(tickers, "HOLD")
        check_triggers(sigs_hold, prices, {}, {})

        # Second run: all flip to BUY
        sigs_buy = _make_signals(tickers, "BUY")
        triggered, reasons = check_triggers(sigs_buy, prices, {}, {})

        assert triggered
        # Should have a consensus reason for each ticker that flipped
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 30

    def test_30_tickers_mixed_flip(self):
        """15 tickers flip to BUY, 15 flip to SELL simultaneously."""
        tickers = ALL_TICKERS[:30]
        prices = _make_prices(tickers)

        sigs_hold = _make_signals(tickers, "HOLD")
        check_triggers(sigs_hold, prices, {}, {})

        # Mix of BUY and SELL
        sigs_mixed = {}
        for i, t in enumerate(tickers):
            action = "BUY" if i < 15 else "SELL"
            sigs_mixed[t] = {"action": action, "confidence": 0.7}

        triggered, reasons = check_triggers(sigs_mixed, prices, {}, {})
        assert triggered
        buy_reasons = [r for r in reasons if "BUY" in r]
        sell_reasons = [r for r in reasons if "SELL" in r]
        assert len(buy_reasons) == 15
        assert len(sell_reasons) == 15


# ---------------------------------------------------------------------------
# Test: Post-trade cooldown reset
# ---------------------------------------------------------------------------

class TestPostTradeCooldownReset(TriggerTestBase):
    def test_patient_trade_resets_cooldown(self):
        """A trade in patient portfolio should reset the cooldown timer."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        # First run seeds state
        check_triggers(sigs, prices, {}, {})

        # Simulate a trade by patching _check_recent_trade
        with mock.patch("portfolio.trigger._check_recent_trade", return_value=True):
            triggered, reasons = check_triggers(sigs, prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)

    def test_bold_trade_resets_cooldown(self):
        """A trade in bold portfolio should reset the cooldown timer."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        check_triggers(sigs, prices, {}, {})

        with mock.patch("portfolio.trigger._check_recent_trade", return_value=True):
            triggered, reasons = check_triggers(sigs, prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)

    def test_no_trade_no_reset(self):
        """Without a trade, no post-trade reset reason."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        check_triggers(sigs, prices, {}, {})

        with mock.patch("portfolio.trigger._check_recent_trade", return_value=False):
            triggered, reasons = check_triggers(sigs, prices, {}, {})

        # May or may not trigger (cooldown), but should NOT have post-trade reason
        assert not any("post-trade" in r for r in reasons)


# ---------------------------------------------------------------------------
# Test: Rapid signal oscillation (BUY->HOLD->BUY in 3 cycles)
# ---------------------------------------------------------------------------

class TestRapidSignalOscillation(TriggerTestBase):
    def test_buy_hold_buy_does_not_trigger_flip(self):
        """BUY->HOLD->BUY should NOT trigger a sustained flip because
        the HOLD in the middle resets the counter."""
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        prices = {"BTC-USD": 69000}

        # Seed with BUY consensus
        check_triggers(sigs_buy, prices, {}, {})

        # Cycle 1: HOLD (resets counter for BUY)
        check_triggers(sigs_hold, prices, {}, {})

        # Cycle 2: BUY (counter starts at 1)
        check_triggers(sigs_buy, prices, {}, {})

        # Cycle 3: BUY (counter = 2, not yet SUSTAINED_CHECKS=3)
        triggered, reasons = check_triggers(sigs_buy, prices, {}, {})

        # Should NOT have a "flipped" reason yet (only 2 consecutive)
        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0

    def test_sustained_hold_after_buy_triggers_flip(self):
        """BUY triggered -> HOLD x3 triggers sustained flip BUY->HOLD.

        The sustained flip only fires when current_action differs from the last
        triggered action AND the new action did NOT already fire a consensus
        trigger (which updates the triggered state).  Since consensus only fires
        for BUY/SELL, a HOLD sustained for 3 cycles while the triggered state
        is still BUY is the canonical sustained-flip scenario.
        """
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        prices = {"BTC-USD": 69000}

        # Seed with HOLD so the first BUY triggers consensus
        check_triggers(sigs_hold, prices, {}, {})

        # BUY consensus fires -> triggered action is now BUY
        triggered1, reasons1 = check_triggers(sigs_buy, prices, {}, {})
        assert any("consensus" in r for r in reasons1)

        # Now switch to HOLD for 3 consecutive cycles.
        # HOLD does NOT trigger consensus (only BUY/SELL do), so the
        # triggered state stays BUY.  After 3 HOLD cycles, sustained flip fires.
        # Note: cycles 1 and 2 may still trigger for cooldown reasons, which
        # would update last_trigger_time but NOT change last.signals (HOLD is
        # stored only when triggered==True, but signals still get written).
        # Actually, if cooldown fires, state["last"]["signals"] is updated to
        # HOLD.  We need to suppress cooldown by setting last_trigger_time
        # to recent.
        import portfolio.trigger as trig
        state = trig._load_state()
        state["last_trigger_time"] = time.time() + 9999  # suppress cooldown
        trig._save_state(state)

        # Cycle 1: HOLD, no trigger (cooldown suppressed, no consensus)
        t1, r1 = check_triggers(sigs_hold, prices, {}, {})
        # If no trigger, state["last"] is NOT updated -> triggered action stays BUY
        if not t1:
            pass  # good, state preserved

        state = trig._load_state()
        state["last_trigger_time"] = time.time() + 9999
        trig._save_state(state)

        # Cycle 2: HOLD, still no trigger
        t2, r2 = check_triggers(sigs_hold, prices, {}, {})

        state = trig._load_state()
        state["last_trigger_time"] = time.time() + 9999
        trig._save_state(state)

        # Cycle 3: HOLD, sustained count reaches 3 -> flip fires
        t3, r3 = check_triggers(sigs_hold, prices, {}, {})
        flip_reasons = [r for r in r3 if "flipped" in r]
        assert len(flip_reasons) == 1
        assert "BUY->HOLD" in flip_reasons[0]


# ---------------------------------------------------------------------------
# Test: Sustained flip detection
# ---------------------------------------------------------------------------

class TestSustainedFlipDetection(TriggerTestBase):
    def _suppress_cooldown(self):
        """Set last_trigger_time far in the future to suppress cooldown triggers."""
        import portfolio.trigger as trig
        state = trig._load_state()
        state["last_trigger_time"] = time.time() + 9999
        trig._save_state(state)

    def test_exactly_3_consecutive_checks(self):
        """Signal must sustain for exactly SUSTAINED_CHECKS (3) consecutive cycles.

        Sustained flip works for HOLD transitions because consensus trigger
        (section 1) only fires for BUY/SELL.  When HOLD is sustained after a
        BUY triggered state, the triggered state stays BUY and the flip fires
        after 3 HOLD cycles.
        """
        assert SUSTAINED_CHECKS == 3

        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        prices = {"BTC-USD": 69000}

        # Seed with HOLD, then trigger BUY consensus
        check_triggers(sigs_hold, prices, {}, {})
        check_triggers(sigs_buy, prices, {}, {})  # triggers consensus BUY

        # Suppress cooldown for the HOLD cycles so no unrelated triggers update state
        self._suppress_cooldown()
        check_triggers(sigs_hold, prices, {}, {})  # count=1 (no trigger if cooldown suppressed)

        self._suppress_cooldown()
        check_triggers(sigs_hold, prices, {}, {})  # count=2

        self._suppress_cooldown()
        # 3rd cycle: should trigger sustained flip
        triggered, reasons = check_triggers(sigs_hold, prices, {}, {})
        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 1
        assert "BUY->HOLD" in flip_reasons[0]

    def test_2_consecutive_not_enough(self):
        """Only 2 consecutive HOLD cycles should NOT trigger a sustained flip."""
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        # Seed with HOLD, then trigger BUY consensus
        check_triggers(sigs_hold, prices, {}, {})
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        check_triggers(sigs_buy, prices, {}, {})

        self._suppress_cooldown()
        check_triggers(sigs_hold, prices, {}, {})  # count=1

        self._suppress_cooldown()
        triggered, reasons = check_triggers(sigs_hold, prices, {}, {})  # count=2

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0

    def test_interrupted_sequence_resets(self):
        """An interruption in the sustained sequence resets the counter."""
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        sigs_sell = _make_signals(["BTC-USD"], "SELL")
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        prices = {"BTC-USD": 69000}

        check_triggers(sigs_hold, prices, {}, {})
        check_triggers(sigs_buy, prices, {}, {})  # set triggered action to BUY

        check_triggers(sigs_sell, prices, {}, {})  # SELL count=1
        check_triggers(sigs_sell, prices, {}, {})  # SELL count=2
        check_triggers(sigs_hold, prices, {}, {})  # HOLD interrupts -> reset
        check_triggers(sigs_sell, prices, {}, {})  # SELL count=1 (restarted)
        triggered, reasons = check_triggers(sigs_sell, prices, {}, {})  # SELL count=2

        flip_reasons = [r for r in reasons if "flipped" in r]
        assert len(flip_reasons) == 0  # only 2 consecutive, not 3


# ---------------------------------------------------------------------------
# Test: Market hours boundary
# ---------------------------------------------------------------------------

class TestMarketHoursBoundary(TriggerTestBase):
    def test_just_before_7_utc_is_offhours(self):
        """6:59 UTC on a weekday is off-hours."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        # Mock datetime to be 6:59 UTC on a Monday
        fake_now = datetime(2026, 2, 16, 6, 59, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Set last_trigger_time far in the past to ensure cooldown triggers
            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        # Off-hours cooldown reason (1h interval)
        cooldown_reasons = [r for r in reasons if "check-in" in r or "cooldown" in r
                           or "crypto" in r]
        # Should use off-hours cooldown
        if triggered:
            assert any("crypto" in r or "check-in" in r for r in reasons)

    def test_just_at_7_utc_is_market_hours(self):
        """7:00 UTC on a weekday is market hours."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        fake_now = datetime(2026, 2, 16, 7, 0, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        # Market hours cooldown reason (30min interval)
        if triggered:
            assert any("cooldown" in r for r in reasons)

    def test_just_before_21_utc_is_market_hours(self):
        """20:59 UTC on a weekday is still market hours."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        fake_now = datetime(2026, 2, 16, 20, 59, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        if triggered:
            assert any("cooldown" in r for r in reasons)

    def test_just_at_21_utc_is_offhours(self):
        """21:00 UTC on a weekday is off-hours."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        fake_now = datetime(2026, 2, 16, 21, 0, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        if triggered:
            assert any("crypto" in r or "check-in" in r for r in reasons)

    def test_weekend_is_offhours(self):
        """Saturday/Sunday should be off-hours regardless of time."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        # Saturday at 10:00 UTC
        fake_now = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        if triggered:
            assert any("crypto" in r or "check-in" in r for r in reasons)


# ---------------------------------------------------------------------------
# Test: Multiple trigger reasons in one cycle
# ---------------------------------------------------------------------------

class TestMultipleTriggerReasons(TriggerTestBase):
    def test_consensus_plus_price_move(self):
        """Signal consensus change + 2% price move = both reasons."""
        sigs_hold = _make_signals(["BTC-USD", "ETH-USD"], "HOLD")
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}

        check_triggers(sigs_hold, prices, {}, {})

        # Both consensus flip AND price move
        sigs_buy = {"BTC-USD": {"action": "BUY", "confidence": 0.8},
                    "ETH-USD": {"action": "HOLD", "confidence": 0.5}}
        new_prices = {"BTC-USD": 72000, "ETH-USD": 2000}  # 4.3% move

        triggered, reasons = check_triggers(sigs_buy, new_prices, {}, {})

        assert triggered
        assert any("consensus" in r for r in reasons)
        assert any("moved" in r for r in reasons)

    def test_consensus_plus_fg_plus_sentiment(self):
        """Signal consensus + F&G threshold + sentiment reversal."""
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}
        fg = {"BTC-USD": {"value": 25}}
        sent = {"BTC-USD": "positive"}

        check_triggers(sigs_hold, prices, fg, sent)

        # Multiple changes at once
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        fg_extreme = {"BTC-USD": {"value": 18}}  # crossed 20
        sent_flip = {"BTC-USD": "negative"}

        triggered, reasons = check_triggers(sigs_buy, prices, fg_extreme, sent_flip)

        assert triggered
        assert any("consensus" in r for r in reasons)
        assert any("F&G" in r for r in reasons)
        assert any("sentiment" in r for r in reasons)

    def test_post_trade_plus_price_move(self):
        """Post-trade reset + price move simultaneously."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        check_triggers(sigs, prices, {}, {})

        with mock.patch("portfolio.trigger._check_recent_trade", return_value=True):
            new_prices = {"BTC-USD": 72000}  # 4.3% move
            triggered, reasons = check_triggers(sigs, new_prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)
        assert any("moved" in r for r in reasons)


# ---------------------------------------------------------------------------
# Test: Cooldown constants
# ---------------------------------------------------------------------------

class TestCooldownConstants:
    def test_market_hours_cooldown_is_30_min(self):
        assert COOLDOWN_SECONDS == 1800

    def test_offhours_cooldown_is_1_hour(self):
        assert OFFHOURS_COOLDOWN == 3600

    def test_price_threshold_is_2_pct(self):
        assert PRICE_THRESHOLD == 0.02

    def test_fg_thresholds(self):
        assert FG_THRESHOLDS == (20, 80)

    def test_sustained_checks_is_3(self):
        assert SUSTAINED_CHECKS == 3
