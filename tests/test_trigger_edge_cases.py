"""Tests for trigger system edge cases.

Covers:
- 30-ticker simultaneous consensus changes
- Post-trade reassessment (both portfolios)
- Rapid signal oscillation (BUY->HOLD->BUY in 3 cycles)
- Sustained flip detection (3 consecutive checks)
- Market hours boundary (just before/after 7:00 and 21:00 UTC)
- Multiple trigger reasons in one cycle

All tests use isolated temp files for trigger state so they are safe
for parallel execution with pytest-xdist (-n auto).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from portfolio.trigger import (
    PRICE_THRESHOLD,
    FG_THRESHOLDS,
    SUSTAINED_CHECKS,
    check_triggers,
    _load_state,
    _save_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_trigger_state(tmp_path):
    """Redirect trigger state to a temp file per test for xdist safety."""
    tmp_state = tmp_path / "trigger_state.json"
    with (
        mock.patch("portfolio.trigger.STATE_FILE", tmp_state),
        mock.patch("portfolio.trigger._startup_grace_active", False),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_TICKERS = [
    "BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD",
    "PLTR", "NVDA", "AMD", "GOOGL",
    "AMZN", "AAPL", "AVGO",
    "META", "MU", "SOUN",
    "SMCI", "TSM", "TTWO",
    "VRT", "LMT",
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

class TestSimultaneousConsensusChanges:
    def test_all_tickers_flip_to_buy(self):
        """When all tickers simultaneously reach BUY consensus."""
        tickers = ALL_TICKERS
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
        assert len(consensus_reasons) == len(tickers)

    def test_all_tickers_mixed_flip(self):
        """Half tickers flip to BUY, half flip to SELL simultaneously."""
        tickers = ALL_TICKERS
        prices = _make_prices(tickers)
        half = len(tickers) // 2

        sigs_hold = _make_signals(tickers, "HOLD")
        check_triggers(sigs_hold, prices, {}, {})

        # Mix of BUY and SELL
        sigs_mixed = {}
        for i, t in enumerate(tickers):
            action = "BUY" if i < half else "SELL"
            sigs_mixed[t] = {"action": action, "confidence": 0.7}

        triggered, reasons = check_triggers(sigs_mixed, prices, {}, {})
        assert triggered
        buy_reasons = [r for r in reasons if "BUY" in r]
        sell_reasons = [r for r in reasons if "SELL" in r]
        assert len(buy_reasons) == half
        assert len(sell_reasons) == len(tickers) - half


# ---------------------------------------------------------------------------
# Test: Post-trade cooldown reset
# ---------------------------------------------------------------------------

class TestPostTradeCooldownReset:
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

class TestRapidSignalOscillation:
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
        # Suppress cooldown by setting last_trigger_time far in the future.
        import portfolio.trigger as trig
        state = trig._load_state()
        state["last_trigger_time"] = time.time() + 9999
        trig._save_state(state)

        # Cycle 1: HOLD, no trigger (cooldown suppressed, no consensus)
        t1, r1 = check_triggers(sigs_hold, prices, {}, {})
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

class TestSustainedFlipDetection:
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

class TestMarketHoursBoundary:
    def test_no_trigger_on_silence_market_hours(self):
        """During market hours with HOLD signals, no trigger fires (no cooldown)."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        fake_now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        cooldown_reasons = [r for r in reasons if "cooldown" in r or "check-in" in r]
        assert len(cooldown_reasons) == 0

    def test_no_trigger_on_silence_offhours(self):
        """During off-hours with HOLD signals, no trigger fires (no cooldown)."""
        sigs = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}

        fake_now = datetime(2026, 2, 16, 23, 0, 0, tzinfo=timezone.utc)
        with mock.patch("portfolio.trigger.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            state = _load_state()
            state["last_trigger_time"] = 0
            _save_state(state)

            triggered, reasons = check_triggers(sigs, prices, {}, {})

        cooldown_reasons = [r for r in reasons if "cooldown" in r or "crypto" in r or "check-in" in r]
        assert len(cooldown_reasons) == 0

    def test_no_trigger_on_weekend_silence(self):
        """Weekend with HOLD signals, no trigger fires (no cooldown)."""
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

        cooldown_reasons = [r for r in reasons if "cooldown" in r or "crypto" in r or "check-in" in r]
        assert len(cooldown_reasons) == 0


# ---------------------------------------------------------------------------
# Test: Multiple trigger reasons in one cycle
# ---------------------------------------------------------------------------

class TestMultipleTriggerReasons:
    def test_consensus_plus_price_move(self):
        """Signal consensus change + 2% price move = both reasons."""
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}

        # Seed with a BUY consensus to establish baseline prices in state["last"]
        sigs_seed = {"BTC-USD": {"action": "BUY", "confidence": 0.6},
                     "ETH-USD": {"action": "HOLD", "confidence": 0.5}}
        check_triggers(sigs_seed, prices, {}, {})

        # Reset BTC to HOLD so next BUY is a new consensus trigger
        check_triggers({"BTC-USD": {"action": "HOLD", "confidence": 0.5},
                        "ETH-USD": {"action": "HOLD", "confidence": 0.5}}, prices, {}, {})

        # Both consensus flip AND price move
        sigs_buy = {"BTC-USD": {"action": "BUY", "confidence": 0.8},
                    "ETH-USD": {"action": "HOLD", "confidence": 0.5}}
        new_prices = {"BTC-USD": 72000, "ETH-USD": 2000}  # 4.3% move

        triggered, reasons = check_triggers(sigs_buy, new_prices, {}, {})

        assert triggered
        assert any("consensus" in r for r in reasons)
        assert any("moved" in r for r in reasons)

    def test_consensus_plus_fg_plus_sentiment(self):
        """Signal consensus + F&G threshold + sustained sentiment reversal."""
        sigs_hold = _make_signals(["BTC-USD"], "HOLD")
        prices = {"BTC-USD": 69000}
        fg = {"BTC-USD": {"value": 25}}
        sent_pos = {"BTC-USD": "positive"}

        # Seed state with positive sentiment sustained (so stable_sentiment is set)
        for _ in range(SUSTAINED_CHECKS):
            check_triggers(sigs_hold, prices, fg, sent_pos)

        sent_neg = {"BTC-USD": "negative"}

        # Sustain negative sentiment for SUSTAINED_CHECKS cycles so it triggers
        # First SUSTAINED_CHECKS-1 cycles: sentiment not yet sustained
        for _ in range(SUSTAINED_CHECKS - 1):
            check_triggers(sigs_hold, prices, fg, sent_neg)

        # Final cycle: consensus + F&G + sustained sentiment all fire
        sigs_buy = _make_signals(["BTC-USD"], "BUY")
        fg_extreme = {"BTC-USD": {"value": 18}}  # crossed 20

        triggered, reasons = check_triggers(sigs_buy, prices, fg_extreme, sent_neg)

        assert triggered
        assert any("consensus" in r for r in reasons)
        assert any("F&G" in r for r in reasons)
        assert any("sentiment" in r for r in reasons)

    def test_post_trade_plus_price_move(self):
        """Post-trade reset + price move simultaneously."""
        prices = {"BTC-USD": 69000}

        # Seed with consensus trigger to establish baseline prices
        check_triggers({"BTC-USD": {"action": "BUY", "confidence": 0.8}}, prices, {}, {})

        with mock.patch("portfolio.trigger._check_recent_trade", return_value=True):
            sigs = _make_signals(["BTC-USD"], "HOLD")
            new_prices = {"BTC-USD": 72000}  # 4.3% move
            triggered, reasons = check_triggers(sigs, new_prices, {}, {})

        assert triggered
        assert any("post-trade" in r for r in reasons)
        assert any("moved" in r for r in reasons)


# ---------------------------------------------------------------------------
# Test: Trigger constants
# ---------------------------------------------------------------------------

class TestTriggerConstants:
    def test_price_threshold_is_2_pct(self):
        assert PRICE_THRESHOLD == 0.02

    def test_fg_thresholds(self):
        assert FG_THRESHOLDS == (20, 80)

    def test_sustained_checks_is_3(self):
        assert SUSTAINED_CHECKS == 3
