"""Tests for metals swing trader — entry/exit logic, warrant selection, state management."""

import json
import os
import sys
import datetime
import pytest

# Ensure data/ is on path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Patch config before importing trader
import metals_swing_config as cfg
cfg.DRY_RUN = True  # always dry run in tests
cfg.STATE_FILE = "data/_test_swing_state.json"
cfg.DECISIONS_LOG = "data/_test_swing_decisions.jsonl"
cfg.TRADES_LOG = "data/_test_swing_trades.jsonl"

from metals_swing_trader import (
    SwingTrader, _load_state, _save_state, _default_state, _compact_signal, _now_utc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_state():
    """Remove test state files before/after each test."""
    for f in [cfg.STATE_FILE, cfg.DECISIONS_LOG, cfg.TRADES_LOG]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in [cfg.STATE_FILE, cfg.DECISIONS_LOG, cfg.TRADES_LOG]:
        if os.path.exists(f):
            os.remove(f)


class MockPage:
    """Mock Playwright page that returns preset data."""

    def __init__(self, prices=None, account=None):
        self._prices = prices or {}
        self._account = account
        self._context = MockContext()

    def evaluate(self, script, args=None):
        # Account overview
        if args and isinstance(args, str) and args == cfg.ACCOUNT_ID:
            return self._account or {"buying_power": 10000, "total_value": 10000}

        # Price fetch
        if args and isinstance(args, list) and len(args) == 2:
            ob_id, api_type = args[0], args[1]
            # Check if this is a BUY/SELL order call (has dict as first arg)
            if isinstance(ob_id, dict):
                return {"status": 200, "body": json.dumps({"orderRequestStatus": "SUCCESS", "orderId": "12345"})}
            return self._prices.get(ob_id)

        return None

    @property
    def context(self):
        return self._context


class MockContext:
    def cookies(self):
        return [{"name": "AZACSRF", "value": "test-csrf-token"}]


def make_signal(action="HOLD", buy_count=0, sell_count=0, rsi=50,
                macd_hist=0, regime="range-bound", confidence=0.5,
                timeframes=None):
    """Create a signal dict matching read_signal_data() output."""
    return {
        "action": action,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "regime": regime,
        "confidence": confidence,
        "voters": buy_count + sell_count,
        "vote_detail": "",
        "bb_position": "inside",
        "atr_pct": 3.0,
        "weighted_confidence": confidence,
        "timeframes": timeframes or {
            "Now": "BUY", "12h": "BUY", "2d": "BUY",
            "7d": "HOLD", "1mo": "HOLD", "3mo": "HOLD", "6mo": "HOLD",
        },
    }


def make_trader(cash=10000, positions=None, macd_history=None, consecutive_losses=0):
    """Create a SwingTrader with preset state (no Avanza API calls)."""
    prices = {
        "2043157": {
            "bid": 10.0, "ask": 10.10, "last": 10.05,
            "underlying": 87.0, "leverage": 1.56, "barrier": 32.45,
            "change_pct": 0.5, "high": 10.5, "low": 9.8,
        },
        "2334960": {
            "bid": 15.0, "ask": 15.10, "last": 15.05,
            "underlying": 87.0, "leverage": 5.0, "barrier": 60.0,
            "change_pct": 0.8, "high": 15.5, "low": 14.5,
        },
    }
    page = MockPage(prices=prices, account={"buying_power": cash})

    state = _default_state()
    state["cash_sek"] = cash
    state["positions"] = positions or {}
    state["macd_history"] = macd_history or {}
    state["consecutive_losses"] = consecutive_losses
    _save_state(state)

    trader = SwingTrader.__new__(SwingTrader)
    trader.page = page
    trader.state = _load_state()
    trader.check_count = 0
    return trader


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_default_state(self):
        s = _default_state()
        assert s["cash_sek"] == 0
        assert s["positions"] == {}
        assert s["consecutive_losses"] == 0
        assert s["total_trades"] == 0

    def test_save_load_roundtrip(self):
        s = _default_state()
        s["cash_sek"] = 5000
        s["positions"]["pos_1"] = {"warrant_key": "test", "units": 10}
        _save_state(s)

        loaded = _load_state()
        assert loaded["cash_sek"] == 5000
        assert "pos_1" in loaded["positions"]

    def test_load_missing_file(self):
        s = _load_state()
        assert s == _default_state()


# ---------------------------------------------------------------------------
# Entry logic tests
# ---------------------------------------------------------------------------

class TestEntryLogic:
    def test_skip_when_no_signal(self):
        trader = make_trader(cash=10000)
        trader._check_entries({}, {})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_hold_consensus(self):
        trader = make_trader(cash=10000)
        sig = make_signal(action="HOLD", buy_count=1, sell_count=1, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_buy_count_too_low(self):
        trader = make_trader(cash=10000)
        sig = make_signal(action="BUY", buy_count=2, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_rsi_overbought(self):
        trader = make_trader(cash=10000)
        sig = make_signal(action="BUY", buy_count=5, rsi=75)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_rsi_oversold(self):
        trader = make_trader(cash=10000)
        sig = make_signal(action="BUY", buy_count=5, rsi=25)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_insufficient_cash(self):
        trader = make_trader(cash=100)
        sig = make_signal(action="BUY", buy_count=5, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_skip_when_max_positions(self):
        trader = make_trader(cash=10000, positions={
            "pos_1": {"underlying": "XAG-USD", "units": 10},
            "pos_2": {"underlying": "XAU-USD", "units": 5},
        })
        sig = make_signal(action="BUY", buy_count=5, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 2  # unchanged

    def test_skip_when_already_holding_underlying(self):
        trader = make_trader(cash=10000, positions={
            "pos_1": {"underlying": "XAG-USD", "units": 10},
        })
        sig = make_signal(action="BUY", buy_count=5, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 1  # unchanged

    def test_entry_on_valid_signal(self):
        """With valid BUY signal + enough cash + good RSI → should create position."""
        trader = make_trader(cash=10000, macd_history={
            "XAG-USD": [0.1, 0.2, 0.3],  # improving
        })
        sig = make_signal(action="BUY", buy_count=5, rsi=50,
                          timeframes={"Now": "BUY", "12h": "BUY", "2d": "BUY",
                                      "7d": "BUY", "1mo": "HOLD", "3mo": "HOLD", "6mo": "HOLD"})
        trader._check_entries({}, {"XAG-USD": sig})
        # Should have created a position (DRY_RUN mode)
        assert len(trader.state["positions"]) == 1

    def test_cooldown_blocks_entry(self):
        """Recent BUY should block new entries."""
        trader = make_trader(cash=10000, macd_history={
            "XAG-USD": [0.1, 0.2, 0.3],
        })
        trader.state["last_buy_ts"] = _now_utc().isoformat()
        sig = make_signal(action="BUY", buy_count=5, rsi=50)
        trader._check_entries({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 0

    def test_cooldown_escalation(self):
        """Consecutive losses should increase cooldown."""
        trader = make_trader(cash=10000, consecutive_losses=3, macd_history={
            "XAG-USD": [0.1, 0.2, 0.3],
        })
        # Set last buy 60 min ago (>30 base but <30*8=240 with 3 losses)
        trader.state["last_buy_ts"] = (
            _now_utc() - datetime.timedelta(minutes=60)
        ).isoformat()
        assert not trader._cooldown_cleared()


# ---------------------------------------------------------------------------
# Exit logic tests
# ---------------------------------------------------------------------------

class TestExitLogic:
    def _make_position(self, und_entry=87.0, entry_price=15.0, hours_ago=1):
        return {
            "warrant_key": "MINI_L_SILVER_AVA_301",
            "warrant_name": "MINI L SILVER AVA 301",
            "ob_id": "2334960",
            "api_type": "warrant",
            "underlying": "XAG-USD",
            "units": 100,
            "entry_price": entry_price,
            "entry_underlying": und_entry,
            "entry_ts": (_now_utc() - datetime.timedelta(hours=hours_ago)).isoformat(),
            "peak_underlying": und_entry,
            "trailing_active": False,
            "stop_order_id": "DRY_RUN",
            "leverage": 6.3,
        }

    def test_take_profit(self):
        """Exit when underlying moves +2%."""
        pos = self._make_position(und_entry=85.0)
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        # Set underlying to 87.0 (+2.35%)
        # The mock returns underlying=87.0 from fetch_price
        trader._check_exits({}, {"XAG-USD": make_signal()})
        # Position should be removed (sold)
        assert len(trader.state["positions"]) == 0

    def test_hard_stop(self):
        """Exit when underlying drops -2%."""
        pos = self._make_position(und_entry=89.0)  # entry at 89, current at 87 = -2.25%
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        trader._check_exits({}, {"XAG-USD": make_signal()})
        assert len(trader.state["positions"]) == 0

    def test_time_limit(self):
        """Exit when held > MAX_HOLD_HOURS."""
        pos = self._make_position(und_entry=87.0, hours_ago=6)
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        trader._check_exits({}, {"XAG-USD": make_signal()})
        assert len(trader.state["positions"]) == 0

    def test_signal_reversal_exit(self):
        """Exit when SELL consensus forms."""
        pos = self._make_position(und_entry=87.0)
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        sell_sig = make_signal(
            action="SELL", sell_count=4,
            timeframes={"Now": "SELL", "12h": "SELL", "2d": "SELL",
                        "7d": "SELL", "1mo": "HOLD", "3mo": "HOLD", "6mo": "HOLD"},
        )
        trader._check_exits({}, {"XAG-USD": sell_sig})
        assert len(trader.state["positions"]) == 0

    def test_hold_when_no_exit_condition(self):
        """Keep position when no exit trigger fires."""
        pos = self._make_position(und_entry=87.0)  # entry=current, no trigger
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        trader._check_exits({}, {"XAG-USD": make_signal()})
        assert len(trader.state["positions"]) == 1

    def test_trailing_stop_activates(self):
        """Trailing stop should activate after +1.5% and exit on pullback."""
        pos = self._make_position(und_entry=85.0)  # current=87, +2.35%
        pos["peak_underlying"] = 88.0  # peak higher than current
        # From peak: (87-88)/88 = -1.14%, which is > -1.0% trail distance
        # So trailing doesn't trigger yet
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        trader._check_exits({}, {"XAG-USD": make_signal()})
        # Should still hold because trailing hasn't been breached (from_peak > -1.0%)
        # Actually (87-88)/88 = -1.14% which is <= -1.0%, so trailing SHOULD trigger
        assert len(trader.state["positions"]) == 0

    def test_consecutive_losses_tracking(self):
        """Losses should increment consecutive_losses counter."""
        pos = self._make_position(und_entry=89.0, entry_price=16.5)  # losing: entry=16.5, bid=15.0
        trader = make_trader(cash=5000, positions={"pos_1": pos})
        trader.state["consecutive_losses"] = 0
        trader._check_exits({}, {"XAG-USD": make_signal()})
        assert trader.state["consecutive_losses"] == 1


# ---------------------------------------------------------------------------
# Warrant selection tests
# ---------------------------------------------------------------------------

class TestWarrantSelection:
    def test_selects_higher_leverage(self):
        """Should prefer AVA 301 (6.3x) over SG (1.56x) when target is 5x."""
        trader = make_trader(cash=10000)
        result = trader._select_warrant("XAG-USD", "LONG")
        assert result is not None
        assert result["key"] == "MINI_L_SILVER_AVA_301"

    def test_no_warrant_for_unknown_underlying(self):
        trader = make_trader(cash=10000)
        result = trader._select_warrant("FOO-USD", "LONG")
        assert result is None


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_compact_signal(self):
        sig = make_signal(action="BUY", buy_count=5, sell_count=2, rsi=55)
        compact = _compact_signal(sig)
        assert compact["action"] == "BUY"
        assert compact["buy"] == 5
        assert compact["sell"] == 2
        assert compact["rsi"] == 55.0

    def test_has_position(self):
        trader = make_trader(positions={
            "pos_1": {"underlying": "XAG-USD"},
        })
        assert trader._has_position("XAG-USD")
        assert not trader._has_position("XAU-USD")

    def test_cooldown_cleared_no_last_buy(self):
        trader = make_trader()
        assert trader._cooldown_cleared()

    def test_cooldown_cleared_after_time(self):
        trader = make_trader()
        trader.state["last_buy_ts"] = (
            _now_utc() - datetime.timedelta(minutes=60)
        ).isoformat()
        assert trader._cooldown_cleared()

    def test_macd_history_tracking(self):
        trader = make_trader()
        sig_data = {"XAG-USD": make_signal(macd_hist=0.5)}
        trader._update_macd_history(sig_data)
        assert "XAG-USD" in trader.state["macd_history"]
        assert trader.state["macd_history"]["XAG-USD"][-1] == 0.5


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cycle_no_signal(self):
        """Full evaluate_and_execute with no signal → no trades."""
        trader = make_trader(cash=10000)
        trader.evaluate_and_execute({}, {})
        assert len(trader.state["positions"]) == 0
        assert trader.check_count == 1

    def test_full_cycle_with_buy_signal(self):
        """Full cycle: BUY signal → creates position (DRY_RUN)."""
        trader = make_trader(cash=10000, macd_history={
            "XAG-USD": [0.1, 0.2, 0.3],
        })
        sig = make_signal(action="BUY", buy_count=5, rsi=50,
                          timeframes={"Now": "BUY", "12h": "BUY", "2d": "BUY",
                                      "7d": "BUY", "1mo": "HOLD", "3mo": "HOLD", "6mo": "HOLD"})
        trader.evaluate_and_execute({}, {"XAG-USD": sig})
        assert len(trader.state["positions"]) == 1
        # Check trade was logged
        assert os.path.exists(cfg.TRADES_LOG)
        assert os.path.exists(cfg.DECISIONS_LOG)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
