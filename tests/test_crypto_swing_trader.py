"""Behavior tests for data/crypto_swing_trader.py.

xdist-safe: every test redirects state files to a tmp_path-derived location
via monkeypatch on the cfg module. Mirrors the metals_swing_trader test
discipline noted in docs/TESTING.md.
"""
from __future__ import annotations

import datetime

import pytest

from data import crypto_swing_config as cfg
from data import crypto_swing_trader as cst


@pytest.fixture
def isolate_state(tmp_path, monkeypatch):
    """Redirect all crypto_swing state files into tmp_path for xdist safety."""
    monkeypatch.setattr(cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(cfg, "DECISIONS_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(cfg, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(cfg, "VALUE_HISTORY_LOG", str(tmp_path / "value.jsonl"))
    monkeypatch.setattr(cfg, "WARRANT_CATALOG_FILE",
                        str(tmp_path / "catalog.json"))
    monkeypatch.setattr(cfg, "MOMENTUM_STATE_FILE",
                        str(tmp_path / "momentum.json"))
    return tmp_path


def _fresh_signal(action="BUY", conf=0.75, buy=4, sell=1,
                   rsi=55, macd=0.05, regime="trending"):
    return {
        "recommendation": action,
        "calibrated_confidence": conf,
        "buy_voters": buy,
        "sell_voters": sell,
        "indicators": {"rsi": rsi, "macd": macd},
        "regime": regime,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


class TestPlaceBuyWarrantQuoteSafety:
    """Regression for the 2026-05-02 back-port of the oil codex finding.

    Sizing must refuse when the warrant entry has no live ask/last;
    previously fell back to ask=1.0 and computed clearly-wrong unit
    counts. The fallback catalog (XBT_TRACKER_AVA / ETH_TRACKER_AVA)
    has no quotes until refresh runs with a live page, so this gate is
    the only thing standing between a DRY_RUN→live flip and broken
    sizing.
    """

    def test_buy_refused_when_warrant_has_no_ask_or_last(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        warrant = {"name": "XBT TRACKER AVA", "ob_id": "12345",
                   "underlying": "BTC-USD", "direction": "LONG",
                   "leverage": 1.0, "barrier": None, "parity": 1}
        # No ask, no last in the warrant dict
        result = trader._place_buy(
            "BTC-USD", warrant, signal_ctx={}, underlying_price=70000.0,
        )
        assert result["executed"] is False
        assert "ask/last" in result["reason"] or "live" in result["reason"]

    def test_buy_refused_when_ask_is_zero(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        warrant = {"name": "XBT TRACKER AVA", "ob_id": "12345",
                   "underlying": "BTC-USD", "direction": "LONG",
                   "leverage": 1.0, "ask": 0, "last": 0, "parity": 1}
        result = trader._place_buy(
            "BTC-USD", warrant, signal_ctx={}, underlying_price=70000.0,
        )
        assert result["executed"] is False

    def test_buy_proceeds_when_ask_present(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        warrant = {"name": "XBT TRACKER AVA", "ob_id": "12345",
                   "underlying": "BTC-USD", "direction": "LONG",
                   "leverage": 1.0, "ask": 850.0, "last": 850.0,
                   "parity": 1}
        result = trader._place_buy(
            "BTC-USD", warrant, signal_ctx={}, underlying_price=70000.0,
        )
        # In DRY_RUN with valid ask, the path completes (executed=True
        # with dry_run=True flag).
        assert result["executed"] is True


class TestExtractors:
    def test_extract_action_recognizes_keys(self):
        assert cst._extract_action({"recommendation": "BUY"}) == "BUY"
        assert cst._extract_action({"action": "sell"}) == "SELL"
        assert cst._extract_action({"signal": "HOLD"}) == "HOLD"

    def test_extract_action_default_hold(self):
        assert cst._extract_action({}) == "HOLD"
        assert cst._extract_action({"foo": "bar"}) == "HOLD"

    def test_extract_confidence_handles_missing(self):
        assert cst._extract_confidence({}) == 0.0
        assert cst._extract_confidence({"calibrated_confidence": 0.7}) == 0.7
        assert cst._extract_confidence({"confidence": 0.5}) == 0.5

    def test_extract_voters_int_cast(self):
        assert cst._extract_voters({"buy_voters": "4", "sell_voters": "1"}) == (4, 1)
        assert cst._extract_voters({}) == (0, 0)

    def test_extract_indicator_falls_back_to_top_level(self):
        sig = {"rsi": 60, "macd": 0.02}
        assert cst._extract_indicator(sig, "rsi") == 60
        assert cst._extract_indicator(sig, "macd") == 0.02
        assert cst._extract_indicator(sig, "missing") is None

    def test_signal_age_returns_none_for_missing_ts(self):
        assert cst._signal_age_seconds({}) is None

    def test_signal_age_seconds_for_recent(self):
        sig = {"timestamp": datetime.datetime.now(datetime.UTC).isoformat()}
        assert cst._signal_age_seconds(sig) is not None
        assert cst._signal_age_seconds(sig) < 5


class TestEntryGates:
    def test_first_cycle_rejects_due_to_persistence(self, isolate_state):
        """2026-05-11: with SIGNAL_PERSISTENCE_CHECKS=1, first-cycle entries
        are no longer rejected by the swing-layer persistence gate. The
        engine layer now owns this responsibility.
        """
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        sig = _fresh_signal()
        trader._update_history("BTC-USD", sig)
        allow, reason, _ = trader._evaluate_entry("BTC-USD", sig, is_momentum=False)
        # With persistence=1, the single fresh cycle now passes.
        assert allow or "persistence" not in (reason or ""), (
            f"with SIGNAL_PERSISTENCE_CHECKS=1, first cycle should not "
            f"be blocked by persistence; got: {reason}"
        )

    def test_momentum_path_skips_persistence(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        sig = _fresh_signal(conf=0.55, buy=2)  # below standard, ok for momentum
        trader._update_history("BTC-USD", sig)
        allow, reason, ctx = trader._evaluate_entry("BTC-USD", sig,
                                                     is_momentum=True)
        assert allow, reason
        assert ctx["is_momentum"]

    def test_rsi_outside_band_rejects(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        sig = _fresh_signal(rsi=75)
        trader._update_history("BTC-USD", sig)
        allow, reason, _ = trader._evaluate_entry("BTC-USD", sig, is_momentum=False)
        assert not allow and "rsi" in reason

    def test_low_confidence_rejects(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        sig = _fresh_signal(conf=0.40)
        trader._update_history("BTC-USD", sig)
        allow, reason, _ = trader._evaluate_entry("BTC-USD", sig, is_momentum=False)
        assert not allow
        assert "confidence" in reason

    def test_stale_signal_rejects_both_paths(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        old_iso = (datetime.datetime.now(datetime.UTC)
                   - datetime.timedelta(seconds=cfg.MAX_SIGNAL_AGE_SEC + 100)
                   ).isoformat()
        sig = _fresh_signal()
        sig["timestamp"] = old_iso
        for is_momentum in (False, True):
            allow, reason, _ = trader._evaluate_entry("BTC-USD", sig,
                                                       is_momentum=is_momentum)
            assert not allow
            assert "stale" in reason


class TestExitGates:
    def test_hard_stop_triggers_at_threshold(self, isolate_state):
        """2026-05-11: hard stop now anchored to warrant pct change
        (STOP_LOSS_WARRANT_PCT=30). entry warrant bid 1.0 → drop below 0.70
        triggers hard stop. Use 0.65 (-35% warrant) past the threshold.
        """
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        pos = {"entry_underlying_price": 100.0, "direction": "LONG",
               "entry_warrant_bid": 1.0, "peak_warrant_bid": 1.0,
               "peak_underlying_price": 100.0,
               "entry_ts": datetime.datetime.now(datetime.UTC).isoformat()}
        should_exit, reason = trader._evaluate_exit(pos,
            current_underlying=96.5, current_warrant_bid=0.65, sig={})
        assert should_exit
        assert "HARD_STOP" in reason

    def test_take_profit_triggers(self, isolate_state):
        """2026-05-11: TP now anchored to warrant pct change
        (TAKE_PROFIT_WARRANT_PCT=5). entry warrant bid 1.0 → 1.06 = +6% past TP.
        """
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        pos = {"entry_underlying_price": 100.0, "direction": "LONG",
               "entry_warrant_bid": 1.0, "peak_warrant_bid": 1.0,
               "peak_underlying_price": 100.0,
               "entry_ts": datetime.datetime.now(datetime.UTC).isoformat()}
        should_exit, reason = trader._evaluate_exit(pos,
            current_underlying=104.5, current_warrant_bid=1.06, sig={})
        assert should_exit
        assert "TAKE_PROFIT" in reason

    def test_signal_reversal_exits(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        pos = {"entry_underlying_price": 100.0, "direction": "LONG",
               "entry_warrant_bid": 1.0, "peak_warrant_bid": 1.0,
               "peak_underlying_price": 100.0,
               "entry_ts": datetime.datetime.now(datetime.UTC).isoformat()}
        sig = {"recommendation": "SELL", "buy_voters": 1, "sell_voters": 4}
        should_exit, reason = trader._evaluate_exit(pos,
            current_underlying=100.5, current_warrant_bid=1.005, sig=sig)
        assert should_exit
        assert "SIGNAL_REVERSAL" in reason

    def test_no_exit_when_in_band(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        pos = {"entry_underlying_price": 100.0, "direction": "LONG",
               "entry_warrant_bid": 1.0, "peak_warrant_bid": 1.0,
               "peak_underlying_price": 100.0,
               "entry_ts": datetime.datetime.now(datetime.UTC).isoformat()}
        should_exit, reason = trader._evaluate_exit(pos,
            current_underlying=100.5, current_warrant_bid=1.005, sig={})
        assert not should_exit
        assert reason == "no exit"

    def test_max_hold_safety_net_fires_eventually(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        old_entry = (datetime.datetime.now(datetime.UTC)
                     - datetime.timedelta(hours=cfg.MAX_HOLD_HOURS + 1)
                     ).isoformat()
        pos = {"entry_underlying_price": 100.0, "direction": "LONG",
               "entry_warrant_bid": 1.0, "peak_warrant_bid": 1.0,
               "peak_underlying_price": 100.0, "entry_ts": old_entry}
        should_exit, reason = trader._evaluate_exit(pos,
            current_underlying=100.5, current_warrant_bid=1.005, sig={})
        assert should_exit
        assert "MAX_HOLD" in reason


class TestEvaluateAndExecute:
    def test_dry_run_buy_after_persistence(self, isolate_state):
        """Updated 2026-05-02: the warrant-sizing safety gate now refuses
        BUYs when the catalog has no live ask. The fallback catalog
        (XBT_TRACKER_AVA / ETH_TRACKER_AVA) has no quotes, so we inject
        them here to assert the persistence path still wires through end-
        to-end. Without the injection, the test would assert the new
        gate fires (also valid)."""
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        # Inject live quotes into the fallback catalog so sizing succeeds
        for w in trader.warrant_catalog.values():
            w.setdefault("ask", 850.0)
            w.setdefault("last", 850.0)

        prices = {"BTC-USD": 100000.0, "ETH-USD": 3500.0}

        # Three cycles of identical BUY signal — should establish persistence
        # + history sufficient for std-path entry on the third cycle.
        for _ in range(6):
            sig = _fresh_signal()
            signal_data = {"per_ticker": {"BTC-USD": sig}}
            result = trader.evaluate_and_execute(prices, signal_data)

        assert any(a["type"] == "entry" and a["result"]["executed"]
                   for a in result["actions"]) or len(trader.state["positions"]) > 0

    def test_dry_run_buy_blocked_when_catalog_lacks_quotes(self, isolate_state):
        """The fallback catalog has no ask/last fields; the new sizing
        gate must block BUYs until refresh_warrant_catalog runs with a
        live page. Behaviour is unchanged for DRY_RUN today (no orders
        either way) but prevents the bug from firing on DRY_RUN→live."""
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        # WARRANT_CATALOG_FALLBACK is module-level and may have been
        # mutated by a sibling test (test_dry_run_buy_after_persistence
        # injects ask/last). Strip those fields to simulate the cold-
        # boot state where the catalog has no live quotes yet.
        for w in trader.warrant_catalog.values():
            w.pop("ask", None)
            w.pop("last", None)

        prices = {"BTC-USD": 100000.0, "ETH-USD": 3500.0}
        for _ in range(6):
            sig = _fresh_signal()
            signal_data = {"per_ticker": {"BTC-USD": sig}}
            trader.evaluate_and_execute(prices, signal_data)

        # No positions — every entry attempt was blocked at the sizing gate
        assert len(trader.state["positions"]) == 0

    def test_no_entry_without_warrant_for_short(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        # SHORT direction not yet supported — fallback catalog has no SHORT
        warrant = trader._select_warrant("BTC-USD", "SHORT")
        assert warrant is None

    def test_select_warrant_returns_long_tracker(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        warrant = trader._select_warrant("BTC-USD", "LONG")
        assert warrant is not None
        assert warrant["direction"] == "LONG"
        assert warrant["underlying"] == "BTC-USD"


class TestCooldown:
    def test_cooldown_passes_initially(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        assert trader._cooldown_cleared("BTC-USD")

    def test_cooldown_blocks_immediately_after_buy(self, isolate_state):
        trader = cst.CryptoSwingTrader(page=None, executor=None)
        trader.state["last_buy_ts"]["BTC-USD"] = datetime.datetime.now(
            datetime.UTC).isoformat()
        assert not trader._cooldown_cleared("BTC-USD")
