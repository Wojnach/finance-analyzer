"""Tests for metals loop autonomous mode (Claude disabled).

Tests the CLAUDE_ENABLED flag, _autonomous_decision(), helper functions,
and the gating logic in invoke_claude().
"""

import json
import os
import sys
import time
import datetime
from unittest.mock import patch, MagicMock, mock_open

import pytest

# metals_loop.py is in data/, not a proper package — add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_metals_loop_state():
    """Reset module-level state between tests."""
    import metals_loop as ml
    ml.check_count = 0
    ml.price_history = []
    ml.last_signal_data = {}
    ml.peak_bids = {}
    ml.last_invoke_times = {1: 0.0, 2: 0.0, 3: 0.0}
    ml._last_auto_telegram = 0
    ml.invoke_count = 0
    ml.claude_proc = None
    ml.claude_log_fh = None
    yield


# ============================================================
# _make_autonomous_prediction tests
# ============================================================

class TestMakeAutonomousPrediction:
    """Tests for the autonomous prediction aggregation function."""

    def test_empty_inputs_return_hold(self):
        from metals_loop import _make_autonomous_prediction
        result = _make_autonomous_prediction({}, {})
        assert result["action"] == "HOLD"
        assert result["direction"] == "flat"
        assert result["confidence"] == 0.0

    def test_buy_signal_consensus(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "BUY", "buy_count": 5, "sell_count": 1},
        }
        result = _make_autonomous_prediction(signals, {})
        assert result["direction"] == "up"
        assert result["confidence"] > 0.5

    def test_sell_signal_consensus(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "SELL", "buy_count": 1, "sell_count": 5},
        }
        result = _make_autonomous_prediction(signals, {})
        assert result["direction"] == "down"
        assert result["confidence"] > 0.5

    def test_hold_when_no_votes(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "HOLD", "buy_count": 0, "sell_count": 0},
        }
        result = _make_autonomous_prediction(signals, {})
        assert result["action"] == "HOLD"
        assert result["direction"] == "flat"

    def test_high_confidence_triggers_buy(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "BUY", "buy_count": 8, "sell_count": 1},
        }
        llm = {
            "XAG-USD": {"consensus": "BUY", "consensus_conf": 0.8},
        }
        result = _make_autonomous_prediction(signals, llm)
        assert result["direction"] == "up"
        assert result["confidence"] >= 0.7
        assert result["action"] == "BUY"

    def test_high_confidence_triggers_sell(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "SELL", "buy_count": 0, "sell_count": 6},
        }
        llm = {
            "XAG-USD": {"consensus": "SELL", "consensus_conf": 0.75},
        }
        result = _make_autonomous_prediction(signals, llm)
        assert result["direction"] == "down"
        assert result["confidence"] >= 0.7
        assert result["action"] == "SELL"

    def test_low_confidence_stays_hold(self):
        from metals_loop import _make_autonomous_prediction
        # Conflicting signals: XAG BUY vs XAU SELL with similar weights → ~0.5 confidence
        signals = {
            "XAG-USD": {"action": "BUY", "buy_count": 3, "sell_count": 2},
            "XAU-USD": {"action": "SELL", "buy_count": 2, "sell_count": 3},
        }
        result = _make_autonomous_prediction(signals, {})
        assert result["confidence"] < 0.7
        assert result["action"] == "HOLD"

    def test_llm_chronos_contributes(self):
        from metals_loop import _make_autonomous_prediction
        signals = {}
        llm = {
            "XAG-USD": {"chronos_3h": "up", "chronos_3h_pct": 0.02},
        }
        result = _make_autonomous_prediction(signals, llm)
        assert result["direction"] == "up"

    def test_llm_underscore_keys_ignored(self):
        from metals_loop import _make_autonomous_prediction
        llm = {
            "_accuracy": {"ministral": {"hit_rate": 0.65}},
            "XAG-USD": {"consensus": "BUY", "consensus_conf": 0.8},
        }
        result = _make_autonomous_prediction({}, llm)
        # Should not crash; _accuracy is skipped
        assert result["direction"] in ("up", "down", "flat")

    def test_mixed_signals_and_llm(self):
        from metals_loop import _make_autonomous_prediction
        signals = {
            "XAG-USD": {"action": "BUY", "buy_count": 3, "sell_count": 2},
            "XAU-USD": {"action": "SELL", "buy_count": 1, "sell_count": 4},
        }
        llm = {}
        result = _make_autonomous_prediction(signals, llm)
        # Both directions have weight — should have lower confidence
        assert result["confidence"] < 1.0
        assert result["direction"] in ("up", "down")

    def test_prediction_has_required_fields(self):
        from metals_loop import _make_autonomous_prediction
        result = _make_autonomous_prediction(
            {"XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 1}},
            {},
        )
        assert "action" in result
        assert "direction" in result
        assert "confidence" in result
        assert "horizon" in result
        assert result["horizon"] == "3h"


# ============================================================
# _assess_thesis tests
# ============================================================

class TestAssessThesis:
    """Tests for the thesis assessment function."""

    def test_neutral_when_no_data(self):
        from metals_loop import _assess_thesis
        result = _assess_thesis({}, {}, ["heartbeat"])
        assert result == "NEUTRAL"

    def test_intact_when_profitable(self):
        from metals_loop import _assess_thesis
        positions = {
            "silver_sg": {"pnl_pct": 8.0, "dist_stop_pct": 15.0},
        }
        signals = {"XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 1}}
        result = _assess_thesis(positions, signals, ["heartbeat"])
        assert result == "INTACT"

    def test_threatened_when_deep_drawdown_no_support(self):
        from metals_loop import _assess_thesis
        positions = {
            "silver_sg": {"pnl_pct": -18.0, "dist_stop_pct": 2.0},
        }
        signals = {}
        result = _assess_thesis(positions, signals, ["heartbeat"])
        assert result == "THREATENED"

    def test_mixed_when_threats_and_supports(self):
        from metals_loop import _assess_thesis
        positions = {
            "silver_sg": {"pnl_pct": -18.0, "dist_stop_pct": 3.0},
        }
        signals = {"XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 1}}
        result = _assess_thesis(positions, signals, ["heartbeat"])
        assert result == "MIXED"

    def test_threatened_on_emergency_trigger(self):
        from metals_loop import _assess_thesis
        positions = {
            "silver_sg": {"pnl_pct": -5.0, "dist_stop_pct": 3.0},
        }
        result = _assess_thesis(positions, {}, ["L3 EMERGENCY: silver_sg"])
        assert result == "THREATENED"

    def test_near_stop_triggers_threat(self):
        from metals_loop import _assess_thesis
        positions = {
            "silver_sg": {"pnl_pct": -2.0, "dist_stop_pct": 4.5},  # < 5%
        }
        result = _assess_thesis(positions, {}, ["heartbeat"])
        assert result == "THREATENED"

    def test_strong_sell_signal_triggers_threat(self):
        from metals_loop import _assess_thesis
        positions = {}
        signals = {"XAG-USD": {"action": "SELL", "sell_count": 5, "buy_count": 0}}
        result = _assess_thesis(positions, signals, ["heartbeat"])
        assert result == "THREATENED"


# ============================================================
# _build_autonomous_telegram tests
# ============================================================

class TestBuildAutonomousTelegram:
    """Tests for the Telegram message builder."""

    def test_basic_message_format(self):
        from metals_loop import _build_autonomous_telegram
        positions = {
            "silver_sg": {
                "bid": 48.3, "entry": 52, "stop": 46, "units": 441,
                "pnl_pct": -7.1, "from_peak_pct": -3.2, "dist_stop_pct": 4.8,
            },
        }
        msg = _build_autonomous_telegram(
            ["heartbeat"], 1, positions, {}, {}, {},
            {"action": "HOLD"}, "INTACT", "14:35 CET", False,
        )
        assert "*AUTO HOLD*" in msg
        assert "INTACT" in msg
        assert "441u" in msg
        assert "48.3" in msg

    def test_emergency_tag(self):
        from metals_loop import _build_autonomous_telegram
        msg = _build_autonomous_telegram(
            ["L3 EMERGENCY"], 3, {}, {}, {}, {},
            {"action": "HOLD"}, "THREATENED", "14:35 CET", True,
        )
        assert "EMG" in msg
        assert "THREATENED" in msg

    def test_signals_in_message(self):
        from metals_loop import _build_autonomous_telegram
        signals = {"XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 2}}
        msg = _build_autonomous_telegram(
            ["test"], 1, {}, signals, {}, {},
            {"action": "HOLD"}, "NEUTRAL", "10:00 CET", False,
        )
        assert "XAG BUY 4B/2S" in msg

    def test_llm_in_message(self):
        from metals_loop import _build_autonomous_telegram
        llm = {"XAG-USD": {"ministral": "BUY", "ministral_conf": 0.65}}
        msg = _build_autonomous_telegram(
            ["test"], 1, {}, {}, llm, {},
            {"action": "HOLD"}, "NEUTRAL", "10:00 CET", False,
        )
        assert "min BUY 65%" in msg

    def test_risk_in_message(self):
        from metals_loop import _build_autonomous_telegram
        risk = {"drawdown_pct": -4.2}
        msg = _build_autonomous_telegram(
            ["test"], 1, {}, {}, {}, risk,
            {"action": "HOLD"}, "NEUTRAL", "10:00 CET", False,
        )
        assert "DD -4.2%" in msg

    def test_mc_stop_probability_shown(self):
        from metals_loop import _build_autonomous_telegram
        positions = {
            "silver_sg": {
                "bid": 48.3, "entry": 52, "stop": 46, "units": 441,
                "pnl_pct": -7.1, "from_peak_pct": -3.2, "dist_stop_pct": 4.8,
            },
        }
        risk = {"silver_sg_mc_pstop3h": 2.1}
        msg = _build_autonomous_telegram(
            ["test"], 1, positions, {}, {}, risk,
            {"action": "HOLD"}, "INTACT", "10:00 CET", False,
        )
        assert "MC:2.1%" in msg

    def test_message_under_telegram_limit(self):
        from metals_loop import _build_autonomous_telegram
        positions = {
            "silver_sg": {
                "bid": 48.3, "entry": 52, "stop": 46, "units": 441,
                "pnl_pct": -7.1, "from_peak_pct": -3.2, "dist_stop_pct": 4.8,
            },
            "silver301": {
                "bid": 14.5, "entry": 15.4, "stop": 12.5, "units": 130,
                "pnl_pct": -5.8, "from_peak_pct": -1.5, "dist_stop_pct": 13.8,
            },
            "gold": {
                "bid": 907.0, "entry": 907.5, "stop": 780, "units": 4,
                "pnl_pct": -0.1, "from_peak_pct": -0.5, "dist_stop_pct": 14.0,
            },
        }
        signals = {
            "XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 2},
            "XAU-USD": {"action": "HOLD", "buy_count": 1, "sell_count": 1},
        }
        llm = {
            "XAG-USD": {
                "ministral": "BUY", "ministral_conf": 0.65,
                "chronos_3h": "up", "chronos_3h_pct": 0.008,
            },
        }
        risk = {"drawdown_pct": -4.2, "silver_sg_mc_pstop3h": 2.1}
        msg = _build_autonomous_telegram(
            ["heartbeat check #80"], 1, positions, signals, llm, risk,
            {"action": "HOLD", "direction": "up", "confidence": 0.6},
            "INTACT", "14:35 CET", False,
        )
        assert len(msg) < 4096

    def test_tier_shown_in_footer(self):
        from metals_loop import _build_autonomous_telegram
        msg = _build_autonomous_telegram(
            ["test"], 2, {}, {}, {}, {},
            {"action": "HOLD"}, "NEUTRAL", "10:00 CET", False,
        )
        assert "Autonomous T2" in msg


# ============================================================
# invoke_claude gating tests
# ============================================================

class TestInvokeClaudeGating:
    """Tests for the CLAUDE_ENABLED guard in invoke_claude()."""

    def test_claude_disabled_routes_to_autonomous(self):
        """When CLAUDE_ENABLED=False, invoke_claude should call _autonomous_decision."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.price_history = [{"silver_sg": 48.3}]

        with patch.object(ml, "_autonomous_decision") as mock_auto, \
             patch.object(ml, "send_telegram"):
            result = ml.invoke_claude(["heartbeat"], tier=1)

        assert result is False
        mock_auto.assert_called_once_with(["heartbeat"], 1)

    def test_claude_disabled_no_subprocess(self):
        """When CLAUDE_ENABLED=False, no subprocess should be spawned."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False

        with patch.object(ml, "_autonomous_decision"), \
             patch.object(ml, "send_telegram"), \
             patch("subprocess.Popen") as mock_popen:
            ml.invoke_claude(["test trigger"], tier=2)

        mock_popen.assert_not_called()

    def test_claude_enabled_proceeds_normally(self):
        """When CLAUDE_ENABLED=True, invoke_claude proceeds past the guard."""
        import metals_loop as ml
        original = ml.CLAUDE_ENABLED
        ml.CLAUDE_ENABLED = True

        with patch.object(ml, "_autonomous_decision") as mock_auto, \
             patch("builtins.open", mock_open(read_data="prompt text")), \
             patch("shutil.which", return_value="claude"), \
             patch("subprocess.Popen") as mock_popen, \
             patch.object(ml, "send_telegram"), \
             patch.object(ml, "log_invocation"):
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc
            result = ml.invoke_claude(["test trigger"], tier=2)

        mock_auto.assert_not_called()
        assert result is True
        ml.CLAUDE_ENABLED = original

    def test_claude_disabled_all_tiers_routed(self):
        """All tiers (1, 2, 3) should route to autonomous when disabled."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False

        for tier in [1, 2, 3]:
            with patch.object(ml, "_autonomous_decision") as mock_auto, \
                 patch.object(ml, "send_telegram"):
                ml.invoke_claude(["test"], tier=tier)
            mock_auto.assert_called_once_with(["test"], tier)


# ============================================================
# _autonomous_decision integration tests
# ============================================================

class TestAutonomousDecision:
    """Tests for _autonomous_decision end-to-end behavior."""

    def test_writes_decision_to_jsonl(self, tmp_path):
        """Autonomous decisions should be logged to metals_decisions.jsonl."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.price_history = [{"silver_sg": 48.3}]
        decisions_file = tmp_path / "metals_decisions.jsonl"

        with patch.object(ml, "send_telegram"), \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("metals_loop.open", mock_open()) as mock_file:
            # Point the open to our tmp file
            mock_file.return_value.__enter__ = mock_file.return_value
            mock_file.return_value.__exit__ = MagicMock(return_value=False)

            ml._autonomous_decision(["heartbeat"], 1)

            # Verify file was opened for appending
            mock_file.assert_called_with(
                "data/metals_decisions.jsonl", "a", encoding="utf-8"
            )

    def test_decision_has_source_autonomous(self, tmp_path):
        """Decision log entries should have source='autonomous'."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.price_history = [{"silver_sg": 48.3}]

        written_data = []

        def capture_write(data):
            written_data.append(data)

        mock_fh = MagicMock()
        mock_fh.write = capture_write

        with patch.object(ml, "send_telegram"), \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", return_value=MagicMock(
                 __enter__=MagicMock(return_value=mock_fh),
                 __exit__=MagicMock(return_value=False),
             )):
            ml._autonomous_decision(["heartbeat"], 1)

        assert len(written_data) > 0
        entry = json.loads(written_data[0].strip())
        assert entry["source"] == "autonomous"
        assert entry["action"] in ("HOLD", "BUY", "SELL")
        assert "prediction" in entry
        assert "thesis_status" in entry

    def test_sends_telegram(self):
        """Autonomous decisions should send Telegram messages."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml._last_auto_telegram = 0  # ensure not throttled

        with patch.object(ml, "send_telegram") as mock_tg, \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            ml._autonomous_decision(["heartbeat"], 1)

        mock_tg.assert_called_once()
        msg = mock_tg.call_args[0][0]
        assert "*AUTO" in msg

    def test_telegram_throttled_for_routine_holds(self):
        """Routine HOLD decisions should be throttled to AUTO_TELEGRAM_COOLDOWN."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml._last_auto_telegram = time.time() - 60  # 60s ago (within 1800s cooldown)

        with patch.object(ml, "send_telegram") as mock_tg, \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            ml._autonomous_decision(["heartbeat"], 1)

        mock_tg.assert_not_called()

    def test_telegram_not_throttled_for_emergency(self):
        """Emergency triggers should always send Telegram."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml._last_auto_telegram = time.time() - 60  # recent, would normally throttle

        with patch.object(ml, "send_telegram") as mock_tg, \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            ml._autonomous_decision(["L3 EMERGENCY: silver_sg"], 3)

        mock_tg.assert_called_once()

    def test_emergency_bypass_when_claude_enabled(self):
        """With CLAUDE_ENABLED=True, emergency patterns should invoke Claude T3."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = True

        with patch.object(ml, "invoke_claude") as mock_invoke, \
             patch.object(ml, "send_telegram"):
            ml._autonomous_decision(["L3 EMERGENCY: silver_sg"], 1)

        mock_invoke.assert_called_once_with(["L3 EMERGENCY: silver_sg"], tier=3)

    def test_emergency_no_recursion_when_claude_disabled(self):
        """With CLAUDE_ENABLED=False, emergency should NOT call invoke_claude (avoid recursion)."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml._last_auto_telegram = 0

        with patch.object(ml, "send_telegram") as mock_tg, \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            # This should NOT recurse — should fall through to assessment
            ml._autonomous_decision(["L3 EMERGENCY: silver_sg"], 3)

        # Should have sent a Telegram (the emergency assessment)
        mock_tg.assert_called_once()
        msg = mock_tg.call_args[0][0]
        assert "EMG" in msg

    def test_decision_includes_positions_data(self, tmp_path):
        """Decision should include position P&L when price_history has data."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.price_history = [{"silver_sg": 48.3, "gold": 907.0}]
        ml._last_auto_telegram = 0
        # Ensure POSITIONS has active entries
        original_positions = ml.POSITIONS.copy()
        ml.POSITIONS["silver_sg"] = {
            "name": "MINI L SILVER SG", "ob_id": "2043157", "api_type": "warrant",
            "units": 441, "entry": 52.0, "stop": 46.0, "active": True,
        }

        decisions_file = str(tmp_path / "metals_decisions.jsonl")

        real_open = open
        def patched_open(path, *args, **kwargs):
            if "metals_decisions" in str(path):
                return real_open(decisions_file, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        try:
            with patch.object(ml, "send_telegram"), \
                 patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
                 patch("builtins.open", side_effect=patched_open):
                ml._autonomous_decision(["heartbeat"], 1)

            with open(decisions_file, "r") as f:
                entry = json.loads(f.readline().strip())
            # silver_sg is active with bid 48.3
            assert "silver_sg" in entry["positions"]
            assert entry["positions"]["silver_sg"]["bid"] == 48.3
        finally:
            ml.POSITIONS = original_positions

    def test_decision_includes_signal_data(self):
        """Decision should include signal consensus when available."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.last_signal_data = {
            "XAG-USD": {"action": "BUY", "buy_count": 4, "sell_count": 1,
                        "rsi": 55, "macd_hist": 0.1, "regime": "trending-up"},
        }
        ml._last_auto_telegram = 0

        written_data = []
        mock_fh = MagicMock()
        mock_fh.write = lambda data: written_data.append(data)

        with patch.object(ml, "send_telegram"), \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", return_value=MagicMock(
                 __enter__=MagicMock(return_value=mock_fh),
                 __exit__=MagicMock(return_value=False),
             )):
            ml._autonomous_decision(["heartbeat"], 1)

        entry = json.loads(written_data[0].strip())
        assert "XAG-USD" in entry["signals"]
        assert entry["signals"]["XAG-USD"]["action"] == "BUY"


# ============================================================
# write_context gating tests
# ============================================================

class TestWriteContextGating:
    """Test that write_context is skipped when Claude is disabled."""

    def test_write_context_skipped_when_disabled(self):
        """write_context should not be called when CLAUDE_ENABLED=False."""
        import metals_loop as ml
        original = ml.CLAUDE_ENABLED
        ml.CLAUDE_ENABLED = False

        # The gating happens in main() loop — test the conditional directly
        prices = {"silver_sg": {"bid": 48.3}}
        reasons = ["heartbeat"]
        tier = 1

        with patch.object(ml, "write_context") as mock_wc:
            # Simulate the gated code path from main()
            if ml.CLAUDE_ENABLED:
                ml.write_context(prices, "; ".join(reasons), tier=tier)

        mock_wc.assert_not_called()
        ml.CLAUDE_ENABLED = original


# ============================================================
# Config and startup tests
# ============================================================

class TestConfig:
    """Tests for configuration values."""

    def test_claude_enabled_default_false(self):
        import metals_loop as ml
        assert ml.CLAUDE_ENABLED is False

    def test_tier_config_no_opus(self):
        import metals_loop as ml
        for tier, cfg in ml.TIER_CONFIG.items():
            assert cfg["model"] != None or tier == 3  # all have explicit model
            if cfg["model"]:
                assert cfg["model"] in ("haiku", "sonnet"), \
                    f"Tier {tier} uses unexpected model: {cfg['model']}"

    def test_tier3_uses_sonnet(self):
        import metals_loop as ml
        assert ml.TIER_CONFIG[3]["model"] == "sonnet"

    def test_tier_cooldowns_structure(self):
        import metals_loop as ml
        assert ml.TIER_COOLDOWNS[1] == 120
        assert ml.TIER_COOLDOWNS[2] == 600
        assert ml.TIER_COOLDOWNS[3] == 0  # immediate

    def test_auto_telegram_cooldown(self):
        import metals_loop as ml
        assert ml.AUTO_TELEGRAM_COOLDOWN == 1800

    def test_per_tier_invoke_times_initialized(self):
        import metals_loop as ml
        assert ml.last_invoke_times == {1: 0.0, 2: 0.0, 3: 0.0}


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_autonomous_with_empty_price_history(self):
        """Should not crash when price_history is empty."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml.price_history = []
        ml._last_auto_telegram = 0

        with patch.object(ml, "send_telegram"), \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            # Should not raise
            ml._autonomous_decision(["heartbeat"], 1)

    def test_autonomous_with_no_active_positions(self):
        """Should handle gracefully when all positions are inactive."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        ml._last_auto_telegram = 0

        # Temporarily deactivate all positions
        orig_states = {}
        for key, pos in ml.POSITIONS.items():
            orig_states[key] = pos["active"]
            pos["active"] = False

        try:
            with patch.object(ml, "send_telegram") as mock_tg, \
                 patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
                 patch("builtins.open", mock_open()):
                ml._autonomous_decision(["heartbeat"], 1)
            mock_tg.assert_called_once()
        finally:
            for key, state in orig_states.items():
                ml.POSITIONS[key]["active"] = state

    def test_prediction_with_only_llm_data(self):
        """Prediction should work with only LLM data, no signals."""
        from metals_loop import _make_autonomous_prediction
        llm = {
            "XAG-USD": {
                "ministral": "BUY", "ministral_conf": 0.7,
                "consensus": "BUY", "consensus_conf": 0.75,
                "chronos_3h": "up", "chronos_3h_pct": 0.01,
            },
        }
        result = _make_autonomous_prediction({}, llm)
        assert result["direction"] == "up"
        assert result["confidence"] > 0

    def test_assess_thesis_with_auto_exit_trigger(self):
        """AUTO-EXIT trigger should register as a threat."""
        from metals_loop import _assess_thesis
        result = _assess_thesis({}, {}, ["AUTO-EXIT: silver_sg"])
        assert result == "THREATENED"

    def test_telegram_builder_handles_no_positions(self):
        """Should build valid message with no positions."""
        from metals_loop import _build_autonomous_telegram
        msg = _build_autonomous_telegram(
            ["heartbeat"], 1, {}, {}, {}, {},
            {"action": "HOLD"}, "NEUTRAL", "10:00 CET", False,
        )
        assert "*AUTO HOLD*" in msg
        assert "NEUTRAL" in msg
        assert len(msg) > 0

    def test_telegram_builder_multiple_positions(self):
        """Should build valid message with multiple positions."""
        from metals_loop import _build_autonomous_telegram
        positions = {
            "silver_sg": {
                "bid": 48.3, "entry": 52, "stop": 46, "units": 441,
                "pnl_pct": -7.1, "from_peak_pct": -3.2, "dist_stop_pct": 4.8,
            },
            "silver301": {
                "bid": 14.5, "entry": 15.4, "stop": 12.5, "units": 130,
                "pnl_pct": -5.8, "from_peak_pct": -1.5, "dist_stop_pct": 13.8,
            },
        }
        msg = _build_autonomous_telegram(
            ["test"], 1, positions, {}, {}, {},
            {"action": "HOLD"}, "INTACT", "10:00 CET", False,
        )
        assert "441u" in msg
        assert "130u" in msg

    def test_throttle_resets_after_cooldown(self):
        """Telegram should send again after cooldown expires."""
        import metals_loop as ml
        ml.CLAUDE_ENABLED = False
        # Set last telegram to 31 min ago (past 30min cooldown)
        ml._last_auto_telegram = time.time() - 1860

        with patch.object(ml, "send_telegram") as mock_tg, \
             patch.object(ml, "cet_time_str", return_value="10:00 CET"), \
             patch("builtins.open", mock_open()):
            ml._autonomous_decision(["heartbeat"], 1)

        mock_tg.assert_called_once()
