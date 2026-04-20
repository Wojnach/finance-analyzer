"""Tests for portfolio.loop_contract — runtime invariant verification.

Covers:
- CycleReport construction and properties
- Each of the 10 contract invariants
- Severity classification
- ViolationTracker escalation (3x warning → critical)
- ViolationTracker persistence
- Self-healing cooldown
- verify_and_act full pipeline
- Integration with _run_post_cycle report tracking
"""

import json
import time
from unittest import mock
from unittest.mock import patch

import pytest

from portfolio.loop_contract import (
    BOT_ERROR_WARNING_THRESHOLD,
    BOT_MAX_CYCLE_DURATION_S,
    ESCALATION_THRESHOLD,
    MAX_CYCLE_DURATION_S,
    METALS_MAX_CYCLE_DURATION_S,
    SELF_HEAL_COOLDOWN_S,
    BotCycleReport,
    CycleReport,
    MetalsCycleReport,
    Violation,
    ViolationTracker,
    _build_heal_prompt,
    verify_and_act,
    verify_bot_contract,
    verify_contract,
    verify_metals_contract,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def good_report():
    """A CycleReport where everything succeeded — should pass all checks."""
    report = CycleReport(
        cycle_id=42,
        active_tickers={"BTC-USD", "ETH-USD", "AAPL"},
    )
    report.signals_ok = 3
    report.signals_failed = 0
    report.signals = {
        "BTC-USD": {"action": "HOLD", "confidence": 0.5, "extra": {"active_voters": 25}},
        "ETH-USD": {"action": "BUY", "confidence": 0.7, "extra": {"active_voters": 25}},
        "AAPL": {"action": "SELL", "confidence": 0.6, "extra": {"active_voters": 20}},
    }
    report.cycle_start = 1000.0
    report.cycle_end = 1045.0  # 45s — well within limit
    report.llm_batch_flushed = True
    report.health_updated = True
    report.heartbeat_updated = True
    report.summary_written = True
    report.post_cycle_results = {
        "daily_digest": True,
        "message_throttle": True,
    }
    return report


@pytest.fixture()
def tracker_dir(tmp_path, monkeypatch):
    """Redirect contract state to temp dir."""
    state_file = tmp_path / "contract_state.json"
    log_file = tmp_path / "contract_violations.jsonl"
    monkeypatch.setattr("portfolio.loop_contract.CONTRACT_STATE_FILE", state_file)
    monkeypatch.setattr("portfolio.loop_contract.CONTRACT_LOG_FILE", log_file)
    return tmp_path


# ---------------------------------------------------------------------------
# CycleReport
# ---------------------------------------------------------------------------

class TestCycleReport:
    """CycleReport dataclass construction and properties."""

    def test_defaults(self):
        r = CycleReport(cycle_id=1)
        assert r.cycle_id == 1
        assert r.active_tickers == set()
        assert r.signals_ok == 0
        assert r.signals_failed == 0
        assert r.signals == {}
        assert r.llm_batch_flushed is False
        assert r.health_updated is False
        assert r.heartbeat_updated is False
        assert r.summary_written is False
        assert r.post_cycle_results == {}
        assert r.errors == []

    def test_cycle_duration(self):
        r = CycleReport(cycle_id=1)
        r.cycle_start = 100.0
        r.cycle_end = 145.5
        assert r.cycle_duration_s == pytest.approx(45.5)

    def test_cycle_duration_zero_when_not_set(self):
        r = CycleReport(cycle_id=1)
        assert r.cycle_duration_s == 0.0


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

class TestVerifyContract:
    """Each invariant is tested for detection and correct severity."""

    def test_clean_report_passes(self, good_report):
        violations = verify_contract(good_report)
        assert violations == []

    # 1. all_tickers_processed
    def test_missing_tickers_detected(self, good_report):
        good_report.signals_ok = 2  # 3 active but only 2 processed
        good_report.signals_failed = 0
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "all_tickers_processed"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"
        assert "silently vanished" in v[0].message

    def test_all_tickers_processed_when_counts_match(self, good_report):
        # 2 ok + 1 failed = 3 active — should pass
        good_report.signals_ok = 2
        good_report.signals_failed = 1
        violations = verify_contract(good_report)
        assert not any(v.invariant == "all_tickers_processed" for v in violations)

    def test_empty_active_tickers_skips_check(self):
        report = CycleReport(cycle_id=1, active_tickers=set())
        violations = verify_contract(report)
        assert not any(v.invariant == "all_tickers_processed" for v in violations)
        assert not any(v.invariant == "min_success_rate" for v in violations)

    # 2. min_success_rate
    def test_low_success_rate_detected(self, good_report):
        good_report.signals_ok = 1
        good_report.signals_failed = 2  # 33% success
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "min_success_rate"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_exactly_50_percent_passes(self, good_report):
        good_report.active_tickers = {"A", "B"}
        good_report.signals_ok = 1
        good_report.signals_failed = 1
        good_report.signals = {
            "A": {"action": "HOLD", "confidence": 0.5, "extra": {}},
        }
        violations = verify_contract(good_report)
        assert not any(v.invariant == "min_success_rate" for v in violations)

    # 3. cycle_duration
    def test_slow_cycle_detected(self, good_report):
        good_report.cycle_start = 1000.0
        good_report.cycle_end = 1000.0 + MAX_CYCLE_DURATION_S + 1
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "cycle_duration"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"
        assert "hanging" in v[0].message

    def test_fast_cycle_passes(self, good_report):
        good_report.cycle_start = 1000.0
        good_report.cycle_end = 1050.0  # 50s
        violations = verify_contract(good_report)
        assert not any(v.invariant == "cycle_duration" for v in violations)

    # 4. llm_batch_flushed
    def test_llm_not_flushed_detected(self, good_report):
        good_report.llm_batch_flushed = False
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "llm_batch_flushed"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    # 5. valid_signals
    def test_missing_action_detected(self, good_report):
        good_report.signals["BTC-USD"] = {"confidence": 0.5, "extra": {}}
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "valid_signals"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"
        assert "BTC-USD" in v[0].message

    def test_none_action_detected(self, good_report):
        good_report.signals["ETH-USD"]["action"] = None
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "valid_signals"]
        assert len(v) == 1

    def test_missing_confidence_detected(self, good_report):
        del good_report.signals["AAPL"]["confidence"]
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "valid_signals"]
        assert len(v) == 1

    def test_non_dict_signal_detected(self, good_report):
        good_report.signals["BTC-USD"] = "not a dict"
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "valid_signals"]
        assert len(v) == 1

    # 6. health_updated
    def test_health_not_updated_detected(self, good_report):
        good_report.health_updated = False
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "health_updated"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    # 7. summary_written
    def test_summary_not_written_detected(self, good_report):
        good_report.summary_written = False
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "summary_written"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    # 8. signal_count_stable
    def test_signal_count_drop_detected(self, good_report):
        previous = {"BTC-USD": 25, "ETH-USD": 25, "AAPL": 20}
        # Drop BTC from 25 to 10 (60% drop)
        good_report.signals["BTC-USD"]["extra"]["active_voters"] = 10
        violations = verify_contract(good_report, previous_signal_counts=previous)
        v = [v for v in violations if v.invariant == "signal_count_stable"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"
        assert "BTC-USD" in v[0].message

    def test_signal_count_stable_when_no_drop(self, good_report):
        previous = {"BTC-USD": 25, "ETH-USD": 25, "AAPL": 20}
        violations = verify_contract(good_report, previous_signal_counts=previous)
        assert not any(v.invariant == "signal_count_stable" for v in violations)

    def test_signal_count_skipped_without_previous(self, good_report):
        violations = verify_contract(good_report, previous_signal_counts=None)
        assert not any(v.invariant == "signal_count_stable" for v in violations)

    def test_signal_count_skipped_for_new_ticker(self, good_report):
        previous = {"ETH-USD": 25}  # BTC and AAPL are new
        violations = verify_contract(good_report, previous_signal_counts=previous)
        assert not any(v.invariant == "signal_count_stable" for v in violations)

    # 9. heartbeat_updated
    def test_heartbeat_not_updated_detected(self, good_report):
        good_report.heartbeat_updated = False
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "heartbeat_updated"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    # 10. post_cycle_complete
    def test_failed_post_cycle_task_detected(self, good_report):
        good_report.post_cycle_results["daily_digest"] = False
        violations = verify_contract(good_report)
        v = [v for v in violations if v.invariant == "post_cycle_complete"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"
        assert "daily_digest" in v[0].message

    def test_all_post_cycle_tasks_pass(self, good_report):
        violations = verify_contract(good_report)
        assert not any(v.invariant == "post_cycle_complete" for v in violations)

    def test_empty_post_cycle_results_passes(self, good_report):
        good_report.post_cycle_results = {}
        violations = verify_contract(good_report)
        assert not any(v.invariant == "post_cycle_complete" for v in violations)


# ---------------------------------------------------------------------------
# ViolationTracker
# ---------------------------------------------------------------------------

class TestViolationTracker:
    """Escalation, persistence, and signal count tracking."""

    def test_clean_cycle_resets_counts(self, good_report, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        tracker._consecutive["health_updated"] = 2
        tracker._save()

        # Clean cycle — no violations
        result = tracker.update([], good_report)
        assert result == []
        assert "health_updated" not in tracker._consecutive

    def test_warning_escalates_after_threshold(self, good_report, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")

        warning = Violation(
            invariant="health_updated",
            severity="WARNING",
            message="Health not updated",
        )
        # Simulate ESCALATION_THRESHOLD consecutive cycles
        for i in range(ESCALATION_THRESHOLD):
            result = tracker.update([warning], good_report)

        # Last update should have escalated
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"
        assert "ESCALATED" in result[0].message

    def test_critical_stays_critical(self, good_report, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        critical = Violation(
            invariant="all_tickers_processed",
            severity="CRITICAL",
            message="Tickers missing",
        )
        result = tracker.update([critical], good_report)
        assert result[0].severity == "CRITICAL"

    def test_consecutive_count_resets_on_pass(self, good_report, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        warning = Violation(
            invariant="llm_batch_flushed",
            severity="WARNING",
            message="LLM not flushed",
        )
        # Accumulate 2 consecutive failures
        tracker.update([warning], good_report)
        tracker.update([warning], good_report)
        assert tracker._consecutive["llm_batch_flushed"] == 2

        # Now pass — count should reset
        tracker.update([], good_report)
        assert "llm_batch_flushed" not in tracker._consecutive

    def test_persistence_survives_reload(self, good_report, tracker_dir):
        state_file = tracker_dir / "contract_state.json"
        tracker1 = ViolationTracker(state_file)
        warning = Violation(
            invariant="heartbeat_updated",
            severity="WARNING",
            message="Heartbeat stale",
        )
        tracker1.update([warning], good_report)
        tracker1.update([warning], good_report)

        # Reload from disk
        tracker2 = ViolationTracker(state_file)
        assert tracker2._consecutive.get("heartbeat_updated") == 2

    def test_signal_counts_updated(self, good_report, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        tracker.update([], good_report)
        assert tracker.previous_signal_counts["BTC-USD"] == 25
        assert tracker.previous_signal_counts["AAPL"] == 20

    def test_self_heal_cooldown(self, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        assert tracker.can_self_heal() is True

        tracker.record_heal()
        assert tracker.can_self_heal() is False

    def test_self_heal_cooldown_expires(self, tracker_dir):
        tracker = ViolationTracker(tracker_dir / "contract_state.json")
        tracker._last_heal_time = time.time() - SELF_HEAL_COOLDOWN_S - 1
        assert tracker.can_self_heal() is True


# ---------------------------------------------------------------------------
# Self-healing prompt
# ---------------------------------------------------------------------------

class TestBuildHealPrompt:
    """Prompt construction for self-healing sessions."""

    def test_includes_violation_details(self):
        violations = [
            Violation(
                invariant="min_success_rate",
                severity="CRITICAL",
                message="Only 30% success",
                details={"success_rate": 0.3},
            ),
        ]
        prompt = _build_heal_prompt(violations)
        assert "min_success_rate" in prompt
        assert "CRITICAL" in prompt
        assert "30%" in prompt
        assert "portfolio/main.py" in prompt

    def test_multiple_violations(self):
        violations = [
            Violation(invariant="a", severity="CRITICAL", message="A failed"),
            Violation(invariant="b", severity="CRITICAL", message="B failed"),
        ]
        prompt = _build_heal_prompt(violations)
        assert "A failed" in prompt
        assert "B failed" in prompt


# ---------------------------------------------------------------------------
# verify_and_act full pipeline
# ---------------------------------------------------------------------------

class TestVerifyAndAct:
    """Integration test for the full verification pipeline."""

    def test_clean_report_no_side_effects(self, good_report, tracker_dir):
        with patch("portfolio.loop_contract._log_violations") as mock_log, \
             patch("portfolio.loop_contract._alert_violations") as mock_alert, \
             patch("portfolio.loop_contract._trigger_self_heal") as mock_heal:
            verify_and_act(good_report, {})
            mock_log.assert_not_called()
            mock_alert.assert_not_called()
            mock_heal.assert_not_called()

    def test_warning_logs_and_alerts(self, good_report, tracker_dir):
        good_report.health_updated = False
        with patch("portfolio.loop_contract._log_violations") as mock_log, \
             patch("portfolio.loop_contract._alert_violations") as mock_alert, \
             patch("portfolio.loop_contract._trigger_self_heal") as mock_heal:
            verify_and_act(good_report, {})
            mock_log.assert_called_once()
            mock_alert.assert_called_once()
            # Warning → no self-heal on first occurrence
            mock_heal.assert_not_called()

    def test_critical_triggers_self_heal(self, good_report, tracker_dir):
        good_report.signals_ok = 0
        good_report.signals_failed = 3  # 0% success → CRITICAL
        with patch("portfolio.loop_contract._log_violations") as mock_log, \
             patch("portfolio.loop_contract._alert_violations") as mock_alert, \
             patch("portfolio.loop_contract._trigger_self_heal") as mock_heal:
            verify_and_act(good_report, {})
            mock_log.assert_called_once()
            mock_alert.assert_called_once()
            mock_heal.assert_called_once()

    def test_escalated_warning_triggers_self_heal(self, good_report, tracker_dir):
        state_file = tracker_dir / "contract_state.json"
        tracker = ViolationTracker(state_file)
        # Pre-load 2 consecutive failures for health_updated
        tracker._consecutive["health_updated"] = ESCALATION_THRESHOLD - 1
        tracker._save()

        good_report.health_updated = False
        with patch("portfolio.loop_contract._log_violations"), \
             patch("portfolio.loop_contract._alert_violations"), \
             patch("portfolio.loop_contract._trigger_self_heal") as mock_heal:
            # This is the 3rd consecutive failure → escalate → self-heal
            verify_and_act(good_report, {}, tracker=ViolationTracker(state_file))
            mock_heal.assert_called_once()

    def test_custom_tracker_used(self, good_report, tracker_dir):
        state_file = tracker_dir / "contract_state.json"
        tracker = ViolationTracker(state_file)
        good_report.health_updated = False
        with patch("portfolio.loop_contract._log_violations"), \
             patch("portfolio.loop_contract._alert_violations"), \
             patch("portfolio.loop_contract._trigger_self_heal"):
            verify_and_act(good_report, {}, tracker=tracker)
        assert tracker._consecutive.get("health_updated") == 1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestViolationLogging:
    """Violations are logged to JSONL."""

    def test_violations_logged_to_jsonl(self, good_report, tracker_dir):
        good_report.health_updated = False
        good_report.heartbeat_updated = False
        with patch("portfolio.loop_contract._alert_violations"), \
             patch("portfolio.loop_contract._trigger_self_heal"):
            verify_and_act(good_report, {})

        log_file = tracker_dir / "contract_violations.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2  # health + heartbeat
        entry = json.loads(lines[0])
        assert "invariant" in entry
        assert "severity" in entry
        assert "ts" in entry
        assert entry["cycle_id"] == 42


# ---------------------------------------------------------------------------
# _run_post_cycle report tracking
# ---------------------------------------------------------------------------

class TestPostCycleReportTracking:
    """_run_post_cycle populates report.post_cycle_results."""

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.daily_digest.maybe_send_daily_digest")
    @mock.patch("portfolio.message_throttle.flush_and_send")
    def test_success_tracked(self, mock_flush, mock_daily, mock_digest):
        from portfolio.main import _run_post_cycle
        report = CycleReport(cycle_id=1)
        config = {"notification": {}}
        _run_post_cycle(config, report=report)
        # Tasks that ran should be tracked
        assert report.post_cycle_results.get("daily_digest") is True
        assert report.post_cycle_results.get("message_throttle") is True

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.daily_digest.maybe_send_daily_digest", side_effect=Exception("boom"))
    @mock.patch("portfolio.message_throttle.flush_and_send")
    def test_failure_tracked(self, mock_flush, mock_daily, mock_digest):
        from portfolio.main import _run_post_cycle
        report = CycleReport(cycle_id=1)
        config = {"notification": {}}
        _run_post_cycle(config, report=report)
        assert report.post_cycle_results.get("daily_digest") is False
        assert ("daily_digest", "boom") in report.errors

    @mock.patch("portfolio.main._maybe_send_digest")
    def test_no_report_still_works(self, mock_digest):
        """Backward compat: _run_post_cycle(config) without report doesn't crash."""
        from portfolio.main import _run_post_cycle
        config = {"notification": {}}
        _run_post_cycle(config)  # No report — should not raise


# ---------------------------------------------------------------------------
# MetalsCycleReport and verify_metals_contract
# ---------------------------------------------------------------------------

@pytest.fixture()
def good_metals_report():
    """A MetalsCycleReport where everything succeeded."""
    report = MetalsCycleReport(cycle_id=10)
    report.cycle_start = 1000.0
    report.cycle_end = 1040.0
    report.underlying_prices_fetched = True
    report.underlying_tickers_ok = {"XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"}
    report.position_prices_updated = True
    report.active_positions = 2
    report.positions_priced = 2
    report.holdings_reconciled = True
    report.session_alive = True
    report.stops_verified = True
    report.probability_computed = True
    return report


class TestMetalsCycleReport:
    """MetalsCycleReport construction and properties."""

    def test_defaults(self):
        r = MetalsCycleReport(cycle_id=1)
        assert r.underlying_prices_fetched is False
        assert r.session_alive is True  # default optimistic
        assert r.active_positions == 0
        assert r.errors == []

    def test_cycle_duration(self):
        r = MetalsCycleReport(cycle_id=1)
        r.cycle_start = 100.0
        r.cycle_end = 140.0
        assert r.cycle_duration_s == pytest.approx(40.0)


class TestVerifyMetalsContract:
    """Each metals invariant is tested for detection and correct severity."""

    def test_clean_report_passes(self, good_metals_report):
        violations = verify_metals_contract(good_metals_report)
        assert violations == []

    def test_missing_xag_detected(self, good_metals_report):
        good_metals_report.underlying_tickers_ok.discard("XAG-USD")
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "underlying_prices_fetched"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"
        assert "XAG-USD" in v[0].message

    def test_missing_xau_detected(self, good_metals_report):
        good_metals_report.underlying_tickers_ok.discard("XAU-USD")
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "underlying_prices_fetched"]
        assert len(v) == 1

    def test_crypto_missing_not_critical(self, good_metals_report):
        """BTC/ETH missing is NOT a violation — only XAG/XAU are required."""
        good_metals_report.underlying_tickers_ok = {"XAG-USD", "XAU-USD"}
        violations = verify_metals_contract(good_metals_report)
        assert not any(v.invariant == "underlying_prices_fetched" for v in violations)

    def test_position_prices_incomplete(self, good_metals_report):
        good_metals_report.active_positions = 3
        good_metals_report.positions_priced = 1
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "position_prices_updated"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    def test_no_positions_skips_price_check(self, good_metals_report):
        good_metals_report.active_positions = 0
        good_metals_report.positions_priced = 0
        violations = verify_metals_contract(good_metals_report)
        assert not any(v.invariant == "position_prices_updated" for v in violations)

    def test_session_dead_critical(self, good_metals_report):
        good_metals_report.session_alive = False
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "session_alive"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_slow_cycle_detected(self, good_metals_report):
        good_metals_report.cycle_end = good_metals_report.cycle_start + METALS_MAX_CYCLE_DURATION_S + 1
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "cycle_duration"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    def test_stops_missing_with_positions(self, good_metals_report):
        good_metals_report.stops_verified = False
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "stops_in_place"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"

    def test_stops_ok_without_positions(self, good_metals_report):
        good_metals_report.active_positions = 0
        good_metals_report.stops_verified = False
        violations = verify_metals_contract(good_metals_report)
        assert not any(v.invariant == "stops_in_place" for v in violations)

    def test_holdings_not_reconciled(self, good_metals_report):
        good_metals_report.holdings_reconciled = False
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "holdings_reconciled"]
        assert len(v) == 1

    def test_errors_recorded(self, good_metals_report):
        good_metals_report.errors.append(("price_fetch", "timeout"))
        violations = verify_metals_contract(good_metals_report)
        v = [v for v in violations if v.invariant == "no_critical_errors"]
        assert len(v) == 1


# ---------------------------------------------------------------------------
# BotCycleReport and verify_bot_contract
# ---------------------------------------------------------------------------

@pytest.fixture()
def good_bot_report():
    """A BotCycleReport where everything succeeded."""
    report = BotCycleReport(cycle_id=5, bot_name="golddigger")
    report.cycle_start = 1000.0
    report.cycle_end = 1020.0
    report.snapshot_collected = True
    report.bot_step_completed = True
    report.session_alive = True
    report.consecutive_errors = 0
    report.max_consecutive_errors = 5
    report.report_on_schedule = True
    return report


class TestBotCycleReport:
    """BotCycleReport construction and properties."""

    def test_defaults(self):
        r = BotCycleReport(cycle_id=1, bot_name="elongir")
        assert r.bot_name == "elongir"
        assert r.bot_step_completed is False
        assert r.session_alive is True
        assert r.consecutive_errors == 0

    def test_cycle_duration(self):
        r = BotCycleReport(cycle_id=1)
        r.cycle_start = 100.0
        r.cycle_end = 125.0
        assert r.cycle_duration_s == pytest.approx(25.0)


class TestVerifyBotContract:
    """Each bot invariant is tested for detection and correct severity."""

    def test_clean_report_passes(self, good_bot_report):
        violations = verify_bot_contract(good_bot_report)
        assert violations == []

    def test_bot_step_failed(self, good_bot_report):
        good_bot_report.bot_step_completed = False
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "bot_step_completed"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_snapshot_not_collected(self, good_bot_report):
        good_bot_report.snapshot_collected = False
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "snapshot_collected"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_session_dead(self, good_bot_report):
        good_bot_report.session_alive = False
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "session_alive"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_errors_approaching_halt_warning(self, good_bot_report):
        good_bot_report.consecutive_errors = BOT_ERROR_WARNING_THRESHOLD
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "consecutive_errors"]
        assert len(v) == 1
        assert v[0].severity == "WARNING"
        assert "until halt" in v[0].message

    def test_errors_one_from_halt_critical(self, good_bot_report):
        good_bot_report.consecutive_errors = good_bot_report.max_consecutive_errors - 1
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "consecutive_errors"]
        assert len(v) == 1
        assert v[0].severity == "CRITICAL"

    def test_no_errors_passes(self, good_bot_report):
        good_bot_report.consecutive_errors = 0
        violations = verify_bot_contract(good_bot_report)
        assert not any(v.invariant == "consecutive_errors" for v in violations)

    def test_slow_cycle(self, good_bot_report):
        good_bot_report.cycle_end = good_bot_report.cycle_start + BOT_MAX_CYCLE_DURATION_S + 1
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "cycle_duration"]
        assert len(v) == 1

    def test_report_missed(self, good_bot_report):
        good_bot_report.report_on_schedule = False
        violations = verify_bot_contract(good_bot_report)
        v = [v for v in violations if v.invariant == "report_on_schedule"]
        assert len(v) == 1

    def test_bot_name_in_messages(self, good_bot_report):
        good_bot_report.bot_step_completed = False
        violations = verify_bot_contract(good_bot_report)
        assert "golddigger" in violations[0].message


# ---------------------------------------------------------------------------
# verify_and_act with custom verify_fn
# ---------------------------------------------------------------------------

class TestVerifyAndActGeneric:
    """verify_and_act works with custom verify functions and loop names."""

    def test_metals_verify_fn(self, good_metals_report, tracker_dir):
        good_metals_report.session_alive = False  # trigger violation
        with patch("portfolio.loop_contract._log_violations") as mock_log, \
             patch("portfolio.loop_contract._alert_violations") as mock_alert, \
             patch("portfolio.loop_contract._trigger_self_heal") as mock_heal:
            verify_and_act(
                good_metals_report, {},
                verify_fn=verify_metals_contract,
                loop_name="metals",
            )
            mock_log.assert_called_once()
            mock_alert.assert_called_once()
            # Check loop_name passed through
            _, kwargs = mock_alert.call_args
            assert kwargs.get("loop_name") == "metals"

    def test_bot_verify_fn(self, good_bot_report, tracker_dir):
        good_bot_report.bot_step_completed = False  # trigger violation
        with patch("portfolio.loop_contract._log_violations"), \
             patch("portfolio.loop_contract._alert_violations") as mock_alert, \
             patch("portfolio.loop_contract._trigger_self_heal"):
            verify_and_act(
                good_bot_report, {},
                verify_fn=verify_bot_contract,
                loop_name="golddigger",
            )
            _, kwargs = mock_alert.call_args
            assert kwargs.get("loop_name") == "golddigger"

    def test_heal_prompt_includes_loop_name(self):
        violations = [
            Violation(invariant="test", severity="CRITICAL", message="broken"),
        ]
        prompt = _build_heal_prompt(violations, loop_name="metals")
        assert "metals" in prompt
        assert "metals_loop.py" in prompt

    def test_heal_prompt_golddigger(self):
        prompt = _build_heal_prompt(
            [Violation(invariant="x", severity="CRITICAL", message="y")],
            loop_name="golddigger",
        )
        assert "golddigger" in prompt
        assert "golddigger/runner.py" in prompt


# ---------------------------------------------------------------------------
# file_utils I/O integration — verify that top-level imports from
# file_utils work correctly and replace the old local helpers.
# ---------------------------------------------------------------------------

class TestFileUtilsIntegration:
    """Verify loop_contract uses file_utils for all I/O."""

    def test_module_imports_from_file_utils(self):
        """Top-level imports of load_json and last_jsonl_entry must work."""
        import portfolio.loop_contract as lc
        from portfolio.file_utils import last_jsonl_entry, load_json
        # Verify the module references the same functions (not local copies)
        assert lc.load_json is load_json
        assert lc.last_jsonl_entry is last_jsonl_entry

    def test_load_json_returns_none_for_missing_file(self, tmp_path):
        """load_json returns None for missing file (same as old _read_json)."""
        from portfolio.file_utils import load_json
        result = load_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_last_jsonl_entry_returns_none_for_missing_file(self, tmp_path):
        """last_jsonl_entry returns None for missing file."""
        from portfolio.file_utils import last_jsonl_entry
        result = last_jsonl_entry(tmp_path / "nonexistent.jsonl")
        assert result is None

    def test_last_jsonl_entry_returns_last_line(self, tmp_path):
        """last_jsonl_entry returns the last valid JSON line."""
        from portfolio.file_utils import last_jsonl_entry
        path = tmp_path / "test.jsonl"
        path.write_text(
            '{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8"
        )
        result = last_jsonl_entry(path)
        assert result == {"a": 3}

    def test_no_json_module_used_directly(self):
        """loop_contract should not use json.load/json.loads directly for file I/O."""
        import inspect
        import portfolio.loop_contract as lc
        source = inspect.getsource(lc.check_layer2_journal_activity)
        assert "json.load(" not in source
        assert "json.loads(" not in source
