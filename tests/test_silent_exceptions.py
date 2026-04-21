"""Tests for BUG-61 through BUG-70: verify previously-silent exception handlers now log.

Each test triggers a code path that previously had `except Exception: pass` and
verifies that the fix now produces a log message instead of silently swallowing errors.
"""

import logging
from unittest.mock import MagicMock, patch

# --- BUG-61: autonomous.py compact summary load ---

def test_autonomous_logs_compact_summary_load_failure(tmp_path, monkeypatch, caplog):
    """autonomous.py — loading compact summary for probability mode should log on failure.

    load_json() catches JSON decode errors internally (returning None) and logs
    a WARNING to portfolio.file_utils. Verify that warning is emitted.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Write corrupt JSON so load_json logs a warning and returns None
    corrupt_file = data_dir / "agent_summary_compact.json"
    corrupt_file.write_text("{bad json", encoding="utf-8")

    import portfolio.autonomous as amod
    monkeypatch.setattr(amod, "DATA_DIR", data_dir)

    # Capture at WARNING level across all loggers (file_utils logs the decode error)
    with caplog.at_level(logging.WARNING):
        # Call the function directly — it has many params, pass minimal valid ones
        result = amod._build_telegram_mode_b(
            actionable={}, hold_count=0, sell_count=0,
            patient_state={"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}},
            bold_state={"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}},
            prices_usd={}, fx_rate=10.5, signals={}, tf_data={},
            predictions={}, config={"notification": {"mode": "probability", "focus_tickers": ["XAG-USD"]}},
            tier=3, regime="range-bound", reflection="", reasons=["test"],
        )

    assert any("compact" in r.message.lower() or "corrupt" in r.message.lower()
               for r in caplog.records), f"Expected log about compact summary failure, got: {[r.message for r in caplog.records]}"


# --- BUG-62: fx_rates.py Telegram alert ---

def test_fx_alert_telegram_logs_on_failure(monkeypatch, caplog):
    """fx_rates.py:63 — FX Telegram alert should log on failure, not silently pass."""
    import portfolio.fx_rates as fx_mod

    # Reset alert cooldown so it fires
    fx_mod._fx_cache["_last_fx_alert"] = 0

    # Make _load_config raise
    monkeypatch.setattr(fx_mod, "_load_config", lambda: (_ for _ in ()).throw(RuntimeError("config broken")))

    with caplog.at_level(logging.DEBUG, logger="portfolio.fx_rates"):
        fx_mod._fx_alert_telegram(7200.0)

    assert any("fx" in r.message.lower() or "alert" in r.message.lower()
               for r in caplog.records), f"Expected FX alert log, got: {[r.message for r in caplog.records]}"


# --- BUG-63: outcome_tracker.py SQLite write ---

def test_outcome_tracker_logs_sqlite_snapshot_failure(tmp_path, monkeypatch, caplog):
    """outcome_tracker.py:150 — SQLite snapshot write should log on failure."""
    import portfolio.outcome_tracker as ot

    monkeypatch.setattr(ot, "SIGNAL_LOG", tmp_path / "signal_log.jsonl")
    monkeypatch.setattr(ot, "DATA_DIR", tmp_path)

    # Make the SignalDB import inside log_signal_snapshot raise
    mock_db = MagicMock()
    mock_db.insert_snapshot.side_effect = RuntimeError("SQLite broken")
    mock_signal_db_module = MagicMock()
    mock_signal_db_module.SignalDB.return_value = mock_db

    with patch.dict("sys.modules", {"portfolio.signal_db": mock_signal_db_module}):
        with caplog.at_level(logging.DEBUG, logger="portfolio.outcome_tracker"):
            ot.log_signal_snapshot(
                {"BTC-USD": {"indicators": {"close": 67000}, "extra": {}, "action": "HOLD"}},
                {"BTC-USD": 67000},
                10.5,
                ["test"],
            )

    assert any("sqlite" in r.message.lower() or "snapshot" in r.message.lower()
               for r in caplog.records), f"Expected SQLite log, got: {[r.message for r in caplog.records]}"


# --- BUG-64: journal.py warrant state load ---

def test_journal_logs_warrant_state_failure(tmp_path, monkeypatch, caplog):
    """journal.py:413 — warrant state load should log on failure, not silently pass."""
    import portfolio.journal as jmod

    entries = [{
        "ts": "2026-03-01T12:00:00+00:00",
        "trigger": "test",
        "regime": "range-bound",
        "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}},
        "tickers": {},
        "prices": {},
    }]

    monkeypatch.setattr(jmod, "PORTFOLIO_FILE", tmp_path / "nonexistent_patient.json")
    monkeypatch.setattr(jmod, "BOLD_FILE", tmp_path / "nonexistent_bold.json")

    # Mock the warrant_portfolio module to raise on import
    mock_wp = MagicMock()
    mock_wp.load_warrant_state.side_effect = RuntimeError("warrant file corrupt")

    with patch.dict("sys.modules", {"portfolio.warrant_portfolio": mock_wp}):
        with caplog.at_level(logging.DEBUG, logger="portfolio.journal"):
            result = jmod.build_context(entries, portfolio_data=None)

    assert any("warrant" in r.message.lower()
               for r in caplog.records), f"Expected warrant log, got: {[r.message for r in caplog.records]}"


# --- BUG-64: journal.py smart retrieval fallthrough ---

def test_journal_logs_smart_retrieval_failure(tmp_path, monkeypatch, caplog):
    """journal.py:567 — smart retrieval failure should log before falling through."""
    import portfolio.journal as jmod

    journal_file = tmp_path / "layer2_journal.jsonl"
    journal_file.write_text("", encoding="utf-8")
    context_file = tmp_path / "layer2_context.md"

    monkeypatch.setattr(jmod, "JOURNAL_FILE", journal_file)
    monkeypatch.setattr(jmod, "CONTEXT_FILE", context_file)
    monkeypatch.setattr(jmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jmod, "_load_config", lambda: {"journal": {"smart_retrieval": True}})
    monkeypatch.setattr(jmod, "_get_current_market_state", lambda: {
        "signals": {}, "held_tickers": [], "regime": "range-bound", "prices": {}
    })

    # Make journal_index.retrieve_relevant_entries raise
    mock_ji = MagicMock()
    mock_ji.retrieve_relevant_entries.side_effect = RuntimeError("retrieval broken")

    with patch.dict("sys.modules", {"portfolio.journal_index": mock_ji}):
        with caplog.at_level(logging.DEBUG, logger="portfolio.journal"):
            jmod.write_context()

    assert any("smart retrieval" in r.message.lower() or "falling back" in r.message.lower()
               or "retrieval" in r.message.lower()
               for r in caplog.records), f"Expected smart retrieval log, got: {[r.message for r in caplog.records]}"


# --- BUG-67: forecast.py Kronos init (restored 2026-04-21 afternoon) ---

def test_forecast_logs_kronos_init_failure(monkeypatch, caplog):
    """forecast.py — Kronos init should log on config load failure instead of
    silently swallowing. Restored 2026-04-21 afternoon after un-retire;
    `_init_kronos_enabled()` now reads config.json again (shadow isolation
    moved into the vote-pool filter in `_health_weighted_vote`, not into the
    init function).
    """
    import portfolio.signals.forecast as fmod

    orig_enabled = fmod._KRONOS_ENABLED
    orig_shadow = fmod._KRONOS_SHADOW

    # Force the load_json inside _init_kronos_enabled to raise.
    with patch(
        "portfolio.file_utils.load_json",
        side_effect=RuntimeError("config corrupt"),
    ):
        with caplog.at_level(logging.DEBUG, logger="portfolio.signals.forecast"):
            fmod._init_kronos_enabled()

    # Restore state so subsequent tests see whatever the real config produced.
    fmod._KRONOS_ENABLED = orig_enabled
    fmod._KRONOS_SHADOW = orig_shadow

    assert any(
        "kronos" in r.message.lower() or "config" in r.message.lower()
        for r in caplog.records
    ), f"Expected Kronos init log, got: {[r.message for r in caplog.records]}"


# --- BUG-67: forecast.py health logging ---

def test_forecast_logs_health_failure(caplog):
    """forecast.py — health logging failure should be logged, not silently pass."""
    import portfolio.signals.forecast as fmod

    # atomic_append_jsonl auto-creates directories so setting a bad path alone won't
    # raise. Instead, mock the function to raise and verify the log fires.
    with caplog.at_level(logging.DEBUG, logger="portfolio.signals.forecast"), \
         patch("portfolio.signals.forecast.atomic_append_jsonl",
               side_effect=OSError("disk full")):
        fmod._log_health("chronos", "BTC-USD", True, 100.0)

    assert any("health" in r.message.lower() or "logging" in r.message.lower()
               or "failed" in r.message.lower()
               for r in caplog.records), f"Expected health log, got: {[r.message for r in caplog.records]}"


# --- BUG-69: main.py DATA_DIR consistency ---

def test_main_post_cycle_uses_data_dir_constant():
    """main.py:161 — _run_post_cycle should use DATA_DIR constant, not re-derive."""
    import inspect

    import portfolio.main as mmod

    source = inspect.getsource(mmod._run_post_cycle)
    assert "Path(__file__)" not in source, \
        "_run_post_cycle still re-derives DATA_DIR instead of using module constant"
    assert "DATA_DIR" in source, \
        "_run_post_cycle should use the DATA_DIR constant"


# --- BUG-70: main.py in-function stdlib import ---

def test_main_run_no_redundant_time_import():
    """main.py — run() should not have `import time` since it's module-level."""
    import inspect

    import portfolio.main as mmod

    source = inspect.getsource(mmod.run)
    assert "import time" not in source, \
        "run() still has redundant `import time` — time is already module-level"
