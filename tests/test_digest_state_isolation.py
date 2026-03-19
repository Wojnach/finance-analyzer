"""Tests for digest state isolation (BUG-40, BUG-41, TEST-10).

Verifies that digest.py and daily_digest.py use their own state files
instead of sharing trigger_state.json with trigger.py.
"""


import pytest

from portfolio.file_utils import atomic_write_json, load_json


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR for digest and daily_digest modules."""
    monkeypatch.setattr("portfolio.digest.DATA_DIR", tmp_path)
    monkeypatch.setattr("portfolio.digest._DIGEST_STATE_FILE", tmp_path / "digest_state.json")
    monkeypatch.setattr("portfolio.daily_digest.DATA_DIR", tmp_path)
    monkeypatch.setattr("portfolio.daily_digest._DAILY_DIGEST_STATE_FILE", tmp_path / "daily_digest_state.json")
    return tmp_path


class TestDigestStateIsolation:
    """digest.py must use digest_state.json, not trigger_state.json."""

    def test_set_last_digest_time_writes_own_file(self, data_dir):
        from portfolio.digest import _set_last_digest_time
        _set_last_digest_time(12345.0)

        state = load_json(data_dir / "digest_state.json", default={})
        assert state["last_digest_time"] == 12345.0

    def test_set_last_digest_time_does_not_modify_trigger_state(self, data_dir):
        # Pre-populate trigger_state.json
        trigger_file = data_dir / "trigger_state.json"
        atomic_write_json(trigger_file, {"some_key": "untouched"})

        from portfolio.digest import _set_last_digest_time
        _set_last_digest_time(99999.0)

        # trigger_state.json must be untouched
        state = load_json(trigger_file, default={})
        assert state == {"some_key": "untouched"}

    def test_get_last_digest_time_reads_own_file(self, data_dir):
        atomic_write_json(data_dir / "digest_state.json", {"last_digest_time": 42.0})

        from portfolio.digest import _get_last_digest_time
        assert _get_last_digest_time() == 42.0

    def test_get_last_digest_time_migrates_from_trigger_state(self, data_dir):
        """When digest_state.json doesn't exist, read from trigger_state.json."""
        atomic_write_json(
            data_dir / "trigger_state.json",
            {"last_digest_time": 77.0, "other": True},
        )

        from portfolio.digest import _get_last_digest_time
        assert _get_last_digest_time() == 77.0

    def test_get_last_digest_time_returns_zero_when_no_files(self, data_dir):
        from portfolio.digest import _get_last_digest_time
        assert _get_last_digest_time() == 0


class TestDailyDigestStateIsolation:
    """daily_digest.py must use daily_digest_state.json, not trigger_state.json."""

    def test_set_last_daily_digest_time_writes_own_file(self, data_dir):
        from portfolio.daily_digest import _set_last_daily_digest_time
        _set_last_daily_digest_time(54321.0)

        state = load_json(data_dir / "daily_digest_state.json", default={})
        assert state["last_daily_digest_time"] == 54321.0

    def test_set_last_daily_digest_time_does_not_modify_trigger_state(self, data_dir):
        trigger_file = data_dir / "trigger_state.json"
        atomic_write_json(trigger_file, {"preserved": True})

        from portfolio.daily_digest import _set_last_daily_digest_time
        _set_last_daily_digest_time(11111.0)

        state = load_json(trigger_file, default={})
        assert state == {"preserved": True}

    def test_get_last_daily_digest_time_reads_own_file(self, data_dir):
        atomic_write_json(
            data_dir / "daily_digest_state.json",
            {"last_daily_digest_time": 88.0},
        )

        from portfolio.daily_digest import _get_last_daily_digest_time
        assert _get_last_daily_digest_time() == 88.0

    def test_get_last_daily_digest_time_migrates_from_trigger_state(self, data_dir):
        atomic_write_json(
            data_dir / "trigger_state.json",
            {"last_daily_digest_time": 55.0},
        )

        from portfolio.daily_digest import _get_last_daily_digest_time
        assert _get_last_daily_digest_time() == 55.0

    def test_get_last_daily_digest_time_returns_zero_when_no_files(self, data_dir):
        from portfolio.daily_digest import _get_last_daily_digest_time
        assert _get_last_daily_digest_time() == 0


class TestSignalEngineSentimentNoTriggerState:
    """REF-7: signal_engine no longer reads trigger_state.json for sentiment."""

    def test_load_prev_sentiments_without_trigger_state(self, tmp_path, monkeypatch):
        """_load_prev_sentiments works even without trigger_state.json."""
        import portfolio.signal_engine as se
        monkeypatch.setattr(se, "DATA_DIR", tmp_path)
        monkeypatch.setattr(se, "_SENTIMENT_STATE_FILE", tmp_path / "sentiment_state.json")
        monkeypatch.setattr(se, "_prev_sentiment_loaded", False)
        monkeypatch.setattr(se, "_prev_sentiment", {})

        # No sentiment_state.json, no trigger_state.json
        se._load_prev_sentiments()
        assert se._prev_sentiment == {}

    def test_load_prev_sentiments_reads_own_file(self, tmp_path, monkeypatch):
        """_load_prev_sentiments reads from sentiment_state.json."""
        import portfolio.signal_engine as se
        monkeypatch.setattr(se, "DATA_DIR", tmp_path)
        sf = tmp_path / "sentiment_state.json"
        monkeypatch.setattr(se, "_SENTIMENT_STATE_FILE", sf)
        monkeypatch.setattr(se, "_prev_sentiment_loaded", False)
        monkeypatch.setattr(se, "_prev_sentiment", {})

        atomic_write_json(sf, {"prev_sentiment": {"BTC-USD": "bullish"}})
        se._load_prev_sentiments()
        assert se._prev_sentiment.get("BTC-USD") == "bullish"
