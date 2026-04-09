"""Tests for portfolio/meta_learner.py.

Covers:
- BUG-145: SQLite connection cleanup on exception (try/finally)
- BUG-147: SIGNAL_NAMES imported from tickers (not duplicated)
- BUG-148: Model cache in predict() with mtime-based staleness
"""

import sqlite3
from unittest import mock

import pytest

# ===========================================================================
# BUG-145: SQLite connection cleanup on exception
# ===========================================================================

class TestSQLiteCleanup:
    """BUG-145: _load_data() must close the SQLite connection even on error."""

    def test_connection_closed_on_query_error(self, tmp_path):
        """If pd.read_sql_query raises, the connection must still be closed."""
        from portfolio import meta_learner

        # Create a minimal SQLite DB (will fail on query since tables don't exist)
        db_path = tmp_path / "signal_log.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.close()

        with mock.patch.object(meta_learner, "SIGNAL_DB", db_path), pytest.raises(Exception):
            meta_learner._load_data("1d")

        # The key test: after the exception, no connections should be leaked.
        # We verify by successfully connecting and performing an operation
        # (WAL mode would block if a connection was left open in some configs).
        verify_conn = sqlite3.connect(str(db_path))
        verify_conn.execute("SELECT * FROM dummy")
        verify_conn.close()

    def test_connection_closed_on_success(self, tmp_path):
        """Normal path: connection is closed after successful query."""
        from portfolio import meta_learner

        db_path = tmp_path / "signal_log.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE snapshots (id INTEGER PRIMARY KEY, ts TEXT)
        """)
        conn.execute("""
            CREATE TABLE ticker_signals (
                id INTEGER PRIMARY KEY, snapshot_id INTEGER,
                ticker TEXT, signals TEXT, regime TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE outcomes (
                id INTEGER PRIMARY KEY, snapshot_id INTEGER,
                ticker TEXT, horizon TEXT, change_pct REAL
            )
        """)
        conn.commit()
        conn.close()

        with mock.patch.object(meta_learner, "SIGNAL_DB", db_path):
            df = meta_learner._load_data("1d")

        assert len(df) == 0  # No data, but query succeeded without leak


# ===========================================================================
# BUG-147: SIGNAL_NAMES imported from tickers
# ===========================================================================

class TestSignalNamesImport:
    """BUG-147: meta_learner.SIGNAL_NAMES should be the same object as tickers.SIGNAL_NAMES."""

    def test_signal_names_is_canonical(self):
        """SIGNAL_NAMES in meta_learner must come from tickers, not be a copy."""
        from portfolio.meta_learner import SIGNAL_NAMES as ml_names
        from portfolio.tickers import SIGNAL_NAMES as tickers_names

        # Same object reference (imported, not copied)
        assert ml_names is tickers_names

    def test_signal_names_count(self):
        """Signal names should have 36 signals (added onchain, credit_spread_risk)."""
        from portfolio.meta_learner import SIGNAL_NAMES
        assert len(SIGNAL_NAMES) == 36


# ===========================================================================
# BUG-148: Model cache in predict()
# ===========================================================================

class TestModelCache:
    """BUG-148: predict() should cache models and reload only when file changes."""

    def test_cache_populated_on_first_call(self, tmp_path):
        """First predict() call should populate _model_cache."""
        from portfolio import meta_learner

        # Create a fake model file
        model_path = tmp_path / "meta_learner_1d.joblib"
        fake_model = mock.MagicMock()
        fake_model.predict_proba.return_value = [[0.4, 0.6]]

        # Clear cache
        meta_learner._model_cache.clear()

        with (
            mock.patch.object(meta_learner, "MODEL_DIR", tmp_path),
            mock.patch("portfolio.meta_learner.joblib") as mock_joblib,
        ):
            # Create the file so model_path.exists() returns True
            model_path.write_bytes(b"fake")

            mock_joblib.load.return_value = fake_model
            votes = {"rsi": "BUY", "macd": "BUY"}
            direction, prob = meta_learner.predict(votes, "BTC-USD", horizon="1d")

            # Model was loaded from disk
            mock_joblib.load.assert_called_once()

            # Cache should now have an entry for "1d"
            assert "1d" in meta_learner._model_cache
            assert meta_learner._model_cache["1d"][0] is fake_model

    def test_cache_reused_on_second_call(self, tmp_path):
        """Second predict() call should use cache, not reload from disk."""
        from portfolio import meta_learner

        model_path = tmp_path / "meta_learner_1d.joblib"
        fake_model = mock.MagicMock()
        fake_model.predict_proba.return_value = [[0.4, 0.6]]

        meta_learner._model_cache.clear()

        with (
            mock.patch.object(meta_learner, "MODEL_DIR", tmp_path),
            mock.patch("portfolio.meta_learner.joblib") as mock_joblib,
        ):
            model_path.write_bytes(b"fake")
            mock_joblib.load.return_value = fake_model

            votes = {"rsi": "BUY"}

            # First call — loads from disk
            meta_learner.predict(votes, "BTC-USD", horizon="1d")
            assert mock_joblib.load.call_count == 1

            # Second call — should use cache (same mtime)
            meta_learner.predict(votes, "BTC-USD", horizon="1d")
            assert mock_joblib.load.call_count == 1  # Still 1, not 2

    def test_cache_invalidated_on_mtime_change(self, tmp_path):
        """If model file is rewritten (retrained), cache should reload."""
        import time

        from portfolio import meta_learner

        model_path = tmp_path / "meta_learner_1d.joblib"
        fake_model = mock.MagicMock()
        fake_model.predict_proba.return_value = [[0.4, 0.6]]

        meta_learner._model_cache.clear()

        with (
            mock.patch.object(meta_learner, "MODEL_DIR", tmp_path),
            mock.patch("portfolio.meta_learner.joblib") as mock_joblib,
        ):
            model_path.write_bytes(b"fake-v1")
            mock_joblib.load.return_value = fake_model
            votes = {"rsi": "BUY"}

            # First call — loads
            meta_learner.predict(votes, "BTC-USD", horizon="1d")
            assert mock_joblib.load.call_count == 1

            # Simulate retraining: change mtime
            time.sleep(0.05)
            model_path.write_bytes(b"fake-v2")

            # Third call — mtime changed, should reload
            meta_learner.predict(votes, "BTC-USD", horizon="1d")
            assert mock_joblib.load.call_count == 2

    def test_missing_model_returns_hold(self, tmp_path):
        """If model file doesn't exist, predict() returns HOLD without error."""
        from portfolio import meta_learner

        meta_learner._model_cache.clear()

        with mock.patch.object(meta_learner, "MODEL_DIR", tmp_path):
            direction, prob = meta_learner.predict({"rsi": "BUY"}, "BTC-USD", horizon="1d")

        assert direction == "HOLD"
        assert prob == 0.0

    def test_different_horizons_cached_independently(self, tmp_path):
        """Each horizon gets its own cache entry."""
        from portfolio import meta_learner

        fake_model_1d = mock.MagicMock()
        fake_model_1d.predict_proba.return_value = [[0.3, 0.7]]
        fake_model_3h = mock.MagicMock()
        fake_model_3h.predict_proba.return_value = [[0.6, 0.4]]

        meta_learner._model_cache.clear()

        with (
            mock.patch.object(meta_learner, "MODEL_DIR", tmp_path),
            mock.patch("portfolio.meta_learner.joblib") as mock_joblib,
        ):
            (tmp_path / "meta_learner_1d.joblib").write_bytes(b"fake")
            (tmp_path / "meta_learner_3h.joblib").write_bytes(b"fake")
            mock_joblib.load.side_effect = [fake_model_1d, fake_model_3h]

            votes = {"rsi": "BUY"}
            d1, _ = meta_learner.predict(votes, "BTC-USD", horizon="1d")
            d3, _ = meta_learner.predict(votes, "BTC-USD", horizon="3h")

            assert "1d" in meta_learner._model_cache
            assert "3h" in meta_learner._model_cache
            assert mock_joblib.load.call_count == 2
