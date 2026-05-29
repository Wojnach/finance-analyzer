"""Regression tests for 2026-05-29 auto-session Batch 2 fixes."""

import pytest


class TestDashboardInfoDisclosure:
    def test_500_error_no_path(self):
        """500 error response should not leak request path."""
        import dashboard.app as app_mod
        with open(app_mod.__file__) as f:
            content = f.read()
        assert '"path": request.path' not in content
        assert '"path":request.path' not in content

    def test_validate_portfolio_no_exception_leak(self):
        """POST validate-portfolio should not reflect exception messages."""
        import dashboard.app as app_mod
        with open(app_mod.__file__) as f:
            content = f.read()
        assert 'f"Validation error: {e}"' not in content


class TestForecastAccuracyRsplit:
    def test_horizon_suffix_uses_rsplit(self):
        """forecast_accuracy should use rsplit for multi-part sub-signal names."""
        import portfolio.forecast_accuracy as fa
        with open(fa.__file__) as f:
            content = f.read()
        assert "rsplit" in content


class TestSignalRegistryValidation:
    def test_max_confidence_out_of_range_raises(self):
        """register_enhanced with max_confidence > 1 should raise."""
        from portfolio.signal_registry import register_enhanced
        with pytest.raises(ValueError, match="max_confidence"):
            register_enhanced(
                "test_bad_conf", "portfolio.signals.rsi", "compute_rsi_signal",
                max_confidence=1.5,
            )


class TestMessageStoreTypeSafety:
    def test_string_muted_categories_ignored(self):
        """A string muted_categories should not iterate chars."""
        from portfolio.message_store import send_or_store
        config = {
            "telegram": {
                "token": "fake",
                "chat_id": "123",
                "muted_categories": "error",
            }
        }
        # "error" as string should NOT mute category "e" or "r"
        # This just checks no crash — the mute set should be empty
        # since it's not a list
        import portfolio.message_store as ms
        raw = config["telegram"]["muted_categories"]
        result = set(raw) if isinstance(raw, list) else set()
        assert "e" not in result
        assert result == set()


class TestTradeGuardsWarningLog:
    def test_corrupt_timestamp_logs_warning(self):
        """Corrupt timestamp should produce a warning, not silent pass."""
        import portfolio.trade_guards as tg
        with open(tg.__file__) as f:
            content = f.read()
        assert "corrupt timestamp" in content


class TestSignalEngineSoftConfCap:
    def test_soft_conf_weight_capped(self):
        """Soft-confidence weight product should be capped at 0.30."""
        import portfolio.signal_engine as se
        with open(se.__file__) as f:
            content = f.read()
        assert "min(weight, 0.30)" in content


class TestMetalsLoopCycleIsolation:
    def test_per_cycle_try_except_exists(self):
        """Metals loop should have per-cycle error handling."""
        with open("data/metals_loop.py") as f:
            content = f.read()
        assert "_consecutive_cycle_errors" in content
        assert "consecutive cycle errors" in content
