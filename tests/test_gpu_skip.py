"""Tests for GPU signal skipping outside US market hours."""

from datetime import UTC, datetime

from portfolio.market_timing import is_us_stock_market_open, should_skip_gpu


class TestIsUsStockMarketOpen:
    """Tests for is_us_stock_market_open()."""

    def test_weekday_during_hours_edt(self):
        # Wednesday 2026-06-10 15:00 UTC = 11:00 ET (EDT)
        now = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is True

    def test_weekday_during_hours_est(self):
        # Wednesday 2026-01-14 16:00 UTC = 11:00 ET (EST)
        now = datetime(2026, 1, 14, 16, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is True

    def test_weekday_before_open_edt(self):
        # Wednesday 2026-06-10 12:00 UTC = 08:00 ET (before 09:30 open)
        now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_weekday_after_close_edt(self):
        # Wednesday 2026-06-10 21:00 UTC = 17:00 ET (after 16:00 close)
        now = datetime(2026, 6, 10, 21, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_weekday_after_close_est(self):
        # Wednesday 2026-01-14 22:00 UTC = 17:00 ET (after 16:00 close)
        now = datetime(2026, 1, 14, 22, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_weekend(self):
        # Saturday 2026-06-13 15:00 UTC
        now = datetime(2026, 6, 13, 15, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_at_open_edt(self):
        # NYSE opens at 13:30 UTC during EDT
        now = datetime(2026, 6, 10, 13, 30, tzinfo=UTC)
        assert is_us_stock_market_open(now) is True

    def test_just_before_open_edt(self):
        now = datetime(2026, 6, 10, 13, 29, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_at_close_edt(self):
        # NYSE closes at 20:00 UTC during EDT — close minute is NOT open
        now = datetime(2026, 6, 10, 20, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now) is False

    def test_just_before_close_edt(self):
        now = datetime(2026, 6, 10, 19, 59, tzinfo=UTC)
        assert is_us_stock_market_open(now) is True

    def test_pre_market_buffer(self):
        # 13:00 UTC = 30 min before 13:30 open (EDT)
        now = datetime(2026, 6, 10, 13, 0, tzinfo=UTC)
        assert is_us_stock_market_open(now, pre_market_buffer_min=30) is True
        assert is_us_stock_market_open(now, pre_market_buffer_min=0) is False

    def test_post_market_buffer(self):
        # 20:10 UTC = 10 min after 20:00 close (EDT)
        now = datetime(2026, 6, 10, 20, 10, tzinfo=UTC)
        assert is_us_stock_market_open(now, post_market_buffer_min=15) is True
        assert is_us_stock_market_open(now, post_market_buffer_min=0) is False


class TestShouldSkipGpu:
    """Tests for should_skip_gpu()."""

    def _off_hours(self):
        """Return a UTC datetime outside US market hours."""
        return datetime(2026, 6, 10, 23, 0, tzinfo=UTC)

    def _market_hours(self):
        """Return a UTC datetime during US market hours."""
        return datetime(2026, 6, 10, 15, 0, tzinfo=UTC)

    def test_crypto_never_skipped(self):
        assert should_skip_gpu("BTC-USD", now=self._off_hours()) is False
        assert should_skip_gpu("ETH-USD", now=self._off_hours()) is False

    def test_metals_never_skipped(self):
        assert should_skip_gpu("XAU-USD", now=self._off_hours()) is False
        assert should_skip_gpu("XAG-USD", now=self._off_hours()) is False

    def test_stock_skipped_off_hours(self):
        # MSTR is the only stock in STOCK_SYMBOLS after the Apr-09 reduction.
        assert should_skip_gpu("MSTR", now=self._off_hours()) is True

    def test_stock_not_skipped_during_hours(self):
        assert should_skip_gpu("MSTR", now=self._market_hours()) is False

    def test_config_disables_feature(self):
        cfg = {"gpu_signals": {"skip_stocks_offhours": False}}
        assert should_skip_gpu("MSTR", config=cfg, now=self._off_hours()) is False

    def test_config_buffers(self):
        # 13:10 UTC = 20 min before NYSE open (EDT 13:30)
        pre_open = datetime(2026, 6, 10, 13, 10, tzinfo=UTC)

        # Default buffer (30 min) — should NOT skip (within buffer)
        assert should_skip_gpu("MSTR", now=pre_open) is False

        # No buffer — should skip (market not open yet)
        cfg = {"gpu_signals": {"pre_market_buffer_min": 0}}
        assert should_skip_gpu("MSTR", config=cfg, now=pre_open) is True

    def test_weekend_stocks_skipped(self):
        weekend = datetime(2026, 6, 13, 15, 0, tzinfo=UTC)  # Saturday
        assert should_skip_gpu("MSTR", now=weekend) is True
        assert should_skip_gpu("BTC-USD", now=weekend) is False
