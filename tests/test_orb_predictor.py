"""Tests for the Opening Range Breakout (ORB) Predictor.

Covers MorningRange calculation, DayResult computation, prediction logic,
warrant translation, summary statistics, and edge cases. All data is mocked
with deterministic values -- no network calls.
"""

import pytest
from datetime import datetime, timezone, timedelta

from portfolio.orb_predictor import (
    ORBPredictor,
    MorningRange,
    DayResult,
    Prediction,
    WarrantTarget,
)


# ---------------------------------------------------------------------------
# Helpers for building mock kline data
# ---------------------------------------------------------------------------

def _candle(dt: datetime, o: float, h: float, l: float, c: float, v: float = 100.0) -> dict:
    """Build a single kline dict at the given UTC datetime."""
    return {
        "ts": dt,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "hour": dt.hour,
        "minute": dt.minute,
        "date": dt.strftime("%Y-%m-%d"),
    }


def _morning_candles(date_str: str, base_price: float = 30.0, spread: float = 0.5,
                     direction: str = "up", n: int = 8, volume: float = 100.0) -> list[dict]:
    """Generate n 15-min morning candles (08:00-09:45 UTC) for a given date.

    direction="up" means close > open across the window.
    direction="down" means close < open.
    """
    dt_base = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=8, minute=0, tzinfo=timezone.utc
    )
    candles = []
    step = spread / n
    for i in range(n):
        dt = dt_base + timedelta(minutes=15 * i)
        if direction == "up":
            o = base_price + step * i
            c = base_price + step * (i + 1)
        else:
            o = base_price + spread - step * i
            c = base_price + spread - step * (i + 1)
        h = max(o, c) + 0.02
        l = min(o, c) - 0.02
        candles.append(_candle(dt, o, h, l, c, volume))
    return candles


def _full_day_candles(date_str: str, morning_base: float = 30.0,
                      morning_spread: float = 0.5, direction: str = "up",
                      day_high_extra: float = 0.3, day_low_extra: float = 0.2,
                      morning_n: int = 8, afternoon_n: int = 48,
                      volume: float = 100.0) -> list[dict]:
    """Generate a full day of klines (morning + afternoon) for a given date.

    morning: 08:00-09:45 UTC  (morning_n candles)
    afternoon: 10:00-21:45 UTC  (afternoon_n candles, covering 12 hours)
    day_high_extra/low_extra: how far beyond morning range the day extends.
    """
    morning = _morning_candles(date_str, morning_base, morning_spread, direction,
                               morning_n, volume)

    # Derive morning high/low from the actual candles
    m_high = max(c["high"] for c in morning)
    m_low = min(c["low"] for c in morning)
    m_mid = (m_high + m_low) / 2

    dt_base = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=10, minute=0, tzinfo=timezone.utc
    )
    afternoon = []
    for i in range(afternoon_n):
        dt = dt_base + timedelta(minutes=15 * i)
        # Most candles stay inside morning range, with one breakout candle
        o = m_mid
        c = m_mid + 0.01
        h = m_mid + 0.05
        l = m_mid - 0.05
        # Place the day high on candle 10, day low on candle 20
        if i == 10:
            h = m_high + day_high_extra
            c = h - 0.02
            o = m_mid
        if i == 20:
            l = m_low - day_low_extra
            o = m_mid
            c = l + 0.02
        afternoon.append(_candle(dt, o, h, l, c, volume))

    return morning + afternoon


def _build_history(num_days: int = 20, base_price: float = 30.0) -> list[dict]:
    """Build num_days worth of full-day klines on consecutive weekdays."""
    all_klines = []
    dt = datetime(2026, 1, 5, tzinfo=timezone.utc)  # Monday
    days_created = 0
    while days_created < num_days:
        if dt.weekday() < 5:  # Skip weekends
            date_str = dt.strftime("%Y-%m-%d")
            direction = "up" if days_created % 3 != 0 else "down"
            day_high_extra = 0.20 + 0.05 * (days_created % 5)
            day_low_extra = 0.10 + 0.03 * (days_created % 4)
            candles = _full_day_candles(
                date_str,
                morning_base=base_price + 0.1 * days_created,
                morning_spread=0.40 + 0.02 * (days_created % 7),
                direction=direction,
                day_high_extra=day_high_extra,
                day_low_extra=day_low_extra,
            )
            all_klines.extend(candles)
            days_created += 1
        dt += timedelta(days=1)
    return all_klines


# ---------------------------------------------------------------------------
# TestMorningRange
# ---------------------------------------------------------------------------


class TestMorningRange:
    """Tests for ORBPredictor.calculate_morning_range."""

    def test_morning_range_basic(self):
        """Normal 8-candle morning returns correct MorningRange."""
        predictor = ORBPredictor()
        candles = _morning_candles("2026-02-10", base_price=30.0, spread=0.5, direction="up")
        mr = predictor.calculate_morning_range(candles)

        assert mr is not None
        assert mr.date == "2026-02-10"
        assert mr.num_candles == 8
        assert mr.high >= mr.low
        assert mr.range_abs == pytest.approx(mr.high - mr.low, abs=1e-9)
        assert mr.midpoint == pytest.approx((mr.high + mr.low) / 2, abs=1e-9)
        assert mr.range_pct == pytest.approx(mr.range_abs / mr.midpoint * 100, abs=1e-6)

    def test_morning_range_up_direction(self):
        """Morning where close > open yields direction='up'."""
        predictor = ORBPredictor()
        candles = _morning_candles("2026-02-10", direction="up")
        mr = predictor.calculate_morning_range(candles)

        assert mr is not None
        assert mr.direction == "up"
        assert mr.close > mr.open

    def test_morning_range_down_direction(self):
        """Morning where close < open yields direction='down'."""
        predictor = ORBPredictor()
        candles = _morning_candles("2026-02-10", direction="down")
        mr = predictor.calculate_morning_range(candles)

        assert mr is not None
        assert mr.direction == "down"
        assert mr.close < mr.open

    def test_morning_range_insufficient_candles(self):
        """Fewer than min_morning_candles (4) returns None."""
        predictor = ORBPredictor()
        candles = _morning_candles("2026-02-10", n=3)
        mr = predictor.calculate_morning_range(candles)

        assert mr is None

    def test_morning_range_exactly_min_candles(self):
        """Exactly min_morning_candles (4) returns a valid MorningRange."""
        predictor = ORBPredictor()
        candles = _morning_candles("2026-02-10", n=4, spread=0.5)
        mr = predictor.calculate_morning_range(candles)

        assert mr is not None
        assert mr.num_candles == 4

    def test_morning_range_tiny_range(self):
        """Range below min_morning_range_pct returns None."""
        predictor = ORBPredictor(min_morning_range_pct=0.5)
        # With spread=0.001 and base=100, the candle helper adds +/-0.02 padding
        # so range_pct ~ 0.04%, which is below our 0.5% threshold.
        candles = _morning_candles("2026-02-10", base_price=100.0, spread=0.001)
        mr = predictor.calculate_morning_range(candles)

        assert mr is None

    def test_morning_range_volume(self):
        """Volume is correctly summed across all morning candles."""
        predictor = ORBPredictor()
        vol = 250.0
        candles = _morning_candles("2026-02-10", volume=vol, n=8)
        mr = predictor.calculate_morning_range(candles)

        assert mr is not None
        assert mr.volume == pytest.approx(vol * 8, abs=1e-9)

    def test_morning_range_filters_by_hour(self):
        """Only candles within morning_start_utc..morning_end_utc are used."""
        predictor = ORBPredictor()
        morning = _morning_candles("2026-02-10", n=8)
        # Add an afternoon candle at 14:00 with extreme price
        afternoon = _candle(
            datetime(2026, 2, 10, 14, 0, tzinfo=timezone.utc),
            100.0, 200.0, 5.0, 150.0, 9999.0,
        )
        mr = predictor.calculate_morning_range(morning + [afternoon])

        assert mr is not None
        # The extreme afternoon candle should NOT affect the morning range
        assert mr.high < 200.0
        assert mr.low > 5.0
        assert mr.num_candles == 8  # Only 8 morning candles counted


# ---------------------------------------------------------------------------
# TestDayResult
# ---------------------------------------------------------------------------


class TestDayResult:
    """Tests for ORBPredictor.calculate_day_result."""

    def test_day_result_basic(self):
        """Full day with morning + afternoon returns correct DayResult."""
        predictor = ORBPredictor()
        candles = _full_day_candles("2026-02-10", day_high_extra=0.30, day_low_extra=0.15)
        dr = predictor.calculate_day_result(candles)

        assert dr is not None
        assert dr.date == "2026-02-10"
        assert dr.day_high >= dr.morning.high
        assert dr.day_low <= dr.morning.low
        assert dr.upside_ext_pct >= 0
        assert dr.downside_ext_pct >= 0

    def test_day_result_high_in_morning(self):
        """When day high == morning high, upside_ext_pct is approximately 0."""
        predictor = ORBPredictor()
        candles = _full_day_candles(
            "2026-02-10", day_high_extra=0.0, day_low_extra=0.15
        )
        dr = predictor.calculate_day_result(candles)

        assert dr is not None
        # Afternoon candles have h = m_mid + 0.05, which is below m_high,
        # so day_high == morning high.
        assert dr.upside_ext_pct == pytest.approx(0.0, abs=0.5)

    def test_day_result_low_in_morning(self):
        """When day low == morning low, downside_ext_pct is approximately 0."""
        predictor = ORBPredictor()
        candles = _full_day_candles(
            "2026-02-10", day_high_extra=0.30, day_low_extra=0.0
        )
        dr = predictor.calculate_day_result(candles)

        assert dr is not None
        assert dr.downside_ext_pct == pytest.approx(0.0, abs=0.5)

    def test_day_result_large_extension(self):
        """Large breakout beyond morning range produces correct ratios."""
        predictor = ORBPredictor()
        candles = _full_day_candles(
            "2026-02-10", morning_base=30.0, morning_spread=0.5,
            day_high_extra=2.0, day_low_extra=1.5,
        )
        dr = predictor.calculate_day_result(candles)

        assert dr is not None
        # upside_ext_pct = (day_high - morning_high) / morning_high * 100
        expected_upside_pct = (dr.day_high - dr.morning.high) / dr.morning.high * 100
        assert dr.upside_ext_pct == pytest.approx(expected_upside_pct, abs=1e-6)

        expected_downside_pct = (dr.morning.low - dr.day_low) / dr.morning.low * 100
        assert dr.downside_ext_pct == pytest.approx(expected_downside_pct, abs=1e-6)

        # Extension ratios
        assert dr.upside_ext_ratio == pytest.approx(
            (dr.day_high - dr.morning.high) / dr.morning.range_abs, abs=1e-4
        )
        assert dr.downside_ext_ratio == pytest.approx(
            (dr.morning.low - dr.day_low) / dr.morning.range_abs, abs=1e-4
        )

    def test_day_result_insufficient_day_candles(self):
        """Fewer than min_day_candles in the full day returns None."""
        predictor = ORBPredictor()
        # Only morning candles, no afternoon -- total < 20
        candles = _morning_candles("2026-02-10", n=8)
        dr = predictor.calculate_day_result(candles)

        assert dr is None

    def test_day_result_insufficient_morning(self):
        """If morning range fails (too few candles), DayResult is None."""
        predictor = ORBPredictor()
        # 2 morning candles + lots of afternoon -- morning fails
        dt_base = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        candles = [_candle(dt_base, 30.0, 30.5, 29.5, 30.2)]
        # Add 30 afternoon candles
        for i in range(30):
            dt = datetime(2026, 2, 10, 10 + i // 4, (i % 4) * 15, tzinfo=timezone.utc)
            candles.append(_candle(dt, 30.0, 30.1, 29.9, 30.0))
        dr = predictor.calculate_day_result(candles)
        assert dr is None

    def test_day_result_records_high_low_hours(self):
        """high_hour_utc and low_hour_utc correctly record when extremes occurred."""
        predictor = ORBPredictor()
        candles = _full_day_candles("2026-02-10", day_high_extra=0.5, day_low_extra=0.4)
        dr = predictor.calculate_day_result(candles)

        assert dr is not None
        assert 8 <= dr.high_hour_utc <= 22
        assert 8 <= dr.low_hour_utc <= 22


# ---------------------------------------------------------------------------
# TestPrediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for ORBPredictor.predict_daily_range."""

    @pytest.fixture
    def predictor(self):
        return ORBPredictor()

    @pytest.fixture
    def history(self, predictor):
        """Build 20 days of history and compute DayResults."""
        klines = _build_history(num_days=20)
        return predictor.calculate_all_days(klines)

    @pytest.fixture
    def morning(self, predictor):
        """A fresh morning range for today."""
        candles = _morning_candles("2026-02-25", base_price=32.0, spread=0.5, direction="up")
        return predictor.calculate_morning_range(candles)

    def test_predict_basic(self, predictor, history, morning):
        """With 20 historical days, returns a valid Prediction."""
        pred = predictor.predict_daily_range(morning, history)

        assert pred is not None
        assert isinstance(pred, Prediction)
        assert pred.date == "2026-02-25"
        assert pred.morning_high == morning.high
        assert pred.morning_low == morning.low
        assert pred.morning_direction == "up"
        assert pred.sample_size > 0

    def test_predict_conservative_below_aggressive_for_highs(self, predictor, history, morning):
        """Conservative high <= median high <= aggressive high."""
        pred = predictor.predict_daily_range(morning, history)

        assert pred is not None
        assert pred.predicted_high_conservative <= pred.predicted_high_median
        assert pred.predicted_high_median <= pred.predicted_high_aggressive

    def test_predict_conservative_above_aggressive_for_lows(self, predictor, history, morning):
        """Conservative low >= median low >= aggressive low (less downside)."""
        pred = predictor.predict_daily_range(morning, history)

        assert pred is not None
        # For lows, conservative = less downside extension = higher price
        # aggressive = more downside extension = lower price
        assert pred.predicted_low_conservative >= pred.predicted_low_median
        assert pred.predicted_low_median >= pred.predicted_low_aggressive

    def test_predict_highs_above_morning(self, predictor, history, morning):
        """All predicted highs should be >= morning high (extensions are positive)."""
        pred = predictor.predict_daily_range(morning, history)

        assert pred is not None
        # At minimum, conservative could equal morning_high (zero extension)
        assert pred.predicted_high_conservative >= morning.high - 0.01

    def test_predict_lows_below_morning(self, predictor, history, morning):
        """All predicted lows should be <= morning low (extensions are positive)."""
        pred = predictor.predict_daily_range(morning, history)

        assert pred is not None
        assert pred.predicted_low_conservative <= morning.low + 0.01

    def test_predict_direction_filter(self, predictor, history, morning):
        """Filtering by direction reduces sample size and may change predictions."""
        pred_no_filter = predictor.predict_daily_range(
            morning, history, use_direction_filter=False
        )
        pred_filtered = predictor.predict_daily_range(
            morning, history, use_direction_filter=True
        )

        assert pred_no_filter is not None
        assert pred_filtered is not None
        # Filtered should have equal or fewer samples
        assert pred_filtered.sample_size <= pred_no_filter.sample_size
        if pred_filtered.sample_size < pred_no_filter.sample_size:
            assert "direction=up" in pred_filtered.filters_applied

    def test_predict_range_filter(self, predictor, history, morning):
        """Filtering by range size reduces sample size."""
        pred = predictor.predict_daily_range(
            morning, history, use_direction_filter=False,
            use_range_filter=True,
        )

        assert pred is not None
        # Range filter applied if enough samples
        if pred.sample_size < len(history):
            assert any("range_size" in f for f in pred.filters_applied)

    def test_predict_insufficient_history(self, predictor, morning):
        """Fewer than min_sample historical days returns None."""
        # Only 3 days of history, min_sample=5 (default)
        klines = _build_history(num_days=3)
        short_history = predictor.calculate_all_days(klines)

        pred = predictor.predict_daily_range(morning, short_history, use_direction_filter=False)
        assert pred is None

    def test_predict_sample_size(self, predictor, history, morning):
        """Prediction.sample_size matches the filtered history count."""
        pred = predictor.predict_daily_range(
            morning, history, use_direction_filter=False
        )
        assert pred is not None
        assert pred.sample_size == len(history)

    def test_predict_with_both_filters(self, predictor, morning):
        """Both direction and range filters can be applied together."""
        # Build enough history so both filters still leave >= min_sample
        klines = _build_history(num_days=40)
        history = predictor.calculate_all_days(klines)

        pred = predictor.predict_daily_range(
            morning, history, use_direction_filter=True,
            use_range_filter=True,
        )
        assert pred is not None
        assert pred.sample_size >= 5

    def test_predict_min_sample_custom(self, predictor, history, morning):
        """Custom min_sample is respected."""
        pred = predictor.predict_daily_range(
            morning, history, use_direction_filter=False, min_sample=100
        )
        # 20 days < 100 min_sample => None
        assert pred is None

    def test_predict_filters_not_applied_when_too_few(self, predictor, morning):
        """Direction filter not applied if it would reduce below min_sample."""
        # Build 6 days: 4 up, 2 down. Direction filter for "up" keeps 4 (< 5*2=10 needed
        # for the filter to even be attempted).
        klines = _build_history(num_days=6)
        history = predictor.calculate_all_days(klines)

        pred = predictor.predict_daily_range(
            morning, history, use_direction_filter=True
        )
        # Should still return a prediction (using unfiltered data)
        assert pred is not None
        # Direction filter should NOT be in filters_applied because
        # len(filtered) < min_sample * 2
        assert "direction=up" not in pred.filters_applied


# ---------------------------------------------------------------------------
# TestWarrant
# ---------------------------------------------------------------------------


class TestWarrant:
    """Tests for ORBPredictor.translate_to_warrant."""

    def test_warrant_at_entry(self):
        """Silver at entry price => 0% change, 0 SEK P&L."""
        wt = ORBPredictor.translate_to_warrant(
            silver_target=90.55, entry_price=90.55, leverage=4.76, position_sek=150_000
        )

        assert isinstance(wt, WarrantTarget)
        assert wt.silver_price == 90.55
        assert wt.warrant_pct_change == pytest.approx(0.0, abs=1e-9)
        assert wt.warrant_sek_pnl == pytest.approx(0.0, abs=1e-6)
        assert wt.warrant_price_factor == pytest.approx(1.0, abs=1e-9)

    def test_warrant_above_entry(self):
        """Silver above entry => positive P&L."""
        wt = ORBPredictor.translate_to_warrant(
            silver_target=91.55, entry_price=90.55, leverage=4.76, position_sek=150_000
        )

        assert wt.warrant_pct_change > 0
        assert wt.warrant_sek_pnl > 0
        assert wt.warrant_price_factor > 1.0

    def test_warrant_below_entry(self):
        """Silver below entry => negative P&L."""
        wt = ORBPredictor.translate_to_warrant(
            silver_target=89.55, entry_price=90.55, leverage=4.76, position_sek=150_000
        )

        assert wt.warrant_pct_change < 0
        assert wt.warrant_sek_pnl < 0
        assert wt.warrant_price_factor < 1.0

    def test_warrant_leverage_effect(self):
        """Warrant % change should be approximately leverage * underlying % change."""
        entry = 90.55
        target = 91.55
        underlying_pct = (target - entry) / entry * 100

        wt = ORBPredictor.translate_to_warrant(
            silver_target=target, entry_price=entry, leverage=4.76, position_sek=150_000
        )

        # The MINI warrant: fl = entry - entry/leverage, intrinsic = entry - fl = entry/leverage
        # Change in intrinsic = target - entry = 1.0
        # % change = change / intrinsic * 100 = 1.0 / (entry/leverage) * 100
        # = 1.0 * leverage / entry * 100
        # So warrant_pct / underlying_pct should be approximately leverage
        ratio = wt.warrant_pct_change / underlying_pct
        assert ratio == pytest.approx(4.76, abs=0.1)

    def test_warrant_pnl_calculation(self):
        """P&L in SEK matches position_sek * pct_change / 100."""
        position = 200_000.0
        wt = ORBPredictor.translate_to_warrant(
            silver_target=92.0, entry_price=90.55, leverage=4.76, position_sek=position
        )

        expected_pnl = position * wt.warrant_pct_change / 100
        assert wt.warrant_sek_pnl == pytest.approx(expected_pnl, abs=1e-6)

    def test_warrant_symmetric_moves(self):
        """Equal up/down moves in silver produce (roughly) equal magnitude changes in warrant."""
        entry = 90.55
        delta = 1.0

        wt_up = ORBPredictor.translate_to_warrant(
            silver_target=entry + delta, entry_price=entry, leverage=4.76, position_sek=150_000
        )
        wt_down = ORBPredictor.translate_to_warrant(
            silver_target=entry - delta, entry_price=entry, leverage=4.76, position_sek=150_000
        )

        # Magnitude should be similar (not exactly equal due to % base difference)
        assert abs(wt_up.warrant_pct_change) == pytest.approx(
            abs(wt_down.warrant_pct_change), abs=0.5
        )


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for ORBPredictor.compute_statistics."""

    @pytest.fixture
    def predictor(self):
        return ORBPredictor()

    @pytest.fixture
    def day_results(self, predictor):
        klines = _build_history(num_days=20)
        return predictor.calculate_all_days(klines)

    def test_statistics_basic(self, predictor, day_results):
        """compute_statistics returns a dict with expected keys."""
        stats = predictor.compute_statistics(day_results)

        assert isinstance(stats, dict)
        expected_keys = [
            "total_days", "high_in_morning_pct", "low_in_morning_pct",
            "upside_ext", "downside_ext", "morning_range_pct",
            "up_morning_days", "down_morning_days",
            "high_hour_distribution", "low_hour_distribution",
            "up_morning_stats", "down_morning_stats",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_statistics_total_days(self, predictor, day_results):
        """total_days matches the number of DayResults passed in."""
        stats = predictor.compute_statistics(day_results)
        assert stats["total_days"] == len(day_results)

    def test_statistics_high_in_morning_count(self, predictor):
        """Correct count of days where the day's high was within the morning range."""
        # Build days where day_high_extra=0 (high stays in morning)
        klines_in_morning = _build_history(num_days=10, base_price=30.0)
        # Override: rebuild with no high extension
        all_klines = []
        dt = datetime(2026, 1, 5, tzinfo=timezone.utc)
        days_created = 0
        while days_created < 10:
            if dt.weekday() < 5:
                date_str = dt.strftime("%Y-%m-%d")
                candles = _full_day_candles(
                    date_str, morning_base=30.0, morning_spread=0.5,
                    direction="up", day_high_extra=0.0, day_low_extra=0.2,
                )
                all_klines.extend(candles)
                days_created += 1
            dt += timedelta(days=1)

        predictor_obj = ORBPredictor()
        results = predictor_obj.calculate_all_days(all_klines)
        stats = predictor_obj.compute_statistics(results)

        # All days should have high in morning (upside_ext_pct < 0.05)
        assert stats["high_in_morning_pct"] > 50.0

    def test_statistics_direction_breakdown(self, predictor, day_results):
        """Up/down morning counts sum to total_days."""
        stats = predictor.compute_statistics(day_results)
        assert stats["up_morning_days"] + stats["down_morning_days"] == stats["total_days"]

    def test_statistics_upside_ext_keys(self, predictor, day_results):
        """upside_ext contains mean, median, max, p25, p75."""
        stats = predictor.compute_statistics(day_results)
        for key in ["mean", "median", "max", "p25", "p75"]:
            assert key in stats["upside_ext"]
            assert key in stats["downside_ext"]

    def test_statistics_empty_results(self, predictor):
        """Empty day_results returns empty dict."""
        stats = predictor.compute_statistics([])
        assert stats == {}

    def test_statistics_morning_range_pct(self, predictor, day_results):
        """morning_range_pct stats are computed correctly."""
        stats = predictor.compute_statistics(day_results)
        ranges = [d.morning.range_pct for d in day_results]
        import statistics as st
        assert stats["morning_range_pct"]["mean"] == pytest.approx(st.mean(ranges), abs=1e-6)
        assert stats["morning_range_pct"]["median"] == pytest.approx(st.median(ranges), abs=1e-6)

    def test_statistics_up_down_morning_stats(self, predictor, day_results):
        """Per-direction stats (avg_upside_ext, avg_downside_ext) are populated."""
        stats = predictor.compute_statistics(day_results)

        if stats["up_morning_days"] > 0:
            assert "avg_upside_ext" in stats["up_morning_stats"]
            assert "avg_downside_ext" in stats["up_morning_stats"]
        if stats["down_morning_days"] > 0:
            assert "avg_upside_ext" in stats["down_morning_stats"]
            assert "avg_downside_ext" in stats["down_morning_stats"]


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for ORBPredictor."""

    def test_empty_klines(self):
        """Empty kline list returns empty group and no results."""
        predictor = ORBPredictor()
        days = predictor.group_by_day([])
        assert days == {}

        results = predictor.calculate_all_days([])
        assert results == []

    def test_single_day(self):
        """One valid day can produce a MorningRange and DayResult but not a Prediction."""
        predictor = ORBPredictor()
        candles = _full_day_candles("2026-02-10")
        results = predictor.calculate_all_days(candles)

        assert len(results) == 1
        assert results[0].date == "2026-02-10"

        # Cannot predict with only 1 day of history (min_sample=5)
        morning = predictor.calculate_morning_range(
            _morning_candles("2026-02-11", base_price=30.5)
        )
        pred = predictor.predict_daily_range(morning, results)
        assert pred is None

    def test_weekend_filtering(self):
        """Saturday and Sunday candles are excluded by group_by_day."""
        predictor = ORBPredictor()
        # Feb 7, 2026 is a Saturday; Feb 8 is Sunday
        sat_candles = _morning_candles("2026-02-07")
        sun_candles = _morning_candles("2026-02-08")
        mon_candles = _morning_candles("2026-02-09")

        # Mark the timestamps with correct weekday
        for c in sat_candles:
            c["ts"] = c["ts"].replace(day=7)  # Sat
        for c in sun_candles:
            c["ts"] = c["ts"].replace(day=8)  # Sun
        for c in mon_candles:
            c["ts"] = c["ts"].replace(day=9)  # Mon

        all_candles = sat_candles + sun_candles + mon_candles
        days = predictor.group_by_day(all_candles, weekdays_only=True)

        # Only Monday should remain
        assert "2026-02-07" not in days  # Saturday
        assert "2026-02-08" not in days  # Sunday
        assert "2026-02-09" in days      # Monday

    def test_weekend_not_filtered_when_disabled(self):
        """When weekdays_only=False, weekend candles are kept."""
        predictor = ORBPredictor()
        sat_candles = _morning_candles("2026-02-07")
        for c in sat_candles:
            c["ts"] = c["ts"].replace(day=7)

        days = predictor.group_by_day(sat_candles, weekdays_only=False)
        assert "2026-02-07" in days

    def test_group_by_day(self):
        """Candles are correctly grouped into per-day lists."""
        predictor = ORBPredictor()
        day1 = _morning_candles("2026-02-09", n=4)
        day2 = _morning_candles("2026-02-10", n=6)
        all_candles = day1 + day2

        days = predictor.group_by_day(all_candles)
        assert "2026-02-09" in days
        assert "2026-02-10" in days
        assert len(days["2026-02-09"]) == 4
        assert len(days["2026-02-10"]) == 6

    def test_parse_klines(self):
        """_parse_klines converts raw Binance arrays to dicts correctly."""
        predictor = ORBPredictor()
        ts_ms = int(datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc).timestamp() * 1000)
        raw = [[ts_ms, "30.0", "30.5", "29.5", "30.2", "1000.0",
                0, "0", 0, "0", "0", "0"]]
        parsed = predictor._parse_klines(raw)

        assert len(parsed) == 1
        assert parsed[0]["open"] == 30.0
        assert parsed[0]["high"] == 30.5
        assert parsed[0]["low"] == 29.5
        assert parsed[0]["close"] == 30.2
        assert parsed[0]["volume"] == 1000.0
        assert parsed[0]["hour"] == 8
        assert parsed[0]["date"] == "2026-02-10"

    def test_calculate_all_days(self):
        """calculate_all_days returns sorted, valid results and skips invalid days."""
        predictor = ORBPredictor()
        klines = _build_history(num_days=10)
        results = predictor.calculate_all_days(klines)

        assert len(results) > 0
        # Results should be sorted by date
        dates = [r.date for r in results]
        assert dates == sorted(dates)

    def test_custom_constructor_params(self):
        """ORBPredictor respects custom constructor parameters."""
        predictor = ORBPredictor(
            morning_start_utc=9,
            morning_end_utc=11,
            min_morning_candles=2,
            min_day_candles=10,
            min_morning_range_pct=0.001,
        )
        assert predictor.morning_start_utc == 9
        assert predictor.morning_end_utc == 11
        assert predictor.min_morning_candles == 2
        assert predictor.min_day_candles == 10
        assert predictor.min_morning_range_pct == 0.001


# ---------------------------------------------------------------------------
# TestFormatPrediction
# ---------------------------------------------------------------------------


class TestFormatPrediction:
    """Tests for ORBPredictor.format_prediction."""

    def test_format_prediction_basic(self):
        """format_prediction returns a non-empty string with expected content."""
        pred = Prediction(
            date="2026-02-25",
            morning_high=31.0,
            morning_low=30.5,
            morning_direction="up",
            morning_range_pct=1.64,
            predicted_high_conservative=31.10,
            predicted_high_median=31.25,
            predicted_high_aggressive=31.40,
            predicted_low_conservative=30.45,
            predicted_low_median=30.35,
            predicted_low_aggressive=30.20,
            sample_size=18,
            filters_applied=["direction=up"],
        )
        predictor = ORBPredictor()
        output = predictor.format_prediction(pred)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "2026-02-25" in output
        assert "UP" in output
        assert "Conservative" in output
        assert "Median" in output
        assert "Aggressive" in output
        assert "18 days" in output
        assert "SEK" in output

    def test_format_prediction_contains_warrant_info(self):
        """Formatted output includes warrant P&L information."""
        pred = Prediction(
            date="2026-02-25",
            morning_high=91.0,
            morning_low=90.0,
            morning_direction="up",
            morning_range_pct=1.1,
            predicted_high_conservative=91.5,
            predicted_high_median=92.0,
            predicted_high_aggressive=92.5,
            predicted_low_conservative=89.8,
            predicted_low_median=89.5,
            predicted_low_aggressive=89.0,
            sample_size=15,
            filters_applied=[],
        )
        predictor = ORBPredictor()
        output = predictor.format_prediction(pred)

        assert "Warrant:" in output
        assert "SEK" in output
        assert "Potential spread:" in output
