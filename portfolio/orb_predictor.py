"""
Opening Range Breakout (ORB) Predictor for Silver (XAGUSDT)

Based on the well-known ORB / Initial Balance trading strategy:
- Observe the price range during 9-11 CET (08:00-10:00 UTC)
- Use historical extension statistics to predict the day's max/min
- Apply filters: morning direction, range size, volume

References:
- Toby Crabel, "Day Trading with Short Term Price Patterns and Opening Range Breakout" (1990)
- Market Profile "Initial Balance" concept (CBOT, 1980s)
- Academic: "Intraday Market Return Predictability" (Management Science, 2025)

Usage:
    from portfolio.orb_predictor import ORBPredictor
    predictor = ORBPredictor()
    days = predictor.fetch_historical_data(num_batches=5)
    morning = predictor.calculate_morning_range(today_klines)
    prediction = predictor.predict_daily_range(morning, days)
"""

import requests
import statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# === Constants ===
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "XAGUSDT"
MORNING_START_UTC = 8   # 09:00 CET = 08:00 UTC (winter)
MORNING_END_UTC = 10    # 11:00 CET = 10:00 UTC (winter)
DAY_START_UTC = 8       # Full trading day starts 08:00 UTC
DAY_END_UTC = 22        # Full trading day ends 22:00 UTC


@dataclass
class MorningRange:
    """Data from the 9-11 CET observation window."""
    date: str                   # YYYY-MM-DD
    open: float                 # First candle open
    high: float                 # Highest price in window
    low: float                  # Lowest price in window
    close: float                # Last candle close
    range_abs: float            # high - low in USD
    range_pct: float            # range as % of midpoint
    direction: str              # "up" if close > open, else "down"
    midpoint: float             # (high + low) / 2
    volume: float               # Total volume in window
    num_candles: int            # Number of 15m candles


@dataclass
class DayResult:
    """Full day outcome for backtesting."""
    date: str
    morning: MorningRange
    day_high: float
    day_low: float
    day_range_pct: float
    upside_ext_pct: float       # (day_high - morning_high) / morning_high * 100
    downside_ext_pct: float     # (morning_low - day_low) / morning_low * 100
    upside_ext_ratio: float     # upside_ext / morning_range
    downside_ext_ratio: float   # downside_ext / morning_range
    high_hour_utc: int          # Hour (UTC) when day's high occurred
    low_hour_utc: int           # Hour (UTC) when day's low occurred


@dataclass
class Prediction:
    """Predicted daily high/low with confidence intervals."""
    date: str
    morning_high: float
    morning_low: float
    morning_direction: str
    morning_range_pct: float
    predicted_high_conservative: float   # 25th percentile
    predicted_high_median: float         # 50th percentile
    predicted_high_aggressive: float     # 75th percentile
    predicted_low_conservative: float    # 25th percentile
    predicted_low_median: float          # 50th percentile
    predicted_low_aggressive: float      # 75th percentile
    sample_size: int                     # Number of historical days used
    filters_applied: list = field(default_factory=list)


@dataclass
class WarrantTarget:
    """Silver price translated to warrant price."""
    silver_price: float
    warrant_pct_change: float   # % change in warrant from entry
    warrant_sek_pnl: float      # SEK P&L on position
    warrant_price_factor: float # Multiply current warrant price by this


class ORBPredictor:
    """Opening Range Breakout predictor for silver."""

    def __init__(
        self,
        symbol: str = SYMBOL,
        morning_start_utc: int = MORNING_START_UTC,
        morning_end_utc: int = MORNING_END_UTC,
        day_start_utc: int = DAY_START_UTC,
        day_end_utc: int = DAY_END_UTC,
        min_morning_candles: int = 4,
        min_day_candles: int = 20,
        min_morning_range_pct: float = 0.01,
    ):
        self.symbol = symbol
        self.morning_start_utc = morning_start_utc
        self.morning_end_utc = morning_end_utc
        self.day_start_utc = day_start_utc
        self.day_end_utc = day_end_utc
        self.min_morning_candles = min_morning_candles
        self.min_day_candles = min_day_candles
        self.min_morning_range_pct = min_morning_range_pct

    # === Data Fetching ===

    def fetch_klines(self, num_batches: int = 5, interval: str = "15m",
                     limit: int = 1000, timeout: int = 10) -> list[dict]:
        """Fetch historical 15m klines from Binance FAPI.

        Returns list of candle dicts sorted by timestamp ascending.
        Each batch fetches `limit` candles going backwards in time.
        5 batches * 1000 candles * 15min = ~52 days of data.
        """
        all_klines = []
        end_time = None

        for _ in range(num_batches):
            params = {"symbol": self.symbol, "interval": interval, "limit": limit}
            if end_time:
                params["endTime"] = end_time

            resp = requests.get(BINANCE_FAPI_KLINES, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_klines = data + all_klines  # prepend older data
            end_time = data[0][0] - 1  # next batch ends before earliest candle

        return self._parse_klines(all_klines)

    def _parse_klines(self, raw_klines: list) -> list[dict]:
        """Parse Binance kline arrays into dicts."""
        parsed = []
        for k in raw_klines:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            parsed.append({
                "ts": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "hour": ts.hour,
                "minute": ts.minute,
                "date": ts.strftime("%Y-%m-%d"),
            })
        return parsed

    # === Grouping ===

    def group_by_day(self, klines: list[dict], weekdays_only: bool = True) -> dict[str, list[dict]]:
        """Group klines by date, optionally filtering weekdays only."""
        days = defaultdict(list)
        for k in klines:
            if weekdays_only and k["ts"].weekday() >= 5:
                continue
            days[k["date"]].append(k)
        return dict(days)

    # === Morning Range Calculation ===

    def calculate_morning_range(self, day_candles: list[dict]) -> Optional[MorningRange]:
        """Calculate the morning range (9-11 CET / 08:00-10:00 UTC) for a single day.

        Returns None if insufficient data.
        """
        morning = [
            c for c in day_candles
            if self.morning_start_utc <= c["hour"] < self.morning_end_utc
        ]

        if len(morning) < self.min_morning_candles:
            return None

        high = max(c["high"] for c in morning)
        low = min(c["low"] for c in morning)
        open_price = morning[0]["open"]
        close_price = morning[-1]["close"]
        mid = (high + low) / 2
        range_abs = high - low
        range_pct = range_abs / mid * 100 if mid > 0 else 0
        volume = sum(c["volume"] for c in morning)

        if range_pct < self.min_morning_range_pct:
            return None

        return MorningRange(
            date=day_candles[0]["date"],
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            range_abs=range_abs,
            range_pct=range_pct,
            direction="up" if close_price > open_price else "down",
            midpoint=mid,
            volume=volume,
            num_candles=len(morning),
        )

    # === Day Result Calculation ===

    def calculate_day_result(self, day_candles: list[dict]) -> Optional[DayResult]:
        """Calculate the full day outcome for backtesting.

        Returns None if insufficient morning or day data.
        """
        morning = self.calculate_morning_range(day_candles)
        if morning is None:
            return None

        full_day = [
            c for c in day_candles
            if self.day_start_utc <= c["hour"] <= self.day_end_utc
        ]

        if len(full_day) < self.min_day_candles:
            return None

        d_high = max(c["high"] for c in full_day)
        d_low = min(c["low"] for c in full_day)
        d_mid = (d_high + d_low) / 2
        d_range_pct = (d_high - d_low) / d_mid * 100 if d_mid > 0 else 0

        upside_ext = d_high - morning.high
        downside_ext = morning.low - d_low

        upside_ext_pct = upside_ext / morning.high * 100
        downside_ext_pct = downside_ext / morning.low * 100

        if morning.range_abs > 0.001:
            upside_ext_ratio = upside_ext / morning.range_abs
            downside_ext_ratio = downside_ext / morning.range_abs
        else:
            upside_ext_ratio = 0.0
            downside_ext_ratio = 0.0

        high_candle = max(full_day, key=lambda c: c["high"])
        low_candle = min(full_day, key=lambda c: c["low"])

        return DayResult(
            date=morning.date,
            morning=morning,
            day_high=d_high,
            day_low=d_low,
            day_range_pct=d_range_pct,
            upside_ext_pct=upside_ext_pct,
            downside_ext_pct=downside_ext_pct,
            upside_ext_ratio=upside_ext_ratio,
            downside_ext_ratio=downside_ext_ratio,
            high_hour_utc=high_candle["hour"],
            low_hour_utc=low_candle["hour"],
        )

    def calculate_all_days(self, klines: list[dict]) -> list[DayResult]:
        """Calculate DayResult for all valid trading days in the dataset."""
        days = self.group_by_day(klines)
        results = []
        for date in sorted(days.keys()):
            result = self.calculate_day_result(days[date])
            if result is not None:
                results.append(result)
        return results

    # === Prediction ===

    def predict_daily_range(
        self,
        morning: MorningRange,
        historical_days: list[DayResult],
        use_direction_filter: bool = True,
        use_range_filter: bool = False,
        min_sample: int = 5,
    ) -> Optional[Prediction]:
        """Predict the day's high/low based on morning range and historical statistics.

        Uses percentile-based extensions from historical data.
        Applies optional filters for morning direction and range size.

        Args:
            morning: Today's morning range data
            historical_days: Past day results to draw statistics from
            use_direction_filter: Filter historical days by same morning direction
            use_range_filter: Filter historical days by similar morning range size
            min_sample: Minimum historical days needed for prediction

        Returns:
            Prediction with conservative/median/aggressive targets, or None if insufficient data
        """
        filtered = list(historical_days)
        filters = []

        # Filter by morning direction
        if use_direction_filter and len(filtered) >= min_sample * 2:
            direction_filtered = [d for d in filtered if d.morning.direction == morning.direction]
            if len(direction_filtered) >= min_sample:
                filtered = direction_filtered
                filters.append(f"direction={morning.direction}")

        # Filter by morning range size (within same half: small or large)
        if use_range_filter and len(filtered) >= min_sample * 2:
            ranges = sorted(d.morning.range_pct for d in filtered)
            median_range = ranges[len(ranges) // 2]
            if morning.range_pct <= median_range:
                size_filtered = [d for d in filtered if d.morning.range_pct <= median_range]
            else:
                size_filtered = [d for d in filtered if d.morning.range_pct > median_range]
            if len(size_filtered) >= min_sample:
                filtered = size_filtered
                filters.append(f"range_size={'small' if morning.range_pct <= median_range else 'large'}")

        if len(filtered) < min_sample:
            return None

        # Extract extension percentages
        up_exts = sorted(d.upside_ext_pct for d in filtered)
        down_exts = sorted(d.downside_ext_pct for d in filtered)

        # Calculate percentiles
        def percentile(sorted_list, pct):
            idx = int(len(sorted_list) * pct / 100)
            idx = max(0, min(idx, len(sorted_list) - 1))
            return sorted_list[idx]

        up_25 = percentile(up_exts, 25)
        up_50 = percentile(up_exts, 50)
        up_75 = percentile(up_exts, 75)
        down_25 = percentile(down_exts, 25)
        down_50 = percentile(down_exts, 50)
        down_75 = percentile(down_exts, 75)

        return Prediction(
            date=morning.date,
            morning_high=morning.high,
            morning_low=morning.low,
            morning_direction=morning.direction,
            morning_range_pct=morning.range_pct,
            predicted_high_conservative=morning.high * (1 + up_25 / 100),
            predicted_high_median=morning.high * (1 + up_50 / 100),
            predicted_high_aggressive=morning.high * (1 + up_75 / 100),
            predicted_low_conservative=morning.low * (1 - down_25 / 100),
            predicted_low_median=morning.low * (1 - down_50 / 100),
            predicted_low_aggressive=morning.low * (1 - down_75 / 100),
            sample_size=len(filtered),
            filters_applied=filters,
        )

    # === Warrant Translation ===

    @staticmethod
    def translate_to_warrant(
        silver_target: float,
        entry_price: float = 90.55,
        leverage: float = 4.76,
        position_sek: float = 150_000,
        current_warrant_price: Optional[float] = None,
    ) -> WarrantTarget:
        """Translate a silver price target to warrant P&L.

        The warrant (MINI Long) has:
        - financing_level = entry_price - (entry_price / leverage)
        - intrinsic_value = silver_price - financing_level
        - warrant % change = (new_intrinsic - entry_intrinsic) / entry_intrinsic * 100
        """
        fl = entry_price - entry_price / leverage
        intrinsic_entry = entry_price - fl
        intrinsic_target = silver_target - fl
        pct_change = (intrinsic_target - intrinsic_entry) / intrinsic_entry * 100
        sek_pnl = position_sek * pct_change / 100

        # Factor to multiply current warrant price by
        if current_warrant_price and current_warrant_price > 0:
            factor = intrinsic_target / (silver_target - fl)  # This simplifies but let's keep explicit
            # Actually: factor = new_intrinsic / current_intrinsic
            # current_intrinsic = current_warrant_price (approximately, ignoring spread)
            factor = intrinsic_target / intrinsic_entry
        else:
            factor = intrinsic_target / intrinsic_entry

        return WarrantTarget(
            silver_price=silver_target,
            warrant_pct_change=pct_change,
            warrant_sek_pnl=sek_pnl,
            warrant_price_factor=factor,
        )

    # === Summary Statistics ===

    def compute_statistics(self, day_results: list[DayResult]) -> dict:
        """Compute summary statistics from historical day results."""
        if not day_results:
            return {}

        up_exts = [d.upside_ext_pct for d in day_results]
        down_exts = [d.downside_ext_pct for d in day_results]
        ranges = [d.morning.range_pct for d in day_results]

        # How often does morning contain the day's extreme?
        high_in_morning = sum(1 for d in day_results if d.upside_ext_pct < 0.05)
        low_in_morning = sum(1 for d in day_results if d.downside_ext_pct < 0.05)

        # Timing of daily highs/lows
        high_hours = defaultdict(int)
        low_hours = defaultdict(int)
        for d in day_results:
            high_hours[d.high_hour_utc] += 1
            low_hours[d.low_hour_utc] += 1

        # Direction breakdown
        up_mornings = [d for d in day_results if d.morning.direction == "up"]
        down_mornings = [d for d in day_results if d.morning.direction == "down"]

        return {
            "total_days": len(day_results),
            "high_in_morning_pct": high_in_morning / len(day_results) * 100,
            "low_in_morning_pct": low_in_morning / len(day_results) * 100,
            "upside_ext": {
                "mean": statistics.mean(up_exts),
                "median": statistics.median(up_exts),
                "max": max(up_exts),
                "p25": sorted(up_exts)[len(up_exts) // 4],
                "p75": sorted(up_exts)[len(up_exts) * 3 // 4],
            },
            "downside_ext": {
                "mean": statistics.mean(down_exts),
                "median": statistics.median(down_exts),
                "max": max(down_exts),
                "p25": sorted(down_exts)[len(down_exts) // 4],
                "p75": sorted(down_exts)[len(down_exts) * 3 // 4],
            },
            "morning_range_pct": {
                "mean": statistics.mean(ranges),
                "median": statistics.median(ranges),
            },
            "up_morning_days": len(up_mornings),
            "down_morning_days": len(down_mornings),
            "high_hour_distribution": dict(high_hours),
            "low_hour_distribution": dict(low_hours),
            "up_morning_stats": {
                "avg_upside_ext": statistics.mean([d.upside_ext_pct for d in up_mornings]) if up_mornings else 0,
                "avg_downside_ext": statistics.mean([d.downside_ext_pct for d in up_mornings]) if up_mornings else 0,
            },
            "down_morning_stats": {
                "avg_upside_ext": statistics.mean([d.upside_ext_pct for d in down_mornings]) if down_mornings else 0,
                "avg_downside_ext": statistics.mean([d.downside_ext_pct for d in down_mornings]) if down_mornings else 0,
            },
        }

    # === Formatting ===

    def format_prediction(self, prediction: Prediction, warrant_entry: float = 90.55,
                          warrant_leverage: float = 4.76, position_sek: float = 150_000) -> str:
        """Format a prediction into a readable string."""
        lines = [
            f"=== ORB Prediction for {prediction.date} ===",
            f"Morning (9-11 CET): HIGH ${prediction.morning_high:.2f} | LOW ${prediction.morning_low:.2f}",
            f"Direction: {prediction.morning_direction.upper()} | Range: {prediction.morning_range_pct:.2f}%",
            f"Sample: {prediction.sample_size} days | Filters: {', '.join(prediction.filters_applied) or 'none'}",
            "",
            "Predicted DAY HIGH:",
        ]

        for label, price in [
            ("Conservative (25th)", prediction.predicted_high_conservative),
            ("Median (50th)", prediction.predicted_high_median),
            ("Aggressive (75th)", prediction.predicted_high_aggressive),
        ]:
            wt = self.translate_to_warrant(price, warrant_entry, warrant_leverage, position_sek)
            lines.append(f"  {label}: ${price:.2f} | Warrant: {wt.warrant_pct_change:+.1f}% = {wt.warrant_sek_pnl:+,.0f} SEK")

        lines.append("")
        lines.append("Predicted DAY LOW:")

        for label, price in [
            ("Conservative (25th)", prediction.predicted_low_conservative),
            ("Median (50th)", prediction.predicted_low_median),
            ("Aggressive (75th)", prediction.predicted_low_aggressive),
        ]:
            wt = self.translate_to_warrant(price, warrant_entry, warrant_leverage, position_sek)
            lines.append(f"  {label}: ${price:.2f} | Warrant: {wt.warrant_pct_change:+.1f}% = {wt.warrant_sek_pnl:+,.0f} SEK")

        lines.append("")
        lines.append("Strategy: BUY at predicted low (median), SELL at predicted high (median)")
        buy_wt = self.translate_to_warrant(prediction.predicted_low_median, warrant_entry, warrant_leverage, position_sek)
        sell_wt = self.translate_to_warrant(prediction.predicted_high_median, warrant_entry, warrant_leverage, position_sek)
        spread_sek = sell_wt.warrant_sek_pnl - buy_wt.warrant_sek_pnl
        lines.append(f"  BUY target: ${prediction.predicted_low_median:.2f} | SELL target: ${prediction.predicted_high_median:.2f}")
        lines.append(f"  Potential spread: {spread_sek:+,.0f} SEK")

        return "\n".join(lines)
