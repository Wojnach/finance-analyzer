"""
TABaseStrategy — Multi-indicator confirmation strategy using TA-Lib.

Indicators (5m): RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), ATR(14),
                 15 candlestick patterns
Indicators (1h): EMA(50), ADX(14) — trend filter
Entry: Multi-confirmation confidence scoring, gated by 1h trend filter
Exit:  Single bearish signal (RSI overbought OR MACD bearish cross)
       + ATR-based trailing stoploss + stale trade timeout
Risk:  ATR-based position sizing, daily loss limit, drawdown kill switch
"""

from datetime import datetime, timezone

import talib
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    informative,
    stoploss_from_absolute,
    timeframe_to_prev_date,
)
from freqtrade.persistence import Trade

BULLISH_PATTERNS = [
    "CDLHAMMER",
    "CDLINVERTEDHAMMER",
    "CDLENGULFING",
    "CDLMORNINGSTAR",
    "CDLPIERCING",
    "CDL3WHITESOLDIERS",
    "CDLHARAMI",
    "CDLMARUBOZU",
    "CDLKICKING",
    "CDLDOJI",
]

BEARISH_PATTERNS = [
    "CDLSHOOTINGSTAR",
    "CDLHANGINGMAN",
    "CDLEVENINGSTAR",
    "CDLDARKCLOUDCOVER",
    "CDL3BLACKCROWS",
    "CDLENGULFING",
    "CDLHARAMI",
    "CDLMARUBOZU",
    "CDLKICKING",
    "CDLDOJI",
]


class TABaseStrategy(IStrategy):

    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0.01,
    }

    stoploss = -0.15
    use_custom_stoploss = True

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 50

    # --- Hyperoptable: entry ---
    rsi_oversold = IntParameter(25, 45, default=35, space="buy")
    volume_spike_mult = DecimalParameter(1.0, 2.5, default=1.5, space="buy")
    adx_threshold = IntParameter(10, 30, default=20, space="buy")

    # --- Hyperoptable: exit ---
    rsi_overbought = IntParameter(60, 85, default=70, space="sell")
    atr_sl_mult = DecimalParameter(1.5, 4.0, default=2.0, space="sell")
    max_trade_candles = IntParameter(100, 500, default=200, space="sell")

    # --- Hyperoptable: risk ---
    max_daily_loss_pct = DecimalParameter(0.02, 0.10, default=0.05, space="sell")
    max_drawdown_pct = DecimalParameter(0.05, 0.20, default=0.15, space="sell")

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = talib.EMA(dataframe["close"], timeperiod=50)
        dataframe["adx"] = talib.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = talib.RSI(dataframe["close"], timeperiod=14)

        macd, signal, hist = talib.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd"] = macd
        dataframe["macd_signal"] = signal
        dataframe["macd_hist"] = hist

        dataframe["ema_fast"] = talib.EMA(dataframe["close"], timeperiod=9)
        dataframe["ema_slow"] = talib.EMA(dataframe["close"], timeperiod=21)

        dataframe["volume_sma"] = talib.SMA(
            dataframe["volume"].astype(np.float64), timeperiod=20
        )

        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe["bb_lower"] = bb_lower
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_upper"] = bb_upper

        # Candlestick patterns
        o, h, l, c = (
            dataframe["open"],
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
        )
        all_patterns = set(BULLISH_PATTERNS + BEARISH_PATTERNS)
        results = {name: getattr(talib, name)(o, h, l, c) for name in all_patterns}

        bullish_score = sum((results[n] > 0).astype(float) for n in BULLISH_PATTERNS)
        bearish_score = sum((results[n] < 0).astype(float) for n in BEARISH_PATTERNS)
        dataframe["cdl_bullish"] = bullish_score
        dataframe["cdl_bearish"] = bearish_score

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi_below = dataframe["rsi"] < self.rsi_oversold.value
        rsi_cross = rsi_below & ~rsi_below.shift(1, fill_value=False)

        macd_hist_pos = dataframe["macd_hist"] > 0
        macd_hist_cross = macd_hist_pos & ~macd_hist_pos.shift(1, fill_value=False)

        bb_touch = (dataframe["close"] < dataframe["bb_lower"]) & (
            dataframe["rsi"] < 40
        )

        trigger = rsi_cross | macd_hist_cross | bb_touch

        guards = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (
                dataframe["volume"]
                > dataframe["volume_sma"] * self.volume_spike_mult.value
            )
            & (dataframe["close_1h"] > dataframe["ema_50_1h"])
            & (dataframe["adx_1h"] > self.adx_threshold.value)
        )

        dataframe.loc[trigger & guards, "enter_long"] = 1
        dataframe.loc[dataframe["volume"] <= 0, "enter_long"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi_overbought = dataframe["rsi"] > self.rsi_overbought.value

        macd_above = dataframe["macd"] > dataframe["macd_signal"]
        macd_bearish = ~macd_above & macd_above.shift(1, fill_value=True)

        dataframe.loc[rsi_overbought | macd_bearish, "exit_long"] = 1
        dataframe.loc[dataframe["volume"] <= 0, "exit_long"] = 0

        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        atr = dataframe.iloc[-1]["atr"]
        if atr is None or np.isnan(atr) or atr <= 0:
            return proposed_stake

        wallet = self.wallets.get_total_stake_amount()
        risk_amount = wallet * 0.01
        position_value = risk_amount / (atr / current_rate)

        return min(position_value, max_stake)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        candle = dataframe.iloc[-1]
        if candle["atr"] is None or np.isnan(candle["atr"]):
            return None

        stop_price = current_rate - (candle["atr"] * self.atr_sl_mult.value)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        trade_candle = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        candle_date = dataframe.iloc[-1]["date"]
        trade_dur = (candle_date - trade_candle).total_seconds() / 300

        if trade_dur >= self.max_trade_candles.value and current_profit < 0.01:
            return "stale_trade_exit"

        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        today_start = current_time.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        closed_today = Trade.get_trades_proxy(is_open=False, close_date=today_start)

        if closed_today:
            wallet = self.wallets.get_total_stake_amount()
            daily_pnl = sum(t.close_profit_abs or 0.0 for t in closed_today)
            if (
                daily_pnl < 0
                and abs(daily_pnl) >= wallet * self.max_daily_loss_pct.value
            ):
                return False

        open_trades = Trade.get_trades_proxy(is_open=True)
        if open_trades:
            wallet = self.wallets.get_total_stake_amount()
            open_pnl = sum(t.close_profit_abs or 0.0 for t in open_trades)
            if open_pnl < 0 and abs(open_pnl) >= wallet * self.max_drawdown_pct.value:
                return False

        return True
