"""
TABaseStrategy — Multi-indicator confirmation strategy using TA-Lib.

Indicators: RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), ATR(14)
Entry: Multi-confirmation (RSI oversold + MACD bullish + EMA cross + volume)
Exit:  Single bearish signal (RSI overbought OR MACD bearish cross)
       + ATR-based trailing stoploss + stale trade timeout
"""

from datetime import datetime

import talib
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    stoploss_from_absolute,
    timeframe_to_prev_date,
)
from freqtrade.persistence import Trade


class TABaseStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # --- ROI table: close position at these profit thresholds ---
    minimal_roi = {
        "0": 0.05,  # 5% immediately
        "30": 0.03,  # 3% after 30 min
        "60": 0.02,  # 2% after 60 min
        "120": 0.01,  # 1% after 120 min
    }

    stoploss = -0.15
    use_custom_stoploss = True

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 30

    # --- Hyperoptable parameters ---
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 85, default=70, space="sell")
    min_confidence = DecimalParameter(0.25, 0.75, default=0.5, space="buy")
    volume_spike_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    base_confidence = DecimalParameter(0.15, 0.35, default=0.25, space="buy")
    atr_sl_mult = DecimalParameter(1.5, 4.0, default=2.0, space="sell")
    max_trade_candles = IntParameter(100, 500, default=200, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = talib.RSI(dataframe["close"], timeperiod=14)

        # MACD
        macd, signal, hist = talib.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd"] = macd
        dataframe["macd_signal"] = signal
        dataframe["macd_hist"] = hist

        # EMA fast/slow
        dataframe["ema_fast"] = talib.EMA(dataframe["close"], timeperiod=9)
        dataframe["ema_slow"] = talib.EMA(dataframe["close"], timeperiod=21)

        # Volume SMA
        dataframe["volume_sma"] = talib.SMA(
            dataframe["volume"].astype(np.float64), timeperiod=20
        )

        # ATR for dynamic stoploss
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bc = self.base_confidence.value

        # Individual signals as float columns (1.0 or 0.0)
        rsi_sig = (dataframe["rsi"] < self.rsi_oversold.value).astype(float)

        macd_above = dataframe["macd"] > dataframe["macd_signal"]
        macd_bullish = (macd_above & ~macd_above.shift(1, fill_value=False)).astype(
            float
        )

        ema_above = dataframe["ema_fast"] > dataframe["ema_slow"]
        ema_bullish = (ema_above & ~ema_above.shift(1, fill_value=False)).astype(float)

        vol_spike = (
            dataframe["volume"] > dataframe["volume_sma"] * self.volume_spike_mult.value
        ).astype(float)

        # Confidence: full signals = 1.0 weight, volume = 0.5 weight
        confidence = (rsi_sig + macd_bullish + ema_bullish + vol_spike * 0.5) * bc

        dataframe.loc[confidence >= self.min_confidence.value, "enter_long"] = 1
        dataframe.loc[dataframe["volume"] <= 0, "enter_long"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi_overbought = dataframe["rsi"] > self.rsi_overbought.value

        macd_above = dataframe["macd"] > dataframe["macd_signal"]
        macd_bearish = ~macd_above & macd_above.shift(1, fill_value=True)

        dataframe.loc[rsi_overbought | macd_bearish, "exit_long"] = 1
        dataframe.loc[dataframe["volume"] <= 0, "exit_long"] = 0

        return dataframe

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
        trade_dur = (candle_date - trade_candle).total_seconds() / 300  # 5m candles

        if trade_dur >= self.max_trade_candles.value and current_profit < 0.01:
            return "stale_trade_exit"

        return None
