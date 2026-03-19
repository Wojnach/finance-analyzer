"""Augmented signal gates for GoldDigger — volatility, momentum, structure.

Fetches 1-minute klines from Binance FAPI, builds a DataFrame, and runs
existing signal modules as secondary gates to filter noise trades.

These signals refresh every ~60 seconds (not every 5s poll) since they
operate on 1-minute candle data, not tick-level.
"""

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.golddigger.augmented")

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
_REFRESH_INTERVAL = 60  # seconds between kline fetches


@dataclass
class AugmentedState:
    """Cached state of secondary signal gates."""
    volatility_action: str = "HOLD"
    volatility_confidence: float = 0.0
    volatility_details: dict = field(default_factory=dict)

    momentum_action: str = "HOLD"
    momentum_confidence: float = 0.0
    momentum_details: dict = field(default_factory=dict)

    structure_action: str = "HOLD"
    structure_confidence: float = 0.0
    structure_details: dict = field(default_factory=dict)

    last_refresh: float = 0.0
    kline_count: int = 0

    def entry_allowed(self, require_vol_confirm: bool = True,
                      block_on_momentum_sell: bool = True,
                      block_on_structure_sell: bool = True) -> tuple[bool, str]:
        """Check if secondary gates allow entry.

        Returns (allowed, reason).
        """
        reasons = []

        # Volatility gate: prefer entry when vol is expanding (BUY)
        if require_vol_confirm and self.volatility_action == "SELL":
            reasons.append(f"vol={self.volatility_action}")

        # Momentum gate: block entry on strong bearish momentum
        if block_on_momentum_sell and self.momentum_action == "SELL":
            reasons.append(f"mom={self.momentum_action}")

        # Structure gate: block entry at structural resistance / bearish structure
        if block_on_structure_sell and self.structure_action == "SELL":
            reasons.append(f"struct={self.structure_action}")

        if reasons:
            return False, "secondary gates blocked: " + ", ".join(reasons)
        return True, "secondary gates OK"

    def summary(self) -> str:
        """One-line summary for logging/Telegram."""
        return (f"vol={self.volatility_action}({self.volatility_confidence:.0%}) "
                f"mom={self.momentum_action}({self.momentum_confidence:.0%}) "
                f"struct={self.structure_action}({self.structure_confidence:.0%})")


class AugmentedSignals:
    """Manages secondary signal computation from Binance FAPI klines."""

    def __init__(self, symbol: str = "XAUUSDT", lookback_bars: int = 120,
                 refresh_interval: float = _REFRESH_INTERVAL):
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.refresh_interval = refresh_interval
        self._state = AugmentedState()
        self._last_fetch_time = 0.0

    @property
    def state(self) -> AugmentedState:
        return self._state

    def refresh_if_needed(self) -> AugmentedState:
        """Refresh signals if enough time has elapsed since last fetch."""
        now = time.time()
        if now - self._last_fetch_time < self.refresh_interval:
            return self._state

        df = self._fetch_klines()
        if df is None or len(df) < 50:
            logger.debug("Not enough klines for augmented signals (%d)",
                         len(df) if df is not None else 0)
            return self._state

        self._compute_signals(df)
        self._last_fetch_time = now
        self._state.last_refresh = now
        self._state.kline_count = len(df)

        logger.info("Augmented signals refreshed: %s (bars=%d)",
                    self._state.summary(), len(df))
        return self._state

    def _fetch_klines(self) -> pd.DataFrame | None:
        """Fetch 1-minute klines from Binance FAPI."""
        try:
            r = fetch_with_retry(
                f"{BINANCE_FAPI}/klines",
                params={
                    "symbol": self.symbol,
                    "interval": "1m",
                    "limit": self.lookback_bars,
                },
                timeout=10,
            )
            if r is None:
                return None
            r.raise_for_status()
            raw = r.json()
            if not raw:
                return None

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_vol",
                "taker_buy_quote_vol", "ignore",
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df

        except Exception as e:
            logger.warning("Kline fetch failed: %s", e)
            return None

    def _compute_signals(self, df: pd.DataFrame):
        """Run volatility, momentum, and structure signals on kline DataFrame."""
        # Volatility signal
        try:
            from portfolio.signals.volatility import compute_volatility_signal
            result = compute_volatility_signal(df)
            self._state.volatility_action = result.get("action", "HOLD")
            self._state.volatility_confidence = result.get("confidence", 0.0)
            self._state.volatility_details = result.get("sub_signals", {})
        except Exception as e:
            logger.warning("Volatility signal failed: %s", e)
            self._state.volatility_action = "HOLD"

        # Momentum factors signal
        try:
            from portfolio.signals.momentum_factors import compute_momentum_factors_signal
            result = compute_momentum_factors_signal(df)
            self._state.momentum_action = result.get("action", "HOLD")
            self._state.momentum_confidence = result.get("confidence", 0.0)
            self._state.momentum_details = result.get("sub_signals", {})
        except Exception as e:
            logger.warning("Momentum signal failed: %s", e)
            self._state.momentum_action = "HOLD"

        # Structure signal
        try:
            from portfolio.signals.structure import compute_structure_signal
            result = compute_structure_signal(df)
            self._state.structure_action = result.get("action", "HOLD")
            self._state.structure_confidence = result.get("confidence", 0.0)
            self._state.structure_details = result.get("sub_signals", {})
        except Exception as e:
            logger.warning("Structure signal failed: %s", e)
            self._state.structure_action = "HOLD"

    def compute_from_klines(self, klines: list) -> AugmentedState:
        """Compute signals from raw kline list (for backtesting).

        klines: list of Binance kline arrays [ts, open, high, low, close, vol, ...]
        """
        if not klines or len(klines) < 50:
            return self._state

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_vol",
            "taker_buy_quote_vol", "ignore",
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        self._compute_signals(df)
        self._state.last_refresh = time.time()
        self._state.kline_count = len(df)
        return self._state
