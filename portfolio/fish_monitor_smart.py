"""Smart fishing monitor — signal-aware position tracking with exit intelligence.

Replaces dumb price polling with continuous signal evaluation.
Monitors cross-asset drivers, conviction shifts, regime changes,
and computes signal-driven exit triggers.

Usage (from CLI):
    .venv/Scripts/python.exe scripts/fin_fish.py --monitor --ticker XAG-USD

Usage (programmatic):
    from portfolio.fish_monitor_smart import SmartFishMonitor
    monitor = SmartFishMonitor("XAG-USD", entry_price=75.20, direction="SHORT")
    monitor.run()  # blocking loop
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import requests

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.instrument_profile import (
    get_cross_asset_drivers,
    get_profile,
)

logger = logging.getLogger("portfolio.fish_monitor_smart")

BASE_DIR = Path(__file__).resolve().parent.parent
MONITOR_LOG = BASE_DIR / "data" / "fish_monitor_log.jsonl"
MONITOR_STATE = BASE_DIR / "data" / "fish_monitor_state.json"
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary_compact.json"

BINANCE_FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FAPI_24HR = "https://fapi.binance.com/fapi/v1/ticker/24hr"

# Monitoring intervals
FAST_INTERVAL = 30       # seconds between price checks
SIGNAL_INTERVAL = 300    # seconds between full signal re-evaluation (5 min)
CROSS_ASSET_INTERVAL = 120  # seconds between cross-asset checks (2 min)

# Alert thresholds
CONVICTION_DROP_ALERT = 20   # alert if conviction drops >20 pts
PRICE_MOVE_ALERT_PCT = 1.0   # alert on >1% move from entry
CROSS_ASSET_ALERT = True     # alert on significant cross-asset moves


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


class SmartFishMonitor:
    """Signal-aware position monitor for fishing sessions.

    Tracks live price, cross-asset drivers, signal conviction,
    and generates exit signals based on multiple intelligence sources.
    """

    def __init__(
        self,
        ticker: str,
        entry_price: float,
        direction: str = "SHORT",
        entry_conviction: int = 50,
        cert_entry_price: float = 0.0,
        cert_units: int = 0,
        cert_leverage: float = 5.0,
        tp_targets: list[float] | None = None,
        sl_price: float = 0.0,
    ):
        self.ticker = ticker
        self.entry_price = entry_price
        self.direction = direction  # "LONG" or "SHORT"
        self.entry_conviction = entry_conviction
        self.cert_entry = cert_entry_price
        self.cert_units = cert_units
        self.cert_leverage = cert_leverage
        self.tp_targets = tp_targets or []
        self.sl_price = sl_price

        self.profile = get_profile(ticker) or {}
        self.binance_symbol = self.profile.get("binance_symbol", "XAGUSDT")

        # State
        self.current_price = entry_price
        self.current_conviction = entry_conviction
        self.session_high = entry_price
        self.session_low = entry_price
        self.last_signal_check = 0.0
        self.last_cross_asset_check = 0.0
        self.alerts_sent: set[str] = set()
        self.cross_asset_prices: dict[str, float] = {}
        self.cross_asset_baselines: dict[str, float] = {}
        self.signal_history: list[dict] = []
        self.start_time = time.time()
        self.check_count = 0

        # Lesson 39: MC stability — track last 3 values, require 2 consecutive
        self.mc_history: list[float] = []
        # Lesson 53: metals loop disagreement counter
        self.metals_disagree_count = 0
        # Lesson 50: news/event flag from last signal check
        self.news_action = "HOLD"

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def _fetch_price(self) -> float | None:
        """Fetch current price from Binance FAPI."""
        try:
            r = requests.get(
                f"{BINANCE_FAPI_TICKER}?symbol={self.binance_symbol}",
                timeout=5,
            )
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception as e:
            logger.warning("Price fetch error: %s", e)
        return None

    def _fetch_cross_asset_prices(self) -> dict[str, float]:
        """Fetch cross-asset driver prices.

        2026-04-14: routed via price_source — commodity drivers (SI=F,
        GC=F, CL=F) hit Binance FAPI for real-time, ETFs hit Alpaca,
        CBOE indices fall back to yfinance only when needed.
        """
        from portfolio.price_source import fetch_klines

        drivers = get_cross_asset_drivers(self.ticker)
        result = {}
        for name, driver in drivers.items():
            ticker = driver.get("ticker")
            if not ticker:
                continue
            with suppress(Exception):
                df = fetch_klines(ticker, interval="1d", limit=2)
                if df is not None and not df.empty:
                    price = float(df["close"].iloc[-1])
                    if price > 0:
                        result[name] = price
        return result

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _load_signal_data(self) -> dict:
        """Load ALL available signal data from both loops.

        Sources:
        - agent_summary_compact.json: 30-signal consensus, MC, focus probs, regime
        - agent_summary.json: price levels (fib, pivot, keltner), forecast signals
        - metals_signal_log.jsonl: metals loop per-check signals + LLM predictions
        - forecast_predictions.jsonl: Chronos/Kronos latest predictions
        """
        # --- Main loop data (agent_summary_compact) ---
        summary = load_json(SUMMARY_PATH) or {}
        signals = (summary.get("signals") or {}).get(self.ticker, {})
        focus = (summary.get("focus_probabilities") or {}).get(self.ticker, {})
        mc = (summary.get("monte_carlo") or {}).get(self.ticker, {})
        forecast = (summary.get("forecast_signals") or {}).get(self.ticker, {})
        extra = signals.get("extra") or {}

        result = {
            # Core signals
            "rsi": _safe_float(signals.get("rsi")),
            "action": signals.get("action", "HOLD"),
            "regime": signals.get("regime", ""),
            "confidence": _safe_float(signals.get("confidence")),
            "w_confidence": _safe_float(signals.get("weighted_confidence",
                                                     signals.get("w_confidence"))),
            "buy_count": extra.get("_buy_count", 0),
            "sell_count": extra.get("_sell_count", 0),
            "voters": extra.get("_voters", 0),
            "vote_detail": extra.get("_vote_detail", ""),
            "confluence": _safe_float(extra.get("_confluence_score")),
            # Directional probabilities
            "focus_3h": focus.get("3h", {}),
            "focus_1d": focus.get("1d", {}),
            "focus_3d": focus.get("3d", {}),
            # Monte Carlo
            "mc_return_1d": _safe_float((mc.get("expected_return_1d") or {}).get("mean_pct")),
            "mc_p_up": _safe_float(mc.get("p_up"), 0.5),
            "mc_bands_1d": mc.get("price_bands_1d", {}),
            "mc_bands_3d": mc.get("price_bands_3d", {}),
            # Forecast models
            "chronos_1h_pct": _safe_float(forecast.get("chronos_1h_pct")),
            "chronos_24h_pct": _safe_float(forecast.get("chronos_24h_pct")),
        }

        # --- Full agent_summary: price levels + news ---
        full_summary = load_json(BASE_DIR / "data" / "agent_summary.json") or {}
        # News/event action (lesson 50)
        enhanced = (full_summary.get("signals") or {}).get(self.ticker, {}).get("enhanced_signals", {})
        result["news_action"] = (enhanced.get("news_event") or {}).get("action", "HOLD")
        result["econ_action"] = (enhanced.get("econ_calendar") or {}).get("action", "HOLD")

        price_levels = (full_summary.get("price_levels") or {}).get(self.ticker, {})
        if price_levels:
            result["fib_382"] = _safe_float(price_levels.get("fib_382"))
            result["fib_50"] = _safe_float(price_levels.get("fib_5"))
            result["fib_618"] = _safe_float(price_levels.get("fib_618"))
            result["pivot_pp"] = _safe_float(price_levels.get("pivot_pp"))
            result["pivot_s1"] = _safe_float(price_levels.get("pivot_s1"))
            result["pivot_r1"] = _safe_float(price_levels.get("pivot_r1"))
            result["keltner_upper"] = _safe_float(price_levels.get("keltner_upper"))
            result["keltner_lower"] = _safe_float(price_levels.get("keltner_lower"))

        # --- Metals loop: LLM predictions ---
        with suppress(Exception):
            metals_log = BASE_DIR / "data" / "metals_signal_log.jsonl"
            lines = metals_log.read_text().strip().split("\n")
            if lines:
                last_metals = json.loads(lines[-1])
                llm = (last_metals.get("llm") or {}).get(self.ticker, {})
                if llm:
                    result["ministral_action"] = llm.get("ministral", "HOLD")
                    result["ministral_conf"] = _safe_float(llm.get("ministral_conf"))
                    result["chronos_1h_dir"] = llm.get("chronos_1h", "flat")
                    result["chronos_3h_dir"] = llm.get("chronos_3h", "flat")
                    result["chronos_1h_move"] = _safe_float(llm.get("chronos_1h_pct_move"))
                    result["chronos_3h_move"] = _safe_float(llm.get("chronos_3h_pct_move"))
                    result["llm_consensus"] = llm.get("consensus_action", "HOLD")

                # Also grab metals loop signal
                metals_sig = (last_metals.get("signals") or {}).get(self.ticker, {})
                if metals_sig:
                    result["metals_action"] = metals_sig.get("action", "HOLD")
                    result["metals_rsi"] = _safe_float(metals_sig.get("rsi"))
                    result["metals_buy"] = metals_sig.get("buy_count", 0)
                    result["metals_sell"] = metals_sig.get("sell_count", 0)

        return result

    def _compute_z_score(self) -> float | None:
        """Compute current z-score for mean reversion tracking.

        2026-04-14: routed via price_source — SI=F/GC=F → Binance FAPI 15m bars.
        """
        try:
            from portfolio.price_source import fetch_klines
            ticker_map = {"XAG-USD": "SI=F", "XAU-USD": "GC=F"}
            sym = ticker_map.get(self.ticker)
            if not sym:
                return None
            df = fetch_klines(sym, interval="15m", limit=120, period="5d")
            if df is None or df.empty or len(df) < 20:
                return None
            close = df["close"].values
            close = close.flatten() if hasattr(close, "flatten") else close
            mean = float(np.mean(close[-60:]))
            std = float(np.std(close[-60:]))
            if std == 0:
                return 0.0
            return float((self.current_price - mean) / std)
        except Exception as e:
            logger.warning("Z-score computation failed: %s", e, exc_info=True)
            return None

    def _compute_half_life(self) -> float | None:
        """Compute Ornstein-Uhlenbeck half-life from recent data.

        2026-04-14: routed via price_source — SI=F/GC=F → Binance FAPI 15m bars.
        """
        try:
            from portfolio.price_source import fetch_klines
            ticker_map = {"XAG-USD": "SI=F", "XAU-USD": "GC=F"}
            sym = ticker_map.get(self.ticker)
            if not sym:
                return None
            df = fetch_klines(sym, interval="15m", limit=120, period="5d")
            if df is None or df.empty or len(df) < 30:
                return None
            close = df["close"].values
            close = close.flatten() if hasattr(close, "flatten") else close
            log_prices = np.log(close[-60:])
            y = np.diff(log_prices)
            x = log_prices[:-1]
            x_mean = x.mean()
            y_mean = y.mean()
            ss_xx = np.sum((x - x_mean) ** 2)
            if ss_xx == 0:
                return None
            theta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
            if theta >= 0:
                return None  # no mean reversion
            return float(-np.log(2) / theta)
        except Exception as e:
            logger.warning("Half-life computation failed: %s", e, exc_info=True)
            return None

    def _recompute_conviction(self) -> int:
        """Re-run preflight-style conviction scoring."""
        try:
            from scripts.fish_preflight import compute_preflight
            pf = compute_preflight(self.ticker)
            if self.direction == "SHORT":
                return pf["bear_score"]
            return pf["bull_score"]
        except Exception:
            return self.current_conviction

    # ------------------------------------------------------------------
    # Exit signal detection
    # ------------------------------------------------------------------

    def update_signal_state(self, signal_data: dict) -> None:
        """Update internal state from latest signal data.

        Call this each signal check to maintain MC history,
        metals loop disagreement tracking, and news flag.
        """
        # MC history for stability checks (lesson 39)
        mc_p_up = signal_data.get("mc_p_up", 0.5)
        self.mc_history.append(mc_p_up)
        if len(self.mc_history) > 3:
            self.mc_history.pop(0)

        # Metals loop disagreement tracking (lesson 53)
        metals_action = signal_data.get("metals_action", "HOLD")
        if metals_action != "HOLD":
            position_bullish = self.direction == "LONG"
            metals_bullish = metals_action == "BUY"
            if position_bullish != metals_bullish:
                self.metals_disagree_count += 1
            else:
                self.metals_disagree_count = 0

        # News flag (lesson 50)
        self.news_action = signal_data.get("news_action", "HOLD")

    def mc_stable_bearish(self) -> bool:
        """MC P(up) < 30% for last 2 consecutive checks."""
        return len(self.mc_history) >= 2 and all(m < 0.30 for m in self.mc_history[-2:])

    def mc_stable_bullish(self) -> bool:
        """MC P(up) > 70% for last 2 consecutive checks."""
        return len(self.mc_history) >= 2 and all(m > 0.70 for m in self.mc_history[-2:])

    def compute_exit_signals(self, signal_data: dict | None = None) -> list[dict]:
        """Evaluate all exit triggers. Returns list of triggered signals.

        Exit rules (from backtested lessons):
        - LONG combined exit: RSI > 62 AND MC < 35% (66.7% backtest win rate)
        - SHORT solo exit: RSI < 30 (backtest shows combined doesn't work for shorts)
        - Signal flip: vote margin > 4
        - Metals loop disagreement: 2+ consecutive checks
        - TP/SL: +2% / -2% underlying
        - Time: 3h tighten, session end force sell
        """
        exits = []
        elapsed_hours = (time.time() - self.start_time) / 3600

        if self.direction == "SHORT":
            move_pct = (self.entry_price - self.current_price) / self.entry_price * 100
        else:
            move_pct = (self.current_price - self.entry_price) / self.entry_price * 100

        rsi = signal_data.get("rsi", 50) if signal_data else 50
        mc_p = signal_data.get("mc_p_up", 0.5) if signal_data else 0.5
        buy_c = signal_data.get("buy_count", 0) if signal_data else 0
        sell_c = signal_data.get("sell_count", 0) if signal_data else 0

        # --- Lesson 45: Combined RSI+MC exit (backtested) ---
        # LONG: RSI > 62 AND MC < 35% → 66.7% win rate, catches peaks early
        if (self.direction == "LONG" and rsi > 62 and mc_p < 0.35
                and "COMBINED_EXIT" not in self.alerts_sent):
            exits.append({
                "trigger": "COMBINED_EXIT",
                "severity": "ACTION",
                "message": f"Combined exit: RSI {rsi:.0f} > 62 AND MC {mc_p:.0%} < 35% (backtested 66.7% win rate)",
            })
        # SHORT: RSI < 30 solo (backtest shows combined doesn't work for shorts)
        if (self.direction == "SHORT" and rsi < 30
                and "RSI_EXIT" not in self.alerts_sent):
            exits.append({
                "trigger": "RSI_EXIT",
                "severity": "ACTION",
                "message": f"RSI oversold exit: RSI {rsi:.0f} < 30",
            })

        # --- Signal flip with 4+ vote margin ---
        if self.direction == "LONG" and sell_c > buy_c + 4 and "SIGNAL_FLIP" not in self.alerts_sent:
            exits.append({
                "trigger": "SIGNAL_FLIP",
                "severity": "ACTION",
                "message": f"Strong SELL flip: {sell_c}S > {buy_c}B + 4",
            })
        if self.direction == "SHORT" and buy_c > sell_c + 4 and "SIGNAL_FLIP" not in self.alerts_sent:
            exits.append({
                "trigger": "SIGNAL_FLIP",
                "severity": "ACTION",
                "message": f"Strong BUY flip: {buy_c}B > {sell_c}S + 4",
            })

        # --- Lesson 53: Metals loop disagreement ---
        if self.metals_disagree_count >= 2 and "METALS_DISAGREE" not in self.alerts_sent:
            exits.append({
                "trigger": "METALS_DISAGREE",
                "severity": "WARNING",
                "message": f"Metals loop disagrees with position for {self.metals_disagree_count} consecutive checks",
            })

        # --- TP/SL ---
        if move_pct >= 2.0 and "TP" not in self.alerts_sent:
            exits.append({
                "trigger": "TP",
                "severity": "ACTION",
                "message": f"Take profit: +{move_pct:.1f}% underlying",
            })
        if move_pct <= -2.0 and "SL" not in self.alerts_sent:
            exits.append({
                "trigger": "SL",
                "severity": "ACTION",
                "message": f"Stop loss: {move_pct:.1f}% underlying",
            })

        # --- Time decay ---
        if elapsed_hours >= 3.0 and "TIME_DECAY_3H" not in self.alerts_sent:
            exits.append({
                "trigger": "TIME_DECAY_3H",
                "severity": "WATCH",
                "message": f"Position held {elapsed_hours:.1f}h. Tighten stops.",
            })

        # --- Adverse move warning ---
        if move_pct <= -3.0 and "ADVERSE_MOVE" not in self.alerts_sent:
            exits.append({
                "trigger": "ADVERSE_MOVE",
                "severity": "WARNING",
                "message": f"Position down {move_pct:.1f}%. Review thesis.",
            })

        # --- Lesson 50: News event flag ---
        news = signal_data.get("news_action", self.news_action) if signal_data else self.news_action
        if news == "SELL" and self.direction == "LONG" and "NEWS_ADVERSE" not in self.alerts_sent:
            exits.append({
                "trigger": "NEWS_ADVERSE",
                "severity": "WATCH",
                "message": "News signal SELL while LONG. Check headlines.",
            })
        if news == "BUY" and self.direction == "SHORT" and "NEWS_ADVERSE" not in self.alerts_sent:
            exits.append({
                "trigger": "NEWS_ADVERSE",
                "severity": "WATCH",
                "message": "News signal BUY while SHORT. Check headlines.",
            })

        # 5. Cross-asset divergence
        for name, baseline in self.cross_asset_baselines.items():
            current = self.cross_asset_prices.get(name)
            if current is None or baseline == 0:
                continue
            ca_move = (current - baseline) / baseline * 100
            driver = (get_cross_asset_drivers(self.ticker) or {}).get(name, {})
            threshold = driver.get("threshold_pct", 1.0)
            corr = driver.get("correlation", 0)

            if abs(ca_move) >= threshold:
                # Check if move is adverse for our position
                expected_impact = ca_move * corr
                is_adverse = (
                    (self.direction == "LONG" and expected_impact < -threshold)
                    or (self.direction == "SHORT" and expected_impact > threshold)
                )
                if is_adverse and f"ca_{name}" not in self.alerts_sent:
                    exits.append({
                        "trigger": f"CROSS_ASSET_{name.upper()}",
                        "severity": "WATCH",
                        "message": (
                            f"{name} moved {ca_move:+.1f}% (corr {corr:+.2f} with {self.ticker}). "
                            f"Expected impact: {expected_impact:+.1f}% -- adverse for {self.direction}."
                        ),
                    })

        return exits

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _format_status(self, signal_data: dict | None = None) -> str:
        """Format current monitoring status for display."""
        elapsed = (time.time() - self.start_time) / 3600
        if self.direction == "SHORT":
            move_pct = (self.entry_price - self.current_price) / self.entry_price * 100
        else:
            move_pct = (self.current_price - self.entry_price) / self.entry_price * 100

        cert_pnl = ""
        if self.cert_entry > 0 and self.cert_units > 0:
            # Estimate cert P&L from underlying move
            cert_move = move_pct * self.cert_leverage
            cert_value = self.cert_entry * self.cert_units * (1 + cert_move / 100)
            cert_cost = self.cert_entry * self.cert_units
            pnl_sek = cert_value - cert_cost
            cert_pnl = f" | Cert P&L: {pnl_sek:+.0f} SEK ({cert_move:+.1f}%)"

        lines = [
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"{self.ticker} ${self.current_price:.2f} "
            f"({move_pct:+.1f}% {'in favor' if move_pct > 0 else 'adverse'})"
            f"{cert_pnl}",
            f"  Session: H ${self.session_high:.2f} / L ${self.session_low:.2f} | "
            f"Elapsed: {elapsed:.1f}h | Conv: {self.current_conviction}/100",
        ]

        if signal_data:
            rsi = signal_data.get("rsi", 0)
            regime = signal_data.get("regime", "?")
            action = signal_data.get("action", "?")
            buy_c = signal_data.get("buy_count", 0)
            sell_c = signal_data.get("sell_count", 0)
            w_conf = signal_data.get("w_confidence", 0)
            lines.append(
                f"  Main loop: {action} ({buy_c}B/{sell_c}S) wConf={w_conf:.0%} | "
                f"RSI {rsi:.1f} | Regime: {regime}"
            )

            # Metals loop signals (independent computation) + disagreement flag
            metals_action = signal_data.get("metals_action")
            if metals_action:
                m_buy = signal_data.get("metals_buy", 0)
                m_sell = signal_data.get("metals_sell", 0)
                m_rsi = signal_data.get("metals_rsi", 0)
                disagree = ""
                if self.metals_disagree_count >= 2:
                    disagree = f" !! DISAGREES x{self.metals_disagree_count}"
                lines.append(
                    f"  Metals loop: {metals_action} ({m_buy}B/{m_sell}S) | RSI {m_rsi:.1f}{disagree}"
                )

            # News/event flags (lesson 50)
            news = signal_data.get("news_action", "HOLD")
            econ = signal_data.get("econ_action", "HOLD")
            if news != "HOLD" or econ != "HOLD":
                lines.append(f"  Events: news={news} econ={econ}")

            # Focus probabilities (all horizons)
            focus_parts = []
            for h_name, h_key in [("3h", "focus_3h"), ("1d", "focus_1d"), ("3d", "focus_3d")]:
                fd = signal_data.get(h_key, {})
                if fd:
                    dir_ = fd.get("direction", "?")
                    prob = _safe_float(fd.get("probability"), 0.5)
                    focus_parts.append(f"{h_name}:{dir_} {prob:.0%}")
            if focus_parts:
                lines.append(f"  Focus: {' | '.join(focus_parts)}")

            # Monte Carlo
            mc_p = signal_data.get("mc_p_up", 0.5)
            mc_ret = signal_data.get("mc_return_1d", 0)
            mc_1d = signal_data.get("mc_bands_1d", {})
            if mc_1d:
                lines.append(
                    f"  MC: P(up)={mc_p:.0%} exp={mc_ret:+.2f}% | "
                    f"1d: ${mc_1d.get('5',0):.2f}-${mc_1d.get('95',0):.2f}"
                )

            # LLM predictions (Chronos + Ministral from metals loop)
            llm_parts = []
            chr_1h = signal_data.get("chronos_1h_dir")
            chr_3h = signal_data.get("chronos_3h_dir")
            chr_1h_move = signal_data.get("chronos_1h_move", 0)
            chr_3h_move = signal_data.get("chronos_3h_move", 0)
            if chr_1h and chr_1h != "flat":
                llm_parts.append(f"Chr1h:{chr_1h}({chr_1h_move:+.2f}%)")
            if chr_3h and chr_3h != "flat":
                llm_parts.append(f"Chr3h:{chr_3h}({chr_3h_move:+.2f}%)")
            ministral = signal_data.get("ministral_action")
            if ministral and ministral != "HOLD":
                llm_parts.append(f"Ministral:{ministral}")
            if llm_parts:
                lines.append(f"  LLM: {' | '.join(llm_parts)}")

            # Key price levels
            fib_382 = signal_data.get("fib_382", 0)
            pivot_s1 = signal_data.get("pivot_s1", 0)
            pivot_r1 = signal_data.get("pivot_r1", 0)
            if fib_382 > 0:
                lines.append(
                    f"  Levels: S1=${pivot_s1:.2f} | PP=${signal_data.get('pivot_pp',0):.2f} | "
                    f"R1=${pivot_r1:.2f} | Fib38=${fib_382:.2f}"
                )

        # Cross-asset status
        if self.cross_asset_prices:
            ca_parts = []
            for name, price in self.cross_asset_prices.items():
                baseline = self.cross_asset_baselines.get(name, price)
                if baseline > 0:
                    ca_move = (price - baseline) / baseline * 100
                    ca_parts.append(f"{name} {ca_move:+.1f}%")
            if ca_parts:
                lines.append(f"  Cross-assets: {' | '.join(ca_parts)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_checks: int = 0, quiet: bool = False) -> list[dict]:
        """Run the monitoring loop.

        Parameters
        ----------
        max_checks : int
            Stop after N checks (0 = run until session end or manual stop).
        quiet : bool
            If True, suppress console output (still logs to file).

        Returns
        -------
        list[dict]
            All exit signals triggered during the session.
        """
        all_exits: list[dict] = []

        if not quiet:
            profile_name = self.profile.get("name", self.ticker)
            print(f"\n{'='*60}")
            print(f"  SMART MONITOR: {profile_name} {self.direction}")
            print(f"  Entry: ${self.entry_price:.2f} | Conviction: {self.entry_conviction}/100")
            print(f"{'='*60}\n")

        # Initialize cross-asset baselines
        self.cross_asset_prices = self._fetch_cross_asset_prices()
        self.cross_asset_baselines = dict(self.cross_asset_prices)

        try:
            while True:
                self.check_count += 1

                # 1. Fetch price
                price = self._fetch_price()
                if price:
                    self.current_price = price
                    self.session_high = max(self.session_high, price)
                    self.session_low = min(self.session_low, price)

                # 2. Signal re-evaluation (every SIGNAL_INTERVAL)
                signal_data = None
                now = time.time()
                if now - self.last_signal_check >= SIGNAL_INTERVAL:
                    signal_data = self._load_signal_data()
                    self.update_signal_state(signal_data)
                    self.last_signal_check = now

                    # Track signal history
                    self.signal_history.append({
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        "price": self.current_price,
                        "conviction": self.current_conviction,
                        "rsi": signal_data.get("rsi"),
                        "regime": signal_data.get("regime"),
                        "action": signal_data.get("action"),
                        "mc_p_up": signal_data.get("mc_p_up"),
                        "metals_action": signal_data.get("metals_action"),
                        "news_action": signal_data.get("news_action"),
                    })

                # 3. Cross-asset check (every CROSS_ASSET_INTERVAL)
                if now - self.last_cross_asset_check >= CROSS_ASSET_INTERVAL:
                    self.cross_asset_prices = self._fetch_cross_asset_prices()
                    self.last_cross_asset_check = now

                # 4. Exit signals
                exits = self.compute_exit_signals(signal_data)
                for ex in exits:
                    trigger = ex["trigger"]
                    if trigger not in self.alerts_sent:
                        self.alerts_sent.add(trigger)
                        all_exits.append(ex)
                        if not quiet:
                            severity = ex["severity"]
                            marker = "!!!" if severity == "ACTION" else "**" if severity == "WARNING" else "--"
                            print(f"\n  {marker} {ex['trigger']}: {ex['message']}")

                # 5. Display status
                if not quiet:
                    status = self._format_status(signal_data)
                    print(f"\n{status}")

                # 6. Log state
                log_entry = {
                    "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                    "ticker": self.ticker,
                    "price": self.current_price,
                    "direction": self.direction,
                    "conviction": self.current_conviction,
                    "session_high": self.session_high,
                    "session_low": self.session_low,
                    "check": self.check_count,
                    "exits_triggered": [e["trigger"] for e in exits],
                }
                atomic_append_jsonl(MONITOR_LOG, log_entry)

                # 7. Check termination
                if max_checks > 0 and self.check_count >= max_checks:
                    break

                # 8. Sleep
                time.sleep(FAST_INTERVAL)

        except KeyboardInterrupt:
            if not quiet:
                print("\n\nMonitoring stopped by user.")

        # Save final state
        final_state = {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "final_price": self.current_price,
            "session_high": self.session_high,
            "session_low": self.session_low,
            "final_conviction": self.current_conviction,
            "total_checks": self.check_count,
            "exits_triggered": [e["trigger"] for e in all_exits],
            "signal_history": self.signal_history[-20:],  # last 20 entries
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        atomic_write_json(MONITOR_STATE, final_state)

        if not quiet:
            if self.direction == "SHORT":
                final_move = (self.entry_price - self.current_price) / self.entry_price * 100
            else:
                final_move = (self.current_price - self.entry_price) / self.entry_price * 100
            print(f"\n  Final: ${self.current_price:.2f} ({final_move:+.1f}%) | {len(all_exits)} exit signals triggered")

        return all_exits
