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
from pathlib import Path
from typing import Any

import numpy as np
import requests

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.instrument_profile import (
    get_cross_asset_drivers,
    get_profile,
    get_regime_behavior,
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
        """Fetch cross-asset driver prices via yfinance (cached)."""
        drivers = get_cross_asset_drivers(self.ticker)
        result = {}
        try:
            import yfinance as yf
            for name, driver in drivers.items():
                yf_ticker = driver.get("ticker")
                if not yf_ticker:
                    continue
                try:
                    t = yf.Ticker(yf_ticker)
                    info = t.fast_info
                    price = getattr(info, "last_price", None)
                    if price and price > 0:
                        result[name] = float(price)
                except Exception:
                    pass
        except ImportError:
            pass
        return result

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _load_signal_data(self) -> dict:
        """Load latest signal data from agent_summary_compact."""
        summary = load_json(SUMMARY_PATH) or {}
        signals = (summary.get("signals") or {}).get(self.ticker, {})
        focus = (summary.get("focus_probabilities") or {}).get(self.ticker, {})
        mc = (summary.get("monte_carlo") or {}).get(self.ticker, {})
        return {
            "rsi": _safe_float(signals.get("rsi")),
            "action": signals.get("action", "HOLD"),
            "regime": signals.get("regime", ""),
            "confidence": _safe_float(signals.get("confidence")),
            "buy_count": (signals.get("extra") or {}).get("_buy_count", 0),
            "sell_count": (signals.get("extra") or {}).get("_sell_count", 0),
            "focus_3h": focus.get("3h", {}),
            "focus_1d": focus.get("1d", {}),
            "mc_return_1d": _safe_float((mc.get("expected_return_1d") or {}).get("mean_pct")),
            "mc_p_up": _safe_float(mc.get("p_up"), 0.5),
        }

    def _compute_z_score(self) -> float | None:
        """Compute current z-score for mean reversion tracking."""
        try:
            import yfinance as yf
            ticker_map = {"XAG-USD": "SI=F", "XAU-USD": "GC=F"}
            yf_sym = ticker_map.get(self.ticker)
            if not yf_sym:
                return None
            df = yf.download(yf_sym, period="5d", interval="15m", progress=False)
            if df.empty or len(df) < 20:
                return None
            close = df["Close"].values if "Close" in df.columns else df.iloc[:, 0].values
            close = close.flatten() if hasattr(close, "flatten") else close
            mean = float(np.mean(close[-60:]))
            std = float(np.std(close[-60:]))
            if std == 0:
                return 0.0
            return float((self.current_price - mean) / std)
        except Exception:
            return None

    def _compute_half_life(self) -> float | None:
        """Compute Ornstein-Uhlenbeck half-life from recent data."""
        try:
            import yfinance as yf
            ticker_map = {"XAG-USD": "SI=F", "XAU-USD": "GC=F"}
            yf_sym = ticker_map.get(self.ticker)
            if not yf_sym:
                return None
            df = yf.download(yf_sym, period="5d", interval="15m", progress=False)
            if df.empty or len(df) < 30:
                return None
            close = df["Close"].values if "Close" in df.columns else df.iloc[:, 0].values
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
        except Exception:
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

    def compute_exit_signals(self) -> list[dict]:
        """Evaluate all exit triggers. Returns list of triggered signals."""
        exits = []
        elapsed_hours = (time.time() - self.start_time) / 3600

        # 1. TP0: +5% underlying move in our favor
        if self.direction == "SHORT":
            move_pct = (self.entry_price - self.current_price) / self.entry_price * 100
        else:
            move_pct = (self.current_price - self.entry_price) / self.entry_price * 100

        if move_pct >= 5.0 and "TP0" not in self.alerts_sent:
            exits.append({
                "trigger": "TP0",
                "severity": "ACTION",
                "message": f"Take profit: +{move_pct:.1f}% in our favor. Sell 30% of position.",
                "move_pct": move_pct,
            })

        if move_pct >= 2.5 and "TP_PARTIAL" not in self.alerts_sent:
            exits.append({
                "trigger": "TP_PARTIAL",
                "severity": "WATCH",
                "message": f"Approaching TP: +{move_pct:.1f}%. Consider partial exit.",
                "move_pct": move_pct,
            })

        # 2. Conviction drop
        conviction_drop = self.entry_conviction - self.current_conviction
        if conviction_drop >= CONVICTION_DROP_ALERT and "CONVICTION_DROP" not in self.alerts_sent:
            exits.append({
                "trigger": "CONVICTION_DROP",
                "severity": "WARNING",
                "message": f"Conviction dropped {conviction_drop} pts ({self.entry_conviction} -> {self.current_conviction})",
            })

        # 3. Time decay
        if elapsed_hours >= 3.0 and "TIME_DECAY_3H" not in self.alerts_sent:
            exits.append({
                "trigger": "TIME_DECAY_3H",
                "severity": "WATCH",
                "message": f"Position held {elapsed_hours:.1f}h. Tighten stops.",
            })
        if elapsed_hours >= 5.0 and "TIME_DECAY_5H" not in self.alerts_sent:
            exits.append({
                "trigger": "TIME_DECAY_5H",
                "severity": "ACTION",
                "message": f"Position held {elapsed_hours:.1f}h. Consider closing.",
            })

        # 4. Adverse move
        if move_pct <= -3.0 and "ADVERSE_MOVE" not in self.alerts_sent:
            exits.append({
                "trigger": "ADVERSE_MOVE",
                "severity": "WARNING",
                "message": f"Position down {move_pct:.1f}%. Review thesis.",
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
            lines.append(
                f"  Signals: {action} ({buy_c}B/{sell_c}S) | RSI {rsi:.1f} | Regime: {regime}"
            )

            focus_3h = signal_data.get("focus_3h", {})
            focus_1d = signal_data.get("focus_1d", {})
            if focus_3h:
                dir_3h = focus_3h.get("direction", "?")
                prob_3h = _safe_float(focus_3h.get("probability"), 0.5)
                lines.append(f"  Focus 3h: {dir_3h} {prob_3h:.0%}", )
            if focus_1d:
                dir_1d = focus_1d.get("direction", "?")
                prob_1d = _safe_float(focus_1d.get("probability"), 0.5)
                lines[-1] += f" | Focus 1d: {dir_1d} {prob_1d:.0%}"

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
                    self.current_conviction = self._recompute_conviction()
                    self.last_signal_check = now

                    # Track signal history
                    self.signal_history.append({
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        "price": self.current_price,
                        "conviction": self.current_conviction,
                        "rsi": signal_data.get("rsi"),
                        "regime": signal_data.get("regime"),
                        "action": signal_data.get("action"),
                    })

                # 3. Cross-asset check (every CROSS_ASSET_INTERVAL)
                if now - self.last_cross_asset_check >= CROSS_ASSET_INTERVAL:
                    self.cross_asset_prices = self._fetch_cross_asset_prices()
                    self.last_cross_asset_check = now

                # 4. Exit signals
                exits = self.compute_exit_signals()
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
