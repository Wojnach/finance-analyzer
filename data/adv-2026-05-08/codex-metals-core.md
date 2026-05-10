OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-metals-core
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e0833-edc2-7873-be1c-6ee42058e81c
--------
user
changes against 'empty-baseline'
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git status --short && git diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9 && git diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-metals-core
2026-05-08T15:28:08.393853Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-metals-core'
'Q:/finance-analyzer/.worktrees/adv-metals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-metals-core

 exited 1 in 566ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-metals-core'
'Q:/finance-analyzer/.worktrees/adv-metals-core/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-metals-core

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-metals-core status --short; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-metals-core diff --stat 13ac6ef928b85b50a617d7b92e9edab2d08705e9; git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-metals-core diff --name-only 13ac6ef928b85b50a617d7b92e9edab2d08705e9' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 615ms:
 data/metals_loop.py               | 7699 +++++++++++++++++++++++++++++++++++++
 portfolio/exit_optimizer.py       |  752 ++++
 portfolio/fin_fish.py             | 1459 +++++++
 portfolio/fin_snipe.py            |  277 ++
 portfolio/fin_snipe_manager.py    | 1765 +++++++++
 portfolio/iskbets.py              |  928 +++++
 portfolio/metals_cross_assets.py  |  310 ++
 portfolio/metals_ladder.py        |  188 +
 portfolio/metals_orderbook.py     |  121 +
 portfolio/metals_precompute.py    | 1218 ++++++
 portfolio/microstructure.py       |  227 ++
 portfolio/microstructure_state.py |  236 ++
 portfolio/orb_postmortem.py       |  318 ++
 portfolio/orb_predictor.py        |  509 +++
 portfolio/price_targets.py        |  536 +++
 portfolio/silver_precompute.py    |   34 +
 16 files changed, 16577 insertions(+)
data/metals_loop.py
portfolio/exit_optimizer.py
portfolio/fin_fish.py
portfolio/fin_snipe.py
portfolio/fin_snipe_manager.py
portfolio/iskbets.py
portfolio/metals_cross_assets.py
portfolio/metals_ladder.py
portfolio/metals_orderbook.py
portfolio/metals_precompute.py
portfolio/microstructure.py
portfolio/microstructure_state.py
portfolio/orb_postmortem.py
portfolio/orb_predictor.py
portfolio/price_targets.py
portfolio/silver_precompute.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import py_compile, sys
files = [
'data/metals_loop.py',
'portfolio/exit_optimizer.py',
'portfolio/fin_fish.py',
'portfolio/fin_snipe.py',
'portfolio/fin_snipe_manager.py',
'portfolio/iskbets.py',
'portfolio/metals_cross_assets.py',
'portfolio/metals_ladder.py',
'portfolio/metals_orderbook.py',
'portfolio/metals_precompute.py',
'portfolio/microstructure.py',
'portfolio/microstructure_state.py',
'portfolio/orb_postmortem.py',
'portfolio/orb_predictor.py',
'portfolio/price_targets.py',
'portfolio/silver_precompute.py',
]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK', f)
    except Exception as e:
        print('ERR', f, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 1005ms:
OK data/metals_loop.py
OK portfolio/exit_optimizer.py
OK portfolio/fin_fish.py
OK portfolio/fin_snipe.py
OK portfolio/fin_snipe_manager.py
OK portfolio/iskbets.py
OK portfolio/metals_cross_assets.py
OK portfolio/metals_ladder.py
OK portfolio/metals_orderbook.py
OK portfolio/metals_precompute.py
OK portfolio/microstructure.py
OK portfolio/microstructure_state.py
OK portfolio/orb_postmortem.py
OK portfolio/orb_predictor.py
OK portfolio/price_targets.py
OK portfolio/silver_precompute.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_snipe.py -TotalCount 250' in Q:\finance-analyzer\.worktrees\adv-metals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_fish.py -TotalCount 250' in Q:\finance-analyzer\.worktrees\adv-metals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_snipe_manager.py -TotalCount 250' in Q:\finance-analyzer\.worktrees\adv-metals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path data/metals_loop.py -TotalCount 250' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 710ms:
"""Fin Fish: intraday dip/spike fishing planner for metals warrants.

Computes optimal limit buy levels for metals warrants based on:
- Recent daily range patterns (how deep do dips / how high do spikes go?)
- ATR-based volatility and first-passage-time fill probabilities
- Structural support/resistance levels (Fibonacci, pivot, smart money)
- Chronos/model drift for directional bias
- RSI-based direction selection (BULL vs BEAR fishing)
- Avanza warrant barrier safety checks

Supports both BULL (buy dips) and BEAR (buy spikes) fishing.

The output is a ranked table of fishing levels with fill probability,
expected gain on bounce, EV in SEK, and barrier distance.

Machine-readable output via ``compute_fishing_plan()`` for snipe manager
integration.  CLI one-shot via ``main()``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
from contextlib import suppress
from pathlib import Path
from typing import Any

import requests

from portfolio.file_utils import atomic_append_jsonl, load_json
from portfolio.monte_carlo import drift_from_probability, volatility_from_atr
from portfolio.price_targets import (
    fill_probability,
    fill_probability_buy,
    structural_levels,
)

# ---------------------------------------------------------------------------
# External config — import from data.fin_fish_config with inline fallbacks
# ---------------------------------------------------------------------------
try:
    from data.fin_fish_config import (
        FISHING_BUDGET_SEK as _CFG_BUDGET,
    )
    from data.fin_fish_config import (
        FISHING_MIN_FILL_PROB as _CFG_MIN_FILL,
    )
    from data.fin_fish_config import (
        FISHING_PREFER_AVA as _CFG_PREFER_AVA,
    )
    from data.fin_fish_config import (
        FISHING_SL_CASCADE as _CFG_SL_CASCADE,
    )
    from data.fin_fish_config import (
        FISHING_TP_CASCADE as _CFG_TP_CASCADE,
    )
    from data.fin_fish_config import (
        PREFERRED_INSTRUMENTS as _CFG_PREFERRED,
    )
    from data.fin_fish_config import (  # type: ignore[import-untyped]
        WARRANT_CATALOG as _CFG_CATALOG,
    )
except Exception:
    _CFG_CATALOG = None
    _CFG_PREFERRED = None
    _CFG_BUDGET = None
    _CFG_MIN_FILL = None
    _CFG_TP_CASCADE = None
    _CFG_SL_CASCADE = None
    _CFG_PREFER_AVA = None

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary.json"
FISH_LOG_PATH = BASE_DIR / "data" / "fin_fish_log.jsonl"

logger = logging.getLogger("portfolio.fin_fish")

BINANCE_FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FAPI_PRICE = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"

UNDERLYING_SYMBOLS = {"XAG-USD": "XAGUSDT", "XAU-USD": "XAUUSDT"}

# ---------------------------------------------------------------------------
# Inline defaults — used when data.fin_fish_config is absent
# ---------------------------------------------------------------------------

# Warrant catalog — BULL and BEAR instruments
_DEFAULT_CATALOG: dict[str, dict] = {
    # --- Silver BULL ---
    "BULL_SILVER_X5_AVA_3": {
        "ob_id": "1069606",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X5 AVA 3",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Silver BEAR ---
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BULL ---
    "BULL_GULD_X5_AVA": {
        "ob_id": "738811",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL GULD X5 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BEAR (no viable AVA X5 — gold rally killed them all) ---
    "BEAR_GULD_X5_VON4": {
        "ob_id": "1047859",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X5 VON4",
        "issuer": "VON",
        "spread_pct": 2.2,
        "commission_sek": 0,
    },
    "BEAR_GULD_X2_AVA": {
        "ob_id": "738805",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 2.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X2 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
}

# Preferred instruments per (underlying, direction) — snipe manager picks these first
_DEFAULT_PREFERRED: dict[tuple[str, str], str] = {
    ("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_3",
    ("XAG-USD", "SHORT"): "BEAR_SILVER_X5_AVA_12",
    ("XAU-USD", "LONG"): "BULL_GULD_X5_AVA",
    ("XAU-USD", "SHORT"): "BEAR_GULD_X5_VON4",
}

_DEFAULT_BUDGET_SEK = 20_000
_DEFAULT_MIN_FILL_PROB = 0.02
_DEFAULT_TP_CASCADE: list[Any] = [
    {"underlying_pct": 1.5, "sell_pct": 40, "action": "move_stop_to_breakeven"},
    {"underlying_pct": 2.5, "sell_pct": 40, "action": "trail_stop_1pct"},
    {"underlying_pct": 4.0, "sell_pct": 20, "action": "close"},
]
_DEFAULT_SL_CASCADE: list[Any] = [
    {"underlying_pct": -1.0, "sell_pct": 50, "action": "partial_stop"},
    {"underlying_pct": -2.0, "sell_pct": 100, "action": "full_stop"},
]
_DEFAULT_PREFER_AVA = True

# Resolve config vs defaults
WARRANT_CATALOG: dict[str, dict] = _CFG_CATALOG if _CFG_CATALOG is not None else _DEFAULT_CATALOG
PREFERRED_INSTRUMENTS: dict[tuple[str, str], str] = (
    _CFG_PREFERRED if _CFG_PREFERRED is not None else _DEFAULT_PREFERRED
)
FISHING_BUDGET_SEK: float = _CFG_BUDGET if _CFG_BUDGET is not None else _DEFAULT_BUDGET_SEK
FISHING_MIN_FILL_PROB: float = _CFG_MIN_FILL if _CFG_MIN_FILL is not None else _DEFAULT_MIN_FILL_PROB
FISHING_TP_CASCADE: list[Any] = _CFG_TP_CASCADE if _CFG_TP_CASCADE is not None else _DEFAULT_TP_CASCADE
FISHING_SL_CASCADE: list[Any] = _CFG_SL_CASCADE if _CFG_SL_CASCADE is not None else _DEFAULT_SL_CASCADE
FISHING_PREFER_AVA: bool = _CFG_PREFER_AVA if _CFG_PREFER_AVA is not None else _DEFAULT_PREFER_AVA

# Avanza warrant hours (CET)
AVANZA_OPEN_H, AVANZA_OPEN_M = 8, 15
AVANZA_CLOSE_H, AVANZA_CLOSE_M = 21, 55

MIN_BARRIER_DISTANCE_PCT = 5.0
DEFAULT_BOUNCE_PCT = 2.0  # +2% underlying = take-profit target


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_live_spot() -> dict[str, dict]:
    """Fetch live spot prices and 24h stats from Binance FAPI."""
    result = {}
    for ticker, symbol in UNDERLYING_SYMBOLS.items():
        try:
            r = requests.get(f"{BINANCE_FAPI_TICKER}?symbol={symbol}", timeout=5)
            d = r.json()
            result[ticker] = {
                "price": float(d["lastPrice"]),
                "high_24h": float(d["highPrice"]),
                "low_24h": float(d["lowPrice"]),
                "change_pct": float(d["priceChangePercent"]),
                "volume_usd": float(d["quoteVolume"]),
            }
        except Exception as e:
            logger.warning("Binance %s error: %s", ticker, e)
    return result


def fetch_daily_ranges(ticker: str, days: int = 10) -> list[dict]:
    """Fetch recent daily candles for range analysis."""
    symbol = UNDERLYING_SYMBOLS.get(ticker)
    if not symbol:
        return []
    try:
        r = requests.get(
            BINANCE_FAPI_KLINES,
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=10,
        )
        candles = r.json()
        result = []
        for c in candles:
            high = float(c[2])
            low = float(c[3])
            close = float(c[4])
            result.append({

 succeeded in 727ms:
"""Fin Snipe: intraday metals bid/exit ladder reporter.

This is the named entry point for the Avanza metals ladder workflow so the
feature is recognizable in the codebase and can later grow into a fuller
intraday snipe/scalp manager.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from portfolio.avanza_session import api_get, verify_session
from portfolio.file_utils import load_json
from portfolio.metals_ladder import build_intraday_ladder, map_underlying_name

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary.json"
SILVER_ANALYSIS_PATH = BASE_DIR / "data" / "silver_analysis.json"

logger = logging.getLogger("portfolio.fin_snipe")


def _value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _load_json(path: Path) -> dict:
    """Load a JSON file safely via file_utils (atomic-read, TOCTOU-safe)."""
    return load_json(path) or {}


def fetch_open_orders() -> list[dict]:
    """Return all open Avanza orders."""
    payload = api_get("/_api/trading/rest/orders")
    if isinstance(payload, dict):
        payload = payload.get("orders", [])
    return list(payload or [])


def fetch_stop_losses() -> list[dict]:
    """Return all Avanza stop-loss orders for the current session."""
    try:
        payload = api_get("/_api/trading/stoploss")
    except Exception:
        logger.warning("Failed to fetch Avanza stop-loss orders", exc_info=True)
        return []
    if isinstance(payload, dict):
        payload = payload.get("orders", [])
    return list(payload or [])


def fetch_positions_by_orderbook() -> dict[str, dict]:
    """Return current positions keyed by Avanza orderbook id."""
    payload = api_get("/_api/position-data/positions")
    positions: dict[str, dict] = {}
    for item in (payload.get("withOrderbook") or []):
        orderbook = ((item.get("instrument") or {}).get("orderbook") or {})
        orderbook_id = str(orderbook.get("id") or "")
        if not orderbook_id:
            continue
        positions[orderbook_id] = {
            "name": (item.get("instrument") or {}).get("name") or orderbook.get("name") or "",
            "instrument_type": ((item.get("instrument") or {}).get("type") or orderbook.get("type") or ""),
            "account_id": str((item.get("account") or {}).get("id") or ""),
            "volume": int(_value(item.get("volume")) or 0),
            "average_price": float(_value(item.get("averageAcquiredPrice")) or 0.0),
            "value_sek": float(_value(item.get("value")) or 0.0),
        }
    return positions


def _fetch_market_guide(orderbook_id: str, instrument_type: str) -> dict:
    inst_type = "warrant" if instrument_type.lower() == "warrant" else "certificate"
    return api_get(f"/_api/market-guide/{inst_type}/{orderbook_id}")


def _analysis_for_ticker(ticker: str, simulate_flash_window: bool = False) -> dict:
    if ticker == "XAG-USD":
        analysis = _load_json(SILVER_ANALYSIS_PATH)
        if simulate_flash_window:
            analysis = copy.deepcopy(analysis)
            analysis.setdefault("us_market_open", {})["phase"] = "pre_open"
        return analysis
    return {}


def _format_line(label: str, underlying: float, price: float) -> str:
    return f"  {label}: underlying {underlying:.4f} -> cert {price:.2f}"


def build_snapshots(
    hours_remaining: float,
    orderbook_filter: set[str] | None = None,
    *,
    simulate_flash_window: bool = False,
) -> list[dict]:
    summary = _load_json(SUMMARY_PATH)
    signals = summary.get("signals") or {}
    focus_probabilities = summary.get("focus_probabilities") or {}
    snapshots: list[dict] = []
    buy_orders = fetch_open_orders()
    stop_losses = fetch_stop_losses()
    positions_by_orderbook = fetch_positions_by_orderbook()
    grouped_orders: dict[str, list[dict]] = {}
    grouped_stop_losses: dict[str, list[dict]] = {}
    for order in buy_orders:
        orderbook_id = str(order.get("orderbookId") or "")
        if orderbook_filter and orderbook_id not in orderbook_filter:
            continue
        grouped_orders.setdefault(orderbook_id, []).append(order)
    for stop_order in stop_losses:
        orderbook_id = str(((stop_order.get("orderbook") or {}).get("id")) or "")
        if not orderbook_id:
            continue
        if orderbook_filter and orderbook_id not in orderbook_filter:
            continue
        grouped_stop_losses.setdefault(orderbook_id, []).append(stop_order)

    managed_orderbooks = set(grouped_orders) | set(grouped_stop_losses) | set(positions_by_orderbook)
    if orderbook_filter:
        managed_orderbooks &= set(orderbook_filter)

    for orderbook_id in sorted(managed_orderbooks):
        active_orders = list(grouped_orders.get(orderbook_id) or [])
        position = positions_by_orderbook.get(orderbook_id) or {}
        exemplar = active_orders[0] if active_orders else {}
        exemplar_book = exemplar.get("orderbook") or {}
        name = (
            exemplar_book.get("name")
            or position.get("name")
            or f"orderbook {orderbook_id}"
        )
        account_id = str((exemplar.get("account") or {}).get("accountId") or position.get("account_id") or "")
        inst_type = exemplar_book.get("instrumentType") or position.get("instrument_type") or "Warrant"
        market = _fetch_market_guide(orderbook_id, inst_type)
        underlying_name = ((market.get("underlying") or {}).get("name") or "").strip()
        ticker = map_underlying_name(underlying_name)
        if not ticker or ticker not in signals:
            if ticker:
                logger.debug(
                    "Skipping orderbook %s (%s): ticker %s not in agent_summary signals",
                    orderbook_id, name, ticker,
                )
            continue

        quote = market.get("quote") or {}
        underlying = (market.get("underlying") or {}).get("quote") or {}
        indicators = market.get("keyIndicators") or {}
        current_price = float(_value(quote.get("sell")) or _value(quote.get("last")) or 0.0)
        current_underlying = float(_value(underlying.get("last")) or 0.0)
        leverage = float(_value(indicators.get("leverage")) or 1.0)
        if current_price <= 0 or current_underlying <= 0 or leverage <= 0:
            continue

        ladder = build_intraday_ladder(
            signals[ticker],
            focus_probabilities.get(ticker),
            ticker=ticker,
            current_instrument_price=current_price,
            current_underlying_price=current_underlying,
            leverage=leverage,
            hours_remaining=hours_remaining,
            analysis=_analysis_for_ticker(
                ticker,
                simulate_flash_window=simulate_flash_window,
            ),
        )

        snapshots.append({
            "orderbook_id": orderbook_id,
            "name": name,
            "instrument_type": inst_type,
            "account_id": account_id,
            "ticker": ticker,
            "signal_entry": signals.get(ticker) or {},
            "focus_probability": focus_probabilities.get(ticker) or {},
            "market": market,
            "quote": quote,
            "current_bid": float(_value(quote.get("buy")) or 0.0),
            "current_ask": float(_value(quote.get("sell")) or 0.0),
            "current_last": float(_value(quote.get("last")) or 0.0),
            "current_underlying": current_underlying,
            "current_instrument_price": current_price,
            "leverage": leverage,
            "open_orders": active_orders,
            "stop_orders": list(grouped_stop_losses.get(orderbook_id) or []),
            "position_volume": int(position.get("volume") or 0),
            "position_average_price": float(position.get("average_price") or 0.0),
            "position_value_sek": float(position.get("value_sek") or 0.0),
            "ladder": ladder,
        })

    return snapshots


def build_reports(
    hours_remaining: float,
    orderbook_filter: set[str] | None = None,
    *,
    simulate_flash_window: bool = False,
) -> list[str]:
    reports: list[str] = []
    snapshots = build_snapshots(
        hours_remaining,
        orderbook_filter,
        simulate_flash_window=simulate_flash_window,
    )
    for snapshot in snapshots:
        active_orders = list(snapshot["open_orders"])
        if not any(str(order.get("side") or "").upper() == "BUY" for order in active_orders):
            continue
        name = snapshot["name"]
        orderbook_id = snapshot["orderbook_id"]
        ticker = snapshot["ticker"]
        quote = snapshot["quote"]
        ladder = snapshot["ladder"]
        active_orders.sort(key=lambda item: float(item.get("price") or 0.0), reverse=True)

        lines = [
            f"{name} ({orderbook_id})",
            f"  live: bid {_value(quote.get('buy'))} / ask {_value(quote.get('sell'))} / last {_value(quote.get('last'))}",
            f"  working bid: {ladder['working_price']:.2f}  | mean entry: {ladder['mean_price']:.2f}",
        ]
        if ladder["flash_price"] > 0:
            lines.append(
                f"  flash reserve: {ladder['flash_price']:.2f}  | extra drop {ladder['flash_crash_drop_pct']:.2f}%"
            )
        elif ticker == "XAG-USD":
            lines.append("  flash reserve: disabled outside the US-open window")
        lines.append(f"  exit target: {ladder['exit_price']:.2f}  | stretch: {ladder['stretch_exit_price']:.2f}")
        lines.append(_format_line("working", ladder["working_underlying"], ladder["working_price"]))
        if ladder["flash_price"] > 0:
            lines.append(_format_line("flash", ladder["flash_underlying"], ladder["flash_price"]))
        lines.append(_format_line("exit", ladder["exit_underlying"], ladder["exit_price"]))
        for existing in active_orders:
            lines.append(
                f"  open BUY: id {existing.get('orderId')}  {existing.get('volume')} @ {float(existing.get('price') or 0.0):.2f}"
            )
        reports.append("\n".join(lines))

    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Fin Snipe: report current metals bid/exit ladders for Avanza BUY orders.")

 succeeded in 737ms:
"""
Unified Market Monitoring Loop v10 (Layer 1 — Autonomous).
Runs every 60s, fully autonomous without Claude Code dependency.
Tracks: XAG/XAU (Binance FAPI), BTC/ETH (Binance SPOT), MSTR (Yahoo).
Core features: probability-focused Telegram, momentum-aware trailing stops,
auto-detect holdings, per-signal accuracy, crypto Fear & Greed, on-chain metrics.

v10: Silver fast-tick monitor merged from silver_monitor.py — 10-second price
checks with instant threshold alerts (-3% to -12.5%) and 3-minute velocity
flush detection.  Replaces the standalone silver_monitor.py process.

Features:
- Silver fast-tick: 10s price checks during 60s cycle sleep (threshold + velocity alerts)
- Tiered Claude invocation (Haiku/Sonnet, no Opus)
- Local LLM inference (Ministral-8B + Chronos for all tracked symbols)
- Monte Carlo VaR for leveraged warrants
- Trade guards (cooldowns, session limits, loss escalation)
- Drawdown circuit breaker (-15% emergency liquidation)
- Multi-level stop-loss (L1 warn / L2 alert / L3 emergency auto-sell)
- Short instrument tracking (BEAR SILVER X5)
- Time server (timeapi.io) for accurate CET
- Daily range analysis (historical percentiles + intraday assessment)
- Spike catcher (limit sell orders before US open)
- Invocation logging (tier/model/trigger tracking)
- Crypto data: Fear & Greed, CryptoCompare news, on-chain (MVRV/SOPR)
- MSTR-BTC NAV premium tracking

Run: .venv/Scripts/python.exe data/metals_loop.py
"""
import atexit
import contextlib
import datetime
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# 2026-04-09 Stage 1 log migration (docs/LOG_MIGRATION_AUDIT_20260409.md):
# The existing print-based `log()` / `_safe_print()` helpers were the only
# log output path in this process before 2026-04-09 afternoon. Fleet v1 added
# `logger` + 43 bare-except observability calls. This stage wires `log()`
# as a thin shim that delegates to `logger.info()` — all future 290+ call
# sites get timestamps + level prefix automatically, and the handful of
# logger.warning/error calls in the file format consistently.
#
# Library discipline (codex adversarial review finding HIGH, 2026-04-09):
# IMPORTING this module must NOT install any handlers or touch the root
# logger. Doing so (a) silently disables parent-process telemetry —
# pytest caplog, file/structured handlers in embedding processes, etc. —
# or (b) causes duplicate output if the parent has its own handler. The
# correct pattern: libraries never configure handlers; the application
# entrypoint configures them at startup.
#
# Handler installation happens only in `_install_stage1_logging()`, which
# is called from the `if __name__ == "__main__":` block at the bottom of
# this file (the real production entrypoint). When imported (pytest, other
# scripts, REPL), no handler is installed — callers that want to capture
# log output should use pytest's `caplog` with
# `caplog.at_level(logging.INFO, logger="metals_loop")`, or attach their
# own handler in `setup_logging()`.
#
# The `_safe_print` helper (defined below) stays in the file for its two
# remaining direct call sites (send_telegram TG error + silver fast-tick
# error path) — those are in the pure-print world, not the logger world.


class _LazyStdoutHandler(logging.StreamHandler):
    """StreamHandler with lazy stdout resolution + UnicodeEncodeError fallback.

    Two behaviors on top of stdlib StreamHandler:

    1. Re-resolves `sys.stdout` on every emit so pytest capsys (which
       swaps sys.stdout per-test) works correctly. In production under
       metals-loop.bat, sys.stdout is stable, so lazy resolution is a
       no-op cost.

    2. Catches UnicodeEncodeError from the underlying write and falls
       back to an ASCII-sanitized form via `_safe_print`. Needed for
       Windows non-UTF consoles when `sys.stdout.reconfigure()` wasn't
       callable at startup (older Python or non-tty stream) — replaces
       the old `_safe_print`-based safety net that was the only reason
       `_safe_print` existed in the first place.
    """

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        # 1. Re-resolve sys.stdout for pytest capsys compatibility.
        self.stream = sys.stdout
        # 2. Write directly (NOT via super().emit()) so we can catch
        #    UnicodeEncodeError ourselves. super().emit() has its own
        #    try/except that routes exceptions through self.handleError,
        #    which would bypass our ASCII sanitization fallback.
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Sanitize the formatted message to ASCII-replace and retry.
            # Same idea as the old `_safe_print` fallback, now integrated
            # into the logging path so every logger.* call benefits.
            try:
                safe = msg.encode("ascii", "replace").decode("ascii")
                self.stream.write(safe + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


logger = logging.getLogger("metals_loop")


def _install_stage1_logging() -> None:
    """Install the Stage 1 stdout handler on the `metals_loop` logger.

    Called from `if __name__ == "__main__":` at the bottom of this file.
    NEVER called at import time — see the library-discipline comment
    block above for the rationale.

    Idempotent: multiple calls (e.g. from test fixtures that simulate
    the production entrypoint) reuse the existing handler instead of
    stacking duplicates, via the `_metals_loop_stage1` marker attribute.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        # Older Python or non-tty stream — _safe_print fallback still catches.
        pass

    if not any(getattr(h, "_metals_loop_stage1", False) for h in logger.handlers):
        _handler = _LazyStdoutHandler()
        _handler._metals_loop_stage1 = True
        _handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # Codex review v6 finding MEDIUM (2026-04-09): set propagate=False
    # after install so records don't duplicate when an embedding process
    # has its own root handler. Our Stage 1 handler is now the sole
    # output owner for `metals_loop` and its children (including
    # `metals_loop.swing_trader`). Telemetry consumers that want these
    # records should attach a handler directly to `metals_loop`, not to
    # root.
    logger.propagate = False


def _has_ancestor_emitter(lg: logging.Logger, target_level: int) -> bool:
    """Walk the logger hierarchy to find a handler that would emit `target_level`.

    Used by log() and metals_swing_trader._log() to decide whether
    logger.info() will actually produce visible output, or whether to
    fall back to a direct stdout print.

    Codex review v6 finding MEDIUM (2026-04-09): the naive check
    `logger.hasHandlers() and isEnabledFor(INFO)` passes for an ancestor
    NullHandler or an ERROR-level StreamHandler — both scenarios drop
    INFO records silently. This walk handles:

    * NullHandler — skipped (it's a no-op absorbing handler)
    * Level-filtered handlers — only returns True if level permits target
    * propagate=False — stops walking when an ancestor blocks propagation

    Returns True only when we're confident an INFO record will actually
    land in at least one handler.
    """
    current = lg
    while current is not None:
        for h in current.handlers:
            if isinstance(h, logging.NullHandler):
                continue
            # Handler level 0 (NOTSET) means "accept whatever the logger
            # passes through"; any other level must be <= target to emit.
            if h.level == logging.NOTSET or h.level <= target_level:
                return True
        if not current.propagate:
            break
        current = current.parent
    return False

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# data/ also on sys.path so bare-module imports (metals_llm, metals_shared, ...)
# resolve here. Previously this was done lazily inside the LLM try-block below;
# hoisted to top-level so critical shared helpers (get_cet_time) can be imported
# as hard deps without a try-block. (2026-04-09 ARCH-12 dedup.)
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
os.chdir(BASE_DIR)

import requests
from metals_shared import get_cet_time
from playwright.sync_api import sync_playwright

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.loop_contract import MetalsCycleReport, ViolationTracker, verify_and_act, verify_metals_contract
from portfolio.market_timing import is_swedish_market_holiday

try:
    from portfolio.notification_text import (
        format_tier_footer,
        format_vote_summary,
        humanize_thesis_status,
        humanize_ticker,
    )
except ImportError:
    def format_tier_footer(source, tier, check_number, cet_str):
        return f"_{source} T{tier} · #{check_number} · {cet_str}_"

    def format_vote_summary(buy_count, sell_count):
        return f"{int(buy_count)}B/{int(sell_count)}S"

    def humanize_thesis_status(status):
        return str(status or "neutral").replace("_", " ").title()

    def humanize_ticker(ticker):
        return str(ticker or "").replace("-USD", "")

try:
    import msvcrt  # Windows file locking for single-instance guard
except ImportError:
    msvcrt = None

try:
    import fcntl  # Linux/WSL file locking for single-instance guard
except ImportError:
    fcntl = None

# --- Optional modules (graceful fallback) ---
try:

 succeeded in 757ms:
"""Fin Snipe Manager: stateful Avanza ladder/order manager.

The manager reconciles three pieces of state each cycle:
1. Live open orders
2. Live held positions
3. The current Fin Snipe ladder for each supported instrument

It then:
- maintains one or two resting BUY limits while flat
- switches to an automated SELL target when a fill creates a position
- cancels stale/mismatched orders so only the intended ladder remains

Dry-run is the default. Use ``--live`` explicitly to execute actions.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Any

from portfolio.avanza_control import (
    delete_order_no_page,
    delete_stop_loss_no_page,
    place_order_no_page,
    place_stop_loss_no_page,
)
from portfolio.avanza_session import verify_session
from portfolio.exit_optimizer import MarketSnapshot, Position, compute_exit_plan
from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_json,
    prune_jsonl,
)
from portfolio.fin_snipe import build_snapshots
from portfolio.metals_ladder import translate_underlying_target
from portfolio.process_lock import acquire_lock_file, release_lock_file
from portfolio.session_calendar import get_session_info

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "fin_snipe_state.json"
MANAGER_LOG_FILE = BASE_DIR / "data" / "fin_snipe_manager_log.jsonl"
PREDICTION_LOG_FILE = BASE_DIR / "data" / "fin_snipe_predictions.jsonl"
FILL_LOG_FILE = BASE_DIR / "data" / "fin_snipe_fills.jsonl"
LOCK_FILE = BASE_DIR / "data" / "fin_snipe_manager.singleton.lock"

FLASH_ENTRY_VOLUME_PCT = 0.30
DEFAULT_HOURS = 6.0
DEFAULT_INTERVAL_SECONDS = 60
LOG_PRUNE_BYTES = 25_000_000
LOG_MAX_ENTRIES = 20_000
EXIT_OPTIMIZER_N_PATHS = 2000
EXIT_OPTIMIZER_SEED = 42
HARD_STOP_CERT_PCT = 0.05
HARD_STOP_SELL_BUFFER_PCT = 0.01
HARD_STOP_VALID_DAYS = 8
MIN_STOP_DISTANCE_PCT = 1.0
FAST_RECHECK_SECONDS = 5
MAX_FAST_RECHECK_CYCLES = 6  # After 6 consecutive fast rechecks, fall back to normal interval
MAX_CANCEL_RETRIES = 3  # After N consecutive failed cancels, mark order as dead
DEAD_ORDER_EXPIRY_HOURS = 4  # Remove dead order reservations after this many hours
CRITICAL_ALERT_COOLDOWN_SECONDS = 1800  # 30 min between same-category alerts

logger = logging.getLogger("portfolio.fin_snipe_manager")

# Throttle state for critical Telegram alerts (category -> last_sent ISO timestamp)
_critical_alert_last: dict[str, str] = {}


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _notify_critical(category: str, message: str) -> None:
    """Send a throttled Telegram alert for critical fin_snipe_manager events.

    Categories: 'session_expired', 'naked_position', 'execution_failure',
    'phantom_orders'. Throttled to one per category per CRITICAL_ALERT_COOLDOWN_SECONDS.
    """
    now = dt.datetime.now(dt.UTC)
    last_raw = _critical_alert_last.get(category)
    if last_raw:
        try:
            last = dt.datetime.fromisoformat(last_raw)
            if (now - last).total_seconds() < CRITICAL_ALERT_COOLDOWN_SECONDS:
                logger.debug("Critical alert throttled: %s", category)
                return
        except (ValueError, TypeError):
            pass

    _critical_alert_last[category] = now.isoformat()
    try:
        from portfolio.message_store import send_or_store
        # BUG-124: Use load_json instead of raw open/json.load
        config = load_json(BASE_DIR / "config.json", default={})
        if config:
            send_or_store(message, config, category="error")
        else:
            logger.warning("Cannot send critical alert — config.json missing or corrupt")
    except Exception:
        logger.warning("Failed to send critical alert: %s", message, exc_info=True)


def _extract_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _new_session_id() -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"fin-snipe-{stamp}-pid{os.getpid()}"


def _host_name() -> str:
    return (
        os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or "unknown-host"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    entry = {
        "ts": _now_utc(),
        "source": "fin_snipe_manager",
        "event": event,
        **_json_safe(payload),
    }
    atomic_append_jsonl(path, entry)


def _log_fill_detected(
    snapshot: dict,
    instrument_state: dict,
    *,
    fill_side: str,
    old_vol: int,
    new_vol: int,
    session_id: str | None = None,
    fill_log_path: Path = FILL_LOG_FILE,
) -> None:
    """Append a fill-detection entry to the fills JSONL log."""
    try:
        entry: dict[str, Any] = {
            "event": "fill_detected",
            "fill_side": fill_side,
            "fill_volume": abs(new_vol - old_vol),
            "instrument_price": float(snapshot.get("current_bid") or snapshot.get("current_last") or 0),
            "underlying_price": float(snapshot.get("current_underlying") or 0),
            "position_avg_price": float(snapshot.get("position_average_price") or 0),
            "orderbook_id": snapshot.get("orderbook_id", ""),
            "name": snapshot.get("name", ""),
            "ticker": snapshot.get("ticker", ""),
            "leverage": float(snapshot.get("leverage") or 1.0),
            "prev_volume": old_vol,
            "new_volume": new_vol,
            "mode": instrument_state.get("mode"),
            "entry_underlying": instrument_state.get("entry_underlying"),
            "managed_order_ids": instrument_state.get("managed_order_ids", []),
        }
        if session_id is not None:
            entry["session_id"] = session_id
        # Compute realized P&L % for SELL fills where position goes to 0
        if fill_side == "SELL" and new_vol == 0:
            entry_underlying = instrument_state.get("entry_underlying")
            current_underlying = snapshot.get("current_underlying")
            leverage = float(snapshot.get("leverage") or 1.0)
            if entry_underlying and current_underlying:
                with contextlib.suppress(ZeroDivisionError, ValueError, TypeError):
                    entry["realized_pnl_pct"] = round(
                        ((float(current_underlying) / float(entry_underlying)) - 1) * 100 * leverage,
                        2,
                    )
        _append_log(fill_log_path, "fill_detected", entry)
    except Exception:
        logger.debug("Failed to log fill detection", exc_info=True)


def _maybe_prune_log(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size >= LOG_PRUNE_BYTES:
            prune_jsonl(path, max_entries=LOG_MAX_ENTRIES)
    except Exception:
        logger.debug("Fin Snipe log prune failed for %s", path, exc_info=True)


def _round_order_price(price: float) -> float:
    if price <= 0:
        return 0.0
    if price < 1:
        return round(price, 3)
    return round(price, 2)


def _price_abs_tolerance(price: float) -> float:
    if price < 1:
        return 0.002
    if price < 20:
        return 0.02
    return 0.25


def _price_matches(left: float, right: float) -> bool:
    ref = max(abs(left), abs(right), 1.0)
    tol = max(_price_abs_tolerance(ref), ref * 0.0025)
    return abs(left - right) <= tol


def _active_orders(snapshot: dict, side: str) -> list[dict]:
    wanted = side.upper()
    return [
        order for order in (snapshot.get("open_orders") or [])
        if str(order.get("side") or "").upper() == wanted
        and str(order.get("state") or "").upper() == "ACTIVE"
    ]


def _stop_order_id(order: dict) -> str:
    return str(order.get("id") or order.get("orderId") or "")


def _stop_order_status(order: dict) -> str:
    return str(order.get("status") or order.get("state") or "").upper()


def _stop_order_trigger(order: dict) -> float:
    trigger = order.get("trigger") or {}
    return float(_extract_value(trigger.get("value")) or 0.0)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Path . -Recurse -File | Select-Object -ExpandProperty FullName' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 585ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\BASELINE.md
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py
Q:\finance-analyzer\.worktrees\adv-metals-core\data\__pycache__\metals_loop.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_cross_assets.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_orderbook.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure_state.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\price_targets.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\silver_precompute.py
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\exit_optimizer.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\fin_fish.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\fin_snipe_manager.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\fin_snipe.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\iskbets.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\metals_cross_assets.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\metals_ladder.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\metals_orderbook.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\metals_precompute.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\microstructure_state.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\microstructure.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\orb_postmortem.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\orb_predictor.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\price_targets.cpython-312.pyc
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\__pycache__\silver_precompute.cpython-312.pyc

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import importlib, traceback
mods = [
'data.metals_loop',
'portfolio.exit_optimizer',
'portfolio.fin_fish',
'portfolio.fin_snipe',
'portfolio.fin_snipe_manager',
'portfolio.iskbets',
'portfolio.metals_cross_assets',
'portfolio.metals_ladder',
'portfolio.metals_orderbook',
'portfolio.metals_precompute',
'portfolio.microstructure',
'portfolio.microstructure_state',
'portfolio.orb_postmortem',
'portfolio.orb_predictor',
'portfolio.price_targets',
'portfolio.silver_precompute',
]
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('ERR', m, type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 1319ms:
ERR data.metals_loop ModuleNotFoundError No module named 'metals_shared'
ERR portfolio.exit_optimizer ModuleNotFoundError No module named 'portfolio.cost_model'
ERR portfolio.fin_fish ModuleNotFoundError No module named 'portfolio.file_utils'
ERR portfolio.fin_snipe ModuleNotFoundError No module named 'portfolio.avanza_session'
ERR portfolio.fin_snipe_manager ModuleNotFoundError No module named 'portfolio.avanza_control'
ERR portfolio.iskbets ModuleNotFoundError No module named 'portfolio.api_utils'
ERR portfolio.metals_cross_assets ModuleNotFoundError No module named 'portfolio.shared_state'
ERR portfolio.metals_ladder ModuleNotFoundError No module named 'scipy'
ERR portfolio.metals_orderbook ModuleNotFoundError No module named 'portfolio.api_utils'
ERR portfolio.metals_precompute ModuleNotFoundError No module named 'portfolio.file_utils'
OK portfolio.microstructure
ERR portfolio.microstructure_state ModuleNotFoundError No module named 'portfolio.file_utils'
ERR portfolio.orb_postmortem ModuleNotFoundError No module named 'portfolio.file_utils'
OK portfolio.orb_predictor
ERR portfolio.price_targets ModuleNotFoundError No module named 'scipy'
ERR portfolio.silver_precompute ModuleNotFoundError No module named 'portfolio.file_utils'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content BASELINE.md' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 572ms:
# Empty baseline for adversarial review diffs

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import pkgutil
mods=['pyflakes','ruff','flake8','pylint']
for m in mods:
    print(m, bool(pkgutil.find_loader(m)))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 576ms:
pyflakes False
ruff False
flake8 False
pylint False
<stdin>:4: DeprecationWarning: 'pkgutil.find_loader' is deprecated and slated for removal in Python 3.14; use importlib.util.find_spec() instead

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/metals_ladder.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 614ms:
"""Intraday ladder planning for metals instruments.

Pure planning utilities that bridge the underlying day-range model to actual
Avanza gold/silver certificates and warrants. The module is read-only and does
not place or cancel orders.
"""

from __future__ import annotations

from typing import Any

from portfolio.price_targets import compute_targets

SUPPORTED_UNDERLYINGS = {
    "gold": "XAU-USD",
    "silver": "XAG-USD",
}

US_OPEN_FLASH_PHASES = {"pre_open", "post_open_active"}
FLASH_RANGE_FRACTION = 0.35


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def map_underlying_name(name: str | None) -> str | None:
    """Map an Avanza underlying display name to the repo ticker."""
    text = (name or "").strip().lower()
    if text in SUPPORTED_UNDERLYINGS:
        return SUPPORTED_UNDERLYINGS[text]
    if "guld" in text or "gold" in text:
        return "XAU-USD"
    if "silver" in text:
        return "XAG-USD"
    return None


def translate_underlying_target(
    current_instrument_price: float,
    current_underlying_price: float,
    target_underlying_price: float,
    leverage: float,
    direction_sign: int = 1,
) -> float:
    """Approximate instrument price from an underlying target."""
    if current_instrument_price <= 0 or current_underlying_price <= 0:
        return 0.0
    underlying_return = (target_underlying_price / current_underlying_price) - 1.0
    instrument_return = direction_sign * leverage * underlying_return
    return round(max(0.01, current_instrument_price * (1.0 + instrument_return)), 4)


def flash_crash_drop_pct(analysis: dict | None) -> float:
    """Heuristic downside allowance for the US-open flash-crash window.

    The silver monitor records that the first US-open hour often has a much
    wider range than the rest of the day. For entry ladders, we only widen bids
    during the active/pre-open window; otherwise the reserve bid is disabled.
    """
    market_open = (analysis or {}).get("us_market_open") or {}
    phase = str(market_open.get("phase") or "").strip().lower()
    if phase not in US_OPEN_FLASH_PHASES:
        return 0.0

    stats = market_open.get("historical_stats") or {}
    mean_drop_pct = abs(_safe_float(stats.get("post_open_mean_pct")))
    avg_range_pct = abs(_safe_float(stats.get("post_open_avg_range_pct")))
    return max(mean_drop_pct, avg_range_pct * FLASH_RANGE_FRACTION)


def build_intraday_ladder(
    signal_entry: dict,
    focus_probabilities: dict | None,
    *,
    ticker: str,
    current_instrument_price: float,
    current_underlying_price: float,
    leverage: float,
    hours_remaining: float,
    analysis: dict | None = None,
    direction_sign: int = 1,
) -> dict:
    """Build a working bid / flash reserve / exit ladder for one instrument."""
    p_up = _safe_float((focus_probabilities or {}).get("3h", {}).get("probability"), 0.5)
    extra = signal_entry.get("extra") if isinstance(signal_entry, dict) else None
    squeeze_on = bool(((extra or {}).get("volatility_sig_indicators") or {}).get("bb_squeeze_on"))

    buy_targets = compute_targets(
        ticker,
        side="buy",
        price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
        atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
        p_up=p_up,
        hours_remaining=hours_remaining,
        indicators=signal_entry,
        extra=extra,
        regime=str(signal_entry.get("regime") or ""),
        bb_squeeze=squeeze_on,
    )
    sell_targets = compute_targets(
        ticker,
        side="sell",
        price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
        atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
        p_up=p_up,
        hours_remaining=hours_remaining,
        indicators=signal_entry,
        extra=extra,
        regime=str(signal_entry.get("regime") or ""),
        bb_squeeze=squeeze_on,
    )

    working_underlying = min(
        _safe_float((buy_targets.get("recommended") or {}).get("price"), current_underlying_price),
        _safe_float((buy_targets.get("extremes") or {}).get("p25"), current_underlying_price),
    )
    mean_underlying = _safe_float((buy_targets.get("recommended") or {}).get("price"), working_underlying)
    flash_drop_pct = flash_crash_drop_pct(analysis)
    flash_underlying = 0.0
    if flash_drop_pct > 0:
        flash_underlying = min(
            working_underlying,
            current_underlying_price * (1.0 - flash_drop_pct / 100.0),
        )

    exit_underlying = _safe_float((sell_targets.get("recommended") or {}).get("price"), current_underlying_price)
    stretch_exit_underlying = _safe_float((sell_targets.get("extremes") or {}).get("p75"), exit_underlying)

    result = {
        "ticker": ticker,
        "current_underlying": round(current_underlying_price, 4),
        "current_instrument": round(current_instrument_price, 4),
        "hours_remaining": round(hours_remaining, 2),
        "working_underlying": round(working_underlying, 4),
        "mean_underlying": round(mean_underlying, 4),
        "exit_underlying": round(exit_underlying, 4),
        "stretch_exit_underlying": round(stretch_exit_underlying, 4),
        "flash_crash_drop_pct": round(flash_drop_pct, 4),
        "working_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            working_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "mean_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            mean_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "exit_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            exit_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "stretch_exit_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            stretch_exit_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "buy_targets": buy_targets,
        "sell_targets": sell_targets,
    }
    result["flash_price"] = (
        translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            flash_underlying,
            leverage,
            direction_sign=direction_sign,
        )
        if flash_underlying > 0
        else 0.0
    )
    result["flash_underlying"] = round(flash_underlying, 4) if flash_underlying > 0 else 0.0
    return result

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Select-String -Path portfolio/*.py,data/*.py -Pattern "direction_sign|direction|BEAR|BULL|SHORT|LONG" | ForEach-Object { "{0}:{1}:{2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 630ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:271:def _first_hit_times(paths: np.ndarray, target: float, direction: str = "above") -> np.ndarray:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:277:direction: "above" (sell target) or "below" (stop level).
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:283:if direction == "above":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:575:hit_times = _first_hit_times(paths, target, direction="above")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:7:- Chronos/model drift for directional bias
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:8:- RSI-based direction selection (BULL vs BEAR fishing)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:11:Supports both BULL (buy dips) and BEAR (buy spikes) fishing.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:90:# Warrant catalog — BULL and BEAR instruments
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:92:# --- Silver BULL ---
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:93:"BULL_SILVER_X5_AVA_3": {
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:97:"direction": "LONG",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:101:"name": "BULL SILVER X5 AVA 3",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:106:# --- Silver BEAR ---
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:107:"BEAR_SILVER_X5_AVA_12": {
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:111:"direction": "SHORT",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:115:"name": "BEAR SILVER X5 AVA 12",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:120:# --- Gold BULL ---
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:121:"BULL_GULD_X5_AVA": {
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:125:"direction": "LONG",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:129:"name": "BULL GULD X5 AVA",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:134:# --- Gold BEAR (no viable AVA X5 — gold rally killed them all) ---
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:135:"BEAR_GULD_X5_VON4": {
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:139:"direction": "SHORT",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:143:"name": "BEAR GULD X5 VON4",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:148:"BEAR_GULD_X2_AVA": {
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:152:"direction": "SHORT",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:156:"name": "BEAR GULD X2 AVA",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:163:# Preferred instruments per (underlying, direction) — snipe manager picks these first
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:165:("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_3",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:166:("XAG-USD", "SHORT"): "BEAR_SILVER_X5_AVA_12",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:167:("XAU-USD", "LONG"): "BULL_GULD_X5_AVA",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:168:("XAU-USD", "SHORT"): "BEAR_GULD_X5_VON4",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:331:direction: str,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:337:direction : str
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:338:``"LONG"`` biases drift downward (we want dips), ``"SHORT"`` biases
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:357:# For LONG fishing we want P(dip) — use p_up as-is (lower p_up = more likely to dip)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:358:# For SHORT fishing we want P(spike) — invert: use 1-p_up
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:359:if direction == "SHORT":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:374:# Direction selection
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:377:def choose_fishing_directions(signal: dict) -> list[dict]:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:378:"""Decide whether to fish BULL, BEAR, or both.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:381:and news severity to set conviction scores for each direction.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:384:[{"direction": "LONG", "conviction": 0.8}, ...]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:389:directions: list[dict] = []
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:393:bull_conv = 0.8 if rsi < 30 else 0.65
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:394:bear_conv = 0.0
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:396:bear_conv = 0.8 if rsi > 70 else 0.65
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:397:bull_conv = 0.0
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:399:bull_conv = 0.4
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:400:bear_conv = 0.4
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:405:bear_conv = max(bear_conv, 0.3) + 0.15
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:406:bull_conv = max(0.0, bull_conv - 0.1)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:408:bull_conv = max(bull_conv, 0.3) + 0.15
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:409:bear_conv = max(0.0, bear_conv - 0.1)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:415:bull_conv += 0.15
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:417:bear_conv += 0.15
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:424:bull_conv += 0.15   # extreme fear → buy dips
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:426:bear_conv += 0.15   # extreme greed → sell peaks
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:431:bear_conv += 0.10
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:432:bull_conv = max(0.0, bull_conv - 0.10)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:437:bear_conv += 0.10
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:439:bull_conv += 0.10
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:442:bull_conv = min(1.0, max(0.0, bull_conv))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:443:bear_conv = min(1.0, max(0.0, bear_conv))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:445:if bull_conv > 0.05:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:446:directions.append({"direction": "LONG", "conviction": round(bull_conv, 2)})
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:447:if bear_conv > 0.05:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:448:directions.append({"direction": "SHORT", "conviction": round(bear_conv, 2)})
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:450:return directions
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:454:# Core fishing level computation — BULL (dip) and BEAR (spike)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:457:def compute_fishing_levels_bull(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:464:"""Compute candidate BULL fishing (dip-buy) levels with fill probabilities.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:468:vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="LONG")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:500:def compute_fishing_levels_bear(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:507:"""Compute candidate BEAR fishing (spike-buy) levels with fill probabilities.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:510:we would buy a BEAR cert at that elevated price.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:512:vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="SHORT")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:551:"""Score BULL fishing candidates (below spot) with fill probability and EV."""
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:586:"""Score BEAR fishing candidates (above spot) with fill probability and EV."""
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:640:Legacy wrapper — delegates to ``compute_fishing_levels_bull``.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:642:return compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:651:direction: str,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:654:"""Select matching warrants for (ticker, direction), preferring the preferred
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:659:direction : str
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:660:``"LONG"`` for BULL, ``"SHORT"`` for BEAR.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:662:pref_key = (ticker, direction)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:668:if w["underlying"] == ticker and w["direction"] == direction
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:702:direction: str = "LONG",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:708:direction : str
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:709:``"LONG"`` for BULL certs, ``"SHORT"`` for BEAR certs.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:711:matching = _select_warrants(ticker, direction, spot)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:730:if direction == "LONG" and spot <= barrier:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:732:if direction == "SHORT" and spot >= barrier:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:733:# BEAR MINIs get knocked out if underlying goes above barrier
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:747:if direction == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:768:if direction == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:830:"direction": direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:856:def _build_instrument_info(warrant_results: list[dict], direction: str) -> dict:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:871:"direction": direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:887:Returns a list of plans (one per direction: LONG and/or SHORT) with
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:895:"direction": "LONG",      # or "SHORT"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:908:directions = choose_fishing_directions(signal)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:911:for d in directions:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:912:direction = d["direction"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:916:if direction == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:917:levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:919:levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:923:ticker, spot, levels, budget_sek, fx_rate, direction=direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:934:instrument_info = _build_instrument_info(warrant_results, direction)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:939:"direction": direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:960:BULL X5: fish $67.50 (-2.5%) fill 11% EV 31
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:961:BEAR X5: fish $71.00 (+2.6%) fill 8% EV 22
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:973:direction = plan["direction"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:974:label = "BULL" if direction == "LONG" else "BEAR"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:980:if direction == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1037:# Direction analysis
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1038:directions = choose_fishing_directions(signal)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1041:f"{'BULL' if d['direction'] == 'LONG' else 'BEAR'} ({d['conviction']:.0%})"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1042:for d in directions
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1044:lines.append(f"  Direction: {dir_labels}")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1054:direction = plan["direction"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1055:label = "BULL" if direction == "LONG" else "BEAR"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1071:if direction == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1130:parser.add_argument("--direction", choices=["bull", "bear", "auto"], default="auto",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1131:help="Force direction: bull, bear, or auto (default: auto).")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1270:# Override direction if requested
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1271:if args.direction != "auto":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1272:forced_dir = "LONG" if args.direction == "bull" else "SHORT"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1273:if forced_dir == "LONG":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1274:levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1276:levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1279:ticker, spot, levels, args.budget, fx_rate, direction=forced_dir,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1285:"ticker": ticker, "spot": spot, "direction": forced_dir,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1311:"direction": p["direction"],
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1362:"bull silver", "bear silver")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1363:gold_keywords = ("guld", "gold", "xau", "bull guld", "bear guld")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1381:"is_short": any(k in name for k in ("bear", "mini s")),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1402:"is_short": pos.get("direction", "").lower() == "short",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1424:# Determine direction
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1425:if args.direction != "auto":
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1426:direction = "LONG" if args.direction == "bull" else "SHORT"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1428:direction = "SHORT" if detected_position.get("is_short") else "LONG"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1431:direction = "LONG" if pf["bull_score"] > pf["bear_score"] else "SHORT"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1433:direction = "SHORT"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1438:entry_conviction = pf["bull_score"] if direction == "LONG" else pf["bear_score"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1445:direction=direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1085:# Prune stale tracking data: remove entries for orders no longer on Avanza
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1099:# longer than DEAD_ORDER_EXPIRY_HOURS, remove it regardless of API state.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:158:bull_conds, _bear_conds, _extra = _evaluate_conditions(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:162:if len(bull_conds) < min_bigbet:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:175:conditions = list(bull_conds)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:277:f"APPROVE if setup is clean. SKIP only for clear red flags (all long TFs opposing, "
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:462:short_ticker = ticker.replace("-USD", "")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:464:f"\U0001f7e1 *ISKBETS BUY {short_ticker}* @ ${price:,.2f}",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:515:lines.append(f"_Bought? Reply:_ `bought {short_ticker} PRICE AMOUNT`")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:516:lines.append(f"_Example:_ `bought {short_ticker} {price:.0f} 100000`")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:556:short_ticker = ticker.replace("-USD", "")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:560:f"{emoji} *{short_ticker} hit target #1* @ ${price:,.2f}",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:573:f"{emoji} *SELL {short_ticker}* @ ${price:,.2f}",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:737:# Normalize short names: "BTC" → "BTC-USD", "ETH" → "ETH-USD", etc.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:738:SHORT_TO_FULL = {k.replace("-USD", ""): k for k in TICKER_SOURCES if "-USD" in k}
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:739:if ticker in SHORT_TO_FULL:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:740:ticker = SHORT_TO_FULL[ticker]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_cross_assets.py:16:for longer-horizon callers; the metals_cross_asset signal switched to
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_cross_assets.py:32:# Intraday TTL is shorter — 60m bars refresh at the start of each hour,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_cross_assets.py:33:# and we want to re-query shortly after the bar closes to pick up the new row.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_cross_assets.py:40:2026-04-14: no longer pinned to yfinance. The router dispatches
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:49:direction_sign: int = 1,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:55:instrument_return = direction_sign * leverage * underlying_return
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:87:direction_sign: int = 1,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:150:direction_sign=direction_sign,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:157:direction_sign=direction_sign,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:164:direction_sign=direction_sign,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:171:direction_sign=direction_sign,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:182:direction_sign=direction_sign,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:418:nc_long = _int(latest.get("noncomm_positions_long_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:419:nc_short = _int(latest.get("noncomm_positions_short_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:420:comm_long = _int(latest.get("comm_positions_long_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:421:comm_short = _int(latest.get("comm_positions_short_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:423:nc_net = (nc_long - nc_short) if nc_long is not None and nc_short is not None else None
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:424:comm_net = (comm_long - comm_short) if comm_long is not None and comm_short is not None else None
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:426:prev_nc_long = _int(prev.get("noncomm_positions_long_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:427:prev_nc_short = _int(prev.get("noncomm_positions_short_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:429:(prev_nc_long - prev_nc_short)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:430:if prev_nc_long is not None and prev_nc_short is not None
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:437:"noncomm_long": nc_long,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:438:"noncomm_short": nc_short,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:440:"comm_long": comm_long,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:441:"comm_short": comm_short,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:464:mm_long = _int(d.get("m_money_positions_long_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:465:mm_short = _int(d.get("m_money_positions_short_all"))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:466:if mm_long is not None and mm_short is not None:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:467:result["managed_money_long"] = mm_long
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:468:result["managed_money_short"] = mm_short
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:469:result["managed_money_net"] = mm_long - mm_short
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:549:# Determine direction from recent trend
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:550:real_yield_direction = None
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:557:real_yield_direction = "rising"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:559:real_yield_direction = "falling"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:561:real_yield_direction = "stable"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:570:"real_yield_direction": real_yield_direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:635:# Compute direction from last 3 weeks
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:643:result["nc_net_direction"] = (
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:652:result["mm_net_direction"] = (
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:728:context["prophecy"] = _extract_prophecy("silver_bull_2026")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:836:price_key: short key for output, e.g. 'xag' or 'xau'
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1070:"note": "Most bullish major bank",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1121:"managed_funds_net_long": 8500,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1132:"bull_market_avg": "60-70",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1158:"note": "Structural bull target",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1197:"primary": "Real yields (10Y - CPI). Negative real yields = bullish.",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:1206:"Crowded long positioning in COT if specs pile in",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure.py:1:"""Microstructure feature computations for short-term metals prediction.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure.py:179:trades in the same direction.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure.py:213:# Same-direction large gap = likely trade-through
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:37:morning_direction: str
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:71:# Errors (positive = actual exceeded prediction, negative = actual fell short)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:111:morning_direction=prediction.morning_direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:189:# Direction analysis
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:190:up_days = [r for r in history if r.morning_direction == "up"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:191:down_days = [r for r in history if r.morning_direction == "down"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:230:lines.append("- Up mornings significantly more predictable. Consider direction filter.")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:232:lines.append("- Down mornings significantly more predictable. Consider direction filter.")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:259:print(f"  Morning: HIGH ${prediction.morning_high:.2f} LOW ${prediction.morning_low:.2f} DIR={prediction.morning_direction}")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:7:- Apply filters: morning direction, range size, volume
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:10:- Toby Crabel, "Day Trading with Short Term Price Patterns and Opening Range Breakout" (1990)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:48:direction: str              # "up" if close > open, else "down"
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:76:morning_direction: str
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:214:direction="up" if close_price > open_price else "down",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:290:use_direction_filter: bool = True,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:297:Applies optional filters for morning direction and range size.
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:302:use_direction_filter: Filter historical days by same morning direction
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:312:# Filter by morning direction
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:313:if use_direction_filter and len(filtered) >= min_sample * 2:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:314:direction_filtered = [d for d in filtered if d.morning.direction == morning.direction]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:315:if len(direction_filtered) >= min_sample:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:316:filtered = direction_filtered
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:317:filters.append(f"direction={morning.direction}")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:355:morning_direction=morning.direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:379:The warrant (MINI Long) has:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:428:# Direction breakdown
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:429:up_mornings = [d for d in day_results if d.morning.direction == "up"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:430:down_mornings = [d for d in day_results if d.morning.direction == "down"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:476:f"Direction: {prediction.morning_direction.upper()} | Range: {prediction.morning_range_pct:.2f}%",
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\price_targets.py:485:# Directional probability
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:20:- Short instrument tracking (BEAR SILVER X5)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:496:# Mirror of the exit-side velocity alert but in the opposite direction and
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:500:# a fresh candidate matches the ticker and direction.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:523:"name": "BULL GULD X8 N", "ob_id": "856394", "api_type": "certificate",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:736:SHORT_INSTRUMENTS = {
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:737:"bear_silver_x5": {
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:738:"name": "BEAR SILVER X5 AVA 12", "ob_id": "2286417", "api_type": "certificate",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:792:short_prices = {}         # latest prices for short instruments
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1142:# Mirror of the exit-side silver fast-tick but direction-reversed and
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1146:# long. When a velocity breakout is detected we write a momentum candidate
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1166:direction: str,
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1177:{"XAG-USD": {"direction": "LONG", "velocity_pct": 0.92, ...}}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1181:"direction": direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1252:Mirror of ``_silver_fast_tick`` but direction-reversed. Runs every
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1298:direction="LONG",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1315:f"_LONG candidate written; swing trader may relax entry gates_"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1930:# zero exit protection (the bull_silver_x5 incident earlier today is
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1939:"856394":  {"key": "gold", "name": "BULL GULD X8 N",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1942:"2286417": {"key": "bear_silver_x5", "name": "BEAR SILVER X5 AVA 12",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1945:"1650161": {"key": "bull_silver_x5", "name": "BULL SILVER X5 AVA 4",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2114:direction = pos.get("direction", "?")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2117:nm = "BULL" if direction == "LONG" else "BEAR"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2134:f"{nm} {volume}u position no longer on Avanza\n"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2251:# Stale migration marker — swing no longer tracks it.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2371:changes.append(f"SOLD {key}: no longer on Avanza")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2372:log(f"Holdings: {key} no longer held on Avanza — deactivating")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2612:"gold", "silver", "precious", "metal", "bullion", "xau", "xag",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2639:("silver AND (price OR market OR ounce OR bullion OR futures)", "XAG-USD"),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2640:("gold AND (price OR market OR ounce OR bullion OR futures)", "XAU-USD"),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2886:"focus_1d_dir": f1d.get("direction", "?"),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2950:than FISH_MAX_SIGNAL_AGE_SEC (default 120s). Today's BULL buy fired
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2976:direction = decision.get("direction", "LONG")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2977:ob_id = decision.get("instrument_ob", "1650161" if direction == "LONG" else "2286417")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2979:leverage = 5.0  # BULL/BEAR SILVER X5
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3047:nm = "BULL X5" if direction == "LONG" else "BEAR X5"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3054:_fish_engine.confirm_entry(direction, ask, volume, price)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3098:# Without this, Avanza rejects with short.sell.not.allowed because
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3117:nm = "BULL" if pos.get("direction") == "LONG" else "BEAR"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3134:# re-arm). Telegram keeps the short form with the operator-
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3191:gold_leads_silver = {"direction": "NEUTRAL", "confidence": 0.0}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3198:"direction": "LONG",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3206:"direction": "SHORT",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3341:# --- Direction probability from signals ---
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3359:entry["chronos_1h"] = {"direction": "flat", "pct_move": 0, "confidence": 0}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3360:entry["chronos_3h"] = {"direction": "flat", "pct_move": 0, "confidence": 0}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3362:entry["llm_consensus"] = {"direction": "flat", "confidence": 0}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3373:"direction": ticker_llm[ckey].get("direction", "flat"),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3385:"direction": cons.get("direction", "flat"),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3391:# --- Combined direction probability ---
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3406:if c.get("direction") in ("up", "down"):
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3408:if c["direction"] == "up":
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3448:if bias in ("bullish", "bearish"):
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3450:if bias == "bullish":
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3520:short_t = ticker.split("-")[0]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3522:if short_t in k and v.get("total", 0) >= 5:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3594:short = ticker.split("-")[0] if "-" in ticker else ticker
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3595:ticker_devs.append((dev, f"{short} {arrow}{prob:.0f}%"))
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3621:short = ticker.split("-")[0] if "-" in ticker else ticker
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3626:lines.append(f"`{short:<4} {price_str}  ↑{prob_up:.0f}%  ↓{prob_down:.0f}%`")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3633:if c.get("direction") in ("up", "down"):
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3634:arrow = "↑" if c["direction"] == "up" else "↓"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3711:short_key = key[:8]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3713:f"`  {short_key} {pos['units']}u b:{bid:.2f} "
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3758:# Without this, Avanza rejects with short.sell.not.allowed because
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3848:elif "short.sell.not.allowed" in message_code:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3883:"emergency_sell: holdings re-check failed after short-sell-not-allowed key=%s",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3889:log(f"  {key}: short-sell-not-allowed but position still held — keeping active")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3896:log(f"  {key}: short-sell-not-allowed and not held — deactivating")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3897:send_telegram(f"*L3* {pos['name']}: no longer held, deactivating")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3920:# separate log(f"...: {e}") preserved as a short operator-friendly
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3922:# the short form. exc_info goes to metals_loop_out.txt for
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3968:the first failure. Today's BULL_SILVER_X5_AVA_4 position was stuck
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4033:Avanza rejects sells with ``short.sell.not.allowed`` when the sum of
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4035:treats the overlap as an attempted short-sale. This helper makes sells
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4072:``short.sell.not.allowed`` on the next sell attempt.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4139:# pre-cancel snapshot. Stops that are no longer present are the ones
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4228:f"SELL aborted to avoid short.sell.not.allowed"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4341:# the operator. Keep `e` in the Telegram text (short form) and
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4554:order_id_short = order.get("id", "?")[:8]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4568:log(f"  Order {order_id_short} expired ({age_s:.0f}s old)")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4592:log(f"  Order {order_id_short} deduplicated (same {action} recently filled)")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4600:log(f"  Order {order_id_short}: cannot fetch live price, skipping")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4615:log(f"  Order {order_id_short}: slippage {slippage:.1f}% > {TRADE_QUEUE_MAX_SLIPPAGE}% "
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4637:# otherwise Avanza rejects with short.sell.not.allowed (sl_vol + sell_vol
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4645:log(f"  Order {order_id_short}: SELL aborted — could not clear stops")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4669:log(f"  Order {order_id_short}: place_order raised: {place_exc}")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4685:log(f"  Order {order_id_short} FILLED: {action} {order['volume']}u @ {exec_price}")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4720:log(f"  Order {order_id_short} FAILED: {error_msg}")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4818:# Telegram alert still fires with the short form; exc_info
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5021:# cancel cycle retries — logging the short exception is
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5257:rejects with ``short.sell.not.allowed``.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5499:# If the order is no longer in the open list, treat as terminal.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5510:f"{spike_order_id} is no longer open — proceeding with restore"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5723:# the place_stop_loss would reject with short.sell.not.allowed.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5757:Returns the volume as an int (0 if the position is no longer held)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5796:size and trigger short.sell.not.allowed on subsequent operations.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5981:"short_instruments": {},
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5987:# Short instrument prices
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5988:for sk, si in SHORT_INSTRUMENTS.items():
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5989:sp = short_prices.get(sk, {})
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5990:ctx["short_instruments"][sk] = {
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5996:"note": "5x short silver certificate, available for hedging",
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6236:direction = consensus.get("direction", "flat")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6242:if direction in ("up", "down") and confidence >= 0.7 and has_accuracy:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6243:action = "BUY" if direction == "up" else "SELL"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6295:directions = []  # (direction, weight)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6320:directions.append(("up", weight))
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6322:directions.append(("down", weight))
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6330:directions.append(("up" if consensus_dir == "BUY" else "down", consensus_conf))
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6334:directions.append((chr_dir, abs(data.get("chronos_3h_pct", 0)) * 10))
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6336:if not directions:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6337:return {"action": "HOLD", "direction": "flat", "confidence": 0.0, "horizon": "3h"}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6339:up_w = sum(w for d, w in directions if d == "up")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6340:down_w = sum(w for d, w in directions if d == "down")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6343:return {"action": "HOLD", "direction": "flat", "confidence": 0.0, "horizon": "3h"}
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6346:direction, confidence = "up", round(up_w / total_w, 2)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6348:direction, confidence = "down", round(down_w / total_w, 2)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6352:action = "BUY" if direction == "up" else "SELL"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6355:"action": action, "direction": direction,
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6362:"""Assess whether the strategic thesis (silver bull 2026) is intact."""
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6442:short_t = humanize_ticker(ticker)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6453:sig_parts.append(f"{short_t} {action} ({conf_part}raw {raw_action}, {votes} votes)")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6455:sig_parts.append(f"{short_t} {action} ({votes} votes)")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6469:direction = data["chronos_3h"]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6470:move = f"{pct:+.1f}%" if direction in ("up", "down") else f"{abs(pct):.1f}%"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6471:ticker_parts.append(f"Chronos {direction} {move} over 3h")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6605:llm_entry[ckey] = data[ckey].get("direction", "?")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6732:"Strategic thesis: Silver bull 2026, target ATH. Bias HOLD. Only sell on structure break.\n\n"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6804:global short_prices, daily_range_stats, _METALS_LOOP_START_TS
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6826:log(f"Short instruments: {', '.join(v['name'] for v in SHORT_INSTRUMENTS.values())}")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7002:# Initialize strategy orchestrator (GoldDigger + Elongir as plugins)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7070:short = t.split("-")[0]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7071:prob_summary += f"\n  {short}: ${r['price']:.2f} ↑{r['prob_up_pct']:.0f}%"
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7175:short_t = t.split("-")[0] if "-" in t else t
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7176:price_tags.append(f"{short_t}=${_format_price(p, t)}")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7228:# Fetch short instrument prices (every 4th check)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7230:for sk, si in SHORT_INSTRUMENTS.items():
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7234:short_prices[sk] = sp
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7236:logger.debug("main_loop: short instrument fetch_price failed key=%s ob_id=%s", sk, si.get("ob_id"), exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7409:log(f"Holdings check: {lost} position(s) no longer held on Avanza")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7506:direction = _spos.get("direction", "LONG")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7508:if direction == "SHORT":
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7521:short_t = t.split("-")[0]
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7526:und_tag += f" {short_t}=${p:.2f}(↑{prob_up:.0f}% v={vel:+.3f}%)"

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/price_targets.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 600ms:
"""Optimal price targets with fill probabilities for buy and sell decisions.

Combines Monte Carlo running max/min simulation, first-passage-time analytics,
and structural price levels (BB, Fibonacci, pivots, Keltner, Donchian, VWAP,
smart money swing levels) to produce ranked price targets with fill probability
and expected value.

Supports regime-aware confidence adjustment, BB squeeze warnings, and
Chronos forecast drift blending for improved accuracy.

Usage:
    from portfolio.price_targets import compute_targets
    result = compute_targets("XAG-USD", side="sell", price_usd=85.28,
                             atr_pct=0.59, p_up=0.45, hours_remaining=3.0)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.stats import norm

from portfolio.monte_carlo import MIN_VOLATILITY, drift_from_probability, volatility_from_atr

logger = logging.getLogger("portfolio.price_targets")

# 24/7 asset suffixes (crypto, metals)
_24H_SUFFIXES = ("-USD",)


def _is_24h(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in _24H_SUFFIXES)


def _year_fraction(hours: float, is_24h: bool = True) -> float:
    """Convert hours to year fraction for GBM."""
    if is_24h:
        return hours / (252.0 * 24.0)
    return hours / (252.0 * 6.5)


def _is_valid_level(val: object) -> bool:
    """Return True if *val* is a finite positive number usable as a price level."""
    if not isinstance(val, (int, float)):
        return False
    if math.isnan(val) or math.isinf(val):
        return False
    return val > 0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fill_probability(price: float, target: float, vol_annual: float,
                     drift_annual: float, hours_remaining: float,
                     is_24h: bool = True) -> float:
    """First-passage-time probability for GBM reaching *target* within window.

    For SELL (target >= price): probability running max hits target.
    For BUY  (target <= price): probability running min hits target.
    """
    if hours_remaining <= 0 or price <= 0 or vol_annual <= 0:
        return 1.0 if _on_easy_side(price, target, "sell") else 0.0

    # Target on the "easy" side (already filled)
    if target <= price:
        # For a sell order at or below current price -> instant fill
        return 1.0
    # target > price: compute first-passage for running max
    T = _year_fraction(hours_remaining, is_24h)
    sigma = max(vol_annual, MIN_VOLATILITY)
    mu_adj = drift_annual - 0.5 * sigma ** 2
    x = math.log(target / price)

    sqrt_T = math.sqrt(T)
    if sqrt_T * sigma < 1e-12:
        return 0.0

    d1 = (-x + mu_adj * T) / (sigma * sqrt_T)
    d2 = (-x - mu_adj * T) / (sigma * sqrt_T)

    exponent = 2.0 * mu_adj * x / (sigma ** 2)
    exponent = max(-500.0, min(500.0, exponent))  # clamp for numerical safety

    p = norm.cdf(d1) + math.exp(exponent) * norm.cdf(d2)
    return float(max(0.0, min(1.0, p)))


def _on_easy_side(price: float, target: float, side: str) -> bool:
    if side == "sell":
        return target <= price
    return target >= price


def fill_probability_buy(price: float, target: float, vol_annual: float,
                         drift_annual: float, hours_remaining: float,
                         is_24h: bool = True) -> float:
    """Fill probability for a BUY limit order (target <= price)."""
    if target >= price:
        return 1.0
    # Flip: P(min <= target) = P(max of -process >= -target)
    # Symmetry: negate drift, swap price/target relationship
    return fill_probability(price, price ** 2 / target if target > 0 else price,
                            vol_annual, -drift_annual, hours_remaining, is_24h)


def running_extremes(price: float, vol_annual: float, drift_annual: float,
                     hours_remaining: float, side: str = "sell",
                     n_paths: int = 10_000, n_steps: int = 50,
                     is_24h: bool = True) -> dict:
    """MC simulation of running max (sell) or running min (buy)."""
    if hours_remaining <= 0 or price <= 0:
        v = float(price) if price > 0 else 0.0
        return {k: v for k in ("p10", "p25", "p50", "p75", "p90")}

    T = _year_fraction(hours_remaining, is_24h)
    sigma = max(vol_annual, MIN_VOLATILITY)
    dt = T / n_steps
    drift_dt = (drift_annual - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)

    rng = np.random.default_rng(42)
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z_anti = -Z
    Z_all = np.concatenate([Z, Z_anti], axis=0)  # (n_paths, n_steps)

    log_increments = drift_dt + vol_dt * Z_all
    log_cum = np.cumsum(log_increments, axis=1)
    # Prepend zero column (start at spot)
    log_cum = np.concatenate([np.zeros((log_cum.shape[0], 1)), log_cum], axis=1)
    price_paths = price * np.exp(log_cum)

    if side == "sell":
        extremes = np.max(price_paths, axis=1)
    else:
        extremes = np.min(price_paths, axis=1)

    pcts = np.percentile(extremes, [10, 25, 50, 75, 90])
    return {
        "p10": round(float(pcts[0]), 4),
        "p25": round(float(pcts[1]), 4),
        "p50": round(float(pcts[2]), 4),
        "p75": round(float(pcts[3]), 4),
        "p90": round(float(pcts[4]), 4),
    }


def structural_levels(price: float, indicators: dict | None,
                      extra: dict | None = None) -> dict:
    """Extract all available price levels from indicators and enhanced signal data.

    Sources:
    - BB mid/upper/lower from main indicators
    - Fibonacci retracement levels, pivots, Camarilla, golden pocket, swings
    - Keltner channels, Donchian channels (volatility signal)
    - VWAP (volume flow signal)
    - Smart money swing highs/lows
    """
    if not indicators and not extra:
        return {}

    levels: dict[str, float] = {}

    # BB levels (from main indicators)
    if indicators:
        for key in ("bb_mid", "bb_upper", "bb_lower"):
            val = indicators.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

    if not extra:
        return levels

    # -- Fibonacci levels --------------------------------------------------
    fib_ind = extra.get("fibonacci_indicators", {})
    if fib_ind:
        fib_levels_dict = fib_ind.get("fib_levels", {})
        for ratio_key, val in fib_levels_dict.items():
            if _is_valid_level(val):
                # Convert "0.236" -> "fib_236", "0.5" -> "fib_5" etc.
                clean_key = str(ratio_key).replace("0.", "")
                levels[f"fib_{clean_key}"] = float(val)

        # Pivot levels
        _pivot_keys = [
            ("pivot", "pivot_pp"),
            ("r1", "pivot_r1"),
            ("r2", "pivot_r2"),
            ("s1", "pivot_s1"),
            ("s2", "pivot_s2"),
        ]
        for src_key, label in _pivot_keys:
            val = fib_ind.get(src_key)
            if _is_valid_level(val):
                levels[label] = float(val)

        # Camarilla pivots
        for key in ("cam_r3", "cam_s3", "cam_r4", "cam_s4"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

        # Golden pocket
        for key in ("gp_upper", "gp_lower"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

        # Fibonacci swing points
        for key in ("swing_high", "swing_low"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[f"fib_{key}"] = float(val)

    # -- Volatility levels (Keltner, Donchian) -----------------------------
    vol_ind = extra.get("volatility_sig_indicators", {})
    if vol_ind:
        for key in ("keltner_upper", "keltner_lower",
                     "donchian_upper", "donchian_lower"):
            val = vol_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

    # -- VWAP from volume flow ---------------------------------------------
    vf_ind = extra.get("volume_flow_indicators", {})
    if vf_ind:
        val = vf_ind.get("vwap")
        if _is_valid_level(val):
            levels["vwap"] = float(val)

    # -- Smart money swing levels ------------------------------------------
    sm_ind = extra.get("smart_money_indicators", {})
    if sm_ind:
        for key in ("last_swing_high", "last_swing_low"):
            val = sm_ind.get(key)
            if _is_valid_level(val):
                levels[f"smc_{key.replace('last_', '')}"] = float(val)

    return levels


def expected_value(fill_prob: float, gain_if_filled: float,
                   gain_at_fallback: float) -> float:
    """Probability-weighted expected value."""
    return fill_prob * gain_if_filled + (1.0 - fill_prob) * gain_at_fallback


def _apply_regime_adjustment(targets: list[dict], regime: str, side: str,
                             price_usd: float, bb_mid: float | None) -> None:
    """Mutate *targets* in-place with regime-aware confidence adjustments.

    - ranging/range-bound: penalize far targets, boost targets near bb_mid
    - trending-up + sell: boost fill_prob slightly for targets above price
    - trending-down + buy: boost fill_prob slightly for targets below price
    - high-vol: no fill_prob change (widening is handled by ATR already)
    """
    regime_lower = regime.lower().replace("-", "").replace("_", "")
    if not regime_lower:
        return

    for t in targets:
        tp = t["price"]
        fp = t["fill_prob"]

        if regime_lower in ("ranging", "rangebound"):
            # Penalize targets far from price (>1% away)
            pct_away = abs(tp - price_usd) / price_usd if price_usd > 0 else 0
            if pct_away > 0.01:
                t["fill_prob"] = round(fp * 0.85, 4)
            # Boost targets near bb_mid (mean-reversion)
            if bb_mid and bb_mid > 0:
                pct_from_mid = abs(tp - bb_mid) / bb_mid
                if pct_from_mid < 0.005:  # within 0.5% of bb_mid
                    t["fill_prob"] = round(min(fp * 1.15, 1.0), 4)

        elif regime_lower in ("trendingup",):
            if side == "sell" and tp > price_usd:
                t["fill_prob"] = round(min(fp * 1.10, 1.0), 4)
            elif side == "buy" and tp < price_usd:
                t["fill_prob"] = round(fp * 0.90, 4)

        elif regime_lower in ("trendingdown",):
            if side == "buy" and tp < price_usd:
                t["fill_prob"] = round(min(fp * 1.10, 1.0), 4)
            elif side == "sell" and tp > price_usd:
                t["fill_prob"] = round(fp * 0.90, 4)


def compute_targets(ticker: str, side: str, price_usd: float,
                    atr_pct: float, p_up: float, hours_remaining: float,
                    indicators: dict | None = None, extra: dict | None = None,
                    warrant_leverage: float = 1.0,
                    position_units: int = 1, fx_rate: float = 1.0,
                    is_24h: bool = True, n_paths: int = 10_000,
                    regime: str = "", bb_squeeze: bool = False,
                    chronos_drift: float | None = None) -> dict:
    """Main entry point: compute ranked price targets with fill probabilities.

    Parameters
    ----------
    extra : dict | None
        Enhanced signal indicator dicts (fibonacci_indicators, etc.)
        passed through to ``structural_levels``.
    regime : str
        Market regime string (e.g. "trending-up", "ranging").
        Used for regime-aware confidence adjustment.
    bb_squeeze : bool
        If True, Bollinger Band squeeze is active -- reduce confidence
        on all targets by 0.7x and flag ``squeeze_warning``.
    chronos_drift : float | None
        Annualised drift from Chronos 24h forecast.  When provided,
        blended 30/70 with the signal-based drift.
    """
    result: dict = {
        "ticker": ticker,
        "side": side,
        "price_usd": price_usd,
        "hours_remaining": hours_remaining,
        "extremes": {},
        "targets": [],
        "recommended": None,
    }

    if hours_remaining <= 0 or price_usd <= 0 or atr_pct <= 0:
        return result

    vol = volatility_from_atr(atr_pct)
    if side == "buy":
        drift = drift_from_probability(1.0 - p_up, vol)
    else:
        drift = drift_from_probability(p_up, vol)

    # Blend Chronos drift when available
    if chronos_drift is not None:
        drift = 0.7 * drift + 0.3 * chronos_drift

    # Structural levels (enriched with extra signal indicators)
    levels = structural_levels(price_usd, indicators, extra=extra)

    # Running extremes
    extremes = running_extremes(price_usd, vol, drift, hours_remaining,
                                side=side, n_paths=n_paths, is_24h=is_24h)
    result["extremes"] = extremes

    # Build candidate targets
    candidates: list[tuple[float, str]] = []

    # MC quantiles
    for pkey in ("p25", "p50", "p75"):
        val = extremes.get(pkey)
        if val is not None:
            candidates.append((val, f"mc_{pkey}"))

    # Structural levels
    for label, val in levels.items():
        if side == "sell" and val > price_usd or side == "buy" and val < price_usd:
            candidates.append((val, label))

    # Fixed offsets
    offsets = [0.005, 0.01, 0.02]
    for off in offsets:
        pct_label = f"{off*100:.1f}%"
        if side == "sell":
            candidates.append((price_usd * (1 + off), f"+{pct_label}"))
        else:
            candidates.append((price_usd * (1 - off), f"-{pct_label}"))

    # Deduplicate (within 0.01% of each other)
    candidates.sort(key=lambda c: c[0])
    deduped: list[tuple[float, str]] = []
    for price_c, label_c in candidates:
        if deduped and abs(price_c - deduped[-1][0]) / max(price_usd, 1e-9) < 0.0001:
            continue
        deduped.append((price_c, label_c))

    # Compute fill probability and EV for each candidate
    min_fill = 0.05
    targets: list[dict] = []
    for target_price, label in deduped:
        if side == "sell":
            if target_price < price_usd:
                continue
            fp = fill_probability(price_usd, target_price, vol, drift,
                                  hours_remaining, is_24h)
            gain_if_filled = (target_price - price_usd) * position_units * warrant_leverage * fx_rate
        else:
            if target_price > price_usd:
                continue
            fp = fill_probability_buy(price_usd, target_price, vol, drift,
                                      hours_remaining, is_24h)
            gain_if_filled = (price_usd - target_price) * position_units * warrant_leverage * fx_rate

        if fp < min_fill:
            continue

        ev = expected_value(fp, gain_if_filled, 0.0)
        targets.append({
            "price": round(target_price, 4),
            "fill_prob": round(fp, 4),
            "ev_sek": round(ev, 2),
            "label": label,
        })

    # Regime-aware adjustments
    bb_mid_val = levels.get("bb_mid")
    if regime:
        _apply_regime_adjustment(targets, regime, side, price_usd, bb_mid_val)
        # Re-compute EV after fill_prob adjustment
        for t in targets:
            if side == "sell":
                gain = (t["price"] - price_usd) * position_units * warrant_leverage * fx_rate
            else:
                gain = (price_usd - t["price"]) * position_units * warrant_leverage * fx_rate
            t["ev_sek"] = round(expected_value(t["fill_prob"], gain, 0.0), 2)

    # BB squeeze warning: reduce confidence on all targets
    if bb_squeeze:
        result["squeeze_warning"] = True
        for t in targets:
            t["fill_prob"] = round(t["fill_prob"] * 0.7, 4)
            # Re-compute EV after squeeze adjustment
            if side == "sell":
                gain = (t["price"] - price_usd) * position_units * warrant_leverage * fx_rate
            else:
                gain = (price_usd - t["price"]) * position_units * warrant_leverage * fx_rate
            t["ev_sek"] = round(expected_value(t["fill_prob"], gain, 0.0), 2)

    # Filter out targets that dropped below min_fill after adjustments
    targets = [t for t in targets if t["fill_prob"] >= min_fill]

    # Sort by EV descending
    targets.sort(key=lambda t: t["ev_sek"], reverse=True)
    result["targets"] = targets
    result["recommended"] = targets[0] if targets else None
    return result


def compute_all_targets(agent_summary: dict, portfolio_states: dict,
                        config: dict) -> dict | None:
    """Batch wrapper for the reporting pipeline."""
    from portfolio.focus_analysis import hours_to_us_close

    default_hours = config.get("default_hours", 6)
    n_paths = config.get("n_paths", 10_000)
    signals = agent_summary.get("signals", {})
    focus_probs = agent_summary.get("focus_probabilities", {})

    # Collect tickers to process: held positions (SELL) + BUY consensus (BUY)
    tasks: list[tuple[str, str]] = []  # (ticker, side)

    held_tickers: set[str] = set()
    for _label, pf in portfolio_states.items():
        for tk, pos in pf.get("holdings", {}).items():
            if pos.get("shares", 0) > 0:
                held_tickers.add(tk)
                tasks.append((tk, "sell"))

    for tk, sig_data in signals.items():
        if sig_data.get("action") == "BUY" and tk not in held_tickers:
            tasks.append((tk, "buy"))

    if not tasks:
        return None

    # Forecast signals for Chronos drift
    forecast_signals = agent_summary.get("forecast_signals", {})

    results: dict = {}
    for ticker, side in tasks:
        if ticker in results:
            continue
        sig = signals.get(ticker, {})
        price = sig.get("price_usd", 0)
        if price <= 0:
            continue

        extra = sig.get("extra", {})
        atr_pct = extra.get("atr_pct") or sig.get("atr_pct", 2.0)
        is_24h_asset = _is_24h(ticker)

        # Directional probability
        tp = focus_probs.get(ticker, {}).get("1d", {})
        p_up = tp.get("probability", 0.5) if tp else 0.5

        # Hours remaining
        if is_24h_asset:
            hours = float(default_hours)
        else:
            hours = hours_to_us_close()
            if hours <= 0:
                hours = float(default_hours)

        # Indicators for structural levels (BB from main signal data)
        indicators = {k: sig.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                      if sig.get(k) is not None}
        if not indicators:
            indicators = {k: extra.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                          if extra.get(k) is not None}

        # Chronos drift from forecast signal
        chronos_drift_val = None
        fc_data = forecast_signals.get(ticker, {})
        chronos_pct = fc_data.get("chronos_24h_pct", 0)
        chronos_conf = fc_data.get("chronos_24h_conf", 0)
        if isinstance(chronos_conf, (int, float)) and chronos_conf > 0.3 \
                and isinstance(chronos_pct, (int, float)) and chronos_pct != 0:
            chronos_drift_val = (chronos_pct / 100.0) * math.sqrt(252)

        # BB squeeze detection
        vol_ind = extra.get("volatility_sig_indicators", {})
        squeeze = bool(vol_ind.get("bb_squeeze_on", False)) if vol_ind else False

        # Regime
        regime = sig.get("regime", "") or ""

        try:
            res = compute_targets(
                ticker, side, price, atr_pct, p_up, hours,
                indicators=indicators or None,
                extra=extra or None,
                is_24h=is_24h_asset,
                n_paths=n_paths,
                regime=regime,
                bb_squeeze=squeeze,
                chronos_drift=chronos_drift_val,
            )
            if res.get("targets"):
                results[ticker] = res
        except Exception:
            logger.warning("price_targets failed for %s", ticker, exc_info=True)

    return results if results else None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/exit_optimizer.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 595ms:
"""Quant exit engine — probabilistic exit optimization for intraday positions.

Three-layer architecture:
1. **Opportunity layer**: Monte Carlo path simulation for remaining-session
   price distribution (max/min/terminal).
2. **Execution layer**: Fill probability and time-to-hit estimation from
   simulated paths.
3. **Decision layer**: EV ranking of candidate exits, net of costs, with
   risk overrides (knock-out proximity, session end, volatility shock).

Designed for Avanza MINI futures (gold/silver warrants) but works for any
instrument with price, volatility, and session data.

Usage:
    from portfolio.exit_optimizer import compute_exit_plan, Position, MarketSnapshot
    plan = compute_exit_plan(position, market, session_end, cost_model)
    print(plan.recommended)  # Best exit by EV

Reference: docs/deep research/deep-research-report.md
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from portfolio.cost_model import CostModel, get_cost_model

logger = logging.getLogger("portfolio.exit_optimizer")

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshot:
    """Current market state for the position's instrument.

    Prices are in the underlying's native currency (USD for metals/crypto).
    """
    asof_ts: datetime
    price: float              # Current underlying price (USD)
    bid: float | None = None
    ask: float | None = None
    volatility_annual: float | None = None   # Annualized vol (decimal)
    atr_pct: float | None = None             # ATR% for vol estimation
    usdsek: float = 10.85                       # FX rate
    drift: float = 0.0                          # Annualized drift (0 = neutral)


@dataclass(frozen=True)
class Position:
    """A held position to evaluate for exit.

    For warrants: prices in SEK, with underlying in USD.
    For stocks/crypto: prices in USD.
    """
    symbol: str                          # Underlying ticker (e.g., "XAG-USD")
    qty: float                           # Units held
    entry_price_sek: float               # What we paid per unit (SEK)
    entry_underlying_usd: float          # Underlying price at entry (USD)
    entry_ts: datetime
    instrument_type: str = "warrant"     # "warrant", "stock", "crypto"
    leverage: float = 1.0                # Effective leverage at entry
    financing_level: float | None = None  # MINI future financing level (USD)
    trailing_peak_usd: float | None = None  # Highest underlying since entry


@dataclass(frozen=True)
class CandidateExit:
    """A ranked exit candidate with probabilistic assessment.

    Attributes:
        price_usd: Target exit price in underlying USD.
        action: Exit method — "limit", "market", "hold_to_close".
        fill_prob: P(price reaches target before session end), 0.0-1.0.
        expected_fill_time_min: E[time to hit target | hit], in minutes.
        pnl_sek: Net P&L if filled at target price (after costs).
        ev_sek: Expected value = fill_prob × pnl + (1-fill_prob) × fallback.
        pnl_pct: P&L as percentage of position value.
        risk_flags: List of active risk warnings.
        quantile: Which quantile of session-max this candidate represents.
    """
    price_usd: float
    action: str
    fill_prob: float
    expected_fill_time_min: float
    pnl_sek: float
    ev_sek: float
    pnl_pct: float
    risk_flags: tuple[str, ...] = ()
    quantile: float | None = None


@dataclass
class ExitPlan:
    """Complete exit plan with ranked candidates.

    Attributes:
        symbol: Underlying ticker.
        asof_ts: When this plan was computed.
        remaining_minutes: Minutes until session close.
        candidates: All evaluated exit candidates, sorted by EV descending.
        recommended: The top candidate (highest EV, respecting risk overrides).
        market_exit: Immediate market exit candidate (always available).
        session_max_distribution: Quantiles of the remaining-session max price.
        session_min_distribution: Quantiles of the remaining-session min price.
        stop_hit_prob: P(price drops to stop level before session end).
        provenance: Audit trail (model version, parameters, data sources).
    """
    symbol: str
    asof_ts: datetime
    remaining_minutes: float
    candidates: list[CandidateExit]
    recommended: CandidateExit
    market_exit: CandidateExit
    session_max_distribution: dict[str, float] = field(default_factory=dict)
    session_min_distribution: dict[str, float] = field(default_factory=dict)
    stop_hit_prob: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for logging/Telegram."""
        rec = self.recommended
        mkt = self.market_exit
        return (
            f"Exit plan: recommended {rec.action} @ ${rec.price_usd:.2f} "
            f"(EV {rec.ev_sek:+,.0f} SEK, fill {rec.fill_prob:.0%}, "
            f"{rec.expected_fill_time_min:.0f}min) | "
            f"market exit {mkt.pnl_sek:+,.0f} SEK | "
            f"{self.remaining_minutes:.0f}min left"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON (agent_summary integration)."""
        return {
            "symbol": self.symbol,
            "remaining_min": round(self.remaining_minutes),
            "recommended": {
                "price": round(self.recommended.price_usd, 2),
                "action": self.recommended.action,
                "fill_prob": round(self.recommended.fill_prob, 3),
                "ev_sek": round(self.recommended.ev_sek),
                "pnl_pct": round(self.recommended.pnl_pct, 2),
                "time_min": round(self.recommended.expected_fill_time_min),
                "risk_flags": list(self.recommended.risk_flags),
            },
            "market_exit_sek": round(self.market_exit.pnl_sek),
            "stop_hit_prob": round(self.stop_hit_prob, 3),
            "session_max": self.session_max_distribution,
            "session_min": self.session_min_distribution,
            "n_candidates": len(self.candidates),
        }


# ---------------------------------------------------------------------------
# Intraday Monte Carlo path engine
# ---------------------------------------------------------------------------

# Trading minutes per day by instrument type (for annualization)
_TRADING_MINUTES = {
    "warrant": 820,    # 08:15-21:55 CET = ~13.67h
    "stock": 390,      # 6.5h
    "crypto": 1440,    # 24h
}
_TRADING_DAYS_PER_YEAR = 252
_MIN_VOLATILITY = 0.05  # 5% annualized floor


def _estimate_volatility(market: MarketSnapshot) -> float:
    """Get annualized volatility from market snapshot."""
    if market.volatility_annual and market.volatility_annual > _MIN_VOLATILITY:
        return market.volatility_annual
    if market.atr_pct and market.atr_pct > 0:
        # Convert ATR% (14-period) to annualized vol
        atr_frac = market.atr_pct / 100.0
        return max(atr_frac * math.sqrt(252.0 / 14), _MIN_VOLATILITY)
    return 0.20  # Default 20% annual vol


def simulate_intraday_paths(
    price: float,
    volatility: float,
    drift: float,
    remaining_minutes: int,
    instrument_type: str = "warrant",
    n_paths: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths at 1-minute resolution.

    Uses antithetic variates for variance reduction (~50% lower variance).

    Args:
        price: Current underlying price (USD).
        volatility: Annualized volatility (decimal, e.g., 0.25 = 25%).
        drift: Annualized drift (decimal). 0 = neutral.
        remaining_minutes: Minutes until session close.
        instrument_type: For annualization ("warrant", "stock", "crypto").
        n_paths: Number of paths to simulate. Even number recommended.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, remaining_minutes + 1) where column 0 is
        the current price and each subsequent column is 1 minute later.
    """
    rng = np.random.default_rng(seed)

    n_steps = max(1, int(remaining_minutes))
    min_per_year = _TRADING_MINUTES.get(instrument_type, 390) * _TRADING_DAYS_PER_YEAR
    dt = 1.0 / min_per_year  # 1 minute in annualized trading time

    vol = max(volatility, _MIN_VOLATILITY)
    drift_per_step = (drift - 0.5 * vol ** 2) * dt
    vol_per_step = vol * math.sqrt(dt)

    # Antithetic variates: generate half, mirror the other half
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z_all = np.vstack([Z, -Z])

    # If odd n_paths, add one extra
    if n_paths % 2 == 1:
        extra = rng.standard_normal((1, n_steps))
        Z_all = np.vstack([Z_all, extra])

    # Log-return increments → cumulative → price paths
    log_inc = drift_per_step + vol_per_step * Z_all  # (n_paths, n_steps)
    log_cum = np.cumsum(log_inc, axis=1)

    # Prepend zero column (current price)
    zeros = np.zeros((Z_all.shape[0], 1))
    log_paths = np.hstack([zeros, log_cum])

    paths = price * np.exp(log_paths)
    return paths


def _path_statistics(paths: np.ndarray) -> dict:
    """Extract key statistics from simulated paths.

    Returns:
        Dict with session_max, session_min, terminal arrays and quantile dicts.
    """
    session_max = np.max(paths[:, 1:], axis=1)  # Exclude t=0
    session_min = np.min(paths[:, 1:], axis=1)
    terminal = paths[:, -1]

    quantiles = [5, 10, 20, 35, 50, 65, 80, 90, 95]
    max_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_max, quantiles))}
    min_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_min, quantiles))}

    return {
        "session_max": session_max,
        "session_min": session_min,
        "terminal": terminal,
        "max_quantiles": max_q,
        "min_quantiles": min_q,
    }


def _first_hit_times(paths: np.ndarray, target: float, direction: str = "above") -> np.ndarray:
    """Compute first passage time for each path to reach target.

    Args:
        paths: Price paths, shape (n_paths, n_steps+1).
        target: Price level to hit.
        direction: "above" (sell target) or "below" (stop level).

    Returns:
        Array of shape (n_paths,). Values are minute indices (1-based).
        -1 means the path never hit the target.
    """
    if direction == "above":
        hits = paths[:, 1:] >= target
    else:
        hits = paths[:, 1:] <= target

    # argmax on axis=1 returns first True index (0-based in the sliced array)
    first_idx = np.argmax(hits, axis=1)

    # Distinguish never-hit: if first_idx=0 but that cell isn't True → never hit
    never_hit = ~np.any(hits, axis=1)
    result = first_idx + 1  # Convert to 1-based minute index
    result[never_hit] = -1

    return result


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def _compute_pnl_sek(
    position: Position,
    exit_price_usd: float,
    market: MarketSnapshot,
    costs: CostModel,
) -> float:
    """Compute net P&L in SEK for exiting at given underlying price.

    For warrants (MINI futures):
        warrant_value = (underlying - financing_level) × usdsek
        pnl = (exit_value - entry_value) × qty - costs

    For stocks/crypto:
        pnl = (exit_price - entry_price) × qty × usdsek - costs
    """
    fx = market.usdsek

    if position.instrument_type == "warrant" and position.financing_level is not None:
        # MINI future: warrant price = (underlying - financing_level) × fx
        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
        exit_warrant_sek = max(exit_warrant_sek, 0)  # Can't go below 0 (knock-out)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    elif position.instrument_type == "warrant":
        # Leveraged product without explicit financing level
        pct_move = (exit_price_usd - position.entry_underlying_usd) / position.entry_underlying_usd
        warrant_move = pct_move * position.leverage
        exit_warrant_sek = position.entry_price_sek * (1 + warrant_move)
        exit_warrant_sek = max(exit_warrant_sek, 0)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    else:
        # Direct position (stock/crypto)
        exit_value = position.qty * exit_price_usd * fx
        entry_value = position.qty * position.entry_underlying_usd * fx

    cost = costs.total_cost_sek(exit_value)
    return exit_value - entry_value - cost


def _pnl_pct(pnl_sek: float, position: Position) -> float:
    """P&L as percentage of initial investment."""
    entry_value = position.entry_price_sek * position.qty
    if entry_value <= 0:
        return 0.0
    return pnl_sek / entry_value * 100.0


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

def _compute_risk_flags(
    target_price: float | None,
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_max: np.ndarray | None = None,
    session_min: np.ndarray | None = None,
) -> list[str]:
    """Generate risk warnings for a candidate exit."""
    flags = []

    # 1. Session end proximity
    if remaining_minutes < 30:
        flags.append("SESSION_END_IMMINENT")
    elif remaining_minutes < 60:
        flags.append("SESSION_END_NEAR")

    # 2. Knock-out proximity (MINI futures)
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            flags.append("KNOCKOUT_DANGER")
        elif distance_pct < 8:
            flags.append("KNOCKOUT_WARNING")

    # 3. Target far from current price (low fill probability expected)
    if target_price and market.price > 0:
        target_distance_pct = abs(target_price - market.price) / market.price * 100
        if target_distance_pct > 5:
            flags.append("TARGET_DISTANT")

    # 4. Underlying session mismatch (warrant still trading but underlying closed)
    # This would be detected by session_calendar, passed as a flag

    # 5. Position aging
    if position.entry_ts:
        hold_hours = (market.asof_ts - position.entry_ts).total_seconds() / 3600
        if hold_hours > 5:
            flags.append("HOLD_TIME_EXTENDED")

    # 6. Stop-loss proximity from MC paths
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03  # 3% above financing
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.10:
            flags.append(f"KNOCKOUT_PROB_{p_knockout:.0%}")

    return flags


# ---------------------------------------------------------------------------
# Risk overrides
# ---------------------------------------------------------------------------

def _apply_risk_overrides(
    candidates: list[CandidateExit],
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_min: np.ndarray | None = None,
) -> CandidateExit:
    """Apply hard risk overrides and select recommended exit.

    Risk overrides can force a market exit even if EV says hold:
    - Knock-out danger (< 3% from financing level)
    - Session end imminent (< 5 min remaining)
    - Stop probability too high (> 25% chance of knock-out)
    """
    if not candidates:
        raise ValueError("No candidates to evaluate")

    # Find the market exit candidate
    market_exits = [c for c in candidates if c.action == "market"]
    market_exit = market_exits[0] if market_exits else candidates[-1]

    # Override 1: Knock-out danger → force market exit
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
                           "forcing market exit", distance_pct)
            return market_exit

    # Override 2: Session about to end → force market exit
    if remaining_minutes < 5:
        logger.info("RISK OVERRIDE: Session ending in %.0f min, forcing market exit",
                     remaining_minutes)
        return market_exit

    # Override 3: High knock-out probability → prefer market exit
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.25:
            logger.warning("RISK OVERRIDE: %.0f%% knock-out probability, "
                           "forcing market exit", p_knockout * 100)
            return market_exit

    # No override triggered — return highest-EV candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_N_PATHS = 5000
DEFAULT_QUANTILES = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]


def compute_exit_plan(
    position: Position,
    market: MarketSnapshot,
    session_end: datetime,
    costs: CostModel | None = None,
    *,
    n_paths: int = DEFAULT_N_PATHS,
    quantiles: list[float] | None = None,
    stop_price_usd: float | None = None,
    seed: int | None = None,
) -> ExitPlan:
    """Compute a full exit plan for a held position.

    This is the main function. It:
    1. Simulates remaining-session price paths (Monte Carlo GBM)
    2. Extracts session-max/min distributions
    3. Generates candidate exits at quantile levels of session max
    4. Computes fill probability, time-to-hit, and EV for each
    5. Adds market exit and hold-to-close baselines
    6. Ranks by EV and applies risk overrides

    Args:
        position: The held position to evaluate.
        market: Current market snapshot.
        session_end: UTC datetime of session close.
        costs: Cost model. If None, auto-selects by instrument type.
        n_paths: Number of Monte Carlo paths.
        quantiles: Quantile levels for candidate generation.
        stop_price_usd: Explicit stop level (for stop-hit probability).
        seed: Random seed for reproducibility.

    Returns:
        ExitPlan with ranked candidates and recommendation.
    """
    if costs is None:
        costs = get_cost_model(position.instrument_type)

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    now = market.asof_ts
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if session_end.tzinfo is None:
        session_end = session_end.replace(tzinfo=UTC)

    remaining_min = max(0, (session_end - now).total_seconds() / 60)

    # ---- Edge case: session over or almost over ----
    if remaining_min < 1:
        mkt_pnl = _compute_pnl_sek(position, market.bid or market.price, market, costs)
        mkt_candidate = CandidateExit(
            price_usd=market.bid or market.price,
            action="market",
            fill_prob=1.0,
            expected_fill_time_min=0,
            pnl_sek=mkt_pnl,
            ev_sek=mkt_pnl,
            pnl_pct=_pnl_pct(mkt_pnl, position),
            risk_flags=("SESSION_ENDED",),
        )
        return ExitPlan(
            symbol=position.symbol,
            asof_ts=now,
            remaining_minutes=0,
            candidates=[mkt_candidate],
            recommended=mkt_candidate,
            market_exit=mkt_candidate,
            provenance={"reason": "session_ended"},
        )

    # ---- 1. Estimate volatility and simulate paths ----
    vol = _estimate_volatility(market)
    drift = market.drift

    paths = simulate_intraday_paths(
        price=market.price,
        volatility=vol,
        drift=drift,
        remaining_minutes=int(remaining_min),
        instrument_type=position.instrument_type,
        n_paths=n_paths,
        seed=seed,
    )

    # ---- 2. Extract path statistics ----
    stats = _path_statistics(paths)
    session_max = stats["session_max"]
    session_min = stats["session_min"]
    terminal = stats["terminal"]

    # ---- 3. Generate candidate exits at session-max quantiles ----
    target_prices = np.quantile(session_max, quantiles)
    candidates: list[CandidateExit] = []

    # Fallback P&L: what we get if we hold to close (median terminal price)
    median_terminal = float(np.median(terminal))
    fallback_pnl = _compute_pnl_sek(position, median_terminal, market, costs)

    for q, target in zip(quantiles, target_prices):
        target = float(target)

        # Skip if target is below current price (can't profit from selling below)
        if target <= market.price * 0.999:
            continue

        # Fill probability: fraction of paths where max >= target
        fill_prob = float(np.mean(session_max >= target))

        # Time to hit
        hit_times = _first_hit_times(paths, target, direction="above")
        hitting_times = hit_times[hit_times > 0]
        expected_time = float(np.mean(hitting_times)) if len(hitting_times) > 0 else remaining_min

        # P&L if filled
        pnl = _compute_pnl_sek(position, target, market, costs)

        # Expected value: fill_prob × conditional_pnl + (1-fill_prob) × fallback
        ev = fill_prob * pnl + (1 - fill_prob) * fallback_pnl

        flags = _compute_risk_flags(target, position, market, remaining_min,
                                     session_max, session_min)

        candidates.append(CandidateExit(
            price_usd=round(target, 4),
            action="limit",
            fill_prob=round(fill_prob, 4),
            expected_fill_time_min=round(expected_time, 1),
            pnl_sek=round(pnl, 2),
            ev_sek=round(ev, 2),
            pnl_pct=round(_pnl_pct(pnl, position), 2),
            risk_flags=tuple(flags),
            quantile=q,
        ))

    # ---- 4. Market exit candidate (immediate fill, certain) ----
    bid = market.bid or market.price
    mkt_pnl = _compute_pnl_sek(position, bid, market, costs)
    market_candidate = CandidateExit(
        price_usd=round(bid, 4),
        action="market",
        fill_prob=1.0,
        expected_fill_time_min=0,
        pnl_sek=round(mkt_pnl, 2),
        ev_sek=round(mkt_pnl, 2),
        pnl_pct=round(_pnl_pct(mkt_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min)),
    )
    candidates.append(market_candidate)

    # ---- 5. Hold-to-close candidate ----
    # EV of holding = mean terminal P&L (expected value across all paths)
    terminal_pnls = np.array([
        _compute_pnl_sek(position, float(p), market, costs)
        for p in np.percentile(terminal, [10, 25, 50, 75, 90])
    ])
    hold_ev = float(np.mean(terminal_pnls))

    hold_candidate = CandidateExit(
        price_usd=round(median_terminal, 4),
        action="hold_to_close",
        fill_prob=1.0,
        expected_fill_time_min=round(remaining_min, 1),
        pnl_sek=round(fallback_pnl, 2),
        ev_sek=round(hold_ev, 2),
        pnl_pct=round(_pnl_pct(fallback_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min,
                                              session_max, session_min)),
    )
    candidates.append(hold_candidate)

    # ---- 6. Sort by EV descending ----
    candidates.sort(key=lambda c: c.ev_sek, reverse=True)

    # ---- 7. Stop-loss hit probability ----
    stop_prob = 0.0
    if stop_price_usd and stop_price_usd > 0:
        stop_prob = float(np.mean(session_min <= stop_price_usd))
    elif position.financing_level:
        # Use knock-out level + 3% buffer as effective stop
        stop_buffer = position.financing_level * 1.03
        stop_prob = float(np.mean(session_min <= stop_buffer))

    # ---- 8. Apply risk overrides to select recommendation ----
    recommended = _apply_risk_overrides(
        candidates, position, market, remaining_min, session_min
    )

    return ExitPlan(
        symbol=position.symbol,
        asof_ts=now,
        remaining_minutes=round(remaining_min, 1),
        candidates=candidates,
        recommended=recommended,
        market_exit=market_candidate,
        session_max_distribution=stats["max_quantiles"],
        session_min_distribution=stats["min_quantiles"],
        stop_hit_prob=round(stop_prob, 4),
        provenance={
            "model": "GBM_antithetic",
            "n_paths": n_paths,
            "volatility": round(vol, 4),
            "drift": round(drift, 4),
            "remaining_min": round(remaining_min),
            "instrument_type": position.instrument_type,
            "cost_model": costs.label,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: compute exit plan from existing system data
# ---------------------------------------------------------------------------

def compute_exit_plan_from_summary(
    ticker: str,
    agent_summary: dict,
    position_state: dict,
    session_end: datetime,
    *,
    instrument_type: str = "warrant",
    financing_level: float | None = None,
    leverage: float = 1.0,
    n_paths: int = DEFAULT_N_PATHS,
) -> ExitPlan | None:
    """Build exit plan from agent_summary and portfolio state data.

    Convenience wrapper that extracts price, volatility, and position data
    from the standard system data structures.

    Args:
        ticker: Underlying ticker (e.g., "XAG-USD").
        agent_summary: Agent summary dict with signals and prices.
        position_state: Position dict with shares, avg_cost, entry info.
        session_end: Session close time (UTC).
        instrument_type: "warrant", "stock", "crypto".
        financing_level: For MINI futures, the knock-out level.
        leverage: Effective leverage.
        n_paths: MC paths.

    Returns:
        ExitPlan or None if insufficient data.
    """
    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker, {})
    if not ticker_data:
        return None

    price = ticker_data.get("price_usd", 0)
    if price <= 0:
        return None

    extra = ticker_data.get("extra", {})
    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct")
    fx_rate = agent_summary.get("fx_rate", 10.85)

    # Build MarketSnapshot
    market = MarketSnapshot(
        asof_ts=datetime.now(UTC),
        price=price,
        atr_pct=atr_pct,
        usdsek=fx_rate,
    )

    # Build Position
    shares = position_state.get("shares", position_state.get("qty", 0))
    entry_price = position_state.get("entry_price_sek",
                                      position_state.get("entry_price", 0))
    entry_underlying = position_state.get("entry_underlying_usd",
                                           position_state.get("entry_underlying", price))
    entry_ts_str = position_state.get("entry_ts")
    entry_ts = datetime.now(UTC)
    if entry_ts_str:
        with contextlib.suppress(ValueError, TypeError):
            entry_ts = datetime.fromisoformat(entry_ts_str)

    position = Position(
        symbol=ticker,
        qty=shares,
        entry_price_sek=entry_price,
        entry_underlying_usd=entry_underlying,
        entry_ts=entry_ts,
        instrument_type=instrument_type,
        leverage=leverage,
        financing_level=financing_level,
    )

    return compute_exit_plan(position, market, session_end, n_paths=n_paths)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern \"compute_exit_plan\\(\" | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 569ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:16:plan = compute_exit_plan(position, market, session_end, cost_model)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:465:def compute_exit_plan(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\exit_optimizer.py:752:return compute_exit_plan(position, market, session_end, n_paths=n_paths)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:467:plan = compute_exit_plan(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_snipe_manager.py -TotalCount 560 | Select-Object -Last 140' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 620ms:
    try:
        market_summary = _summarize_market(snapshot)
        underlying_summary = market_summary.get("underlying") or {}
        # A-MC-2 (2026-04-11): was hardcoded `usdsek=1.0` which made every
        # SEK calculation downstream of compute_exit_plan wrong by ~10x.
        # exit_optimizer.py:312 multiplies underlying USD by usdsek to get
        # warrant value in SEK, and uses that in the optimizer's reward
        # function — so usdsek=1.0 understated SEK values by an order of
        # magnitude. Fetch the live rate (with the same 15-min cache the
        # rest of the system uses) and fall back to 10.85 only on total
        # API failure (matching fx_rates' own fallback behavior).
        from portfolio.fx_rates import fetch_usd_sek
        live_usdsek = fetch_usd_sek() or 10.85
        # A-MC-4 (2026-04-11): Real entry timestamp instead of `now()`.
        # Previously fin_snipe_manager always passed entry_ts=now() to the
        # exit_optimizer, which made `hold_hours = (now - entry_ts) ≈ 0`
        # for every cycle. The HOLD_TIME_EXTENDED risk flag at
        # exit_optimizer.py:390-393 fires only when hold_hours > 5, so it
        # never fired — the flag was permanently disabled despite being
        # in the live code path.
        #
        # Fix: persist the entry timestamp in instrument_state on first
        # observation of a non-zero position, then read it back on
        # subsequent calls. Bootstrap behavior: existing positions get
        # entry_ts = "first time we see them after this fix ships", which
        # is acceptable — it just means HOLD_TIME_EXTENDED becomes
        # available 5h after restart for already-open positions, vs.
        # never. New positions get the real first-observation time.
        entry_ts_iso = instrument_state.get("entry_ts")
        if entry_ts_iso:
            try:
                position_entry_ts = dt.datetime.fromisoformat(entry_ts_iso)
                if position_entry_ts.tzinfo is None:
                    position_entry_ts = position_entry_ts.replace(tzinfo=dt.UTC)
            except (TypeError, ValueError):
                position_entry_ts = dt.datetime.now(dt.UTC)
                instrument_state["entry_ts"] = position_entry_ts.isoformat()
        else:
            position_entry_ts = dt.datetime.now(dt.UTC)
            instrument_state["entry_ts"] = position_entry_ts.isoformat()
        # BUG-228: If estimate returns -1.0 sentinel (invalid inputs),
        # fall back to current_underlying for the exit plan (prevents
        # ZeroDivisionError) but don't persist it.
        _entry_und = _estimate_entry_underlying(snapshot, instrument_state)
        if _entry_und <= 0:
            _entry_und = current_underlying
        plan = compute_exit_plan(
            Position(
                symbol=snapshot["ticker"],
                qty=position_volume,
                entry_price_sek=position_avg,
                entry_underlying_usd=_entry_und,
                entry_ts=position_entry_ts,
                instrument_type="warrant",
                leverage=leverage,
                financing_level=None,
            ),
            MarketSnapshot(
                asof_ts=dt.datetime.now(dt.UTC),
                price=current_underlying,
                bid=float(underlying_summary.get("bid") or current_underlying),
                ask=float(underlying_summary.get("ask") or current_underlying),
                atr_pct=atr_pct if atr_pct > 0 else None,
                usdsek=live_usdsek,
                drift=0.0,
            ),
            session.session_end,
            n_paths=EXIT_OPTIMIZER_N_PATHS,
            seed=EXIT_OPTIMIZER_SEED,
        )
        target_underlying = float(plan.recommended.price_usd or fallback_underlying)
        translated = translate_underlying_target(
            float(snapshot.get("current_instrument_price") or current_bid or position_avg),
            current_underlying,
            target_underlying,
            leverage,
        )
        exit_price = _round_order_price(translated or fallback_price)
        minimum_profit_price = _round_order_price(max(position_avg, 0.0))
        if plan.recommended.action == "market" and current_bid > 0:
            exit_price = _round_order_price(current_bid)
            source = "quant_exit_optimizer_market"
        elif current_bid > 0:
            exit_price = max(exit_price, _round_order_price(current_bid))
            source = "quant_exit_optimizer"
            if exit_price < minimum_profit_price:
                exit_price = max(fallback_price, minimum_profit_price)
                source = "ladder_profit_guard"
        else:
            source = "quant_exit_optimizer"

        result.update({
            "price": exit_price,
            "underlying_price": target_underlying,
            "source": source,
            "fill_prob": float(plan.recommended.fill_prob),
            "expected_fill_time_min": float(plan.recommended.expected_fill_time_min),
            "stop_hit_prob": float(plan.stop_hit_prob),
            "risk_flags": list(plan.recommended.risk_flags),
            "action": plan.recommended.action,
            "optimizer_price": _round_order_price(translated or 0.0),
            "optimizer_underlying_price": target_underlying,
        })
    except Exception:
        logger.warning("Exit optimizer failed for %s", snapshot.get("orderbook_id"), exc_info=True)
    return result


def _compute_stop_plan(snapshot: dict, *, has_existing_stop: bool = False) -> dict[str, Any] | None:
    position_volume = int(snapshot.get("position_volume") or 0)
    position_avg = float(snapshot.get("position_average_price") or 0.0)
    current_bid = float(snapshot.get("current_bid") or 0.0)
    if position_volume <= 0 or position_avg <= 0:
        return None

    trigger_price = _round_order_price(position_avg * (1.0 - HARD_STOP_CERT_PCT))
    sell_price = _round_order_price(trigger_price * (1.0 - HARD_STOP_SELL_BUFFER_PCT))
    if trigger_price <= 0 or sell_price <= 0:
        return None

    distance_pct = ((current_bid - trigger_price) / current_bid * 100.0) if current_bid > 0 else None

    # Hysteresis: if we already have a managed stop, keep it regardless of distance.
    # Only skip placement of NEW stops when too close.
    if not has_existing_stop and distance_pct is not None and distance_pct < MIN_STOP_DISTANCE_PCT:
        return {
            "skip": True,
            "reason": "stop_too_close",
            "distance_pct": round(distance_pct, 2),
            "trigger_price": trigger_price,
            "sell_price": sell_price,
            "volume": position_volume,
        }

    return {
        "skip": False,
        "reason": "keep_existing" if has_existing_stop and distance_pct is not None and distance_pct < MIN_STOP_DISTANCE_PCT else "entry_minus_5pct",
        "distance_pct": round(distance_pct, 2) if distance_pct is not None else None,
        "trigger_price": trigger_price,
        "sell_price": sell_price,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_fish.py -TotalCount 900' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 684ms:
"""Fin Fish: intraday dip/spike fishing planner for metals warrants.

Computes optimal limit buy levels for metals warrants based on:
- Recent daily range patterns (how deep do dips / how high do spikes go?)
- ATR-based volatility and first-passage-time fill probabilities
- Structural support/resistance levels (Fibonacci, pivot, smart money)
- Chronos/model drift for directional bias
- RSI-based direction selection (BULL vs BEAR fishing)
- Avanza warrant barrier safety checks

Supports both BULL (buy dips) and BEAR (buy spikes) fishing.

The output is a ranked table of fishing levels with fill probability,
expected gain on bounce, EV in SEK, and barrier distance.

Machine-readable output via ``compute_fishing_plan()`` for snipe manager
integration.  CLI one-shot via ``main()``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
from contextlib import suppress
from pathlib import Path
from typing import Any

import requests

from portfolio.file_utils import atomic_append_jsonl, load_json
from portfolio.monte_carlo import drift_from_probability, volatility_from_atr
from portfolio.price_targets import (
    fill_probability,
    fill_probability_buy,
    structural_levels,
)

# ---------------------------------------------------------------------------
# External config — import from data.fin_fish_config with inline fallbacks
# ---------------------------------------------------------------------------
try:
    from data.fin_fish_config import (
        FISHING_BUDGET_SEK as _CFG_BUDGET,
    )
    from data.fin_fish_config import (
        FISHING_MIN_FILL_PROB as _CFG_MIN_FILL,
    )
    from data.fin_fish_config import (
        FISHING_PREFER_AVA as _CFG_PREFER_AVA,
    )
    from data.fin_fish_config import (
        FISHING_SL_CASCADE as _CFG_SL_CASCADE,
    )
    from data.fin_fish_config import (
        FISHING_TP_CASCADE as _CFG_TP_CASCADE,
    )
    from data.fin_fish_config import (
        PREFERRED_INSTRUMENTS as _CFG_PREFERRED,
    )
    from data.fin_fish_config import (  # type: ignore[import-untyped]
        WARRANT_CATALOG as _CFG_CATALOG,
    )
except Exception:
    _CFG_CATALOG = None
    _CFG_PREFERRED = None
    _CFG_BUDGET = None
    _CFG_MIN_FILL = None
    _CFG_TP_CASCADE = None
    _CFG_SL_CASCADE = None
    _CFG_PREFER_AVA = None

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary.json"
FISH_LOG_PATH = BASE_DIR / "data" / "fin_fish_log.jsonl"

logger = logging.getLogger("portfolio.fin_fish")

BINANCE_FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FAPI_PRICE = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"

UNDERLYING_SYMBOLS = {"XAG-USD": "XAGUSDT", "XAU-USD": "XAUUSDT"}

# ---------------------------------------------------------------------------
# Inline defaults — used when data.fin_fish_config is absent
# ---------------------------------------------------------------------------

# Warrant catalog — BULL and BEAR instruments
_DEFAULT_CATALOG: dict[str, dict] = {
    # --- Silver BULL ---
    "BULL_SILVER_X5_AVA_3": {
        "ob_id": "1069606",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X5 AVA 3",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Silver BEAR ---
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BULL ---
    "BULL_GULD_X5_AVA": {
        "ob_id": "738811",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL GULD X5 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BEAR (no viable AVA X5 — gold rally killed them all) ---
    "BEAR_GULD_X5_VON4": {
        "ob_id": "1047859",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X5 VON4",
        "issuer": "VON",
        "spread_pct": 2.2,
        "commission_sek": 0,
    },
    "BEAR_GULD_X2_AVA": {
        "ob_id": "738805",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 2.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X2 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
}

# Preferred instruments per (underlying, direction) — snipe manager picks these first
_DEFAULT_PREFERRED: dict[tuple[str, str], str] = {
    ("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_3",
    ("XAG-USD", "SHORT"): "BEAR_SILVER_X5_AVA_12",
    ("XAU-USD", "LONG"): "BULL_GULD_X5_AVA",
    ("XAU-USD", "SHORT"): "BEAR_GULD_X5_VON4",
}

_DEFAULT_BUDGET_SEK = 20_000
_DEFAULT_MIN_FILL_PROB = 0.02
_DEFAULT_TP_CASCADE: list[Any] = [
    {"underlying_pct": 1.5, "sell_pct": 40, "action": "move_stop_to_breakeven"},
    {"underlying_pct": 2.5, "sell_pct": 40, "action": "trail_stop_1pct"},
    {"underlying_pct": 4.0, "sell_pct": 20, "action": "close"},
]
_DEFAULT_SL_CASCADE: list[Any] = [
    {"underlying_pct": -1.0, "sell_pct": 50, "action": "partial_stop"},
    {"underlying_pct": -2.0, "sell_pct": 100, "action": "full_stop"},
]
_DEFAULT_PREFER_AVA = True

# Resolve config vs defaults
WARRANT_CATALOG: dict[str, dict] = _CFG_CATALOG if _CFG_CATALOG is not None else _DEFAULT_CATALOG
PREFERRED_INSTRUMENTS: dict[tuple[str, str], str] = (
    _CFG_PREFERRED if _CFG_PREFERRED is not None else _DEFAULT_PREFERRED
)
FISHING_BUDGET_SEK: float = _CFG_BUDGET if _CFG_BUDGET is not None else _DEFAULT_BUDGET_SEK
FISHING_MIN_FILL_PROB: float = _CFG_MIN_FILL if _CFG_MIN_FILL is not None else _DEFAULT_MIN_FILL_PROB
FISHING_TP_CASCADE: list[Any] = _CFG_TP_CASCADE if _CFG_TP_CASCADE is not None else _DEFAULT_TP_CASCADE
FISHING_SL_CASCADE: list[Any] = _CFG_SL_CASCADE if _CFG_SL_CASCADE is not None else _DEFAULT_SL_CASCADE
FISHING_PREFER_AVA: bool = _CFG_PREFER_AVA if _CFG_PREFER_AVA is not None else _DEFAULT_PREFER_AVA

# Avanza warrant hours (CET)
AVANZA_OPEN_H, AVANZA_OPEN_M = 8, 15
AVANZA_CLOSE_H, AVANZA_CLOSE_M = 21, 55

MIN_BARRIER_DISTANCE_PCT = 5.0
DEFAULT_BOUNCE_PCT = 2.0  # +2% underlying = take-profit target


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_live_spot() -> dict[str, dict]:
    """Fetch live spot prices and 24h stats from Binance FAPI."""
    result = {}
    for ticker, symbol in UNDERLYING_SYMBOLS.items():
        try:
            r = requests.get(f"{BINANCE_FAPI_TICKER}?symbol={symbol}", timeout=5)
            d = r.json()
            result[ticker] = {
                "price": float(d["lastPrice"]),
                "high_24h": float(d["highPrice"]),
                "low_24h": float(d["lowPrice"]),
                "change_pct": float(d["priceChangePercent"]),
                "volume_usd": float(d["quoteVolume"]),
            }
        except Exception as e:
            logger.warning("Binance %s error: %s", ticker, e)
    return result


def fetch_daily_ranges(ticker: str, days: int = 10) -> list[dict]:
    """Fetch recent daily candles for range analysis."""
    symbol = UNDERLYING_SYMBOLS.get(ticker)
    if not symbol:
        return []
    try:
        r = requests.get(
            BINANCE_FAPI_KLINES,
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=10,
        )
        candles = r.json()
        result = []
        for c in candles:
            high = float(c[2])
            low = float(c[3])
            close = float(c[4])
            result.append({
                "date": datetime.datetime.fromtimestamp(c[0] / 1000).strftime("%m-%d"),
                "high": high,
                "low": low,
                "close": close,
                "range_pct": round((high - low) / low * 100, 2) if low > 0 else 0,
            })
        return result
    except Exception as e:
        logger.warning("Daily candles error for %s: %s", ticker, e)
        return []


def session_hours_remaining() -> float:
    """Compute hours remaining in the Avanza warrant session (CET)."""
    try:
        import zoneinfo
        cet = zoneinfo.ZoneInfo("Europe/Stockholm")
    except Exception:
        import dateutil.tz  # type: ignore[import-untyped]
        cet = dateutil.tz.gettz("Europe/Stockholm")

    now = datetime.datetime.now(cet)
    close = now.replace(hour=AVANZA_CLOSE_H, minute=AVANZA_CLOSE_M, second=0, microsecond=0)
    open_time = now.replace(hour=AVANZA_OPEN_H, minute=AVANZA_OPEN_M, second=0, microsecond=0)

    if now < open_time:
        return 0.0  # before market open
    if now >= close:
        return 0.0  # after market close

    remaining = (close - now).total_seconds() / 3600
    return max(0.0, round(remaining, 2))


def load_signal_data(ticker: str) -> dict:
    """Load signal data for a ticker from agent_summary."""
    summary = load_json(SUMMARY_PATH) or {}
    signals = summary.get("signals", {})
    entry = signals.get(ticker, {})
    focus = (summary.get("focus_probabilities") or {}).get(ticker, {})
    extra = entry.get("extra") or {}
    return {
        "entry": entry,
        "focus": focus,
        "price_usd": _safe_float(entry.get("price_usd")),
        "rsi": _safe_float(entry.get("rsi")),
        "atr_pct": _safe_float(entry.get("atr_pct"), 0.5),
        "regime": str(entry.get("regime") or ""),
        "action": str(entry.get("action") or "HOLD"),
        "weighted_confidence": _safe_float(entry.get("weighted_confidence")),
        "fear_greed": extra.get("fear_greed"),
        "econ_action": str(extra.get("econ_calendar_action") or "HOLD"),
        "news_action": str(extra.get("news_event_action") or "HOLD"),
    }


def fetch_fx_rate() -> float:
    """Fetch USD/SEK exchange rate."""
    try:
        from portfolio.fx_rates import fetch_usd_sek
        return fetch_usd_sek()
    except Exception:
        return 10.0  # fallback


def _get_chronos_drift(signal: dict) -> float | None:
    """Extract Chronos 24h drift from signal data, or None if unavailable."""
    extra = (signal["entry"].get("extra") or {}) if signal.get("entry") else None
    if not extra:
        return None
    forecast_ind = extra.get("forecast_indicators") or {}
    chronos_24h = forecast_ind.get("chronos_24h_pct")
    if chronos_24h is None:
        return None
    return float(chronos_24h)


def _compute_vol_and_drift(
    signal: dict,
    daily_ranges: list[dict],
    direction: str,
) -> tuple[float, float]:
    """Compute annualized volatility and drift for GBM.

    Parameters
    ----------
    direction : str
        ``"LONG"`` biases drift downward (we want dips), ``"SHORT"`` biases
        drift upward (we want spikes).
    """
    atr_pct = signal["atr_pct"]
    p_up = _safe_float((signal["focus"].get("3h") or {}).get("probability"), 0.5)

    # Hourly ATR path — volatility_from_atr assumes hourly candles
    vol = volatility_from_atr(atr_pct)

    # Daily range path — annualize daily sigma directly with sqrt(252)
    # (volatility_from_atr uses sqrt(252/14) which is wrong for daily data)
    if daily_ranges and len(daily_ranges) >= 3:
        recent_ranges = [c["range_pct"] for c in daily_ranges[-5:] if c["range_pct"] > 0.5]
        if recent_ranges:
            avg_range = sum(recent_ranges) / len(recent_ranges)
            daily_sigma = avg_range / 1.5 / 100.0
            vol_from_daily = daily_sigma * math.sqrt(252.0)
            vol = max(vol, vol_from_daily)

    # For LONG fishing we want P(dip) — use p_up as-is (lower p_up = more likely to dip)
    # For SHORT fishing we want P(spike) — invert: use 1-p_up
    if direction == "SHORT":
        drift = drift_from_probability(1.0 - p_up, vol)
    else:
        drift = drift_from_probability(p_up, vol)

    # Blend Chronos drift if available
    chronos_pct = _get_chronos_drift(signal)
    if chronos_pct is not None:
        chronos_annual = (chronos_pct / 100.0) * 252
        drift = 0.7 * drift + 0.3 * chronos_annual

    return vol, drift


# ---------------------------------------------------------------------------
# Direction selection
# ---------------------------------------------------------------------------

def choose_fishing_directions(signal: dict) -> list[dict]:
    """Decide whether to fish BULL, BEAR, or both.

    Uses RSI, Chronos, consensus signal, Fear & Greed, econ calendar,
    and news severity to set conviction scores for each direction.

    Returns a list of dicts:
        [{"direction": "LONG", "conviction": 0.8}, ...]
    """
    rsi = signal["rsi"]
    chronos_pct = _get_chronos_drift(signal)

    directions: list[dict] = []

    # --- Base conviction from RSI ---
    if rsi < 45:
        bull_conv = 0.8 if rsi < 30 else 0.65
        bear_conv = 0.0
    elif rsi > 65:
        bear_conv = 0.8 if rsi > 70 else 0.65
        bull_conv = 0.0
    else:
        bull_conv = 0.4
        bear_conv = 0.4

    # --- Chronos 24h forecast ---
    if chronos_pct is not None:
        if chronos_pct < -0.3:
            bear_conv = max(bear_conv, 0.3) + 0.15
            bull_conv = max(0.0, bull_conv - 0.1)
        elif chronos_pct > 0.3:
            bull_conv = max(bull_conv, 0.3) + 0.15
            bear_conv = max(0.0, bear_conv - 0.1)

    # --- Consensus signal (30-signal weighted vote) ---
    action = signal.get("action", "HOLD")
    confidence = signal.get("weighted_confidence", 0)
    if action == "BUY" and confidence > 0.6:
        bull_conv += 0.15
    elif action == "SELL" and confidence > 0.6:
        bear_conv += 0.15

    # --- Fear & Greed (contrarian) ---
    fg = signal.get("fear_greed")
    if fg is not None:
        fg_val = _safe_float(fg)
        if fg_val <= 20:
            bull_conv += 0.15   # extreme fear → buy dips
        elif fg_val >= 80:
            bear_conv += 0.15   # extreme greed → sell peaks

    # --- Econ calendar (FOMC/CPI imminent → risk-off) ---
    econ_action = signal.get("econ_action", "HOLD")
    if econ_action == "SELL":
        bear_conv += 0.10
        bull_conv = max(0.0, bull_conv - 0.10)

    # --- News severity ---
    news_action = signal.get("news_action", "HOLD")
    if news_action == "SELL":
        bear_conv += 0.10
    elif news_action == "BUY":
        bull_conv += 0.10

    # Clamp to [0, 1]
    bull_conv = min(1.0, max(0.0, bull_conv))
    bear_conv = min(1.0, max(0.0, bear_conv))

    if bull_conv > 0.05:
        directions.append({"direction": "LONG", "conviction": round(bull_conv, 2)})
    if bear_conv > 0.05:
        directions.append({"direction": "SHORT", "conviction": round(bear_conv, 2)})

    return directions


# ---------------------------------------------------------------------------
# Core fishing level computation — BULL (dip) and BEAR (spike)
# ---------------------------------------------------------------------------

def compute_fishing_levels_bull(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate BULL fishing (dip-buy) levels with fill probabilities.

    Looks for levels BELOW current price where price might dip to.
    """
    vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="LONG")
    extra = (signal["entry"].get("extra") or {}) if signal["entry"] else None

    candidates: dict[float, str] = {}

    # 1. ATR-based offsets below spot
    atr_pct = signal["atr_pct"]
    for n, label in [(0.5, "ATR 0.5x"), (1.0, "ATR 1x"), (1.5, "ATR 1.5x"),
                     (2.0, "ATR 2x"), (3.0, "ATR 3x")]:
        level = spot * (1 - n * atr_pct / 100)
        candidates[round(level, 4)] = label

    # 2. Fixed percentage offsets below spot
    for pct in [1, 2, 3, 5, 7, 10]:
        level = spot * (1 - pct / 100)
        candidates[round(level, 4)] = f"-{pct}%"

    # 3. Recent daily lows
    for candle in daily_ranges[-5:]:
        low = candle["low"]
        if 0 < low < spot:
            candidates[round(low, 4)] = f"Daily low {candle['date']}"

    # 4. Structural levels below current price
    levels = structural_levels(spot, signal["entry"], extra)
    for name, lvl in levels.items():
        if 0 < lvl < spot * 0.99:
            candidates[round(lvl, 4)] = f"Struct: {name}"

    return _score_candidates_buy(candidates, spot, vol, drift, hours)


def compute_fishing_levels_bear(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate BEAR fishing (spike-buy) levels with fill probabilities.

    Looks for levels ABOVE current price where price might spike to —
    we would buy a BEAR cert at that elevated price.
    """
    vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="SHORT")
    extra = (signal["entry"].get("extra") or {}) if signal["entry"] else None

    candidates: dict[float, str] = {}

    # 1. ATR-based offsets above spot
    atr_pct = signal["atr_pct"]
    for n, label in [(0.5, "ATR 0.5x"), (1.0, "ATR 1x"), (1.5, "ATR 1.5x"),
                     (2.0, "ATR 2x"), (3.0, "ATR 3x")]:
        level = spot * (1 + n * atr_pct / 100)
        candidates[round(level, 4)] = label

    # 2. Fixed percentage offsets above spot
    for pct in [1, 2, 3, 5, 7, 10]:
        level = spot * (1 + pct / 100)
        candidates[round(level, 4)] = f"+{pct}%"

    # 3. Recent daily highs
    for candle in daily_ranges[-5:]:
        high = candle["high"]
        if high > spot:
            candidates[round(high, 4)] = f"Daily high {candle['date']}"

    # 4. Structural levels above current price (resistance)
    levels = structural_levels(spot, signal["entry"], extra)
    for name, lvl in levels.items():
        if lvl > spot * 1.01:
            candidates[round(lvl, 4)] = f"Struct: {name}"

    return _score_candidates_sell(candidates, spot, vol, drift, hours)


def _score_candidates_buy(
    candidates: dict[float, str],
    spot: float,
    vol: float,
    drift: float,
    hours: float,
) -> list[dict]:
    """Score BULL fishing candidates (below spot) with fill probability and EV."""
    results = []
    for level, source in sorted(candidates.items(), reverse=True):
        if level <= 0 or level >= spot:
            continue
        dip_pct = round((spot - level) / spot * 100, 2)
        if dip_pct < 0.3:
            continue

        fp = fill_probability_buy(spot, level, vol, drift, hours, is_24h=True)
        bounce_pct = dip_pct  # symmetric bounce back to current
        modest_bounce_pct = round(DEFAULT_BOUNCE_PCT, 2)
        modest_bounce_target = round(level * (1 + DEFAULT_BOUNCE_PCT / 100), 4)

        results.append({
            "level": level,
            "source": source,
            "dip_pct": dip_pct,
            "move_pct": dip_pct,  # normalized: how far from spot
            "fill_prob": round(fp, 4),
            "bounce_to_spot_pct": round(bounce_pct, 2),
            "modest_bounce_pct": modest_bounce_pct,
            "modest_bounce_target": modest_bounce_target,
        })

    return _dedupe_and_rank(results, key_field="level")


def _score_candidates_sell(
    candidates: dict[float, str],
    spot: float,
    vol: float,
    drift: float,
    hours: float,
) -> list[dict]:
    """Score BEAR fishing candidates (above spot) with fill probability and EV."""
    results = []
    for level, source in sorted(candidates.items()):
        if level <= spot:
            continue
        spike_pct = round((level - spot) / spot * 100, 2)
        if spike_pct < 0.3:
            continue

        # fill_probability gives P(running max >= target) — exactly what we need
        fp = fill_probability(spot, level, vol, drift, hours, is_24h=True)
        bounce_pct = spike_pct  # symmetric drop back to current
        modest_bounce_pct = round(DEFAULT_BOUNCE_PCT, 2)
        modest_bounce_target = round(level * (1 - DEFAULT_BOUNCE_PCT / 100), 4)

        results.append({
            "level": level,
            "source": source,
            "dip_pct": spike_pct,   # legacy name — represents distance from spot
            "move_pct": spike_pct,
            "fill_prob": round(fp, 4),
            "bounce_to_spot_pct": round(bounce_pct, 2),
            "modest_bounce_pct": modest_bounce_pct,
            "modest_bounce_target": modest_bounce_target,
        })

    return _dedupe_and_rank(results, key_field="level")


def _dedupe_and_rank(results: list[dict], key_field: str = "level") -> list[dict]:
    """Deduplicate levels within 0.2% of each other and rank by EV."""
    sorted_results = sorted(results, key=lambda x: x[key_field], reverse=True)
    deduped: list[dict] = []
    for r in sorted_results:
        if not deduped or abs(r[key_field] - deduped[-1][key_field]) / r[key_field] > 0.002:
            deduped.append(r)

    for r in deduped:
        r["ev_score"] = round(r["fill_prob"] * r["bounce_to_spot_pct"], 4)

    deduped.sort(key=lambda x: x["ev_score"], reverse=True)
    return deduped


# Legacy alias — keep backward compatibility
def compute_fishing_levels(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate fishing (dip-buy) levels with fill probabilities.

    Legacy wrapper — delegates to ``compute_fishing_levels_bull``.
    """
    return compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)


# ---------------------------------------------------------------------------
# Warrant evaluation
# ---------------------------------------------------------------------------

def _select_warrants(
    ticker: str,
    direction: str,
    spot: float,
) -> list[dict]:
    """Select matching warrants for (ticker, direction), preferring the preferred
    instrument and falling back to catalog search.

    Parameters
    ----------
    direction : str
        ``"LONG"`` for BULL, ``"SHORT"`` for BEAR.
    """
    pref_key = (ticker, direction)
    preferred_id = PREFERRED_INSTRUMENTS.get(pref_key)

    # Collect all matching warrants
    matching = [
        w for w in WARRANT_CATALOG.values()
        if w["underlying"] == ticker and w["direction"] == direction
    ]

    if not matching:
        return []

    # If we have a preferred instrument and it exists, put it first
    if preferred_id and preferred_id in WARRANT_CATALOG:
        pref_warrant = WARRANT_CATALOG[preferred_id]
        if pref_warrant in matching:
            matching.remove(pref_warrant)
            matching.insert(0, pref_warrant)

    # If FISHING_PREFER_AVA, sort AVA warrants before others (after preferred)
    if FISHING_PREFER_AVA:
        ava_first = []
        others = []
        for w in matching:
            if w.get("issuer") == "AVA":
                ava_first.append(w)
            else:
                others.append(w)
        # Preferred is already first if it's AVA; otherwise keep order
        matching = ava_first + others

    return matching


def evaluate_warrants(
    ticker: str,
    spot: float,
    fishing_levels: list[dict],
    budget_sek: float,
    fx_rate: float,
    direction: str = "LONG",
) -> list[dict]:
    """Match fishing levels to available warrants, compute sizing and EV.

    Parameters
    ----------
    direction : str
        ``"LONG"`` for BULL certs, ``"SHORT"`` for BEAR certs.
    """
    matching = _select_warrants(ticker, direction, spot)

    results = []
    for warrant in matching:
        barrier = warrant["barrier"]
        name = warrant["name"]
        is_daily_cert = warrant["api_type"] == "certificate" and barrier == 0

        # Dynamic leverage: compute from spot and barrier for warrants.
        # Config leverage is stale (set when cert was added, not at current price).
        # Daily certs (no barrier) keep config leverage.
        if not is_daily_cert and barrier > 0:
            dist = abs(spot - barrier)
            leverage = spot / dist if dist > 0 else warrant["leverage"]
        else:
            leverage = warrant["leverage"]

        # Barrier checks only for MINI warrants (barrier > 0)
        if not is_daily_cert and barrier > 0:
            if direction == "LONG" and spot <= barrier:
                continue  # knocked out
            if direction == "SHORT" and spot >= barrier:
                # BEAR MINIs get knocked out if underlying goes above barrier
                # (depends on product, but skip if too close)
                pass
            barrier_distance = abs(spot - barrier) / spot * 100
            if barrier_distance < MIN_BARRIER_DISTANCE_PCT:
                continue
        else:
            barrier_distance = 100.0

        for fl in fishing_levels:
            level = fl["level"]

            # Check barrier safety at fishing level (MINI warrants only)
            if not is_daily_cert and barrier > 0:
                if direction == "LONG":
                    fish_barrier_dist = round((level - barrier) / level * 100, 2)
                else:
                    fish_barrier_dist = round(abs(level - barrier) / level * 100, 2)
                if fish_barrier_dist < MIN_BARRIER_DISTANCE_PCT:
                    continue
            else:
                fish_barrier_dist = 100.0

            # Estimate warrant price at fishing level
            parity = warrant.get("parity", 10)
            spread_pct = warrant.get("spread_pct", 1.0)
            commission = warrant.get("commission_sek", 0)
            issuer = warrant.get("issuer", "?")

            if is_daily_cert:
                # Daily leverage cert: we cannot compute price from underlying.
                # Use budget / leverage for sizing.
                # Gain = underlying_move_pct * leverage - spread_pct
                warrant_price_at_fish = None  # unknown without Avanza quote
            else:
                if direction == "LONG":
                    warrant_price_at_fish = max(0.01, (level - barrier) / parity * fx_rate)
                else:
                    warrant_price_at_fish = max(0.01, (barrier - level) / parity * fx_rate)

            # Underlying move that generates profit
            bounce_underlying_pct = fl["bounce_to_spot_pct"]

            if is_daily_cert:
                # gain = underlying_move% * leverage - spread%
                warrant_gain_pct = round(bounce_underlying_pct * leverage, 2)
                net_gain_pct = round(warrant_gain_pct - spread_pct, 2)

                if net_gain_pct <= 0:
                    continue

                # For sizing daily certs, estimate cost from budget
                # We don't know the exact cert price, so compute EV as % of budget
                invest_sek = budget_sek
                gain_sek = round(budget_sek * net_gain_pct / 100, 2)
                spread_cost_sek = round(budget_sek * spread_pct / 100, 2)
                qty = 0  # unknown without live cert price
                display_price = 0.0
            else:
                warrant_gain_pct = round(bounce_underlying_pct * leverage, 2)
                net_gain_pct = round(warrant_gain_pct - spread_pct, 2)

                if net_gain_pct <= 0:
                    continue

                assert warrant_price_at_fish is not None  # MINI warrants always have a price
                qty = max(1, int(budget_sek / warrant_price_at_fish))
                invest_sek = round(qty * warrant_price_at_fish, 0)
                gross_sek = round(qty * warrant_price_at_fish * warrant_gain_pct / 100, 2)
                spread_cost_sek = round(qty * warrant_price_at_fish * spread_pct / 100, 2)
                gain_sek = round(gross_sek - spread_cost_sek - commission, 2)
                display_price = round(warrant_price_at_fish, 2)

            ev_sek = round(fl["fill_prob"] * gain_sek, 2)

            results.append({
                "level": level,
                "source": fl["source"],
                "dip_pct": fl["dip_pct"],
                "move_pct": fl.get("move_pct", fl["dip_pct"]),
                "fill_prob": fl["fill_prob"],
                "warrant": name,
                "ob_id": warrant["ob_id"],
                "issuer": issuer,
                "leverage": leverage,
                "barrier": barrier,
                "barrier_dist_pct": fish_barrier_dist,
                "warrant_price": display_price,
                "qty": qty,
                "invest_sek": invest_sek,
                "bounce_pct": bounce_underlying_pct,
                "warrant_gain_pct": warrant_gain_pct,
                "spread_pct": spread_pct,
                "spread_cost_sek": spread_cost_sek,
                "net_gain_pct": net_gain_pct,
                "gain_sek": gain_sek,
                "ev_sek": ev_sek,
                "direction": direction,
                "is_daily_cert": is_daily_cert,
            })

    results.sort(key=lambda x: x["ev_sek"], reverse=True)

    # Deduplicate: keep only the best warrant per price level (within 0.2%).
    # Without this, the same level appears N times for N different warrants,
    # drowning out other price levels in the output.
    deduped: list[dict] = []
    for r in results:
        is_dup = False
        for kept in deduped:
            if abs(r["level"] - kept["level"]) / max(r["level"], 1e-9) < 0.002:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)

    return deduped


# ---------------------------------------------------------------------------
# Structured plan output (for snipe manager)
# ---------------------------------------------------------------------------

def _build_instrument_info(warrant_results: list[dict], direction: str) -> dict:
    """Extract instrument metadata from the best warrant result."""
    if not warrant_results:
        return {}
    best_name = warrant_results[0]["warrant"]
    for w in WARRANT_CATALOG.values():
        if w["name"] == best_name:
            return {
                "ob_id": w["ob_id"],
                "name": w["name"],
                "leverage": w["leverage"],
                "barrier": w["barrier"],
                "issuer": w.get("issuer", "?"),
                "spread_pct": w.get("spread_pct", 1.0),
                "api_type": w.get("api_type", "warrant"),
                "direction": direction,
            }
    return {}


def compute_fishing_plan(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
    budget_sek: float | None = None,
    fx_rate: float | None = None,
) -> list[dict]:
    """Compute a structured fishing plan for a ticker.

    Returns a list of plans (one per direction: LONG and/or SHORT) with
    machine-readable structure for the snipe manager.

    Each plan dict::

        {
            "ticker": "XAG-USD",
            "spot": 69.21,
            "direction": "LONG",      # or "SHORT"
            "conviction": 0.65,
            "levels": [...],           # fishing level dicts with fill_prob, ev_sek
            "instrument": {"ob_id": ..., "name": ..., "leverage": ...},
            "tp_cascade": [...],       # from FISHING_TP_CASCADE config
            "sl_cascade": [...],       # from FISHING_SL_CASCADE config

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content -Path portfolio/fin_fish.py -Tail 500' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 720ms:
        BULL X5: fish $67.50 (-2.5%) fill 11% EV 31
        BEAR X5: fish $71.00 (+2.6%) fill 8% EV 22
        Avanza: OFFLINE
    """
    if not plans:
        return "FISH: no viable plans"

    lines: list[str] = []
    ticker = plans[0]["ticker"]
    spot = plans[0]["spot"]
    lines.append(f"FISH PLAN {ticker} ${spot:.2f}")

    for plan in plans:
        direction = plan["direction"]
        label = "BULL" if direction == "LONG" else "BEAR"
        leverage = plan["instrument"].get("leverage", 1)
        best_levels = plan["levels"][:1]  # top level only for Telegram

        for lvl in best_levels:
            level_price = lvl["level"]
            if direction == "LONG":
                move_str = f"-{lvl['move_pct']:.1f}%"
            else:
                move_str = f"+{lvl['move_pct']:.1f}%"
            fill_pct = lvl["fill_prob"] * 100
            ev = lvl["ev_sek"]
            lines.append(
                f"{label} X{leverage:.0f}: fish ${level_price:.2f} ({move_str}) "
                f"fill {fill_pct:.0f}% EV {ev:.0f}"
            )

    avanza_status = "online" if avanza_online else "OFFLINE"
    lines.append(f"Avanza: {avanza_status}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI report formatting
# ---------------------------------------------------------------------------

def format_report(
    ticker: str,
    spot_data: dict,
    signal: dict,
    daily_ranges: list[dict],
    plans: list[dict],
    hours: float,
    max_levels: int = 8,
) -> str:
    """Format the fishing plan as a readable CLI report."""
    spot = spot_data["price"]
    lines: list[str] = []
    lines.append(f"=== {ticker} -- ${spot:.2f} ===")
    lines.append(f"  24h: ${spot_data['low_24h']:.2f} - ${spot_data['high_24h']:.2f} "
                 f"({spot_data['change_pct']:+.2f}%) | Vol ${spot_data['volume_usd']/1e6:.0f}M")
    lines.append(f"  Regime: {signal['regime']} | RSI: {signal['rsi']:.1f} | "
                 f"ATR: {signal['atr_pct']:.2f}% | Signal: {signal['action']}")
    # Signal boosters line
    fg = signal.get("fear_greed")
    fg_str = f"F&G: {fg}" if fg is not None else "F&G: n/a"
    conf_str = f"Consensus: {signal['action']} {signal['weighted_confidence']:.0%}"
    econ_str = f"Econ: {signal.get('econ_action', 'HOLD')}"
    news_str = f"News: {signal.get('news_action', 'HOLD')}"
    lines.append(f"  {fg_str} | {conf_str} | {news_str} | {econ_str}")
    lines.append(f"  Session hours left: {hours:.1f}h")

    # Daily range pattern
    if daily_ranges:
        recent = daily_ranges[-5:]
        lines.append(f"  Daily ranges (last {len(recent)}d):")
        for c in recent:
            lines.append(f"    {c['date']}: ${c['low']:.2f}-${c['high']:.2f} ({c['range_pct']:.1f}%)")
        avg_range = sum(c["range_pct"] for c in recent) / len(recent)
        avg_low_dip = sum((c["high"] - c["low"]) / c["high"] * 100 for c in recent) / len(recent)
        lines.append(f"  Avg daily range: {avg_range:.1f}% | Avg dip from high: {avg_low_dip:.1f}%")

    # Direction analysis
    directions = choose_fishing_directions(signal)
    chronos_pct = _get_chronos_drift(signal)
    dir_labels = ", ".join(
        f"{'BULL' if d['direction'] == 'LONG' else 'BEAR'} ({d['conviction']:.0%})"
        for d in directions
    )
    lines.append(f"  Direction: {dir_labels}")
    if chronos_pct is not None:
        lines.append(f"  Chronos 24h: {chronos_pct:+.2f}%")
    lines.append("")

    if not plans:
        lines.append("  No viable fishing plans found.")
        return "\n".join(lines)

    for plan in plans:
        direction = plan["direction"]
        label = "BULL" if direction == "LONG" else "BEAR"
        conviction = plan["conviction"]
        instrument = plan.get("instrument", {})
        inst_name = instrument.get("name", "?")
        inst_lev = instrument.get("leverage", 1)
        warrant_results = plan["levels"]

        lines.append(f"  --- {label} fishing (conviction {conviction:.0%}) "
                     f"via {inst_name} (X{inst_lev:.1f}) ---")

        if not warrant_results:
            lines.append("    No viable levels.")
            lines.append("")
            continue

        # Table header
        if direction == "LONG":
            lines.append(f"    {'Level':>9} {'Dip%':>6} {'Fill%':>6} {'Gross':>6} {'Sprd':>5} "
                         f"{'Net%':>5} {'EV/SEK':>7} {'Barr%':>6} {'Source':<20}")
        else:
            lines.append(f"    {'Level':>9} {'Spike%':>6} {'Fill%':>6} {'Gross':>6} {'Sprd':>5} "
                         f"{'Net%':>5} {'EV/SEK':>7} {'Barr%':>6} {'Source':<20}")
        lines.append(f"    {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*5} "
                     f"{'-'*5} {'-'*7} {'-'*6} {'-'*20}")

        shown = 0
        for r in warrant_results:
            if shown >= max_levels:
                break
            if r["fill_prob"] < FISHING_MIN_FILL_PROB:
                continue

            lines.append(
                f"    ${r['level']:>8.2f} {r['move_pct']:>5.1f}% {r['fill_prob']:>5.1%} "
                f"{r['warrant_gain_pct']:>5.1f}% {r['spread_pct']:>4.1f}% "
                f"{r['net_gain_pct']:>4.1f}% {r['ev_sek']:>6.0f} {r['barrier_dist_pct']:>5.1f}% "
                f"{r['source']:>20s}"
            )
            shown += 1

        if shown == 0:
            lines.append(f"    All levels have <{FISHING_MIN_FILL_PROB:.0%} fill probability.")

        # TP/SL cascade (handles both dict-style and float-style entries)
        tp_str = ", ".join(
            f"+{t['underlying_pct']:.1f}%" if isinstance(t, dict) else f"+{t:.1f}%"
            for t in plan["tp_cascade"]
        )
        sl_str = ", ".join(
            f"{s['underlying_pct']:.1f}%" if isinstance(s, dict) else f"-{s:.1f}%"
            for s in plan["sl_cascade"]
        )
        lines.append(f"    TP cascade (underlying): {tp_str}")
        lines.append(f"    SL cascade (underlying): {sl_str}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fin Fish: compute optimal dip/spike fishing levels for metals warrants."
    )
    parser.add_argument("--hours", type=float, default=0,
                        help="Override planning horizon (default: auto-compute from session).")
    parser.add_argument("--budget", type=float, default=FISHING_BUDGET_SEK,
                        help=f"Budget per fishing level in SEK (default: {FISHING_BUDGET_SEK}).")
    parser.add_argument("--metals", default="silver",
                        help="Comma-separated: silver,gold (default: silver).")
    parser.add_argument("--max-levels", type=int, default=8,
                        help="Max fishing levels to display per metal (default: 8).")
    parser.add_argument("--direction", choices=["bull", "bear", "auto"], default="auto",
                        help="Force direction: bull, bear, or auto (default: auto).")
    parser.add_argument("--telegram", action="store_true",
                        help="Print Telegram-format summary instead of full report.")
    parser.add_argument("--monitor", action="store_true",
                        help="Enter smart monitoring mode after analysis.")
    parser.add_argument("--entry-price", type=float, default=0,
                        help="Entry price for monitoring (default: current spot).")
    parser.add_argument("--cert-price", type=float, default=0,
                        help="Certificate entry price in SEK for P&L tracking.")
    parser.add_argument("--cert-units", type=int, default=0,
                        help="Number of certificate units held.")
    parser.add_argument("--leverage", type=float, default=5.0,
                        help="Certificate leverage (default: 5x).")
    args = parser.parse_args()

    metals = [m.strip().lower() for m in args.metals.split(",")]
    ticker_map = {"silver": "XAG-USD", "gold": "XAU-USD"}
    tickers = [ticker_map[m] for m in metals if m in ticker_map]

    if not tickers:
        print("No valid metals specified. Use: silver, gold")
        return 1

    # Session hours
    hours = args.hours if args.hours > 0 else session_hours_remaining()
    if hours <= 0:
        # Outside trading hours — use next session (planning mode)
        hours = 13.67  # full session 08:15-21:55
        print("Outside Avanza hours -- showing NEXT SESSION plan (13.7h horizon)\n")
    else:
        print(f"Session: {hours:.1f}h remaining until 21:55 CET\n")

    # Fetch data
    print("Fetching live prices...")
    spot_data = fetch_live_spot()
    fx_rate = fetch_fx_rate()
    print(f"FX rate: {fx_rate:.2f} SEK/USD\n")

    # --- Preflight GO/NO-GO check ---
    print("Running preflight check...")
    preflight_results = {}
    try:
        from scripts.fish_preflight import compute_preflight, print_preflight
        for ticker in tickers:
            pf = compute_preflight(ticker)
            preflight_results[ticker] = pf
            print_preflight(pf)
    except Exception as e:
        print(f"  Preflight unavailable: {e}")

    # --- Instrument profile briefing ---
    try:
        from portfolio.instrument_profile import (
            format_profile_briefing,
            get_profile,
        )
        signal_data = load_json(BASE_DIR / "data" / "agent_summary_compact.json")
        for ticker in tickers:
            profile = get_profile(ticker)
            if profile:
                print(f"\n{'='*60}")
                print(f"  INSTRUMENT PROFILE: {profile['name']}")
                print(f"{'='*60}")
                print(format_profile_briefing(ticker, signal_data))

                # Show signal reliability ranking for this ticker
                reliability = (signal_data or {}).get("signal_reliability", {}).get(ticker, {})
                if reliability:
                    ranked = sorted(
                        [(k, v) for k, v in reliability.items() if isinstance(v, dict) and v.get("total", 0) >= 30],
                        key=lambda x: x[1].get("accuracy", 0),
                        reverse=True,
                    )
                    if ranked:
                        print("\n  Signal reliability (top 10 / bottom 3):")
                        for name, data in ranked[:10]:
                            acc = data.get("accuracy", 0)
                            n = data.get("total", 0)
                            marker = " *" if name in profile.get("trusted_signals", []) else ""
                            print(f"    {name:20s} {acc:5.1%} ({n:4d} samples){marker}")
                        if len(ranked) > 10:
                            print("    ...")
                            for name, data in ranked[-3:]:
                                acc = data.get("accuracy", 0)
                                n = data.get("total", 0)
                                marker = " X" if name in profile.get("ignored_signals", []) else ""
                                print(f"    {name:20s} {acc:5.1%} ({n:4d} samples){marker}")

                # Show deep context summary if available (with staleness check)
                precompute_path = BASE_DIR / profile.get("precompute_file", "")
                deep_ctx = load_json(precompute_path)
                if deep_ctx:
                    import datetime as _dt
                    _gen = deep_ctx.get("generated_at", "")
                    if _gen:
                        with suppress(Exception):
                            _age_s = (_dt.datetime.now(_dt.UTC) - _dt.datetime.fromisoformat(_gen)).total_seconds()
                            if _age_s > 7200:  # 2 hours
                                print(f"  ⚠ Deep context STALE ({_age_s/3600:.1f}h old)")
                    analyst = deep_ctx.get("analyst_targets", {})
                    if analyst:
                        targets = []
                        for bank, data in analyst.items():
                            if isinstance(data, dict) and data.get("target"):
                                targets.append(f"{bank}: ${data['target']}")
                            elif isinstance(data, (int, float)):
                                targets.append(f"{bank}: ${data}")
                        if targets:
                            print(f"\n  Analyst targets: {', '.join(targets[:5])}")

                    cot = deep_ctx.get("cot_positioning", {})
                    if cot:
                        trend = cot.get("trend", "")
                        if trend:
                            print(f"  COT trend: {trend}")

                print()
    except Exception as e:
        logger.debug("Profile briefing error: %s", e)

    print()

    all_plans: list[dict] = []
    reports: list[str] = []
    log_entries: list[dict] = []

    for ticker in tickers:
        if ticker not in spot_data:
            print(f"  {ticker}: no live price available, skipping")
            continue

        sd = spot_data[ticker]
        spot = sd["price"]
        print(f"Fetching daily ranges for {ticker}...")
        daily_ranges = fetch_daily_ranges(ticker, days=10)

        print(f"Loading signals for {ticker}...")
        signal = load_signal_data(ticker)

        # Override direction if requested
        if args.direction != "auto":
            forced_dir = "LONG" if args.direction == "bull" else "SHORT"
            if forced_dir == "LONG":
                levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
            else:
                levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)

            warrant_results = evaluate_warrants(
                ticker, spot, levels, args.budget, fx_rate, direction=forced_dir,
            )
            warrant_results = [r for r in warrant_results if r["fill_prob"] >= FISHING_MIN_FILL_PROB]

            inst_info = _build_instrument_info(warrant_results, forced_dir)
            plans = [{
                "ticker": ticker, "spot": spot, "direction": forced_dir,
                "conviction": 1.0, "levels": warrant_results,
                "instrument": inst_info,
                "tp_cascade": list(FISHING_TP_CASCADE),
                "sl_cascade": list(FISHING_SL_CASCADE),
            }] if warrant_results else []
        else:
            print(f"Computing fishing plan for {ticker}...")
            plans = compute_fishing_plan(ticker, spot, signal, hours, daily_ranges,
                                         budget_sek=args.budget, fx_rate=fx_rate)

        all_plans.extend(plans)

        report = format_report(ticker, sd, signal, daily_ranges, plans,
                               hours, max_levels=args.max_levels)
        reports.append(report)

        # Build log entry
        log_entries.append({
            "ticker": ticker,
            "spot": spot,
            "hours": hours,
            "regime": signal["regime"],
            "rsi": signal["rsi"],
            "atr_pct": signal["atr_pct"],
            "plans": [{
                "direction": p["direction"],
                "conviction": p["conviction"],
                "top_levels": [{
                    "level": r["level"],
                    "fill_prob": r["fill_prob"],
                    "ev_sek": r["ev_sek"],
                    "warrant": r["warrant"],
                } for r in p["levels"][:args.max_levels]],
            } for p in plans],
        })

    if args.telegram:
        # Print Telegram-format summary for each ticker's plans
        for ticker in tickers:
            ticker_plans = [p for p in all_plans if p["ticker"] == ticker]
            if ticker_plans:
                print(format_telegram_plan(ticker_plans))
                print()
    else:
        print("\n" + "=" * 80)
        print(f"FISHING PLAN -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} CET")
        print(f"Budget: {args.budget:.0f} SEK per level | FX: {fx_rate:.2f}")
        print("=" * 80 + "\n")

        for report in reports:
            print(report)
            print()

    # Log
    log_entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "fin-fish",
        "budget_sek": args.budget,
        "fx_rate": fx_rate,
        "hours_remaining": hours,
        "metals": log_entries,
    }
    atomic_append_jsonl(FISH_LOG_PATH, log_entry)

    # --- Smart monitoring mode ---
    # Auto-detect active positions from Avanza or metals_positions_state
    # Start monitoring automatically when a position exists (no --monitor flag needed)
    should_monitor = args.monitor
    detected_position = None

    if not should_monitor and tickers:
        # Try to detect active positions
        try:
            from portfolio.avanza_session import get_positions
            positions = get_positions()
            silver_keywords = ("silver", "silv", "xag", "mini s silver", "mini l silver",
                               "bull silver", "bear silver")
            gold_keywords = ("guld", "gold", "xau", "bull guld", "bear guld")

            for pos in (positions or []):
                name = (pos.get("name") or "").lower()
                vol = pos.get("volume", 0)
                if vol <= 0:
                    continue

                for ticker in tickers:
                    keywords = silver_keywords if "XAG" in ticker else gold_keywords
                    if any(kw in name for kw in keywords):
                        detected_position = {
                            "ticker": ticker,
                            "name": pos.get("name", ""),
                            "volume": vol,
                            "value": pos.get("value", 0),
                            "avg_price": pos.get("averageAcquiredPrice", 0),
                            "last_price": pos.get("lastPrice", 0),
                            "is_short": any(k in name for k in ("bear", "mini s")),
                        }
                        should_monitor = True
                        print(f"\n  Active position detected: {pos.get('name')} ({vol}u)")
                        print("  Auto-starting smart monitor...\n")
                        break
                if detected_position:
                    break
        except Exception:
            # Avanza unavailable — check persisted state
            with suppress(Exception):
                pos_state = load_json(BASE_DIR / "data" / "metals_positions_state.json") or {}
                for key, pos in pos_state.items():
                    if pos.get("active") and any(t.lower().replace("-", "") in key.lower()
                                                  for t in tickers):
                        detected_position = {
                            "ticker": tickers[0],
                            "name": key,
                            "volume": pos.get("units", 0),
                            "value": 0,
                            "avg_price": pos.get("entry", 0),
                            "is_short": pos.get("direction", "").lower() == "short",
                        }
                        should_monitor = True
                        print(f"\n  Active position from state: {key}")
                        print("  Auto-starting smart monitor...\n")
                        break

    if should_monitor and tickers:
        ticker = tickers[0]
        if ticker in spot_data:
            spot = spot_data[ticker]["price"]
            entry = args.entry_price if args.entry_price > 0 else spot

            # Use detected position info if available
            cert_price = args.cert_price
            cert_units = args.cert_units
            cert_leverage = args.leverage

            if detected_position and cert_price == 0:
                cert_price = detected_position.get("avg_price", 0) or 0
                cert_units = detected_position.get("volume", 0)

            # Determine direction
            if args.direction != "auto":
                direction = "LONG" if args.direction == "bull" else "SHORT"
            elif detected_position:
                direction = "SHORT" if detected_position.get("is_short") else "LONG"
            elif ticker in preflight_results:
                pf = preflight_results[ticker]
                direction = "LONG" if pf["bull_score"] > pf["bear_score"] else "SHORT"
            else:
                direction = "SHORT"

            entry_conviction = 50
            if ticker in preflight_results:
                pf = preflight_results[ticker]
                entry_conviction = pf["bull_score"] if direction == "LONG" else pf["bear_score"]

            try:
                from portfolio.fish_monitor_smart import SmartFishMonitor
                monitor = SmartFishMonitor(
                    ticker=ticker,
                    entry_price=entry,
                    direction=direction,
                    entry_conviction=entry_conviction,
                    cert_entry_price=cert_price,
                    cert_units=cert_units,
                    cert_leverage=cert_leverage,
                )
                monitor.run()
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/price_targets.py | Select-Object -Index (300..380)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 603ms:
    """Main entry point: compute ranked price targets with fill probabilities.

    Parameters
    ----------
    extra : dict | None
        Enhanced signal indicator dicts (fibonacci_indicators, etc.)
        passed through to ``structural_levels``.
    regime : str
        Market regime string (e.g. "trending-up", "ranging").
        Used for regime-aware confidence adjustment.
    bb_squeeze : bool
        If True, Bollinger Band squeeze is active -- reduce confidence
        on all targets by 0.7x and flag ``squeeze_warning``.
    chronos_drift : float | None
        Annualised drift from Chronos 24h forecast.  When provided,
        blended 30/70 with the signal-based drift.
    """
    result: dict = {
        "ticker": ticker,
        "side": side,
        "price_usd": price_usd,
        "hours_remaining": hours_remaining,
        "extremes": {},
        "targets": [],
        "recommended": None,
    }

    if hours_remaining <= 0 or price_usd <= 0 or atr_pct <= 0:
        return result

    vol = volatility_from_atr(atr_pct)
    if side == "buy":
        drift = drift_from_probability(1.0 - p_up, vol)
    else:
        drift = drift_from_probability(p_up, vol)

    # Blend Chronos drift when available
    if chronos_drift is not None:
        drift = 0.7 * drift + 0.3 * chronos_drift

    # Structural levels (enriched with extra signal indicators)
    levels = structural_levels(price_usd, indicators, extra=extra)

    # Running extremes
    extremes = running_extremes(price_usd, vol, drift, hours_remaining,
                                side=side, n_paths=n_paths, is_24h=is_24h)
    result["extremes"] = extremes

    # Build candidate targets
    candidates: list[tuple[float, str]] = []

    # MC quantiles
    for pkey in ("p25", "p50", "p75"):
        val = extremes.get(pkey)
        if val is not None:
            candidates.append((val, f"mc_{pkey}"))

    # Structural levels
    for label, val in levels.items():
        if side == "sell" and val > price_usd or side == "buy" and val < price_usd:
            candidates.append((val, label))

    # Fixed offsets
    offsets = [0.005, 0.01, 0.02]
    for off in offsets:
        pct_label = f"{off*100:.1f}%"
        if side == "sell":
            candidates.append((price_usd * (1 + off), f"+{pct_label}"))
        else:
            candidates.append((price_usd * (1 - off), f"-{pct_label}"))

    # Deduplicate (within 0.01% of each other)
    candidates.sort(key=lambda c: c[0])
    deduped: list[tuple[float, str]] = []
    for price_c, label_c in candidates:
        if deduped and abs(price_c - deduped[-1][0]) / max(price_usd, 1e-9) < 0.0001:
            continue
        deduped.append((price_c, label_c))

    # Compute fill probability and EV for each candidate
    min_fill = 0.05

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/metals_orderbook.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 654ms:
"""Binance FAPI order book depth and recent trades for metals.

Fetches L2 snapshots and trade ticks for XAUUSDT / XAGUSDT.
Data feeds into microstructure feature computations (OFI, depth
imbalance, VPIN, spread metrics).

Uses the same rate-limiting and caching patterns as futures_data.py.
"""
from __future__ import annotations

import logging
import time
from functools import wraps

from portfolio.api_utils import BINANCE_FAPI_BASE
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _binance_limiter, _cached

logger = logging.getLogger("portfolio.metals_orderbook")

SYMBOL_MAP = {
    "XAU-USD": "XAUUSDT",
    "XAG-USD": "XAGUSDT",
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

_DEPTH_TTL = 10
_TRADES_TTL = 10


def _fetch_fapi_json(url, params=None, timeout=10):
    """Fetch JSON from Binance FAPI with rate limiting and retry."""
    _binance_limiter.wait()
    return fetch_json(url, params=params, timeout=timeout, label="metals_orderbook")


def _nocache(func):
    """Mark a function so tests can call func.__wrapped__ to bypass _cached."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


@_nocache
def get_orderbook_depth(ticker: str, limit: int = 20) -> dict | None:
    """Fetch order book depth snapshot from Binance FAPI.

    Returns dict with bids, asks (as [[price, qty], ...] floats), best_bid, best_ask,
    mid_price, spread, spread_bps. None on failure.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_fapi_json(
            f"{BINANCE_FAPI_BASE}/depth",
            params={"symbol": symbol, "limit": limit},
        )
        if data is None or "bids" not in data or "asks" not in data:
            return None
        bids = [[float(p), float(q)] for p, q in data["bids"]]
        asks = [[float(p), float(q)] for p, q in data["asks"]]
        if not bids or not asks:
            return None
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread": spread,
            "spread_bps": (spread / mid) * 10000 if mid > 0 else 0.0,
            "bid_depth_total": sum(q for _, q in bids),
            "ask_depth_total": sum(q for _, q in asks),
            "ts": int(time.time() * 1000),
        }

    return _cached(f"depth_{ticker}_{limit}", _DEPTH_TTL, _fetch)


@_nocache
def get_recent_trades(ticker: str, limit: int = 100) -> list[dict] | None:
    """Fetch recent trades from Binance FAPI.

    Each trade includes a sign: +1 for buyer-initiated (taker buy),
    -1 for seller-initiated (taker sell). isBuyerMaker=True means the maker
    was the buyer, so the taker (aggressor) was the seller -> sign = -1.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_fapi_json(
            f"{BINANCE_FAPI_BASE}/trades",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "id": d["id"],
                "price": float(d["price"]),
                "qty": float(d["qty"]),
                "time": d["time"],
                "is_buyer_maker": d.get("isBuyerMaker", False),
                "sign": -1 if d.get("isBuyerMaker", False) else 1,
            }
            for d in data
        ]

    return _cached(f"trades_{ticker}_{limit}", _TRADES_TTL, _fetch)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/microstructure.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 763ms:
"""Microstructure feature computations for short-term metals prediction.

Implements order-flow and market-microstructure metrics from academic literature.
All functions are pure — they take raw data and return numeric features.

Key features:
    - Depth Imbalance: ln(V_bid) - ln(V_ask)  [Lipton et al.]
    - Trade Flow Imbalance: signed volume ratio
    - VPIN: Volume-synchronized probability of informed trading
    - OFI: Order Flow Imbalance from quote changes  [Cont et al. 2014]
    - Spread Z-Score: current spread vs rolling distribution
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger("portfolio.microstructure")


def depth_imbalance(depth: dict, levels: int | None = None) -> float:
    """Log ratio of bid vs ask volume: F_t = ln(V_bid) - ln(V_ask).

    Positive → bid-heavy (buying pressure).
    Negative → ask-heavy (selling pressure).
    """
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if levels is not None:
        bids = bids[:levels]
        asks = asks[:levels]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    if bid_vol <= 0 or ask_vol <= 0:
        return 0.0
    return math.log(bid_vol) - math.log(ask_vol)


def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
    """Compute signed volume imbalance from recent trades.
    Each trade dict must have 'qty' (float) and 'sign' (+1 buyer, -1 seller).
    """
    if not trades:
        return None
    buy_vol = sum(t["qty"] for t in trades if t["sign"] == 1)
    sell_vol = sum(t["qty"] for t in trades if t["sign"] == -1)
    total_vol = buy_vol + sell_vol
    signed_vol = buy_vol - sell_vol
    imbalance = signed_vol / total_vol if total_vol > 0 else 0.0
    return {
        "signed_volume": signed_vol,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "imbalance_ratio": imbalance,
        "trade_count": len(trades),
        "buy_count": sum(1 for t in trades if t["sign"] == 1),
        "sell_count": sum(1 for t in trades if t["sign"] == -1),
    }


def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
    """VPIN: average absolute buy-sell imbalance per volume bucket.

    High VPIN (>0.6) → toxic flow, likely informed trading.
    Low VPIN (<0.3) → balanced, uninformed flow.
    """
    if len(trades) < n_buckets:
        return None
    total_vol = sum(t["qty"] for t in trades)
    if total_vol <= 0:
        return None
    bucket_size = total_vol / n_buckets

    imbalances = []
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_vol = 0.0

    for t in trades:
        qty = t["qty"]
        remaining = qty
        while remaining > 0:
            space = bucket_size - bucket_vol
            fill = min(remaining, space)
            if t["sign"] == 1:
                bucket_buy += fill
            else:
                bucket_sell += fill
            bucket_vol += fill
            remaining -= fill
            if bucket_vol >= bucket_size - 1e-12:
                if bucket_buy + bucket_sell > 0:
                    imbalances.append(
                        abs(bucket_buy - bucket_sell) / (bucket_buy + bucket_sell)
                    )
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_vol = 0.0

    if not imbalances:
        return None
    return float(np.mean(imbalances))


def compute_ofi(snapshots: list[dict]) -> float:
    """Order Flow Imbalance from consecutive order book snapshots.

    Implements the Cont et al. (2014) OFI formula.
    Positive OFI → net buying pressure.
    Negative OFI → net selling pressure.
    """
    if len(snapshots) < 2:
        return 0.0

    total_ofi = 0.0
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]

        prev_bid = prev["best_bid"]
        curr_bid = curr["best_bid"]
        prev_bid_vol = prev["bids"][0][1] if prev["bids"] else 0.0
        curr_bid_vol = curr["bids"][0][1] if curr["bids"] else 0.0

        if curr_bid > prev_bid:
            delta_bid = curr_bid_vol
        elif curr_bid == prev_bid:
            delta_bid = curr_bid_vol - prev_bid_vol
        else:
            delta_bid = -prev_bid_vol

        prev_ask = prev["best_ask"]
        curr_ask = curr["best_ask"]
        prev_ask_vol = prev["asks"][0][1] if prev["asks"] else 0.0
        curr_ask_vol = curr["asks"][0][1] if curr["asks"] else 0.0

        if curr_ask < prev_ask:
            delta_ask = curr_ask_vol
        elif curr_ask == prev_ask:
            delta_ask = curr_ask_vol - prev_ask_vol
        else:
            delta_ask = -prev_ask_vol

        total_ofi += delta_bid - delta_ask

    return total_ofi


def spread_zscore(spread_history: list[float], min_samples: int = 5) -> float | None:
    """Z-score of current spread vs recent history."""
    if len(spread_history) < min_samples:
        return None
    arr = np.array(spread_history, dtype=float)
    mean = arr[:-1].mean()
    std = arr[:-1].std()
    if std < 1e-12:
        # Zero variance: if current matches mean it's normal (0.0),
        # otherwise it's an extreme outlier — return large signed value.
        diff = arr[-1] - mean
        if abs(diff) < 1e-12:
            return 0.0
        return float(np.sign(diff) * 10.0)
    return float((arr[-1] - mean) / std)


# ---------------------------------------------------------------------------
# Trade-Through Detection (approximate)
# ---------------------------------------------------------------------------

def detect_trade_throughs(trades: list[dict], threshold_bps: float = 5.0) -> dict:
    """Detect trade-throughs: trades that jump across multiple price levels.

    A trade-through occurs when a market order is large enough to consume
    multiple levels of the order book, causing the execution price to jump
    significantly from the previous trade.  We approximate this from
    the trades list by detecting price gaps > threshold between consecutive
    trades in the same direction.

    Args:
        trades: List of trade dicts with 'price', 'qty', 'sign'.
        threshold_bps: Minimum price jump in basis points to count as
                       trade-through (default 5 bps = 0.05%).

    Returns:
        Dict with buy_throughs, sell_throughs (counts), total_volume_throughs,
        and max_gap_bps.  Returns zeros if insufficient trades.
    """
    if len(trades) < 2:
        return {
            "buy_throughs": 0,
            "sell_throughs": 0,
            "total_throughs": 0,
            "through_volume": 0.0,
            "max_gap_bps": 0.0,
        }

    buy_throughs = 0
    sell_throughs = 0
    through_volume = 0.0
    max_gap_bps = 0.0

    for i in range(1, len(trades)):
        prev = trades[i - 1]
        curr = trades[i]
        mid_price = (prev["price"] + curr["price"]) / 2.0
        if mid_price <= 0:
            continue
        gap_bps = abs(curr["price"] - prev["price"]) / mid_price * 10000

        if gap_bps >= threshold_bps and curr["sign"] == prev["sign"]:
            # Same-direction large gap = likely trade-through
            if curr["sign"] == 1:
                buy_throughs += 1
            else:
                sell_throughs += 1
            through_volume += curr["qty"]
            max_gap_bps = max(max_gap_bps, gap_bps)

    return {
        "buy_throughs": buy_throughs,
        "sell_throughs": sell_throughs,
        "total_throughs": buy_throughs + sell_throughs,
        "through_volume": round(through_volume, 4),
        "max_gap_bps": round(max_gap_bps, 2),
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/microstructure_state.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 685ms:
"""Microstructure snapshot accumulator for rolling OFI and spread history.

Called each cycle by metals_loop.py to build order book snapshot history.
The orderbook_flow signal reads the accumulated OFI and spread z-score
from the persisted state.

State is kept in memory (ring buffer) and persisted to
data/microstructure_state.json for cross-process access.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.microstructure import compute_ofi, spread_zscore

logger = logging.getLogger("portfolio.microstructure_state")

_BASE_DIR = Path(__file__).resolve().parent.parent
_STATE_FILE = _BASE_DIR / "data" / "microstructure_state.json"
_MAX_SNAPSHOTS = 60  # ~30-60 min at 30-60s intervals
_MIN_SNAPSHOTS_FOR_OFI = 3
_MIN_SPREADS_FOR_ZSCORE = 10
_MIN_OFI_HISTORY_FOR_ZSCORE = 10
_MAX_OFI_HISTORY = 120  # ~2h of OFI readings for z-score normalization

# Multi-scale OFI windows (snapshot counts)
_OFI_WINDOW_FAST = 5   # ~5 min
_OFI_WINDOW_MEDIUM = 15  # ~15 min
# slow = all snapshots (full buffer)

# In-memory ring buffers per ticker.
# Protected by _buffer_lock — metals_loop fast-tick (10s) and main cycle
# (60s) can both call accumulate_snapshot / get_state concurrently.
_buffer_lock = threading.Lock()
_snapshot_buffers: dict[str, deque] = {}
_spread_buffers: dict[str, deque] = {}
_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score


def _ensure_buffer(ticker: str) -> None:
    """Initialize ring buffers for a ticker if needed."""
    if ticker not in _snapshot_buffers:
        _snapshot_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
    if ticker not in _spread_buffers:
        _spread_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
    if ticker not in _ofi_history:
        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)


def accumulate_snapshot(ticker: str, depth: dict) -> None:
    """Add an order book snapshot to the rolling buffer.

    Args:
        ticker: Canonical ticker (e.g. "XAG-USD")
        depth: Order book depth dict from metals_orderbook.get_orderbook_depth()
               Must have: best_bid, best_ask, bids, asks, spread
    """
    if depth is None:
        return
    snapshot = {
        "best_bid": depth["best_bid"],
        "best_ask": depth["best_ask"],
        "bids": depth["bids"][:5],   # keep top 5 levels only
        "asks": depth["asks"][:5],
        "ts": depth.get("ts", int(time.time() * 1000)),
    }
    with _buffer_lock:
        _ensure_buffer(ticker)
        _snapshot_buffers[ticker].append(snapshot)
        _spread_buffers[ticker].append(depth["spread"])


def get_rolling_ofi(ticker: str) -> float:
    """Compute OFI from accumulated snapshots for a ticker.

    Returns cumulative OFI over the last N snapshots.
    Returns 0.0 if insufficient history.
    """
    with _buffer_lock:
        _ensure_buffer(ticker)
        snapshots = list(_snapshot_buffers[ticker])
    if len(snapshots) < _MIN_SNAPSHOTS_FOR_OFI:
        return 0.0
    return compute_ofi(snapshots)


def record_ofi(ticker: str, ofi_val: float) -> None:
    """Record an OFI value for z-score history tracking.

    Called once per cycle from get_microstructure_state to avoid
    double-appending if get_rolling_ofi is called multiple times.
    """
    with _buffer_lock:
        _ensure_buffer(ticker)
        _ofi_history[ticker].append(ofi_val)


def get_ofi_zscore(ticker: str, current_ofi: float | None = None) -> float:
    """Z-score of OFI relative to its own rolling distribution.

    Scores ``current_ofi`` (or the latest recorded value) against the
    historical distribution WITHOUT including the current value.
    This prevents self-contamination that compresses z-scores toward zero.

    Returns 0.0 if insufficient history.
    """
    with _buffer_lock:
        _ensure_buffer(ticker)
        history = list(_ofi_history[ticker])
    if len(history) < _MIN_OFI_HISTORY_FOR_ZSCORE:
        return 0.0
    import numpy as np
    arr = np.array(history, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    value = current_ofi if current_ofi is not None else arr[-1]
    return float((value - mean) / std)


def get_multiscale_ofi(ticker: str) -> dict:
    """Compute OFI at 3 time scales: fast (~5min), medium (~15min), slow (full).

    Returns dict with ofi_fast, ofi_medium, ofi_slow, and flow_acceleration
    (fast z-score minus slow z-score — positive = accelerating buying).
    """
    with _buffer_lock:
        _ensure_buffer(ticker)
        snapshots = list(_snapshot_buffers[ticker])
    n = len(snapshots)

    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
    ofi_medium = compute_ofi(snapshots[-_OFI_WINDOW_MEDIUM:]) if n >= _OFI_WINDOW_MEDIUM else ofi_slow
    ofi_fast = compute_ofi(snapshots[-_OFI_WINDOW_FAST:]) if n >= _OFI_WINDOW_FAST else ofi_medium

    # Flow acceleration: compare fast to slow (normalized by snapshot counts).
    # Only meaningful when we have enough snapshots for distinct windows;
    # during warmup (n < _OFI_WINDOW_FAST), fast==slow so acceleration
    # would produce misleading non-zero values (code review MEDIUM-2).
    if n >= _OFI_WINDOW_FAST:
        fast_per_snap = ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)
        slow_per_snap = ofi_slow / max(n - 1, 1) if n > 1 else 0.0
        flow_acceleration = fast_per_snap - slow_per_snap
    else:
        flow_acceleration = 0.0

    return {
        "ofi_fast": round(ofi_fast, 4),
        "ofi_medium": round(ofi_medium, 4),
        "ofi_slow": round(ofi_slow, 4),
        "flow_acceleration": round(flow_acceleration, 4),
    }


def get_spread_zscore(ticker: str) -> float | None:
    """Compute spread z-score from accumulated spread history.

    Returns z-score of current spread vs recent history.
    Returns None if insufficient data.
    """
    with _buffer_lock:
        _ensure_buffer(ticker)
        spreads = list(_spread_buffers[ticker])
    if len(spreads) < _MIN_SPREADS_FOR_ZSCORE:
        return None
    return spread_zscore(spreads)


def get_microstructure_state(ticker: str) -> dict:
    """Get current accumulated microstructure state for a ticker.

    Returns dict with ofi, ofi_zscore, multiscale OFI, and spread_zscore.
    """
    ofi = get_rolling_ofi(ticker)
    # Compute z-score BEFORE appending current value to history —
    # scoring against a population that includes itself compresses
    # the z-score toward zero (code review finding CRITICAL-1).
    ofi_z = get_ofi_zscore(ticker, current_ofi=ofi)
    record_ofi(ticker, ofi)
    sz = get_spread_zscore(ticker)
    ms_ofi = get_multiscale_ofi(ticker)
    with _buffer_lock:
        _ensure_buffer(ticker)
        snap_count = len(_snapshot_buffers[ticker])
        spread_count = len(_spread_buffers[ticker])
    return {
        "ofi": ofi,
        "ofi_zscore": ofi_z,
        "ofi_fast": ms_ofi["ofi_fast"],
        "ofi_medium": ms_ofi["ofi_medium"],
        "ofi_slow": ms_ofi["ofi_slow"],
        "flow_acceleration": ms_ofi["flow_acceleration"],
        "spread_zscore": sz if sz is not None else 0.0,
        "snapshot_count": snap_count,
        "spread_count": spread_count,
    }


def persist_state() -> None:
    """Write current microstructure state to disk for cross-process access."""
    state = {}
    for ticker in _snapshot_buffers:
        ms = get_microstructure_state(ticker)
        ms["ts"] = int(time.time() * 1000)
        state[ticker] = ms
    if state:
        atomic_write_json(_STATE_FILE, state)


def load_persisted_state(ticker: str) -> dict | None:
    """Read persisted microstructure state for a ticker.

    Used by orderbook_flow signal when running in a different process
    from metals_loop.
    """
    data = load_json(_STATE_FILE)
    if not data or ticker not in data:
        return None
    entry = data[ticker]
    age_ms = int(time.time() * 1000) - entry.get("ts", 0)
    if age_ms > 120_000:  # stale if >2 minutes old
        return None
    return entry


def snapshot_count(ticker: str) -> int:
    """Return current snapshot buffer size for a ticker."""
    with _buffer_lock:
        _ensure_buffer(ticker)
        return len(_snapshot_buffers[ticker])

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/orb_postmortem.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 757ms:
"""
ORB Postmortem — End-of-day analysis comparing ORB predictions to actual results.

Tracks prediction accuracy over time, identifies which filters work best,
and generates actionable recommendations.

Usage:
    python -u portfolio/orb_postmortem.py
"""

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.orb_predictor import ORBPredictor, Prediction

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POSTMORTEM_PATH = DATA_DIR / "orb_postmortem.jsonl"
PREDICTIONS_TODAY_PATH = DATA_DIR / "orb_predictions_today.json"


@dataclass
class PostmortemResult:
    """Result of comparing one day's ORB prediction to actual outcome."""
    date: str
    # Predicted values
    predicted_high_conservative: float
    predicted_high_median: float
    predicted_high_aggressive: float
    predicted_low_conservative: float
    predicted_low_median: float
    predicted_low_aggressive: float
    morning_direction: str
    morning_range_pct: float
    sample_size: int
    filters_applied: list
    # Actual values
    actual_high: float
    actual_low: float
    # Errors
    high_error_abs: float       # actual_high - predicted_high_median
    high_error_pct: float       # error as % of predicted
    low_error_abs: float        # actual_low - predicted_low_median
    low_error_pct: float        # error as % of predicted
    # Target hit analysis
    high_within_conservative: bool  # actual_high <= predicted_high_aggressive
    high_within_aggressive: bool    # actual_high >= predicted_high_conservative
    low_within_conservative: bool   # actual_low >= predicted_low_aggressive
    low_within_aggressive: bool     # actual_low <= predicted_low_conservative
    # P&L simulation (if traded median targets)
    buy_target_hit: bool        # actual_low <= predicted_low_median
    sell_target_hit: bool       # actual_high >= predicted_high_median
    simulated_pnl_pct: float    # % P&L if bought at pred_low_med and sold at pred_high_med


def run_postmortem(prediction: Prediction, actual_high: float, actual_low: float) -> PostmortemResult:
    """Compare predicted vs actual highs/lows for a single day.

    Args:
        prediction: The day's ORB prediction
        actual_high: Actual day high from market data
        actual_low: Actual day low from market data

    Returns:
        PostmortemResult with errors, hit/miss analysis, and simulated P&L
    """
    # Errors (positive = actual exceeded prediction, negative = actual fell short)
    high_error_abs = actual_high - prediction.predicted_high_median
    high_error_pct = high_error_abs / prediction.predicted_high_median * 100

    low_error_abs = actual_low - prediction.predicted_low_median
    low_error_pct = low_error_abs / prediction.predicted_low_median * 100

    # Target hit analysis
    # "Within conservative" means actual stayed within the tighter bounds
    high_within_conservative = actual_high <= prediction.predicted_high_aggressive
    high_within_aggressive = actual_high >= prediction.predicted_high_conservative
    low_within_conservative = actual_low >= prediction.predicted_low_aggressive
    low_within_aggressive = actual_low <= prediction.predicted_low_conservative

    # Buy/sell target hit
    buy_target_hit = actual_low <= prediction.predicted_low_median
    sell_target_hit = actual_high >= prediction.predicted_high_median

    # Simulated P&L: if we bought at predicted low median and sold at predicted high median
    if buy_target_hit and sell_target_hit:
        # Both targets hit -- full predicted spread captured
        simulated_pnl_pct = (prediction.predicted_high_median - prediction.predicted_low_median) / prediction.predicted_low_median * 100
    elif buy_target_hit:
        # Bought at low target, but high target never reached -- use actual high as exit
        simulated_pnl_pct = (actual_high - prediction.predicted_low_median) / prediction.predicted_low_median * 100
    elif sell_target_hit:
        # Never got buy fill -- no trade
        simulated_pnl_pct = 0.0
    else:
        # Neither target hit -- no trade
        simulated_pnl_pct = 0.0

    return PostmortemResult(
        date=prediction.date,
        predicted_high_conservative=prediction.predicted_high_conservative,
        predicted_high_median=prediction.predicted_high_median,
        predicted_high_aggressive=prediction.predicted_high_aggressive,
        predicted_low_conservative=prediction.predicted_low_conservative,
        predicted_low_median=prediction.predicted_low_median,
        predicted_low_aggressive=prediction.predicted_low_aggressive,
        morning_direction=prediction.morning_direction,
        morning_range_pct=prediction.morning_range_pct,
        sample_size=prediction.sample_size,
        filters_applied=prediction.filters_applied,
        actual_high=actual_high,
        actual_low=actual_low,
        high_error_abs=round(high_error_abs, 4),
        high_error_pct=round(high_error_pct, 3),
        low_error_abs=round(low_error_abs, 4),
        low_error_pct=round(low_error_pct, 3),
        high_within_conservative=high_within_conservative,
        high_within_aggressive=high_within_aggressive,
        low_within_conservative=low_within_conservative,
        low_within_aggressive=low_within_aggressive,
        buy_target_hit=buy_target_hit,
        sell_target_hit=sell_target_hit,
        simulated_pnl_pct=round(simulated_pnl_pct, 4),
    )


def log_postmortem(result: PostmortemResult, filepath: str = str(POSTMORTEM_PATH)) -> None:
    """Append one JSON line per day to the postmortem log."""
    entry = asdict(result)
    entry["logged_at"] = datetime.now(UTC).isoformat()
    atomic_append_jsonl(filepath, entry)


def load_postmortem_history(filepath: str = str(POSTMORTEM_PATH)) -> list[PostmortemResult]:
    """Read all past postmortems and return as list of PostmortemResult."""
    path = Path(filepath)
    if not path.exists():
        return []

    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Remove non-dataclass fields
                data.pop("logged_at", None)
                results.append(PostmortemResult(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def format_lessons_learned(history: list[PostmortemResult]) -> str:
    """Analyze postmortem history for patterns and output recommendations."""
    if not history:
        return "No postmortem history available yet."

    lines = [f"=== ORB Lessons Learned ({len(history)} days) ===\n"]

    # Overall accuracy
    buy_hits = sum(1 for r in history if r.buy_target_hit)
    sell_hits = sum(1 for r in history if r.sell_target_hit)
    both_hits = sum(1 for r in history if r.buy_target_hit and r.sell_target_hit)
    lines.append(f"Buy target hit rate:  {buy_hits}/{len(history)} ({buy_hits/len(history)*100:.0f}%)")
    lines.append(f"Sell target hit rate: {sell_hits}/{len(history)} ({sell_hits/len(history)*100:.0f}%)")
    lines.append(f"Both targets hit:     {both_hits}/{len(history)} ({both_hits/len(history)*100:.0f}%)")

    # High prediction accuracy
    high_errors = [abs(r.high_error_pct) for r in history]
    low_errors = [abs(r.low_error_pct) for r in history]
    lines.append(f"\nHigh prediction error: median {statistics.median(high_errors):.2f}%, mean {statistics.mean(high_errors):.2f}%")
    lines.append(f"Low prediction error:  median {statistics.median(low_errors):.2f}%, mean {statistics.mean(low_errors):.2f}%")

    # Simulated P&L
    pnls = [r.simulated_pnl_pct for r in history]
    traded_pnls = [p for p in pnls if p != 0.0]
    lines.append(f"\nSimulated P&L (all days): total {sum(pnls):.3f}%, mean {statistics.mean(pnls):.3f}%")
    if traded_pnls:
        lines.append(f"Simulated P&L (traded days only): total {sum(traded_pnls):.3f}%, mean {statistics.mean(traded_pnls):.3f}%")
        lines.append(f"Trade days: {len(traded_pnls)}/{len(history)}")

    # Direction analysis
    up_days = [r for r in history if r.morning_direction == "up"]
    down_days = [r for r in history if r.morning_direction == "down"]

    if up_days:
        up_both = sum(1 for r in up_days if r.buy_target_hit and r.sell_target_hit)
        up_pnl = sum(r.simulated_pnl_pct for r in up_days)
        lines.append(f"\nUp mornings ({len(up_days)} days): both-hit {up_both/len(up_days)*100:.0f}%, total P&L {up_pnl:.3f}%")

    if down_days:
        down_both = sum(1 for r in down_days if r.buy_target_hit and r.sell_target_hit)
        down_pnl = sum(r.simulated_pnl_pct for r in down_days)
        lines.append(f"Down mornings ({len(down_days)} days): both-hit {down_both/len(down_days)*100:.0f}%, total P&L {down_pnl:.3f}%")

    # Range size analysis
    if len(history) >= 6:
        sorted_by_range = sorted(history, key=lambda r: r.morning_range_pct)
        half = len(sorted_by_range) // 2
        small_range = sorted_by_range[:half]
        large_range = sorted_by_range[half:]

        small_both = sum(1 for r in small_range if r.buy_target_hit and r.sell_target_hit)
        large_both = sum(1 for r in large_range if r.buy_target_hit and r.sell_target_hit)
        lines.append(f"\nSmall morning range ({len(small_range)} days): both-hit {small_both/len(small_range)*100:.0f}%")
        lines.append(f"Large morning range ({len(large_range)} days): both-hit {large_both/len(large_range)*100:.0f}%")

    # Recommendations
    lines.append("\n--- Recommendations ---")
    if len(history) < 10:
        lines.append("- Need more data (< 10 days). Keep tracking.")
    else:
        if buy_hits / len(history) < 0.5:
            lines.append("- Low buy target hit rate. Consider using conservative (25th pctl) instead of median for buy targets.")
        if sell_hits / len(history) < 0.5:
            lines.append("- Low sell target hit rate. Consider using conservative (25th pctl) instead of median for sell targets.")
        if both_hits / len(history) > 0.6:
            lines.append("- Good both-target hit rate. Median targets are reliable for this market.")
        if up_days and down_days:
            up_rate = sum(1 for r in up_days if r.buy_target_hit and r.sell_target_hit) / len(up_days)
            down_rate = sum(1 for r in down_days if r.buy_target_hit and r.sell_target_hit) / len(down_days)
            if up_rate > down_rate + 0.15:
                lines.append("- Up mornings significantly more predictable. Consider direction filter.")
            elif down_rate > up_rate + 0.15:
                lines.append("- Down mornings significantly more predictable. Consider direction filter.")

    return "\n".join(lines)


def generate_daily_report() -> PostmortemResult | None:
    """Run end-of-day postmortem for today.

    Reads today's prediction from orb_predictions_today.json,
    fetches actual day data from Binance, runs postmortem, logs it.

    Returns:
        PostmortemResult if successful, None otherwise
    """
    # Load today's prediction
    if not PREDICTIONS_TODAY_PATH.exists():
        print("No prediction found for today (data/orb_predictions_today.json missing)")
        return None

    with open(PREDICTIONS_TODAY_PATH, encoding="utf-8") as f:
        pred_data = json.load(f)

    # Remove non-Prediction fields
    pred_data.pop("generated_at", None)
    prediction = Prediction(**pred_data)

    print(f"Loaded prediction for {prediction.date}")
    print(f"  Morning: HIGH ${prediction.morning_high:.2f} LOW ${prediction.morning_low:.2f} DIR={prediction.morning_direction}")
    print(f"  Predicted HIGH (med): ${prediction.predicted_high_median:.2f}")
    print(f"  Predicted LOW  (med): ${prediction.predicted_low_median:.2f}")

    # Fetch actual day data from Binance
    predictor = ORBPredictor()
    print("\nFetching today's actual data from Binance...")
    try:
        klines = predictor.fetch_klines(num_batches=1, limit=200)
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None

    days = predictor.group_by_day(klines, weekdays_only=False)
    today_candles = days.get(prediction.date, [])

    if not today_candles:
        print(f"No candles found for {prediction.date}")
        return None

    # Filter to trading hours (08:00-22:00 UTC)
    day_candles = [
        c for c in today_candles
        if predictor.day_start_utc <= c["hour"] <= predictor.day_end_utc
    ]

    if len(day_candles) < 10:
        print(f"Insufficient day candles ({len(day_candles)}), market may still be open")
        return None

    actual_high = max(c["high"] for c in day_candles)
    actual_low = min(c["low"] for c in day_candles)

    print(f"  Actual: HIGH ${actual_high:.2f} LOW ${actual_low:.2f}")

    # Run postmortem
    result = run_postmortem(prediction, actual_high, actual_low)

    # Log it
    log_postmortem(result)
    print(f"\nPostmortem logged to {POSTMORTEM_PATH}")

    # Print result summary
    print(f"\n=== Postmortem for {result.date} ===")
    print(f"High: predicted ${result.predicted_high_median:.2f} vs actual ${result.actual_high:.2f} (error: {result.high_error_pct:+.2f}%)")
    print(f"Low:  predicted ${result.predicted_low_median:.2f} vs actual ${result.actual_low:.2f} (error: {result.low_error_pct:+.2f}%)")
    print(f"Buy target hit:  {'YES' if result.buy_target_hit else 'NO'}")
    print(f"Sell target hit: {'YES' if result.sell_target_hit else 'NO'}")
    print(f"Simulated P&L:   {result.simulated_pnl_pct:+.3f}%")

    # Load history and print lessons
    history = load_postmortem_history()
    if history:
        print(f"\n{format_lessons_learned(history)}")

    return result


if __name__ == "__main__":
    generate_daily_report()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/orb_predictor.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 733ms:
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

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

import requests

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
            ts = datetime.fromtimestamp(k[0] / 1000, tz=UTC)
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

    def calculate_morning_range(self, day_candles: list[dict]) -> MorningRange | None:
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

    def calculate_day_result(self, day_candles: list[dict]) -> DayResult | None:
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
    ) -> Prediction | None:
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
        current_warrant_price: float | None = None,
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

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Select-String -Path portfolio/*.py,data/*.py -Pattern "high_within_conservative|high_within_aggressive|low_within_conservative|low_within_aggressive" | ForEach-Object { "{0}:{1}:{2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 624ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:50:high_within_conservative: bool  # actual_high <= predicted_high_aggressive
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:51:high_within_aggressive: bool    # actual_high >= predicted_high_conservative
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:52:low_within_conservative: bool   # actual_low >= predicted_low_aggressive
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:53:low_within_aggressive: bool     # actual_low <= predicted_low_conservative
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:80:high_within_conservative = actual_high <= prediction.predicted_high_aggressive
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:81:high_within_aggressive = actual_high >= prediction.predicted_high_conservative
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:82:low_within_conservative = actual_low >= prediction.predicted_low_aggressive
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:83:low_within_aggressive = actual_low <= prediction.predicted_low_conservative
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:121:high_within_conservative=high_within_conservative,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:122:high_within_aggressive=high_within_aggressive,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:123:low_within_conservative=low_within_conservative,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_postmortem.py:124:low_within_aggressive=low_within_aggressive,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/metals_cross_assets.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 659ms:
"""Cross-asset data for metals prediction.

Fetches correlated markets that carry predictive information for
1-3h gold and silver moves:
    - Copper (HG=F): industrial demand proxy, correlated with silver
    - GVZ: CBOE Gold ETF Volatility Index (implied vol for gold)
    - SPY: S&P 500 ETF (risk-on/risk-off gauge)
    - Gold/Silver ratio: mean-reverting ratio, extreme readings signal

All data fetched via yfinance with caching to avoid rate limits.

2026-04-13: Added intraday (60m bar) fetchers next to the existing daily
ones after 4,916-sample measurement showed metals_cross_asset at 29.1%
on XAG 3h — root cause was 5-day lookbacks evaluated against 3h outcomes
(see docs/AVANZA_RESILIENCE_PLAN.md follow-up). Daily fetchers preserved
for longer-horizon callers; the metals_cross_asset signal switched to
intraday by default.
"""
from __future__ import annotations

import logging
from functools import wraps

import pandas as pd

from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.metals_cross_assets")

_CROSS_TTL = 300
_GVZ_TTL = 600
# Intraday TTL is shorter — 60m bars refresh at the start of each hour,
# and we want to re-query shortly after the bar closes to pick up the new row.
_CROSS_INTRADAY_TTL = 180


def _yf_download(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV OHLCV bars. Routed via ``portfolio.price_source``.

    2026-04-14: no longer pinned to yfinance. The router dispatches
    commodity futures (HG=F, CL=F) to Binance FAPI for 7.7s-fresh data,
    stocks/ETFs (SPY, USO) to Alpaca, and falls back to yfinance only
    for tickers with no live alternative (^GVZ). Returns DataFrame with
    capitalized column names for backward compatibility with callers
    in this module that reference ``df["Close"]``.
    """
    # Rough period-to-limit mapping — price_source/Binance/Alpaca use row
    # limits while yfinance uses period strings. Slight over-fetch is OK.
    _limit_map = {
        "1d": 10, "5d": 120, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730,
    }
    limit = _limit_map.get(period, 90)

    try:
        from portfolio.price_source import fetch_klines

        df = fetch_klines(ticker, interval=interval, limit=limit, period=period)
    except Exception as e:
        logger.warning("price_source fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Callers in this module use capitalized column names (legacy
    # yfinance convention). Backends normalize to lowercase; re-capitalize
    # here so downstream getters ``df["Close"]`` keep working.
    rename = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _nocache(func):
    """Mark function so tests can bypass _cached via func.__wrapped__."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


def _pct_change(series: pd.Series, periods: int) -> float:
    """Percentage change over N periods, returns NaN on insufficient data."""
    if len(series) < periods + 1:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


@_nocache
def get_copper_data() -> dict | None:
    """Copper futures (HG=F) price and momentum."""
    def _fetch():
        df = _yf_download("HG=F", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 20:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
            "sma20": float(close.rolling(20).mean().iloc[-1]),
            "vs_sma20_pct": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
        }
    return _cached("cross_copper", _CROSS_TTL, _fetch)


@_nocache
def get_gvz() -> dict | None:
    """CBOE Gold ETF Volatility Index (^GVZ)."""
    def _fetch():
        df = _yf_download("^GVZ", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 10:
            return None
        level = float(close.iloc[-1])
        mean20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
        std20 = float(close.rolling(20, min_periods=10).std().iloc[-1])
        zscore = (level - mean20) / std20 if std20 > 0.01 else 0.0
        return {
            "level": level,
            "change_1d_pct": _pct_change(close, 1),
            "sma20": mean20,
            "zscore": zscore,
        }
    return _cached("cross_gvz", _GVZ_TTL, _fetch)


@_nocache
def get_gold_silver_ratio() -> dict | None:
    """Gold/Silver price ratio and deviation from mean."""
    def _fetch():
        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
        if gold_df.empty or silver_df.empty:
            return None
        gold_close = gold_df["Close"].dropna()
        silver_close = silver_df["Close"].dropna()
        if len(gold_close) < 20 or len(silver_close) < 20:
            return None
        common = gold_close.index.intersection(silver_close.index)
        if len(common) < 20:
            return None
        g = gold_close.loc[common]
        s = silver_close.loc[common]
        ratio = g / s
        current = float(ratio.iloc[-1])
        sma20 = float(ratio.rolling(20).mean().iloc[-1])
        std20 = float(ratio.rolling(20).std().iloc[-1])
        zscore = (current - sma20) / std20 if std20 > 0.01 else 0.0
        return {
            "ratio": current,
            "sma20": sma20,
            "zscore": zscore,
            "change_5d_pct": _pct_change(ratio, 5),
        }
    return _cached("cross_gs_ratio", _CROSS_TTL, _fetch)


@_nocache
def get_oil_data() -> dict | None:
    """WTI Crude Oil futures (CL=F) price and momentum."""
    def _fetch():
        df = _yf_download("CL=F", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 10:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
        }
    return _cached("cross_oil", _CROSS_TTL, _fetch)


@_nocache
def get_spy_return() -> dict | None:
    """S&P 500 ETF (SPY) recent returns for risk-on/risk-off."""
    def _fetch():
        df = _yf_download("SPY", period="1mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 5:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
        }
    return _cached("cross_spy", _CROSS_TTL, _fetch)


def get_all_cross_asset_data() -> dict:
    """Fetch all cross-asset features in one call (daily bars)."""
    return {
        "copper": get_copper_data(),
        "gvz": get_gvz(),
        "gold_silver_ratio": get_gold_silver_ratio(),
        "spy": get_spy_return(),
        "oil": get_oil_data(),
    }


# --- Intraday variants (60m bars, for 1-3h prediction horizons) ---
#
# yfinance 60m interval supports up to 730 days of history. We use 5d
# period which yields ~35 hourly bars — enough for 3h change (3 bars) and
# intraday rolling stats. On weekends/holidays the last ~2 days of bars
# may be sparse; `_pct_change` returns NaN and signal votes HOLD.


@_nocache
def get_copper_intraday() -> dict | None:
    """Copper 60m bars. Exposes change_1h_pct + change_3h_pct."""
    def _fetch():
        df = _yf_download("HG=F", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_copper_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_gold_silver_ratio_intraday() -> dict | None:
    """Gold/Silver ratio 60m bars. Exposes ratio_change_3h_pct."""
    def _fetch():
        gold_df = _yf_download("GC=F", period="5d", interval="60m")
        silver_df = _yf_download("SI=F", period="5d", interval="60m")
        if gold_df.empty or silver_df.empty:
            return None
        gold_close = gold_df["Close"].dropna()
        silver_close = silver_df["Close"].dropna()
        if len(gold_close) < 4 or len(silver_close) < 4:
            return None
        common = gold_close.index.intersection(silver_close.index)
        if len(common) < 4:
            return None
        ratio = gold_close.loc[common] / silver_close.loc[common]
        return {
            "ratio": float(ratio.iloc[-1]),
            "change_1h_pct": _pct_change(ratio, 1),
            "change_3h_pct": _pct_change(ratio, 3),
        }
    return _cached("cross_gs_ratio_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_oil_intraday() -> dict | None:
    """WTI crude 60m bars."""
    def _fetch():
        df = _yf_download("CL=F", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_oil_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_spy_intraday() -> dict | None:
    """SPY 60m bars — captures intraday risk-on/risk-off."""
    def _fetch():
        df = _yf_download("SPY", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_spy_intraday", _CROSS_INTRADAY_TTL, _fetch)


def get_all_cross_asset_intraday() -> dict:
    """Fetch all intraday (60m) cross-asset features in one call.

    GVZ is intentionally absent — it's a daily-published index with no
    intraday bars. Callers should still read `get_gvz()` for GVZ context.
    """
    return {
        "copper": get_copper_intraday(),
        "gold_silver_ratio": get_gold_silver_ratio_intraday(),
        "spy": get_spy_intraday(),
        "oil": get_oil_intraday(),
    }

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern \"\\bpass\\b\" | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 605ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:735:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:96:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:57:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:99:# Fetch all shared market data in one pass
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:543:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:138:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:935:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:952:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:964:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:971:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:977:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1437:pass  # outer log gate inside the helper handles repeated failures
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1759:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2028:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4437:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4590:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5356:pass
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7368:# Pass POSITIONS so cancel_spike_orders can restore

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern \"\\[\\\"instrument\\\"\\].*leverage|instrument\\].get\\(\\\"leverage\\\"|instrument.*\\[\\\"leverage\\\"\\]\" | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
2026-05-08T15:31:31.143827Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mSelect-String: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mSelect-String -Path portfolio/*.py,data/*.py -Pattern "\[\"instrument[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mA positional parameter cannot be found that accepts argument 'instrument\\].*leverage|instrument\].get\(\leverage\|instrument.*\[\leverage\\]'.[0m

 exited 1 in 786ms:
[31;1mSelect-String: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mSelect-String -Path portfolio/*.py,data/*.py -Pattern "\[\"instrument[0m …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mA positional parameter cannot be found that accepts argument 'instrument\\].*leverage|instrument\].get\(\leverage\|instrument.*\[\leverage\\]'.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'get\\(\"leverage\"' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 641ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:975:leverage = plan["instrument"].get("leverage", 1)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1059:inst_lev = instrument.get("leverage", 1)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:176:"leverage": float(snapshot.get("leverage") or 1.0),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:189:leverage = float(snapshot.get("leverage") or 1.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:356:leverage = float(snapshot.get("leverage") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:398:leverage = float(snapshot.get("leverage") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:596:"leverage": float(snapshot.get("leverage") or 0.0),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:608:"leverage": float(_extract_value(indicators.get("leverage")) or snapshot.get("leverage") or 0.0),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe.py:156:leverage = float(_value(indicators.get("leverage")) or 1.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1455:leverage = silver_pos.get("leverage", 4.76)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1969:"leverage": wv.get("leverage", 5.0),
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1998:"leverage": _wv.get("leverage") or 1.0,
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4475:entry["current_leverage"] = p.get("leverage") or winfo.get("leverage")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7505:lev = _spos.get("leverage") or 1.0

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/silver_precompute.py' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 637ms:
"""Silver Deep Context Precomputer — delegation wrapper.

Delegates to the consolidated metals_precompute module.
Kept for backwards compatibility (manual runs, existing references).

Run: .venv/Scripts/python.exe portfolio/silver_precompute.py
Or:  .venv/Scripts/python.exe portfolio/metals_precompute.py  (preferred)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import load_json


def maybe_precompute_silver(config=None):
    """Delegate to consolidated metals precompute."""
    from portfolio.metals_precompute import maybe_precompute_metals
    return maybe_precompute_metals(config)


def precompute(config=None):
    """Delegate to consolidated metals precompute."""
    from portfolio.metals_precompute import precompute as metals_precompute
    return metals_precompute(config)


if __name__ == "__main__":
    from portfolio.metals_precompute import precompute as metals_precompute

    config = load_json("config.json")
    metals_precompute(config)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/iskbets.py -TotalCount 260' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 582ms:
"""ISKBETS — Intraday quick-gamble mode.

Scans for entry conditions every 60s cycle, sends Telegram alerts, monitors
exit conditions using an ATR-based ladder. User trades manually on Avanza
and confirms via Telegram replies.
"""

import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("portfolio.iskbets")

from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, get_alpaca_headers
from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.http_retry import fetch_with_retry
from portfolio.message_store import send_or_store
from portfolio.shared_state import _cached

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_FILE = DATA_DIR / "iskbets_config.json"
STATE_FILE = DATA_DIR / "iskbets_state.json"

# Ticker → source mapping (imported from shared tickers module)
import contextlib

from portfolio.tickers import (
    SYMBOLS as TICKER_SOURCES,
)

# ── State I/O ────────────────────────────────────────────────────────────


def _load_config():
    """Load per-session ISKBETS config. Returns dict or None if disabled/expired."""
    cfg = load_json(CONFIG_FILE)
    if cfg is None:
        return None
    if not cfg.get("enabled", False):
        return None
    # Check expiry
    expiry = cfg.get("expiry")
    if expiry:
        try:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if datetime.now(UTC) > exp_dt:
                # Auto-disable
                cfg["enabled"] = False
                _save_config(cfg)
                logger.info("ISKBETS: Session expired, auto-disabled")
                return None
        except (ValueError, TypeError):
            pass
    return cfg


def _save_config(cfg):
    """Atomic write of iskbets config."""
    atomic_write_json(CONFIG_FILE, cfg)


def _load_state():
    """Load ISKBETS runtime state."""
    result = load_json(STATE_FILE)
    if result is not None:
        return result
    return {"active_position": None, "trade_history": []}


def _save_state(state):
    """Atomic write of ISKBETS state."""
    atomic_write_json(STATE_FILE, state)


# ── Telegram ─────────────────────────────────────────────────────────────


def _send_telegram(msg, config):
    """Send a Telegram message via central routing (category: iskbets → always sent)."""
    send_or_store(msg, config, category="iskbets")


def _log_telegram(msg):
    """Append message to telegram log (legacy — new messages use send_or_store)."""
    from portfolio.message_store import log_message
    log_message(msg, category="iskbets", sent=False)


# ── ATR Computation ──────────────────────────────────────────────────────


def compute_atr_15m(ticker, config):
    """Fetch 15-min candles and compute ATR(14). Returns ATR value in USD.

    Results are cached for 5 minutes via shared_state._cached() to avoid
    redundant API calls when called per ticker per entry check.
    """
    return _cached(f"atr_15m_{ticker}", 300, _compute_atr_15m_impl, ticker, config)


def _compute_atr_15m_impl(ticker, config):
    """Implementation: fetch 15-min candles and compute ATR(14).

    Delegates kline fetching to data_collector._fetch_klines to avoid
    duplicating Binance/Alpaca/yfinance API code.
    """
    from portfolio.data_collector import _fetch_klines

    source = TICKER_SOURCES.get(ticker)
    if not source:
        raise ValueError(f"Unknown ticker: {ticker}")

    df = _fetch_klines(source, interval="15m", limit=20)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    return float(atr)


# ── Entry Evaluation ─────────────────────────────────────────────────────


def _evaluate_entry(ticker, signals, prices_usd, tf_data, iskbets_cfg, app_config):
    """Check if entry conditions are met for a ticker.

    Returns (should_enter, conditions_list).
    Gate 1: ≥min_bigbet_conditions of 6 bigbet conditions.
    Gate 2: ≥min_buy_votes from main signal grid.
    Time gate: No entry after entry_cutoff_et.
    """
    # Time gate
    cutoff_str = iskbets_cfg.get("entry_cutoff_et", "14:30")
    if not _before_cutoff(cutoff_str):
        return False, ["Past entry cutoff"]

    min_bigbet = iskbets_cfg.get("min_bigbet_conditions", 2)
    min_votes = iskbets_cfg.get("min_buy_votes", 3)

    # Gate 1: Reuse bigbet condition evaluation
    from portfolio.bigbet import _evaluate_conditions

    bull_conds, _bear_conds, _extra = _evaluate_conditions(
        ticker, signals, prices_usd, tf_data
    )

    if len(bull_conds) < min_bigbet:
        return False, []

    # Gate 2: Buy vote count from signals
    sig = signals.get(ticker)
    if not sig:
        return False, []
    extra = sig.get("extra", {})
    buy_count = extra.get("_buy_count", 0)

    if buy_count < min_votes:
        return False, []

    conditions = list(bull_conds)
    conditions.append(f"Signal grid: {buy_count} BUY votes")
    return True, conditions


def _before_cutoff(cutoff_str):
    """Check if current time is before the ET cutoff. cutoff_str like '14:30'."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    parts = cutoff_str.split(":")
    cutoff_hour = int(parts[0])
    cutoff_min = int(parts[1]) if len(parts) > 1 else 0
    return (now_et.hour < cutoff_hour) or (
        now_et.hour == cutoff_hour and now_et.minute < cutoff_min
    )


def _past_time_exit(time_exit_str):
    """Check if current time is past the ET time exit."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    parts = time_exit_str.split(":")
    exit_hour = int(parts[0])
    exit_min = int(parts[1]) if len(parts) > 1 else 0
    return (now_et.hour > exit_hour) or (
        now_et.hour == exit_hour and now_et.minute >= exit_min
    )


# ── Layer 2 Gate ─────────────────────────────────────────────────────────


GATE_LOG_FILE = DATA_DIR / "iskbets_gate_log.jsonl"


def _build_gate_prompt(ticker, price, conditions, signals, tf_data, atr, config):
    """Build a minimal prompt for the Layer 2 APPROVE/SKIP gate."""
    sig = signals.get(ticker, {})
    extra = sig.get("extra", {})
    ind = sig.get("indicators", {})

    buy_c = extra.get("_buy_count", 0)
    sell_c = extra.get("_sell_count", 0)
    hold_c = extra.get("_total_applicable", 21) - buy_c - sell_c

    rsi = ind.get("rsi", "N/A")
    macd = ind.get("macd_hist", "N/A")
    bb = ind.get("price_vs_bb", "N/A")
    atr_pct = (atr / price * 100) if price > 0 else 0

    fg = extra.get("fear_greed", "N/A")

    # Build TF heatmap row for this ticker
    tf_row = ""
    tf_list = tf_data.get(ticker, [])
    if tf_list:
        labels = []
        actions = []
        for label, td in tf_list:
            labels.append(label)
            a = td.get("action")
            if a == "BUY":
                actions.append("B")
            elif a == "SELL":
                actions.append("S")
            else:
                actions.append("H")
        tf_row = "/".join(labels) + ": " + " ".join(actions)

    # FOMC proximity from agent_summary if available
    fomc_days = "N/A"
    try:
        summary_file = DATA_DIR / "agent_summary.json"
        summary = load_json(summary_file)
        if summary:
            macro = summary.get("macro", {})
            fed_info = macro.get("fed", {})
            fomc_days = fed_info.get("days_until", "N/A")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
for path,start,end in [
('portfolio/fin_snipe.py',145,170),
('portfolio/fin_snipe_manager.py',455,505),
('portfolio/fin_fish.py',726,738),
('portfolio/orb_postmortem.py',76,84),
('data/metals_loop.py',210,220),
]:
    print(f'--- {path} ---')
    lines=Path(path).read_text().splitlines()
    for i in range(start,end+1):
        if i<=len(lines):
            print(f'{i}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 661ms:
--- portfolio/fin_snipe.py ---
145:                 logger.debug(
146:                     "Skipping orderbook %s (%s): ticker %s not in agent_summary signals",
147:                     orderbook_id, name, ticker,
148:                 )
149:             continue
150: 
151:         quote = market.get("quote") or {}
152:         underlying = (market.get("underlying") or {}).get("quote") or {}
153:         indicators = market.get("keyIndicators") or {}
154:         current_price = float(_value(quote.get("sell")) or _value(quote.get("last")) or 0.0)
155:         current_underlying = float(_value(underlying.get("last")) or 0.0)
156:         leverage = float(_value(indicators.get("leverage")) or 1.0)
157:         if current_price <= 0 or current_underlying <= 0 or leverage <= 0:
158:             continue
159: 
160:         ladder = build_intraday_ladder(
161:             signals[ticker],
162:             focus_probabilities.get(ticker),
163:             ticker=ticker,
164:             current_instrument_price=current_price,
165:             current_underlying_price=current_underlying,
166:             leverage=leverage,
167:             hours_remaining=hours_remaining,
168:             analysis=_analysis_for_ticker(
169:                 ticker,
170:                 simulate_flash_window=simulate_flash_window,
--- portfolio/fin_snipe_manager.py ---
455:             except (TypeError, ValueError):
456:                 position_entry_ts = dt.datetime.now(dt.UTC)
457:                 instrument_state["entry_ts"] = position_entry_ts.isoformat()
458:         else:
459:             position_entry_ts = dt.datetime.now(dt.UTC)
460:             instrument_state["entry_ts"] = position_entry_ts.isoformat()
461:         # BUG-228: If estimate returns -1.0 sentinel (invalid inputs),
462:         # fall back to current_underlying for the exit plan (prevents
463:         # ZeroDivisionError) but don't persist it.
464:         _entry_und = _estimate_entry_underlying(snapshot, instrument_state)
465:         if _entry_und <= 0:
466:             _entry_und = current_underlying
467:         plan = compute_exit_plan(
468:             Position(
469:                 symbol=snapshot["ticker"],
470:                 qty=position_volume,
471:                 entry_price_sek=position_avg,
472:                 entry_underlying_usd=_entry_und,
473:                 entry_ts=position_entry_ts,
474:                 instrument_type="warrant",
475:                 leverage=leverage,
476:                 financing_level=None,
477:             ),
478:             MarketSnapshot(
479:                 asof_ts=dt.datetime.now(dt.UTC),
480:                 price=current_underlying,
481:                 bid=float(underlying_summary.get("bid") or current_underlying),
482:                 ask=float(underlying_summary.get("ask") or current_underlying),
483:                 atr_pct=atr_pct if atr_pct > 0 else None,
484:                 usdsek=live_usdsek,
485:                 drift=0.0,
486:             ),
487:             session.session_end,
488:             n_paths=EXIT_OPTIMIZER_N_PATHS,
489:             seed=EXIT_OPTIMIZER_SEED,
490:         )
491:         target_underlying = float(plan.recommended.price_usd or fallback_underlying)
492:         translated = translate_underlying_target(
493:             float(snapshot.get("current_instrument_price") or current_bid or position_avg),
494:             current_underlying,
495:             target_underlying,
496:             leverage,
497:         )
498:         exit_price = _round_order_price(translated or fallback_price)
499:         minimum_profit_price = _round_order_price(max(position_avg, 0.0))
500:         if plan.recommended.action == "market" and current_bid > 0:
501:             exit_price = _round_order_price(current_bid)
502:             source = "quant_exit_optimizer_market"
503:         elif current_bid > 0:
504:             exit_price = max(exit_price, _round_order_price(current_bid))
505:             source = "quant_exit_optimizer"
--- portfolio/fin_fish.py ---
726:             leverage = warrant["leverage"]
727: 
728:         # Barrier checks only for MINI warrants (barrier > 0)
729:         if not is_daily_cert and barrier > 0:
730:             if direction == "LONG" and spot <= barrier:
731:                 continue  # knocked out
732:             if direction == "SHORT" and spot >= barrier:
733:                 # BEAR MINIs get knocked out if underlying goes above barrier
734:                 # (depends on product, but skip if too close)
735:                 pass
736:             barrier_distance = abs(spot - barrier) / spot * 100
737:             if barrier_distance < MIN_BARRIER_DISTANCE_PCT:
738:                 continue
--- portfolio/orb_postmortem.py ---
76:     low_error_pct = low_error_abs / prediction.predicted_low_median * 100
77: 
78:     # Target hit analysis
79:     # "Within conservative" means actual stayed within the tighter bounds
80:     high_within_conservative = actual_high <= prediction.predicted_high_aggressive
81:     high_within_aggressive = actual_high >= prediction.predicted_high_conservative
82:     low_within_conservative = actual_low >= prediction.predicted_low_aggressive
83:     low_within_aggressive = actual_low <= prediction.predicted_low_conservative
84: 
--- data/metals_loop.py ---
210: 
211: import requests
212: from metals_shared import get_cet_time
213: from playwright.sync_api import sync_playwright
214: 
215: from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
216: from portfolio.loop_contract import MetalsCycleReport, ViolationTracker, verify_and_act, verify_metals_contract
217: from portfolio.market_timing import is_swedish_market_holiday
218: 
219: try:
220:     from portfolio.notification_text import (

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/fin_snipe_manager.py -Pattern 'ladder\\]|ladder\\[|snapshot\\[\"ladder\"\\]' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 580ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:377:ladder = snapshot["ladder"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:914:ladder = snapshot["ladder"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:936:ladder = snapshot["ladder"]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1254:"last_working_price": snapshot["ladder"].get("working_price"),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1255:"last_flash_price": snapshot["ladder"].get("flash_price"),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1256:"last_exit_price": snapshot["ladder"].get("exit_price"),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/fin_snipe_manager.py | Select-Object -Index (360..410)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 594ms:
            current_underlying, current_price, entry_price, leverage,
        )
        return -1.0  # sentinel: caller should use current_underlying but not save

    instrument_return = (current_price / entry_price) - 1.0
    underlying_return = instrument_return / leverage
    base = 1.0 + underlying_return
    if base <= 0:
        logger.warning(
            "BUG-228: entry_underlying formula degenerate (base=%.4f)", base,
        )
        return -1.0
    return current_underlying / base


def _compute_exit_target(snapshot: dict, instrument_state: dict) -> dict[str, Any]:
    ladder = snapshot["ladder"]
    fallback_price = _round_order_price(ladder.get("exit_price") or 0.0)
    fallback_underlying = float(ladder.get("exit_underlying") or snapshot.get("current_underlying") or 0.0)
    current_bid = float(snapshot.get("current_bid") or 0.0)

    result = {
        "price": max(fallback_price, _round_order_price(current_bid)) if current_bid > 0 else fallback_price,
        "underlying_price": fallback_underlying,
        "source": "ladder",
        "fill_prob": None,
        "expected_fill_time_min": None,
        "stop_hit_prob": None,
        "risk_flags": [],
        "action": "limit",
        "optimizer_price": None,
        "optimizer_underlying_price": None,
    }

    position_volume = int(snapshot.get("position_volume") or 0)
    position_avg = float(snapshot.get("position_average_price") or 0.0)
    current_underlying = float(snapshot.get("current_underlying") or 0.0)
    leverage = float(snapshot.get("leverage") or 0.0)
    signal_entry = snapshot.get("signal_entry") or {}
    atr_pct = float(
        (signal_entry.get("extra") or {}).get("atr_pct")
        or signal_entry.get("atr_pct")
        or 0.0
    )
    session = get_session_info("warrant", underlying=snapshot.get("ticker"))
    if (
        position_volume <= 0
        or position_avg <= 0
        or current_underlying <= 0
        or leverage <= 0
        or not session.is_open

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/fin_snipe_manager.py -Pattern 'is_short|direction' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 551ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'warrant_price_factor' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 580ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:94:warrant_price_factor: float # Multiply current warrant price by this
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\orb_predictor.py:403:warrant_price_factor=factor,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'accumulate_snapshot\\(' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 566ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure_state.py:55:def accumulate_snapshot(ticker: str, depth: dict) -> None:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1798:accumulate_snapshot(ticker, depth)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1822:accumulate_snapshot(ticker, depth)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'persist_state\\(' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 566ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\microstructure_state.py:205:def persist_state() -> None:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1838:persist_state()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content data/metals_loop.py | Select-Object -Index (1778..1845)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 595ms:

def _accumulate_orderbook_snapshot_for(ticker: str) -> None:
    """Single-ticker orderbook snapshot accumulator.

    2026-04-13: factored out of _accumulate_orderbook_snapshots so the
    silver fast-tick (10s cadence) can poll XAG depth without paying the
    cost of XAU on every fast tick. Failures are logged at WARNING with
    1-in-30 throttling so a transient FAPI outage doesn't blow the log.

    Uses its own _snapshot_for_call_counter — NOT the cycle-level
    _microstructure_persist_counter — so the throttle gate measures actual
    helper invocations rather than cycle ticks.
    """
    global _snapshot_for_call_counter
    if not _MICROSTRUCTURE_AVAILABLE:
        return
    try:
        depth = get_orderbook_depth(ticker, limit=20)
        if depth:
            accumulate_snapshot(ticker, depth)
    except Exception:
        _snapshot_for_call_counter += 1
        if _snapshot_for_call_counter % 30 == 0:
            logger.warning(
                "_accumulate_microstructure: get_orderbook_depth failed for %s",
                ticker,
                exc_info=True,
            )


def _accumulate_orderbook_snapshots():
    """Poll order book depth and accumulate snapshots for OFI computation.

    Called each cycle (~30-60s).  Fetches depth for metals tickers,
    adds to ring buffer, and persists state every 5th call.
    """
    global _microstructure_persist_counter
    if not _MICROSTRUCTURE_AVAILABLE:
        return
    for ticker in _MICROSTRUCTURE_TICKERS:
        try:
            depth = get_orderbook_depth(ticker, limit=20)
            if depth:
                accumulate_snapshot(ticker, depth)
        except Exception:
            # 2026-04-09 Stage 3: WARNING — preserved 1-in-30 throttling
            # so this doesn't blow up the log file. Microstructure
            # snapshots are best-effort rolling accumulation — skipping
            # a few doesn't break signal computation, just degrades
            # VPIN/OFI slightly.
            if _microstructure_persist_counter % 30 == 0:  # log rarely
                logger.warning(
                    "_accumulate_microstructure: get_orderbook_depth failed for %s",
                    ticker,
                    exc_info=True,
                )
    _microstructure_persist_counter += 1
    if _microstructure_persist_counter % 5 == 0:  # persist every ~2.5-5 min
        try:
            persist_state()
        except Exception as e:
            log(f"Microstructure state persist error: {e}")


def fetch_underlying_klines(ticker, interval="1h", limit=100):
    """Fetch OHLCV klines from Binance (FAPI for metals, SPOT for crypto). Cached 5 min."""
    symbol = UNDERLYING_SYMBOLS.get(ticker) or CRYPTO_SYMBOLS.get(ticker)
    if not symbol:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path data/metals_loop.py -Pattern '_accumulate_orderbook_snapshot_for\\(' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 553ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1435:_accumulate_orderbook_snapshot_for("XAG-USD")
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1780:def _accumulate_orderbook_snapshot_for(ticker: str) -> None:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content data/metals_loop.py | Select-Object -Index (1410..1450)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 612ms:


def _silver_fast_tick():
    """10-second silver price check with threshold and velocity alerts.

    Merged from silver_monitor.py.  Fetches XAG-USD from Binance FAPI,
    checks for significant drops from the entry reference price, and detects
    rapid 3-minute flushes.  Only runs when an active silver position exists.
    """
    global _silver_session_low, _silver_session_high
    global _silver_consecutive_down, _silver_prev_price

    silver_key, silver_pos = _get_active_silver()
    if silver_key is None:
        return

    # 2026-04-13: Opportunistic XAG microstructure snapshot at 10s cadence.
    # Cycle-level _accumulate_orderbook_snapshots already runs this every
    # ~60s; the fast-tick adds 5x more snapshots while a position is held,
    # giving OFI/VPIN better resolution. Gated by _FAST_TICK_ORDERBOOK env
    # toggle for ops to disable if FAPI rate-limit pressure shows up.
    # Best-effort: never blocks the price-check or alerts below.
    if _FAST_TICK_ORDERBOOK:
        try:
            _accumulate_orderbook_snapshot_for("XAG-USD")
        except Exception:
            pass  # outer log gate inside the helper handles repeated failures

    price = _silver_fetch_xag()
    if price is None or price <= 0:
        return

    # Ensure reference is initialized
    _silver_init_ref()
    ref = _silver_underlying_ref
    if ref is None or ref <= 0:
        return

    # Underlying % change from entry reference
    pct_change = (price - ref) / ref * 100


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'FIXME|TODO|XXX|BUG' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 632ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_fish.py:1249:logger.debug("Profile briefing error: %s", e)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:93:logger.debug("Critical alert throttled: %s", category)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:101:# BUG-124: Use load_json instead of raw open/json.load
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:198:logger.debug("Failed to log fill detection", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:206:logger.debug("Fin Snipe log prune failed for %s", path, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:346:from an unresolvable failure (BUG-228: previously returned
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:359:"BUG-228: cannot estimate entry_underlying — missing inputs "
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:370:"BUG-228: entry_underlying formula degenerate (base=%.4f)", base,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:461:# BUG-228: If estimate returns -1.0 sentinel (invalid inputs),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1240:# BUG-228: Only save entry_underlying if estimate is valid (> 0).
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe.py:145:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\iskbets.py:332:# BUG-201 (2026-04-16): Route through detect_auth_failure. This gate
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:471:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:487:logger.debug("FRED API key not configured, skipping")
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_precompute.py:600:logger.debug("COT history prune failed: %s", e)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:555:logging.getLogger(__name__).debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:816:#   2026-04-07: first live test, 6 integration bugs cost 590 SEK.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:817:#   2026-04-09: all 6 bugs fixed; position reconciliation added.
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:821:# The previous disable comment said "re-enable only after 6 bugs fixed".
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:822:# That framing misdirects: the bugs ARE fixed, and the strategy itself
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:824:# on "the bugs are fixed" is insufficient — the bugs were fixed and the
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1120:logger.debug("_silver_fetch_xag: Binance FAPI call failed, using cached XAG-USD", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1135:logger.debug("_gold_fetch_xau: Binance FAPI call failed, using cached XAU-USD", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1233:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1318:logger.debug("entry_fast_tick telegram failed", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:1981:# so detect_holdings knows to log them at debug level and NOT add them to
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2009:logger.debug("dynamic warrant catalog load at module init failed: %s", _dyn_exc)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2050:logger.debug("Could not propagate FISHING_OB_IDS to swing_trader", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2213:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2241:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2246:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2312:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2329:# of SwingTrader's own management. Log once at debug so
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2333:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2752:logger.debug("_run_fish_engine_tick: agent_summary_compact load failed, using empty dict", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2781:logger.debug("_run_fish_engine_tick: enhanced signals parse failed, defaulting news/econ to HOLD", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2816:logger.debug("_run_fish_engine_tick: layer2 journal line parse failed, skipping", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2819:logger.debug("_run_fish_engine_tick: layer2_journal.jsonl read failed, defaulting layer2_* to empty", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2840:logger.debug("_run_fish_engine_tick: prophecy parse failed, defaulting to 0", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2855:logger.debug("_run_fish_engine_tick: check_trade_guard raised, defaulting trade_guard_ok=True", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2868:logger.debug("_run_fish_engine_tick: spread fetch (ob 1650161) failed, using default 0.3%%", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2896:"news_spike": False,  # TODO: read from headlines
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2949:2026-04-13 (Bug 1): reject decisions whose underlying signal is older
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:2962:# Signal-age guard (Bug 1)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3389:logger.debug("compute_probability_report: get_llm_signals failed ticker=%s, entry defaults to flat", ticker, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3439:logger.debug("compute_probability_report: get_fear_greed failed ticker=%s", ticker, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3456:logger.debug("compute_probability_report: get_onchain_summary failed ticker=%s", ticker, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3528:logger.debug("compute_probability_report: get_accuracy_report failed ticker=%s", ticker, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3555:logger.debug("_parse_vote_detail: vote_detail parse failed, returning empty", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3607:logger.debug("build_probability_telegram: get_fear_greed failed, omitting F&G tag", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3679:logger.debug("build_probability_telegram: MSTR price/NAV fetch failed, skipping line", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:3967:2026-04-13 (Bug 2a): retry once with a fresh Playwright context on
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4002:logger.debug(
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4089:logger.debug("_ensure_stops_cancelled_before_sell: telegram notify failed ob=%s (snapshot-failed path)", ob_str, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4182:logger.debug("_ensure_stops_cancelled_before_sell: telegram notify failed ob=%s (reconcile-failed path)", ob_str, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4231:logger.debug("_ensure_stops_cancelled_before_sell: telegram notify failed ob=%s (cancel-failed path)", ob_str, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4350:logger.debug("_rearm_stops_after_failed_sell: telegram notify failed ob=%s (naked-position alert path)", ob_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:4372:logger.debug("_rearm_stops_after_failed_sell: telegram notify failed ob=%s (partial-rearm alert path)", ob_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5407:logger.debug("place_spike_orders: telegram notify failed key=%s (rollback-notice path)", key, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5533:logger.debug("_rollback_spike_order_and_restore: telegram notify failed order=%s (rollback-incomplete path)", spike_order_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5833:logger.debug("_restore_full_stop_protection: telegram notify failed ob=%s (restore-deferred path)", ob_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5861:logger.debug("_restore_full_stop_protection: telegram notify failed ob=%s (restore-partial path)", ob_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:5877:logger.debug("_restore_full_stop_protection: telegram notify failed ob=%s (restore-failed path)", ob_id, exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6025:logger.debug("write_context: get_llm_summary failed, leaving llm_predictions empty", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6227:logger.debug("check_triggers: get_fear_greed failed, skipping F&G trigger check", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6246:logger.debug("check_triggers: get_llm_signals/accuracy failed, skipping LLM consensus trigger", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:6613:logger.debug("_autonomous_decision: get_llm_signals failed, llm_data stays empty", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7236:logger.debug("main_loop: short instrument fetch_price failed key=%s ob_id=%s", sk, si.get("ob_id"), exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7514:logger.debug("main_loop: swing summary build failed", exc_info=True)
Q:\finance-analyzer\.worktrees\adv-metals-core\data\metals_loop.py:7533:logger.debug("main_loop: get_accuracy_summary failed, omitting ACC tag", exc_info=True)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/fin_snipe_manager.py | Select-Object -Index (344..372)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 564ms:
    formula inputs are invalid so callers can distinguish a real estimate
    from an unresolvable failure (BUG-228: previously returned
    current_underlying, which was saved permanently and never retried).
    """
    saved = float(instrument_state.get("entry_underlying") or 0.0)
    if saved > 0:
        return saved

    current_underlying = float(snapshot.get("current_underlying") or 0.0)
    current_price = float(snapshot.get("current_instrument_price") or 0.0)
    entry_price = float(snapshot.get("position_average_price") or 0.0)
    leverage = float(snapshot.get("leverage") or 0.0)
    if current_underlying <= 0 or current_price <= 0 or entry_price <= 0 or leverage <= 0:
        logger.warning(
            "BUG-228: cannot estimate entry_underlying — missing inputs "
            "(underlying=%.2f, price=%.2f, entry=%.2f, lev=%.1f)",
            current_underlying, current_price, entry_price, leverage,
        )
        return -1.0  # sentinel: caller should use current_underlying but not save

    instrument_return = (current_price / entry_price) - 1.0
    underlying_return = instrument_return / leverage
    base = 1.0 + underlying_return
    if base <= 0:
        logger.warning(
            "BUG-228: entry_underlying formula degenerate (base=%.4f)", base,
        )
        return -1.0
    return current_underlying / base

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/fin_fish.py | Select-Object -Index (885..955)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 602ms:

    Returns a list of plans (one per direction: LONG and/or SHORT) with
    machine-readable structure for the snipe manager.

    Each plan dict::

        {
            "ticker": "XAG-USD",
            "spot": 69.21,
            "direction": "LONG",      # or "SHORT"
            "conviction": 0.65,
            "levels": [...],           # fishing level dicts with fill_prob, ev_sek
            "instrument": {"ob_id": ..., "name": ..., "leverage": ...},
            "tp_cascade": [...],       # from FISHING_TP_CASCADE config
            "sl_cascade": [...],       # from FISHING_SL_CASCADE config
        }
    """
    if budget_sek is None:
        budget_sek = FISHING_BUDGET_SEK
    if fx_rate is None:
        fx_rate = fetch_fx_rate()

    directions = choose_fishing_directions(signal)
    plans: list[dict] = []

    for d in directions:
        direction = d["direction"]
        conviction = d["conviction"]

        # Compute fishing levels
        if direction == "LONG":
            levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
        else:
            levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)

        # Evaluate warrants for these levels
        warrant_results = evaluate_warrants(
            ticker, spot, levels, budget_sek, fx_rate, direction=direction,
        )

        # Filter by minimum fill probability
        warrant_results = [
            r for r in warrant_results if r["fill_prob"] >= FISHING_MIN_FILL_PROB
        ]

        if not warrant_results:
            continue

        instrument_info = _build_instrument_info(warrant_results, direction)

        plans.append({
            "ticker": ticker,
            "spot": spot,
            "direction": direction,
            "conviction": conviction,
            "levels": warrant_results,
            "instrument": instrument_info,
            "tp_cascade": list(FISHING_TP_CASCADE),
            "sl_cascade": list(FISHING_SL_CASCADE),
        })

    return plans


# ---------------------------------------------------------------------------
# Telegram summary
# ---------------------------------------------------------------------------

def format_telegram_plan(plans: list[dict], avanza_online: bool = False) -> str:
    """Format fishing plans into a concise Telegram message.


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
# Stub missing modules
file_utils = types.ModuleType('portfolio.file_utils')
file_utils.atomic_append_jsonl = lambda *a, **k: None
file_utils.load_json = lambda *a, **k: {}
sys.modules['portfolio.file_utils'] = file_utils
mc = types.ModuleType('portfolio.monte_carlo')
mc.drift_from_probability = lambda p, v: 0.0
mc.volatility_from_atr = lambda atr: 0.2
sys.modules['portfolio.monte_carlo'] = mc
pt = types.ModuleType('portfolio.price_targets')
pt.fill_probability = lambda *a, **k: 0.5
pt.fill_probability_buy = lambda *a, **k: 0.5
pt.structural_levels = lambda *a, **k: {}
sys.modules['portfolio.price_targets'] = pt
mod = importlib.import_module('portfolio.fin_fish')
mod.WARRANT_CATALOG = {
    'TEST_BEAR': {
        'ob_id': '1', 'api_type': 'warrant', 'underlying': 'XAG-USD',
        'direction': 'SHORT', 'leverage': 5.0, 'barrier': 100.0, 'parity': 1,
        'name': 'TEST BEAR', 'issuer': 'AVA', 'spread_pct': 0.5, 'commission_sek': 0,
    }
}
mod.PREFERRED_INSTRUMENTS = {('XAG-USD','SHORT'): 'TEST_BEAR'}
levels = [{'level': 105.0, 'source': 'test', 'dip_pct': 5.0, 'move_pct': 5.0, 'fill_prob': 0.5, 'bounce_to_spot_pct': 5.0}]
print(mod.evaluate_warrants('XAG-USD', 120.0, levels, 1000.0, 10.0, direction='SHORT'))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 690ms:
[]

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
file_utils = types.ModuleType('portfolio.file_utils')
file_utils.atomic_append_jsonl = lambda *a, **k: None
file_utils.load_json = lambda *a, **k: {}
sys.modules['portfolio.file_utils'] = file_utils
mc = types.ModuleType('portfolio.monte_carlo')
mc.drift_from_probability = lambda p, v: 0.0
mc.volatility_from_atr = lambda atr: 0.2
sys.modules['portfolio.monte_carlo'] = mc
pt = types.ModuleType('portfolio.price_targets')
pt.fill_probability = lambda *a, **k: 0.5
pt.fill_probability_buy = lambda *a, **k: 0.5
pt.structural_levels = lambda *a, **k: {}
sys.modules['portfolio.price_targets'] = pt
mod = importlib.import_module('portfolio.fin_fish')
mod.WARRANT_CATALOG = {
    'TEST_BEAR': {
        'ob_id': '1', 'api_type': 'warrant', 'underlying': 'XAG-USD',
        'direction': 'SHORT', 'leverage': 5.0, 'barrier': 100.0, 'parity': 1,
        'name': 'TEST BEAR', 'issuer': 'AVA', 'spread_pct': 0.5, 'commission_sek': 0,
    }
}
mod.PREFERRED_INSTRUMENTS = {('XAG-USD','SHORT'): 'TEST_BEAR'}
levels = [{'level': 130.0, 'source': 'test', 'dip_pct': 8.33, 'move_pct': 8.33, 'fill_prob': 0.5, 'bounce_to_spot_pct': 8.33}]
from pprint import pprint
pprint(mod.evaluate_warrants('XAG-USD', 120.0, levels, 1000.0, 10.0, direction='SHORT'))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 763ms:
[{'barrier': 100.0,
  'barrier_dist_pct': 23.08,
  'bounce_pct': 8.33,
  'dip_pct': 8.33,
  'direction': 'SHORT',
  'ev_sek': 247.4,
  'fill_prob': 0.5,
  'gain_sek': 494.8,
  'invest_sek': 1000.0,
  'is_daily_cert': False,
  'issuer': 'AVA',
  'level': 130.0,
  'leverage': 6.0,
  'move_pct': 8.33,
  'net_gain_pct': 49.48,
  'ob_id': '1',
  'qty': 100000,
  'source': 'test',
  'spread_cost_sek': 5.0,
  'spread_pct': 0.5,
  'warrant': 'TEST BEAR',
  'warrant_gain_pct': 49.98,
  'warrant_price': 0.01}]

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, types, importlib
file_utils = types.ModuleType('portfolio.file_utils')
file_utils.atomic_append_jsonl = lambda *a, **k: None
sys.modules['portfolio.file_utils'] = file_utils
post = importlib.import_module('portfolio.orb_postmortem')
from portfolio.orb_predictor import Prediction
pred = Prediction(
    date='2026-05-08',
    morning_high=100, morning_low=90, morning_direction='up', morning_range_pct=5,
    predicted_high_conservative=102, predicted_high_median=104, predicted_high_aggressive=106,
    predicted_low_conservative=88, predicted_low_median=86, predicted_low_aggressive=84,
    sample_size=10, filters_applied=[]
)
r = post.run_postmortem(pred, actual_high=110, actual_low=80)
print(r.high_within_conservative, r.high_within_aggressive, r.low_within_conservative, r.low_within_aggressive)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 721ms:
False True False True

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/fin_snipe_manager.py -Pattern 'build_snapshots\\(' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 564ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1583:snapshots = build_snapshots(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/fin_snipe_manager.py -Pattern 'working_price|flash_price|exit_price' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 624ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:378:fallback_price = _round_order_price(ladder.get("exit_price") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:498:exit_price = _round_order_price(translated or fallback_price)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:501:exit_price = _round_order_price(current_bid)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:504:exit_price = max(exit_price, _round_order_price(current_bid))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:506:if exit_price < minimum_profit_price:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:507:exit_price = max(fallback_price, minimum_profit_price)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:513:"price": exit_price,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:915:working_price = _round_order_price(ladder.get("working_price") or ladder.get("mean_price") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:916:if working_price <= 0:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:918:return max(int(budget_sek // working_price), 0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:938:working_price = _round_order_price(ladder.get("working_price") or ladder.get("mean_price") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:940:working_price = min(working_price, _round_order_price(current_bid))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:942:flash_price = _round_order_price(ladder.get("flash_price") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:943:if flash_price > 0 and current_bid > 0:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:944:flash_price = min(flash_price, _round_order_price(current_bid))
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:946:if flash_price <= 0 or entry_volume < 2 or _price_matches(working_price, flash_price):
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:947:return [{"side": "BUY", "price": working_price, "volume": entry_volume, "role": "working", "order_type": "limit_order"}]
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:953:{"side": "BUY", "price": working_price, "volume": working_volume, "role": "working", "order_type": "limit_order"},
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:954:{"side": "BUY", "price": flash_price, "volume": flash_volume, "role": "flash", "order_type": "limit_order"},
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:982:exit_price = _round_order_price(exit_target.get("price") or 0.0)
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:983:if exit_price <= 0:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:987:"price": exit_price,
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1254:"last_working_price": snapshot["ladder"].get("working_price"),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1255:"last_flash_price": snapshot["ladder"].get("flash_price"),
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe_manager.py:1256:"last_exit_price": snapshot["ladder"].get("exit_price"),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path='data/metals_loop.py'
lines=Path(path).read_text().splitlines()
for i in range(198,210):
    print(f'{i}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 609ms:
198: 
199: BASE_DIR = Path(__file__).resolve().parents[1]
200: DATA_DIR = BASE_DIR / "data"
201: if str(BASE_DIR) not in sys.path:
202:     sys.path.insert(0, str(BASE_DIR))
203: # data/ also on sys.path so bare-module imports (metals_llm, metals_shared, ...)
204: # resolve here. Previously this was done lazily inside the LLM try-block below;
205: # hoisted to top-level so critical shared helpers (get_cet_time) can be imported
206: # as hard deps without a try-block. (2026-04-09 ARCH-12 dedup.)
207: if str(DATA_DIR) not in sys.path:
208:     sys.path.insert(0, str(DATA_DIR))
209: os.chdir(BASE_DIR)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Content portfolio/fin_snipe_manager.py | Select-Object -Index (930..955)' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 577ms:

def _desired_buy_orders(snapshot: dict, entry_volume: int) -> list[dict]:
    if entry_volume <= 0:
        return []

    ladder = snapshot["ladder"]
    current_bid = float(snapshot.get("current_bid") or 0.0)
    working_price = _round_order_price(ladder.get("working_price") or ladder.get("mean_price") or 0.0)
    if current_bid > 0:
        working_price = min(working_price, _round_order_price(current_bid))

    flash_price = _round_order_price(ladder.get("flash_price") or 0.0)
    if flash_price > 0 and current_bid > 0:
        flash_price = min(flash_price, _round_order_price(current_bid))

    if flash_price <= 0 or entry_volume < 2 or _price_matches(working_price, flash_price):
        return [{"side": "BUY", "price": working_price, "volume": entry_volume, "role": "working", "order_type": "limit_order"}]

    flash_volume = max(1, int(round(entry_volume * FLASH_ENTRY_VOLUME_PCT)))
    flash_volume = min(flash_volume, entry_volume - 1)
    working_volume = entry_volume - flash_volume
    return [
        {"side": "BUY", "price": working_price, "volume": working_volume, "role": "working", "order_type": "limit_order"},
        {"side": "BUY", "price": flash_price, "volume": flash_volume, "role": "flash", "order_type": "limit_order"},
    ]


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path='portfolio/metals_ladder.py'
lines=Path(path).read_text().splitlines()
for i in range(84,107):
    print(f'{i}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 544ms:
84:     leverage: float,
85:     hours_remaining: float,
86:     analysis: dict | None = None,
87:     direction_sign: int = 1,
88: ) -> dict:
89:     """Build a working bid / flash reserve / exit ladder for one instrument."""
90:     p_up = _safe_float((focus_probabilities or {}).get("3h", {}).get("probability"), 0.5)
91:     extra = signal_entry.get("extra") if isinstance(signal_entry, dict) else None
92:     squeeze_on = bool(((extra or {}).get("volatility_sig_indicators") or {}).get("bb_squeeze_on"))
93: 
94:     buy_targets = compute_targets(
95:         ticker,
96:         side="buy",
97:         price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
98:         atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
99:         p_up=p_up,
100:         hours_remaining=hours_remaining,
101:         indicators=signal_entry,
102:         extra=extra,
103:         regime=str(signal_entry.get("regime") or ""),
104:         bb_squeeze=squeeze_on,
105:     )
106:     sell_targets = compute_targets(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
path='portfolio/metals_ladder.py'
lines=Path(path).read_text().splitlines()
for i in range(106,130):
    print(f'{i}: {lines[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 590ms:
106:     sell_targets = compute_targets(
107:         ticker,
108:         side="sell",
109:         price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
110:         atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
111:         p_up=p_up,
112:         hours_remaining=hours_remaining,
113:         indicators=signal_entry,
114:         extra=extra,
115:         regime=str(signal_entry.get("regime") or ""),
116:         bb_squeeze=squeeze_on,
117:     )
118: 
119:     working_underlying = min(
120:         _safe_float((buy_targets.get("recommended") or {}).get("price"), current_underlying_price),
121:         _safe_float((buy_targets.get("extremes") or {}).get("p25"), current_underlying_price),
122:     )
123:     mean_underlying = _safe_float((buy_targets.get("recommended") or {}).get("price"), working_underlying)
124:     flash_drop_pct = flash_crash_drop_pct(analysis)
125:     flash_underlying = 0.0
126:     if flash_drop_pct > 0:
127:         flash_underlying = min(
128:             working_underlying,
129:             current_underlying_price * (1.0 - flash_drop_pct / 100.0),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path portfolio/*.py,data/*.py -Pattern 'build_intraday_ladder\\(' | ForEach-Object { \"{0}:{1}:{2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-metals-core
 succeeded in 566ms:
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\fin_snipe.py:160:ladder = build_intraday_ladder(
Q:\finance-analyzer\.worktrees\adv-metals-core\portfolio\metals_ladder.py:77:def build_intraday_ladder(

codex
The patch is not correct as submitted: a fresh checkout cannot import the new modules because required internal dependencies were not added, and several live trading paths mis-handle inverse/BEAR products. Those issues can stop the tooling from starting or cause it to compute/recommend orders on the wrong side of the market.

Full review comments:

- [P0] Include the support modules these entry points import — Q:/finance-analyzer/.worktrees/adv-metals-core/data/metals_loop.py:212-217
  This diff adds only the entry-point modules, but the imports here (`metals_shared`, `portfolio.file_utils`, `portfolio.loop_contract`, `portfolio.market_timing`, etc.) are not present anywhere in the tree. A clean checkout fails before any trading logic runs (`import data.metals_loop` currently dies on `metals_shared`), and the same problem hits other new modules such as `portfolio.fin_snipe_manager` and `portfolio.exit_optimizer`. The patch is not runnable until those internal dependencies are added to the branch as well.

- [P2] Avoid changing the caller's working directory on import — Q:/finance-analyzer/.worktrees/adv-metals-core/data/metals_loop.py:209-209
  The comment block above explicitly says this module should be safe to import from pytest and other scripts, but `os.chdir(BASE_DIR)` mutates global process state as soon as the import succeeds. Any caller that imports `data.metals_loop` and then opens relative paths or spawns subprocesses relative to its own cwd will start behaving differently. This needs to move into the CLI startup path instead of running at import time.

- [P1] Don't build BEAR ladders from long-side defaults — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_snipe.py:160-166
  This path never derives whether the orderbook is BULL or BEAR before calling `build_intraday_ladder()`, so inverse products inherit that helper's long-only defaults (`buy` targets below spot, `sell` targets above spot, positive translation sign). That works for BULL certificates but inverts the BEAR products added elsewhere in the branch (`BEAR_SILVER_X5_AVA_12`, `BEAR_GULD_*`): the manager/reporter will compute working bids and fallback exits on the wrong side whenever it sees a BEAR orderbook.

- [P1] Translate optimizer exits with inverse sign for BEAR holdings — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_snipe_manager.py:492-497
  This sell path always converts the optimizer's underlying target back to a certificate price as if the held product were long. For BEAR/MINI-short positions, a favorable drop in XAG/XAU should increase the cert price; here a lower `target_underlying` instead produces a lower `exit_price`, and `_estimate_entry_underlying()` earlier in the function has the same long-only assumption. In that scenario the managed SELL order gets pushed toward or below the bid instead of toward the profitable side.

- [P1] Skip BEAR MINI warrants once spot has crossed the barrier — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_fish.py:732-735
  When `direction == "SHORT"` and `spot >= barrier`, the code falls through instead of rejecting the instrument. If price is more than `MIN_BARRIER_DISTANCE_PCT` above the barrier, `evaluate_warrants()` can still return that BEAR MINI even though it is already knocked out and cannot be traded, so the planner may recommend an impossible instrument.
The patch is not correct as submitted: a fresh checkout cannot import the new modules because required internal dependencies were not added, and several live trading paths mis-handle inverse/BEAR products. Those issues can stop the tooling from starting or cause it to compute/recommend orders on the wrong side of the market.

Full review comments:

- [P0] Include the support modules these entry points import — Q:/finance-analyzer/.worktrees/adv-metals-core/data/metals_loop.py:212-217
  This diff adds only the entry-point modules, but the imports here (`metals_shared`, `portfolio.file_utils`, `portfolio.loop_contract`, `portfolio.market_timing`, etc.) are not present anywhere in the tree. A clean checkout fails before any trading logic runs (`import data.metals_loop` currently dies on `metals_shared`), and the same problem hits other new modules such as `portfolio.fin_snipe_manager` and `portfolio.exit_optimizer`. The patch is not runnable until those internal dependencies are added to the branch as well.

- [P2] Avoid changing the caller's working directory on import — Q:/finance-analyzer/.worktrees/adv-metals-core/data/metals_loop.py:209-209
  The comment block above explicitly says this module should be safe to import from pytest and other scripts, but `os.chdir(BASE_DIR)` mutates global process state as soon as the import succeeds. Any caller that imports `data.metals_loop` and then opens relative paths or spawns subprocesses relative to its own cwd will start behaving differently. This needs to move into the CLI startup path instead of running at import time.

- [P1] Don't build BEAR ladders from long-side defaults — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_snipe.py:160-166
  This path never derives whether the orderbook is BULL or BEAR before calling `build_intraday_ladder()`, so inverse products inherit that helper's long-only defaults (`buy` targets below spot, `sell` targets above spot, positive translation sign). That works for BULL certificates but inverts the BEAR products added elsewhere in the branch (`BEAR_SILVER_X5_AVA_12`, `BEAR_GULD_*`): the manager/reporter will compute working bids and fallback exits on the wrong side whenever it sees a BEAR orderbook.

- [P1] Translate optimizer exits with inverse sign for BEAR holdings — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_snipe_manager.py:492-497
  This sell path always converts the optimizer's underlying target back to a certificate price as if the held product were long. For BEAR/MINI-short positions, a favorable drop in XAG/XAU should increase the cert price; here a lower `target_underlying` instead produces a lower `exit_price`, and `_estimate_entry_underlying()` earlier in the function has the same long-only assumption. In that scenario the managed SELL order gets pushed toward or below the bid instead of toward the profitable side.

- [P1] Skip BEAR MINI warrants once spot has crossed the barrier — Q:/finance-analyzer/.worktrees/adv-metals-core/portfolio/fin_fish.py:732-735
  When `direction == "SHORT"` and `spot >= barrier`, the code falls through instead of rejecting the instrument. If price is more than `MIN_BARRIER_DISTANCE_PCT` above the barrier, `evaluate_warrants()` can still return that BEAR MINI even though it is already knocked out and cannot be traded, so the planner may recommend an impossible instrument.
