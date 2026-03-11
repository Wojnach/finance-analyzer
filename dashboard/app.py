"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import json
import functools
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider


def _json_safe(value):
    """Convert NaN/Infinity to JSON-safe null recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


class SafeJSONProvider(DefaultJSONProvider):
    """Flask JSON provider that strips non-finite floats."""

    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


app = Flask(__name__, static_folder="static")
app.json = SafeJSONProvider(app)


@app.after_request
def add_cors_headers(response):
    """Allow same-network browser access (e.g. phone on LAN)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from portfolio.file_utils import load_json as _load_json_impl, load_jsonl as _load_jsonl_impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path):
    return _load_json_impl(path)


def _read_jsonl(path, limit=100):
    return _load_jsonl_impl(path, limit=limit)


def _get_config():
    return _read_json(CONFIG_PATH) or {}


def _parse_limit_arg(name, default, max_value):
    """Parse integer query arg with sane bounds and fallback."""
    try:
        value = int(request.args.get(name, default))
    except (ValueError, TypeError):
        value = default
    return max(1, min(value, max_value))


def _iter_latest_dict_entries(path, read_limit):
    """Yield JSONL entries newest-first, skipping non-dict shapes."""
    raw = _read_jsonl(path, limit=read_limit)
    for entry in reversed(raw):
        if isinstance(entry, dict):
            yield entry


def _aggregate_accuracy_bucket(bucket):
    """Aggregate nested accuracy stats into one accuracy/total pair."""
    if not isinstance(bucket, dict):
        return {"accuracy": None, "total": 0, "correct": 0}

    correct = 0
    total = 0
    for stats in bucket.values():
        if not isinstance(stats, dict):
            continue
        correct += int(stats.get("correct", 0) or 0)
        total += int(stats.get("total", 0) or 0)

    return {
        "accuracy": round(correct / total, 3) if total else None,
        "correct": correct,
        "total": total,
    }


def _build_local_llm_trend_point(entry, ticker=None):
    """Flatten one local-LLM history entry into chart-friendly metrics."""
    ticker = (ticker or "").upper() or None
    ministral = ((entry.get("ministral") or {}).get("overall") or {})
    by_ticker = ((entry.get("ministral") or {}).get("by_ticker") or {})
    ticker_stats = by_ticker.get(ticker, {}) if ticker else {}
    health = entry.get("health") or {}
    forecast = entry.get("forecast") or {}
    gating = (entry.get("gating_counts") or {}).get("forecast") or {}

    raw_1h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("1h"))
    raw_24h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("24h"))
    effective_1h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("1h"))
    effective_24h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("24h"))

    return {
        "date": entry.get("date"),
        "exported_at": entry.get("exported_at"),
        "days": entry.get("days"),
        "ticker": ticker,
        "ministral_accuracy": ministral.get("accuracy"),
        "ministral_samples": ministral.get("samples", 0),
        "ministral_ticker_accuracy": ticker_stats.get("accuracy"),
        "ministral_ticker_samples": ticker_stats.get("samples", 0),
        "chronos_success_rate": (health.get("chronos") or {}).get("success_rate"),
        "chronos_total": (health.get("chronos") or {}).get("total", 0),
        "kronos_success_rate": (health.get("kronos") or {}).get("success_rate"),
        "kronos_total": (health.get("kronos") or {}).get("total", 0),
        "forecast_raw_1h_accuracy": raw_1h["accuracy"],
        "forecast_raw_1h_total": raw_1h["total"],
        "forecast_raw_24h_accuracy": raw_24h["accuracy"],
        "forecast_raw_24h_total": raw_24h["total"],
        "forecast_effective_1h_accuracy": effective_1h["accuracy"],
        "forecast_effective_1h_total": effective_1h["total"],
        "forecast_effective_24h_accuracy": effective_24h["accuracy"],
        "forecast_effective_24h_total": effective_24h["total"],
        "forecast_gating_raw": gating.get("raw", 0),
        "forecast_gating_held": gating.get("held", 0),
        "forecast_gating_insufficient_data": gating.get("insufficient_data", 0),
        "forecast_gating_vol_gated": gating.get("vol_gated", 0),
    }


# ---------------------------------------------------------------------------
# Token authentication middleware
# ---------------------------------------------------------------------------

def _get_dashboard_token():
    """Return the configured dashboard_token, or None if not set."""
    cfg = _get_config()
    return cfg.get("dashboard_token") or None


def require_auth(f):
    """Decorator: checks ?token= query param or Authorization: Bearer header.

    If no dashboard_token is configured, access is allowed (backwards compatible).
    Returns 401 for invalid tokens.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        expected = _get_dashboard_token()
        if expected is None:
            # No token configured — allow unauthenticated access
            return f(*args, **kwargs)

        # Check query param
        token = request.args.get("token")
        if token and token == expected:
            return f(*args, **kwargs)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:].strip()
            if bearer_token == expected:
                return f(*args, **kwargs)

        return jsonify({"error": "Unauthorized", "message": "Invalid or missing token"}), 401

    return decorated


# ---------------------------------------------------------------------------
# Routes — Static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Routes — API (all require auth)
# ---------------------------------------------------------------------------

@app.route("/api/summary")
@require_auth
def api_summary():
    """Combined endpoint for auto-refresh: signals + both portfolios + telegrams."""
    sig = _read_json(DATA_DIR / "agent_summary.json")
    port = _read_json(DATA_DIR / "portfolio_state.json")
    port_bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
    tel = list(_iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=50))
    return jsonify({
        "signals": sig,
        "portfolio": port,
        "portfolio_bold": port_bold,
        "telegrams": tel,
    })


@app.route("/api/signals")
@require_auth
def api_signals():
    data = _read_json(DATA_DIR / "agent_summary.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/portfolio")
@require_auth
def api_portfolio():
    data = _read_json(DATA_DIR / "portfolio_state.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/portfolio-bold")
@require_auth
def api_portfolio_bold():
    data = _read_json(DATA_DIR / "portfolio_state_bold.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/invocations")
@require_auth
def api_invocations():
    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/telegrams")
@require_auth
def api_telegrams():
    """Return telegram messages with optional filtering.

    Query params:
      - limit: max entries (default 200, max 2000)
      - category: filter by category (trade, analysis, iskbets, bigbet, digest, etc.)
      - search: text search in message body
    """
    limit = _parse_limit_arg("limit", default=200, max_value=2000)
    category_filter = request.args.get("category", "").strip().lower()
    search_filter = request.args.get("search", "").strip().lower()

    results = []
    for entry in _iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=5000):
        if category_filter and (entry.get("category", "") or "").lower() != category_filter:
            continue
        if search_filter and search_filter not in (entry.get("text", "") or "").lower():
            continue
        results.append(entry)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/api/signal-log")
@require_auth
def api_signal_log():
    entries = _read_jsonl(DATA_DIR / "signal_log.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/accuracy")
@require_auth
def api_accuracy():
    try:
        from portfolio.accuracy_stats import (
            signal_accuracy,
            consensus_accuracy,
            per_ticker_accuracy,
        )

        result = {}
        for horizon in ["1d", "3d", "5d", "10d"]:
            sa = signal_accuracy(horizon)
            ca = consensus_accuracy(horizon)
            ta = per_ticker_accuracy(horizon)
            if ca["total"] > 0:
                result[horizon] = {
                    "signals": sa,
                    "consensus": ca,
                    "per_ticker": ta,
                }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/iskbets")
@require_auth
def api_iskbets():
    config = _read_json(DATA_DIR / "iskbets_config.json")
    state = _read_json(DATA_DIR / "iskbets_state.json")
    return jsonify({"config": config, "state": state})


@app.route("/api/lora-status")
@require_auth
def api_lora_status():
    state = _read_json(TRAINING_DIR / "state.json")
    progress = _read_json(TRAINING_DIR / "training_progress.json")
    return jsonify({"state": state, "training_progress": progress})


# ---------------------------------------------------------------------------
# New: Portfolio validation
# ---------------------------------------------------------------------------

@app.route("/api/validate-portfolio", methods=["POST"])
@require_auth
def api_validate_portfolio():
    """Validate a portfolio JSON for integrity.

    Delegates to portfolio_validator.validate_portfolio() which performs
    comprehensive checks: cash, holdings, fees, transactions, avg_cost.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400

    try:
        from portfolio.portfolio_validator import validate_portfolio
        errors = validate_portfolio(data)
    except Exception as e:
        return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500

    return jsonify({
        "valid": len(errors) == 0,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# New: Equity curve
# ---------------------------------------------------------------------------

@app.route("/api/equity-curve")
@require_auth
def api_equity_curve():
    """Return portfolio value history for charting.

    Reads data/portfolio_value_history.jsonl. Returns empty array if missing.
    """
    entries = _read_jsonl(DATA_DIR / "portfolio_value_history.jsonl", limit=5000)
    return jsonify(entries)


# ---------------------------------------------------------------------------
# New: Signal heatmap (30 signals x all tickers)
# ---------------------------------------------------------------------------

@app.route("/api/signal-heatmap")
@require_auth
def api_signal_heatmap():
    """Return the full 30-signal x all-tickers grid.

    Each cell is BUY/SELL/HOLD. Built from agent_summary.json signals + enhanced_signals.
    """
    summary = _read_json(DATA_DIR / "agent_summary.json")
    if not summary:
        return jsonify({"error": "no data"}), 404

    signals_data = summary.get("signals", {})

    # Core signal names (11 total: 8 active + 3 disabled)
    core_signals = [
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "ministral", "volume", "ml", "funding", "custom_lora"
    ]
    # Enhanced composite signal names (19 modules, signals #12-#30)
    enhanced_signals = [
        "trend", "momentum", "volume_flow", "volatility_sig",
        "candlestick", "structure", "fibonacci", "smart_money",
        "oscillators", "heikin_ashi", "mean_reversion", "calendar",
        "macro_regime", "momentum_factors", "news_event", "econ_calendar",
        "forecast", "claude_fundamental", "futures_flow"
    ]
    all_signals = core_signals + enhanced_signals

    heatmap = {}
    tickers = list(signals_data.keys())

    for ticker in tickers:
        sig = signals_data[ticker]
        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})

        # _votes contains all 30 signal keys (core + enhanced)
        row = {}
        for s in all_signals:
            row[s] = (votes.get(s, "HOLD") or "HOLD").upper()
        heatmap[ticker] = row

    return jsonify({
        "tickers": tickers,
        "signals": all_signals,
        "core_signals": core_signals,
        "enhanced_signals": enhanced_signals,
        "heatmap": heatmap,
    })


# ---------------------------------------------------------------------------
# New: Trigger activity timeline
# ---------------------------------------------------------------------------

@app.route("/api/triggers")
@require_auth
def api_triggers():
    """Return last 50 trigger/invocation events from invocations.jsonl."""
    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/accuracy-history")
@require_auth
def api_accuracy_history():
    """Return accuracy snapshots over time for charting trend lines."""
    entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
    return jsonify(entries)


@app.route("/api/local-llm-trends")
@require_auth
def api_local_llm_trends():
    """Return local-LLM report trend data for dashboard charts.

    Query params:
      - limit: number of history points to return (default 90, max 366)
      - ticker: optional ticker filter for Ministral per-ticker series
    """
    limit = _parse_limit_arg("limit", default=90, max_value=366)
    ticker = request.args.get("ticker", "").strip().upper() or None
    latest = _read_json(DATA_DIR / "local_llm_report_latest.json")
    history = _read_jsonl(DATA_DIR / "local_llm_report_history.jsonl", limit=limit)

    return jsonify({
        "ticker": ticker,
        "latest": latest,
        "series": [
            _build_local_llm_trend_point(entry, ticker=ticker)
            for entry in history
            if isinstance(entry, dict)
        ],
    })


@app.route("/api/metals-accuracy")
@require_auth
def api_metals_accuracy():
    """Return metals loop signal accuracy (1h/3h horizons)."""
    data = _read_json(DATA_DIR / "metals_signal_accuracy.json")
    if not data:
        return jsonify({"error": "no data", "stats": {}})
    return jsonify(data)


@app.route("/api/trades")
@require_auth
def api_trades():
    """Return combined transactions from both portfolio states for chart annotations."""
    patient = _read_json(DATA_DIR / "portfolio_state.json")
    bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
    trades = []
    if patient and patient.get("transactions"):
        for tx in patient["transactions"]:
            trades.append({
                "ts": tx.get("timestamp", ""),
                "ticker": tx.get("ticker", ""),
                "action": tx.get("action", ""),
                "total_sek": tx.get("total_sek", 0),
                "price_usd": tx.get("price_usd", 0),
                "strategy": "patient",
            })
    if bold and bold.get("transactions"):
        for tx in bold["transactions"]:
            trades.append({
                "ts": tx.get("timestamp", ""),
                "ticker": tx.get("ticker", ""),
                "action": tx.get("action", ""),
                "total_sek": tx.get("total_sek", 0),
                "price_usd": tx.get("price_usd", 0),
                "strategy": "bold",
            })
    trades.sort(key=lambda t: t.get("ts", ""))
    return jsonify(trades)


@app.route("/api/decisions")
@require_auth
def api_decisions():
    """Return Layer 2 decision history with optional filtering.

    Query params:
      - limit: max entries (default 50, max 500)
      - ticker: filter by ticker (e.g., BTC-USD)
      - action: filter by action (BUY, SELL, HOLD)
      - strategy: filter by strategy (patient, bold)
    """
    limit = _parse_limit_arg("limit", default=50, max_value=500)
    ticker_filter = request.args.get("ticker", "").upper()
    action_filter = request.args.get("action", "").upper()
    strategy_filter = request.args.get("strategy", "").lower()

    results = []
    for entry in _iter_latest_dict_entries(DATA_DIR / "layer2_journal.jsonl", read_limit=1000):
        # Apply action/strategy filters
        if action_filter or strategy_filter:
            decisions = entry.get("decisions", {})
            matched = False
            for strat, dec in decisions.items():
                if strategy_filter and strat != strategy_filter:
                    continue
                if action_filter and dec.get("action", "").upper() != action_filter:
                    continue
                matched = True
            if not matched:
                continue

        if ticker_filter:
            tickers = entry.get("tickers", {})
            if ticker_filter not in tickers:
                continue

        results.append(entry)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/api/health")
@require_auth
def api_health():
    """Return system health summary (loop heartbeat, errors, agent silence)."""
    try:
        from portfolio.health import get_health_summary
        return jsonify(get_health_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# New: Warrant portfolio
# ---------------------------------------------------------------------------

@app.route("/api/warrants")
@require_auth
def api_warrants():
    """Return warrant holdings with leverage P&L.

    Reads data/portfolio_state_warrants.json. Returns empty structure if missing.
    """
    data = _read_json(DATA_DIR / "portfolio_state_warrants.json")
    if not data:
        return jsonify({"holdings": {}, "transactions": []})
    return jsonify(data)


# ---------------------------------------------------------------------------
# New: Risk data (Monte Carlo + VaR)
# ---------------------------------------------------------------------------

@app.route("/api/risk")
@require_auth
def api_risk():
    """Return Monte Carlo price bands and Portfolio VaR from compact summary.

    Reads monte_carlo and portfolio_var sections from agent_summary_compact.json.
    """
    compact = _read_json(DATA_DIR / "agent_summary_compact.json")
    if not compact:
        return jsonify({"monte_carlo": {}, "portfolio_var": {}})
    return jsonify({
        "monte_carlo": compact.get("monte_carlo", {}),
        "portfolio_var": compact.get("portfolio_var", {}),
    })


# ---------------------------------------------------------------------------
# New: Metals monitoring
# ---------------------------------------------------------------------------

@app.route("/api/metals")
@require_auth
def api_metals():
    """Return combined metals monitoring data.

    Reads:
      - data/metals_context.json — live positions, P&L, risk, signals, prices
      - data/metals_decisions.jsonl — decision log (newest first, last 50)
      - data/metals_history.json — YTD stats + daily OHLCV
      - data/silver_analysis.json — multi-TF technicals
    """
    context = _read_json(DATA_DIR / "metals_context.json")
    decisions = list(_iter_latest_dict_entries(DATA_DIR / "metals_decisions.jsonl", read_limit=50))
    history = _read_json(DATA_DIR / "metals_history.json")
    technicals = _read_json(DATA_DIR / "silver_analysis.json")
    return jsonify({
        "context": context,
        "decisions": decisions,
        "history": history,
        "technicals": technicals,
    })


# ---------------------------------------------------------------------------
# New: GoldDigger monitoring
# ---------------------------------------------------------------------------

@app.route("/api/golddigger")
@require_auth
def api_golddigger():
    """Return GoldDigger signal data: state, log entries, trades.

    Reads:
      - data/golddigger_state.json — live composite score, z-scores, position, session
      - data/golddigger_log.jsonl — signal log (newest first, last 100)
      - data/golddigger_trades.jsonl — trade history (newest first, last 50)
    """
    state = _read_json(DATA_DIR / "golddigger_state.json")
    log = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_log.jsonl", read_limit=100))
    trades = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_trades.jsonl", read_limit=50))
    return jsonify({
        "state": state,
        "log": log,
        "trades": trades,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
