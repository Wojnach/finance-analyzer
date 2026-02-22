"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import json
import functools
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _read_jsonl(path, limit=100):
    if not path.exists():
        return []
    entries = deque(maxlen=limit)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(entries)


def _get_config():
    return _read_json(CONFIG_PATH) or {}


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
    tel = _read_jsonl(DATA_DIR / "telegram_messages.jsonl", limit=50)
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
    entries = _read_jsonl(DATA_DIR / "telegram_messages.jsonl", limit=50)
    return jsonify(entries)


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
# New: Signal heatmap (25 signals x all tickers)
# ---------------------------------------------------------------------------

@app.route("/api/signal-heatmap")
@require_auth
def api_signal_heatmap():
    """Return the full 25-signal x all-tickers grid.

    Each cell is BUY/SELL/HOLD. Built from agent_summary.json signals + enhanced_signals.
    """
    summary = _read_json(DATA_DIR / "agent_summary.json")
    if not summary:
        return jsonify({"error": "no data"}), 404

    signals_data = summary.get("signals", {})

    # Core signal names (extracted from _votes in extra)
    core_signals = [
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "ministral", "ml", "funding", "volume"
    ]
    # Enhanced composite signal names
    enhanced_signals = [
        "trend", "momentum", "volume_flow", "volatility_sig",
        "candlestick", "structure", "fibonacci", "smart_money",
        "oscillators", "heikin_ashi", "mean_reversion", "calendar",
        "macro_regime", "momentum_factors"
    ]
    all_signals = core_signals + enhanced_signals

    heatmap = {}
    tickers = list(signals_data.keys())

    for ticker in tickers:
        sig = signals_data[ticker]
        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})

        # _votes contains all 25 signal keys (core + enhanced)
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
    try:
        limit = min(int(request.args.get("limit", 50)), 500)
    except (ValueError, TypeError):
        limit = 50
    ticker_filter = request.args.get("ticker", "").upper()
    action_filter = request.args.get("action", "").upper()
    strategy_filter = request.args.get("strategy", "").lower()

    raw = _read_jsonl(DATA_DIR / "layer2_journal.jsonl", limit=1000)

    results = []
    for entry in reversed(raw):  # newest first
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
