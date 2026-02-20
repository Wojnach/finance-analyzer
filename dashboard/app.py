"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import json
import functools
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
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:]


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
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    Checks:
      - cash_sek is non-negative
      - All share counts are non-negative
      - Transaction math: starting cash - BUY allocs + SELL proceeds = cash_sek
      - Holdings integrity: total bought - total sold = current shares per ticker
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400

    errors = []

    # Check cash non-negative
    cash = data.get("cash_sek")
    if cash is None:
        errors.append("Missing cash_sek field")
    elif cash < 0:
        errors.append(f"cash_sek is negative: {cash}")

    # Check shares non-negative
    holdings = data.get("holdings", {})
    for ticker, info in holdings.items():
        shares = info.get("shares", 0) if isinstance(info, dict) else 0
        if shares < 0:
            errors.append(f"Negative shares for {ticker}: {shares}")

    # Check transaction math
    initial = data.get("initial_value_sek", 500000)
    transactions = data.get("transactions", [])
    computed_cash = initial
    ticker_bought = {}
    ticker_sold = {}

    for i, tx in enumerate(transactions):
        action = tx.get("action", "").upper()
        ticker = tx.get("ticker", "unknown")
        shares = tx.get("shares", 0)
        total = tx.get("total_sek", 0)
        fee = tx.get("fee_sek", 0)

        if action == "BUY":
            # BUY: full alloc deducted from cash
            computed_cash -= total
            ticker_bought.setdefault(ticker, 0)
            ticker_bought[ticker] += shares
        elif action == "SELL":
            # SELL: net proceeds added to cash
            computed_cash += total
            ticker_sold.setdefault(ticker, 0)
            ticker_sold[ticker] += shares

    # Cash reconciliation (allow small float tolerance)
    if cash is not None:
        diff = abs(computed_cash - cash)
        if diff > 1.0:  # 1 SEK tolerance for float rounding
            errors.append(
                f"Cash mismatch: computed {computed_cash:.2f} vs recorded {cash:.2f} "
                f"(diff {diff:.2f} SEK)"
            )

    # Holdings reconciliation
    all_tickers = set(list(ticker_bought.keys()) + list(ticker_sold.keys()))
    for ticker in all_tickers:
        bought = ticker_bought.get(ticker, 0)
        sold = ticker_sold.get(ticker, 0)
        expected_remaining = bought - sold

        actual = 0
        if ticker in holdings:
            h = holdings[ticker]
            actual = h.get("shares", 0) if isinstance(h, dict) else 0

        diff = abs(expected_remaining - actual)
        if diff > 0.0001:  # tolerance for float precision
            errors.append(
                f"Holdings mismatch for {ticker}: bought {bought:.6f} - sold {sold:.6f} "
                f"= expected {expected_remaining:.6f}, actual {actual:.6f}"
            )

    return jsonify({
        "valid": len(errors) == 0,
        "errors": errors,
        "computed_cash": round(computed_cash, 2),
        "ticker_balances": {
            t: round(ticker_bought.get(t, 0) - ticker_sold.get(t, 0), 6)
            for t in all_tickers
        },
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
        "ministral", "ml", "funding", "volume", "custom_lora"
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
