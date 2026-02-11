"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import json
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"


def _read_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def _read_jsonl(path, limit=100):
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries[-limit:]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/signals")
def api_signals():
    data = _read_json(DATA_DIR / "agent_summary.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/portfolio")
def api_portfolio():
    data = _read_json(DATA_DIR / "portfolio_state.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/invocations")
def api_invocations():
    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/telegrams")
def api_telegrams():
    entries = _read_jsonl(DATA_DIR / "telegram_messages.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/signal-log")
def api_signal_log():
    entries = _read_jsonl(DATA_DIR / "signal_log.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/accuracy")
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


@app.route("/api/lora-status")
def api_lora_status():
    state = _read_json(TRAINING_DIR / "state.json")
    progress = _read_json(TRAINING_DIR / "training_progress.json")
    return jsonify({"state": state, "training_progress": progress})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
