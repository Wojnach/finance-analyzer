"""Persistent Chronos forecast server.

Loads the Chronos model once, then processes JSON-line requests from stdin.
Eliminates cold-start CPU cost of reimporting torch + loading the model
on every call.

Usage:
    .venv/Scripts/python.exe -u data/chronos_server.py

Protocol (JSON lines over stdin/stdout):
    Request:  {"close_prices": [...], "horizons": [1, 3]}
    Response: {"1h": {...}, "3h": {...}} or {"error": "..."}
"""

import json
import sys

sys.path.insert(0, "Q:/finance-analyzer")

# Load the model ONCE at startup
try:
    from portfolio.forecast_signal import _get_chronos_pipeline, forecast_chronos

    pipeline = _get_chronos_pipeline()
    if pipeline is None:
        sys.stderr.write("CHRONOS_FAILED: could not load pipeline\n")
        sys.stderr.flush()
        sys.exit(1)

    sys.stderr.write("CHRONOS_READY\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"CHRONOS_FAILED: {e}\n")
    sys.stderr.flush()
    sys.exit(1)

# Process requests
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        close_prices = req["close_prices"]
        horizons = tuple(req.get("horizons", [1, 3]))
        result = forecast_chronos("", close_prices, horizons=horizons)
        if result:
            sys.stdout.write(json.dumps(result) + "\n")
        else:
            sys.stdout.write(json.dumps({}) + "\n")
    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
    sys.stdout.flush()
