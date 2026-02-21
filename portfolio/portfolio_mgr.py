"""Portfolio state management — load, save, atomic writes, value calculation."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
INITIAL_CASH_SEK = 500_000


def _atomic_write_json(path, data):
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "cash_sek": INITIAL_CASH_SEK,
        "holdings": {},
        "transactions": [],
        "start_date": datetime.now(timezone.utc).isoformat(),
        "initial_value_sek": INITIAL_CASH_SEK,
    }


def save_state(state):
    _atomic_write_json(STATE_FILE, state)


def portfolio_value(state, prices_usd, fx_rate):
    total = state["cash_sek"]
    for ticker, h in state.get("holdings", {}).items():
        if h["shares"] > 0 and ticker in prices_usd:
            total += h["shares"] * prices_usd[ticker] * fx_rate
    return total
