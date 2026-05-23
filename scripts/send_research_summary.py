"""Send signal research Telegram summary and update SQLite."""
import json
import sqlite3
import datetime
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# Send Telegram
config = json.loads((root / "config.json").read_text())
from portfolio.telegram_notifications import send_telegram

lines = [
    "*SIGNAL RESEARCH REPORT*",
    "",
    "Date: 2026-05-23",
    "Assets researched: ETH-USD, BTC-USD, XAU-USD, XAG-USD, cross-asset",
    "Papers found: 9",
    "Web sources: 7",
    "New candidates: 4",
    "",
    "IMPLEMENTED: autotune_adaptive_cycle",
    "Score: 8.1/10",
    "Backtest 1d: 48.4% avg (XAU 51.8%, ETH 49.2%)",
    "Backtest 5d: 53.2% avg (XAU 59.6%, ETH 55.2%)",
    "Status: disabled (shadow mode)",
    "Source: Ehlers TASC May 2026",
    "",
    "Other new candidates:",
    "- adaptive_momentum_trailing_stop: 7.6 (crypto, Sharpe 2.41)",
    "- macro_composite_timing: 7.15 (cross-asset, Sharpe 1.01)",
    "- funding_extreme_retracement: 6.9 (crypto, 60% WR)",
    "",
    f"Backlog size: ~59 candidates",
]
send_telegram("\n".join(lines), config)
print("Telegram sent")

# Update SQLite
db = sqlite3.connect(str(root / "data" / "signal_log.db"))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
db.execute(
    "UPDATE signal_candidates SET status='implemented', implemented_module=?, implemented_date=?, updated_at=? WHERE name=? AND status='new'",
    ("portfolio/signals/autotune_adaptive_cycle.py", "2026-05-23", now, "autotune_adaptive_cycle"),
)
db.execute(
    "UPDATE signal_candidates SET backtest_winrate=?, backtest_notes=?, updated_at=? WHERE name=? AND status='implemented'",
    (0.484, json.dumps({
        "BTC-USD": {"total": 259, "acc_1d": 0.463, "acc_3d": 0.490, "acc_5d": 0.548},
        "ETH-USD": {"total": 250, "acc_1d": 0.492, "acc_3d": 0.560, "acc_5d": 0.552},
        "XAU-USD": {"total": 166, "acc_1d": 0.518, "acc_3d": 0.548, "acc_5d": 0.596},
        "XAG-USD": {"total": 181, "acc_1d": 0.492, "acc_3d": 0.459, "acc_5d": 0.508},
        "MSTR": {"total": 244, "acc_1d": 0.455, "acc_3d": 0.516, "acc_5d": 0.455},
    }), now, "autotune_adaptive_cycle"),
)
db.commit()
print("SQLite updated")

# Update progress
progress = {
    "current_phase": "DONE",
    "phase_started": "2026-05-23T15:00:00+00:00",
    "last_update": now,
    "status": "done",
    "notes": "Implemented autotune_adaptive_cycle (Ehlers AutoTune). 8.1/10, 20/20 tests, merged and pushed.",
    "phases_completed": ["PHASE 0: BASELINE", "PHASE 1: ACADEMIC SEARCH", "PHASE 2: WEB RESEARCH", "PHASE 3-4: SCORING", "PHASE 5-7: IMPLEMENT", "PHASE 8: SHIP"],
}
(root / "data" / "signal-research-progress.json").write_text(json.dumps(progress, indent=2))
print("Progress updated")
