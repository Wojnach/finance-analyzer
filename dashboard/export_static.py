"""Export dashboard API endpoints as static JSON files.

Imports the Flask app, uses its test client to call each API endpoint,
and writes the JSON responses to dashboard/static/api-data/<name>.json.

Run periodically (e.g. via scheduled task) to keep the static site updated.

Usage:
    python dashboard/export_static.py
    python dashboard/export_static.py --out /path/to/output  # custom output dir
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "config.json"

from dashboard.app import app  # noqa: E402

# Endpoints to export: (Flask route, static filename)
ENDPOINTS = [
    ("/api/summary", "summary.json"),
    ("/api/signals", "signals.json"),
    ("/api/portfolio", "portfolio.json"),
    ("/api/portfolio-bold", "portfolio-bold.json"),
    ("/api/signal-heatmap", "signal-heatmap.json"),
    ("/api/equity-curve", "equity-curve.json"),
    ("/api/triggers", "triggers.json"),
    ("/api/decisions", "decisions.json"),
    ("/api/telegrams", "telegrams.json"),
    ("/api/accuracy-history", "accuracy-history.json"),
    ("/api/trades", "trades.json"),
    ("/api/warrants", "warrants.json"),
    ("/api/risk", "risk.json"),
    ("/api/metals", "metals.json"),
    ("/api/metals-accuracy", "metals-accuracy.json"),
    ("/api/lora-status", "lora-status.json"),
    ("/api/health", "health.json"),
]

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "static" / "api-data"


def _get_dashboard_token():
    """Read dashboard token from config.json, if configured."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cfg.get("dashboard_token") or None


def export_all(out_dir: Path | None = None) -> dict:
    """Export all API endpoints to static JSON files.

    Args:
        out_dir: Directory to write files to. Defaults to dashboard/static/api-data/.

    Returns:
        Dict with 'ok' (list of exported filenames) and 'failed' (list of dicts).
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _get_dashboard_token()

    ok = []
    failed = []

    with app.test_client() as client:
        for route, filename in ENDPOINTS:
            try:
                req_route = route
                if token:
                    sep = "&" if "?" in req_route else "?"
                    req_route = f"{req_route}{sep}token={token}"
                resp = client.get(req_route)
                if resp.status_code != 200:
                    failed.append({
                        "route": route,
                        "filename": filename,
                        "status": resp.status_code,
                        "error": resp.get_data(as_text=True)[:200],
                    })
                    continue

                data = resp.get_json()
                dest = out_dir / filename
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

                ok.append(filename)
            except Exception as e:
                failed.append({
                    "route": route,
                    "filename": filename,
                    "error": str(e),
                })

    return {"ok": ok, "failed": failed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export dashboard API to static JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    result = export_all(out_dir=args.out)

    print(f"Exported {len(result['ok'])}/{len(ENDPOINTS)} endpoints")
    for name in result["ok"]:
        print(f"  OK  {name}")
    for fail in result["failed"]:
        print(f"  FAIL {fail['filename']}: {fail.get('status', '')} {fail.get('error', '')}")

    if result["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
