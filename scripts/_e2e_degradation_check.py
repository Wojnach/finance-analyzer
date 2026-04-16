"""End-to-end verification of the degradation tracker.

Hand-crafts a 7-day-old snapshot with strong baseline accuracy, stubs the
current accuracy sources to a clear collapse, and asserts the violation
list contains the expected severity. Restores the original snapshot file
on exit.

Run with:
  PYTHONPATH=Q:\\finance-analyzer-degradation Q:\\finance-analyzer\\.venv\\Scripts\\python.exe scripts\\_e2e_degradation_check.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import portfolio.accuracy_degradation as deg
import portfolio.accuracy_stats as acc_mod
import portfolio.forecast_accuracy as forecast_mod

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAP_PATH = DATA_DIR / "accuracy_snapshots.jsonl"
ALERT_PATH = DATA_DIR / "degradation_alert_state.json"
BACKUP_DIR = Path.home() / ".e2e_degradation_backup"
BACKUP_DIR.mkdir(exist_ok=True)
BACKUP_SNAP = BACKUP_DIR / "snapshots.jsonl"
BACKUP_ALERT = BACKUP_DIR / "alert_state.json"


def main() -> int:
    if SNAP_PATH.exists():
        shutil.copy(SNAP_PATH, BACKUP_SNAP)
        print(f"backed up snapshots -> {BACKUP_SNAP}")
    if ALERT_PATH.exists():
        shutil.copy(ALERT_PATH, BACKUP_ALERT)
        print(f"backed up alert state -> {BACKUP_ALERT}")

    try:
        # Drop alert state so the throttle starts fresh
        if ALERT_PATH.exists():
            ALERT_PATH.unlink()

        baseline_ts = datetime.now(UTC) - timedelta(days=7, hours=2)
        baseline = {
            "ts": baseline_ts.isoformat(),
            "signals": {"rsi": {"accuracy": 0.65, "total": 1500}},
            "signals_recent": {
                "rsi": {"accuracy": 0.65, "total": 280},
                "macd": {"accuracy": 0.60, "total": 250},
            },
            "consensus": {"accuracy": 0.55, "total": 4200},
            "consensus_recent": {"accuracy": 0.55, "total": 320},
        }

        # Append the synthetic baseline
        from portfolio.file_utils import atomic_append_jsonl
        atomic_append_jsonl(SNAP_PATH, baseline)
        print(f"injected synthetic baseline at ts={baseline.get('ts')}")

        # Stub current accuracy to a clear collapse so we exercise both
        # the per-signal WARNING path AND the consensus CRITICAL path:
        # - 3 signals dropping (rsi, macd, bb) -> CRITICAL via 3-signal rule
        # - Consensus 55% -> 35% -> 20pp drop -> CRITICAL via consensus rule
        deg_recent = {
            "rsi": {"accuracy": 0.42, "total": 320},   # 23pp drop, <50%
            "macd": {"accuracy": 0.41, "total": 280},  # 19pp drop, <50%
            "bb":   {"accuracy": 0.39, "total": 260},  # synthesized 3rd
        }
        # Add bb to baseline.signals_recent so the diff has something to compare
        baseline["signals_recent"]["bb"] = {"accuracy": 0.58, "total": 240}
        original_recent = acc_mod.signal_accuracy_recent
        original_consensus = acc_mod.consensus_accuracy
        original_per_ticker = acc_mod.accuracy_by_ticker_signal_cached
        original_per_ticker_recent = deg._per_ticker_recent
        original_forecast = forecast_mod.cached_forecast_accuracy

        acc_mod.signal_accuracy_recent = lambda h="1d", days=7: deg_recent

        def stub_consensus(horizon="1d", entries=None, days=None):
            if entries is None and days is not None:
                # 35% recent vs 55% baseline = 20pp drop, <50% absolute
                return {"accuracy": 0.35, "total": 320, "correct": 112}
            return {"accuracy": 0.55, "total": 4200, "correct": 2310}

        acc_mod.consensus_accuracy = stub_consensus
        acc_mod.accuracy_by_ticker_signal_cached = lambda h: {}
        deg._per_ticker_recent = lambda horizon, days: {}
        forecast_mod.cached_forecast_accuracy = (
            lambda horizon="24h", days=7, use_raw_sub_signals=True: {}
        )

        # Skip econ blackout for this test
        import portfolio.econ_dates as econ_mod
        original_within = econ_mod.events_within_hours
        original_recent_high = econ_mod.recent_high_impact_events
        econ_mod.events_within_hours = lambda hours: []
        econ_mod.recent_high_impact_events = lambda hours, impact_filter=("high",): []

        try:
            # First call: full check
            violations = deg.check_degradation()
            print(f"\nFirst check returned {len(violations)} violation(s):")
            for v in violations:
                print(f"  - severity={v.severity} invariant={v.invariant}")
                # cp1252 console can't render the Unicode arrow in messages
                safe_msg = v.message.encode("ascii", errors="replace").decode("ascii")
                print(f"    message: {safe_msg[:200]}")
                alerts = v.details.get("alerts", [])
                for a in alerts:
                    print(f"    alert: {a}")

            assert len(violations) == 1, f"expected 1 violation, got {len(violations)}"
            v = violations[0]
            assert v.severity == deg.SEVERITY_CRITICAL, (
                f"expected CRITICAL (consensus dropped), got {v.severity}"
            )
            print("\nPASS: first call returned 1 CRITICAL violation")

            # Second call: throttle replay
            second = deg.check_degradation()
            assert len(second) == 1, f"throttle replay broken: {len(second)} != 1"
            assert second[0].message == violations[0].message
            print("PASS: throttled second call replayed cached violation")

            # Daily summary
            snapshots = deg._load_snapshots()
            latest_synthetic = {
                "ts": datetime.now(UTC).isoformat(),
                "signals_recent": deg_recent,
                "consensus_recent": {"accuracy": 0.43, "total": 200},
                "forecast_recent": {
                    "chronos_24h": {"accuracy": 0.51, "total": 200},
                    "kronos_24h": {"accuracy": 0.49, "total": 180},
                },
            }
            body = deg.build_daily_summary(
                latest=latest_synthetic, baseline=baseline,
                now=datetime.now(UTC),
            )
            print("\n--- Daily summary preview ---")
            print(body.encode("ascii", errors="replace").decode("ascii"))
            print("--- end ---")
            assert "*ACCURACY DAILY*" in body
            assert "rsi" in body
            print("\nPASS: daily summary builder emits expected lines")

        finally:
            acc_mod.signal_accuracy_recent = original_recent
            acc_mod.consensus_accuracy = original_consensus
            acc_mod.accuracy_by_ticker_signal_cached = original_per_ticker
            deg._per_ticker_recent = original_per_ticker_recent
            forecast_mod.cached_forecast_accuracy = original_forecast
            econ_mod.events_within_hours = original_within
            econ_mod.recent_high_impact_events = original_recent_high

    finally:
        # Restore snapshot file
        if BACKUP_SNAP.exists():
            shutil.copy(BACKUP_SNAP, SNAP_PATH)
            print(f"\nrestored snapshots from {BACKUP_SNAP}")
        else:
            # No backup means original was empty; clean up our injection.
            if SNAP_PATH.exists():
                SNAP_PATH.unlink()
                print("removed injected snapshot file (no backup existed)")
        if BACKUP_ALERT.exists():
            shutil.copy(BACKUP_ALERT, ALERT_PATH)
            print(f"restored alert state from {BACKUP_ALERT}")
        else:
            if ALERT_PATH.exists():
                ALERT_PATH.unlink()
                print("removed test alert state (no backup existed)")

    print("\nALL E2E ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
