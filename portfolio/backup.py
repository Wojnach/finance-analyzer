"""Daily backup utility for critical portfolio data files."""

import datetime
import pathlib
import shutil

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "backups"

BACKUP_FILES = [
    "portfolio_state.json",
    "portfolio_state_bold.json",
    "signal_log.jsonl",
    "layer2_journal.jsonl",
    "accuracy_cache.json",
    "activation_cache.json",
]

MAX_BACKUPS = 7  # Keep 7 days of backups


def backup_all():
    """Create timestamped copies of critical data files."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for filename in BACKUP_FILES:
        src = DATA_DIR / filename
        if not src.exists():
            results.append({"file": filename, "status": "not_found"})
            continue

        dst = BACKUP_DIR / f"{src.stem}_{ts}{src.suffix}"
        shutil.copy2(src, dst)
        results.append({
            "file": filename,
            "status": "backed_up",
            "backup": str(dst.name),
            "size_kb": round(dst.stat().st_size / 1024, 1)
        })

    # Cleanup old backups
    cleanup_old_backups()
    return results


def cleanup_old_backups():
    """Remove backups older than MAX_BACKUPS days."""
    if not BACKUP_DIR.exists():
        return

    cutoff = datetime.datetime.now() - datetime.timedelta(days=MAX_BACKUPS)
    removed = 0
    for f in BACKUP_DIR.iterdir():
        if f.is_file():
            try:
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def print_status():
    """Show current backup status."""
    if not BACKUP_DIR.exists():
        print("No backups directory found.")
        return

    total_size = 0
    files = sorted(BACKUP_DIR.iterdir())
    print(f"Backup directory: {BACKUP_DIR}")
    print(f"Files: {len(files)}")
    for f in files:
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size/1024:.1f} KB")
    print(f"Total: {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    import sys
    if "--status" in sys.argv:
        print_status()
    else:
        results = backup_all()
        for r in results:
            print(f"  {r['file']}: {r['status']}" +
                  (f" -> {r.get('backup', '')} ({r.get('size_kb', 0)} KB)" if r['status'] == 'backed_up' else ''))
