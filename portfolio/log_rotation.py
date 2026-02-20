"""Log rotation for finance-analyzer data files.

Handles rotation for all JSONL and log files in the data/ directory.
Supports age-based rotation (JSONL files) and size-based rotation (plain text).

Usage:
    python -m portfolio.log_rotation          # rotate all files
    python -m portfolio.log_rotation --dry-run # show what would be rotated
    python -m portfolio.log_rotation --status  # show data dir sizes

Integration with main loop (add to portfolio/main.py loop() function):
    # At the top of the while True loop, once per day:
    from portfolio.log_rotation import rotate_all
    _last_rotation = 0
    ...
    if time.time() - _last_rotation > 86400:  # 24 hours
        try:
            rotate_all()
            _last_rotation = time.time()
        except Exception as e:
            print(f"  WARNING: log rotation failed: {e}")

Or run standalone via scheduled task (e.g. PF-LogRotate, daily at 03:00):
    .venv\\Scripts\\python.exe -m portfolio.log_rotation
"""

import datetime
import gzip
import json
import os
import pathlib
import shutil
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"

# Rotation policies per file
# - max_age_days: for JSONL files, archive entries older than this
# - max_size_mb: rotate when file exceeds this size
# - keep_rotations: for plain text files, how many rotated copies to keep
# - compress: gzip archived/rotated files
ROTATION_POLICIES = {
    "signal_log.jsonl": {
        "max_age_days": 30,
        "max_size_mb": 50,
        "compress": True,
        "type": "jsonl",
        "ts_field": "ts",
    },
    "agent.log": {
        "max_size_mb": 10,
        "keep_rotations": 3,
        "compress": True,
        "type": "text",
    },
    "loop_out.txt": {
        "max_size_mb": 5,
        "keep_rotations": 3,
        "compress": True,
        "type": "text",
    },
    "telegram_messages.jsonl": {
        "max_age_days": 90,
        "max_size_mb": 20,
        "compress": True,
        "type": "jsonl",
        "ts_field": "ts",
    },
    "ab_test_log.jsonl": {
        "max_age_days": 30,
        "max_size_mb": 20,
        "compress": True,
        "type": "jsonl",
        "ts_field": "ts",
    },
    "layer2_journal.jsonl": {
        "max_age_days": 60,
        "max_size_mb": 10,
        "compress": True,
        "type": "jsonl",
        "ts_field": "ts",
    },
}


def _ensure_archive_dir():
    """Create data/archive/ if it does not exist."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _file_size_mb(path):
    """Return file size in MB, or 0 if file does not exist."""
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _parse_ts(ts_str):
    """Parse ISO-8601 timestamp string to datetime (UTC).

    Handles both timezone-aware (with +00:00) and naive timestamps.
    """
    if ts_str is None:
        return None
    try:
        # Python 3.7+ fromisoformat handles most ISO formats
        dt = datetime.datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _gzip_file(src_path, dst_path):
    """Compress src_path to dst_path using gzip."""
    with open(src_path, "rb") as f_in:
        with gzip.open(dst_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def rotate_jsonl(filename, policy, dry_run=False):
    """Rotate a JSONL file by age: archive old entries, keep recent ones.

    Old entries are grouped by year-month and written to
    data/archive/FILENAME.YYYY-MM.jsonl.gz

    Returns dict with rotation stats.
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return {"file": filename, "status": "not_found"}

    size_mb = _file_size_mb(filepath)
    ts_field = policy.get("ts_field", "ts")
    max_age_days = policy.get("max_age_days", 30)
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=max_age_days)

    # Read all lines and classify as keep vs archive
    keep_lines = []
    archive_buckets = {}  # "YYYY-MM" -> list of raw lines
    parse_failures = 0
    total_lines = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total_lines += 1
            try:
                entry = json.loads(line)
                ts = _parse_ts(entry.get(ts_field))
                if ts is None:
                    # Can't determine age -- keep the entry to be safe
                    keep_lines.append(line)
                    parse_failures += 1
                elif ts >= cutoff:
                    keep_lines.append(line)
                else:
                    # Archive this entry, grouped by month
                    month_key = ts.strftime("%Y-%m")
                    archive_buckets.setdefault(month_key, []).append(line)
            except json.JSONDecodeError:
                # Malformed line -- keep it to avoid data loss
                keep_lines.append(line)
                parse_failures += 1

    archived_count = sum(len(v) for v in archive_buckets.values())
    result = {
        "file": filename,
        "size_mb": round(size_mb, 2),
        "total_lines": total_lines,
        "kept": len(keep_lines),
        "archived": archived_count,
        "archive_months": sorted(archive_buckets.keys()),
        "parse_failures": parse_failures,
    }

    if archived_count == 0:
        result["status"] = "nothing_to_archive"
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    _ensure_archive_dir()

    # Write archived entries to monthly files
    stem = pathlib.Path(filename).stem  # e.g. "signal_log"
    suffix = pathlib.Path(filename).suffix  # e.g. ".jsonl"

    for month_key, lines in sorted(archive_buckets.items()):
        archive_name = f"{stem}.{month_key}{suffix}"
        archive_path = ARCHIVE_DIR / archive_name
        gz_path = ARCHIVE_DIR / f"{archive_name}.gz"

        # Append to existing archive for this month (may already have entries
        # from a previous rotation)
        if gz_path.exists() and policy.get("compress", True):
            # Decompress existing, append, re-compress
            existing_lines = []
            with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
                for existing_line in gf:
                    existing_line = existing_line.rstrip("\n")
                    if existing_line.strip():
                        existing_lines.append(existing_line)
            all_lines = existing_lines + lines
            with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
                for l in all_lines:
                    gf.write(l + "\n")
        elif policy.get("compress", True):
            with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
                for l in lines:
                    gf.write(l + "\n")
        else:
            with open(archive_path, "a", encoding="utf-8") as af:
                for l in lines:
                    af.write(l + "\n")

    # Rewrite the original file with only kept lines
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in keep_lines:
            f.write(line + "\n")

    # Atomic-ish replace: remove original, rename tmp
    # On Windows, os.replace is atomic within the same volume
    os.replace(tmp_path, filepath)

    result["status"] = "rotated"
    return result


def rotate_text(filename, policy, dry_run=False):
    """Rotate a plain text file by size.

    When file exceeds max_size_mb:
    - Shift existing rotations: .2 -> .3, .1 -> .2, current -> .1
    - Delete rotations beyond keep_rotations
    - Compress old rotations if policy says so

    Returns dict with rotation stats.
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return {"file": filename, "status": "not_found"}

    size_mb = _file_size_mb(filepath)
    max_size_mb = policy.get("max_size_mb", 10)
    keep_rotations = policy.get("keep_rotations", 3)
    compress = policy.get("compress", True)

    result = {
        "file": filename,
        "size_mb": round(size_mb, 2),
        "max_size_mb": max_size_mb,
    }

    if size_mb < max_size_mb:
        result["status"] = "under_threshold"
        return result

    if dry_run:
        result["status"] = "dry_run_would_rotate"
        return result

    _ensure_archive_dir()

    stem = pathlib.Path(filename).stem
    ext = pathlib.Path(filename).suffix

    # Delete the oldest rotation if it exists
    oldest = ARCHIVE_DIR / f"{stem}{ext}.{keep_rotations}"
    oldest_gz = ARCHIVE_DIR / f"{stem}{ext}.{keep_rotations}.gz"
    if oldest_gz.exists():
        oldest_gz.unlink()
    if oldest.exists():
        oldest.unlink()

    # Shift existing rotations: N-1 -> N, N-2 -> N-1, ..., 1 -> 2
    for i in range(keep_rotations - 1, 0, -1):
        src = ARCHIVE_DIR / f"{stem}{ext}.{i}"
        src_gz = ARCHIVE_DIR / f"{stem}{ext}.{i}.gz"
        dst_num = i + 1
        dst = ARCHIVE_DIR / f"{stem}{ext}.{dst_num}"
        dst_gz = ARCHIVE_DIR / f"{stem}{ext}.{dst_num}.gz"

        if src_gz.exists():
            src_gz.rename(dst_gz)
        elif src.exists():
            src.rename(dst)

    # Move current file to .1 (in archive dir)
    rotation_1 = ARCHIVE_DIR / f"{stem}{ext}.1"
    rotation_1_gz = ARCHIVE_DIR / f"{stem}{ext}.1.gz"

    if compress:
        _gzip_file(filepath, rotation_1_gz)
    else:
        shutil.copy2(filepath, rotation_1)

    # Truncate the original file (creates a fresh empty file)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("")

    result["status"] = "rotated"
    result["rotated_to"] = str(rotation_1_gz if compress else rotation_1)
    return result


def rotate_file(filename, policy, dry_run=False):
    """Route to the appropriate rotation function based on file type."""
    file_type = policy.get("type", "text")
    if file_type == "jsonl":
        return rotate_jsonl(filename, policy, dry_run=dry_run)
    else:
        return rotate_text(filename, policy, dry_run=dry_run)


def rotate_all(dry_run=False):
    """Rotate all files defined in ROTATION_POLICIES.

    Returns list of result dicts, one per file.
    """
    results = []
    for filename, policy in ROTATION_POLICIES.items():
        try:
            result = rotate_file(filename, policy, dry_run=dry_run)
            results.append(result)
        except Exception as e:
            results.append({
                "file": filename,
                "status": "error",
                "error": str(e),
            })
    return results


def get_data_dir_size():
    """Return total size of data/ directory in MB (including subdirectories)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = pathlib.Path(dirpath) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def get_file_stats():
    """Return a list of dicts with size info for each managed file."""
    stats = []
    for filename, policy in ROTATION_POLICIES.items():
        filepath = DATA_DIR / filename
        size_mb = _file_size_mb(filepath)

        # Count lines for JSONL files
        line_count = None
        if policy.get("type") == "jsonl" and filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    line_count = sum(1 for line in f if line.strip())
            except OSError:
                pass

        # Check for existing archives
        archives = []
        if ARCHIVE_DIR.exists():
            stem = pathlib.Path(filename).stem
            for p in sorted(ARCHIVE_DIR.glob(f"{stem}*")):
                archives.append({
                    "name": p.name,
                    "size_mb": round(_file_size_mb(p), 3),
                })

        stats.append({
            "file": filename,
            "size_mb": round(size_mb, 2),
            "lines": line_count,
            "policy": policy,
            "archives": archives,
        })
    return stats


def print_status():
    """Print a human-readable status report of all managed files."""
    total_mb = get_data_dir_size()
    print(f"Data directory total: {total_mb:.1f} MB")
    print()

    stats = get_file_stats()
    for s in stats:
        lines_str = f", {s['lines']:,} lines" if s["lines"] is not None else ""
        policy = s["policy"]

        threshold = ""
        if "max_age_days" in policy:
            threshold = f"age>{policy['max_age_days']}d"
        if "max_size_mb" in policy:
            pct = (s["size_mb"] / policy["max_size_mb"]) * 100 if policy["max_size_mb"] > 0 else 0
            size_note = f"size>{policy['max_size_mb']}MB ({pct:.0f}% used)"
            threshold = f"{threshold}, {size_note}" if threshold else size_note

        print(f"  {s['file']:30s}  {s['size_mb']:7.2f} MB{lines_str}")
        print(f"    Policy: {threshold}")

        if s["archives"]:
            for a in s["archives"]:
                print(f"    Archive: {a['name']} ({a['size_mb']:.3f} MB)")
    print()


def print_results(results):
    """Print rotation results in a human-readable format."""
    for r in results:
        status = r.get("status", "unknown")
        file = r.get("file", "?")

        if status == "not_found":
            print(f"  {file}: not found, skipped")
        elif status == "nothing_to_archive":
            print(f"  {file}: {r.get('total_lines', '?')} lines, all within retention -- no action")
        elif status == "under_threshold":
            print(f"  {file}: {r.get('size_mb', '?')} MB < {r.get('max_size_mb', '?')} MB -- no action")
        elif status == "dry_run":
            print(f"  {file}: WOULD archive {r.get('archived', 0)} of {r.get('total_lines', '?')} lines")
            if r.get("archive_months"):
                print(f"    Months: {', '.join(r['archive_months'])}")
        elif status == "dry_run_would_rotate":
            print(f"  {file}: WOULD rotate ({r.get('size_mb', '?')} MB > {r.get('max_size_mb', '?')} MB)")
        elif status == "rotated":
            if "archived" in r:
                print(f"  {file}: archived {r['archived']} lines, kept {r['kept']}")
                if r.get("archive_months"):
                    print(f"    Months: {', '.join(r['archive_months'])}")
            else:
                print(f"  {file}: rotated to {r.get('rotated_to', '?')}")
        elif status == "error":
            print(f"  {file}: ERROR -- {r.get('error', 'unknown')}")
        else:
            print(f"  {file}: {status}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    show_status = "--status" in args

    if show_status:
        print("=== Log Rotation Status ===\n")
        print_status()
        sys.exit(0)

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"=== Log Rotation ({mode}) ===\n")

    results = rotate_all(dry_run=dry_run)
    print_results(results)

    total_mb = get_data_dir_size()
    print(f"\nData directory total: {total_mb:.1f} MB")
