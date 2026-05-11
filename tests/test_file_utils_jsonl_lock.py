"""Tests for jsonl_sidecar_lock + atomic_append_jsonl + rotate_jsonl race.

Regression for the 2026-05-11 signal_log_reconciliation divergence:
log_rotation.rotate_jsonl was rewriting the JSONL without holding the
sidecar lock that atomic_append_jsonl uses. Appends arriving between
rotation's read and replace were dropped (~400 per pass on signal_log).
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio import file_utils as fu
from portfolio import log_rotation as lr


# ---------------------------------------------------------------------------
# Sidecar lock direct usage
# ---------------------------------------------------------------------------


class TestJsonlSidecarLock:
    def test_creates_sidecar_file(self, tmp_path):
        target = tmp_path / "log.jsonl"
        target.write_text("")
        with fu.jsonl_sidecar_lock(target):
            assert (tmp_path / ".log.jsonl.lock").exists()

    def test_serialises_concurrent_holders(self, tmp_path):
        target = tmp_path / "log.jsonl"
        target.write_text("")
        events = []
        barrier = threading.Barrier(2)

        def worker(tag):
            barrier.wait()
            with fu.jsonl_sidecar_lock(target):
                events.append(("enter", tag))
                time.sleep(0.05)
                events.append(("exit", tag))

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start(); t2.start(); t1.join(); t2.join()

        # Either A or B goes first, but each owner's enter/exit must
        # bracket the other's pair — they cannot interleave.
        first_enter = events[0][1]
        assert events[1] == ("exit", first_enter)
        assert events[2][0] == "enter"
        assert events[3][0] == "exit"


# ---------------------------------------------------------------------------
# atomic_append_jsonl still works after refactor
# ---------------------------------------------------------------------------


class TestAtomicAppend:
    def test_single_append_round_trip(self, tmp_path):
        target = tmp_path / "log.jsonl"
        fu.atomic_append_jsonl(target, {"ts": "2026-05-11T12:00:00Z", "i": 1})
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["i"] == 1

    def test_concurrent_appends_no_torn_lines(self, tmp_path):
        target = tmp_path / "log.jsonl"
        n_threads = 8
        per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker(thread_id):
            barrier.wait()
            for i in range(per_thread):
                fu.atomic_append_jsonl(target, {
                    "ts": _dt.datetime.now(_dt.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"),
                    "tid": thread_id, "i": i,
                })

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()

        lines = target.read_text().strip().split("\n")
        # Every line must be valid JSON (no torn lines).
        records = [json.loads(line) for line in lines]
        assert len(records) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Rotate-vs-append race regression — the primary fix
# ---------------------------------------------------------------------------


class TestRotateAppendRace:
    def test_appends_during_rotate_are_preserved(self, tmp_path, monkeypatch):
        """8 threads append while rotate_jsonl runs. No append should
        be lost — the count after rotation must equal the seed_count
        plus appends_during_rotation.
        """
        monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
        monkeypatch.setattr(lr, "ARCHIVE_DIR", tmp_path / "archive")
        filename = "test_log.jsonl"
        filepath = tmp_path / filename

        # Seed with 50 OLD entries (>30 days) so rotation has something
        # to archive, plus 5 FRESH entries that must survive.
        now = _dt.datetime.now(_dt.timezone.utc)
        old_ts = (now - _dt.timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        fresh_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"ts": old_ts, "i": i, "tag": "old"})
                        + "\n")
            for i in range(5):
                f.write(json.dumps({"ts": fresh_ts, "i": i, "tag": "seed"})
                        + "\n")

        # Launch appender threads and a rotation thread simultaneously.
        n_appenders = 8
        per_thread = 25
        appended = []
        barrier = threading.Barrier(n_appenders + 1)

        def appender(tid):
            barrier.wait()
            for i in range(per_thread):
                entry = {"ts": _dt.datetime.now(_dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                    "tid": tid, "i": i, "tag": "race"}
                fu.atomic_append_jsonl(filepath, entry)
                appended.append(entry)

        rotation_result = []

        def rotator():
            barrier.wait()
            # Small sleep so a few appends queue first; the race is the
            # window we care about.
            time.sleep(0.01)
            rotation_result.append(lr.rotate_jsonl(
                filename, {"ts_field": "ts", "max_age_days": 30,
                           "compress": True}))

        threads = [threading.Thread(target=appender, args=(t,))
                   for t in range(n_appenders)]
        rotator_thread = threading.Thread(target=rotator)
        for t in threads: t.start()
        rotator_thread.start()
        for t in threads: t.join()
        rotator_thread.join()

        # Read survivors. 5 seed + n_appenders*per_thread race appends.
        survivors = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    survivors.append(json.loads(line))

        seed_count = sum(1 for r in survivors if r.get("tag") == "seed")
        race_count = sum(1 for r in survivors if r.get("tag") == "race")
        old_count = sum(1 for r in survivors if r.get("tag") == "old")

        assert seed_count == 5, f"seed appends lost: {seed_count}/5"
        assert race_count == n_appenders * per_thread, \
            f"race appends lost: {race_count}/{n_appenders * per_thread}"
        assert old_count == 0, f"old entries should have been archived"

        # Rotation also ran exactly once.
        assert len(rotation_result) == 1
        assert rotation_result[0]["status"] == "rotated"
        assert rotation_result[0]["archived"] == 50

    def test_rotate_fsyncs_tmp_before_replace(self, tmp_path, monkeypatch):
        """Regression for the missing fsync before os.replace."""
        monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
        monkeypatch.setattr(lr, "ARCHIVE_DIR", tmp_path / "archive")
        filename = "test_log.jsonl"
        filepath = tmp_path / filename
        # Seed with old entries to force a rotation.
        old_ts = (_dt.datetime.now(_dt.timezone.utc)
                  - _dt.timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"ts": old_ts, "i": i}) + "\n")

        fsync_calls = []
        real_fsync = lr.os.fsync

        def tracking_fsync(fd):
            fsync_calls.append(fd)
            return real_fsync(fd)

        with patch.object(lr.os, "fsync", side_effect=tracking_fsync):
            lr.rotate_jsonl(filename, {"ts_field": "ts",
                                       "max_age_days": 30,
                                       "compress": True})

        # At least one fsync ran (the tmp file before replace).
        assert len(fsync_calls) >= 1


# ---------------------------------------------------------------------------
# Edge cases preserved across the refactor
# ---------------------------------------------------------------------------


class TestRotateEdgeCases:
    def test_missing_file_returns_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
        result = lr.rotate_jsonl("does_not_exist.jsonl",
                                  {"ts_field": "ts", "max_age_days": 30})
        assert result["status"] == "not_found"

    def test_nothing_to_archive_returns_early(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
        filename = "test_log.jsonl"
        filepath = tmp_path / filename
        fresh_ts = _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"ts": fresh_ts, "i": i}) + "\n")
        result = lr.rotate_jsonl(filename,
                                  {"ts_field": "ts", "max_age_days": 30})
        assert result["status"] == "nothing_to_archive"
        assert result["archived"] == 0

    def test_dry_run_does_not_rewrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
        monkeypatch.setattr(lr, "ARCHIVE_DIR", tmp_path / "archive")
        filename = "test_log.jsonl"
        filepath = tmp_path / filename
        old_ts = (_dt.datetime.now(_dt.timezone.utc)
                  - _dt.timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"ts": old_ts, "i": i}) + "\n")
        before = filepath.read_text()
        result = lr.rotate_jsonl(filename,
                                  {"ts_field": "ts", "max_age_days": 30},
                                  dry_run=True)
        assert result["status"] == "dry_run"
        assert filepath.read_text() == before
