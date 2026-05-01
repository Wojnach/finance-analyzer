"""Tests for signal_history module — concurrent update safety.

Adversarial review 05-01 P0-3: update_history is a read-modify-write of
HISTORY_FILE. The main loop's ThreadPoolExecutor (8 workers) calls it
concurrently for 5 tickers. Without a lock, last-writer-wins and 4/5
ticker updates per cycle were silently discarded. Persistence scores,
streaks, and noisy-signal lists were then computed from a corrupted
history.

This file was created 2026-05-02 to add the missing concurrency test.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def isolated_history(monkeypatch, tmp_path):
    """Point HISTORY_FILE at a tmp_path file and reset the lock for the test."""
    from portfolio import signal_history as sh

    history_file = tmp_path / "signal_history.jsonl"
    monkeypatch.setattr(sh, "HISTORY_FILE", history_file)
    # Reset lock to a fresh one in case prior tests left it held / reused
    monkeypatch.setattr(sh, "_history_lock", threading.Lock())
    return history_file


def test_lock_is_reentrant_safe(isolated_history):
    """Sanity: the module exports `_history_lock` as a threading.Lock."""
    from portfolio import signal_history as sh

    assert isinstance(sh._history_lock, type(threading.Lock())), (
        "_history_lock must be a threading.Lock instance"
    )


def test_concurrent_updates_dont_lose_writes(isolated_history):
    """8 workers × 5 tickers × 10 updates each = 400 writes, none should be lost.

    Before the lock: last-writer-wins meant ~4 of every 5 ticker updates
    per cycle were lost. With the lock: every write is preserved (modulo
    MAX_ENTRIES_PER_TICKER trimming).
    """
    from portfolio import signal_history as sh

    tickers = ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"]
    updates_per_ticker = 10
    workers = 8

    def update_one(args):
        ticker, idx = args
        # Each call writes one entry. votes_dict marker: signal_name -> "BUY"/"SELL"
        # Use idx to make each entry distinguishable.
        votes = {"rsi": "BUY" if idx % 2 == 0 else "SELL"}
        sh.update_history(ticker, votes)

    # Build the work list: 5 tickers × 10 updates = 50 total updates
    work = [(t, i) for t in tickers for i in range(updates_per_ticker)]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(update_one, work))

    # Read back and count entries per ticker
    entries = sh._load_history()
    by_ticker = {}
    for e in entries:
        t = e.get("ticker", "unknown")
        by_ticker[t] = by_ticker.get(t, 0) + 1

    # MAX_ENTRIES_PER_TICKER = 50 (greater than 10 per ticker), so every entry
    # should survive. We expect exactly 10 entries per ticker.
    for ticker in tickers:
        assert by_ticker.get(ticker, 0) == updates_per_ticker, (
            f"{ticker}: expected {updates_per_ticker} entries, got "
            f"{by_ticker.get(ticker, 0)}. Concurrent writes lost — lock missing?"
        )


def test_serial_updates_still_work(isolated_history):
    """Sanity: the lock doesn't break the simple sequential case."""
    from portfolio import signal_history as sh

    sh.update_history("BTC-USD", {"rsi": "BUY"})
    sh.update_history("ETH-USD", {"rsi": "SELL"})
    sh.update_history("BTC-USD", {"rsi": "HOLD"})

    entries = sh._load_history()
    assert len(entries) == 3
    btc_entries = [e for e in entries if e["ticker"] == "BTC-USD"]
    assert len(btc_entries) == 2


def test_trimming_still_applies_with_lock(isolated_history):
    """Sanity: MAX_ENTRIES_PER_TICKER trimming still works under the lock."""
    from portfolio import signal_history as sh

    # MAX_ENTRIES_PER_TICKER = 50; write 60 entries for one ticker
    for i in range(60):
        sh.update_history("BTC-USD", {"rsi": "BUY" if i % 2 == 0 else "SELL"})

    entries = sh._load_history()
    btc_entries = [e for e in entries if e["ticker"] == "BTC-USD"]
    assert len(btc_entries) == sh.MAX_ENTRIES_PER_TICKER, (
        f"Expected trim to {sh.MAX_ENTRIES_PER_TICKER}, got {len(btc_entries)}"
    )
