"""Tests for data/oil_loop.py.

Mirrors test_crypto_swing_trader.py / test_metals_loop_singleton.py pattern.
Verifies:
  - Singleton lock acquire/release.
  - Stale lock cleanup (PID-not-alive path).
  - fetch_live_prices uses portfolio.price_source (NOT Binance HTTP directly).
  - CLI argument parsing.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import oil_loop


# ---------------------------------------------------------------------------
# Singleton lock
# ---------------------------------------------------------------------------
def test_acquire_singleton_lock_creates_lockfile(tmp_path):
    lock = tmp_path / "oil.lock"
    handle = oil_loop.acquire_singleton_lock(str(lock))
    try:
        assert handle is not None
        assert lock.exists()
        # Lock file contains the PID
        assert lock.read_text().strip() == str(os.getpid())
    finally:
        oil_loop.release_singleton_lock(handle)


def test_release_singleton_lock_removes_lockfile(tmp_path):
    lock = tmp_path / "oil.lock"
    handle = oil_loop.acquire_singleton_lock(str(lock))
    oil_loop.release_singleton_lock(handle)
    assert not lock.exists()


def test_acquire_singleton_lock_rejects_when_held_by_live_pid(tmp_path):
    lock = tmp_path / "oil.lock"
    h1 = oil_loop.acquire_singleton_lock(str(lock))
    try:
        # Second attempt while first holds the lock — must return None
        h2 = oil_loop.acquire_singleton_lock(str(lock))
        assert h2 is None
    finally:
        oil_loop.release_singleton_lock(h1)


def test_acquire_singleton_lock_clears_stale_lock(tmp_path, monkeypatch):
    """If lock file's PID is dead, we should remove and retry once."""
    lock = tmp_path / "oil.lock"
    # Plant a stale lock with a PID that's almost certainly not alive
    stale_pid = 999999
    lock.write_text(str(stale_pid))

    # Force _pid_alive to return False so the stale-lock path triggers
    monkeypatch.setattr(oil_loop, "_pid_alive", lambda pid: False)

    handle = oil_loop.acquire_singleton_lock(str(lock))
    try:
        assert handle is not None
        # Lock is now ours — file contains *our* PID
        assert lock.read_text().strip() == str(os.getpid())
    finally:
        oil_loop.release_singleton_lock(handle)


# ---------------------------------------------------------------------------
# fetch_live_prices — must route via portfolio.price_source, not Binance HTTP
# ---------------------------------------------------------------------------
def test_fetch_live_prices_uses_price_source_router():
    """Critical: oil prices must come from portfolio.price_source.fetch_klines,
    NOT from a direct Binance call (oil's CL=F is not a Binance spot symbol)."""
    fake_df = pd.DataFrame({
        "open": [78.0, 78.2, 78.4],
        "high": [78.5, 78.7, 78.9],
        "low": [77.8, 78.0, 78.1],
        "close": [78.3, 78.45, 78.42],
        "volume": [1000, 1100, 1200],
    })

    with patch("portfolio.price_source.fetch_klines",
               return_value=fake_df) as mock_fetch:
        prices = oil_loop.fetch_live_prices()

    mock_fetch.assert_called()
    # First positional arg must be the yfinance symbol from cfg.DATA_SOURCES
    first_call_args = mock_fetch.call_args_list[0][0]
    assert first_call_args[0] == "CL=F"

    # Returned dict keyed by config ticker
    assert "OIL-USD" in prices
    assert prices["OIL-USD"] == pytest.approx(78.42)


def test_fetch_live_prices_returns_empty_when_router_returns_none():
    with patch("portfolio.price_source.fetch_klines", return_value=None):
        prices = oil_loop.fetch_live_prices()
    assert prices == {}


def test_fetch_live_prices_returns_empty_when_router_returns_empty_df():
    with patch("portfolio.price_source.fetch_klines",
               return_value=pd.DataFrame()):
        prices = oil_loop.fetch_live_prices()
    assert prices == {}


def test_fetch_live_prices_handles_router_exception():
    """A raised exception per-instrument shouldn't crash the loop."""
    with patch("portfolio.price_source.fetch_klines",
               side_effect=RuntimeError("network")):
        prices = oil_loop.fetch_live_prices()
    assert prices == {}


# ---------------------------------------------------------------------------
# Signal snapshot loading
# ---------------------------------------------------------------------------
def test_load_signal_snapshot_falls_back_to_full_summary(tmp_path, monkeypatch):
    """Compact missing → fall back to data/agent_summary.json."""
    # Both files missing → empty dict
    monkeypatch.chdir(tmp_path)
    snap = oil_loop.load_signal_snapshot()
    assert snap == {}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_cli_supports_loop_once_report(monkeypatch):
    for flag in ("--loop", "--once", "--report"):
        monkeypatch.setattr(sys, "argv", ["oil_loop", flag])
        args = oil_loop._parse_args()
        attr = flag.lstrip("-")
        assert getattr(args, attr) is True


def test_cli_no_args_prints_usage_returns_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["oil_loop"])
    rc = oil_loop.main()
    assert rc == 2


# ---------------------------------------------------------------------------
# Cycle wiring
# ---------------------------------------------------------------------------
def test_run_one_cycle_returns_not_ok_when_no_prices(tmp_path, monkeypatch):
    """Empty price dict => the cycle reports ok=False with reason — the
    loop intentionally surfaces "skip this cycle" rather than pretending
    success."""
    from data import oil_swing_config as cfg
    monkeypatch.setattr(cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(cfg, "DECISIONS_LOG", str(tmp_path / "dec.jsonl"))
    monkeypatch.setattr(cfg, "TRADES_LOG", str(tmp_path / "trd.jsonl"))
    monkeypatch.setattr(cfg, "VALUE_HISTORY_LOG", str(tmp_path / "val.jsonl"))
    monkeypatch.setattr(cfg, "WARRANT_CATALOG_FILE",
                        str(tmp_path / "cat.json"))
    monkeypatch.setattr(cfg, "MOMENTUM_STATE_FILE", str(tmp_path / "mom.json"))

    from data.oil_swing_trader import OilSwingTrader

    trader = OilSwingTrader(page=None, executor=None)
    with patch.object(oil_loop, "fetch_live_prices", return_value={}), \
         patch.object(oil_loop, "load_signal_snapshot", return_value={}):
        result = oil_loop.run_one_cycle(trader)
    assert result.get("ok") is False
    assert "price" in (result.get("reason") or "").lower()


def test_run_one_cycle_returns_ok_when_prices_available(tmp_path, monkeypatch):
    """A valid price + empty signal data should complete cleanly (HOLD)."""
    from data import oil_swing_config as cfg
    monkeypatch.setattr(cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(cfg, "DECISIONS_LOG", str(tmp_path / "dec.jsonl"))
    monkeypatch.setattr(cfg, "TRADES_LOG", str(tmp_path / "trd.jsonl"))
    monkeypatch.setattr(cfg, "VALUE_HISTORY_LOG", str(tmp_path / "val.jsonl"))
    monkeypatch.setattr(cfg, "WARRANT_CATALOG_FILE",
                        str(tmp_path / "cat.json"))
    monkeypatch.setattr(cfg, "MOMENTUM_STATE_FILE", str(tmp_path / "mom.json"))

    from data.oil_swing_trader import OilSwingTrader

    trader = OilSwingTrader(page=None, executor=None)
    with patch.object(oil_loop, "fetch_live_prices",
                      return_value={"OIL-USD": 78.40}), \
         patch.object(oil_loop, "load_signal_snapshot", return_value={}):
        result = oil_loop.run_one_cycle(trader)
    assert result.get("ok") is True
