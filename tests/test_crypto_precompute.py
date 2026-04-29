"""Schema + cache-gate tests for portfolio/crypto_precompute.py."""
from __future__ import annotations

import time

import pytest

from portfolio import crypto_precompute as cp


class TestBuildContext:
    def test_full_market_data(self):
        market = {
            "fear_greed": {"value": 50, "classification": "Neutral"},
            "btc_dominance": 55.2,
            "btc_price_usd": 105_000.0, "btc_24h_pct": 2.5,
            "btc_funding_rate": 0.0001, "btc_open_interest": 100000,
            "onchain_btc": {"mvrv": 1.8},
            "eth_price_usd": 3500.0, "eth_24h_pct": 1.2,
            "eth_funding_rate": 0.00008, "eth_open_interest": 50000,
            "dxy_close": 100.5, "dxy_change_pct": -0.3,
            "spy_close": 580.4, "spy_change_pct": 0.5,
            "gold_close": 4500.0, "gold_change_pct": 0.8,
        }
        ctx = cp._build_context(market, "2026-04-30T00:00:00+00:00")
        assert ctx["schema_version"] == 1
        assert ctx["generated_at"] == "2026-04-30T00:00:00+00:00"
        assert "shared" in ctx and "btc" in ctx and "eth" in ctx
        assert ctx["btc"]["price_usd"] == 105_000.0
        assert ctx["eth"]["funding_rate"] == 0.00008
        assert ctx["shared"]["fear_greed"]["value"] == 50

    def test_partial_data_passes_none_through(self):
        ctx = cp._build_context({}, "2026-04-30T00:00:00+00:00")
        assert ctx["btc"]["price_usd"] is None
        assert ctx["eth"]["price_usd"] is None
        assert ctx["shared"]["dxy"]["close"] is None

    def test_schema_keys_stable(self):
        ctx = cp._build_context({}, "2026-04-30T00:00:00+00:00")
        assert sorted(ctx.keys()) == sorted(
            ["generated_at", "schema_version", "shared", "btc", "eth"])
        assert sorted(ctx["btc"].keys()) == sorted(
            ["price_usd", "change_24h_pct", "funding_rate",
             "open_interest", "onchain"])


class TestCacheGate:
    def test_skips_when_recent(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        out_file = tmp_path / "out.json"
        monkeypatch.setattr(cp, "_STATE_FILE", str(state_file))
        monkeypatch.setattr(cp, "_OUTPUT_FILE", str(out_file))

        # Seed state — pretends we ran 1 second ago
        import json
        state_file.write_text(json.dumps({
            "last_run_epoch": time.time() - 1,
            "last_run_iso": "now",
            "status": "ok",
        }))
        result = cp.maybe_precompute_crypto({})
        assert result is None  # gate engaged

    def test_runs_when_stale(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        out_file = tmp_path / "out.json"
        monkeypatch.setattr(cp, "_STATE_FILE", str(state_file))
        monkeypatch.setattr(cp, "_OUTPUT_FILE", str(out_file))
        # Mock _fetch_market_data to bypass the network
        monkeypatch.setattr(cp, "_fetch_market_data", lambda c=None: {
            "fear_greed": None,
        })

        # Seed stale state — 999 hours ago
        import json
        state_file.write_text(json.dumps({
            "last_run_epoch": time.time() - 999 * 3600,
            "last_run_iso": "ages ago",
            "status": "ok",
        }))
        result = cp.maybe_precompute_crypto({})
        assert result is not None
        assert out_file.exists()
        content = json.loads(out_file.read_text())
        assert content["schema_version"] == 1


def test_module_has_cli_main_block():
    """precompute() should be callable (sanity)."""
    assert callable(cp.precompute)
    assert callable(cp.maybe_precompute_crypto)
