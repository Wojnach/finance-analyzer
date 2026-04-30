"""Tests for portfolio/mstr_precompute.py (NAV premium math + schema)."""
from __future__ import annotations

import json
import time

from portfolio import mstr_precompute as mp


class TestNavPremiumMath:
    def test_typical_premium(self):
        # MSTR at $400, BTC at $105k, 471k holdings, $8.5B debt, 287M shares
        nav = mp._compute_nav_premium(400.0, 105_000.0, 471_107,
                                       8.5e9, 287e6)
        assert nav["premium"] is not None
        assert nav["premium"] > 0  # historically MSTR trades at premium to BTC NAV
        assert nav["market_cap_usd"] == round(400 * 287e6, 2)
        # net_btc_nav = 471107 * 105000 - 8.5e9
        expected_nav = 471_107 * 105_000.0 - 8.5e9
        assert nav["net_btc_nav_usd"] == round(expected_nav, 2)

    def test_premium_none_on_negative_nav(self):
        # Underwater: holdings * price < debt
        nav = mp._compute_nav_premium(400.0, 1.0, 100, 1e9, 1e6)
        assert nav["premium"] is None  # net_btc_nav_usd <= 0

    def test_premium_none_on_missing_inputs(self):
        nav = mp._compute_nav_premium(None, 105_000.0, 471_107, 8.5e9, 287e6)
        assert nav["premium"] is None
        nav = mp._compute_nav_premium(400.0, None, 471_107, 8.5e9, 287e6)
        assert nav["premium"] is None

    def test_zero_shares_returns_none(self):
        nav = mp._compute_nav_premium(400.0, 105_000.0, 471_107, 8.5e9, 0)
        assert nav["premium"] is None

    def test_returned_dict_has_expected_keys(self):
        nav = mp._compute_nav_premium(400.0, 105_000.0, 471_107, 8.5e9, 287e6)
        for k in ("mstr_price", "btc_price", "btc_holdings", "debt_usd",
                  "shares_outstanding", "net_btc_nav_usd",
                  "market_cap_usd", "premium"):
            assert k in nav


class TestPrecomputeWriting:
    def test_writes_output(self, tmp_path, monkeypatch):
        out_file = tmp_path / "mstr_deep.json"
        state_file = tmp_path / "mstr_state.json"
        monkeypatch.setattr(mp, "_OUTPUT_FILE", str(out_file))
        monkeypatch.setattr(mp, "_STATE_FILE", str(state_file))

        # Stub fetch to bypass network
        monkeypatch.setattr(mp, "_fetch_market_data", lambda c=None: {
            "mstr_price": 400.0, "btc_price_usd": 105_000.0,
            "mstr_change_pct": 1.5, "mstr_volume": 1_000_000,
            "mstr_52w_high": 500, "mstr_52w_low": 200,
            "btc_24h_pct": 2.5, "options": None,
            "correlation_btc_30d": 0.85, "short_interest_pct": 0.18,
            "analyst_consensus": "buy",
            "btc_holdings": 471_107, "debt_usd": 8.5e9,
            "shares_outstanding": 287e6,
        })

        result = mp.precompute()
        assert out_file.exists()
        content = json.loads(out_file.read_text())
        assert content["schema_version"] == 1
        assert content["nav"]["premium"] is not None
        assert content["correlation_btc_30d"] == 0.85
        assert content["analyst_consensus"] == "buy"

    def test_cache_gate(self, tmp_path, monkeypatch):
        out_file = tmp_path / "out.json"
        state_file = tmp_path / "st.json"
        monkeypatch.setattr(mp, "_OUTPUT_FILE", str(out_file))
        monkeypatch.setattr(mp, "_STATE_FILE", str(state_file))
        state_file.write_text(json.dumps({
            "last_run_epoch": time.time() - 1,
            "last_run_iso": "now", "status": "ok",
        }))
        assert mp.maybe_precompute_mstr({}) is None
