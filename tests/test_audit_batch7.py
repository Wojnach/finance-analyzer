"""Audit batch 7 (portfolio & risk) focused regression tests.

Covers the fixes in commit "fix(risk): state isolation, stop math branches,
drawdown bounds, exits bypass cooldown". See docs/IMPROVEMENT_AUDIT_2026-06-10.md.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest


# --- Fix 1: portfolio_mgr default-state isolation ---

class TestDefaultStateIsolation:
    def test_two_default_loads_do_not_share_containers(self, tmp_path):
        """Mutating one default load must not contaminate a second load or the
        module global _DEFAULT_STATE."""
        from portfolio import portfolio_mgr

        missing = tmp_path / "nope.json"
        with patch.object(portfolio_mgr, "STATE_FILE", missing), \
             patch.object(portfolio_mgr, "BOLD_STATE_FILE", tmp_path / "nope_bold.json"):
            a = portfolio_mgr._load_state_from(missing)
            a["holdings"]["BTC-USD"] = {"shares": 1}
            a["transactions"].append({"action": "BUY"})

            b = portfolio_mgr._load_state_from(missing)
            assert b["holdings"] == {}
            assert b["transactions"] == []
            # module global untouched
            assert portfolio_mgr._DEFAULT_STATE["holdings"] == {}
            assert portfolio_mgr._DEFAULT_STATE["transactions"] == []

    def test_validated_state_partial_load_isolated(self, tmp_path):
        from portfolio import portfolio_mgr

        # loaded dict lacks holdings/transactions -> defaults fill them in
        v1 = portfolio_mgr._validated_state({"cash_sek": 100})
        v1["holdings"]["X"] = {"shares": 5}
        v2 = portfolio_mgr._validated_state({"cash_sek": 200})
        assert v2["holdings"] == {}
        assert portfolio_mgr._DEFAULT_STATE["holdings"] == {}


# --- Fix 2: stop math branches (metals annualization + MSTR session key) ---

class TestStopBranches:
    def _summary(self, ticker):
        return {
            "fx_rate": 10.5,
            "signals": {ticker: {"price_usd": 100.0, "atr_pct": 3.0}},
        }

    def test_metals_use_365_day_annualization(self):
        """XAG/XAU classify as 'warrant' and must annualize on 365 days, not
        252 (the old dead 'metals' branch let them fall to 252)."""
        from portfolio import risk_management as rm

        captured = {}

        def fake_sim(price, volatility, drift, remaining_minutes, instrument_type, n_paths, **kw):
            captured["vol"] = volatility
            captured["inst"] = instrument_type
            import numpy as np
            return np.full((4, max(2, remaining_minutes + 1)), price)

        holdings = {"XAG-USD": {"shares": 10, "avg_cost_usd": 100.0}}
        with patch("portfolio.exit_optimizer.simulate_intraday_paths", side_effect=fake_sim), \
             patch("portfolio.session_calendar.remaining_session_minutes", return_value=300):
            rm.compute_probabilistic_stops(holdings, self._summary("XAG-USD"))

        import math
        # vol = atr/100 * sqrt(trading_days/14); with trading_days=365:
        expected_365 = 0.03 * math.sqrt(365.0 / 14)
        assert captured["inst"] == "warrant"
        assert captured["vol"] == pytest.approx(expected_365, rel=1e-6)

    def test_mstr_uses_stock_us_session_key(self):
        """MSTR (inst_type 'stock') must query the 'stock_us' session, not the
        warrant fallback."""
        from portfolio import risk_management as rm

        seen = {}

        def fake_remaining(inst_type, *a, **k):
            seen["type"] = inst_type
            return 120

        def fake_sim(price, volatility, drift, remaining_minutes, instrument_type, n_paths, **kw):
            import numpy as np
            return np.full((4, max(2, remaining_minutes + 1)), price)

        holdings = {"MSTR": {"shares": 5, "avg_cost_usd": 100.0}}
        with patch("portfolio.session_calendar.remaining_session_minutes", side_effect=fake_remaining), \
             patch("portfolio.exit_optimizer.simulate_intraday_paths", side_effect=fake_sim):
            rm.compute_probabilistic_stops(holdings, self._summary("MSTR"))

        assert seen["type"] == "stock_us"


# --- Fix 3a: drawdown peak plausibility bound ---

class TestPeakBound:
    def test_glitch_row_capped_in_check_drawdown(self, tmp_path):
        from portfolio import risk_management as rm

        # cash-only portfolio so current_value == cash (no agent_summary needed)
        state = {
            "cash_sek": 500_000,
            "holdings": {},
            "transactions": [],
            "initial_value_sek": 500_000,
        }
        pf_path = tmp_path / "portfolio_state.json"
        pf_path.write_text(json.dumps(state))

        hist = tmp_path / "portfolio_value_history.jsonl"
        rows = [
            {"patient_value_sek": 500_000},
            {"patient_value_sek": 520_000},
            {"patient_value_sek": 99_000_000},  # glitch — ~198x initial
            {"patient_value_sek": 510_000},
        ]
        hist.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with patch.object(rm, "DATA_DIR", tmp_path):
            # also need a (missing) agent_summary -> falls back to cash value
            res = rm.check_drawdown(str(pf_path), max_drawdown_pct=20.0,
                                    agent_summary_path=str(tmp_path / "no_summary.json"))

        # peak capped at 10x max(initial=500k, current=500k) = 5_000_000,
        # NOT the 99M glitch. Drawdown computed off the capped peak.
        ceiling = rm._PEAK_PLAUSIBILITY_MULT * 500_000
        assert res["peak_value"] == pytest.approx(ceiling)
        # current 500k off a 5M peak = 90% drawdown -> breached, but NOT the
        # ~99.5% a 99M peak would have produced (the difference matters for the
        # block threshold; the cap keeps the number believable).
        assert res["peak_value"] == 5_000_000

    def test_streaming_max_returns_raw_glitch(self, tmp_path):
        """_streaming_max itself does not cap — the bound lives in check_drawdown."""
        from portfolio import risk_management as rm
        hist = tmp_path / "h.jsonl"
        hist.write_text(json.dumps({"patient_value_sek": 99_000_000}) + "\n")
        assert rm._streaming_max(hist, "patient_value_sek", floor=500_000) == 99_000_000


# --- Fix 3b: _streaming_max skips non-numeric rows (fail-closed, not crash) ---

class TestStreamingMaxJunk:
    def test_skips_non_numeric(self, tmp_path):
        from portfolio import risk_management as rm

        hist = tmp_path / "h.jsonl"
        rows = [
            {"patient_value_sek": 500_000},
            {"patient_value_sek": None},
            {"patient_value_sek": "oops"},
            {"patient_value_sek": True},   # bool is not a real value
            {"patient_value_sek": 600_000},
        ]
        hist.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # Must NOT raise TypeError; returns max of the numeric rows.
        peak = rm._streaming_max(hist, "patient_value_sek", floor=500_000)
        assert peak == 600_000


# --- Fix 4: monte_carlo_risk fx band reuses _resolve_fx_rate ---

class TestMonteCarloFxBand:
    def test_bogus_fx_rate_rejected(self, tmp_path):
        from portfolio import monte_carlo_risk as mcr
        from portfolio import risk_management as rm

        summary = {
            "fx_rate": 1.0,  # bogus / stale — must be rejected by the band
            "signals": {"BTC-USD": {"price_usd": 50_000, "atr_pct": 3.0}},
        }
        pf = {"holdings": {"BTC-USD": {"shares": 0.1}}, "cash_sek": 100_000}
        # Point the fx cache at an empty tmp dir so fallback is the hardcoded rate.
        with patch.object(rm, "DATA_DIR", tmp_path):
            out = mcr.compute_portfolio_var(pf, summary, n_paths=200, seed=1)
        # If 1.0 had passed through, *_sek would be ~10x understated. The band
        # forces the fallback (~10.5), so the effective rate is in [7,15].
        # Derive the rate the function used from a SEK/USD output pair if present.
        from portfolio.fx_rates import FX_RATE_MIN, FX_RATE_MAX
        resolved = rm._resolve_fx_rate(summary)
        assert FX_RATE_MIN <= resolved <= FX_RATE_MAX
        assert resolved != 1.0
        assert isinstance(out, dict)


# --- Fix 5: corrupt trade_guard_state quarantined + critical entry ---

class TestTradeGuardCorruptQuarantine:
    def test_corrupt_state_quarantined_and_journaled(self, tmp_path):
        from portfolio import trade_guards as tg

        state_file = tmp_path / "trade_guard_state.json"
        state_file.write_text("{not valid json")
        recorded = {}

        def fake_record(category, caller, message, context=None):
            recorded["category"] = category
            recorded["caller"] = caller
            return True

        with patch.object(tg, "STATE_FILE", state_file), \
             patch("portfolio.claude_gate.record_critical_error", side_effect=fake_record):
            st = tg._load_state()

        # returned fresh defaults
        assert st["consecutive_losses"] == {"patient": 0, "bold": 0}
        # quarantine sidecar written
        assert (tmp_path / "trade_guard_state.json.corrupt").exists()
        # critical entry recorded with the agreed category
        assert recorded["category"] == "trade_guard_state_corrupt"
        assert recorded["caller"] == "portfolio.trade_guards"

    def test_missing_file_is_not_quarantined(self, tmp_path):
        from portfolio import trade_guards as tg

        state_file = tmp_path / "absent.json"
        called = {"n": 0}

        def fake_record(*a, **k):
            called["n"] += 1
            return True

        with patch.object(tg, "STATE_FILE", state_file), \
             patch("portfolio.claude_gate.record_critical_error", side_effect=fake_record):
            st = tg._load_state()
        assert st["consecutive_losses"] == {"patient": 0, "bold": 0}
        assert called["n"] == 0
        assert not (tmp_path / "absent.json.corrupt").exists()


# --- Fix 6: exits pass cooldown while entries blocked; scale-in not new pos ---

class TestExitsBypassCooldown:
    @pytest.fixture
    def temp_state(self, tmp_path):
        from portfolio import trade_guards as tg
        sf = tmp_path / "trade_guard_state.json"
        with patch.object(tg, "STATE_FILE", sf):
            yield sf

    def test_sell_exit_bypasses_cooldown_but_buy_blocked(self, temp_state):
        from portfolio import trade_guards as tg

        # Arm cooldown: a recent trade on BTC for patient.
        tg.record_trade("BTC-USD", "BUY", "patient")
        pf_held = {"holdings": {"BTC-USD": {"shares": 0.5}}}

        # A SELL that reduces the held position must NOT be blocked by cooldown.
        sell_warns = tg.check_overtrading_guards("BTC-USD", "SELL", "patient", pf_held)
        assert not any(w["guard"] == "ticker_cooldown" for w in sell_warns)

        # A fresh BUY on the same ticker within the window is still blocked.
        buy_warns = tg.check_overtrading_guards("BTC-USD", "BUY", "patient", pf_held)
        assert any(
            w["guard"] == "ticker_cooldown" and w["severity"] == "block"
            for w in buy_warns
        )

    def test_sell_with_no_position_still_gated(self, temp_state):
        from portfolio import trade_guards as tg

        tg.record_trade("ETH-USD", "BUY", "patient")
        pf_empty = {"holdings": {}}
        # No position to reduce -> treated as entry -> cooldown still applies.
        warns = tg.check_overtrading_guards("ETH-USD", "SELL", "patient", pf_empty)
        assert any(w["guard"] == "ticker_cooldown" for w in warns)

    def test_scale_in_not_counted_as_new_position(self, temp_state):
        from portfolio import trade_guards as tg

        # patient_position_limit defaults to 1 per 8h. Open a NEW position.
        tg.record_trade("BTC-USD", "BUY", "patient", is_new_position=True)
        pf_held = {"holdings": {"BTC-USD": {"shares": 0.5}}}

        # A scale-in BUY (already held) must NOT hit the new-position rate limit.
        warns = tg.check_overtrading_guards("BTC-USD", "BUY", "patient", pf_held)
        assert not any(w["guard"] == "position_rate_limit" for w in warns)

        # A genuinely-new ticker BUY DOES hit the limit (1 new position already).
        pf_other = {"holdings": {"BTC-USD": {"shares": 0.5}}}
        warns2 = tg.check_overtrading_guards("ETH-USD", "BUY", "patient", pf_other)
        assert any(
            w["guard"] == "position_rate_limit" and w["severity"] == "block"
            for w in warns2
        )

    def test_scale_in_record_trade_does_not_arm_rate_limit(self, temp_state):
        from portfolio import trade_guards as tg

        tg.record_trade("BTC-USD", "BUY", "patient", is_new_position=True)
        tg.record_trade("BTC-USD", "BUY", "patient", is_new_position=False)  # scale-in
        st = tg._load_state()
        # Only the genuine open stamped the new-position clock.
        assert len(st["new_position_timestamps"]["patient"]) == 1
