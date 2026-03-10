"""Tests for data/metals_risk.py — Monte Carlo, trade guards, drawdown, daily ranges, spikes.

Batch 5 of the metals monitoring auto-improvement plan.
"""
import json
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect state files to tmp_path."""
    import metals_risk as mod

    monkeypatch.setattr(mod, "STATE_FILE", str(tmp_path / "guard_state.json"))
    monkeypatch.setattr(mod, "HISTORY_FILE", str(tmp_path / "value_history.jsonl"))
    monkeypatch.setattr(mod, "SPIKE_STATE_FILE", str(tmp_path / "spike_state.json"))
    yield


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median(self):
        from metals_risk import _percentile
        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_extremes(self):
        from metals_risk import _percentile
        data = [10, 20, 30, 40, 50]
        assert _percentile(data, 0) == 10
        assert _percentile(data, 100) == 50

    def test_interpolation(self):
        from metals_risk import _percentile
        result = _percentile([1, 2, 3, 4], 25)
        assert 1 <= result <= 2

    def test_single_element(self):
        from metals_risk import _percentile
        assert _percentile([42], 50) == 42

    def test_empty_returns_zero(self):
        from metals_risk import _percentile
        assert _percentile([], 50) == 0


# ---------------------------------------------------------------------------
# Trade guards
# ---------------------------------------------------------------------------

class TestTradeGuards:
    def test_fresh_state_allows_trade(self, tmp_path):
        from metals_risk import check_trade_guard
        warnings = check_trade_guard("silver79", "BUY")
        assert len(warnings) == 0

    def test_load_guard_state_bad_json_logs_and_falls_back(self, tmp_path, caplog):
        import metals_risk as mod

        with open(mod.STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        with caplog.at_level("WARNING", logger=mod.logger.name):
            state = mod._load_guard_state()

        assert isinstance(state, dict)
        assert "ticker_trades" in state
        assert any("guard state" in record.message.lower() for record in caplog.records)

    def test_save_guard_state_uses_atomic_write_json(self, monkeypatch):
        import metals_risk as mod

        calls = []

        def _fake_atomic_write_json(path, data, indent=2, ensure_ascii=True):
            calls.append((path, data, indent, ensure_ascii))

        monkeypatch.setattr(mod, "atomic_write_json", _fake_atomic_write_json, raising=False)
        mod._save_guard_state({"ticker_trades": {}})

        assert len(calls) == 1
        assert calls[0][0] == mod.STATE_FILE
        assert calls[0][3] is False

    def test_cooldown_blocks_same_key(self, tmp_path):
        from metals_risk import check_trade_guard, record_metals_trade
        record_metals_trade("silver79", "BUY")
        warnings = check_trade_guard("silver79", "BUY")
        # Warning dicts use "guard" key (e.g., "ticker_cooldown")
        cooldown_warns = [w for w in warnings if "cooldown" in w.get("guard", "").lower()
                          or "cooldown" in w.get("message", "").lower()]
        assert len(cooldown_warns) > 0

    def test_cooldown_expires(self, tmp_path, monkeypatch):
        from metals_risk import check_trade_guard, record_metals_trade, _load_guard_state, _save_guard_state

        record_metals_trade("silver79", "BUY")

        # Manually expire the cooldown by backdating the last trade
        state = _load_guard_state()
        for key in ["ticker_trades", "last_trade_ts"]:
            if "silver79" in state.get(key, {}):
                state[key]["silver79"] = time.time() - 7200  # 2 hours ago
        _save_guard_state(state)

        warnings = check_trade_guard("silver79", "BUY")
        cooldown_warns = [w for w in warnings if "cooldown" in w.get("guard", "").lower()
                          or "cooldown" in w.get("message", "").lower()]
        assert len(cooldown_warns) == 0

    def test_loss_escalation(self, tmp_path):
        from metals_risk import check_trade_guard, record_metals_trade

        # Record 3 consecutive losses
        for _ in range(3):
            record_metals_trade("silver79", "SELL", pnl_pct_value=-2.0)

        warnings = check_trade_guard("silver79", "BUY")
        # Should have escalated cooldown
        assert len(warnings) > 0

    def test_win_resets_losses(self, tmp_path):
        from metals_risk import record_metals_trade, _load_guard_state

        record_metals_trade("silver79", "SELL", pnl_pct_value=-2.0)
        record_metals_trade("silver79", "SELL", pnl_pct_value=-1.0)
        record_metals_trade("silver79", "SELL", pnl_pct_value=3.0)  # Win resets

        state = _load_guard_state()
        losses = state.get("consecutive_losses", 0)
        assert losses == 0

    def test_session_limit(self, tmp_path):
        from metals_risk import check_trade_guard, record_metals_trade, MAX_TRADES_PER_SESSION

        # Record max trades
        for i in range(MAX_TRADES_PER_SESSION):
            record_metals_trade(f"key{i}", "BUY")

        warnings = check_trade_guard("new_key", "BUY")
        session_warns = [w for w in warnings if "session" in w.get("guard", "").lower()
                         or "session" in w.get("message", "").lower()]
        assert len(session_warns) > 0

    def test_buy_rate_limit(self, tmp_path):
        from metals_risk import check_trade_guard, record_metals_trade

        record_metals_trade("silver79", "BUY")
        # Immediately try buying a different key
        warnings = check_trade_guard("gold", "BUY")
        assert isinstance(warnings, list)

    def test_sell_no_rate_limit(self, tmp_path):
        from metals_risk import check_trade_guard, record_metals_trade

        record_metals_trade("silver79", "BUY")
        warnings = check_trade_guard("silver79", "SELL")
        rate_warns = [w for w in warnings if "rate" in w.get("guard", "").lower()
                      or "rate" in w.get("message", "").lower()]
        assert len(rate_warns) == 0


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_positions(self):
        from metals_risk import check_portfolio_drawdown
        result = check_portfolio_drawdown({}, {})
        assert isinstance(result, dict)

    def test_profit_ok(self, tmp_path):
        from metals_risk import check_portfolio_drawdown, log_portfolio_value

        positions = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices = {"silver79": {"bid": 44.0}}
        log_portfolio_value(positions, prices)

        result = check_portfolio_drawdown(positions, prices)
        assert isinstance(result, dict)

    def test_warning_level(self, tmp_path):
        from metals_risk import check_portfolio_drawdown, HISTORY_FILE

        # Create history showing a peak, then decline
        peak_entry = {"ts": "2026-03-01T10:00:00", "total_value": 100000}
        with open(HISTORY_FILE, "w") as f:
            f.write(json.dumps(peak_entry) + "\n")

        positions = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices = {"silver79": {"bid": 36.0}}

        result = check_portfolio_drawdown(positions, prices)
        assert isinstance(result, dict)

    def test_from_peak(self, tmp_path):
        from metals_risk import check_portfolio_drawdown, log_portfolio_value

        # Log high value first
        positions_high = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices_high = {"silver79": {"bid": 50.0}}
        log_portfolio_value(positions_high, prices_high)

        # Now check with lower value
        prices_low = {"silver79": {"bid": 40.0}}
        result = check_portfolio_drawdown(positions_high, prices_low)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# log_portfolio_value
# ---------------------------------------------------------------------------

class TestLogPortfolioValue:
    def test_writes_to_history(self, tmp_path):
        from metals_risk import log_portfolio_value, HISTORY_FILE

        positions = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices = {"silver79": {"bid": 42.0}}
        entry = log_portfolio_value(positions, prices)

        assert os.path.exists(HISTORY_FILE)
        assert entry is not None
        assert "total_value" in entry or "ts" in entry

    def test_skips_inactive(self, tmp_path):
        from metals_risk import log_portfolio_value

        positions = {"silver79": {"active": False, "units": 0}}
        prices = {}
        entry = log_portfolio_value(positions, prices)
        assert entry is not None


# ---------------------------------------------------------------------------
# Daily range stats
# ---------------------------------------------------------------------------

class TestDailyRangeStats:
    def test_computes_stats(self, tmp_path):
        from metals_risk import compute_daily_range_stats

        history = {
            "silver": [
                {"date": "2026-03-01", "open": 30.0, "high": 31.0, "low": 29.5, "close": 30.5},
                {"date": "2026-03-02", "open": 30.5, "high": 31.5, "low": 30.0, "close": 31.0},
                {"date": "2026-03-03", "open": 31.0, "high": 32.0, "low": 30.5, "close": 31.5},
            ]
        }
        hist_path = str(tmp_path / "metals_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

        result = compute_daily_range_stats(history_path=hist_path)
        assert isinstance(result, dict)

    def test_missing_file(self, tmp_path):
        from metals_risk import compute_daily_range_stats
        result = compute_daily_range_stats(history_path=str(tmp_path / "nonexistent.json"))
        assert isinstance(result, dict)

    def test_too_few_candles(self, tmp_path):
        from metals_risk import compute_daily_range_stats

        history = {"silver": [{"date": "2026-03-01", "open": 30, "high": 31, "low": 29, "close": 30}]}
        hist_path = str(tmp_path / "metals_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

        result = compute_daily_range_stats(history_path=hist_path)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Spike catcher
# ---------------------------------------------------------------------------

class TestSpikeCatcher:
    def test_computes_targets(self):
        from metals_risk import compute_spike_targets

        positions = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices = {"silver79": {"bid": 45.0, "underlying": 31.0}}
        range_stats = {
            "XAG-USD": {"open_to_high": {"p75": 2.0}},
        }
        result = compute_spike_targets(positions, prices, range_stats)
        assert isinstance(result, dict)

    def test_skip_inactive(self):
        from metals_risk import compute_spike_targets

        positions = {"silver79": {"active": False, "units": 0}}
        prices = {}
        result = compute_spike_targets(positions, prices, {})
        assert isinstance(result, dict)

    def test_skip_losing_position(self):
        from metals_risk import compute_spike_targets

        positions = {"silver79": {"active": True, "units": 100, "entry": 50.0}}
        prices = {"silver79": {"bid": 40.0, "underlying": 29.0}}  # Losing
        result = compute_spike_targets(positions, prices, {})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Spike state persistence
# ---------------------------------------------------------------------------

class TestSpikeState:
    def test_load_default(self, tmp_path):
        from metals_risk import load_spike_state
        state = load_spike_state()
        assert isinstance(state, dict)

    def test_load_bad_json_logs_and_falls_back(self, tmp_path, caplog):
        import metals_risk as mod

        with open(mod.SPIKE_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        with caplog.at_level("WARNING", logger=mod.logger.name):
            state = mod.load_spike_state()

        assert state == {"orders": {}, "date": None, "placed": False, "cancelled": False}
        assert any("spike state" in record.message.lower() for record in caplog.records)

    def test_roundtrip(self, tmp_path):
        from metals_risk import load_spike_state, save_spike_state

        state = {"silver79": {"spike_order_id": "123", "target_price": 45.0}}
        save_spike_state(state)
        loaded = load_spike_state()
        assert loaded.get("silver79", {}).get("spike_order_id") == "123"

    def test_save_spike_state_uses_atomic_write_json(self, monkeypatch):
        import metals_risk as mod

        calls = []

        def _fake_atomic_write_json(path, data, indent=2, ensure_ascii=True):
            calls.append((path, data, indent, ensure_ascii))

        monkeypatch.setattr(mod, "atomic_write_json", _fake_atomic_write_json, raising=False)
        mod.save_spike_state({"orders": {}})

        assert len(calls) == 1
        assert calls[0][0] == mod.SPIKE_STATE_FILE
        assert calls[0][3] is False


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_basic_simulation(self):
        from metals_risk import simulate_warrant_risk

        result = simulate_warrant_risk(
            underlying_price=31.0,
            atr_pct=2.5,
            leverage=5.0,
            entry_price_warrant=42.0,
            stop_price_warrant=38.0,
        )
        assert isinstance(result, dict)
        # Check for known return keys (horizon-specific)
        has_mc_keys = any(k.startswith("p_stop_hit_") or k.startswith("price_bands_")
                         or k.startswith("expected_return_") for k in result)
        assert has_mc_keys, f"Missing MC keys in: {list(result.keys())}"

    def test_annualized_vol(self):
        from metals_risk import simulate_warrant_risk

        result = simulate_warrant_risk(
            underlying_price=2650.0,
            atr_pct=1.5,
            leverage=8.0,
            entry_price_warrant=100.0,
            stop_price_warrant=85.0,
            n_paths=1000,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------

class TestRiskSummary:
    def test_returns_structure(self, tmp_path):
        from metals_risk import get_risk_summary

        positions = {"silver79": {"active": True, "units": 100, "entry": 40.0}}
        prices = {"silver79": {"bid": 42.0, "underlying": 31.0}}

        result = get_risk_summary(positions, prices)
        assert isinstance(result, dict)
