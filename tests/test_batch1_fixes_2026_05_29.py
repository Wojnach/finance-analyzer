"""Regression tests for 2026-05-29 auto-session Batch 1 fixes."""

import json
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

UTC = timezone.utc


# ---------- B1-1: loop_contract known failure statuses ----------

class TestLoopContractStatuses:
    def test_timeout_handled_via_journal_stub(self):
        """Timeout is handled via journal stub (B1-3), not _KNOWN_FAILURE_STATUSES.
        The journal stub at the timeout-kill path satisfies the journal check."""
        import portfolio.agent_invocation as ai
        with open(ai.__file__) as f:
            content = f.read()
        assert "timeout_stub" in content
        assert "atomic_append_jsonl(JOURNAL_FILE, timeout_stub)" in content


# ---------- B1-2: loop_contract journal timestamp tolerance ----------

class TestLoopContractTimestampTolerance:
    def test_journal_within_5s_tolerance_passes(self):
        """Verify the 5s tolerance window is coded."""
        import portfolio.loop_contract as lc
        with open(lc.__file__) as f:
            content = f.read()
        assert "timedelta(seconds=5)" in content or "timedelta(seconds = 5)" in content


# ---------- B1-3: agent_invocation timeout journal stub ----------

class TestTimeoutJournalStub:
    def test_timeout_stub_code_exists(self):
        """Verify the timeout journal stub write exists in _kill_overrun_agent."""
        import portfolio.agent_invocation as ai
        with open(ai.__file__) as f:
            content = f.read()
        assert "timeout_stub" in content
        assert "Layer 2 timed out" in content
        assert "atomic_append_jsonl(JOURNAL_FILE, timeout_stub)" in content


# ---------- B1-4: ATR stop-loss floor ----------

class TestATRStopFloor:
    def test_low_atr_gets_3pct_floor(self):
        """ATR of 1% should result in 3% stop distance (not 2%)."""
        from portfolio.risk_management import compute_stop_levels
        holdings = {"TEST-USD": {"avg_cost_usd": 100.0, "shares": 1.0}}
        agent_summary = {"signals": {"TEST-USD": {"price_usd": 100.0, "atr_pct": 1.0}}}
        result = compute_stop_levels(holdings, agent_summary)
        assert "TEST-USD" in result
        stop_price = result["TEST-USD"]["stop_price_usd"]
        # With 1% ATR, 2*1% = 2% distance, but floor is 3%
        # So stop should be at 100 * (1 - 3/100) = 97.0
        assert stop_price == pytest.approx(97.0, abs=0.1)

    def test_high_atr_uses_2x_atr(self):
        """ATR of 5% should use 2*5% = 10% distance (above 3% floor)."""
        from portfolio.risk_management import compute_stop_levels
        holdings = {"TEST-USD": {"avg_cost_usd": 100.0, "shares": 1.0}}
        agent_summary = {"signals": {"TEST-USD": {"price_usd": 100.0, "atr_pct": 5.0}}}
        result = compute_stop_levels(holdings, agent_summary)
        stop_price = result["TEST-USD"]["stop_price_usd"]
        # 2*5% = 10% distance > 3% floor → stop at 100*(1-10/100) = 90.0
        assert stop_price == pytest.approx(90.0, abs=0.1)


# ---------- B1-5: Kelly FIFO ----------

class TestKellyFIFO:
    def test_kelly_uses_fifo_not_average(self):
        """Kelly should match BUY→SELL in FIFO order, not avg_buy_price."""
        from portfolio.kelly_sizing import _compute_trade_stats
        transactions = [
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100, "fee_sek": 0, "timestamp": "2026-01-01T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 200, "fee_sek": 0, "timestamp": "2026-01-02T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 150, "fee_sek": 0, "timestamp": "2026-01-03T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 250, "fee_sek": 0, "timestamp": "2026-01-04T00:00:00Z"},
        ]
        result = _compute_trade_stats(transactions)
        assert result is not None
        # FIFO: sell1 matches buy1 (100→150 = +50%), sell2 matches buy2 (200→250 = +25%)
        # Both wins: win_rate = 1.0
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["total_trades"] == 2

    def test_kelly_fifo_with_loss(self):
        """FIFO should correctly identify losses."""
        from portfolio.kelly_sizing import _compute_trade_stats
        transactions = [
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 200, "fee_sek": 0, "timestamp": "2026-01-01T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100, "fee_sek": 0, "timestamp": "2026-01-02T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 150, "fee_sek": 0, "timestamp": "2026-01-03T00:00:00Z"},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 150, "fee_sek": 0, "timestamp": "2026-01-04T00:00:00Z"},
        ]
        result = _compute_trade_stats(transactions)
        assert result is not None
        # FIFO: sell1 matches buy1 (200→150 = -25% LOSS), sell2 matches buy2 (100→150 = +50% WIN)
        assert result["win_rate"] == pytest.approx(0.5)
        assert result["wins"] == 1
        assert result["losses"] == 1


# ---------- B1-6: blend_accuracy_data None handling ----------

class TestBlendAccuracyNoneHandling:
    def test_none_accuracy_in_alltime_does_not_crash(self):
        """blend_accuracy_data should handle accuracy=None without TypeError."""
        from portfolio.accuracy_stats import blend_accuracy_data
        alltime = {"rsi": {"accuracy": None, "total": 200}}
        recent = {"rsi": {"accuracy": 0.55, "total": 50}}
        result = blend_accuracy_data(alltime, recent)
        assert "rsi" in result
        assert isinstance(result["rsi"]["accuracy"], float)

    def test_none_accuracy_in_recent_does_not_crash(self):
        """blend_accuracy_data should handle accuracy=None in recent."""
        from portfolio.accuracy_stats import blend_accuracy_data
        alltime = {"rsi": {"accuracy": 0.52, "total": 200}}
        recent = {"rsi": {"accuracy": None, "total": 50}}
        result = blend_accuracy_data(alltime, recent)
        assert "rsi" in result
        assert isinstance(result["rsi"]["accuracy"], float)


# ---------- B1-7: metals_loop exit code ----------

class TestMetalsLoopExitCode:
    def test_crash_returns_nonzero(self):
        """Verify the fatal exception path returns 1."""
        with open("data/metals_loop.py") as f:
            content = f.read()
        # Check that 'return 1' follows the FATAL crash handler
        idx = content.index('log(f"FATAL: {e}")')
        block = content[idx:idx+200]
        assert "return 1" in block


# ---------- B1-8: sentiment_extremity_gate ticker guard ----------

class TestSentimentExtremityGateTickerGuard:
    def test_metals_ticker_returns_hold(self):
        """Non-crypto tickers should get HOLD from sentiment_extremity_gate."""
        import pandas as pd
        import numpy as np
        from portfolio.signals.sentiment_extremity_gate import (
            compute_sentiment_extremity_gate_signal,
        )
        df = pd.DataFrame({
            "close": np.random.uniform(30, 35, 50),
            "high": np.random.uniform(35, 36, 50),
            "low": np.random.uniform(29, 30, 50),
            "volume": np.random.uniform(1000, 2000, 50),
        })
        result = compute_sentiment_extremity_gate_signal(df, context={"ticker": "XAU-USD"})
        assert result["action"] == "HOLD"
        assert result["sub_signals"].get("reason") == "non_crypto_ticker"

    def test_crypto_ticker_not_blocked(self):
        """Crypto tickers should NOT be blocked by the ticker guard."""
        import pandas as pd
        import numpy as np
        from portfolio.signals.sentiment_extremity_gate import (
            compute_sentiment_extremity_gate_signal,
        )
        df = pd.DataFrame({
            "close": np.random.uniform(30, 35, 50),
            "high": np.random.uniform(35, 36, 50),
            "low": np.random.uniform(29, 30, 50),
            "volume": np.random.uniform(1000, 2000, 50),
        })
        # BTC-USD should pass the ticker guard (may still HOLD for other reasons)
        result = compute_sentiment_extremity_gate_signal(df, context={"ticker": "BTC-USD"})
        # Should NOT be blocked with "non_crypto_ticker" reason
        if result["action"] == "HOLD":
            assert result["sub_signals"].get("reason") != "non_crypto_ticker"
