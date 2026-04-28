"""Tests for portfolio.signals.claude_fundamental — three-tier LLM cascade."""

import json
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.claude_fundamental import (
    _MAX_CONFIDENCE,
    SUB_SIGNAL_NAMES,
    _cache,
    _call_claude_cli,
    _extract_json,
    _get_best_result,
    _lock,
    _parse_haiku_response,
    _parse_opus_response,
    _parse_sonnet_response,
    compute_claude_fundamental_signal,
)


def _make_df(n=100):
    """Create minimal OHLCV dataframe."""
    return pd.DataFrame({
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
    })


def _reset_cache():
    """Reset the module-level cache between tests."""
    with _lock:
        for tier in ("haiku", "sonnet", "opus"):
            _cache[tier]["results"] = {}
            _cache[tier]["ts"] = 0


@pytest.fixture(autouse=True)
def clean_cache():
    """Reset cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


# --- JSON extraction ---

class TestExtractJson:
    def test_plain_json(self):
        text = '{"BTC-USD": {"action": "BUY", "confidence": 0.6}}'
        result = _extract_json(text)
        assert result["BTC-USD"]["action"] == "BUY"

    def test_markdown_code_block(self):
        text = '```json\n{"BTC-USD": {"action": "SELL", "confidence": 0.5}}\n```'
        result = _extract_json(text)
        assert result["BTC-USD"]["action"] == "SELL"

    def test_text_around_json(self):
        text = 'Here is my analysis:\n{"BTC-USD": {"action": "HOLD", "confidence": 0.0}}\nThat is all.'
        result = _extract_json(text)
        assert result["BTC-USD"]["action"] == "HOLD"

    def test_malformed_returns_empty(self):
        assert _extract_json("not json at all") == {}

    def test_empty_string(self):
        assert _extract_json("") == {}


# --- Haiku parsing ---

class TestParseHaiku:
    def test_valid_response(self):
        text = json.dumps({
            "BTC-USD": {"action": "BUY", "confidence": 0.6},
            "ETH-USD": {"action": "HOLD", "confidence": 0.0},
        })
        results = _parse_haiku_response(text)
        assert results["BTC-USD"]["action"] == "BUY"
        assert results["BTC-USD"]["confidence"] <= _MAX_CONFIDENCE
        assert results["ETH-USD"]["action"] == "HOLD"
        assert results["BTC-USD"]["indicators"]["_tier"] == "haiku"

    def test_confidence_capped(self):
        text = json.dumps({"BTC-USD": {"action": "BUY", "confidence": 0.95}})
        results = _parse_haiku_response(text)
        assert results["BTC-USD"]["confidence"] <= _MAX_CONFIDENCE

    def test_invalid_action_becomes_hold(self):
        text = json.dumps({"BTC-USD": {"action": "MAYBE", "confidence": 0.5}})
        results = _parse_haiku_response(text)
        assert results["BTC-USD"]["action"] == "HOLD"

    def test_malformed_json(self):
        results = _parse_haiku_response("not json")
        assert results == {}


# --- Sonnet parsing ---

class TestParseSonnet:
    def test_valid_response(self):
        text = json.dumps({
            "NVDA": {
                "sub_signals": {
                    "fundamental_quality": "BUY",
                    "sector_positioning": "BUY",
                    "valuation": "HOLD",
                    "catalyst_assessment": "BUY",
                    "macro_sensitivity": "HOLD",
                },
                "reasoning": "Strong AI demand cycle",
            }
        })
        results = _parse_sonnet_response(text)
        assert results["NVDA"]["action"] == "BUY"
        assert results["NVDA"]["confidence"] > 0
        assert results["NVDA"]["confidence"] <= _MAX_CONFIDENCE
        assert results["NVDA"]["sub_signals"]["fundamental_quality"] == "BUY"
        assert results["NVDA"]["indicators"]["_tier"] == "sonnet"
        assert "AI demand" in results["NVDA"]["indicators"]["reasoning"]

    def test_all_hold(self):
        text = json.dumps({
            "NVDA": {
                "sub_signals": {s: "HOLD" for s in SUB_SIGNAL_NAMES},
                "reasoning": "No strong view",
            }
        })
        results = _parse_sonnet_response(text)
        assert results["NVDA"]["action"] == "HOLD"
        assert results["NVDA"]["confidence"] == 0.0

    def test_missing_sub_signals_default_hold(self):
        text = json.dumps({
            "NVDA": {
                "sub_signals": {"fundamental_quality": "BUY"},
                "reasoning": "Partial data",
            }
        })
        results = _parse_sonnet_response(text)
        # Missing sub_signals should default to HOLD
        assert results["NVDA"]["sub_signals"]["valuation"] == "HOLD"


# --- Opus parsing ---

class TestParseOpus:
    def test_valid_response_with_contrarian(self):
        text = json.dumps({
            "MU": {
                "sub_signals": {
                    "fundamental_quality": "BUY",
                    "sector_positioning": "BUY",
                    "valuation": "BUY",
                    "catalyst_assessment": "HOLD",
                    "macro_sensitivity": "SELL",
                },
                "conviction": 0.8,
                "reasoning": "HBM supply pre-sold, earnings trajectory strong",
                "contrarian_flag": True,
            }
        })
        results = _parse_opus_response(text)
        assert results["MU"]["action"] == "BUY"
        assert results["MU"]["confidence"] > 0
        assert results["MU"]["confidence"] <= _MAX_CONFIDENCE
        assert results["MU"]["indicators"]["contrarian_flag"] is True
        assert results["MU"]["indicators"]["conviction"] == 0.8
        assert results["MU"]["indicators"]["_tier"] == "opus"

    def test_conviction_scales_confidence(self):
        text = json.dumps({
            "MU": {
                "sub_signals": {s: "BUY" for s in SUB_SIGNAL_NAMES},
                "conviction": 0.3,
                "reasoning": "Low conviction",
                "contrarian_flag": False,
            }
        })
        results = _parse_opus_response(text)
        # All BUY with low conviction should still have scaled-down confidence
        assert results["MU"]["confidence"] < _MAX_CONFIDENCE


# --- Cascade logic ---

class TestCascade:
    def test_opus_wins_over_sonnet(self):
        """When Opus has a non-HOLD result, it should win."""
        with _lock:
            _cache["opus"]["results"] = {
                "BTC-USD": {
                    "action": "BUY", "confidence": 0.6,
                    "sub_signals": {}, "indicators": {"_tier": "opus"},
                }
            }
            _cache["opus"]["ts"] = time.time()
            _cache["sonnet"]["results"] = {
                "BTC-USD": {
                    "action": "SELL", "confidence": 0.5,
                    "sub_signals": {}, "indicators": {"_tier": "sonnet"},
                }
            }
            _cache["sonnet"]["ts"] = time.time()

        result = _get_best_result("BTC-USD")
        assert result["action"] == "BUY"
        assert result["indicators"]["_tier"] == "opus"

    def test_sonnet_fallback_when_opus_hold(self):
        """When Opus says HOLD, Sonnet's vote should be used."""
        with _lock:
            _cache["opus"]["results"] = {
                "BTC-USD": {
                    "action": "HOLD", "confidence": 0.0,
                    "sub_signals": {}, "indicators": {"_tier": "opus"},
                }
            }
            _cache["opus"]["ts"] = time.time()
            _cache["sonnet"]["results"] = {
                "BTC-USD": {
                    "action": "SELL", "confidence": 0.5,
                    "sub_signals": {}, "indicators": {"_tier": "sonnet"},
                }
            }
            _cache["sonnet"]["ts"] = time.time()

        result = _get_best_result("BTC-USD")
        assert result["action"] == "SELL"
        assert result["indicators"]["_tier"] == "sonnet"

    def test_haiku_fallback_when_all_hold(self):
        """When all tiers say HOLD, return the highest-tier available."""
        with _lock:
            _cache["opus"]["results"] = {
                "BTC-USD": {
                    "action": "HOLD", "confidence": 0.0,
                    "sub_signals": {}, "indicators": {"_tier": "opus"},
                }
            }
            _cache["opus"]["ts"] = time.time()
            _cache["sonnet"]["results"] = {
                "BTC-USD": {
                    "action": "HOLD", "confidence": 0.0,
                    "sub_signals": {}, "indicators": {"_tier": "sonnet"},
                }
            }
            _cache["sonnet"]["ts"] = time.time()
            _cache["haiku"]["results"] = {
                "BTC-USD": {
                    "action": "HOLD", "confidence": 0.0,
                    "sub_signals": {}, "indicators": {"_tier": "haiku"},
                }
            }
            _cache["haiku"]["ts"] = time.time()

        result = _get_best_result("BTC-USD")
        # Should return opus (highest tier) even though all HOLD
        assert result["indicators"]["_tier"] == "opus"

    def test_unknown_ticker_returns_none(self):
        assert _get_best_result("UNKNOWN") is None


# --- Bias detection (added 2026-04-25, extended per-ticker 2026-04-28) ---

class TestPerTickerBiasDetection:
    """The global _is_tier_biased misses per-ticker patterns. The audit on
    2026-04-28 found Opus voting 100% BUY on BTC over 5 cycles while overall
    Opus rate was 70% BUY (below the 75% global threshold). Per-ticker bias
    detection catches that."""

    def test_per_ticker_biased_buy_becomes_synthesized_hold(self, tmp_path):
        """With Opus 100% BUY on BTC over recent journal entries, the cascade
        should suppress Opus's BUY into a synthesized HOLD with
        `_bias_suppressed: True` (rather than passing the biased BUY through)."""
        from portfolio.signals import claude_fundamental as cf

        # Journal: Opus 5x BUY on BTC, 5x SELL on ETH (overall balanced —
        # global check would NOT flag Opus, but per-ticker BTC will).
        log_path = tmp_path / "claude_fundamental_log.jsonl"
        rows = []
        for _ in range(5):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "BUY",
                         "is_abstention": False})
        for _ in range(5):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "ETH-USD", "action": "SELL",
                         "is_abstention": False})
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with mock.patch.object(cf, "_CF_LOG", log_path):
            with _lock:
                _cache["opus"]["results"] = {
                    "BTC-USD": {
                        "action": "BUY", "confidence": 0.7,
                        "sub_signals": {}, "indicators": {"_tier": "opus"},
                    }
                }
                _cache["opus"]["ts"] = time.time()
                _cache["sonnet"]["results"] = {
                    "BTC-USD": {
                        "action": "HOLD", "confidence": 0.0,
                        "sub_signals": {}, "indicators": {"_tier": "sonnet"},
                    }
                }
                _cache["sonnet"]["ts"] = time.time()

            assert cf._is_tier_biased("opus") is False
            assert cf._is_tier_biased_for_ticker("opus", "BTC-USD") is True

            result = _get_best_result("BTC-USD")
            # Opus's biased BUY suppressed -> HOLD; Opus is highest tier with
            # any result so cascade fallback returns Opus's synthesized HOLD.
            assert result["action"] == "HOLD"
            assert result["confidence"] == 0.0
            assert result["indicators"].get("_bias_suppressed") is True
            assert result["indicators"].get("_original_action") == "BUY"
            assert result["indicators"]["_tier"] == "opus"

    def test_per_ticker_biased_buy_lets_lower_tier_non_hold_win(self, tmp_path):
        """When Opus is per-ticker-biased on BTC (BUY) but Sonnet has a
        non-biased SELL, the cascade should pick Sonnet's SELL — not Opus's
        suppressed HOLD."""
        from portfolio.signals import claude_fundamental as cf

        log_path = tmp_path / "claude_fundamental_log.jsonl"
        rows = []
        # Opus per-ticker biased on BTC (5/5 BUY)
        for _ in range(5):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "BUY",
                         "is_abstention": False})
        # Sonnet on BTC: 3 BUY + 3 SELL (balanced, not biased)
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "sonnet",
                         "ticker": "BTC-USD", "action": "BUY",
                         "is_abstention": False})
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "sonnet",
                         "ticker": "BTC-USD", "action": "SELL",
                         "is_abstention": False})
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with mock.patch.object(cf, "_CF_LOG", log_path):
            with _lock:
                _cache["opus"]["results"] = {
                    "BTC-USD": {
                        "action": "BUY", "confidence": 0.7,
                        "sub_signals": {}, "indicators": {"_tier": "opus"},
                    }
                }
                _cache["opus"]["ts"] = time.time()
                _cache["sonnet"]["results"] = {
                    "BTC-USD": {
                        "action": "SELL", "confidence": 0.5,
                        "sub_signals": {}, "indicators": {"_tier": "sonnet"},
                    }
                }
                _cache["sonnet"]["ts"] = time.time()

            result = _get_best_result("BTC-USD")
            # Opus BUY suppressed; Sonnet's non-biased SELL wins.
            assert result["action"] == "SELL"
            assert result["indicators"]["_tier"] == "sonnet"

    def test_per_ticker_below_min_samples_not_biased(self, tmp_path):
        """4 BUY votes on BTC is below the per-ticker min-samples floor (5);
        bias should not fire."""
        from portfolio.signals import claude_fundamental as cf

        log_path = tmp_path / "claude_fundamental_log.jsonl"
        rows = [
            {"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
             "ticker": "BTC-USD", "action": "BUY", "is_abstention": False}
            for _ in range(4)
        ]
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with mock.patch.object(cf, "_CF_LOG", log_path):
            assert cf._is_tier_biased_for_ticker("opus", "BTC-USD") is False

    def test_per_ticker_balanced_not_biased(self, tmp_path):
        """3 BUY + 3 SELL on BTC: balanced, should not fire."""
        from portfolio.signals import claude_fundamental as cf

        log_path = tmp_path / "claude_fundamental_log.jsonl"
        rows = []
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "BUY",
                         "is_abstention": False})
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "SELL",
                         "is_abstention": False})
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with mock.patch.object(cf, "_CF_LOG", log_path):
            assert cf._is_tier_biased_for_ticker("opus", "BTC-USD") is False

    def test_per_ticker_check_ignores_other_tickers(self, tmp_path):
        """Opus is 100% BUY on ETH but balanced on BTC — bias check for BTC
        should not fire."""
        from portfolio.signals import claude_fundamental as cf

        log_path = tmp_path / "claude_fundamental_log.jsonl"
        rows = []
        for _ in range(10):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "ETH-USD", "action": "BUY",
                         "is_abstention": False})
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "BUY",
                         "is_abstention": False})
        for _ in range(3):
            rows.append({"ts": "2026-04-28T10:00:00+00:00", "tier": "opus",
                         "ticker": "BTC-USD", "action": "SELL",
                         "is_abstention": False})
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with mock.patch.object(cf, "_CF_LOG", log_path):
            # ETH is 100% biased
            assert cf._is_tier_biased_for_ticker("opus", "ETH-USD") is True
            # BTC is balanced
            assert cf._is_tier_biased_for_ticker("opus", "BTC-USD") is False


# --- Main compute function ---

class TestComputeSignal:
    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_claude_fundamental_signal(df, context=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_disabled_returns_hold(self):
        df = _make_df()
        result = compute_claude_fundamental_signal(
            df,
            context={
                "ticker": "BTC-USD",
                "config": {"claude_fundamental": {"enabled": False}},
            },
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @mock.patch("portfolio.signals.claude_fundamental._refresh_tier")
    def test_returns_cached_result(self, mock_refresh):
        """When cache is populated, should return result without refreshing."""
        # Pre-populate cache
        with _lock:
            _cache["haiku"]["results"] = {
                "BTC-USD": {
                    "action": "BUY", "confidence": 0.5,
                    "sub_signals": {}, "indicators": {"_tier": "haiku"},
                }
            }
            _cache["haiku"]["ts"] = time.time()
            _cache["sonnet"]["ts"] = time.time()
            _cache["opus"]["ts"] = time.time()

        df = _make_df()
        result = compute_claude_fundamental_signal(
            df,
            context={"ticker": "BTC-USD", "config": {"claude_fundamental": {"enabled": True}}},
        )
        assert result["action"] == "BUY"
        # No refresh should have happened (cache is fresh)
        mock_refresh.assert_not_called()

    @mock.patch("portfolio.signals.claude_fundamental._refresh_tier")
    def test_cooldown_prevents_refresh(self, mock_refresh):
        """Second call within TTL should use cache, not refresh."""
        with _lock:
            _cache["haiku"]["results"] = {
                "BTC-USD": {
                    "action": "SELL", "confidence": 0.4,
                    "sub_signals": {}, "indicators": {"_tier": "haiku"},
                }
            }
            _cache["haiku"]["ts"] = time.time()
            _cache["sonnet"]["ts"] = time.time()
            _cache["opus"]["ts"] = time.time()

        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"claude_fundamental": {"enabled": True}}}

        result1 = compute_claude_fundamental_signal(df, context=ctx)
        result2 = compute_claude_fundamental_signal(df, context=ctx)

        assert result1["action"] == "SELL"
        assert result2["action"] == "SELL"
        mock_refresh.assert_not_called()

    @mock.patch("portfolio.market_timing.get_market_state", return_value=("open", None, None))
    def test_expired_cache_triggers_refresh(self, mock_market):
        """Expired cache should trigger background refresh threads."""
        # Cache is at ts=0 (expired), market is "open" so refresh is allowed
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"claude_fundamental": {"enabled": True}}}
        compute_claude_fundamental_signal(df, context=ctx)
        # The function spawns daemon threads — verify the cache timestamps
        # were updated (marking them as "refreshing to prevent duplicate spawns")
        for tier in ("haiku", "sonnet", "opus"):
            assert _cache[tier]["ts"] > 0

    @mock.patch("portfolio.signals.claude_fundamental._refresh_tier", side_effect=Exception("API down"))
    def test_api_failure_graceful(self, mock_refresh):
        """API failure should return HOLD, not crash."""
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"claude_fundamental": {"enabled": True}}}
        result = compute_claude_fundamental_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_result_is_copy(self):
        """Returned result should be a copy, not a reference to cache."""
        with _lock:
            _cache["haiku"]["results"] = {
                "BTC-USD": {
                    "action": "BUY", "confidence": 0.5,
                    "sub_signals": {}, "indicators": {"_tier": "haiku"},
                }
            }
            _cache["haiku"]["ts"] = time.time()
            _cache["sonnet"]["ts"] = time.time()
            _cache["opus"]["ts"] = time.time()

        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"claude_fundamental": {"enabled": True}}}
        result = compute_claude_fundamental_signal(df, context=ctx)
        result["action"] = "MODIFIED"
        # Original cache should be unchanged
        assert _cache["haiku"]["results"]["BTC-USD"]["action"] == "BUY"


# --- Contrarian flag ---

class TestContrarianFlag:
    def test_contrarian_flag_surfaced(self):
        """Opus contrarian flag should be in indicators."""
        with _lock:
            _cache["opus"]["results"] = {
                "NVDA": {
                    "action": "SELL", "confidence": 0.6,
                    "sub_signals": {},
                    "indicators": {
                        "_tier": "opus",
                        "contrarian_flag": True,
                        "conviction": 0.9,
                        "reasoning": "Fundamentals say sell despite technical buy",
                    },
                }
            }
            _cache["opus"]["ts"] = time.time()
            _cache["sonnet"]["ts"] = time.time()
            _cache["haiku"]["ts"] = time.time()

        df = _make_df()
        ctx = {"ticker": "NVDA", "config": {"claude_fundamental": {"enabled": True}}}
        result = compute_claude_fundamental_signal(df, context=ctx)
        assert result["indicators"]["contrarian_flag"] is True
        assert result["indicators"]["_tier"] == "opus"


# --- CLI layer (now routes through claude_gate) ---

class TestCallClaudeCli:
    @mock.patch("portfolio.claude_gate.invoke_claude_text")
    def test_success(self, mock_gate):
        """Successful CLI call returns stdout text."""
        mock_gate.return_value = (
            '{"BTC-USD": {"action": "BUY", "confidence": 0.6}}',
            True,
            0,
        )
        result = _call_claude_cli("haiku", "test prompt", timeout=30)
        assert "BTC-USD" in result
        mock_gate.assert_called_once_with(
            prompt="test prompt",
            caller="claude_fundamental_haiku",
            model="haiku",
            timeout=30,
        )

    @mock.patch("portfolio.claude_gate.invoke_claude_text")
    def test_nonzero_exit_raises(self, mock_gate):
        """Non-zero exit code should raise RuntimeError."""
        mock_gate.return_value = ("", False, 1)
        with pytest.raises(RuntimeError, match="claude_gate returned exit_code=1"):
            _call_claude_cli("haiku", "test", timeout=30)

    @mock.patch("portfolio.claude_gate.invoke_claude_text")
    def test_blocked_raises(self, mock_gate):
        """Blocked invocation (gate returns -1) should raise RuntimeError."""
        mock_gate.return_value = ("", False, -1)
        with pytest.raises(RuntimeError, match="claude_gate returned exit_code=-1"):
            _call_claude_cli("haiku", "test", timeout=30)

    @mock.patch("portfolio.claude_gate.invoke_claude_text")
    def test_model_passed_to_gate(self, mock_gate):
        """Model alias should be forwarded to invoke_claude_text."""
        mock_gate.return_value = ("{}", True, 0)
        _call_claude_cli("sonnet", "test", timeout=60)
        mock_gate.assert_called_once_with(
            prompt="test",
            caller="claude_fundamental_sonnet",
            model="sonnet",
            timeout=60,
        )

    @mock.patch("portfolio.claude_gate.invoke_claude_text")
    def test_timeout_forwarded(self, mock_gate):
        """Timeout should be forwarded to invoke_claude_text."""
        mock_gate.return_value = ("{}", True, 0)
        _call_claude_cli("opus", "test", timeout=180)
        _, kwargs = mock_gate.call_args
        assert kwargs["timeout"] == 180
