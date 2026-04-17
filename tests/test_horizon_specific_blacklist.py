"""Tests for horizon-specific per-ticker blacklist (Batch 4 of 2026-04-16
accuracy gating reconfiguration).

The W15/W16 MSTR consensus collapse was caused by a blacklist built from
3h accuracy data applied at all horizons; at 1d the blacklisted signals
were 66-81% accurate. Batch 4 adds the per-horizon mechanism (entries empty
by default — populated by future audits).
"""

import pytest

from portfolio.signal_engine import (
    _TICKER_DISABLED_BY_HORIZON,
    _TICKER_DISABLED_SIGNALS,
    _get_horizon_disabled_signals,
    _weighted_consensus,
)


class TestHorizonDisabledStructure:
    """Guardrail: the dict has the expected shape."""

    def test_has_default_key(self):
        assert "_default" in _TICKER_DISABLED_BY_HORIZON

    def test_has_all_supported_horizons(self):
        for h in ("3h", "4h", "12h", "1d", "3d", "5d", "10d"):
            assert h in _TICKER_DISABLED_BY_HORIZON, f"missing horizon {h}"

    def test_legacy_alias_points_to_default(self):
        """_TICKER_DISABLED_SIGNALS must remain as a view of _default for
        backward compatibility with the signal dispatch loop."""
        assert _TICKER_DISABLED_SIGNALS is _TICKER_DISABLED_BY_HORIZON["_default"]

    def test_default_includes_mstr_trimmed_to_two(self):
        """Batch 1 trim: MSTR default has claude_fundamental + credit_spread_risk."""
        mstr = _TICKER_DISABLED_BY_HORIZON["_default"].get("MSTR", frozenset())
        assert mstr == frozenset({"claude_fundamental", "credit_spread_risk"})


class TestGetHorizonDisabledSignals:
    """Helper function: returns union of _default + horizon-specific."""

    def test_returns_default_set_when_no_horizon(self):
        result = _get_horizon_disabled_signals("MSTR", horizon=None)
        assert result == frozenset({"claude_fundamental", "credit_spread_risk"})

    def test_returns_default_union_horizon_for_mstr_1d(self):
        # MSTR has 1d-specific entries (ema, bb) added 2026-04-16 after-hours.
        result = _get_horizon_disabled_signals("MSTR", horizon="1d")
        assert result == frozenset({
            "claude_fundamental", "credit_spread_risk",  # _default
            "ema", "bb",  # 1d-specific
        })

    def test_unions_horizon_specific_with_default(self):
        """3h-specific entries for MSTR (volume, volatility_sig) union with _default."""
        result = _get_horizon_disabled_signals("MSTR", horizon="3h")
        assert result == frozenset({
            "claude_fundamental", "credit_spread_risk",  # _default
            "volume", "volatility_sig",  # 3h-specific
        })

    def test_monkeypatch_horizon_entry_for_unknown_ticker(self, monkeypatch):
        """Adding a synthetic horizon entry via monkeypatch still unions correctly."""
        monkeypatch.setitem(
            _TICKER_DISABLED_BY_HORIZON["4h"], "FAKE-TICKER", frozenset({"rsi"}),
        )
        result = _get_horizon_disabled_signals("FAKE-TICKER", horizon="4h")
        assert result == frozenset({"rsi"})
        result_1d = _get_horizon_disabled_signals("FAKE-TICKER", horizon="1d")
        assert result_1d == frozenset()

    def test_empty_frozenset_for_unknown_ticker(self):
        result = _get_horizon_disabled_signals("UNKNOWN-TICKER", horizon="1d")
        assert result == frozenset()

    def test_empty_frozenset_for_none_ticker(self):
        result = _get_horizon_disabled_signals(None, horizon="1d")
        assert result == frozenset()


class TestWeightedConsensusAppliesHorizonBlacklist:
    """_weighted_consensus must force-HOLD horizon-blacklisted signals."""

    def _make_stats(self, acc, total=100):
        return {
            "accuracy": acc, "total": total,
            "buy_accuracy": acc, "sell_accuracy": acc,
            "total_buy": total // 2, "total_sell": total // 2,
        }

    def test_consensus_ignores_horizon_blacklisted_signal(self, monkeypatch):
        """If a ticker+horizon combo blacklists a signal, consensus drops its vote.

        2026-04-16 review quality fix: use monkeypatch.setitem so the mutation
        is automatically reverted even if the test raises between set and
        cleanup. Previously used try/finally which had a small xdist race
        window.
        """
        monkeypatch.setitem(
            _TICKER_DISABLED_BY_HORIZON["3h"], "TEST-TICKER", frozenset({"rsi"}),
        )

        votes = {"rsi": "BUY", "ema": "SELL", "macd": "SELL",
                 "bb": "SELL", "volume": "SELL", "mean_reversion": "SELL"}
        accuracy = {k: self._make_stats(0.60) for k in votes}

        # At 3h: rsi is blacklisted, so BUY vote dropped. Consensus SELL.
        action_3h, _ = _weighted_consensus(
            votes, accuracy, regime="unknown", horizon="3h", ticker="TEST-TICKER",
        )
        assert action_3h == "SELL"

        # At 1d: rsi is NOT blacklisted, vote counts. Still SELL (5 vs 1)
        # but the rsi BUY reduces SELL confidence.
        action_1d, _ = _weighted_consensus(
            votes, accuracy, regime="unknown", horizon="1d", ticker="TEST-TICKER",
        )
        assert action_1d == "SELL"

    def test_consensus_applies_default_blacklist_without_horizon_entry(self):
        """Default (MSTR claude_fundamental) applies at every horizon."""
        votes = {
            "claude_fundamental": "BUY",
            "rsi": "SELL", "ema": "SELL", "macd": "SELL",
            "bb": "SELL", "volume": "SELL",
        }
        accuracy = {k: self._make_stats(0.60) for k in votes}

        for horizon in ("3h", "1d", "3d"):
            action, _ = _weighted_consensus(
                votes, accuracy, regime="unknown", horizon=horizon, ticker="MSTR",
            )
            # claude_fundamental should be HOLD'd regardless of horizon (in _default)
            assert action == "SELL", f"Expected SELL at {horizon}, got {action}"

    def test_alias_mutation_synchronizes_with_default(self, monkeypatch):
        """2026-04-16 review (Reviewer 1 I3 + Reviewer 3 P1-4): the legacy
        alias _TICKER_DISABLED_SIGNALS is a reference binding to
        _TICKER_DISABLED_BY_HORIZON['_default']. If a future refactor
        accidentally rebinds the alias to a copy, the two names diverge
        silently, re-breaking the horizon-mismatch fix (compute time sees
        one blacklist, consensus sees another).

        This test pins the mutation behavior: adding an entry through either
        name must be visible through both.
        """
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        # Mutate through the alias; should leak into _TICKER_DISABLED_BY_HORIZON['_default'].
        monkeypatch.setitem(
            _TICKER_DISABLED_SIGNALS, "TEST-TICKER-ALIAS", frozenset({"rsi"}),
        )
        assert _TICKER_DISABLED_BY_HORIZON["_default"].get("TEST-TICKER-ALIAS") \
            == frozenset({"rsi"}), (
                "Alias mutation didn't propagate to _default - rebinding bug."
            )

    def test_consensus_unchanged_without_ticker(self):
        """No ticker -> no horizon-specific gating applied (matches pre-Batch-4 behavior)."""
        votes = {
            "claude_fundamental": "BUY",
            "rsi": "BUY", "ema": "BUY", "macd": "BUY",
        }
        accuracy = {k: self._make_stats(0.60) for k in votes}
        action, _ = _weighted_consensus(
            votes, accuracy, regime="unknown", horizon="1d", ticker=None,
        )
        # claude_fundamental NOT gated (no ticker -> no blacklist lookup) -> BUY
        assert action == "BUY"


class TestAfterHoursAuditEntries:
    """2026-04-16 after-hours audit populated 3h/1d horizon-specific entries.
    Pin the exact entries to catch accidental mutation or drift."""

    def test_3h_btc(self):
        result = _get_horizon_disabled_signals("BTC-USD", "3h")
        expected_default = {"smart_money", "heikin_ashi"}
        expected_3h = {"volatility_sig", "bb"}
        assert result == frozenset(expected_default | expected_3h)

    def test_3h_eth(self):
        result = _get_horizon_disabled_signals("ETH-USD", "3h")
        expected_default = {"news_event", "qwen3", "smart_money"}
        expected_3h = {"credit_spread_risk"}
        assert result == frozenset(expected_default | expected_3h)

    def test_3h_xau(self):
        result = _get_horizon_disabled_signals("XAU-USD", "3h")
        expected_default = {"ministral", "metals_cross_asset"}
        expected_3h = {"credit_spread_risk"}
        assert result == frozenset(expected_default | expected_3h)

    def test_3h_xag(self):
        result = _get_horizon_disabled_signals("XAG-USD", "3h")
        expected_default = {"ministral", "credit_spread_risk", "metals_cross_asset",
                            "smart_money"}
        expected_3h = {"forecast", "qwen3"}
        assert result == frozenset(expected_default | expected_3h)

    def test_3h_mstr(self):
        result = _get_horizon_disabled_signals("MSTR", "3h")
        expected_default = {"claude_fundamental", "credit_spread_risk"}
        expected_3h = {"volume", "volatility_sig"}
        assert result == frozenset(expected_default | expected_3h)

    def test_1d_btc(self):
        result = _get_horizon_disabled_signals("BTC-USD", "1d")
        expected_default = {"smart_money", "heikin_ashi"}
        expected_1d = {"news_event", "forecast"}
        assert result == frozenset(expected_default | expected_1d)

    def test_1d_xau(self):
        result = _get_horizon_disabled_signals("XAU-USD", "1d")
        expected_default = {"ministral", "metals_cross_asset"}
        expected_1d = {"candlestick"}
        assert result == frozenset(expected_default | expected_1d)

    def test_1d_mstr(self):
        result = _get_horizon_disabled_signals("MSTR", "1d")
        expected_default = {"claude_fundamental", "credit_spread_risk"}
        expected_1d = {"ema", "bb"}
        assert result == frozenset(expected_default | expected_1d)

    def test_4h_has_no_entries_yet(self):
        """4h horizon still empty — no data-driven entries yet."""
        for ticker in ("BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"):
            result = _get_horizon_disabled_signals(ticker, "4h")
            default = _TICKER_DISABLED_BY_HORIZON["_default"].get(ticker, frozenset())
            assert result == default, f"4h should only have _default for {ticker}"


class TestGenerateSignalHorizonBlacklistE2E:
    """2026-04-16 review (Reviewer 3 P1-3): end-to-end coverage that ticker
    propagates from generate_signal through to the horizon blacklist. The
    unit tests only call _weighted_consensus directly; this test covers
    the full dispatch path where an argument could be accidentally dropped
    in a refactor.
    """

    def _make_df(self, n=100, close_start=100.0):
        import numpy as np
        import pandas as pd
        dates = pd.date_range("2026-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)
        closes = close_start + np.cumsum(rng.standard_normal(n) * 0.5)
        highs = closes + rng.random(n) * 2
        lows = closes - rng.random(n) * 2
        volumes = rng.integers(100, 10000, n).astype(float)
        return pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )

    def test_mstr_claude_fundamental_is_hold_at_every_horizon(self, monkeypatch):
        """P1-F (2026-04-17 adversarial-review): the previous assertion only
        checked `isinstance(extra, dict)` and re-called the helper. Now
        actually spy on _weighted_consensus to capture the votes AFTER all
        gating, then assert claude_fundamental appears as HOLD there.

        This catches a refactor that drops the horizon-disable lookup inside
        _weighted_consensus but leaves the helper intact - a pure-helper test
        would pass spuriously.
        """
        from unittest import mock

        import portfolio.signal_engine as se
        from portfolio.indicators import compute_indicators

        df = self._make_df(150)

        monkeypatch.setattr("portfolio.market_timing.should_skip_gpu",
                            mock.MagicMock(return_value=True))
        monkeypatch.setattr("portfolio.shared_state._cached",
                            mock.MagicMock(return_value=None))

        # Spy on _weighted_consensus to capture the post-gated votes dict.
        captured_votes: dict = {}
        real_fn = se._weighted_consensus

        def _spy(votes, *args, **kwargs):
            # The votes dict inside _weighted_consensus is mutated by
            # horizon-disable + regime-gate BEFORE consensus. We capture
            # a snapshot at call time.
            captured_votes["snapshot"] = dict(votes)
            return real_fn(votes, *args, **kwargs)

        monkeypatch.setattr("portfolio.signal_engine._weighted_consensus", _spy)

        for horizon in ("3h", "1d"):
            captured_votes.clear()
            ind = compute_indicators(df, horizon=horizon if horizon == "3h" else None)
            se.generate_signal(
                ind, ticker="MSTR", df=df,
                horizon=horizon if horizon == "3h" else None,
            )
            # The OUTER caller already applies horizon-disable (P1-B fix),
            # so by the time _weighted_consensus receives votes, blacklisted
            # MSTR signals should be HOLD.
            snapshot = captured_votes.get("snapshot", {})
            assert snapshot, f"spy didn't capture votes at horizon={horizon}"
            # claude_fundamental is in _default blacklist for MSTR.
            cf = snapshot.get("claude_fundamental")
            assert cf == "HOLD", (
                f"At horizon={horizon}, claude_fundamental should be HOLD "
                f"post-gating (was {cf!r}). Horizon-disable failed to "
                f"propagate through the pipeline."
            )
            # credit_spread_risk also in _default for MSTR.
            csr = snapshot.get("credit_spread_risk")
            assert csr == "HOLD", (
                f"At horizon={horizon}, credit_spread_risk should be HOLD "
                f"post-gating (was {csr!r})."
            )

    def test_generate_signal_passes_ticker_into_weighted_consensus(self, monkeypatch):
        """Directly verify the call-site plumbing by spying on _weighted_consensus."""
        from unittest import mock

        import portfolio.signal_engine as se
        from portfolio.indicators import compute_indicators

        captured_kwargs = {}
        real_fn = se._weighted_consensus

        def _spy(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr("portfolio.signal_engine._weighted_consensus", _spy)
        monkeypatch.setattr("portfolio.market_timing.should_skip_gpu",
                            mock.MagicMock(return_value=True))
        monkeypatch.setattr("portfolio.shared_state._cached",
                            mock.MagicMock(return_value=None))

        df = self._make_df(150)
        ind = compute_indicators(df)
        se.generate_signal(ind, ticker="MSTR", df=df)

        assert captured_kwargs.get("ticker") == "MSTR", (
            "generate_signal must propagate ticker='MSTR' into _weighted_consensus "
            "or the horizon-specific blacklist silently stops firing."
        )
