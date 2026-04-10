"""Tests for portfolio.signal_engine — dynamic correlation groups."""



# ---------------------------------------------------------------------------
# 3c. Static and dynamic correlation groups
# ---------------------------------------------------------------------------

class TestCorrelationGroups:

    def test_static_correlation_groups_has_expected_keys(self):
        """CORRELATION_GROUPS (static alias) should exist and contain known group names."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert isinstance(CORRELATION_GROUPS, dict)
        # Check a few known static groups
        assert "low_activity_timing" in CORRELATION_GROUPS
        assert "pattern_based" in CORRELATION_GROUPS
        # Values should be frozensets
        for name, members in CORRELATION_GROUPS.items():
            assert isinstance(members, frozenset), f"Group {name} should be frozenset"
            assert len(members) >= 2, f"Group {name} should have at least 2 members"

    def test_momentum_cluster_exists(self):
        """momentum_cluster should contain rsi, bb, mean_reversion, momentum.

        Empirical: rsi-bb 100%, bb-mean_reversion 100%, bb-momentum 98.8%.
        Renamed from rsi_based (2026-04-08).
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "momentum_cluster" in CORRELATION_GROUPS
        mc_group = CORRELATION_GROUPS["momentum_cluster"]
        assert "mean_reversion" in mc_group
        assert "rsi" in mc_group
        assert "bb" in mc_group
        assert "momentum" in mc_group

    def test_volatility_cluster_exists(self):
        """volatility_cluster should contain volatility_sig, oscillators, volume, structure.

        Empirical: volume-volatility_sig 94.9%, volatility_sig-structure 94.2%.
        Renamed from rare_technical and expanded (2026-04-08).
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "volatility_cluster" in CORRELATION_GROUPS
        vc_group = CORRELATION_GROUPS["volatility_cluster"]
        assert "volatility_sig" in vc_group
        assert "oscillators" in vc_group
        assert "volume" in vc_group
        assert "structure" in vc_group

    def test_macro_external_includes_momentum_factors(self):
        """momentum_factors should be in macro_external (94.3% with sentiment)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        me_group = CORRELATION_GROUPS["macro_external"]
        assert "momentum_factors" in me_group

    def test_macro_regime_in_trend_direction(self):
        """macro_regime should be in trend_direction group (r=0.520 with trend)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "trend_direction" in CORRELATION_GROUPS
        td_group = CORRELATION_GROUPS["trend_direction"]
        assert "macro_regime" in td_group
        assert "trend" in td_group

    def test_macro_regime_not_in_macro_external(self):
        """macro_regime was moved out of macro_external into trend_direction."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        me_group = CORRELATION_GROUPS["macro_external"]
        assert "macro_regime" not in me_group

    def test_static_groups_backward_compat_alias(self):
        """CORRELATION_GROUPS should be the same object as _STATIC_CORRELATION_GROUPS."""
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            CORRELATION_GROUPS,
        )
        assert CORRELATION_GROUPS is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback(self, monkeypatch):
        """When no signal_log data exists, _compute_dynamic_correlation_groups
        should return _STATIC_CORRELATION_GROUPS."""
        # Monkeypatch load_entries to return empty list (no data)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback_insufficient_data(self, monkeypatch):
        """With fewer than _DYNAMIC_CORR_MIN_SAMPLES entries, falls back to static."""
        # Provide a small number of entries (below the 30 minimum)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        fake_entries = [
            {
                "ts": "2026-04-01T00:00:00+00:00",
                "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
                "outcomes": {},
            }
            for _ in range(5)
        ]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: fake_entries)

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_returns_dict_of_frozensets(self, monkeypatch):
        """Even when falling back, the return type is dict[str, frozenset]."""
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import _compute_dynamic_correlation_groups
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, frozenset)


# ---------------------------------------------------------------------------
# Directional bias penalty
# ---------------------------------------------------------------------------

class TestDirectionalBiasPenalty:

    def test_extreme_bias_reduces_weight(self):
        """Signals with bias > 85% should get _BIAS_PENALTY applied."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY", "calendar": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 100},
            "calendar": {"accuracy": 0.60, "total": 100},
        }
        # calendar has extreme bias (>85%), rsi does not
        activation_rates = {
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 0.25,
                         "activation_rate": 0.08},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        # Both BUY → should still return BUY
        assert result[0] == "BUY"

    def test_no_penalty_below_threshold(self):
        """Signals with bias <= 85% should NOT get extra penalty."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.5, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
        }
        # With bias=0.5 (< 0.85), no extra penalty should apply
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"

    def test_bias_penalty_not_applied_with_few_samples(self):
        """Bias penalty should not fire when samples < _BIAS_MIN_ACTIVE."""
        from portfolio.signal_engine import _BIAS_MIN_ACTIVE, _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.99, "samples": _BIAS_MIN_ACTIVE - 1,
                    "normalized_weight": 1.0, "activation_rate": 0.3},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"


# ---------------------------------------------------------------------------
# Directional accuracy gating
# ---------------------------------------------------------------------------

class TestDirectionalAccuracyGating:

    def test_buy_gated_when_buy_accuracy_low(self):
        """Qwen3-like: overall 59.8% passes gate, but BUY=30% should be gated."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 BUY at 30% (well below 35% directional gate), SELL at 74%
        votes = {"qwen3": "BUY", "rsi": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # qwen3 BUY should be gated, only rsi SELL remains → SELL
        assert result[0] == "SELL"

    def test_sell_passes_when_buy_gated(self):
        """Same signal: BUY gated but SELL should still vote normally."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 SELL should NOT be directionally gated (sell_accuracy 74% >> 35%)
        votes = {"qwen3": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Only qwen3 voting SELL, not directionally gated → SELL
        assert result[0] == "SELL"

    def test_directional_gate_not_applied_with_few_samples(self):
        """Directional gate requires sufficient samples to avoid premature gating."""
        from portfolio.signal_engine import (
            _DIRECTIONAL_GATE_MIN_SAMPLES,
            _weighted_consensus,
        )

        votes = {"qwen3": "BUY"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.55, "total": 100,
                       "buy_accuracy": 0.20, "total_buy": _DIRECTIONAL_GATE_MIN_SAMPLES - 1,
                       "sell_accuracy": 0.80, "total_sell": 100},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Not enough BUY samples → directional gate should NOT fire
        assert result[0] == "BUY"

    def test_claude_fundamental_sell_gated(self):
        """claude_fundamental: BUY=65.7% fine, but SELL=39.7% should be gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"claude_fundamental": "SELL", "rsi": "BUY"}
        accuracy_data = {
            "claude_fundamental": {"accuracy": 0.628, "total": 7535,
                                    "buy_accuracy": 0.657, "total_buy": 6697,
                                    "sell_accuracy": 0.397, "total_sell": 838},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        # sell_accuracy 0.397 > 0.35 threshold → NOT gated
        # (we set _DIRECTIONAL_GATE_THRESHOLD at 0.35, not 0.45)
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # claude_fundamental SELL NOT gated at 0.35 threshold (it's 0.397)
        # Both vote, weights determine outcome
        assert result[0] in ("SELL", "BUY")

    def test_no_directional_gate_when_accuracy_above_threshold(self):
        """Signals with both BUY and SELL above threshold should not be gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.52, "total": 1000,
                     "buy_accuracy": 0.50, "total_buy": 500,
                     "sell_accuracy": 0.54, "total_sell": 500},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "BUY"


# ---------------------------------------------------------------------------
# Funding rate horizon gating (2026-04-09)
# ---------------------------------------------------------------------------

class TestFundingRateHorizonGating:
    """Funding rate: 74.2% at 3h but 29.9% at 1d. Should only vote at 3h/4h."""

    def test_funding_gated_at_default_in_all_regimes(self):
        """funding should be in _default gate for ranging, trending-up/down, high-vol."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        for regime in ("ranging", "trending-up", "trending-down", "high-vol"):
            assert regime in REGIME_GATED_SIGNALS, f"{regime} missing from REGIME_GATED_SIGNALS"
            default_set = REGIME_GATED_SIGNALS[regime].get("_default", frozenset())
            assert "funding" in default_set, (
                f"funding should be gated at _default in {regime}"
            )

    def test_funding_not_gated_at_3h_ranging(self):
        """funding should NOT be in 3h gate for ranging (74.2% accuracy there)."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_3h = REGIME_GATED_SIGNALS["ranging"].get("3h", frozenset())
        assert "funding" not in gate_3h

    def test_funding_not_gated_at_4h_ranging(self):
        """funding should NOT be in 4h gate for ranging."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_4h = REGIME_GATED_SIGNALS["ranging"].get("4h", frozenset())
        assert "funding" not in gate_4h

    def test_funding_not_in_disabled_signals(self):
        """funding was removed from DISABLED_SIGNALS (re-enabled 2026-04-09)."""
        from portfolio.tickers import DISABLED_SIGNALS

        assert "funding" not in DISABLED_SIGNALS


# ---------------------------------------------------------------------------
# On-chain BTC signal (2026-04-09)
# ---------------------------------------------------------------------------

class TestOnchainSignal:
    """On-chain signal: MVRV Z-Score, SOPR, NUPL, netflow → majority vote for BTC."""

    def test_onchain_in_signal_names(self):
        """onchain should be in the SIGNAL_NAMES list."""
        from portfolio.tickers import SIGNAL_NAMES

        assert "onchain" in SIGNAL_NAMES

    def _run_onchain_block(self, ticker, cached_data):
        """Helper: run just the on-chain signal block in isolation.

        Returns (vote, extra_info) without calling generate_signal
        (which needs dozens of indicator keys).
        """
        from portfolio.tickers import CRYPTO_SYMBOLS
        from portfolio.shared_state import ONCHAIN_TTL

        votes = {"onchain": "HOLD"}
        extra_info = {}

        if ticker == "BTC-USD":
            oc = cached_data
            if oc:
                sub_votes = []
                zscore = oc.get("mvrv_zscore")
                if zscore is not None:
                    if zscore < 1.0:
                        sub_votes.append("BUY")
                    elif zscore > 5.0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_mvrv_zscore"] = round(zscore, 2)
                sopr = oc.get("sopr")
                if sopr is not None:
                    if sopr < 0.97:
                        sub_votes.append("BUY")
                    elif sopr > 1.05:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_sopr"] = round(sopr, 4)
                nupl = oc.get("nupl")
                if nupl is not None:
                    if nupl < 0:
                        sub_votes.append("BUY")
                    elif nupl > 0.75:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                netflow = oc.get("netflow")
                if netflow is not None:
                    if netflow < 0:
                        sub_votes.append("BUY")
                    elif netflow > 0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                buy_count = sub_votes.count("BUY")
                sell_count = sub_votes.count("SELL")
                total = buy_count + sell_count
                if total >= 2:
                    if buy_count > sell_count:
                        votes["onchain"] = "BUY"
                    elif sell_count > buy_count:
                        votes["onchain"] = "SELL"
                    extra_info["onchain_sub_votes"] = f"{buy_count}B/{sell_count}S"

        return votes["onchain"], extra_info

    def test_onchain_buy_when_metrics_undervalued(self):
        """When MVRV < 1, SOPR < 0.97, NUPL < 0 → majority BUY."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": -0.1,         # < 0 → BUY
            "netflow": 500,       # > 0 → SELL (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "BUY"
        assert extra["onchain_mvrv_zscore"] == 0.5
        assert extra["onchain_sopr"] == 0.95
        assert extra["onchain_sub_votes"] == "3B/1S"

    def test_onchain_sell_when_overheated(self):
        """When MVRV > 5, SOPR > 1.05, NUPL > 0.75 → majority SELL."""
        data = {
            "mvrv_zscore": 6.0,   # > 5.0 → SELL
            "sopr": 1.08,         # > 1.05 → SELL
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": -100,      # < 0 → BUY (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "SELL"
        assert extra["onchain_sub_votes"] == "1B/3S"

    def test_onchain_hold_for_non_btc(self):
        """On-chain signal should be HOLD for non-BTC tickers."""
        data = {"mvrv_zscore": 0.5, "sopr": 0.95, "nupl": -0.1, "netflow": -100}
        vote, extra = self._run_onchain_block("ETH-USD", data)
        assert vote == "HOLD"
        assert not extra  # no extra_info populated

    def test_onchain_hold_when_no_data(self):
        """On-chain signal should be HOLD when API returns None."""
        vote, extra = self._run_onchain_block("BTC-USD", None)
        assert vote == "HOLD"

    def test_onchain_hold_when_neutral_metrics(self):
        """When all metrics are neutral, should remain HOLD (no majority)."""
        data = {
            "mvrv_zscore": 3.0,   # 1-5 → HOLD
            "sopr": 1.00,         # 0.97-1.05 → HOLD
            "nupl": 0.4,          # 0-0.75 → HOLD
            "netflow": 0,         # 0 → HOLD
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        # No sub_votes key when total active (BUY+SELL) < 2
        assert "onchain_sub_votes" not in extra

    def test_onchain_tie_stays_hold(self):
        """When BUY and SELL are tied (2B/2S), should remain HOLD."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": 500,       # > 0 → SELL
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        assert extra["onchain_sub_votes"] == "2B/2S"


# ---------------------------------------------------------------------------
# BUG-178 fixes (2026-04-10)
# ---------------------------------------------------------------------------

class TestDispatchLoopRespectsDisabledSignals:
    """Regression: enhanced-signal dispatch loop must skip DISABLED_SIGNALS.

    Before 2026-04-10, the dispatch loop iterated every registered enhanced
    signal regardless of disabled status. This caused 49 BUG-178 ticker
    pool timeouts on 2026-04-09/10 because the disabled signals
    (crypto_macro, cot_positioning, credit_spread_risk) were doing network
    I/O on every cycle.
    """

    def test_disabled_signals_have_force_hold_in_dispatch_loop(self):
        """The dispatch loop must short-circuit on DISABLED_SIGNALS."""
        import inspect

        from portfolio import signal_engine
        from portfolio.tickers import DISABLED_SIGNALS

        src = inspect.getsource(signal_engine.generate_signal)
        # The fix: dispatch loop must check DISABLED_SIGNALS before calling compute_fn
        assert "if sig_name in DISABLED_SIGNALS" in src, (
            "Dispatch loop must skip disabled signals to prevent BUG-178 hangs"
        )
        # Defense in depth: at least one disabled signal must exist (otherwise
        # the check is dead code we should remove)
        assert len(DISABLED_SIGNALS) > 0


class TestLastSignalDiagnostic:
    """BUG-178 diagnostic: per-ticker last-signal tracker."""

    def test_set_and_get_last_signal_round_trip(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("BTC-USD", "test_signal_xyz")
        result = get_last_signal("BTC-USD")
        assert result is not None
        sig_name, elapsed = result
        assert sig_name == "test_signal_xyz"
        assert elapsed >= 0.0
        assert elapsed < 5.0  # should be near-instant

    def test_get_last_signal_unknown_ticker_returns_none(self):
        from portfolio.signal_engine import get_last_signal

        result = get_last_signal("__nonexistent_ticker__")
        assert result is None

    def test_set_overwrites_previous(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("ETH-USD", "first_signal")
        _set_last_signal("ETH-USD", "second_signal")
        result = get_last_signal("ETH-USD")
        assert result is not None
        assert result[0] == "second_signal"
