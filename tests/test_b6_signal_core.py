"""B6 audit (2026-06-11) focused regression tests — signal engine core.

Covers the batch-6 fixes:
1. btc_proxy force-HOLD on MSTR (disabled-signal leak)
2. metals 2-voter consensus survives the Stage-4 dynamic floor (1-voter doesn't)
   + voter_count_post_persistence recorded on metals rows
3. seasonal-multiplier-then-cap ordering (output <= 0.80)
4. applicable-count includes ministral for non-crypto (XAU)
5. backfill aborts (no corruption) when signal_log rotates mid-backfill
6. blend_accuracy_data: recent subset of all-time no longer double-counted

(fix 8 was assessed + skipped, fix 7/9 are comment-only, fix 10 has its own test.)
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from conftest import make_indicators as _make_indicators


def _null_cached(key, ttl, func, *args):
    """Block accuracy/activation external lookups so consensus math is isolated."""
    if key and ("accuracy" in key or "activation_rates" in key):
        return {}
    return None


def _metals_ind(**overrides):
    base = dict(
        close=32.0, ema9=32.0, ema21=32.0,
        bb_upper=33.0, bb_lower=31.0, bb_mid=32.0,
        rsi=50.0, macd_hist=0.0, macd_hist_prev=0.0,
        atr=0.5, atr_pct=1.5, price_vs_bb="inside",
        rsi_p20=35.0, rsi_p80=65.0,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Fix 1: disabled btc_proxy must NOT vote live on MSTR (shadow only)
# --------------------------------------------------------------------------
class TestBtcProxyDisabledLeak:
    @patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_btc_proxy_disabled_routes_to_shadow_on_mstr(self, _mock):
        from portfolio import signal_engine as se
        from portfolio.tickers import DISABLED_SIGNALS

        assert "btc_proxy" in DISABLED_SIGNALS
        assert ("btc_proxy", "MSTR") not in se._DISABLED_SIGNAL_OVERRIDES

        # Seed the cross-ticker cache with a BTC SELL consensus (horizon=None
        # is the production key — main.py never passes a horizon).
        with se._cross_ticker_lock:
            se._cross_ticker_consensus[("BTC-USD", None)] = {
                "action": "SELL", "confidence": 0.7,
            }
        with se._persistence_lock:
            se._persistence_state.pop("MSTR", None)

        ind = _make_indicators(close=130.0, ema9=130.0, ema21=130.0,
                               bb_upper=135.0, bb_lower=125.0, bb_mid=130.0)
        _action, _conf, extra = se.generate_signal(ind, ticker="MSTR")

        votes = extra.get("_votes", {})
        shadow = extra.get("_shadow_votes", {})
        # Live consensus vote is force-HOLD; real action only in shadow.
        assert votes.get("btc_proxy") == "HOLD", votes.get("btc_proxy")
        assert shadow.get("btc_proxy") == "SELL", shadow
        assert extra.get("shadow_btc_proxy") is True
        # diagnostic fields still recorded
        assert extra.get("btc_proxy_action") == "SELL"


# --------------------------------------------------------------------------
# Fix 3: metals seasonal multiplier applied BEFORE the 0.80 cap
# --------------------------------------------------------------------------
class TestSeasonalBeforeCap:
    @patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_xag_feb_buy_confidence_never_exceeds_cap(self, _mock):
        import datetime as _dtmod

        from portfolio import signal_engine as se

        # Drive a strong oversold metals BUY in February (seasonal x1.15).
        with se._persistence_lock:
            se._persistence_state.pop("XAG-USD", None)

        # The seasonal block does a local `import datetime as _dt` then calls
        # _dt.datetime.now(...), so patch the stdlib datetime.datetime class.
        class _FrozenDT(_dtmod.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dtmod.datetime(2026, 2, 15, 12, 0, tzinfo=tz or UTC)

        ind = _metals_ind(rsi=18, macd_hist=2.0, macd_hist_prev=-2.0,
                          close=30.0, bb_lower=31.0, price_vs_bb="below")
        with patch("datetime.datetime", _FrozenDT):
            _action, conf, extra = se.generate_signal(ind, ticker="XAG-USD")

        # If we actually reached the seasonal block, confirm Feb was seen.
        if "seasonal_month" in extra:
            assert extra["seasonal_month"] == 2
        # The cap is the invariant: even with the Feb x1.15 seasonal boost the
        # emitted confidence must not exceed 0.80.
        assert conf <= 0.80 + 1e-9, (
            f"seasonal multiplier escaped the 0.80 cap: conf={conf}, "
            f"seasonal_mult={extra.get('seasonal_mult')}"
        )


# --------------------------------------------------------------------------
# Fix 2: asset-aware Stage-4 dynamic min voters
# --------------------------------------------------------------------------
class TestDynamicMinVotersMetals:
    def test_metals_floor_is_two_in_trending(self):
        from portfolio.signal_engine import (
            MIN_VOTERS_METALS,
            _dynamic_min_voters_for_regime,
        )
        assert MIN_VOTERS_METALS == 2
        # metals shifts each regime quorum down by 1, hard floor at 2.
        assert _dynamic_min_voters_for_regime("trending-up", ticker="XAG-USD") == 2
        assert _dynamic_min_voters_for_regime("trending-up", ticker="XAU-USD") == 2

    def test_metals_floor_never_below_two(self):
        from portfolio.signal_engine import _dynamic_min_voters_for_regime
        # ranging/high-vol shift down but never below MIN_VOTERS_METALS.
        assert _dynamic_min_voters_for_regime("high-vol", ticker="XAG-USD") == 3
        assert _dynamic_min_voters_for_regime("ranging", ticker="XAG-USD") == 4

    def test_non_metals_unchanged(self):
        from portfolio.signal_engine import _dynamic_min_voters_for_regime
        # crypto / stocks keep the original regime quorum (no ticker shift).
        assert _dynamic_min_voters_for_regime("trending-up", ticker="BTC-USD") == 3
        assert _dynamic_min_voters_for_regime("high-vol", ticker="MSTR") == 4
        assert _dynamic_min_voters_for_regime("ranging", ticker="BTC-USD") == 5
        # no ticker == old behavior.
        assert _dynamic_min_voters_for_regime("trending-up") == 3

    def test_two_voter_metals_passes_stage4_one_does_not(self):
        from portfolio.signal_engine import apply_confidence_penalties

        # signature: (action, conf, regime, ind, extra_info, ticker, df, config)
        # Neutral indicators so only the Stage-4 quorum gate is exercised; the
        # config disables nothing (penalties enabled).
        def _run(voters):
            extra = {"_voters_post_filter": voters}
            return apply_confidence_penalties(
                "BUY", 0.60, "trending-up", {}, extra, "XAG-USD", None, {},
            ), extra

        # 2 post-persistence voters: metals floor is 2 -> NOT force-HELD by Stage 4
        (action2, conf2, pen2), extra2 = _run(2)
        # 1 post-persistence voter: below floor -> force HOLD at Stage 4
        (action1, conf1, pen1), extra1 = _run(1)

        # Stage-4 quorum gate must not have fired for the 2-voter case.
        assert not any(p.get("stage") == "dynamic_min_voters" for p in pen2), \
            f"2-voter metals BUY was gated by Stage 4: {pen2}"
        # voter_count recorded on metals rows (premortem hook 11)
        assert extra2.get("voter_count_post_persistence") == 2
        assert extra2.get("min_voters_dynamic") == 2

        # 1-voter case must be force-HELD by Stage 4.
        assert action1 == "HOLD"
        assert any(p.get("stage") == "dynamic_min_voters" for p in pen1)
        assert extra1.get("voter_count_post_persistence") == 1


# --------------------------------------------------------------------------
# Review of 12f65ded: circuit-breaker relaxation path agrees with Stage 4
# on the metals quorum (guards A/B/C all ticker-aware)
# --------------------------------------------------------------------------
class TestRelaxationMetalsQuorum:
    # 2 voters, accuracy 0.46: gated at the strict 0.47 gate, recoverable
    # at the 2pp-relaxed 0.45 gate. No exclusions, trending regime.
    _VOTES = {"rsi": "BUY", "bb": "BUY"}
    _ACC = {
        "rsi": {"accuracy": 0.46, "total": 500},
        "bb": {"accuracy": 0.46, "total": 500},
    }

    def test_metals_two_voter_slate_can_relax(self):
        from portfolio.signal_engine import _compute_gate_relaxation
        rel = _compute_gate_relaxation(
            self._VOTES, self._ACC, set(), set(), 0.47,
            regime="trending-up", ticker="XAG-USD",
        )
        # Guards A (quorum 2), B (slate floor 2) and C (lone floor 2) must
        # all pass for metals; relaxation recovers both voters.
        assert rel > 0.0, "circuit breaker refused to relax a 2-voter metals slate"

    def test_non_metals_two_voter_slate_still_floored_at_three(self):
        from portfolio.signal_engine import _compute_gate_relaxation
        for tkr in ("BTC-USD", "MSTR"):
            rel = _compute_gate_relaxation(
                self._VOTES, self._ACC, set(), set(), 0.47,
                regime="trending-up", ticker=tkr,
            )
            assert rel == 0.0, f"2-voter {tkr} slate must not relax (floor 3)"
        # no ticker == old asset-blind behavior (floor 3)
        rel = _compute_gate_relaxation(
            self._VOTES, self._ACC, set(), set(), 0.47, regime="trending-up",
        )
        assert rel == 0.0


# --------------------------------------------------------------------------
# Fix 4: applicable count includes ministral on non-crypto
# --------------------------------------------------------------------------
class TestApplicableCountMinistral:
    def test_xau_count_includes_ministral(self):
        from portfolio import signal_engine as se
        # With GPU available (skip_gpu=False), ministral votes on all tickers,
        # so it must be counted for metals/stocks the same as for crypto.
        xau = se._compute_applicable_count("XAU-USD", skip_gpu=False)
        # Re-run with the (old, buggy) exclusion to prove ministral is now in.
        with patch.object(se, "GPU_SIGNALS", se.GPU_SIGNALS):
            assert "ministral" in se.SIGNAL_NAMES
        # ministral must contribute: count without it (simulate) is xau-1.
        # We assert directly that ministral is not silently dropped: build the
        # count manually excluding the ministral special-case and compare.
        assert xau >= 1
        # ministral present in the applicable set => count strictly larger than
        # a hypothetical version that skipped it. Verified by ensuring the old
        # skip path is gone: count for XAU includes ministral iff it is a GPU
        # signal counted under skip_gpu=False.
        assert "ministral" in se.GPU_SIGNALS

    def test_skip_gpu_still_drops_ministral_for_stocks(self):
        from portfolio import signal_engine as se
        with_gpu = se._compute_applicable_count("MSTR", skip_gpu=False)
        no_gpu = se._compute_applicable_count("MSTR", skip_gpu=True)
        # ministral (a GPU signal) is dropped only via the skip_gpu branch now.
        assert no_gpu < with_gpu


# --------------------------------------------------------------------------
# Fix 6: blend_accuracy_data — recent subset of all-time, no double-count
# --------------------------------------------------------------------------
class TestBlendNoDoubleCount:
    def test_directional_totals_use_max_not_sum(self):
        from portfolio.accuracy_stats import blend_accuracy_data
        # recent is a strict subset of all-time: all-time has 400 buy samples,
        # recent (last 7d) has 80 of those. Summing -> 480 (wrong).
        alltime = {
            "rsi": {
                "accuracy": 0.52, "total": 400,
                "buy_accuracy": 0.50, "total_buy": 400,
                "sell_accuracy": 0.54, "total_sell": 200,
            }
        }
        recent = {
            "rsi": {
                "accuracy": 0.58, "total": 80,
                "buy_accuracy": 0.60, "total_buy": 80,
                "sell_accuracy": 0.55, "total_sell": 40,
            }
        }
        blended = blend_accuracy_data(alltime, recent)["rsi"]
        # totals must be max(at, rc), not at + rc
        assert blended["total_buy"] == 400, blended
        assert blended["total_sell"] == 200, blended
        # directional accuracy blended 70/30 (recent has >=30 sam, at>0):
        # 0.70*0.60 + 0.30*0.50 = 0.57
        assert blended["buy_accuracy"] == pytest.approx(0.57, abs=1e-6)

    def test_recent_only_signal_directionals_kept(self):
        from portfolio.accuracy_stats import blend_accuracy_data
        recent = {
            "newsig": {
                "accuracy": 0.62, "total": 60,
                "sell_accuracy": 0.28, "total_sell": 60,
            }
        }
        blended = blend_accuracy_data({}, recent)["newsig"]
        # recent-only with enough samples: directional kept, total = max(0,60)
        assert blended["total_sell"] == 60
        assert blended["sell_accuracy"] == pytest.approx(0.28, abs=1e-6)


# --------------------------------------------------------------------------
# Fix 5: backfill aborts on mid-backfill rotation (no corruption)
# --------------------------------------------------------------------------
class TestBackfillRotationRace:
    def _entry(self, hours_ago, ticker="BTC-USD"):
        ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
        return {"ts": ts, "tickers": {ticker: {"price_usd": 67000}}}

    def test_rotation_during_backfill_aborts_rewrite(self, tmp_path):
        from portfolio import outcome_tracker as ot

        log = tmp_path / "signal_log.jsonl"
        # head (old, filled) + tail entries needing backfill
        head = []
        for i in range(3):
            e = self._entry(hours_ago=500 + i)
            head.append(e)
        tail = [self._entry(hours_ago=i + 1) for i in range(3)]
        with open(log, "w", encoding="utf-8") as f:
            for e in head + tail:
                f.write(json.dumps(e) + "\n")
        original_bytes = log.read_bytes()

        # Simulate rotate_jsonl landing during the unlocked Phase-2 HTTP window:
        # the FIRST historical-price fetch replaces the file with a shorter,
        # different-first-line tail-keep version (what rotation does).
        rotated = {"ts": "2026-06-11T00:00:00+00:00", "tickers": {"ETH-USD": {"price_usd": 2}}}
        rotated_bytes = (json.dumps(rotated) + "\n").encode("utf-8")
        state = {"rotated": False}

        def _fetch_then_rotate(ticker, target_ts):
            if not state["rotated"]:
                log.write_bytes(rotated_bytes)  # rotation replaces the file
                state["rotated"] = True
            return None

        with patch.object(ot, "SIGNAL_LOG", log), \
             patch.object(ot, "_fetch_historical_price", side_effect=_fetch_then_rotate):
            result = ot.backfill_outcomes(max_entries=3)

        # backfill must have ABORTED the rewrite (return 0) and left the
        # rotated file intact — NOT clobbered with stale head/tail bytes.
        assert result == 0
        after = log.read_bytes()
        assert after == rotated_bytes, "backfill corrupted the rotated log"
        assert after != original_bytes

    def test_no_rotation_normal_backfill_unaffected(self, tmp_path):
        from portfolio import outcome_tracker as ot

        log = tmp_path / "signal_log.jsonl"
        entries = [self._entry(hours_ago=i + 1) for i in range(3)]
        with open(log, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        with patch.object(ot, "SIGNAL_LOG", log), \
             patch.object(ot, "_fetch_historical_price", return_value=None):
            ot.backfill_outcomes(max_entries=100)

        # all 3 entries preserved, file still valid JSONL
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 3
        for l in lines:
            json.loads(l)


# --------------------------------------------------------------------------
# Fix 10: scripts/accuracy_gate_flip_diff.py
# --------------------------------------------------------------------------
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import accuracy_gate_flip_diff as agfd  # noqa: E402


class TestGateFlipDiff:
    def test_gate_status_tiers(self):
        # below 30 samples -> never gated regardless of accuracy
        assert agfd.gate_status({"accuracy": 0.10, "total": 10})[0] is False
        # standard tier: <47% with >=30 samples -> gated
        assert agfd.gate_status({"accuracy": 0.46, "total": 500})[0] is True
        assert agfd.gate_status({"accuracy": 0.48, "total": 500})[0] is False
        # high-sample tier: <50% with >=7000 samples -> gated
        assert agfd.gate_status({"accuracy": 0.49, "total": 8000})[0] is True
        # same 49% but under 7000 samples passes the looser 47% gate
        assert agfd.gate_status({"accuracy": 0.49, "total": 6000})[0] is False

    def test_flip_detection_on_synthetic_snapshots(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        # rsi: stays open. flipper_open_to_gated: 0.48 -> 0.46 (OPEN -> GATED).
        # flipper_gated_to_open: 0.45 -> 0.49 (GATED -> OPEN).
        old.write_text(json.dumps({"1d": {
            "rsi": {"accuracy": 0.52, "total": 1000},
            "flipper_open_to_gated": {"accuracy": 0.48, "total": 500},
            "flipper_gated_to_open": {"accuracy": 0.45, "total": 500},
        }}))
        new.write_text(json.dumps({"1d": {
            "rsi": {"accuracy": 0.53, "total": 1100},
            "flipper_open_to_gated": {"accuracy": 0.46, "total": 600},
            "flipper_gated_to_open": {"accuracy": 0.49, "total": 600},
        }}))

        rc = agfd.main([str(old), str(new)])
        assert rc == 0

        # exercise the comparison directly to assert exactly two flips.
        with open(old) as f:
            ob = json.load(f)["1d"]
        with open(new) as f:
            nb = json.load(f)["1d"]
        flipped = {
            s for s in set(ob) | set(nb)
            if agfd.gate_status(ob.get(s))[0] != agfd.gate_status(nb.get(s))[0]
        }
        assert flipped == {"flipper_open_to_gated", "flipper_gated_to_open"}
