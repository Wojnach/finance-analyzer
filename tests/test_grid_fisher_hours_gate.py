"""Tests for grid_fisher Gate A (quote-staleness) and Gate B (rapid-cancel).

Both gates were added 2026-05-18 after the FNSE OLJAB X5 silent-rejection
loop: Avanza accepted orders into a closed orderbook and silently auto-
cancelled them, so grid_fisher re-armed every cycle. Gate A blocks
placement when the underlying orderbook hasn't traded recently; Gate B
detects the place / fast-cancel pattern at runtime and arms a multi-hour
cooldown.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any

import pytest

from portfolio import grid_fisher as gf
from portfolio.grid_fisher import (
    GridFisher,
    InstrumentState,
    ORDER_ARMED,
    ORDER_CANCELLED,
    TierOrder,
    load_state,
    save_state,
)


CATALOG = {
    "BULL_SILVER_X5_AVA_4": {
        "ob_id": "1650161",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
    },
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
    },
}


class _StaleQuoteSession:
    """Minimal FakeSession used by Gate A unit tests.

    The only behaviour the gate exercises is ``get_quote``; everything
    else is a no-op stub so the test never reaches real Avanza state.
    """

    def __init__(self, *, time_of_last_ms: int | None,
                 omit_field: bool = False) -> None:
        self.time_of_last_ms = time_of_last_ms
        self.omit_field = omit_field
        self.get_quote_calls = 0

    def get_quote(self, ob_id: str) -> dict[str, Any]:
        self.get_quote_calls += 1
        q: dict[str, Any] = {"buy": 42.5, "sell": 42.6, "last": 42.55}
        if not self.omit_field:
            q["timeOfLast"] = self.time_of_last_ms
        return q

    # Stubs so ``GridFisher.__init__`` doesn't choke on missing attrs.
    def place_buy_order(self, *a, **k): return {"orderRequestStatus": "SUCCESS", "orderId": "BUY1"}
    def place_sell_order(self, *a, **k): return {"orderRequestStatus": "SUCCESS", "orderId": "SELL1"}
    def place_stop_loss(self, *a, **k): return {"status": "SUCCESS", "stoplossOrderId": "STOP1"}
    def cancel_order(self, *a, **k): return {"orderRequestStatus": "SUCCESS"}
    def cancel_stop_loss(self, *a, **k): return {"status": "SUCCESS"}
    def get_open_orders(self): return []
    def get_positions(self): return []


def _build_fisher(session, tmp_path):
    f = GridFisher(
        session=session,
        catalog=CATALOG,
        state_path=tmp_path / "state.json",
        decisions_path=tmp_path / "decisions.jsonl",
    )
    f._order_delay_s = 0.0
    return f


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Gate A — quote-staleness
# ---------------------------------------------------------------------------


class TestGateAQuoteStaleness:

    def test_recent_quote_not_stale(self, tmp_path):
        """A trade <30 s ago does not trip the gate."""
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms() - 30_000)
        f = _build_fisher(sess, tmp_path)
        is_stale, age_s = f._is_quote_stale("1650161")
        assert is_stale is False
        assert age_s is not None and 0 <= age_s < 60

    def test_quote_just_over_threshold_is_stale(self, tmp_path, monkeypatch):
        """1 s past the configured threshold trips the gate."""
        # Lower the threshold so we can prove the boundary in a fast test.
        monkeypatch.setattr(
            gf, "GRID_QUOTE_STALENESS_THRESHOLD_S", 600,
            raising=False,
        )
        # The gate reads the value via the config import inside
        # _is_quote_stale — patch the source module too.
        from portfolio import grid_fisher_config as gfc
        monkeypatch.setattr(gfc, "GRID_QUOTE_STALENESS_THRESHOLD_S", 600,
                            raising=True)
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms() - 601_000)
        f = _build_fisher(sess, tmp_path)
        is_stale, age_s = f._is_quote_stale("1650161")
        assert is_stale is True
        assert age_s is not None and age_s > 600

    def test_missing_time_of_last_treated_as_stale(self, tmp_path):
        """Missing timeOfLast → fail-safe skip + one-shot warning."""
        sess = _StaleQuoteSession(time_of_last_ms=None, omit_field=True)
        f = _build_fisher(sess, tmp_path)
        is_stale, age_s = f._is_quote_stale("1650161")
        assert is_stale is True
        assert age_s is None
        # Calling again does not re-emit the warning (one-shot).
        assert "1650161" in f._quote_shape_warned
        # Second call still flagged stale, no additional shape warning.
        f._is_quote_stale("1650161")
        assert "1650161" in f._quote_shape_warned

    def test_quote_fetch_failure_treated_as_stale(self, tmp_path):
        """``_fetch_quote_cached`` returning None → fail-safe skip."""

        class _BrokenSession(_StaleQuoteSession):
            def get_quote(self, ob_id):  # noqa: ARG002
                raise RuntimeError("simulated network failure")

        sess = _BrokenSession(time_of_last_ms=None)
        f = _build_fisher(sess, tmp_path)
        is_stale, age_s = f._is_quote_stale("1650161")
        assert is_stale is True
        assert age_s is None

    def test_quote_cache_hits_within_ttl(self, tmp_path):
        """A second call within GRID_QUOTE_CACHE_SECS does not hit Avanza."""
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        f._is_quote_stale("1650161")
        f._is_quote_stale("1650161")
        assert sess.get_quote_calls == 1

    def test_future_dated_timeoflast_treated_as_stale(self, tmp_path):
        """A future ``timeOfLast`` (Avanza data corruption / clock skew)
        must fail-safe to stale — never silently clamp to age=0 and
        let placement through (review finding 2026-05-18)."""
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms() + 60_000)
        f = _build_fisher(sess, tmp_path)
        is_stale, age_s = f._is_quote_stale("1650161")
        assert is_stale is True
        assert age_s is None


# ---------------------------------------------------------------------------
# Gate B — rapid-cancel back-off
# ---------------------------------------------------------------------------


def _seed_cancelled_tier(
    inst: InstrumentState, *, tier: int, age_s: float,
) -> None:
    """Append a CANCELLED tier whose ``placed_ts`` is *age_s* in the past."""
    placed = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=age_s)
    inst.buy_ladder.append(
        TierOrder(
            tier=tier,
            order_id=f"BUY-{tier}",
            price=10.0,
            qty=100,
            placed_ts=placed.strftime("%Y-%m-%dT%H:%M:%SZ"),
            status=ORDER_CANCELLED,
        )
    )


class TestGateBRapidCancel:

    def test_single_rapid_cancel_increments_counter(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        _seed_cancelled_tier(inst, tier=0, age_s=10)

        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=0)

        assert inst.rapid_cancel_count == 1
        assert inst.cooldown_until is None

    def test_two_rapid_cancels_arm_cooldown(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        _seed_cancelled_tier(inst, tier=0, age_s=10)
        _seed_cancelled_tier(inst, tier=1, age_s=10)

        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=0)
        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=1)

        assert inst.rapid_cancel_count == 2
        assert inst.cooldown_until is not None
        # Cooldown should be ~6 h out.
        cd = _dt.datetime.fromisoformat(
            inst.cooldown_until.replace("Z", "+00:00")
        )
        delta = (cd - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
        assert 5 * 3600 < delta < 7 * 3600

    def test_slow_cancel_resets_streak(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        _seed_cancelled_tier(inst, tier=0, age_s=10)
        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=0)
        assert inst.rapid_cancel_count == 1

        _seed_cancelled_tier(inst, tier=1, age_s=600)  # 10 min — not rapid
        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=1)
        assert inst.rapid_cancel_count == 0
        assert inst.last_rapid_cancel_ts is None

    def test_fill_clears_streak(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        _seed_cancelled_tier(inst, tier=0, age_s=10)
        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=0)
        assert inst.rapid_cancel_count == 1

        # Simulate a real fill via record_fill.
        inst.buy_ladder.append(
            TierOrder(
                tier=2, order_id="BUY-fill", price=10.0, qty=50,
                placed_ts="2026-05-18T18:00:00Z", status=ORDER_ARMED,
            )
        )
        gf.record_fill(inst, tier_idx=2, fill_price=10.0, side="buy")

        assert inst.rapid_cancel_count == 0
        assert inst.last_rapid_cancel_ts is None

    def test_session_roll_resets_streak(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        inst.rapid_cancel_count = 2
        inst.last_rapid_cancel_ts = "2026-05-17T20:00:00Z"
        # Force a session date that's "older than today".
        f.state.session_id = "1970-01-01"

        rolled = gf.roll_session_if_new_day(f.state)

        assert rolled is True
        assert inst.rapid_cancel_count == 0
        assert inst.last_rapid_cancel_ts is None

    def test_missing_placed_ts_does_not_increment(self, tmp_path):
        sess = _StaleQuoteSession(time_of_last_ms=_now_ms())
        f = _build_fisher(sess, tmp_path)
        inst = f.state.by_instrument["1650161"]
        inst.buy_ladder.append(
            TierOrder(
                tier=0, order_id="BUY-orphan", price=10.0, qty=100,
                placed_ts="",  # malformed
                status=ORDER_CANCELLED,
            )
        )

        f._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=0)

        assert inst.rapid_cancel_count == 0


# ---------------------------------------------------------------------------
# State schema-version forward-compat
# ---------------------------------------------------------------------------


class TestStateVersionForwardAssert:

    def test_newer_version_starts_fresh(self, tmp_path, caplog):
        from portfolio.file_utils import atomic_write_json

        state_path = tmp_path / "state.json"
        forward = {
            "version": 999,
            "session_id": "2026-05-18",
            "halted": False,
            "halt_reason": None,
            "global_session_pnl_sek": 0.0,
            "global_max_dd_sek": 0.0,
            "by_instrument": {},
        }
        atomic_write_json(state_path, forward)

        import logging
        with caplog.at_level(logging.CRITICAL, logger="portfolio.grid_fisher"):
            state = load_state(state_path)

        # Fresh state means empty by_instrument.
        assert state.by_instrument == {}
        # And a critical log entry was emitted.
        crit = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert any("forward-incompatible" in (r.getMessage().lower())
                   for r in crit)

    def test_older_version_migrates_via_defaults(self, tmp_path):
        from portfolio.file_utils import atomic_write_json

        state_path = tmp_path / "state.json"
        # v1 shape — no rapid_cancel_count field.
        legacy = {
            "version": 1,
            "session_id": "2026-05-18",
            "halted": False,
            "halt_reason": None,
            "global_session_pnl_sek": 0.0,
            "global_max_dd_sek": 0.0,
            "by_instrument": {
                "1650161": {
                    "ob_id": "1650161",
                    "ticker": "XAG-USD",
                    "cert_name": "BULL_SILVER_X5_AVA_4",
                    "active_direction": "LONG",
                    "buy_ladder": [],
                    "sell_ladder": [],
                    "stop_loss_id": None,
                    "stop_loss_price": None,
                    "inventory_units": 0,
                    "avg_entry_price": 0.0,
                    "session_pnl_sek": 0.0,
                    "fills_this_session": 0,
                    "consecutive_losses": 0,
                    "cooldown_until": None,
                    "last_direction_flip_ts": None,
                    "eod_sell_order_id": None,
                },
            },
        }
        atomic_write_json(state_path, legacy)

        state = load_state(state_path)
        inst = state.by_instrument["1650161"]
        assert inst.rapid_cancel_count == 0
        assert inst.last_rapid_cancel_ts is None
        # Version field is migrated forward on load.
        assert state.version == gf.GRID_STATE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2026-06-12 (audit B4 fix 5): dynamic EOD cutoff from todayClosingTime.
# Normal day → keep 21:55 (todayClosingTime reports the EXCHANGE close,
# not the AVA market-maker window — memory reference_avanza_trading_hours);
# half-day (early close) → close minus margin; fetch failure → 21:50
# fallback + eod_close_time_fallback decision-log entry.
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_close_cache():
    def _reset():
        with gf._close_time_lock:
            gf._close_time_cache.clear()
            gf._close_fetch_failed_mono.clear()
            gf._close_fallback_logged.clear()
    _reset()
    yield
    _reset()


class TestEodCloseTimeResolver:
    def test_normal_day_keeps_default_cutoff(self, _reset_close_cache, tmp_path):
        cutoff = gf.resolve_eod_cutoff_hm(
            lambda: {"marketPlace": {"todayClosingTime": "17:30"}},
            log_path=tmp_path / "decisions.jsonl",
        )
        assert cutoff == (21, 55)

    def test_half_day_uses_early_close_minus_margin(
        self, _reset_close_cache, tmp_path,
    ):
        cutoff = gf.resolve_eod_cutoff_hm(
            lambda: {"marketPlace": {"todayClosingTime": "13:00"}},
            log_path=tmp_path / "decisions.jsonl",
        )
        assert cutoff == (12, 55)

    def test_fetch_failure_falls_back_and_logs(
        self, _reset_close_cache, tmp_path,
    ):
        log_file = tmp_path / "decisions.jsonl"

        def _boom():
            raise RuntimeError("session down")

        cutoff = gf.resolve_eod_cutoff_hm(_boom, log_path=log_file)
        assert cutoff == (21, 50)  # 21:55 minus safety margin
        entries = log_file.read_text(encoding="utf-8")
        assert "eod_close_time_fallback" in entries

    def test_missing_field_falls_back(self, _reset_close_cache, tmp_path):
        cutoff = gf.resolve_eod_cutoff_hm(
            lambda: {"quote": {"buy": 1.0}},
            log_path=tmp_path / "decisions.jsonl",
        )
        assert cutoff == (21, 50)

    def test_success_cached_for_the_day(self, _reset_close_cache, tmp_path):
        calls = []

        def _fetch():
            calls.append(1)
            return {"marketPlace": {"todayClosingTime": "13:00"}}

        first = gf.resolve_eod_cutoff_hm(_fetch, log_path=tmp_path / "d.jsonl")
        second = gf.resolve_eod_cutoff_hm(_fetch, log_path=tmp_path / "d.jsonl")
        assert first == second == (12, 55)
        assert len(calls) == 1

    def test_minutes_until_eod_honors_cutoff_param(self):
        # 10:00 UTC summer = 12:00 Stockholm (CEST). Cutoff 13:00 → 60 min.
        now = _dt.datetime(2026, 6, 12, 10, 0, tzinfo=_dt.timezone.utc)
        mins = gf.minutes_until_eod(now, cutoff_hm=(13, 0))
        assert abs(mins - 60.0) < 0.01

    def test_minutes_until_eod_default_unchanged(self):
        # Default cutoff stays 21:55 Stockholm — 10:00 UTC = 12:00 local
        # in June → 595 minutes.
        now = _dt.datetime(2026, 6, 12, 10, 0, tzinfo=_dt.timezone.utc)
        mins = gf.minutes_until_eod(now)
        assert abs(mins - 595.0) < 0.01


class TestEodCloseTimePlausibilityBand:
    """2026-06-12 (review fix 18d9d0cc #3): early-close accepted only in
    [12:00, 17:00) — garbage like 09:00 must NOT gate the grid all day."""

    def test_implausibly_early_close_treated_as_fetch_failure(
        self, _reset_close_cache, tmp_path,
    ):
        log_file = tmp_path / "decisions.jsonl"
        cutoff = gf.resolve_eod_cutoff_hm(
            lambda: {"marketPlace": {"todayClosingTime": "09:00"}},
            log_path=log_file,
        )
        assert cutoff == (21, 50)  # fallback, not 08:55
        entries = log_file.read_text(encoding="utf-8")
        assert "eod_close_time_fallback" in entries
        assert "implausible" in entries

    def test_boundary_noon_close_accepted(self, _reset_close_cache, tmp_path):
        cutoff = gf.resolve_eod_cutoff_hm(
            lambda: {"marketPlace": {"todayClosingTime": "12:00"}},
            log_path=tmp_path / "decisions.jsonl",
        )
        assert cutoff == (11, 55)
