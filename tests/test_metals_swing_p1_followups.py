"""P1 follow-up regression tests for metals_swing_trader.

Carved out 2026-05-02 on worktree fix/metals-p1-followups-20260502 to
pin five specific defects from docs/PLAN_metals_followups_20260502.md:

  * MC-P1-3 — pos_id collision on same-second buys
  * MC-P1-4 — zero-price sell on price-fetch failure
  * P1-9    — hardcoded usdsek=10.85 in exit optimizer
  * MC-P1-1 — orphan stop computed from entry not bid

Each test pins the *post-fix* behaviour. Add new tests here when
re-touching this code so future regressions are caught.
"""

from __future__ import annotations

import datetime
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_trader as mst  # noqa: E402

UTC = datetime.UTC


def _make_trader(cash: float = 50_000.0):
    """Minimal SwingTrader bypassing __init__ for these unit tests."""
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.page = object()
    trader.state = mst._default_state()
    trader.state["cash_sek"] = cash
    trader.state["positions"] = {}
    trader.check_count = 0
    trader.regime_history = {}
    trader.cash_sync_ok = True
    trader.cash_sync_was_ok = True
    trader.recon_failure_streak = 0
    trader.reconciled_once = True
    trader._jit_sync_tick = -1
    trader.kelly_no_edge_count = {}
    trader._orphans_migrated = True  # skip migration for these tests
    trader.warrant_catalog = {}
    return trader


def _silence_io(monkeypatch, trader=None):
    """Mute logging + Telegram + state save side-effects."""
    monkeypatch.setattr(mst, "_log", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_trade", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_decision", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    if trader is not None:
        monkeypatch.setattr(trader, "_set_stop_loss", lambda pid, **kw: None)


# ---------------------------------------------------------------------------
# MC-P1-3: pos_id must be collision-proof across same-second buys
# ---------------------------------------------------------------------------

def test_pos_id_unique_when_two_buys_land_in_same_second(monkeypatch):
    """Two _execute_buy calls in the same epoch second on different
    ob_ids must produce two distinct positions in state. Previous code
    used pos_id = f"pos_{int(time.time())}" which collided."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    # Pin time.time() so both calls see the same second.
    monkeypatch.setattr(mst.time, "time", lambda: 1735000000.0)

    warrant_a = {
        "key": "MINI_L_SILVER_AVA_301", "name": "MINI L SILVER AVA 301",
        "ob_id": "2334960", "api_type": "warrant",
        "live_leverage": 5.0, "underlying_price": 30.0,
    }
    warrant_b = {
        "key": "MINI_L_SILVER_SG", "name": "MINI L SILVER SG",
        "ob_id": "2043157", "api_type": "warrant",
        "live_leverage": 1.56, "underlying_price": 30.0,
    }
    sig = {"buy_count": 4, "sell_count": 1, "rsi": 45.0, "action": "BUY"}

    trader._execute_buy(
        warrant=warrant_a, units=50, ask_price=10.0,
        underlying_ticker="XAG-USD", sig=sig, total_cost=500.0,
        direction="LONG",
    )
    trader._execute_buy(
        warrant=warrant_b, units=30, ask_price=12.0,
        underlying_ticker="XAG-USD", sig=sig, total_cost=360.0,
        direction="LONG",
    )

    assert len(trader.state["positions"]) == 2, (
        f"expected 2 positions, got {len(trader.state['positions'])} — "
        f"keys: {list(trader.state['positions'].keys())}"
    )
    # Both positions retained — different ob_ids
    ob_ids = {p["ob_id"] for p in trader.state["positions"].values()}
    assert ob_ids == {"2334960", "2043157"}


# ---------------------------------------------------------------------------
# MC-P1-4: zero-price sell guard on price fetch failure
# ---------------------------------------------------------------------------

def _build_long_position(trader, *, pos_id="pos1"):
    """Add a LONG XAG position with a long-ago entry so EOD/MAX_HOLD
    don't intercept and only our injected exit reasons can fire."""
    entry_ts = datetime.datetime.now(UTC) - datetime.timedelta(minutes=20)
    trader.state["positions"][pos_id] = {
        "warrant_key": "MINI_L_SILVER_AVA_301",
        "warrant_name": "MINI L SILVER AVA 301",
        "ob_id": "2334960",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "units": 50,
        "entry_price": 10.0,
        "entry_underlying": 30.0,
        "entry_ts": entry_ts.isoformat(),
        "peak_underlying": 30.0,
        "trough_underlying": 30.0,
        "trailing_active": False,
        "stop_order_id": "STOP_X",
        "leverage": 5.0,
        "fill_verified": True,
        "buy_order_id": "BUY_X",
    }


def test_zero_price_sell_aborted_on_fetch_failure(monkeypatch):
    """When fetch_price returns None (or a dict with bid<=0), and an
    exit_reason would otherwise fire, _execute_sell MUST NOT be called.
    The position must remain in state for re-evaluation next cycle."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    _build_long_position(trader)

    # Drive the underlying down 5% so HARD_STOP fires (-2% to -3%
    # depending on which fix landed). This forces _execute_sell to be
    # the next call — which the guard must abort.
    prices = {"silver301": {"underlying": 28.0}}  # -6.6% from entry 30.0

    # fetch_price returns None — simulates Avanza API hiccup mid-cycle.
    monkeypatch.setattr(mst, "fetch_price", lambda *a, **k: None)

    # Spy on _execute_sell — must NOT be called.
    sell_calls = []
    monkeypatch.setattr(
        trader,
        "_execute_sell",
        lambda *a, **k: sell_calls.append((a, k)),
    )

    trader._check_exits(prices, signal_data=None)

    assert sell_calls == [], (
        f"_execute_sell must not run when current_bid<=0; called: {sell_calls}"
    )
    # Position must stay in state for next-cycle retry.
    assert "pos1" in trader.state["positions"]


def test_zero_price_sell_aborted_on_missing_bid_field(monkeypatch):
    """Variant: fetch_price returns a dict but missing 'bid' (or bid=0).
    Same guard: no sell, position retained."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    _build_long_position(trader)
    prices = {"silver301": {"underlying": 28.0}}

    monkeypatch.setattr(
        mst, "fetch_price",
        lambda *a, **k: {"underlying": 28.0, "ask": 9.0},  # no 'bid'
    )

    sell_calls = []
    monkeypatch.setattr(
        trader,
        "_execute_sell",
        lambda *a, **k: sell_calls.append((a, k)),
    )

    trader._check_exits(prices, signal_data=None)

    assert sell_calls == []
    assert "pos1" in trader.state["positions"]


def test_normal_sell_proceeds_when_bid_positive(monkeypatch):
    """Sanity check: when bid is positive, the SELL path runs as before.

    2026-05-11: HARD_STOP is now anchored on warrant pct change
    (STOP_LOSS_WARRANT_PCT=30). entry_price=10.0 → bid must be <= 7.0 to
    breach. Set bid=6.5 (-35% warrant), underlying=28.0 (irrelevant now).
    """
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    _build_long_position(trader)
    prices = {"silver301": {"underlying": 28.0}}

    monkeypatch.setattr(
        mst, "fetch_price",
        lambda *a, **k: {"underlying": 28.0, "bid": 6.50},  # -35% warrant
    )

    sell_calls = []
    monkeypatch.setattr(
        trader,
        "_execute_sell",
        lambda *a, **k: sell_calls.append((a, k)),
    )

    trader._check_exits(prices, signal_data=None)

    assert len(sell_calls) == 1, (
        f"expected exactly 1 sell call, got {len(sell_calls)}"
    )
    # Verify the bid passed to _execute_sell is positive
    args, _ = sell_calls[0]
    # _execute_sell(pos_id, pos, current_bid, underlying_price, reason)
    assert args[2] == 6.50


# ---------------------------------------------------------------------------
# P1-9: exit-optimizer must use live USD/SEK rate, not hardcoded 10.85
# ---------------------------------------------------------------------------

def test_exit_optimizer_uses_live_fx_rate(monkeypatch):
    """The MarketSnapshot passed to compute_exit_plan must carry the
    rate returned by portfolio.fx_rates.fetch_usd_sek, not a hardcoded
    constant. Tests by capturing the kwarg and asserting on it."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    _build_long_position(trader)
    prices = {"silver301": {"underlying": 30.5}}  # +1.66% — neither TP nor stop

    monkeypatch.setattr(
        mst, "fetch_price",
        lambda *a, **k: {"underlying": 30.5, "bid": 10.5},
    )

    # Patch the FX module to return a distinctive value (not 10.85).
    captured = {}
    fake_rate = 11.27

    def _fake_fetch_usd_sek():
        return fake_rate

    monkeypatch.setattr(
        "portfolio.fx_rates.fetch_usd_sek",
        _fake_fetch_usd_sek,
    )

    # Mock compute_exit_plan to capture its market arg and return a
    # benign plan that won't actually drive an exit.
    fake_plan = MagicMock()
    fake_plan.recommended.action = "hold"
    fake_plan.recommended.price_usd = 30.5
    fake_plan.recommended.ev_sek = 0.0
    fake_plan.recommended.fill_prob = 0.0
    fake_plan.recommended.risk_flags = []
    fake_plan.stop_hit_prob = 0.0
    fake_plan.market_exit.pnl_sek = 0.0

    def _capture_plan(opt_pos, opt_market, *args, **kwargs):
        captured["market"] = opt_market
        return fake_plan

    # Make session look open with plenty of remaining time so the
    # optimizer branch actually runs.
    fake_session = MagicMock()
    fake_session.is_open = True
    fake_session.remaining_minutes = 60
    fake_session.session_end = (
        datetime.datetime.now(UTC) + datetime.timedelta(hours=1)
    )

    with patch("portfolio.exit_optimizer.compute_exit_plan", _capture_plan), \
         patch("portfolio.session_calendar.get_session_info", lambda *a, **k: fake_session), \
         patch("portfolio.cost_model.get_cost_model", lambda *a, **k: MagicMock()):
        trader._check_exits(prices, signal_data=None)

    assert "market" in captured, (
        "compute_exit_plan was never called — branch did not execute. "
        "Check session/exits gating in _check_exits."
    )
    assert captured["market"].usdsek == pytest.approx(fake_rate), (
        f"MarketSnapshot.usdsek={captured['market'].usdsek!r} should be live "
        f"FX rate {fake_rate}, not the legacy hardcoded 10.85"
    )


def test_exit_optimizer_falls_back_to_10_85_when_fx_fails(monkeypatch):
    """If fetch_usd_sek raises (e.g. import failure / pathological state),
    the optimizer call must still proceed — falling back to 10.85 keeps
    the existing safety-net behaviour."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    _build_long_position(trader)
    prices = {"silver301": {"underlying": 30.5}}

    monkeypatch.setattr(
        mst, "fetch_price",
        lambda *a, **k: {"underlying": 30.5, "bid": 10.5},
    )

    def _broken_fetch_usd_sek():
        raise RuntimeError("FX module simulated failure")

    monkeypatch.setattr(
        "portfolio.fx_rates.fetch_usd_sek",
        _broken_fetch_usd_sek,
    )

    captured = {}

    fake_plan = MagicMock()
    fake_plan.recommended.action = "hold"
    fake_plan.recommended.price_usd = 30.5
    fake_plan.recommended.ev_sek = 0.0
    fake_plan.recommended.fill_prob = 0.0
    fake_plan.recommended.risk_flags = []
    fake_plan.stop_hit_prob = 0.0
    fake_plan.market_exit.pnl_sek = 0.0

    def _capture_plan(opt_pos, opt_market, *args, **kwargs):
        captured["market"] = opt_market
        return fake_plan

    fake_session = MagicMock()
    fake_session.is_open = True
    fake_session.remaining_minutes = 60
    fake_session.session_end = (
        datetime.datetime.now(UTC) + datetime.timedelta(hours=1)
    )

    with patch("portfolio.exit_optimizer.compute_exit_plan", _capture_plan), \
         patch("portfolio.session_calendar.get_session_info", lambda *a, **k: fake_session), \
         patch("portfolio.cost_model.get_cost_model", lambda *a, **k: MagicMock()):
        # Must not raise
        trader._check_exits(prices, signal_data=None)

    # The optimizer still runs and gets the documented fallback.
    assert "market" in captured
    assert captured["market"].usdsek == pytest.approx(10.85)


# ---------------------------------------------------------------------------
# MC-P1-1: orphan stop must be computed from current bid, not entry price
# ---------------------------------------------------------------------------

def test_set_stop_loss_uses_anchor_when_provided(monkeypatch):
    """_set_stop_loss(pos_id, anchor_price=BID) must compute the stop
    relative to the anchor bid, not pos['entry_price']. This is the
    orphan-ingest scenario: position entered at 100, current bid is 80,
    stop must be at 80*(1-leverage*und%) = 68 (5x@3%), NOT 100*(1-...)
    = 85 which is ABOVE current price and would trigger immediately."""
    trader = _make_trader()
    # NOTE: don't use _silence_io with trader= here — that stubs out
    # _set_stop_loss, which is the very method we're testing.
    monkeypatch.setattr(mst, "_log", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_trade", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_decision", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    # Position with entry well above current bid (orphan ingest scenario).
    trader.state["positions"]["pos1"] = {
        "warrant_key": "BULL_X5",
        "warrant_name": "BULL SILVER X5 AVA 4",
        "ob_id": "1650161",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "units": 100,
        "entry_price": 100.0,        # original orphan purchase
        "entry_underlying": 32.0,    # original orphan underlying
        "entry_ts": datetime.datetime.now(UTC).isoformat(),
        "leverage": 5.0,
        "fill_verified": True,
    }

    # Default path (no anchor) — stop computed from entry_price 100.
    trader._set_stop_loss("pos1")
    default_stop = trader.state["positions"]["pos1"].get("stop_order_id")
    # In DRY_RUN this just sets stop_order_id="DRY_RUN" as a marker.
    assert default_stop == "DRY_RUN"

    # Reset and compute again with an anchor at the (lower) current bid.
    trader.state["positions"]["pos1"]["stop_order_id"] = None

    captured = {}
    real_round = round

    def _capture(*a, **k):
        # Capture the trigger_price computed inside _set_stop_loss by
        # replacing place_stop_loss — we want the math, not the API call.
        # In DRY_RUN this isn't called, so we instead inspect the log.
        return True, "STOP_TEST"

    monkeypatch.setattr(mst, "place_stop_loss", _capture)
    # Switch to non-DRY so place_stop_loss is invoked.
    monkeypatch.setattr(mst, "DRY_RUN", False)

    # Capture the trigger/sell prices via the place_stop_loss mock.
    def _capture_args(page, account, ob_id, trigger, sell_price, units, valid_days=None):
        captured["trigger"] = trigger
        captured["sell"] = sell_price
        return True, "STOP_TEST_2"

    monkeypatch.setattr(mst, "place_stop_loss", _capture_args)

    # Anchor at current bid 80 (20% below entry).
    trader._set_stop_loss("pos1", anchor_price=80.0)

    # 2026-05-11: stop now anchored to warrant pct change (STOP_LOSS_WARRANT_PCT),
    # not underlying × leverage. trigger = anchor × (1 - STOP_LOSS_WARRANT_PCT/100)
    # For STOP_LOSS_WARRANT_PCT=30: 80 × 0.70 = 56.0
    expected_drop = mst.STOP_LOSS_WARRANT_PCT / 100
    expected_trigger = real_round(80.0 * (1 - expected_drop), 2)
    assert captured.get("trigger") == expected_trigger, (
        f"With anchor=80 expected trigger={expected_trigger}; "
        f"got {captured.get('trigger')!r}"
    )
    # Sanity: the trigger must be BELOW current bid 80
    assert captured["trigger"] < 80.0


def test_ingest_orphan_passes_current_bid_as_stop_anchor(monkeypatch):
    """MC-P1-1 end-to-end: ingest_position fetches the current bid and
    forwards it to _set_stop_loss as anchor_price. Spies on
    _set_stop_loss to capture the kwargs."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: None)
    monkeypatch.setattr(mst, "_collect_existing_stops_for", lambda ob: [])
    # Provide a known catalog entry so meta lookup succeeds.
    trader.warrant_catalog = {
        "BULL_X5": {
            "ob_id": "1650161",
            "api_type": "certificate",
            "underlying": "XAG-USD",
            "leverage": 5.0,
            "name": "BULL SILVER X5 AVA 4",
        },
    }
    monkeypatch.setattr(mst, "_lookup_known_warrant", lambda ob: None)
    # Current bid is 80, but orphan was originally bought at 100.
    monkeypatch.setattr(mst, "fetch_price", lambda *a, **k: {"bid": 80.0})

    captured_kwargs = {}

    def _spy_set_stop(pos_id, **kw):
        captured_kwargs[pos_id] = kw
        trader.state["positions"][pos_id]["stop_order_id"] = "STOP_SPIED"

    trader._set_stop_loss = _spy_set_stop

    pos_id = trader.ingest_position(
        ob_id="1650161",
        units=100,
        entry_price=100.0,        # original orphan purchase, far above current
        underlying_price=32.0,
        set_stop_loss=True,
    )

    assert pos_id is not None, "ingest should succeed for known orphan"
    assert pos_id in captured_kwargs, "_set_stop_loss should have been called"
    assert captured_kwargs[pos_id].get("anchor_price") == 80.0, (
        f"Expected anchor_price=80.0 (current bid), got "
        f"{captured_kwargs[pos_id]!r}"
    )


def test_ingest_orphan_falls_back_when_bid_fetch_fails(monkeypatch):
    """If the bid fetch raises or returns None, ingest still calls
    _set_stop_loss but with anchor_price=None — _set_stop_loss falls back
    to the entry-price anchor (current pre-fix behaviour). Better an
    entry-anchored stop than no stop at all on an orphan."""
    trader = _make_trader()
    _silence_io(monkeypatch, trader)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: None)
    monkeypatch.setattr(mst, "_collect_existing_stops_for", lambda ob: [])
    trader.warrant_catalog = {
        "BULL_X5": {
            "ob_id": "1650161",
            "api_type": "certificate",
            "underlying": "XAG-USD",
            "leverage": 5.0,
            "name": "BULL SILVER X5 AVA 4",
        },
    }
    monkeypatch.setattr(mst, "_lookup_known_warrant", lambda ob: None)

    def _broken_fetch_price(*a, **k):
        raise ConnectionError("Avanza unreachable")

    monkeypatch.setattr(mst, "fetch_price", _broken_fetch_price)

    captured_kwargs = {}

    def _spy_set_stop(pos_id, **kw):
        captured_kwargs[pos_id] = kw
        trader.state["positions"][pos_id]["stop_order_id"] = "STOP_FALLBACK"

    trader._set_stop_loss = _spy_set_stop

    pos_id = trader.ingest_position(
        ob_id="1650161",
        units=100,
        entry_price=100.0,
        underlying_price=32.0,
        set_stop_loss=True,
    )

    assert pos_id is not None
    assert pos_id in captured_kwargs
    assert captured_kwargs[pos_id].get("anchor_price") is None, (
        f"Expected anchor_price=None on fetch failure, got "
        f"{captured_kwargs[pos_id]!r}"
    )


def test_hard_stop_widened_per_user_5x_cert_preference():
    """P1-8 (2026-05-02): user feedback memory 'Wider stop-losses' explicitly
    states '5x certs need -15%+ stops, not -8%, to survive intraday wicks'.
    HARD_STOP_UNDERLYING_PCT * 5 must produce a 15%+ certificate stop.
    Pin the value so a future tune-down silently regressing to 2.0% is
    caught at test time, not in production after a wick stop-out."""
    import metals_swing_config as cfg
    assert cfg.HARD_STOP_UNDERLYING_PCT >= 3.0, (
        f"HARD_STOP_UNDERLYING_PCT={cfg.HARD_STOP_UNDERLYING_PCT} too tight; "
        f"on 5x lev that's only {cfg.HARD_STOP_UNDERLYING_PCT * 5}% cert stop. "
        f"User memory: '5x certs need -15%+ stops to survive intraday wicks'."
    )


def test_hardware_stop_at_least_as_loose_as_software_stop():
    """MC-P1-2 (2026-05-02): the hardware (Avanza) stop is a safety net
    for process-down scenarios. It must NOT trigger before the software
    HARD_STOP. STOP_LOSS_UNDERLYING_PCT (HW) must be >= HARD_STOP_UNDERLYING_PCT
    (SW). If HW < SW, the broker fires first and pre-empts SW's
    smarter exit logic."""
    import metals_swing_config as cfg
    assert cfg.STOP_LOSS_UNDERLYING_PCT >= cfg.HARD_STOP_UNDERLYING_PCT, (
        f"HW stop {cfg.STOP_LOSS_UNDERLYING_PCT}% tighter than SW stop "
        f"{cfg.HARD_STOP_UNDERLYING_PCT}% — Avanza will fire before "
        f"in-process logic, defeating the safety-net design."
    )


def test_set_stop_loss_default_path_unchanged(monkeypatch):
    """Without anchor_price, behaviour must match pre-fix: stop computed
    from pos['entry_price']. Pinned so the default code path of fresh
    buys never accidentally regresses."""
    trader = _make_trader()
    # Don't stub _set_stop_loss — that's the method under test.
    monkeypatch.setattr(mst, "_log", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_trade", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_decision", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(mst, "DRY_RUN", False)

    trader.state["positions"]["pos1"] = {
        "warrant_key": "BULL_X5",
        "warrant_name": "BULL SILVER X5 AVA 4",
        "ob_id": "1650161",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "units": 100,
        "entry_price": 100.0,
        "entry_underlying": 32.0,
        "entry_ts": datetime.datetime.now(UTC).isoformat(),
        "leverage": 5.0,
        "fill_verified": True,
    }

    captured = {}

    def _capture_args(page, account, ob_id, trigger, sell_price, units, valid_days=None):
        captured["trigger"] = trigger
        return True, "STOP_FRESH"

    monkeypatch.setattr(mst, "place_stop_loss", _capture_args)

    trader._set_stop_loss("pos1")  # no anchor

    # 2026-05-11: stop anchored to warrant pct change (STOP_LOSS_WARRANT_PCT).
    # entry=100, STOP_LOSS_WARRANT_PCT=30 → 100 × 0.70 = 70.0
    expected_drop = mst.STOP_LOSS_WARRANT_PCT / 100
    expected_trigger = round(100.0 * (1 - expected_drop), 2)
    assert captured["trigger"] == expected_trigger
