"""Pricing-layer tests. Offline — the Avanza call is injected.

Synthetic fixtures only; this repo is public.
"""

import pytest

from portfolio.swedbank.instruments import INSTRUMENTS, by_key
from portfolio.swedbank.pricing import Quote, build_quote, sweep, value_holding

NOW_MS = 1_785_500_000_000


def raw(bid=100.0, ask=100.2, last=100.1, updated=NOW_MS, vol=1000.0, rt=True):
    return {
        "buy": bid,
        "sell": ask,
        "last": last,
        "updated": updated,
        "totalVolumeTraded": vol,
        "isRealTime": rt,
    }


class TestMarkingRule:
    def test_last_inside_spread_is_used(self):
        q = build_quote(by_key("NVDA"), raw(100.0, 100.2, 100.1), now_ms=NOW_MS)
        assert q.mark == pytest.approx(100.1)
        assert q.mark_basis == "last"
        assert not q.stale_last

    def test_stale_last_above_spread_marks_at_mid(self):
        # The measured real case: a thin warrant whose last print sits well
        # above the live offer. Marking at last would overstate the position.
        q = build_quote(by_key("MINI-TSMC"), raw(304.47, 304.68, 314.36), now_ms=NOW_MS)
        assert q.mark_basis == "mid"
        assert q.mark == pytest.approx(304.575)
        assert q.stale_last
        assert "mid" in q.note

    def test_stale_last_below_spread_marks_at_mid(self):
        q = build_quote(by_key("NVDA"), raw(200.0, 200.4, 180.0), now_ms=NOW_MS)
        assert q.mark_basis == "mid"
        assert q.stale_last

    def test_marginally_outside_spread_is_tolerated(self):
        # Within STALE_LAST_TOL of mid — a tick outside the quote is normal
        # microstructure, not a stale print. Do not flap.
        q = build_quote(by_key("NVDA"), raw(100.0, 100.2, 100.3), now_ms=NOW_MS)
        assert q.mark_basis == "last"
        assert not q.stale_last

    def test_missing_last_falls_back_to_mid(self):
        q = build_quote(by_key("NVDA"), raw(last=None), now_ms=NOW_MS)
        assert q.mark_basis == "mid"
        assert q.mark == pytest.approx(100.1)

    def test_no_usable_price_raises(self):
        with pytest.raises(ValueError, match="neither usable last nor bid/ask"):
            build_quote(by_key("NVDA"), {"buy": None, "sell": None, "last": None})

    def test_spread_percent_computed(self):
        q = build_quote(by_key("BEAMMW-B"), raw(14.20, 14.44, 14.30), now_ms=NOW_MS)
        assert q.spread_pct == pytest.approx(1.674, abs=0.01)

    def test_age_from_updated_timestamp(self):
        q = build_quote(by_key("NVDA"), raw(updated=NOW_MS - 45_000), now_ms=NOW_MS)
        assert q.age_s == pytest.approx(45.0)

    def test_age_never_negative_on_clock_skew(self):
        q = build_quote(by_key("NVDA"), raw(updated=NOW_MS + 10_000), now_ms=NOW_MS)
        assert q.age_s == 0.0


class TestSweep:
    def test_all_ok(self):
        s = sweep(keys=["NVDA", "MU"], quote_fn=lambda ob: raw(), now_ms=NOW_MS)
        assert s.ok
        assert set(s.quotes) == {"NVDA", "MU"}
        assert all(q.source == "avanza" for q in s.quotes.values())

    def test_partial_failure_does_not_abort_the_sweep(self):
        def fn(ob):
            if ob == by_key("MU").avanza_ob:
                raise RuntimeError("boom")
            return raw()

        s = sweep(
            keys=["NVDA", "MU"],
            quote_fn=fn,
            now_ms=NOW_MS,
            # Inject the fallback too. Without this the test passes or fails
            # depending on whether Alpaca credentials happen to be reachable,
            # which silently changes WHICH code path is under test.
            fallback_fn=lambda inst: None,
            cache={
                "MU": {
                    "key": "MU",
                    "mark": 5.0,
                    "currency": "USD",
                    "source": "avanza",
                    "mark_basis": "last",
                    "as_of_ms": NOW_MS - 3600_000,
                }
            },
        )
        assert "NVDA" in s.quotes
        assert s.quotes["MU"].degraded
        assert s.quotes["MU"].source.endswith(":cached")
        assert not s.ok

    def test_cached_fallback_recomputes_age(self):
        s = sweep(
            keys=["MINI-TSMC"],
            fallback_fn=lambda inst: None,
            quote_fn=lambda ob: (_ for _ in ()).throw(RuntimeError()),
            now_ms=NOW_MS,
            cache={
                "MINI-TSMC": {
                    "key": "MINI-TSMC",
                    "mark": 300.0,
                    "currency": "SEK",
                    "source": "avanza",
                    "mark_basis": "mid",
                    "as_of_ms": NOW_MS - 7200_000,
                }
            },
        )
        q = s.quotes["MINI-TSMC"]
        assert q.age_s == pytest.approx(7200.0)
        assert q.degraded

    def test_stockholm_has_no_alpaca_fallback_and_errors_without_cache(self):
        s = sweep(
            keys=["INVE-B"],
            quote_fn=lambda ob: (_ for _ in ()).throw(RuntimeError("session dead")),
            now_ms=NOW_MS,
            cache={},
        )
        assert "INVE-B" not in s.quotes
        assert "no price available" in s.errors["INVE-B"]

    def test_fx_failure_is_recorded_not_swallowed(self):
        def bad_fx():
            raise RuntimeError("fx down")

        s = sweep(keys=["NVDA"], quote_fn=lambda ob: raw(), now_ms=NOW_MS, fx_fn=bad_fx)
        assert "__fx__" in s.errors
        assert "USD/SEK unavailable" in s.errors["__fx__"]

    def test_fx_recorded_when_available(self):
        s = sweep(
            keys=["NVDA"], quote_fn=lambda ob: raw(), now_ms=NOW_MS, fx_fn=lambda: 9.5
        )
        assert s.fx["USDSEK"] == pytest.approx(9.5)

    def test_sequential_not_concurrent(self):
        # P2-6: the real-money metals loop shares this Avanza session. Assert
        # the sweep issues calls one at a time.
        concurrent, active = [], []

        def fn(ob):
            active.append(ob)
            concurrent.append(len(active))
            active.pop()
            return raw()

        sweep(keys=list(INSTRUMENTS)[:8], quote_fn=fn, now_ms=NOW_MS)
        assert max(concurrent) == 1

    def test_sweep_covers_every_instrument_by_default(self):
        s = sweep(quote_fn=lambda ob: raw(), now_ms=NOW_MS)
        assert set(s.quotes) == set(INSTRUMENTS)


class TestValueHolding:
    def test_local_currency_ignores_fx(self):
        q = Quote(
            key="INVE-B", mark=400.0, currency="SEK", source="avanza", mark_basis="last"
        )
        assert value_holding(10, q, fx=9.5) == pytest.approx(4000.0)

    def test_foreign_applies_fx(self):
        q = Quote(
            key="NVDA", mark=200.0, currency="USD", source="avanza", mark_basis="last"
        )
        assert value_holding(3, q, fx=9.5) == pytest.approx(5700.0)

    def test_missing_fx_raises_rather_than_defaulting_to_one(self):
        # Defaulting to 1.0 would undervalue a USD position by ~90% silently.
        q = Quote(
            key="NVDA", mark=200.0, currency="USD", source="avanza", mark_basis="last"
        )
        with pytest.raises(ValueError, match="no FX rate"):
            value_holding(3, q, fx=None)


class TestCorruptCacheDoesNotCrashSweep:
    """The cache branch only runs after live sources failed. An exception there
    would turn a degraded-but-working sweep into a crash, and the bad cache
    persists on disk so the loop would crashloop every cycle."""

    def _dead(self, ob):
        raise RuntimeError("session dead")

    @pytest.mark.parametrize(
        "bad",
        [
            {},                                   # empty
            {"key": "MU"},                        # missing mark/currency/source
            {"mark": 5.0},                        # missing key
            {"key": "MU", "mark": None, "currency": "USD",
             "source": "avanza", "mark_basis": "last"},
            {"key": "MU", "mark": 5.0, "currency": "USD", "source": "avanza",
             "mark_basis": "last", "bogus_field": 1},   # unknown key is dropped
        ],
    )
    def test_malformed_entry_is_skipped_not_raised(self, bad):
        s = sweep(
            keys=["MU"],
            quote_fn=self._dead,
            fallback_fn=lambda inst: None,
            now_ms=NOW_MS,
            cache={"MU": bad},
        )
        # Must not raise. Either it produced a usable quote or recorded an error.
        assert "MU" in s.quotes or "MU" in s.errors

    def test_wholly_unusable_cache_records_error(self):
        s = sweep(
            keys=["MU"],
            quote_fn=self._dead,
            fallback_fn=lambda inst: None,
            now_ms=NOW_MS,
            cache={"MU": {"nonsense": True}},
        )
        assert "MU" not in s.quotes
        assert "no price available" in s.errors["MU"]
