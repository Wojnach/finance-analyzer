"""Tests for the Swedbank OHLCV + signals layer.

Fully OFFLINE — the Avanza chart call and the Alpaca fallback are both injected.
Fixtures are SYNTHETIC; this repo is public, so never seed from the live book.

Four contract bugs hid in these two modules and none of them raised: a 3-tuple
read as a dict, `regime` vs `_regime`, `LONG/SHORT` vs `buy/sell`, and hardcoded
UTC session closes that were an hour wrong all winter. The tests below pin each
of those contracts so a regression fails loudly instead of silently producing
confident, well-formed, wrong output.
"""

import datetime

import pytest

from portfolio.swedbank import ohlcv, signals
from portfolio.swedbank.instruments import AssetClass, by_key


def _bars(n=260, start=100.0, step=0.25, vol=True):
    """Synthetic ascending series, Avanza chart payload shape."""
    t0 = 1_700_000_000_000
    out = []
    for i in range(n):
        c = start + i * step
        row = {
            "timestamp": t0 + i * 86_400_000,
            "open": c - 0.1,
            "high": c + 0.4,
            "low": c - 0.4,
            "close": c,
        }
        if vol:
            row["totalVolumeTraded"] = 1000 + i
        out.append(row)
    return {"ohlc": out}


class TestToFrame:
    def test_happy_path_columns(self):
        df = ohlcv.to_frame(_bars())
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 260

    def test_missing_volume_defaults_rather_than_failing(self):
        # Some instruments omit volume. Volume-dependent signals should degrade
        # on their own, not sink the whole fetch.
        df = ohlcv.to_frame(_bars(vol=False))
        assert (df["volume"] == 0).all()

    def test_empty_series_raises(self):
        with pytest.raises(ohlcv.OhlcvError):
            ohlcv.to_frame({"ohlc": []})
        with pytest.raises(ohlcv.OhlcvError):
            ohlcv.to_frame({})

    def test_too_few_bars_raises(self):
        with pytest.raises(ohlcv.OhlcvError, match="usable bars"):
            ohlcv.to_frame(_bars(n=10))

    def test_missing_price_column_raises(self):
        payload = {"ohlc": [{"timestamp": 1, "open": 1, "high": 2, "low": 0}] * 40}
        with pytest.raises(ohlcv.OhlcvError, match="missing columns"):
            ohlcv.to_frame(payload)

    def test_sorted_ascending_even_if_input_is_not(self):
        p = _bars(n=40)
        p["ohlc"] = list(reversed(p["ohlc"]))
        df = ohlcv.to_frame(p)
        assert df["timestamp"].is_monotonic_increasing

    def test_non_numeric_close_dropped(self):
        p = _bars(n=40)
        p["ohlc"][5]["close"] = "not-a-number"
        df = ohlcv.to_frame(p)
        assert len(df) == 39


class TestFetchFallback:
    def _boom(self, *a, **k):
        raise RuntimeError("avanza down")

    def test_avanza_success_reports_source(self):
        df, src = ohlcv.fetch(by_key("NVDA"), chart_fn=lambda *a: _bars())
        assert src == "avanza" and len(df) == 260

    def test_us_falls_back_to_alpaca(self):
        df, src = ohlcv.fetch(
            by_key("NVDA"),
            chart_fn=self._boom,
            alpaca_fn=lambda sym, hz: ohlcv.to_frame(_bars()),
        )
        assert src == "alpaca:fallback"

    def test_stockholm_has_no_fallback_and_raises(self):
        # Critical: Alpaca cannot see Swedish listings. Returning anything here
        # would mask an Avanza outage instead of surfacing it.
        with pytest.raises(ohlcv.OhlcvError, match="no fallback"):
            ohlcv.fetch(by_key("INVE-B"), chart_fn=self._boom)

    def test_alpaca_returning_too_few_bars_raises(self):
        with pytest.raises(ohlcv.OhlcvError):
            ohlcv.fetch(
                by_key("NVDA"),
                chart_fn=self._boom,
                alpaca_fn=lambda sym, hz: ohlcv.to_frame(_bars(n=30)).head(5),
            )


class TestApplicable:
    def test_excludes_crypto_and_metals_families(self):
        got = set(signals.applicable_for(by_key("NVDA")))
        assert not got & {"futures_flow", "funding", "crypto_macro", "onchain"}
        assert "metals_cross_asset" not in got

    def test_orderbook_flow_excluded_for_equity(self):
        # The bug this guards: an unregistered ticker reads as "not a stock" via
        # global set membership, so this metals/crypto-only signal leaks onto
        # equities. We gate on the instrument's explicit asset_class instead.
        assert "orderbook_flow" not in signals.applicable_for(by_key("NVDA"))

    def test_uninformative_excluded(self):
        got = set(signals.applicable_for(by_key("NVDA")))
        assert not got & signals._UNINFORMATIVE

    def test_non_empty(self):
        assert len(signals.applicable_for(by_key("NVDA"))) > 3


class TestUnderlyingMapping:
    @pytest.mark.parametrize(
        "key,under",
        [("XBT-BTC", "BTC-USD"), ("XBT-ETH", "ETH-USD"), ("MINI-TSMC", "TSM")],
    )
    def test_leveraged_products_use_underlying(self, key, under):
        assert signals.UNDERLYING[key] == under

    def test_plain_equity_is_its_own_signal_ticker(self):
        assert signals.UNDERLYING.get("NVDA") is None


class TestEvaluateErrorPath:
    def test_missing_data_yields_error_not_a_fabricated_hold(self):
        # THE important one. A monitoring page that renders HOLD when it has no
        # data is worse than one that renders nothing: "cannot evaluate" and
        # "evaluation says do nothing" are different statements.
        r = signals.evaluate(
            by_key("INVE-B"),
            chart_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert r.get("error")
        assert r.get("action") is None
        assert r.get("confidence") is None

    def test_short_history_errors(self):
        r = signals.evaluate(
            by_key("NVDA"),
            chart_fn=lambda *a: _bars(n=8),
            alpaca_fn=lambda *a: None,  # isolate: NVDA has a real fallback
        )
        assert r.get("error")
        assert r.get("action") is None


class TestHoursRemaining:
    """Guards the DST bug: hardcoded UTC closes were an hour wrong all winter."""

    def test_returns_tuple_of_hours_and_open_flag(self):
        h, is_open = signals._hours_remaining(by_key("NVDA"))
        assert isinstance(h, float) and isinstance(is_open, bool)

    def test_hours_always_positive_and_bounded(self):
        for key in ("NVDA", "INVE-B"):
            h, _ = signals._hours_remaining(by_key(key))
            assert 0 < h <= 9.0, f"{key}: {h}"

    def test_closed_market_capped_to_one_session(self):
        # Wall-clock to the next close is 60h+ over a weekend; feeding that in
        # scales the projection by sqrt(60) and yields a meaningless ladder.
        for key, cap in (("NVDA", 6.5), ("INVE-B", 8.5)):
            h, is_open = signals._hours_remaining(by_key(key))
            if not is_open:
                assert h <= cap + 1e-9

    def test_dst_is_not_hardcoded(self):
        """The venue close must track local time, not a fixed UTC constant."""
        from zoneinfo import ZoneInfo

        for tz, hh, mm, summer_utc, winter_utc in (
            ("Europe/Stockholm", 17, 30, 15, 16),
            ("America/New_York", 16, 0, 20, 21),
        ):
            for date, expect in (
                ("2026-07-15", summer_utc),
                ("2026-01-15", winter_utc),
            ):
                local = (
                    datetime.datetime.fromisoformat(date + "T00:00")
                    .replace(tzinfo=ZoneInfo(tz))
                    .replace(hour=hh, minute=mm)
                )
                assert local.astimezone(datetime.timezone.utc).hour == expect


class TestNoLayer1Contamination:
    def test_own_signal_log_not_tier1s(self):
        # accuracy_stats blends every ticker in data/signal_log.jsonl into one
        # global per-signal figure Tier-1 falls back on, and our rows would evict
        # real history from its 50k-row tail.
        assert signals.SIGNAL_LOG != "data/signal_log.jsonl"
        assert "swedbank" in signals.SIGNAL_LOG

    def test_import_does_not_mutate_ticker_registries(self):
        import importlib

        from portfolio import tickers

        before = tuple(
            set(getattr(tickers, n))
            for n in ("CRYPTO_SYMBOLS", "METALS_SYMBOLS", "STOCK_SYMBOLS", "SYMBOLS")
        )
        importlib.reload(signals)
        after = tuple(
            set(getattr(tickers, n))
            for n in ("CRYPTO_SYMBOLS", "METALS_SYMBOLS", "STOCK_SYMBOLS", "SYMBOLS")
        )
        assert before == after

    def test_module_never_flushes_sentiment_state(self):
        import pathlib

        # Must check for a CALL, not the string — the module docstring names it
        # while explaining why we never invoke it.
        import ast

        tree = ast.parse(pathlib.Path(signals.__file__).read_text())
        called = {
            (
                n.func.attr
                if isinstance(n.func, ast.Attribute)
                else getattr(n.func, "id", "")
            )
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        assert "flush_sentiment_state" not in called

    def test_log_snapshot_writes_one_line(self, tmp_path):
        path = tmp_path / "log.jsonl"
        signals.log_snapshot(
            {"NVDA": {"action": "BUY", "confidence": 0.6}}, path=str(path)
        )
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1


class TestSideContract:
    """price_targets branches on lowercase buy/sell; LONG/SHORT matched neither
    and every trajectory ran with inverted directional logic."""

    def test_price_targets_uses_lowercase_sides(self):
        import inspect

        from portfolio import price_targets

        src = inspect.getsource(price_targets)
        assert 'side == "sell"' in src
        assert 'side == "buy"' in src

    def test_module_maps_to_lowercase(self):
        import pathlib

        src = pathlib.Path(signals.__file__).read_text()
        assert 'side = "sell" if action == "SELL" else "buy"' in src


class TestAssetClasses:
    def test_certificates_and_warrant_are_not_equity(self):
        for key in ("XBT-BTC", "XBT-ETH", "MINI-TSMC"):
            assert by_key(key).asset_class is not AssetClass.EQUITY


class TestUnderlyingIsNeverSelfReferencing:
    """The seventh contract bug: BTC-USD and ETH-USD are NOT pinned in
    INSTRUMENTS, so `INSTRUMENTS.get(signal_ticker, inst)` silently fell back to
    the CERTIFICATE — computing indicators on exactly the thin decaying SEK
    series the underlying mapping exists to avoid, while the payload still
    claimed the underlying was used. Only MINI-TSMC worked, because TSM happens
    to be pinned, and that was the one verified live."""

    def test_crypto_underlyings_are_not_pinned(self):
        from portfolio.swedbank.instruments import INSTRUMENTS

        assert "BTC-USD" not in INSTRUMENTS
        assert "ETH-USD" not in INSTRUMENTS
        assert "TSM" in INSTRUMENTS

    @pytest.mark.parametrize("key", ["XBT-BTC", "XBT-ETH"])
    def test_never_fetches_the_certificates_own_orderbook(self, key):
        seen = []

        def chart(ob, *a, **k):
            seen.append(ob)
            return _bars()

        signals.evaluate(
            by_key(key),
            chart_fn=chart,
            alpaca_fn=lambda *a: None,
            ticker_fn=lambda t, horizon="1d": (ohlcv.to_frame(_bars()), "price_source"),
        )
        assert (
            by_key(key).avanza_ob not in seen
        ), "fetched the certificate's own orderbook while claiming the underlying"

    @pytest.mark.parametrize("key", ["XBT-BTC", "XBT-ETH"])
    def test_unfetchable_underlying_errors_rather_than_self_referencing(self, key):
        r = signals.evaluate(
            by_key(key),
            chart_fn=lambda *a: _bars(),
            ticker_fn=lambda t, horizon="1d": (_ for _ in ()).throw(
                RuntimeError("down")
            ),
        )
        assert r.get("error")
        assert r.get("action") is None


class TestUniverseInjectionAndStoHours:
    def test_evaluate_universe_forwards_ticker_fn(self):
        """evaluate_universe is the LOOP's entry point. Without ticker_fn
        passthrough, a fully-faked loop test still hit live Binance for the XBT
        rows — reintroducing, one level up, the exact isolation hole that adding
        alpaca_fn was meant to close."""
        used = []
        signals.evaluate_universe(
            keys=["XBT-BTC"],
            chart_fn=lambda *a: _bars(),
            alpaca_fn=lambda *a: None,
            ticker_fn=lambda t, horizon="1d": used.append(t)
            or (ohlcv.to_frame(_bars()), "price_source"),
        )
        assert used == ["BTC-USD"], "real fetch_by_ticker was used instead of the fake"

    def test_sto_hours_normalised_to_us_equivalent(self):
        """price_targets._year_fraction hardcodes 6.5h as one trading day, but a
        Stockholm session is 8.5h and one daily bar spans it — so a full session
        must equal ONE trading day of variance. Uncorrected this inflated every
        STO band by sqrt(8.5/6.5) = 1.143, permanently."""
        r = signals.evaluate(
            by_key("INVE-B"), chart_fn=lambda *a: _bars(), alpaca_fn=lambda *a: None
        )
        t = r.get("trajectory") or {}
        raw, _ = signals._hours_remaining(by_key("INVE-B"))
        assert t.get("hours_remaining") == pytest.approx(raw * (6.5 / 8.5), abs=0.0051)

    def test_us_hours_not_rescaled(self):
        r = signals.evaluate(
            by_key("NVDA"), chart_fn=lambda *a: _bars(), alpaca_fn=lambda *a: None
        )
        raw, _ = signals._hours_remaining(by_key("NVDA"))
        assert (r.get("trajectory") or {}).get("hours_remaining") == pytest.approx(
            raw, abs=0.0051
        )
