"""Watchlist tracking: daily refresh, dedupe against the book, fail-closed rows."""

from __future__ import annotations

import datetime

import pytest

from portfolio.swedbank import watchlist as wl


def _cache(entries=None, refreshed_at=None, dead=None):
    return {
        "refreshed_at": refreshed_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lists": ["Min bevakningslista"],
        "entries": entries or [],
        "dead": dead or [],
    }


class TestRefreshCache:
    def test_resolves_and_records_dead_orderbooks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wl, "CACHE_FILE", str(tmp_path / "c.json"))

        def fake_api(path):
            assert path == wl.WATCHLIST_PATH
            return [{"name": "L", "orderbookIds": ["1", "2", "1"]}]

        def fake_resolve(ob):
            if ob == "2":
                raise RuntimeError("Avanza API error 404")
            return {"ob": ob, "name": f"N{ob}", "ticker": f"T{ob}", "currency": "USD"}

        cache = wl.refresh_cache(api_get_fn=fake_api, resolve_fn=fake_resolve)
        assert [e["ob"] for e in cache["entries"]] == ["1"]  # deduped, dead removed
        assert cache["dead"][0]["ob"] == "2"

    def test_non_list_response_raises(self):
        with pytest.raises(RuntimeError):
            wl.fetch_watchlists(api_get_fn=lambda p: {"error": "html login page"})


class TestEnsureFresh:
    def test_fresh_cache_is_not_refetched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wl, "CACHE_FILE", str(tmp_path / "c.json"))
        from portfolio.file_utils import atomic_write_json

        atomic_write_json(wl.CACHE_FILE, _cache())
        out = wl.ensure_fresh(
            api_get_fn=lambda p: pytest.fail("must not hit the API when fresh")
        )
        assert out["entries"] == []

    def test_stale_cache_triggers_refresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wl, "CACHE_FILE", str(tmp_path / "c.json"))
        from portfolio.file_utils import atomic_write_json

        old = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=25)
        ).isoformat()
        atomic_write_json(wl.CACHE_FILE, _cache(refreshed_at=old))
        called = {}

        def fake_api(path):
            called["yes"] = True
            return [{"name": "L", "orderbookIds": []}]

        wl.ensure_fresh(api_get_fn=fake_api)
        assert called.get("yes")

    def test_failed_refresh_serves_stale_cache(self, tmp_path, monkeypatch):
        # A watchlist from yesterday beats no watchlist; the snapshot exposes age.
        monkeypatch.setattr(wl, "CACHE_FILE", str(tmp_path / "c.json"))
        from portfolio.file_utils import atomic_write_json

        old = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=30)
        ).isoformat()
        atomic_write_json(
            wl.CACHE_FILE,
            _cache(
                entries=[{"ob": "9", "name": "X", "currency": "USD"}], refreshed_at=old
            ),
        )

        def broken_api(path):
            raise OSError("session dead")

        out = wl.ensure_fresh(api_get_fn=broken_api)
        assert out["entries"][0]["ob"] == "9"


class TestSplitEntries:
    def test_book_instruments_become_pointers_not_reevaluations(self):
        from portfolio.swedbank.instruments import INSTRUMENTS

        nvda_ob = INSTRUMENTS["NVDA"].avanza_ob
        cache = _cache(
            entries=[
                {"ob": nvda_ob, "name": "NVIDIA", "ticker": "NVDA", "currency": "USD"},
                {
                    "ob": "999999",
                    "name": "Foreign",
                    "ticker": "FRGN",
                    "currency": "USD",
                },
            ]
        )
        overlap, fresh, quote_only = wl.split_entries(cache)
        assert overlap[0]["swedbank_key"] == "NVDA"
        assert [e["ob"] for e in fresh] == ["999999"]
        assert quote_only == []

    def test_quote_only_products_are_separated(self):
        ob = next(iter(wl.QUOTE_ONLY_OBS))
        cache = _cache(entries=[{"ob": ob, "name": "Fund", "currency": "SEK"}])
        overlap, fresh, quote_only = wl.split_entries(cache)
        assert not fresh and len(quote_only) == 1


class TestEvaluateWatchlist:
    def test_broken_instrument_gets_error_row_not_silence(self):
        cache = _cache(
            entries=[
                {"ob": "424242", "name": "Broken", "ticker": "BRK", "currency": "USD"}
            ]
        )

        def boom(inst, horizon="1d", **kw):
            raise RuntimeError("no OHLCV")

        results, pointers = wl.evaluate_watchlist(cache=cache, evaluate_fn=boom)
        (row,) = results.values()
        assert "no OHLCV" in row["error"] and row["watch_only"]

    def test_leveraged_product_evaluates_on_underlying(self):
        from portfolio.swedbank import signals as sigmod

        ob = "2224675"  # BEAR MSTR X5 -> MSTR
        cache = _cache(
            entries=[
                {"ob": ob, "name": "BEAR MSTR X5 SG5", "ticker": "", "currency": "SEK"}
            ]
        )
        seen = {}

        def spy(inst, horizon="1d", **kw):
            seen["underlying"] = sigmod.UNDERLYING.get(inst.key)
            return {"key": inst.key, "action": "HOLD"}

        results, _ = wl.evaluate_watchlist(cache=cache, evaluate_fn=spy)
        assert seen["underlying"] == "MSTR"
        # the temporary mapping must not leak into the shared module dict
        assert not any("BEAR" in k for k in sigmod.UNDERLYING)

    def test_snapshot_separates_tracked_from_book_pointers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wl, "SNAPSHOT_FILE", str(tmp_path / "s.json"))
        monkeypatch.setattr(wl, "SIGNAL_LOG", str(tmp_path / "l.jsonl"))
        snap = wl.write_snapshot(
            {"FRGN": {"key": "FRGN", "action": "HOLD", "confidence": 0.0}},
            {"NVDA": {"ob": "4478", "see": "swedbank", "swedbank_key": "NVDA"}},
            _cache(),
        )
        assert "FRGN" in snap["tracked"]
        assert snap["in_swedbank_book"]["NVDA"]["see"] == "swedbank"
