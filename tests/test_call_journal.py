"""Call journal: append-only, mechanically scored, no self-flattering verdicts."""

from __future__ import annotations

import datetime

import pytest

from portfolio import call_journal as cj


@pytest.fixture
def jpath(tmp_path):
    return str(tmp_path / "calls.jsonl")


def _log(jpath, **kw):
    base = dict(
        instrument="ALAB",
        call="SELL",
        thesis="t",
        price_at_call=100.0,
        horizon_days=1,
        expected_move_pct=-4.0,
        path=jpath,
    )
    base.update(kw)
    return cj.log_call(**base)


class TestLogCall:
    def test_rejects_unknown_call(self, jpath):
        with pytest.raises(ValueError):
            _log(jpath, call="MOON")

    def test_sets_resolve_after_from_horizon(self, jpath):
        e = _log(jpath, horizon_days=30)
        t0 = datetime.datetime.fromisoformat(e["ts"])
        t1 = datetime.datetime.fromisoformat(e["resolve_after"])
        assert (t1 - t0).days == 30

    def test_records_the_evidence(self, jpath):
        e = _log(jpath, basis=["PT -8.4%", "3/8 up"])
        assert e["basis"] == ["PT -8.4%", "3/8 up"]


class TestResolution:
    def test_bearish_call_correct_when_price_falls(self, jpath):
        e = _log(jpath)
        r = cj.resolve_call(e, 96.0, path=jpath)
        assert r["direction_correct"] is True
        assert r["realised_move_pct"] == -4.0
        assert r["verdict"] == "correct"

    def test_bearish_call_wrong_when_price_rises(self, jpath):
        e = _log(jpath)
        r = cj.resolve_call(e, 112.0, path=jpath)
        assert r["direction_correct"] is False
        assert r["verdict"] == "wrong"

    def test_bullish_call_scored_the_other_way(self, jpath):
        e = _log(jpath, call="BUY", expected_move_pct=6.0)
        assert cj.resolve_call(e, 110.0, path=jpath)["verdict"] == "correct"
        e2 = _log(jpath, call="BUY", expected_move_pct=6.0)
        assert cj.resolve_call(e2, 90.0, path=jpath)["verdict"] == "wrong"

    def test_right_direction_wrong_size_is_not_banked_as_a_clean_win(self, jpath):
        # Called -4%, got -40%. Directionally right, but the magnitude claim was
        # badly wrong and must stay visible rather than counting as a hit.
        e = _log(jpath)
        r = cj.resolve_call(e, 60.0, path=jpath)
        assert r["direction_correct"] is True
        assert r["within_expected"] is False
        assert r["verdict"] == "right-direction-wrong-size"

    def test_avoid_is_scored_bearishly(self, jpath):
        e = _log(jpath, instrument="MSFT", call="AVOID", expected_move_pct=0.0)
        assert cj.resolve_call(e, 90.0, path=jpath)["direction_correct"] is True

    def test_resolution_does_not_mutate_the_call(self, jpath):
        e = _log(jpath)
        cj.resolve_call(e, 96.0, path=jpath)
        rows = cj.load_all(jpath)
        calls = [r for r in rows if r["kind"] == "call"]
        assert len(calls) == 1 and calls[0]["status"] == "open"
        assert sum(1 for r in rows if r["kind"] == "resolution") == 1


class TestOpenCalls:
    def test_resolved_calls_drop_out(self, jpath):
        e = _log(jpath)
        assert len(cj.open_calls(jpath)) == 1
        cj.resolve_call(e, 96.0, path=jpath)
        assert cj.open_calls(jpath) == []

    def test_due_flag_respects_horizon(self, jpath):
        past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)
        _log(jpath, horizon_days=1, ts=past)
        _log(jpath, instrument="ASML", call="BUY", horizon_days=30)
        rows = {c["instrument"]: c["due"] for c in cj.open_calls(jpath)}
        assert rows["ALAB"] is True and rows["ASML"] is False


class TestScorecard:
    def test_empty_journal(self, jpath):
        assert cj.scorecard(jpath) == {"n": 0}

    def test_hit_rate_and_per_instrument(self, jpath):
        cj.resolve_call(_log(jpath), 96.0, path=jpath)  # correct
        cj.resolve_call(_log(jpath), 110.0, path=jpath)  # wrong
        cj.resolve_call(
            _log(jpath, instrument="ASML", call="BUY", expected_move_pct=6.0),
            106.0,
            path=jpath,
        )  # correct
        sc = cj.scorecard(jpath)
        assert sc["n"] == 3
        assert sc["direction_hit_rate"] == pytest.approx(66.7, abs=0.1)
        assert sc["by_instrument"]["ALAB"]["hit_rate"] == 50.0
        assert sc["by_instrument"]["ASML"]["hit_rate"] == 100.0

    def test_missing_price_is_unscorable_not_a_win(self, jpath):
        e = _log(jpath, price_at_call=None)
        r = cj.resolve_call(e, 96.0, path=jpath)
        assert r["verdict"] == "unscorable"
        assert r["direction_correct"] is None
