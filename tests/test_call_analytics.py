"""Lifetime accuracy breakdowns: calibration, dimensions, amendment effect."""

from __future__ import annotations

import datetime

import pytest

from portfolio import call_analytics as ca
from portfolio import call_journal as cj


@pytest.fixture
def jpath(tmp_path):
    return str(tmp_path / "calls.jsonl")


def _call(jpath, **kw):
    base = dict(
        instrument="ALAB",
        call="SELL",
        thesis="t",
        price_at_call=100.0,
        horizon_days=1,
        expected_move_pct=-4.0,
        p_up=38,
        confidence="medium",
        path=jpath,
    )
    base.update(kw)
    return cj.log_call(**base)


class TestEvidenceTagging:
    def test_tags_recognised_from_free_text(self):
        tags = ca._tags_for(
            [
                "consensus PT $317.92 = -8.4% vs spot",
                "own earnings history: 3/8 up",
                "insiders sold $445M, zero buys",
                "RSI 79 and +42.9% run-in",
            ]
        )
        assert {"analyst_target", "own_history", "insider", "technical"} <= set(tags)

    def test_no_tags_when_nothing_matches(self):
        assert ca._tags_for(["a hunch"]) == []


class TestOverall:
    def test_empty_journal(self, jpath):
        assert ca.build(jpath)["n"] == 0

    def test_hit_rate_and_abs_error(self, jpath):
        cj.resolve_call(_call(jpath), 96.0, path=jpath)  # -4%, exactly expected
        cj.resolve_call(_call(jpath), 110.0, path=jpath)  # +10%, wrong direction
        o = ca.build(jpath)["overall"]
        assert o["n"] == 2
        assert o["direction_hit_rate"] == 50.0
        # errors: |−4 − (−4)| = 0 and |+10 − (−4)| = 14 -> mean 7.0
        assert o["mean_abs_error_pct"] == 7.0
        assert o["thin_sample"] is True


class TestCalibration:
    def test_confident_and_right_scores_well(self, jpath):
        # p_up=10 on a bearish call that fell: |0.1 - 0|^2 = 0.01
        cj.resolve_call(_call(jpath, p_up=10), 96.0, path=jpath)
        assert ca.build(jpath)["calibration"]["brier"] == pytest.approx(0.01)

    def test_confident_and_wrong_scores_badly(self, jpath):
        # p_up=10 but it rose: |0.1 - 1|^2 = 0.81, far worse than a coin flip
        cj.resolve_call(_call(jpath, p_up=10), 120.0, path=jpath)
        b = ca.build(jpath)["calibration"]
        assert b["brier"] == pytest.approx(0.81)
        assert b["brier"] > b["coin_flip"]

    def test_absent_p_up_is_excluded_not_assumed(self, jpath):
        cj.resolve_call(_call(jpath, p_up=None), 96.0, path=jpath)
        assert ca.build(jpath)["calibration"] is None


class TestDimensions:
    def test_splits_by_confidence_call_type_instrument_horizon(self, jpath):
        cj.resolve_call(_call(jpath, confidence="high"), 96.0, path=jpath)
        cj.resolve_call(
            _call(
                jpath, instrument="ASML", call="BUY", confidence="low", horizon_days=30
            ),
            110.0,
            path=jpath,
        )
        a = ca.build(jpath)
        assert a["by_confidence"]["high"]["hit_rate"] == 100.0
        assert a["by_confidence"]["low"]["hit_rate"] == 100.0
        assert a["by_call_type"]["SELL"]["n"] == 1
        assert a["by_instrument"]["ASML"]["n"] == 1
        assert a["by_horizon"]["8-90d"]["n"] == 1
        assert a["by_horizon"]["1-7d"]["n"] == 1

    def test_evidence_rows_count_every_citing_call(self, jpath):
        cj.resolve_call(
            _call(jpath, basis=["consensus PT -8%", "RSI 79"]), 96.0, path=jpath
        )
        cj.resolve_call(_call(jpath, basis=["consensus PT +20%"]), 110.0, path=jpath)
        ev = ca.build(jpath)["by_evidence_cited"]
        assert ev["analyst_target"]["n"] == 2
        assert ev["technical"]["n"] == 1

    def test_thin_sample_flag_clears_past_threshold(self, jpath):
        for _ in range(ca.MIN_N_FOR_SIGNAL):
            cj.resolve_call(_call(jpath), 96.0, path=jpath)
        assert ca.build(jpath)["by_instrument"]["ALAB"]["thin_sample"] is False


class TestAmendmentEffect:
    def _amend(self, jpath, call_id, revised):
        from portfolio.file_utils import atomic_append_jsonl

        atomic_append_jsonl(
            jpath,
            {
                "kind": "amendment",
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "amends_id": call_id,
                "instrument": "ALAB",
                "revised_expected_move_pct": revised,
            },
        )

    def test_amendment_that_moved_away_from_truth_is_recorded_as_worse(self, jpath):
        # Original -4% was closer to the realised -7% than the revised +1.2%.
        # This is exactly what happened on ALAB and must not be hidden.
        e = _call(jpath, expected_move_pct=-4.0)
        self._amend(jpath, e["id"], 1.2)
        cj.resolve_call(e, 93.0, path=jpath)
        eff = ca.build(jpath)["amendment_effect"]
        assert eff["n"] == 1
        assert eff["amendment_worsened"] == 1
        assert eff["amendment_improved"] == 0

    def test_amendment_toward_truth_is_recorded_as_better(self, jpath):
        e = _call(jpath, expected_move_pct=-4.0)
        self._amend(jpath, e["id"], -10.0)
        cj.resolve_call(e, 90.0, path=jpath)
        assert ca.build(jpath)["amendment_effect"]["amendment_improved"] == 1

    def test_revised_figures_supersede_the_original_for_scoring(self, jpath):
        e = _call(jpath, expected_move_pct=-4.0, p_up=38)
        self._amend(jpath, e["id"], 1.2)
        cj.resolve_call(e, 93.0, path=jpath)
        # abs error is measured against the revised +1.2, not the original -4
        assert ca.build(jpath)["overall"]["mean_abs_error_pct"] == pytest.approx(8.2)


class TestRetroactive:
    def test_retroactive_calls_are_counted_and_flagged(self, jpath):
        e = _call(jpath, retroactive=True)
        cj.resolve_call(e, 110.0, path=jpath)
        a = ca.build(jpath)
        assert a["retroactive_count"] == 1
        # and they still count against the hit rate
        assert a["overall"]["direction_hit_rate"] == 0.0


class TestReport:
    def test_report_renders_and_flags_bad_calibration(self, jpath):
        cj.resolve_call(_call(jpath, p_up=10), 120.0, path=jpath)
        text = ca.report(jpath)
        assert "LIFETIME CALL ACCURACY" in text
        assert "WORSE than coin flip" in text

    def test_report_handles_empty(self, jpath):
        assert "No resolved calls yet" in ca.report(jpath)
