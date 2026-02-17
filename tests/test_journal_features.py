import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from portfolio.journal import build_context
from tests.test_journal import make_entry


class TestReflection:
    def test_reflection_rendered_when_present(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=0.5,
                now=now,
                reflection="Previous bullish thesis on BTC was correct, price up 3%",
            )
        ]
        md = build_context(entries)
        assert "Previous bullish thesis" in md

    def test_no_reflection_when_absent(self):
        now = datetime.now(timezone.utc)
        md = build_context([make_entry(age_hours=0.5, now=now)])
        assert "Reflection" not in md

    def test_reflection_in_full_context_output(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=1.0,
                now=now,
                reflection="thesis was wrong",
                patient_action="BUY BTC-USD",
            ),
            make_entry(age_hours=0.5, now=now),
        ]
        assert "thesis was wrong" in build_context(entries)


class TestConviction:
    def test_conviction_displayed_as_percentage(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=0.5,
                now=now,
                tickers={
                    "BTC-USD": {
                        "outlook": "bullish",
                        "thesis": "breakout",
                        "conviction": 0.7,
                        "levels": [],
                    }
                },
            )
        ]
        assert "[70%]" in build_context(entries)

    def test_missing_conviction_no_display(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=0.5,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []}
                },
            )
        ]
        md = build_context(entries)
        assert "%" not in md

    def test_zero_conviction_no_display(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=0.5,
                now=now,
                tickers={
                    "BTC-USD": {
                        "outlook": "bullish",
                        "thesis": "up",
                        "conviction": 0.0,
                        "levels": [],
                    }
                },
            )
        ]
        assert "[0%]" not in build_context(entries)


class TestContinuationChains:
    def test_simple_chain_of_2(self):
        now = datetime.now(timezone.utc)
        e1 = make_entry(
            age_hours=2,
            now=now,
            tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
            },
        )
        e2 = make_entry(
            age_hours=1,
            now=now,
            continues=e1["ts"],
            tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "still up", "levels": []},
            },
        )
        assert "Thesis Chains" in build_context([e1, e2])

    def test_chain_of_3(self):
        now = datetime.now(timezone.utc)
        e1 = make_entry(age_hours=3, now=now)
        e2 = make_entry(age_hours=2, now=now, continues=e1["ts"])
        e3 = make_entry(age_hours=1, now=now, continues=e2["ts"])
        md = build_context([e1, e2, e3])
        assert "Thesis Chains" in md
        assert "→" in md

    def test_no_continues_no_chains(self):
        now = datetime.now(timezone.utc)
        assert "Thesis Chains" not in build_context(
            [make_entry(age_hours=0.5, now=now)]
        )

    def test_dangling_reference_handled(self):
        now = datetime.now(timezone.utc)
        e = make_entry(age_hours=0.5, now=now, continues="2026-01-01T00:00:00+00:00")
        assert "Thesis Chains" not in build_context([e])


class TestPortfolioPnl:
    def test_reads_patient_portfolio(self, tmp_path):
        from portfolio.journal import _load_portfolio_pnl

        pf = tmp_path / "portfolio_state.json"
        pf.write_text(
            json.dumps(
                {
                    "cash_sek": 425000.0,
                    "initial_value_sek": 500000.0,
                    "total_fees_sek": 150.0,
                    "holdings": {"BTC-USD": {"shares": 0.5}},
                    "transactions": [{"action": "BUY"}, {"action": "SELL"}],
                }
            )
        )
        with patch("portfolio.journal.PORTFOLIO_FILE", pf):
            data = _load_portfolio_pnl()
        assert data["patient"]["cash_sek"] == 425000.0
        assert data["patient"]["trades"] == 2

    def test_reads_bold_portfolio(self, tmp_path):
        from portfolio.journal import _load_portfolio_pnl

        bf = tmp_path / "portfolio_state_bold.json"
        bf.write_text(
            json.dumps(
                {
                    "cash_sek": 350000.0,
                    "initial_value_sek": 500000.0,
                    "total_fees_sek": 75.0,
                    "holdings": {},
                    "transactions": [{"action": "BUY"}],
                }
            )
        )
        with patch("portfolio.journal.BOLD_FILE", bf):
            data = _load_portfolio_pnl()
        assert data["bold"]["cash_sek"] == 350000.0

    def test_missing_file_returns_empty(self, tmp_path):
        from portfolio.journal import _load_portfolio_pnl

        missing = tmp_path / "nope.json"
        with patch("portfolio.journal.PORTFOLIO_FILE", missing), patch(
            "portfolio.journal.BOLD_FILE", missing
        ):
            assert _load_portfolio_pnl() == {}

    def test_malformed_json_handled(self, tmp_path):
        from portfolio.journal import _load_portfolio_pnl

        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with patch("portfolio.journal.PORTFOLIO_FILE", bad), patch(
            "portfolio.journal.BOLD_FILE", bad
        ):
            assert _load_portfolio_pnl() == {}

    def test_portfolio_section_in_context(self):
        now = datetime.now(timezone.utc)
        pdata = {
            "patient": {
                "cash_sek": 425000.0,
                "initial_value_sek": 500000.0,
                "total_fees_sek": 150.0,
                "trades": 2,
                "holdings": ["BTC-USD"],
            },
            "bold": {
                "cash_sek": 350000.0,
                "initial_value_sek": 500000.0,
                "total_fees_sek": 75.0,
                "trades": 1,
                "holdings": [],
            },
        }
        md = build_context([make_entry(age_hours=0.5, now=now)], portfolio_data=pdata)
        assert "Portfolio Snapshot" in md
        assert "425,000" in md


class TestLayeredCompression:
    def test_recent_entry_gets_full_detail(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=0.5,
                now=now,
                patient_action="BUY BTC-USD",
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []}
                },
            )
        ]
        md = build_context(entries, now=now)
        assert "trigger:" in md
        assert "BUY BTC-USD" in md

    def test_medium_entry_gets_compact(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=3,
                now=now,
                patient_action="BUY BTC-USD",
                tickers={
                    "BTC-USD": {
                        "outlook": "bullish",
                        "thesis": "up",
                        "conviction": 0.7,
                        "levels": [],
                    }
                },
            )
        ]
        md = build_context(entries, now=now)
        assert "trigger:" not in md

    def test_old_entry_gets_oneline(self):
        now = datetime.now(timezone.utc)
        entries = [make_entry(age_hours=6, now=now, patient_action="BUY BTC-USD")]
        md = build_context(entries, now=now)
        lines = [l for l in md.split("\n") if "BUY" in l]
        for line in lines:
            assert len(line) < 120

    def test_recent_hold_not_compressed(self):
        now = datetime.now(timezone.utc)
        md = build_context([make_entry(age_hours=0.5, now=now)], now=now)
        assert "trigger:" in md

    def test_old_holds_compressed(self):
        now = datetime.now(timezone.utc)
        entries = [make_entry(age_hours=h, now=now) for h in [5, 4.5, 4]]
        assert "HOLD" in build_context(entries, now=now)

    def test_mixed_ages_in_single_context(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=6, now=now, patient_action="BUY BTC-USD"),
            make_entry(age_hours=3, now=now, patient_action="SELL BTC-USD"),
            make_entry(
                age_hours=0.5,
                now=now,
                patient_action="BUY ETH-USD",
                tickers={
                    "ETH-USD": {
                        "outlook": "bullish",
                        "thesis": "recovery",
                        "levels": [],
                    }
                },
            ),
        ]
        md = build_context(entries, now=now)
        assert "trigger:" in md
        assert "ETH-USD" in md


class TestWarnings:
    def test_thesis_contradiction_detected(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=3,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
                },
                prices={"BTC-USD": 70000.0},
            ),
            make_entry(
                age_hours=2,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
                },
                prices={"BTC-USD": 69500.0},
            ),
            make_entry(
                age_hours=1,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
                },
                prices={"BTC-USD": 69000.0},
            ),
        ]
        warnings = _detect_warnings(entries)
        assert any("contradict" in w.lower() for w in warnings)

    def test_no_warning_when_thesis_correct(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=h,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
                },
                prices={"BTC-USD": 72000.0 - h * 500},
            )
            for h in [3, 2, 1]
        ]
        contradiction_warnings = [
            w for w in _detect_warnings(entries) if "contradict" in w.lower()
        ]
        assert len(contradiction_warnings) == 0

    def test_small_price_move_not_contradiction(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(
                age_hours=h,
                now=now,
                tickers={
                    "BTC-USD": {"outlook": "bullish", "thesis": "up", "levels": []},
                },
                prices={"BTC-USD": 70000.0 - h * 30},
            )
            for h in [3, 2, 1]
        ]
        contradiction_warnings = [
            w for w in _detect_warnings(entries) if "contradict" in w.lower()
        ]
        assert len(contradiction_warnings) == 0

    def test_whipsaw_detected(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=3, now=now, patient_action="BUY BTC-USD"),
            make_entry(age_hours=2, now=now),
            make_entry(age_hours=1, now=now, patient_action="SELL BTC-USD"),
        ]
        assert any("whipsaw" in w.lower() for w in _detect_warnings(entries))

    def test_regime_stuck_detected(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=h, now=now, regime="range-bound") for h in [9, 5, 1]
        ]
        assert any("regime" in w.lower() for w in _detect_warnings(entries))

    def test_regime_not_stuck_under_8h(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=h, now=now, regime="range-bound") for h in [6, 3, 1]
        ]
        assert not any("regime" in w.lower() for w in _detect_warnings(entries))

    def test_churning_detected(self):
        from portfolio.journal import _detect_warnings

        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=3, now=now, patient_action="BUY BTC-USD"),
            make_entry(age_hours=2, now=now, patient_action="SELL BTC-USD"),
            make_entry(age_hours=1, now=now, patient_action="BUY BTC-USD"),
        ]
        assert any("churn" in w.lower() for w in _detect_warnings(entries))

    def test_empty_entries_no_warnings(self):
        from portfolio.journal import _detect_warnings

        assert _detect_warnings([]) == []
