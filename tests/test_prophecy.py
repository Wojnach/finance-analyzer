"""Tests for portfolio.prophecy — Prophecy/Belief system."""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from pathlib import Path
from io import StringIO

import portfolio.prophecy as prophecy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_prophecy_file(tmp_path, monkeypatch):
    """Redirect PROPHECY_FILE to a temp dir for test isolation."""
    test_file = tmp_path / "prophecy.json"
    monkeypatch.setattr(prophecy, "PROPHECY_FILE", test_file)
    return test_file


def _make_belief(**overrides):
    """Helper to build a belief dict with sensible defaults."""
    base = {
        "id": "test_belief",
        "ticker": "XAG-USD",
        "thesis": "Test thesis",
        "direction": "bullish",
        "conviction": 0.7,
        "target_price": 100.0,
        "entry_price": 30.0,
        "status": "active",
        "supporting_evidence": ["evidence 1"],
        "opposing_evidence": ["counter 1"],
        "checkpoints": [],
        "tags": ["metals"],
    }
    base.update(overrides)
    return base


def _make_checkpoint(**overrides):
    """Helper to build a checkpoint dict."""
    base = {
        "id": "cp_test",
        "condition": "XAG breaks $35",
        "target_value": 35.0,
        "comparison": "above",
        "status": "pending",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_belief_template_has_all_keys(self):
        expected = {
            "id", "ticker", "thesis", "direction", "conviction",
            "target_price", "target_timeframe", "entry_price",
            "created_at", "updated_at", "status",
            "supporting_evidence", "opposing_evidence",
            "checkpoints", "tags", "notes",
        }
        assert set(prophecy.BELIEF_TEMPLATE.keys()) == expected

    def test_checkpoint_template_has_all_keys(self):
        expected = {
            "id", "condition", "target_value", "comparison",
            "deadline", "status", "triggered_at", "created_at",
        }
        assert set(prophecy.CHECKPOINT_TEMPLATE.keys()) == expected

    def test_belief_template_defaults(self):
        t = prophecy.BELIEF_TEMPLATE
        assert t["direction"] == "neutral"
        assert t["conviction"] == 0.5
        assert t["status"] == "active"
        assert t["checkpoints"] == []

    def test_checkpoint_template_defaults(self):
        t = prophecy.CHECKPOINT_TEMPLATE
        assert t["comparison"] == "above"
        assert t["status"] == "pending"
        assert t["triggered_at"] is None


# ---------------------------------------------------------------------------
# load_beliefs
# ---------------------------------------------------------------------------

class TestLoadBeliefs:
    def test_load_missing_file(self):
        result = prophecy.load_beliefs()
        assert result["beliefs"] == []
        assert result["metadata"]["version"] == 1
        assert result["metadata"]["last_review"] is None

    def test_load_valid_file(self, _isolate_prophecy_file):
        data = {
            "beliefs": [_make_belief()],
            "metadata": {"version": 1, "last_review": "2026-01-01T00:00:00+00:00"},
        }
        _isolate_prophecy_file.write_text(json.dumps(data), encoding="utf-8")
        result = prophecy.load_beliefs()
        assert len(result["beliefs"]) == 1
        assert result["beliefs"][0]["id"] == "test_belief"

    def test_load_legacy_list_format(self, _isolate_prophecy_file):
        """Legacy format: just a list of beliefs (no metadata wrapper)."""
        beliefs = [_make_belief()]
        _isolate_prophecy_file.write_text(json.dumps(beliefs), encoding="utf-8")
        result = prophecy.load_beliefs()
        assert len(result["beliefs"]) == 1
        assert result["metadata"]["version"] == 1

    def test_load_invalid_json(self, _isolate_prophecy_file):
        _isolate_prophecy_file.write_text("not valid json {{{", encoding="utf-8")
        result = prophecy.load_beliefs()
        assert result["beliefs"] == []

    def test_load_empty_beliefs(self, _isolate_prophecy_file):
        data = {"beliefs": [], "metadata": {"version": 1, "last_review": None}}
        _isolate_prophecy_file.write_text(json.dumps(data), encoding="utf-8")
        result = prophecy.load_beliefs()
        assert result["beliefs"] == []


# ---------------------------------------------------------------------------
# save_beliefs
# ---------------------------------------------------------------------------

class TestSaveBeliefs:
    def test_save_writes_json(self, _isolate_prophecy_file):
        data = {"beliefs": [_make_belief()], "metadata": {"version": 1, "last_review": None}}
        prophecy.save_beliefs(data)
        saved = json.loads(_isolate_prophecy_file.read_text(encoding="utf-8"))
        assert len(saved["beliefs"]) == 1
        assert saved["metadata"]["last_review"] is not None

    def test_save_updates_last_review(self, _isolate_prophecy_file):
        data = {"beliefs": [], "metadata": {"version": 1, "last_review": None}}
        prophecy.save_beliefs(data)
        saved = json.loads(_isolate_prophecy_file.read_text(encoding="utf-8"))
        # Should be a recent ISO timestamp
        dt = datetime.fromisoformat(saved["metadata"]["last_review"])
        assert (datetime.now(timezone.utc) - dt).total_seconds() < 10


# ---------------------------------------------------------------------------
# add_belief
# ---------------------------------------------------------------------------

class TestAddBelief:
    def test_add_basic(self):
        result = prophecy.add_belief(_make_belief())
        assert result["id"] == "test_belief"
        assert result["created_at"] != ""
        assert result["updated_at"] != ""
        # Verify persisted
        data = prophecy.load_beliefs()
        assert len(data["beliefs"]) == 1

    def test_add_fills_defaults(self):
        result = prophecy.add_belief({"id": "minimal", "ticker": "BTC-USD"})
        assert result["direction"] == "neutral"
        assert result["conviction"] == 0.5
        assert result["status"] == "active"

    def test_add_duplicate_id_raises(self):
        prophecy.add_belief(_make_belief(id="dup"))
        with pytest.raises(ValueError, match="already exists"):
            prophecy.add_belief(_make_belief(id="dup"))

    def test_add_preserves_existing(self):
        prophecy.add_belief(_make_belief(id="first"))
        prophecy.add_belief(_make_belief(id="second"))
        data = prophecy.load_beliefs()
        assert len(data["beliefs"]) == 2

    def test_add_respects_explicit_timestamps(self):
        ts = "2026-01-15T12:00:00+00:00"
        result = prophecy.add_belief(_make_belief(created_at=ts, updated_at=ts))
        assert result["created_at"] == ts
        assert result["updated_at"] == ts


# ---------------------------------------------------------------------------
# update_belief
# ---------------------------------------------------------------------------

class TestUpdateBelief:
    def test_update_existing(self):
        prophecy.add_belief(_make_belief(id="upd1"))
        result = prophecy.update_belief("upd1", {"conviction": 0.9, "thesis": "Updated"})
        assert result is not None
        assert result["conviction"] == 0.9
        assert result["thesis"] == "Updated"

    def test_update_sets_updated_at(self):
        prophecy.add_belief(_make_belief(id="upd2", updated_at="2026-01-01T00:00:00+00:00"))
        result = prophecy.update_belief("upd2", {"conviction": 0.3})
        assert result["updated_at"] != "2026-01-01T00:00:00+00:00"

    def test_update_missing_returns_none(self):
        result = prophecy.update_belief("nonexistent", {"conviction": 0.9})
        assert result is None

    def test_update_persists(self):
        prophecy.add_belief(_make_belief(id="upd3"))
        prophecy.update_belief("upd3", {"direction": "bearish"})
        belief = prophecy.get_belief("upd3")
        assert belief["direction"] == "bearish"


# ---------------------------------------------------------------------------
# remove_belief
# ---------------------------------------------------------------------------

class TestRemoveBelief:
    def test_remove_existing(self):
        prophecy.add_belief(_make_belief(id="rem1"))
        assert prophecy.remove_belief("rem1") is True
        assert prophecy.get_belief("rem1") is None

    def test_remove_missing(self):
        assert prophecy.remove_belief("nonexistent") is False

    def test_remove_preserves_others(self):
        prophecy.add_belief(_make_belief(id="keep"))
        prophecy.add_belief(_make_belief(id="drop"))
        prophecy.remove_belief("drop")
        data = prophecy.load_beliefs()
        assert len(data["beliefs"]) == 1
        assert data["beliefs"][0]["id"] == "keep"


# ---------------------------------------------------------------------------
# get_belief
# ---------------------------------------------------------------------------

class TestGetBelief:
    def test_get_existing(self):
        prophecy.add_belief(_make_belief(id="get1"))
        result = prophecy.get_belief("get1")
        assert result is not None
        assert result["id"] == "get1"

    def test_get_missing(self):
        assert prophecy.get_belief("missing") is None


# ---------------------------------------------------------------------------
# get_active_beliefs
# ---------------------------------------------------------------------------

class TestGetActiveBeliefs:
    def test_returns_active_only(self):
        prophecy.add_belief(_make_belief(id="active1", status="active"))
        prophecy.add_belief(_make_belief(id="paused1", status="paused"))
        prophecy.add_belief(_make_belief(id="active2", status="active"))
        result = prophecy.get_active_beliefs()
        assert len(result) == 2
        ids = {b["id"] for b in result}
        assert ids == {"active1", "active2"}

    def test_filter_by_ticker(self):
        prophecy.add_belief(_make_belief(id="xag1", ticker="XAG-USD"))
        prophecy.add_belief(_make_belief(id="btc1", ticker="BTC-USD"))
        result = prophecy.get_active_beliefs(ticker="XAG-USD")
        assert len(result) == 1
        assert result[0]["id"] == "xag1"

    def test_empty_when_none_active(self):
        prophecy.add_belief(_make_belief(id="paused", status="paused"))
        assert prophecy.get_active_beliefs() == []

    def test_empty_when_no_beliefs(self):
        assert prophecy.get_active_beliefs() == []


# ---------------------------------------------------------------------------
# add_checkpoint
# ---------------------------------------------------------------------------

class TestAddCheckpoint:
    def test_add_to_existing_belief(self):
        prophecy.add_belief(_make_belief(id="cp_test"))
        cp = prophecy.add_checkpoint("cp_test", _make_checkpoint())
        assert cp is not None
        assert cp["status"] == "pending"
        belief = prophecy.get_belief("cp_test")
        assert len(belief["checkpoints"]) == 1

    def test_add_generates_id(self):
        prophecy.add_belief(_make_belief(id="cp_id"))
        cp = prophecy.add_checkpoint("cp_id", {"condition": "test", "target_value": 50.0})
        assert cp["id"] == "cp_0"

    def test_add_to_missing_belief(self):
        result = prophecy.add_checkpoint("nonexistent", _make_checkpoint())
        assert result is None

    def test_add_fills_created_at(self):
        prophecy.add_belief(_make_belief(id="cp_ts"))
        cp = prophecy.add_checkpoint("cp_ts", _make_checkpoint())
        assert cp["created_at"] != ""

    def test_add_multiple_checkpoints(self):
        prophecy.add_belief(_make_belief(id="cp_multi"))
        prophecy.add_checkpoint("cp_multi", _make_checkpoint(id="cp_0"))
        prophecy.add_checkpoint("cp_multi", _make_checkpoint(id="cp_1"))
        belief = prophecy.get_belief("cp_multi")
        assert len(belief["checkpoints"]) == 2


# ---------------------------------------------------------------------------
# evaluate_checkpoints
# ---------------------------------------------------------------------------

class TestEvaluateCheckpoints:
    def test_above_triggered(self):
        prophecy.add_belief(_make_belief(id="eval_above", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_above", _make_checkpoint(
            id="cp_0", target_value=35.0, comparison="above"
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        assert len(triggered) == 1
        assert triggered[0]["belief_id"] == "eval_above"
        assert triggered[0]["price"] == 36.0

    def test_above_not_triggered(self):
        prophecy.add_belief(_make_belief(id="eval_below", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_below", _make_checkpoint(
            id="cp_0", target_value=35.0, comparison="above"
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 34.0})
        assert len(triggered) == 0

    def test_below_triggered(self):
        prophecy.add_belief(_make_belief(id="eval_b", ticker="BTC-USD"))
        prophecy.add_checkpoint("eval_b", _make_checkpoint(
            id="cp_0", target_value=60000.0, comparison="below"
        ))
        triggered = prophecy.evaluate_checkpoints({"BTC-USD": 59000.0})
        assert len(triggered) == 1

    def test_between_triggered(self):
        prophecy.add_belief(_make_belief(id="eval_between", ticker="ETH-USD"))
        prophecy.add_checkpoint("eval_between", _make_checkpoint(
            id="cp_0", target_value=[1800.0, 2200.0], comparison="between"
        ))
        triggered = prophecy.evaluate_checkpoints({"ETH-USD": 2000.0})
        assert len(triggered) == 1

    def test_between_not_triggered(self):
        prophecy.add_belief(_make_belief(id="eval_between2", ticker="ETH-USD"))
        prophecy.add_checkpoint("eval_between2", _make_checkpoint(
            id="cp_0", target_value=[1800.0, 2200.0], comparison="between"
        ))
        triggered = prophecy.evaluate_checkpoints({"ETH-USD": 2500.0})
        assert len(triggered) == 0

    def test_deadline_expiry(self):
        prophecy.add_belief(_make_belief(id="eval_exp", ticker="XAG-USD"))
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        prophecy.add_checkpoint("eval_exp", _make_checkpoint(
            id="cp_0", target_value=35.0, comparison="above", deadline=past
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        assert len(triggered) == 0
        # Verify status changed to expired
        belief = prophecy.get_belief("eval_exp")
        assert belief["checkpoints"][0]["status"] == "expired"

    def test_no_price_available(self):
        prophecy.add_belief(_make_belief(id="eval_noprice", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_noprice", _make_checkpoint(
            id="cp_0", target_value=35.0
        ))
        triggered = prophecy.evaluate_checkpoints({"BTC-USD": 70000.0})
        assert len(triggered) == 0

    def test_skip_non_active_beliefs(self):
        prophecy.add_belief(_make_belief(id="eval_paused", ticker="XAG-USD", status="paused"))
        prophecy.add_checkpoint("eval_paused", _make_checkpoint(
            id="cp_0", target_value=35.0
        ))
        # Force the checkpoint onto the paused belief manually
        data = prophecy.load_beliefs()
        data["beliefs"][0]["checkpoints"] = [_make_checkpoint(id="cp_0", target_value=35.0)]
        prophecy.save_beliefs(data)
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        assert len(triggered) == 0

    def test_skip_already_triggered_checkpoint(self):
        prophecy.add_belief(_make_belief(id="eval_already", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_already", _make_checkpoint(
            id="cp_0", target_value=35.0, status="triggered"
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        assert len(triggered) == 0

    def test_no_target_value_skipped(self):
        prophecy.add_belief(_make_belief(id="eval_notarget", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_notarget", _make_checkpoint(
            id="cp_0", target_value=None
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        assert len(triggered) == 0

    def test_triggered_persisted(self):
        prophecy.add_belief(_make_belief(id="eval_persist", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_persist", _make_checkpoint(
            id="cp_0", target_value=35.0, comparison="above"
        ))
        prophecy.evaluate_checkpoints({"XAG-USD": 36.0})
        belief = prophecy.get_belief("eval_persist")
        cp = belief["checkpoints"][0]
        assert cp["status"] == "triggered"
        assert cp["triggered_at"] is not None

    def test_exact_boundary_above(self):
        """Price exactly at target_value should trigger 'above'."""
        prophecy.add_belief(_make_belief(id="eval_exact", ticker="XAG-USD"))
        prophecy.add_checkpoint("eval_exact", _make_checkpoint(
            id="cp_0", target_value=35.0, comparison="above"
        ))
        triggered = prophecy.evaluate_checkpoints({"XAG-USD": 35.0})
        assert len(triggered) == 1

    def test_exact_boundary_below(self):
        """Price exactly at target_value should trigger 'below'."""
        prophecy.add_belief(_make_belief(id="eval_exact_b", ticker="BTC-USD"))
        prophecy.add_checkpoint("eval_exact_b", _make_checkpoint(
            id="cp_0", target_value=60000.0, comparison="below"
        ))
        triggered = prophecy.evaluate_checkpoints({"BTC-USD": 60000.0})
        assert len(triggered) == 1


# ---------------------------------------------------------------------------
# get_context_for_layer2
# ---------------------------------------------------------------------------

class TestGetContextForLayer2:
    def test_empty_state(self):
        result = prophecy.get_context_for_layer2()
        assert result["beliefs"] == []
        assert result["total_active"] == 0

    def test_active_beliefs_included(self):
        prophecy.add_belief(_make_belief(id="ctx1", ticker="XAG-USD"))
        prophecy.add_belief(_make_belief(id="ctx2", ticker="BTC-USD"))
        result = prophecy.get_context_for_layer2()
        assert result["total_active"] == 2
        assert len(result["beliefs"]) == 2

    def test_progress_calculation(self):
        prophecy.add_belief(_make_belief(
            id="prog1", ticker="XAG-USD",
            entry_price=30.0, target_price=120.0,
        ))
        # Price at 57 -> (57 - 30) / (120 - 30) * 100 = 30%
        result = prophecy.get_context_for_layer2({"XAG-USD": 57.0})
        belief = result["beliefs"][0]
        assert belief["progress_pct"] == 30.0
        assert belief["current_price"] == 57.0

    def test_progress_no_prices(self):
        prophecy.add_belief(_make_belief(
            id="prog2", ticker="XAG-USD",
            entry_price=30.0, target_price=120.0,
        ))
        result = prophecy.get_context_for_layer2()
        belief = result["beliefs"][0]
        assert "progress_pct" not in belief

    def test_progress_no_entry_price(self):
        prophecy.add_belief(_make_belief(
            id="prog3", ticker="XAG-USD",
            entry_price=None, target_price=120.0,
        ))
        result = prophecy.get_context_for_layer2({"XAG-USD": 57.0})
        belief = result["beliefs"][0]
        assert "progress_pct" not in belief

    def test_progress_no_target_price(self):
        prophecy.add_belief(_make_belief(
            id="prog4", ticker="BTC-USD",
            entry_price=67000.0, target_price=None,
        ))
        result = prophecy.get_context_for_layer2({"BTC-USD": 70000.0})
        belief = result["beliefs"][0]
        assert "progress_pct" not in belief

    def test_thesis_truncated(self):
        long_thesis = "A" * 500
        prophecy.add_belief(_make_belief(id="trunc", thesis=long_thesis))
        result = prophecy.get_context_for_layer2()
        assert len(result["beliefs"][0]["thesis"]) == 200

    def test_checkpoint_summary(self):
        prophecy.add_belief(_make_belief(id="cps"))
        prophecy.add_checkpoint("cps", _make_checkpoint(id="cp_0", status="triggered"))
        prophecy.add_checkpoint("cps", _make_checkpoint(id="cp_1", status="pending"))
        prophecy.add_checkpoint("cps", _make_checkpoint(id="cp_2", status="pending"))
        result = prophecy.get_context_for_layer2()
        assert result["beliefs"][0]["checkpoints_summary"] == "1/3 triggered"

    def test_excludes_inactive(self):
        prophecy.add_belief(_make_belief(id="act", status="active"))
        prophecy.add_belief(_make_belief(id="exp", status="expired"))
        result = prophecy.get_context_for_layer2()
        assert result["total_active"] == 1

    def test_progress_same_entry_target(self):
        """Edge case: entry_price == target_price should not divide by zero."""
        prophecy.add_belief(_make_belief(
            id="same", ticker="XAG-USD",
            entry_price=50.0, target_price=50.0,
        ))
        result = prophecy.get_context_for_layer2({"XAG-USD": 50.0})
        belief = result["beliefs"][0]
        assert "progress_pct" not in belief


# ---------------------------------------------------------------------------
# print_prophecy_review
# ---------------------------------------------------------------------------

class TestPrintProphecyReview:
    def test_empty_state(self, capsys):
        prophecy.print_prophecy_review()
        output = capsys.readouterr().out
        assert "No beliefs configured" in output

    def test_active_beliefs_shown(self, capsys):
        prophecy.add_belief(_make_belief(
            id="rev1", ticker="XAG-USD", direction="bullish",
            conviction=0.8, target_price=120.0, entry_price=31.50,
            thesis="Silver bull thesis",
        ))
        prophecy.add_checkpoint("rev1", _make_checkpoint(
            id="cp_0", condition="XAG breaks $35", status="pending"
        ))
        prophecy.print_prophecy_review()
        output = capsys.readouterr().out
        assert "Prophecy / Belief Review" in output
        assert "rev1" in output
        assert "XAG-USD" in output
        assert "Silver bull thesis" in output
        assert "$120" in output

    def test_inactive_count_shown(self, capsys):
        prophecy.add_belief(_make_belief(id="active1"))
        prophecy.add_belief(_make_belief(id="paused1", status="paused"))
        prophecy.add_belief(_make_belief(id="expired1", status="expired"))
        prophecy.print_prophecy_review()
        output = capsys.readouterr().out
        assert "2 inactive beliefs not shown" in output

    def test_bearish_direction_symbol(self, capsys):
        prophecy.add_belief(_make_belief(id="bear1", direction="bearish"))
        prophecy.print_prophecy_review()
        output = capsys.readouterr().out
        assert "v [bear1]" in output

    def test_neutral_direction_symbol(self, capsys):
        prophecy.add_belief(_make_belief(id="neut1", direction="neutral"))
        prophecy.print_prophecy_review()
        output = capsys.readouterr().out
        assert "> [neut1]" in output
