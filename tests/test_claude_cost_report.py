"""Regression tests for scripts/claude_cost_report.summarise."""

import json

from scripts import claude_cost_report as ccr


class TestSummariseTierKeys:
    def test_mixed_tier_types_serialize_with_sort_keys(self):
        """Rows with int tiers alongside '?' unknowns crashed Flask's
        sort_keys jsonify (2026-07-19, /api/claude_cost 500). Tier keys
        must always be strings."""
        bundle = {
            "cutoff": 0,
            "gate_rows": [],
            "layer2_rows": [
                {"tier": 1, "status": "ok", "duration_s": 2.0},
                {"tier": 2, "status": "ok", "duration_s": 1.0},
                {"tier": "?", "status": "error", "duration_s": None},
            ],
        }
        summary = ccr.summarise(bundle)
        json.dumps(summary, sort_keys=True)
        assert sorted(summary["layer2_by_tier"]) == ["1", "2", "?"]
        assert all(isinstance(k, str) for k in summary["layer2_by_tier"])
