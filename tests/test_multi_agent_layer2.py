"""Tests for multi-agent Layer 2 orchestration."""
from __future__ import annotations


class TestBuildSpecialistPrompts:
    def test_technical_prompt_includes_signals(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "technical" in prompts
        assert "signal" in prompts["technical"].lower()

    def test_risk_prompt_includes_portfolio(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "risk" in prompts
        assert "portfolio" in prompts["risk"].lower()

    def test_microstructure_prompt_includes_orderflow(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "microstructure" in prompts
        text = prompts["microstructure"].lower()
        assert "order" in text or "flow" in text or "depth" in text

    def test_prompts_include_trigger_reasons(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%", "RSI crossover"],
        )
        for name, prompt in prompts.items():
            assert "price move" in prompt

    def test_prompts_include_output_file(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["test"],
        )
        assert "_specialist_technical.md" in prompts["technical"]
        assert "_specialist_risk.md" in prompts["risk"]
        assert "_specialist_microstructure.md" in prompts["microstructure"]


class TestBuildSynthesisPrompt:
    def test_references_report_paths(self):
        from portfolio.multi_agent_layer2 import build_synthesis_prompt
        prompt = build_synthesis_prompt(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
            report_paths=["/tmp/tech.md", "/tmp/risk.md", "/tmp/micro.md"],
        )
        assert "/tmp/tech.md" in prompt
        assert "/tmp/risk.md" in prompt
        assert "/tmp/micro.md" in prompt

    def test_includes_playbook(self):
        from portfolio.multi_agent_layer2 import build_synthesis_prompt
        prompt = build_synthesis_prompt(
            ticker="XAG-USD",
            trigger_reasons=["test"],
        )
        assert "TRADING_PLAYBOOK" in prompt

    def test_includes_both_strategies(self):
        from portfolio.multi_agent_layer2 import build_synthesis_prompt
        prompt = build_synthesis_prompt(
            ticker="XAG-USD",
            trigger_reasons=["test"],
        )
        assert "Patient" in prompt and "Bold" in prompt


class TestMultiAgentConfig:
    def test_specialist_count(self):
        from portfolio.multi_agent_layer2 import SPECIALISTS
        assert len(SPECIALISTS) == 3

    def test_each_specialist_has_required_fields(self):
        from portfolio.multi_agent_layer2 import SPECIALISTS
        for name, spec in SPECIALISTS.items():
            assert "data_files" in spec, f"{name} missing data_files"
            assert "focus" in spec, f"{name} missing focus"
            assert "output_file" in spec, f"{name} missing output_file"
            assert "timeout" in spec, f"{name} missing timeout"
            assert "max_turns" in spec, f"{name} missing max_turns"

    def test_get_report_paths(self):
        from portfolio.multi_agent_layer2 import get_report_paths
        paths = get_report_paths()
        assert len(paths) == 3
        assert all("_specialist_" in p for p in paths)
