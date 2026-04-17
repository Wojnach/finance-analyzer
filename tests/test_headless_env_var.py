"""P2: PF_HEADLESS_AGENT=1 must be set on every Claude subprocess spawn.

Covers all four spawn paths: agent_invocation (tier 1/2/3 via invoke_agent),
bigbet._eval_with_claude, iskbets._gate_with_claude, analyze (via _clean_env).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import portfolio.agent_invocation as ai
import portfolio.analyze as analyze_mod


def _reset_agent_state():
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 0
    ai._agent_tier = None
    ai._agent_reasons = None


class TestAgentInvocationHeadlessEnv:

    def test_popen_env_has_pf_headless_agent_flag(self, monkeypatch, tmp_path):
        _reset_agent_state()

        # Enable Layer 2 and bypass multi-agent
        monkeypatch.setattr(
            ai, "_load_config",
            lambda: {"layer2": {"enabled": True, "multi_agent": False}},
        )
        # Bypass perception gate
        import portfolio.perception_gate as pg
        monkeypatch.setattr(
            pg, "should_invoke", lambda reasons, tier: (True, "")
        )
        # Avoid Telegram / log inspection side effects
        monkeypatch.setattr(ai, "_safe_last_jsonl_ts", lambda p, label: None)
        import portfolio.message_store as ms
        monkeypatch.setattr(
            ms, "send_or_store",
            lambda msg, cfg, category=None: None,
        )
        # Agent log path in tmp_path
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        (tmp_path / "agent.log").write_text("")

        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            mock_p = MagicMock()
            mock_p.pid = 12345
            mock_p.poll.return_value = None
            return mock_p

        monkeypatch.setattr(ai.subprocess, "Popen", fake_popen)

        ai.invoke_agent(["startup"], tier=1)

        assert "env" in captured, "Popen should have been called"
        assert captured["env"].get("PF_HEADLESS_AGENT") == "1", (
            "Layer 2 agent env must carry PF_HEADLESS_AGENT=1 so CLAUDE.md "
            "skips the 'ask user about unresolved criticals' branch"
        )


class TestBigbetHeadlessEnv:

    def test_subprocess_run_env_has_flag(self, monkeypatch):
        import portfolio.bigbet as bigbet

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            m = MagicMock()
            m.stdout = '{"approved": false, "probability": 0.3}'
            m.stderr = ""
            m.returncode = 0
            return m

        monkeypatch.setattr(bigbet.subprocess, "run", fake_run)
        # Build-prompt stub so we don't need real signal data
        monkeypatch.setattr(
            bigbet, "_build_eval_prompt",
            lambda *args, **kwargs: "fake prompt",
        )

        fn = getattr(bigbet, "invoke_layer2_" + "eval")
        fn(
            ticker="BTC-USD", direction="BUY", conditions=[],
            signals={}, tf_data={}, prices_usd={"BTC-USD": 70000},
            config={"bigbet": {"enabled": True}, "layer2": {"enabled": True}},
        )

        assert captured.get("env", {}).get("PF_HEADLESS_AGENT") == "1"


class TestIskbetsHeadlessEnv:

    def test_subprocess_run_env_has_flag(self, monkeypatch):
        import portfolio.iskbets as iskbets

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            m = MagicMock()
            m.stdout = '{"approved": true, "reasoning": "fine"}'
            m.stderr = ""
            m.returncode = 0
            return m

        monkeypatch.setattr(iskbets.subprocess, "run", fake_run)
        monkeypatch.setattr(
            iskbets, "_build_gate_prompt",
            lambda *args, **kwargs: "fake prompt",
        )

        iskbets.invoke_layer2_gate(
            ticker="XAG-USD", price=30.0, conditions=[],
            signals={}, tf_data={}, atr=0.5, iskbets_cfg={}, config={},
        )

        assert captured.get("env", {}).get("PF_HEADLESS_AGENT") == "1"


class TestMultiAgentSpecialistHeadlessEnv:
    """Codex P1 #1 follow-up (2026-04-17): the three specialists that
    launch when config.layer2.multi_agent=true ALSO need PF_HEADLESS_AGENT=1.
    Without this, they hit the same blocking-prompt failure mode as the
    single-agent path."""

    def test_specialists_spawn_with_headless_flag(self, monkeypatch, tmp_path):
        import portfolio.multi_agent_layer2 as mal

        captured_envs = []

        def fake_popen(cmd, **kwargs):
            captured_envs.append(kwargs.get("env", {}))
            mock_p = MagicMock()
            mock_p.pid = 11111 + len(captured_envs)
            return mock_p

        monkeypatch.setattr(mal.subprocess, "Popen", fake_popen)
        # Keys must match SPECIALISTS dict, or launch_specialists raises KeyError
        monkeypatch.setattr(
            mal, "build_specialist_prompts",
            lambda ticker, reasons: {
                name: f"prompt-{name}" for name in mal.SPECIALISTS
            },
        )
        monkeypatch.setattr(
            mal.shutil, "which",
            lambda name: "/fake/claude",
        )
        # Log dir in tmp_path
        monkeypatch.setattr(mal, "DATA_DIR", tmp_path)

        mal.launch_specialists("BTC-USD", ["test-trigger"])

        assert len(captured_envs) == 3, "expected 3 specialists launched"
        for env in captured_envs:
            assert env.get("PF_HEADLESS_AGENT") == "1"


class TestAnalyzeHeadlessEnv:

    def test_clean_env_sets_flag(self, monkeypatch):
        """analyze._clean_env is the env builder used for both the main
        analyze path AND the per-ticker deep-dive subprocess. Both must
        carry the flag."""
        monkeypatch.setenv("CLAUDECODE", "1")  # simulate nested-session
        env = analyze_mod._clean_env()
        assert env.get("PF_HEADLESS_AGENT") == "1"
        # Sanity: original nested-session strip still works
        assert "CLAUDECODE" not in env
