"""Master pause switch for ALL local model inference.

2026-07-02: added at user request — one bool to pause every local model
going forward. Mirrors claude_gate.CLAUDE_ENABLED in spirit, but
runtime-togglable without a code edit or loop restart.

Covers every local-model backend across both loops (choke points):
- ``llama_server.query_llama_server`` / ``query_llama_server_batch`` —
  all shared GGUF models (ministral3, qwen3, fingpt, shadow LLMs,
  ministral8_lora for metals, prewarmer) and prevents the server from
  ever starting.
- ``llama_server.model_load_safe`` — returns False when paused, which
  converts the cold-start subprocess fallbacks in ministral_signal /
  qwen3_signal / shadow-LLM signals into clean abstains.
- ``bert_sentiment.predict`` — in-process CPU/GPU BERT sentiment
  (CryptoBERT / Trading-Hero-LLM / FinBERT); returns zero-confidence
  neutral placeholders so sentiment.py does NOT fall back to the legacy
  subprocess path.
- ``forecast_signal.forecast_chronos`` — Chronos in main loop,
  data/chronos_server.py (imports it), and the metals subprocess fallback.
- ``data/metals_llm._run_ministral_metals`` / ``_run_chronos_metals`` —
  seals the metals-loop-only legacy paths (stdin ministral server,
  one-shot subprocesses) that never touch model_load_safe.

Two independent switches — paused when EITHER trips:

1. Flag file ``data/local_llm.disabled`` exists::

       touch data/local_llm.disabled    # pause
       rm data/local_llm.disabled       # resume

2. config.json ``{"local_llm": {"enabled": false}}`` (key absent → enabled).

Both are re-checked live (flag: os.path.exists per call; config: mtime
cache inside api_utils.load_config), so toggling takes effect mid-cycle
in the running loops — same ergonomics as data/fix_agent.disabled.

Fail-open: if config.json is unreadable (e.g. worktrees don't replicate
the config symlink) only the flag file decides. A tooling failure must
never silently change trading behavior.
"""

import logging
import pathlib

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DISABLE_FLAG = _REPO_ROOT / "data" / "local_llm.disabled"

# Last state seen, for transition logging only (None = never checked).
# Benign race under threads — worst case is a duplicate log line.
_last_state = None


def local_llm_enabled() -> bool:
    """Return True if local model inference is allowed right now.

    Checked at every choke point listed in the module docstring. Cheap:
    one os.stat for the flag file + the mtime-cached config read.
    """
    global _last_state

    enabled = True
    reason = ""
    if DISABLE_FLAG.exists():
        enabled = False
        reason = f"flag file {DISABLE_FLAG.name} present"
    else:
        try:
            # Lazy import: keeps this module import-safe from data/ scripts
            # and avoids any import-cycle risk with portfolio internals.
            from portfolio.api_utils import load_config

            cfg = load_config()
            if not cfg.get("local_llm", {}).get("enabled", True):
                enabled = False
                reason = "config local_llm.enabled=false"
        except Exception:
            pass  # config unreadable → flag file alone decides (fail-open)

    if enabled != _last_state:
        if enabled:
            logger.info("local LLM gate: ENABLED — local model inference resumed")
        else:
            logger.info(
                "local LLM gate: PAUSED (%s) — all local model inference skipped",
                reason,
            )
        _last_state = enabled
    return enabled
