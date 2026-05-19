# Main Thread — Independent Adversarial Pass (Cross-Cutting)

**Scope:** issues that single-subsystem agents will systematically miss because they require seeing both sides of a boundary (signal-engine ↔ Layer 2, dashboard ↔ portfolio_state, main loop ↔ signal_history, etc.) or holding the whole repo's atomic-I/O discipline in mind.

**Approach:** sweep for anti-patterns across the entire codebase, not per-subsystem.

---

## P0 — Cross-cutting silent failure / data loss

### M01. `data/crypto_monitor.py:617`: raw `open(... "w") + json.dump` for crypto_analysis.json

```python
with open(analysis_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

This file is read by the same process's next analysis cycle and by Layer 2 when Crypto trigger fires. If killed mid-write (loop crash, force restart) the JSON is truncated; next reader gets ValueError. The CLAUDE.md atomic-I/O rule is **mandatory** and this violates it. Fix: `from portfolio.file_utils import atomic_write_json; atomic_write_json(analysis_path, data, indent=2, ensure_ascii=False)`.

### M02. `data/metals_history_fetch.py:206-207`: same raw open+json.dump for metals_history.json

Same pattern. `data/metals_history.json` is consumed by metals-loop and Layer 2. Replace with `atomic_write_json`.

### M03. `data/silver_monitor.py:782`: same raw open+json.dump

Verified via grep. Same fix.

### M04. Dashboard `_get_config()` in `dashboard/auth.py:60-78` allows access if `dashboard_token` is unset

```python
expected = _get_dashboard_token()
if expected is None:
    return f(*args, **kwargs)
```

If `config.json` becomes corrupt or is mid-edit when read, `_read_config_uncached()` returns `{}`, `_get_dashboard_token()` returns `None`, and EVERY route is open. The backwards-compat comment acknowledges this but the failure mode is silent. The TTL cache prevents permanent damage but a freshly-started dashboard process is exposed for one cache miss. Fix: log WARNING and refuse to serve unless a sentinel `disable_auth=true` is present in config.

## P1 — Cross-cutting real bugs

### M05. Singleton lock code duplicated between `portfolio/main.py` and `portfolio/process_lock.py`

`portfolio/main.py:95-109` reimplements fcntl + msvcrt singleton locking inline instead of using `portfolio/process_lock.py`. The inline version is missing the Linux/WSL `fcntl.LOCK_UN` release path (Agent 2 finding P0-1) — `process_lock.py` does this correctly. Migrating eliminates ~70 lines and the asymmetric-unlock P0.

### M06. Env-var pollution: Layer 2 subprocess inherits all parent env

`portfolio/agent_invocation.py:1038`: `agent_env = os.environ.copy()`. The subsequent `pop("CLAUDECODE", None)` and `pop("CLAUDE_CODE_ENTRYPOINT", None)` only sanitize known Claude Code markers. Any other secret-bearing env var (AWS credentials, GitHub tokens, anything the parent shell set) flows into the spawned Claude process and into its child Bash tool — exposed to whatever the LLM decides to print or log. The agent is a `claude -p` subprocess running with elevated trust; treating its env as untrusted is correct. Recommend env allowlist (start empty, opt-in each var) rather than the current blocklist.

Same finding applies at `portfolio/multi_agent_layer2.py:148`.

### M07. `datetime.utcnow()` deprecated usage (Python 3.12+)

`portfolio/escalation_gate.py:160`: `_dt.datetime.utcnow().isoformat() + "Z"` — `utcnow()` returns a NAIVE datetime; appending `"Z"` is a lie. A reader doing `datetime.fromisoformat()` gets a naive datetime that doesn't compare correctly against tz-aware timestamps elsewhere. Cross-cutting: anywhere reading the escalation_gate logs hits this inconsistency. Fix: `datetime.now(UTC).isoformat()`.

### M08. signal_engine and signal_history use different locks for the same data flow

`portfolio/signal_engine.py` has `_persistence_lock`, `_cross_ticker_lock`, `_phase_log_lock`, etc. — eight module-level threading.Locks. `portfolio/signal_history.py:30` has its own `_history_lock`. The signal-engine compute path writes signal votes that signal_history then reads to compute persistence. The two locks don't compose — signal_engine can write a vote and emit a "phase change" event into one log while signal_history's read-modify-write sees a stale snapshot, producing diverged persistence scores. Whether this is a real bug depends on ordering, but the multi-lock design is brittle.

### M09. Layer 2 subprocess prompts embed unsanitized trigger reasons via `escape_markdown_v1(", ".join(reasons[:3]))`

`portfolio/agent_invocation.py:1141-1145`. The reasons strings come from signal-engine code paths that include ticker symbols and signal names. Today both are constants, but a future signal that includes user-fetched headlines or news strings in its reason would inject content directly into the Telegram notification. The `escape_markdown_v1` helps with Telegram parse-mode escapes only — it does NOT sanitize for the agent's prompt. Worth a defense-in-depth length cap and ASCII filter.

### M10. `portfolio/main.py:874` `os.environ.get("NO_TELEGRAM")` truthiness check

`if os.environ.get("NO_TELEGRAM"):` — any non-empty value (including `"0"`, `"false"`, `"no"`) suppresses Telegram. Common footgun: setting `NO_TELEGRAM=false` to enable. Spread across 5+ files (`bigbet.py:159`, `loop_contract.py:2025`, `message_store.py:111`, `telegram_notifications.py:36`). Pick one canonical truthy check and centralize.

## P2 — Cross-cutting latent

### M11. `signal_log.jsonl` and `signal_log.db` write paths diverge

`portfolio/accuracy_stats.py:50` comments acknowledge that signal_log.db is the read source while signal_log.jsonl is the legacy write target. If a backfill writes only to one (or one rotation truncates one and not the other), accuracy numbers split.

### M12. ThreadPoolExecutor exception aggregation in `main.py`

If two tickers raise different exceptions concurrently, only the first is observable through `future.result()`. Agent 2 caught the broader case but the multi-error aggregation question is its own latent loss.

### M13. Layer 2 subprocess can mutate portfolio_state.json while dashboard `/api/portfolio` is reading

`dashboard/app.py` reads via `load_json` which is fine for atomic-rename writes, but if Layer 2 ever switches back to raw `json.dump`, dashboard hits TOCTOU. Verified portfolio_mgr is correct today but the contract is fragile.

### M14. CHANGELOG / SESSION_PROGRESS / IMPROVEMENT_BACKLOG drift

Three different docs all promise to track the same thing. No automatic check that an "implemented" item in SESSION_PROGRESS got removed from IMPROVEMENT_BACKLOG. Drift is the norm.

## P3 — Cross-cutting minor

### M15. `requirements.txt` vs actual `.venv` not validated

The venv is checked in (`.venv/Scripts/python.exe` is the canonical interpreter) but no pre-commit check that `requirements.txt` matches `.venv` `pip freeze`. Drift causes "works on my machine" loop crashes.

### M16. Test count claim of "~5,994 tests across 242 files" in CLAUDE.md

Memory note + CLAUDE.md disagree on count vs `ls tests/ | wc -l` (415). Doc rot. Pick one number, automate.

---

## Findings Summary

| Severity | Count |
|----------|-------|
| P0 | 4 |
| P1 | 6 |
| P2 | 4 |
| P3 | 2 |
| **Total** | **16** |

Note: many cross-cutting findings overlap with single-subsystem agent findings — see synthesis doc for deduplication.
