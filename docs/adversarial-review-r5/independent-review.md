# Independent Cross-Cutting Adversarial Review — Round 5 (2026-04-14)

**Reviewer**: Manual cross-cutting analysis (orthogonal to 8 subsystem agents)
**Focus**: Systemic issues that span subsystem boundaries

---

## THE BIG QUESTION: Is the risk management subsystem protecting the portfolio?

**Answer: NO.** The risk management subsystem is theater.

Six independent risk/validation functions exist, are well-tested in isolation, but have
**zero callers** in any production execution path:

| Function | File:Line | Purpose | Callers (production) |
|----------|-----------|---------|---------------------|
| `check_drawdown()` | risk_management.py:86 | 20% drawdown circuit breaker | **ZERO** |
| `record_trade()` | trade_guards.py:177 | Per-ticker cooldown, overtrading prevention | **ZERO** |
| `validate_trade()` | trade_validation.py:22 | Pre-order parameter validation | **ZERO** |
| `classify_trade_risk()` | trade_risk_classifier.py:29 | Trade risk scoring (6-factor) | **ZERO** |
| `recommended_size()` | kelly_sizing.py:204 | Kelly criterion position sizing | **ZERO** (main portfolios) |
| `exposure_coach` enforcement | exposure_coach.py:89 | `new_entries_allowed=False` flag | **ADVISORY ONLY** |

The only hard enforcement point in the entire codebase is the 1000 SEK minimum in
`avanza_session.py:590`, which raises `ValueError` at the last possible moment.

The trade execution path is:
```
metals_loop → metals_execution_engine → avanza_session.place_order()
Layer 2 (LLM subprocess) → reads agent_summary.json → LLM decides → writes journal
```

No Python code enforces risk checks between signal output and trade execution.
Layer 2 "enforcement" depends entirely on the LLM reading and obeying JSON fields.

**Priority: P1 CRITICAL — has been flagged since Round 4 (C6) and remains unfixed.**

---

## Cross-Cutting Findings

### XC-R5-1 (P1 CRITICAL) — Claude subprocess bypass: kill switch ineffective

Four call sites bypass `claude_gate.py`'s `_invoke_lock`, `CLAUDE_ENABLED` kill switch,
CLAUDECODE env stripping, and auth failure detection:

1. `bigbet.py:169` — `subprocess.run(["claude", "-p", ...])` directly
2. `multi_agent_layer2.py:154-168` — `Popen(["claude", ...])` for 3 specialists
3. `analyze.py:272` — `subprocess.run(["claude", "-p", ...])`
4. `analyze.py:727` — `subprocess.run(["claude", "-p", ...])`

Impact: The kill switch (`CLAUDE_ENABLED = False`) cannot actually disable all Claude
invocations. During an outage, setting the kill switch still allows bigbet and specialist
processes to spawn. Up to 5+ concurrent Claude processes can run simultaneously without
serialization. The CLAUDECODE env var inheritance risk (which caused a 34h outage on
Feb 18-19) is present in bigbet.py (no env stripping at all).

### XC-R5-2 (P2 HIGH) — Error swallowing cascade: failures never escalate

`main.py` contains 25+ `except Exception: logger.warning()` blocks in post-cycle operations.
Each individually correct for loop reliability, but:
- No aggregation of repeated failures across cycles
- No circuit breaker for modules that fail every cycle
- No Telegram alert for persistent module failures
- `report.post_cycle_results` tracks success/failure per cycle but is not persisted

A dead signal module or broken API can fail 1440 times/day (every cycle) generating
only WARNING-level logs. No escalation to the operator.

### XC-R5-3 (P2 HIGH) — Cross-subsystem stale data reads

Multiple modules read prices/signals from `agent_summary_compact.json` on disk instead of
live APIs, with no freshness check:
- `crypto_scheduler.py:110-137` — Telegram report uses stale prices
- `perception_gate.py:59-65` — Signal strength gate reads prior-cycle summary
- `crypto_macro_data.py:202-259` — Gold/BTC ratio from disk, cached 1h
- `fin_fish.py` session_hours_remaining — hardcoded 21:55 ignoring DST

When the main loop is down (crash, restart, CLAUDECODE outage), these modules operate on
potentially hours-old data without any indication to the user.

### XC-R5-4 (P2 HIGH) — Inconsistent minimum trade size across subsystems

| Location | Threshold | Enforcement |
|----------|-----------|-------------|
| `avanza_session.py:590` | 1000 SEK | Hard `ValueError` |
| `trade_validation.py:32` | 500 SEK | Advisory (never called anyway) |
| `kelly_sizing.py:290` | 500 SEK | Advisory |
| `kelly_metals.py:44` | 500 SEK | Advisory |

A Kelly sizer returning `recommended_sek=700` will lead Layer 2 to believe 700 SEK is
valid. The hard stop at avanza_session.py catches it only at execution time with an exception.

### XC-R5-5 (P2 HIGH) — meta_learner.py bypasses file_utils

`meta_learner.py` uses `json.loads(path.read_text())` directly at lines 390 and 437,
bypassing `file_utils.load_json()`. This is a TOCTOU race when the ML retrain task
(`PF-MLRetrain`) rewrites the metrics file while the main loop reads it.

### XC-R5-6 (P3 MEDIUM) — `atomic_append_jsonl` is not truly atomic on Windows

The function uses `open(path, "a")` + `f.write(line)` + `f.flush()` + `os.fsync()`.
On POSIX, writes to O_APPEND fds are atomic for writes ≤ PIPE_BUF. On Windows, there is
no such guarantee. The name "atomic" is misleading. In practice, main loop and metals loop
write to different JSONL files, so interleaving risk is low, but the function should at
least document the limitation.

### XC-R5-7 (P3 MEDIUM) — Specialist report files never cleaned up

`multi_agent_layer2.py` has `cleanup_reports()` but `agent_invocation.py` never calls it.
Synthesis agent reads `data/_specialist_*.md` files that may be stale from a prior run.
A failed specialist run leaves prior-run data on disk; the synthesis agent acts on it.

---

## Summary

| ID | Priority | Category | Description |
|----|----------|----------|-------------|
| XC-R5-1 | P1 | Risk | Risk management subsystem entirely disconnected from production |
| XC-R5-1b | P1 | Orchestration | Kill switch ineffective — 4 bypass paths |
| XC-R5-2 | P2 | Reliability | Error swallowing cascade — no escalation for persistent failures |
| XC-R5-3 | P2 | Data integrity | Stale data reads across subsystems (no freshness check) |
| XC-R5-4 | P2 | Risk | Inconsistent min trade size (1000 vs 500 SEK) |
| XC-R5-5 | P2 | Data integrity | meta_learner.py raw JSON reads bypass file_utils |
| XC-R5-6 | P3 | Infrastructure | atomic_append_jsonl not truly atomic on Windows |
| XC-R5-7 | P3 | Orchestration | Specialist report files never cleaned up |
