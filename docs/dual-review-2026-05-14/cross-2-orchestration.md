# Cross-Critique — 2 orchestration

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/agent_invocation.py:572` — `_agent_proc = None` after failed kill is P0.** Both reviewers locked onto the same line, same docstring, same `kill_ok=False` flag. **Independent rediscovery — confidence very high.** Production effect (Codex states explicitly): the auth-error scanner relies on `_agent_log_start_offset` which the second concurrent claude resets, so any `"Not logged in"` from the old process between offsets is now invisible — recreates the 3-week silent-auth failure mode. Action: gate the clear on `kill_ok`.

- **Auth cooldown fails open on file-read failure (`agent_invocation.py:597-624`)** — both reviewers flag the load_jsonl swallow. Codex notes the file is read every cycle and the comment "Better to attempt and fail loudly" is wrong-headed given the 8-spawn-in-30-min auth storm pattern from 2026-05-10. Action: cache last-auth-error in module state OR fail closed on read failure.

- **Ticker pool zombie leak (`main.py:608-661`)** — both flag `pool.shutdown(wait=False, cancel_futures=True)` doesn't actually cancel running-but-stuck threads. Codex quantifies: 5 stuck threads × N stuck cycles → unbounded. Claude proposes `as_completed(timeout=...)` + log thread IDs. Real, ongoing, documented in code comments. Action: per-call HTTP timeout inside `_process_ticker` OR a semaphore around it.

- **Stale journal tail read after `journal_written` (`agent_invocation.py:1338-1342`)** — both flag the `last_jsonl_entry(JOURNAL_FILE)` race vs metals_loop/autonomous appends. Codex links this to fishing_context.json → grid_fisher direction_bias → real-money mis-direction. P0 in impact. Action: capture entry from the same diff that decided `journal_written`.

## Codex found, Claude missed

- **`portfolio/multi_agent_layer2.py:163-185, 210` — specialists bypass `claude_gate` entirely (P0).** Claude flagged this at P2 with `proc.kill()` not killing the tree, but Codex elevates correctly: bypassed lock, bypassed tree-kill, no invocations.jsonl entry (cost/rate-limiter blind), and auth scan only post-wait. This is the same pattern as the documented gate.py forbidden direct-Popen call. **Should be P0**, not P2.

- **`portfolio/analyze.py:282-289` — `subprocess.run(["claude", "-p", ...])` bypasses `claude_gate`.** Claude didn't even open `analyze.py`. Codex catches it: same forbidden direct-Popen pattern, plus subset-of-clean-env (no `CLAUDE_CODE_ENTRYPOINT` pop), plus auth check uses concat that could land beyond `_AUTH_SCAN_LINE_LIMIT`. Plausible silent path. P1.

- **`portfolio/loop_contract.py:1004` — `_JOURNAL_UNIQUENESS_WINDOW_S = 600` shorter than T3 timeout (900s).** Subtle one Claude missed. A T3 timeout-and-respawn writes two journal entries 15min apart and slips the duplicate-detection window. Real P2 with a fix in one line.

- **`portfolio/trigger.py:165-199` corruption-masking startup grace.** Claude flagged `_load_state` general corruption (P1); Codex extends to the specific path: `state={}` from corruption triggers grace, overwrites with empty state, masks corruption forever. **Stronger framing — same line of code, sharper consequence.**

- **`portfolio/agent_invocation.py:1411-1423` — `_agent_timeout` not cleared in cleanup.** Codex notes this is a footgun for any future refactor. Claude flagged the same uncleared-state at P2 line 1226 — both right, Codex's framing is "future-proofing", Claude's is "current cosmetic". Same risk.

## Claude found, Codex missed

- **`portfolio/agent_invocation.py:562` — `_agent_tier=None` produces literal `"layer2_tNone_timeout"` caller field.** Codex didn't flag this. Small but real — the critical_errors caller field is supposed to be searchable; the string `"None"` is a debug-time lie. P2.

- **`portfolio/agent_invocation.py:946-953` — `log_fh` opened in text mode with CRLF translation breaking the byte offset.** Codex says agent.log "opened in non-atomic append mode" (P1) but doesn't surface the Windows-CRLF-vs-binary-fd offset bug specifically. Claude's catch is sharper: line `_agent_log_start_offset = agent_log_path.stat().st_size` is bytes, subprocess writes via OS-level fd are bytes, but `log_fh` in text mode means file *flushes* may insert CR before LF — auth marker bytes can land before/after the recorded offset depending on platform encoder. Fix: `"ab"`. Codex missed the encoding interaction.

- **`portfolio/main.py:541` — `ind['rsi'], ind['macd_hist']` direct subscript in log format.** Codex missed. Real P1 — KeyError converts to silent ticker skip via outer try/except.

- **`portfolio/journal.py:23-40` — `load_recent` raw `open()` + `json.loads` per line.** Codex flagged `llm_calibration.py`'s version of this in the signals-core review but not the orchestration version. Claude is right that the dashboard + agent_invocation both read this path mid-`atomic_append_jsonl` rename.

## Disagreements

None on substance. Codex elevates `multi_agent_layer2` to P0 where Claude had P2 — both can defend; Codex's framing (gate bypass = silent auth failure surface = 3-week-outage shape) is the more defensible call. **Adopt Codex's severity.**

## What BOTH missed (third pass)

- **`portfolio/agent_invocation.py` `_invoke_lock` not held across the "auth cooldown decided to allow" → "Popen spawn" critical section.** If two Layer 1 cycles overlap (slow cycle + watchdog tick), both can pass the cooldown check and both spawn. Neither reviewer cross-referenced with `claude_gate._invoke_lock` — there *is* a lock around Popen, but the cooldown check is OUTSIDE it. Reorder: check cooldown inside `_invoke_lock`.

- **`portfolio/main.py` `pool` is reset every cycle but `_TICKER_POOL_TIMEOUT` is enforced via `as_completed(timeout=...)`** — the *timeout* is loop-scoped, not pool-scoped. If Cycle N-1 leaves 5 stuck threads, Cycle N starts a fresh pool with 8 *new* workers, but the platform's CreateThread limit doesn't care about Python's pool object. Neither reviewer counted total live threads — `threading.active_count()` would be the real probe.

- **`portfolio/multi_agent_layer2.py:163-185` calls `subprocess.Popen` with `stdout=open("data/_specialist_*.log", "w")`**, but the `proc.stdout` attribute is None (it's redirected). The wait_for_specialists code (codex flagged at P0) reads from log files post-wait — if those files were opened in "w" mode by another thread mid-spawn, they truncate, losing the specialist output. Neither reviewer checked the file-open mode race.

- **`portfolio/digest.py` 4h-cadence sends.** Codex briefly mentions it; Claude doesn't dive in. Neither reviewer checked whether `digest.send_digest` re-reads `metals_swing_state.json` and produces stale claims about live positions during a position transition.

- **`portfolio/loop_health.py`** — not opened by either reviewer. It's a heartbeat-decision module and its drift detection is what flags "loop dead" alerts. Untested.

## Verdict

P0 list after cross: **3 confirmed** (kill_ok=False race, multi_agent gate bypass, journal-tail-race → fishing_context). 
P1 list after cross: **~7 confirmed** (auth-cooldown read-failure, ticker-pool zombies, analyze.py direct-Popen, trigger corruption-masking, agent_timeout uncleared, ind[] direct subscript, journal.load_recent non-atomic).
P2 list after cross: ~6.
