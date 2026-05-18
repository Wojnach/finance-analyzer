# Own Independent Review — Claude main pass

Reviewed alongside subagents. Findings verified by source inspection.

## P1 / 🔴 CONFIRMED bugs (read-and-verified)

1. **agent_invocation.py:1395-1396** — VERIFIED. `_check_agent_completion_locked()` (def line 1334) reads `_journal_count_before` and `_telegram_count_before` without `global` declaration. These names ONLY exist as locals inside `invoke_agent()` (set at 1059-1060 without `global`). Module scope has `_journal_ts_before` and `_telegram_ts_before` only — NOT the `_count_` variants. **Outcome: `NameError` on first watchdog tick after a subprocess actually completes.** Code was added 2026-05-17. Subagent finding confirmed. Fix: declare `global _journal_count_before, _telegram_count_before` at top of `_check_agent_completion_locked` AND at top of `invoke_agent`, and initialize module-scope to 0.

2. **file_utils.py:371-415 (prune_jsonl)** — VERIFIED. `prune_jsonl` does NOT acquire `jsonl_sidecar_lock(path)`. Reads file, then atomic-replaces it. Any `atomic_append_jsonl(path, entry)` call from another thread between read and `os.replace` is LOST — exactly the divergence class the 2026-05-11 signal_log_reconciliation contract is supposed to detect. Same bug class as the `log_rotation` issue the lock primitive was extracted to solve. Used on `invocations.jsonl`, `signal_log.jsonl` (via main.py and others — 28 grep hits). Fix: wrap entire read+replace in `with jsonl_sidecar_lock(path):`.

3. **risk_management.py:760-761** — VERIFIED. `if total_value <= 0: return None`. Silent blind spot on all-cash or net-negative portfolio. Caller can't distinguish "concentration safe" from "skipped".

4. **signal_engine.py:3933 / 2644** — Confirmed: gated today by line 3929 `active_voters < min_voters` (min_voters >= 2 for metals via MIN_VOTERS_METALS, >=3 elsewhere). Not a live bug today. **Demote to P3 robustness** ("future refactor risk"); not P1.

## P2 / 🟡 OWN findings

5. **agent_invocation.py:55-56 + 1059-1060** — Module-scope `_journal_ts_before`/`_telegram_ts_before` ARE in the `global` declaration at 1340, but `invoke_agent` at line 1061-1062 reassigns them as locals without `global` declaration. After invoke_agent runs, the locals shadow nothing; module-scope retains its initial `None`. The completion-watchdog reads stale Nones forever. Same class as #1 above but for the `_ts_` variants. Fix: `global _journal_ts_before, _telegram_ts_before` inside `invoke_agent`.

6. **signal_engine.py CRITICAL-2 (per .claude/rules/signals.md)** — Documented pre-existing bug: `ticker=""` dispatch. Listed as known but appears unfixed. No timeline. Worth surfacing.

7. **.claude/rules/signals.md:14** — Documentation drift: rule says "Applicable signal counts: crypto=29, stocks=25, metals=27" but CLAUDE.md says "crypto=16, stocks=10, metals=12". One is wrong.

8. **critical_errors.jsonl current state** — System has unresolved `accuracy_degradation` entry from 2026-05-18T14:29:20: 5 signals dropped >15pp vs 7d baseline AND below 50% absolute (sentiment 65.6%→39.3%, structure 61.1%→37.0%, macro_regime 54.3%→37.0%, econ_calendar 71.2%→33.4%, crypto_macro). The `econ_calendar` 71.2%→33.4% drop matches the hardcoded-time bug from data-external review (events at 14:00 UTC instead of 19:00/13:30 UTC). Likely root cause discovered by review.

## Cross-check observations

- Subagent's `signal_engine.py:3933` ZeroDivisionError claim → demoted to P3 after verification (gated).
- Subagent's `agent_invocation.py:1395` UnboundLocalError claim → ESCALATED to P1 after verification (real NameError on first watchdog tick).
- Subagent's `risk_management.py:760` silent None claim → verified P1.
- File_utils.py atomic_write_json / atomic_append_jsonl design looks correct EXCEPT prune_jsonl gap (own P1 above).

## P3 / 🔵 OWN findings (lower severity)

- prune_jsonl logs `INFO` after pruning — fine — but doesn't write a critical_errors entry if write fails (only logs); silent partial-prune state hard to diagnose.
- count_jsonl_lines (line 360) opens file in binary mode and reads line by line — fine, but a concurrent appender's last line may be visible mid-write as a blank-after-strip → undercount by 1. Used as completion-check baseline at agent_invocation.py:1059 → race window of one entry. Low probability.
- atomic_write_json (line 53) uses `default=str` for json.dump — silently coerces non-serializable types to str; could mask Decimal/datetime/Path leaks into JSON files (drift over time).
