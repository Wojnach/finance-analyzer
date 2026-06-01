# FGL Review — Independent Pass (session author, 2026-06-01)

This is the orchestrator's own adversarial pass, written *before* collecting the
8 subagent reviews, to provide a cross-check independent of them. It focuses on
what I personally read and verified against live code, and on the **cross-cutting
root cause** that no single subsystem owns. Line numbers verified against `main`
at commit `3518bfed`.

---

## P0 — Layer 2 persists trades by hand-editing JSON (recurring corruption root cause)

**The corruption that fired 12× today is architectural, not a portfolio_mgr bug.**

Evidence chain (all verified):
- `docs/TRADING_PLAYBOOK.md:127` — the documented persistence step is literally
  **"Edit `data/portfolio_state.json` (patient) or `data/portfolio_state_bold.json` (bold)."**
- `portfolio/agent_invocation.py:1159` — the Layer 2 subprocess is launched with
  `--allowedTools "Edit,Read,Bash,Write"`. So the LLM modifies the live state file
  with its generic **Edit/Write** tool.
- `data/layer2_action.py:49-50`, `data/layer2_exec.py:46-47` — L2-authored throwaway
  scripts (left in `data/`) persist via **raw `open(path,"a") + json.dump`**, not
  `atomic_append_jsonl`. Same modus operandi, different file.
- `memory/now.md` 13:53–15:00: *"L2: portfolio_state.json corrupted; XAG 26.92oz prior
  unverifiable."* The agent itself produced the unparseable file.

**Causal chain:** an LLM editing a 500-line JSON document with a line-based Edit tool
will, eventually, drop a bracket / append after the closing `]` / mis-indent →
`json.loads` fails → `load_json` returns `None` → `_load_state_from` quarantines +
returns fresh 500K defaults → the loop trades against a blank portfolio until a human
restores. This **bypasses Critical Rule #4 (atomic I/O only) on the system's own
primary trade-write path.** Every defensive fix shipped today (quarantine, fail-loud,
diagnostics — commits `57de1814`/`c5ba4ae0`) is downstream mitigation; the write path
itself was never changed, which is why corruption **recurred** at 15:11–15:19.

**Fix (the real one):** give Layer 2 a typed, atomic persistence CLI and forbid raw
edits. e.g. `python -m portfolio.layer2_apply --strategy patient --action BUY
--ticker XAG-USD --shares N --price P --reason "..."` that routes through
`portfolio_mgr.update_state()` (atomic write + per-file lock + `_validated_state`
schema merge). Then drop `Edit,Write` from the L2 allow-list (keep `Read,Bash`),
and rewrite playbook §"Execute" to call the CLI. This makes a malformed write
*impossible* instead of *recoverable-after-the-fact*. **This is the highest-value
change in the whole codebase right now.**

---

## P1 — Portfolio state lock is per-process; Layer 2 + Layer 1 are different processes

`portfolio/portfolio_mgr.py:34-47` — `_state_locks` is a dict of `threading.Lock`.
`_save_state_to` (183-188) and `update_state` (211-234) serialize **only threads
within one process**. But `portfolio_state.json` is written by *both* the Layer 1
main loop process *and* the Layer 2 `claude -p` subprocess (and read by the dashboard
process). A `threading.Lock` provides **zero** mutual exclusion across processes.

Contrast: `file_utils.atomic_append_jsonl` (269-292) correctly uses the
**cross-process** `jsonl_sidecar_lock` (msvcrt/fcntl). The primitive to do this right
already exists in the codebase — the state-save path just doesn't use it.

**Causal chain:** Layer 1 `update_state` reads state, Layer 2 Edit writes state, Layer 1
`os.replace`s its now-stale copy → Layer 2's trade is silently lost (a real BUY/SELL
vanishes from the record). `os.replace` keeps the file *parseable*, so this is a
**silent lost-update**, arguably worse than the loud corruption. **Fix:** wrap
`_save_state_to`/`update_state`'s read-modify-write in `jsonl_sidecar_lock(path)` (or a
dedicated cross-process state lock), and route L2 through the same lock via the CLI above.

## P1 — `default=str` + `allow_nan=True` lets non-JSON / wrong-typed values persist silently

`portfolio/file_utils.py:64` — `json.dump(data, f, ..., default=str)` with the json
default `allow_nan=True`. Two failure modes:
1. A `float('nan')`/`inf` in state (e.g. a price that came back NaN) serializes to the
   bare literal `NaN`/`Infinity` — **invalid per the JSON spec**. Python's `json.loads`
   tolerates it, but the dashboard's browser-side `JSON.parse` (and any strict consumer)
   throws → the "corruption" surfaces only client-side.
2. `default=str` silently coerces any non-serializable object (numpy float, pandas
   Timestamp, Decimal) to its `repr` string → a `shares` field becomes `"1.0"` or a
   price becomes `"Timestamp(...)"`; `_validated_state` does not coerce numeric types
   back, so downstream `shares * price` math mis-behaves.

**Fix:** `allow_nan=False` (raise loudly on NaN at write time, when the bug is local)
and replace `default=str` with an explicit encoder that rejects unexpected types or
converts numerics deliberately.

## P2 — Backup ring can be poisoned by the corrupt file; backups created only on save

`portfolio/portfolio_mgr.py:50-68` (`_rotate_backups`) runs **only** inside
`save_state`/`update_state`. Most Layer 2 invocations are HOLD (no save) → backups are
refreshed rarely, so a corruption that lands between trades has **no recent `.bak`**
(matches today's "no backup recovered"). Worse: when `_load_state_from` recovers from a
`.bak` (166-172), the subsequent `update_state` calls `_rotate_backups` while the
on-disk `path` is *still the corrupt content* → it copies corrupt bytes into `.bak`
(66) before the atomic write replaces `path`. Repeated corruption cycles shift corrupt
copies through `.bak`→`.bak2`→`.bak3`, eventually poisoning the whole ring.
**Fix:** back up the *validated in-memory state* after a successful load, not the raw
on-disk file; and/or write a backup on every load that parses cleanly, not only on save.

---

## Verified-OK (adversarial checks that did NOT find a bug — recorded so they aren't re-litigated)

- **Accuracy gate is not inverting.** `signal_engine.py:426-434,2002-2021` — sub-47%
  (50% for ≥7K-sample) signals are force-HOLD, explicitly *not* inverted (the comment
  records that inversion caused whiplash). Today's `accuracy_degradation` alerts are the
  gate working as designed; consistent with the prior session's ACCEPT. *Minor:* a 0.46
  literal at `2512` vs `ACCURACY_GATE_THRESHOLD=0.47` — flagged to the signals-core
  reviewer to confirm it's an intentional separate borderline gate, not drift.
- **Metals stop-loss uses the dedicated path.** `data/metals_loop.py:348,2480` call
  `place_stop_loss(...)`, not the regular order endpoint; regular `order/new` (3782) is
  used only for BUY/SELL entries (3040,3108) — correct. (Internals of `place_stop_loss`
  delegated to the metals-core/avanza-api reviewers.)
- **`atomic_write_json`/`atomic_write_text` themselves are correctly atomic** —
  `tempfile.mkstemp` in the dest dir + `flush` + `os.fsync` + `os.replace`, with tmp
  cleanup on failure (`file_utils.py:32-71`). The corruption is NOT from this function;
  it's from writers that don't call it (see P0).

---

## Summary
- **1 P0** (Layer 2 raw-edit persistence — the recurring-corruption root cause; spans
  orchestration + portfolio-risk + infrastructure, owned by none).
- **2 P1** (per-process-only state lock → silent lost update; `default=str`/`allow_nan`
  silent bad persists).
- **1 P2** (backup-ring poisoning + save-only backups).
- Single most important action: **route Layer 2 trade writes through an atomic CLI and
  remove Edit/Write from its tool allow-list.** Fixing only the recovery path (as done
  today) treats the symptom; this fixes the cause.
