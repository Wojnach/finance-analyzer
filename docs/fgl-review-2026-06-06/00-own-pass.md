# Adversarial Review 2026-06-06 — Orchestrator's Independent Pass

Written BEFORE collecting subagent results, from a first-principles read of the
foundational + integration-seam files. Goal: catch cross-subsystem / systemic
issues a per-subsystem reviewer structurally cannot see, and establish ground
truth on the atomic-I/O layer so subagent findings on it can be cross-checked.

Files read in full: `portfolio/file_utils.py`. Seams verified by targeted grep:
`agent_invocation.py`, `claude_gate.py`, `grid_fisher.py`, `signal_history.py`.

Severity: P0 = money loss / data corruption / outage / silent-failure-masking /
secret leak. P1 = real bug. P2 = robustness/race-under-load. P3 = maintainability.

---

## Confirmed findings

### OWN-1 (P1) — Layer-2 trade spawn ignores the `CLAUDE_ENABLED` master kill-switch
`portfolio/agent_invocation.py:853` gates Layer-2 invocation **only** on
`config.layer2.enabled`. The `claude_gate` module exposes a master enable/detector
path, but the trade-spawn path imports only `detect_auth_failure` (line 15) — it
never consults a master `CLAUDE_ENABLED` switch. Operational consequence: flipping
the master switch off does **not** stop trade subprocesses; only the per-feature
`layer2.enabled` flag does. This is a "silent half-on" trap — an operator who kills
the master switch during an incident still gets live trade spawns. Independently
corroborates the orchestration reviewer's same-line P0. Severity P1 (it does not
itself produce a bad trade, but defeats an incident-response control).
→ Route the spawn through the gate, or have the `layer2.enabled` check also honor the master switch.

### OWN-2 (P2) — `atomic_write_jsonl` bypasses the `jsonl_sidecar_lock` contract
`portfolio/file_utils.py:352`. `prune_jsonl` (line 445) and `rotate_jsonl` both
wrap their read→rewrite→`os.replace` in `jsonl_sidecar_lock` precisely so a
concurrent `atomic_append_jsonl` cannot be silently dropped between the reader's
"read all lines" and the `os.replace` (the documented `signal_log_reconciliation`
divergence). `atomic_write_jsonl` does the *same* read-elsewhere → write-tmp →
replace **without acquiring the lock**. Any file that is both full-rewritten via
`atomic_write_jsonl` and appended via `atomic_append_jsonl` can lose appends.
Current callers (`signal_history.py:48`, `forecast_accuracy.py`, `fin_evolve.py`)
appear single-writer per file, which masks the bug today — but the primitive is a
latent foot-gun that contradicts the module's own lock contract.
→ Acquire `jsonl_sidecar_lock(path)` inside `atomic_write_jsonl`, mirroring `prune_jsonl`.

### OWN-3 (P3) — No parent-directory fsync after `os.replace` (POSIX-only durability gap)
`portfolio/file_utils.py` `atomic_write_json/text/jsonl` + `prune_jsonl` fsync the
**temp file** before `os.replace` (good) but never fsync the **parent directory**
after. On POSIX, the rename's durability requires a directory fsync; a power loss
between replace and the dir flush can lose the rename. Production is Windows
(`os.replace` → `MoveFileEx`, unaffected), so this is prod-safe today and only bites
if a loop is ever run under WSL/Linux. Documented as P3 / ACCEPT-on-Windows.
→ If any loop migrates to Linux, add an `os.open(dir, O_DIRECTORY); os.fsync; close` after replace.

---

## Ground-truth notes for cross-critique (things that are CORRECT, to catch false positives)

- **`atomic_append_jsonl` self-heal is NOT buggy on empty files.** Lines 342–346
  guard `if f.tell() > 0:` *before* the `f.seek(-1, os.SEEK_END)`. A brand-new /
  size-0 file never reaches the seek-backward, so there is no undefined `seek(-1)`
  and no spurious-newline path. Any subagent P0 claiming "seek(-1) on a new file is
  UB" is a **false positive** — the guard already exists.
- **`grid_fisher` global cap is well-guarded.** `_effective_global_cap`
  (grid_fisher.py:978) fails *closed* on account-resolve failure and bypasses only
  when no `account_id` is configured; the per-tier loop re-checks `projected >
  global_cap_sek` per tier (line 1417–1424, the 2026-05-28 fix), not just at entry.
  The metals loop drives this single-threaded per tick, so there is no cross-thread
  cap race. Reject any P0 claiming an unguarded global-cap breach.
- **Stop-loss endpoint discipline holds.** No regular-order endpoint is used for
  stop placement in the avanza layer (grep clean); stops route to
  `/_api/trading/stoploss/new`.
- **Atomic-I/O discipline holds repo-wide.** No `json.loads(open(...))` /
  `open(...,'w').write(json...)` anti-pattern anywhere under `portfolio/`.

---

## Cross-cutting themes to weight in synthesis
1. **Fail-toward-trading on stale/degraded inputs** is the recurring shape of the
   real risk here (drawdown cash-only fallback, VaR fx_rate=1.0, warrant book
   unlocked). The system is good at *atomic* writes but weaker at *refusing to act*
   when an input is known-stale. This is the same class as the 3-week auth outage:
   exit-0 / default-value masking a degraded state.
2. **Lock-contract completeness.** The sidecar-lock contract is correct where
   applied but not uniformly applied (OWN-2; warrant book). Audit every JSONL/state
   rewrite path for lock coverage.
3. **Kill-switch / control-plane asymmetry** (OWN-1): controls that an operator
   reaches for during an incident must actually gate the money path.
