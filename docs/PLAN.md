# PLAN — Error-investigation fixes (2026-06-01)

Branch: `fix/state-resilience-20260601`

## Context

Investigation of the dashboard "Unresolved (12 err)" list surfaced three error
categories. Data-level fixes already applied on `main` (the malformed
`portfolio_state.json` was repaired and the 8 stale journal entries resolved).
This branch ships the **code-level** fixes that prevent recurrence and remove
the noise at its source.

### Root causes found

1. **`portfolio_arithmetic` (CRITICAL, real):** the 13:53 session hand-edited
   `data/portfolio_state.json` to append a 12:40 XAG SELL, inserting the new
   object *after* the `]` that closes the `transactions` array → invalid JSON
   (`Expecting property name … line 518`). Violated Critical Rule #4 (atomic
   I/O only). `loop_contract` detected it but reported only "not a dict",
   which is misleading (the file is unparseable, not a non-dict).

2. **`avanza_account_mismatch` ×5 (noise):** `verify_default_account()` logs a
   **`level: critical`** `critical_errors.jsonl` entry on every `fetch_failed`,
   including the routine case where the Avanza BankID session has expired
   (~24 h lifetime). Session expiry is an *operational* state only a human
   relogin can fix — yet it (a) clutters the unresolved-critical list daily and
   (b) triggers `PF-FixAgentDispatcher` to spawn a fix agent (`Read,Edit,Bash`,
   no commit/login powers) that **cannot** resolve it → wasted tokens + backoff.

3. **`accuracy_degradation` ×3 (benign):** real signal decay
   (`statistical_jump_regime` family 62→44 %, `XAG-USD::momentum_factors`
   63→35 %, `MSTR::econ_calendar` 55→33 %). All sit < `ACCURACY_GATE_THRESHOLD`
   (0.47) recent-window → already auto force-HOLD, no live-trade harm. Alert
   re-fires hourly against the 13.3-day-old baseline high-water mark and
   self-clears once the baseline rolls forward. **ACCEPT — no code change**
   (signal weights/thresholds are off-limits per protocol; the gate already
   neutralises these signals).

### Latent danger uncovered (the reason this is worth a branch)

`portfolio_mgr._load_state_from()` recovers a corrupt state file from `.bak`
backups, but if the file is corrupt **and** all backups are missing/corrupt it
silently returns `_DEFAULT_STATE` (fresh 500 K, no holdings). The next
`save_state()` then overwrites the corrupt file with those defaults — the entire
portfolio track record is **lost with only a `logger.critical` line** (no
journal entry, no Telegram). There are currently **no `.bak` files on disk**, so
during today's corruption window a single trade trigger would have wiped the
Patient portfolio silently. This is the highest-value fix.

## What this branch changes

### Fix 1 — `portfolio/portfolio_mgr.py`: no silent portfolio wipe (P0)
In `_load_state_from`, the fall-through that returns fresh defaults after a
corrupt file with no usable backup currently only calls `logger.critical`.
Change it to **fail loud and preserve evidence** before returning defaults
(loop must keep running, so we still return defaults):

- Quarantine the corrupt file: copy `path` → `path` + `.corrupt-<utc-stamp>` so
  the unparseable content (often hand-recoverable, as today's was) is not lost
  when the next save overwrites it.
- Append a `critical_errors.jsonl` entry (`category: "portfolio_state_corrupt"`)
  via `atomic_append_jsonl` so it reaches the canonical surfacing path.
- Best-effort Telegram alert (lazy import, swallow failures — same pattern as
  `avanza_account_check._send_telegram`).
- Only THEN return defaults.

Guard the quarantine+alert so it never raises (a failure here must not crash the
read path). Idempotency: only quarantine when `path.exists()` and we are on the
all-backups-failed branch. Quarantine copy uses raw bytes (file is unparseable),
not load_json.

### Fix 2 — `portfolio/loop_contract.py`: accurate corruption diagnostics (P1)
In `_check_portfolio_arithmetic`, when `load_json` returns a non-dict for an
existing file, re-read the raw bytes and attempt `json.loads` to capture the
real `JSONDecodeError` (msg + line/col). Put that in the violation message
instead of the bare "not a dict", e.g.:
`"… invalid JSON: Expecting property name enclosed in double quotes (line 518 col 5)"`.
Read-only, additive; falls back to the existing message if the re-read also
can't explain it (e.g. genuinely a JSON list/number at top level, or the file
vanished between reads).

### Fix 3 — `portfolio/avanza_account_check.py`: de-escalate session expiry (P2)
- Add `_is_session_expiry(reason)` → True when the reason contains
  `"session expired"` (case-insensitive).
- In the `fetch_failed` branch, when it's a session expiry, write the journal
  entry with **`level: "warning"`** and `category: "avanza_session_expired"`
  (operational), NOT `level: "critical"`. Genuine fetch failures (DNS/5xx/auth
  blip) keep `level: "critical"` + `category: "avanza_account_mismatch"`.
- On a successful verify (`ok=True`), best-effort **auto-resolve** any still-
  unresolved `avanza_account_mismatch` / `avanza_session_expired` originals by
  appending resolution lines (so the relogin closes the loop). Guarded; never
  raises into the verify path. Only targets entries whose `level` is
  critical/warning and `category != "resolution"` — never resolves a resolution.

The dashboard tile and `check_critical_errors.py` both key on
`level == "critical"`, so the downgrade removes the daily clutter and stops the
useless fix-agent dispatch while keeping full visibility (the warning is still
journaled).

## Files

| File | Change |
|------|--------|
| `portfolio/portfolio_mgr.py` | quarantine + journal + telegram on corrupt-no-backup |
| `portfolio/loop_contract.py` | real JSON error in portfolio_arithmetic message |
| `portfolio/avanza_account_check.py` | session-expiry → warning + auto-resolve on success |
| `tests/test_portfolio_mgr_corrupt_quarantine.py` | NEW — wipe-prevention tests |
| `tests/test_loop_contract.py` | extend — diagnostic message asserts real JSON error |
| `tests/test_avanza_account_check.py` | extend — expiry-downgrade + auto-resolve |

## Execution order

1. Batch 1 — Fix 1 (portfolio_mgr) + tests. Test targeted.
2. Batch 2 — Fix 2 (loop_contract) + tests. Test targeted.
3. Batch 3 — Fix 3 (avanza_account_check) + tests. Test targeted.
4. Adversarial review (cavecrew-reviewer) on diff; fix P1/P2.
5. Full suite `pytest -n auto`; merge; push (Windows git); cleanup worktree.

## What could break (pre-premortem seed)
- Quarantine writes inside the locked read path → I/O failure must not crash load.
- `loop_contract` re-read races a concurrent atomic write → tolerate, fall back.
- Auto-resolve appends could loop (resolution entry re-read as needing resolve) —
  must only target `level critical/warning` originals, never `category resolution`.
- Telegram import at module load could slow read path — keep lazy.

## Premortem
_(filled in after the premortem agent returns)_
