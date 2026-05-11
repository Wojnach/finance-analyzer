# PLAN — Cleanup (rotation race + account verify) — 2026-05-11

## Goal

Address two unresolved follow-ups that surfaced in session memory but
aren't tied to the grid-fisher feature:

1. **`signal_log_rotation_race`** — `portfolio/log_rotation.rotate_jsonl`
   reads-rewrites-os.replace without holding the sidecar lock used by
   `atomic_append_jsonl`. Any append between rotation's read and
   replace is silently discarded, and the SQLite dual-write keeps the
   record — driving the `signal_log_reconciliation` contract invariant
   into ESCALATED state. ~400 entries lost per pass.

2. **`avanza_account_mismatch_20260511`** — User reported the dashboard
   showing Beammwave / NextEra / Vertiv holdings on account `1625505`,
   which is the hard-coded `DEFAULT_ACCOUNT_ID`. Those look like an ISK
   account, not a warrant-trading account. Risk: every live bot order
   has been routed at the wrong account ID. As of 2026-05-11 the live
   bots are paper / signal-only on those paths, so blast radius is
   limited, but live grid-fisher placement is now wired up — confirm
   before any real fill lands.

Both items are independent of the grid fisher work but both block / risk
production reliability.

## Why

- Contract invariant `signal_log_reconciliation` has been ESCALATED 22x
  in a row. We're spending alert budget on a real divergence we can fix.
- Grid fisher just shipped to live trading. If `DEFAULT_ACCOUNT_ID` is
  wrong, the first qualifying signal could place orders into the wrong
  account. We can't trust the existing whitelist guard (it allows
  exactly the one ID that's apparently wrong).

## Scope

### Batch 1 — rotation race
- Extract sidecar-lock context manager from `atomic_append_jsonl` into
  `portfolio/file_utils.jsonl_sidecar_lock(path)` so any code that
  needs the same lock can borrow it.
- Refactor `atomic_append_jsonl` to use the helper.
- Refactor `rotate_jsonl` to:
  1. Acquire the sidecar lock for *path*.
  2. Read all lines.
  3. Write the tmp file.
  4. `fsync` the tmp file (missing today).
  5. `os.replace` to swap.
  6. Release the lock.
  All under the same lock, so any concurrent `atomic_append_jsonl`
  blocks behind us and lands in the rotated file (kept_lines path),
  not into a tmp that's about to be replaced.
- Tests: concurrent append + rotate under thread/process load, assert
  no append is lost.

### Batch 2 — account verification
- Add `portfolio/avanza_account_check.py` — fetches
  `/_api/account-overview/overview/categorizedAccounts`, walks all
  accounts, asserts the configured `DEFAULT_ACCOUNT_ID` is present and
  tagged with a trading-class category (`AKTIE_DEPÅ`, `AKTIEFOND`,
  `DEPÅ` — not `INVESTERINGSSPARKONTO` / `KAPITALFÖRSÄKRING` / etc.).
- If mismatch: write `critical_errors` entry + Telegram alert and
  (unless `PF_SKIP_ACCOUNT_CHECK=1` env is set) raise. Default-on so
  fresh deploys catch the misconfig immediately, override available
  for known-bad windows.
- Cache result on the module so repeated startup checks across
  metals_loop / golddigger / grid_fisher hit the API once per process.
- Tests: mocked categorizedAccounts response covering OK case,
  missing account, ISK-only account, network failure (raises — fail
  closed).

## Execution order

### Batch 1
- `portfolio/file_utils.py` — new `jsonl_sidecar_lock`; refactor
  `atomic_append_jsonl` to use it.
- `portfolio/log_rotation.py` — wrap `rotate_jsonl` body in the lock,
  add fsync before replace.
- `tests/test_file_utils_jsonl_lock.py` (new) — concurrency regression.

### Batch 2
- `portfolio/avanza_account_check.py` (new) — verification helper.
- `tests/test_avanza_account_check.py` (new).
- Wire startup call into metals_loop init (grid_fisher / golddigger
  follow-up — keep this PR focused on the helper landing first; the
  consumers can be wired in a tight follow-up once the helper has a
  green test suite).

## What could break

- **Rotation lock** — extending the lock from append-only to
  read-rewrite-rename means rotation now holds the lock for hundreds
  of ms instead of microseconds. Concurrent loops *block* briefly on
  append. Acceptable: trades "data loss" for "transient latency".
- **Account verification raising** — if it fires on a legitimate
  trading account with a non-standard `category` tag, every live bot
  halts at startup. Mitigation: log the raw category alongside the
  failure so the operator can expand the whitelist or set the
  override env var.
- **Concurrent rotation passes** — two PF-LogRotate runs racing the
  same file. The lock serialises them; the second run sees the output
  of the first. Fine.

## Out of scope

- Picking the *correct* account ID — that's a user decision. We surface
  the mismatch via critical_errors + Telegram and let the operator
  update config.
- Rotating non-JSONL logs (`loop_out.txt` etc.) — those use shell `>>`
  redirection and never had the lock contract.

## Verification

- Concurrency stress test: 8 threads doing `atomic_append_jsonl` while
  rotation runs every 50ms. Final count = appended count.
- Unit tests cover all branches of the account check before merge.
- After merge: run `python -m portfolio.log_rotation` once and observe
  no decrease in JSONL count vs SQLite next cycle.
