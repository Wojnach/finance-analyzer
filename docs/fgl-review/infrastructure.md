# FGL Review — infrastructure

Scope: `portfolio/file_utils.py`, `portfolio/health.py`, `portfolio/shared_state.py`,
`portfolio/journal.py`, `portfolio/http_retry.py`, `portfolio/gpu_gate.py`,
`portfolio/telegram_notifications.py`, `portfolio/message_store.py`,
`dashboard/auth.py`, `dashboard/app.py` (+ cross-file context:
`portfolio/portfolio_mgr.py`, `dashboard/house_blueprint.py`, `dashboard/cf_access.py`,
`tests/test_portfolio_mgr_core.py`).

## The live `portfolio_state_corrupt` event today (15:11–15:19) — diagnosis

This was the #1 target. Conclusion: **the live portfolio state was NOT corrupted.**
Every `portfolio_state_corrupt` entry written to `data/critical_errors.jsonl` today
came from `pytest` temp directories, not production:

```
"path": "C:\\Users\\...\\pytest-of-herc2\\pytest-2873\\popen-gw3\\test_load_all_corrupt_returns_0\\portfolio_state.json"
"path": "...\\test_corrupt_json_returns_defa0\\state.json"   (bytes: 15)
"path": "...\\test_null_json_returns_default0\\state.json"   (bytes: 4 → "null")
"path": "...\\test_returns_defaults_when_all0\\portfolio_state.json"
```

Live verification: `data/portfolio_state.json` and `_bold.json` both parse (6 keys
each); there are **zero** `.bak`/`.corrupt-*` files in the live `data/` dir. So the
alert is a **false positive injected by the test suite into the production journal**.
That is itself a real, high-severity bug (P1 below) because CLAUDE.md's mandatory
startup check and the `PF-FixAgentDispatcher` task both treat unresolved
`critical_errors.jsonl` entries as actionable — phantom criticals trigger false
startup-check failures and spurious auto-spawned fix agents.

Separately, the backup-recovery design in `portfolio_mgr.py` has a real
data-loss flaw that explains how a *genuine* "no backup recovered" could occur
(P0 below) — the rotation clobbers the last good backup with corrupt content.

---

## Critical (90-100)

- **[P0] portfolio/portfolio_mgr.py:50-68 (`_rotate_backups`) + :183-188 / :211-234** —
  Backup rotation **destroys the last good backup when the live file is already
  corrupt**, which is the exact mechanism that yields "unparseable and no backup
  recovered". Flow: `_save_state_to`/`update_state` call `_rotate_backups(path)`
  unconditionally before the atomic write. `_rotate_backups` copies the *current*
  on-disk file to `.bak` (`shutil.copy2(path, .bak)`) **without checking that the
  current file is valid JSON**. The documented 2026-06-01 trigger is a hand-edit
  that leaves the file unparseable while the loop is running. Sequence: (1) external
  edit corrupts `portfolio_state.json`; (2) before the next *save*, some path does a
  lockless `load_state()` → corruption detected, but at that moment no save has run
  so backups may still be good; (3) the next `update_state` for an unrelated mutation
  runs `_rotate_backups`, which copies the **corrupt** current file over `.bak`
  (and cascades `.bak`→`.bak2`→`.bak3` over the rotation horizon), then writes fresh
  defaults. After `_MAX_BACKUPS` rotations every backup holds corrupt-or-default
  content and the real portfolio is gone. **Fix:** in `_rotate_backups`, validate the
  source parses as a dict (`load_json(..., default=None) is not None`) before copying
  it to `.bak`; if it does not parse, SKIP rotating it forward so the last-known-good
  `.bak` is preserved. Equivalently, only rotate inside `_save_state_to` when the
  state about to be written is itself valid (it always is — it's the in-memory dict),
  and never overwrite a good `.bak` with an unparseable current file.

## Important (80-89)

- **[P1] tests/test_portfolio_mgr_core.py:279, 295, 377, 483 (and test_io_safety_sweep.py:199,
  test_portfolio_mgr.py:133) → portfolio/portfolio_mgr.py:84-138 / :175** —
  Test-isolation defect that **writes phantom `critical`/`portfolio_state_corrupt`
  entries into the real `data/critical_errors.jsonl`**. These tests patch
  `portfolio_mgr.STATE_FILE` to `tmp_path` but do **not** patch
  `portfolio_mgr.CRITICAL_ERRORS_LOG`. When `_load_state_from` reaches the
  "all backups corrupt" branch, `_quarantine_corrupt_state` writes the corrupt bytes
  to tmp_path (fine) but `atomic_append_jsonl(str(CRITICAL_ERRORS_LOG), …)` appends to
  the **module-level live constant** (`DATA_DIR / "critical_errors.jsonl"`). CLAUDE.md
  STARTUP CHECK runs `scripts/check_critical_errors.py` which exits non-zero on
  unresolved criticals in the last 7 days, and `PF-FixAgentDispatcher` auto-spawns a
  Claude fix agent on unresolved entries — so every test run now triggers false
  startup failures and spurious fix-agent spawns. The repo rule "Tests using
  module-level file paths must patch to `tmp_path` for xdist safety"
  (`.claude/rules/testing.md`) is violated. **Fix:** add a `monkeypatch.setattr(pm,
  "CRITICAL_ERRORS_LOG", tmp_path / "critical_errors.jsonl")` fixture to every test that
  drives `load_state()`/`_load_state_from` with corrupt input (the
  `test_portfolio_mgr_corrupt_quarantine.py::journal` fixture is the correct pattern to
  copy). Resolve the polluting entries with the `resolution` follow-up lines per CLAUDE.md.

- **[P1] portfolio/portfolio_mgr.py:191-193 / :228 (lockless `load_state` vs locked
  writer on Windows)** — `load_state()`/`load_bold_state()` call `_load_state_from`
  with **no lock**, while `update_state`/`_save_state_to` hold the per-file lock and do
  `os.replace(tmp, path)`. On Windows `os.replace` fails with `PermissionError` if the
  destination is held open by a concurrent reader, and a reader can also observe the
  brief window around the replace. The reader's `load_json` swallows `OSError`/parse
  errors to `default=None` (file_utils.py:86-94), and `_load_state_from` then treats
  `None` as **corruption** — entering the backup-recovery path and (per P0) eventually
  the quarantine/critical branch. With two loop processes plus the dashboard and the
  journal module all calling `load_state()`/`_load_portfolio_pnl()` lockless, a transient
  read failure during a write masquerades as corruption. The thread-level `_get_lock`
  does nothing across the two real OS processes. **Fix:** make the corruption verdict
  require a *re-read after a short backoff* (load_json returned None twice in a row)
  before declaring corruption, and/or acquire the per-file lock on the read path. At
  minimum, distinguish `OSError` (transient/locked — retry, do not quarantine) from
  `JSONDecodeError` (genuine corruption) in `_load_state_from` instead of collapsing
  both into `loaded is None`.

## Low (51-75)

- **[P2] portfolio/file_utils.py:269-292 (`atomic_append_jsonl`) cross-process lock
  vs the non-locked `open(path,"ab")` writers** — `atomic_append_jsonl` correctly
  serializes via `jsonl_sidecar_lock` (msvcrt/fcntl, cross-process). Good. But the
  guarantee only holds if **all** writers to a given JSONL go through it. `O_APPEND`
  on Windows does not give the same atomic-append guarantee as POSIX for writes larger
  than a sector, and any code path that opens the same file directly (outside this
  helper) is unserialized against it. This is a latent torn-line risk for high-rate
  logs (`signal_log.jsonl`, `telegram_messages.jsonl`). Within scope the writers are
  disciplined; flagging so future writers are required to use the helper, not raw
  `open(..., "a")`. **Fix:** none required now; document the invariant at the top of
  `file_utils.py` and consider a lint check.

- **[P2] portfolio/file_utils.py:74-94 (`load_json`) silently maps locked/transient
  reads to `default`** — `load_json` swallows `OSError` (incl. Windows
  `PermissionError` from a file locked by AV or a concurrent writer) and returns
  `default`. For most callers that degrade gracefully this is fine and intended, but
  for `portfolio_mgr._load_state_from` it is the trigger that turns a transient lock
  into a "corruption" verdict (see P1 above). The two concerns are coupled; fixing P1
  removes the sharp edge. No change needed in `load_json` itself.

- **[P2] portfolio/health.py:64-86 (`heartbeat`) — heartbeat freshness does not
  imply work progress** — `heartbeat_keepalive` ticks `last_heartbeat` on a daemon
  thread every 60s for the entire duration of a blocking Layer 2 subprocess (up to
  900s). `check_staleness` (300s gate) and `/api/health` therefore report
  **"healthy" even if the loop is wedged inside a hung subprocess** — the keepalive
  thread keeps writing while no cycle work advances. The design comment acknowledges
  this is intentional ("the loop is alive, just waiting"), and `check_agent_silence`
  (2h/4h gate) is the intended backstop for a stuck agent. So this is a known
  trade-off, not a regression. Residual gap: a hang in the *main cycle* path that is
  NOT wrapped by `heartbeat_keepalive` and NOT a Layer 2 invocation would correctly go
  stale, so the false-OK is bounded to the keepalive-wrapped window. **Fix (optional):**
  pair the heartbeat with a monotonically-increasing `cycle_count` freshness check so
  the dashboard can show "alive but no cycle completed in N min" distinctly from
  "heartbeat stale".

- **[P2] portfolio/shared_state.py:270-287 (`_RateLimiter.wait`) under-throttles on
  a cold limiter** — `last_call` initializes to `0.0`. On the first call,
  `elapsed = now - 0.0` is enormous so `wait_time = 0` and `last_call = now`. Fine.
  But the reservation math `self.last_call = self.last_call + interval if wait_time > 0
  else now` means once the bucket goes idle (elapsed > interval), the next burst resets
  to `now` and N threads can each see `wait_time == 0` in the same instant before any of
  them advances `last_call` — they read `last_call` under the lock sequentially, so the
  first sets `last_call=now`, the second sees `elapsed≈0 < interval` and reserves
  `now+interval`. So it is actually serialized correctly under the lock. No bug; the
  reservation-before-sleep pattern (BUG-212) is sound. Noting only that a long idle gap
  permits a single immediate call (intended token-bucket behavior).

## Notes — verified SAFE (no finding)

- **dashboard/auth.py + cf_access.py — CF-Access header bypass is properly closed.**
  `require_auth` no longer trusts `Cf-Access-*` headers by presence; it calls
  `verify_cf_jwt`, which verifies the RS256 signature against the team JWKS, requires
  `exp/iat/aud/iss`, checks `aud`==configured tag and `iss`==`https://<team_domain>`,
  and asserts the header email equals the signed `email` claim (cf_access.py:99-166).
  Missing config → `None` → fail-closed fall-through. A forged LAN/Tailscale header
  cannot bypass auth. Cold-start config read failure fails CLOSED (503) via
  `_config_is_known()` (auth.py:114-118, 161-174). Token comparisons use
  `hmac.compare_digest`.
- **dashboard/app.py — every API route is `@require_auth`.** Verified all `@app.route`
  handlers (summary, portfolio, trades, decisions, avanza_account, system/trading
  status, crypto/eth/btc/mstr/oil/metals/golddigger, etc.) carry `@require_auth`.
  `/logout` is intentionally bare (only wipes a cookie). The Flask default static
  handler is unauthenticated, but `house_blueprint.py` documents + unit-tests that no
  sensitive asset is ever placed/symlinked under `dashboard/static/`.
- **dashboard/app.py:64-72 — `/house/heatmap` same-origin framing change is safe.**
  `X-Frame-Options: SAMEORIGIN` + CSP `frame-ancestors 'self'` only on that one
  auth-gated, read-only route; every other route keeps `DENY` / `'none'`. The `/house`
  hub embeds it via a same-origin `<iframe>`. No clickjacking exposure beyond the
  operator's own origin.
- **dashboard/house_blueprint.py — path-traversal guarded.** `_validate_run_id`
  (regex) and `_validate_slug` (`secure_filename` + regex) gate every filesystem
  component; all routes `@require_auth`; user content goes through `escape()`/markdown.
- **portfolio/gpu_gate.py — cross-process lock is correct.** `O_CREAT|O_EXCL` atomic
  create avoids TOCTOU; stale-break is gated on `_is_stale()` AND owner pid dead
  (`_pid_alive` fails safe to "dead" when psutil missing, but only after the stale-mtime
  check). Sweeper daemon is idempotent and exception-swallowing. Re-entry by same pid is
  handled. No VRAM double-load vector found.
- **portfolio/telegram_notifications.py + message_store.py — swallowed send failures
  are appropriate.** Send failures return `False`/log WARNING and never raise into the
  loop (correct for best-effort alerts). `_do_send_telegram` distinguishes a real
  config error (missing token/chat_id → WARNING + `False`) from a transient send
  failure, so a misconfig is observable in logs rather than hidden. Markdown-parse 400s
  fall back to plain-text resend.
- **portfolio/http_retry.py — redacts bot tokens in logs (`_redact_url`), honors
  `retry_after` on 429, separates fatal (4xx) from retryable status. Sound.**
- **portfolio/journal.py — append/build path is read-mostly; the only write is
  `atomic_write_text(CONTEXT_FILE, …)` (atomic). No append-only-integrity violation.**

## Summary

The headline "live corruption" alert is a **false alarm**: production state is intact;
the `portfolio_state_corrupt` journal lines were injected by unit tests that fail to
patch `CRITICAL_ERRORS_LOG` (**P1**) — this pollutes the journal that the mandatory
startup check and the auto-spawn fix-agent dispatcher read. However, the audit found a
genuine **P0** data-loss path in `portfolio_mgr._rotate_backups`: it copies the current
on-disk file to `.bak` without validating it, so once the live file is corrupt
(e.g. a hand-edit, the documented 2026-06-01 trigger) subsequent rotations overwrite the
last good backups with corrupt content, producing exactly "no backup recovered" and a
silent wipe-to-defaults. A related **P1** is that the read path (`load_state`) is
lockless and conflates Windows `PermissionError`/transient locked-file reads with
genuine `JSONDecodeError` corruption, so a concurrent-write race can falsely trip the
corruption/quarantine machinery. The dashboard auth surface (CF-Access JWT verification,
fail-closed cold start, per-route `@require_auth`, the `/house/heatmap` framing change,
path-traversal guards) is solid — no auth-bypass or data-leak route found. gpu_gate,
http_retry, telegram/message_store, and the file_utils atomic writers are correct.

Priority order: fix `_rotate_backups` to never clobber a good backup with an unparseable
file (P0); patch `CRITICAL_ERRORS_LOG` in the corrupt-state tests + append `resolution`
lines for the phantom criticals (P1); harden `_load_state_from` to distinguish transient
OSError from JSONDecodeError and/or take the lock on read (P1).
