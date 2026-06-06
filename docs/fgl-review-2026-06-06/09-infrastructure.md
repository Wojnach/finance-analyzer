# Adversarial review — infrastructure (2026-06-06)

Reviewer: `caveman:cavecrew-reviewer` (returns text only — captured here verbatim).
Scope: atomic I/O, shared state, health, notifications, logging.

> ⚠ **Cross-critique note (orchestrator):** two of this reviewer's high-severity findings were
> **REJECTED as false positives** after the orchestrator read the cited code. They are kept below
> with the verdict inline so the record is honest. See `01-SYNTHESIS.md` → Cross-critique.

## P0 (as reported)
- `portfolio/file_utils.py:342-346`: ~~P0: `atomic_append_jsonl` self-heal `seek(-1, SEEK_END)` on a
  brand-new (size-0) file is UB → spurious newlines.~~ **REJECTED (false positive):** lines 343 guard
  `if f.tell() > 0:` *before* the backward seek; empty files never reach it.
- `portfolio/telegram_poller.py:312-369`: P0→**RE-GRADED P1**: `/mode` command modifies `config.json`
  (the external secrets symlink) with no exclusive lock; if `load_json` transiently returns `{}` the
  size guard may accept it and overwrite. Real *design* smell — notification mode should not live in
  the secrets file. Key-destruction unconfirmed (size guard). → Move mode to a separate file.

## P1 (as reported)
- `portfolio/shared_state.py:123`: ~~P1: `return` inside the exception handler doesn't release
  `_cache_lock` → deadlock; "context-manager exit is skipped".~~ **REJECTED (false positive):** the
  `return` is inside `with _cache_lock:` (line 111); Python runs `__exit__` on return — lock IS released.
- `portfolio/shared_state.py:208`: P1: `enqueue_fn` called without holding `_cache_lock`; if it raises,
  cleanup of `_loading_keys`/`_loading_timestamps` can race a concurrent `_cached()` → a key can be left
  permanently in `_loading_keys`, blocking all future fetches for that key. → Wrap `enqueue_fn` in
  try/except inside the lock. *(survives — verify.)*
- `portfolio/telegram_poller.py:245-253`: P1: a handler exception is tagged `raised:*` but the `finally`
  block still persists the offset → the user's command can be dropped on restart (marked processed
  without running). → Only persist offset when `outcome["processed"]` is True.
- `portfolio/config_validator.py:66-68`: P1: `validate_config_file` raises with the config path on an
  unreadable external symlink → aborts loop startup with no fallback (and leaks the path). → Degrade or
  log-only; don't expose the path.
- `portfolio/message_store.py:209-215`: P1: non-400 send failures (403/500) log a warning and return
  False with no retry/queue → callers may silently drop the message. → Queue failed messages / escalate
  to `critical_errors.jsonl`.
- `portfolio/health.py:341-343`: P1→**RE-GRADED P2**: newsapi quota reset uses wall-clock `now(UTC)`;
  a backward clock jump can double-reset the quota mid-day. Robustness only.

## P2
- `portfolio/log_rotation.py:386-408`: P2: `rotate_jsonl` holds the sidecar lock while reading the
  entire (68 MB) file → blocks all `atomic_append_jsonl` writers for hundreds of ms (perf cliff, not
  data loss). → Streaming rotation / offset-based.
- `portfolio/shared_state.py:82-87`: P2: stale-while-revalidate returns None when a key is loading and
  no stale data exists → under 8 workers, 7 return None instead of a 1-cycle-old value.
- `portfolio/health.py:152-166`: P2: `check_staleness` parses `last_heartbeat` without holding
  `_health_lock`; a concurrent write window → transient false "stale" alert.
- `portfolio/message_store.py:210` / `telegram_poller.py:371-387`: P2: send status / reply failures not
  persisted with enough metadata to distinguish "never sent" from "network hiccup" from "auth failure".

## Risk summary (orchestrator-adjusted)
After cross-critique the real infra exposure is narrower than reported: the `file_utils` atomic layer
and the `_cached` lock discipline are sound (the two headline P0/P1s were false positives). The genuine
issues are the `/mode`→`config.json` write coupling (P1), the `_loading_keys` stuck-key race on
`enqueue_fn` failure (P1), and telegram offset-persist-on-failure dropping user commands (P1) — the last
is the same silent-failure family as the 3-week auth outage.
