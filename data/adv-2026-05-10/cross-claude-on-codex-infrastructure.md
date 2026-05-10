# Claude critique of codex findings — infrastructure

## Verdicts

- [P1] Ship the dashboard helper modules before importing them — dashboard/app.py:736-742
  Verdict: FALSE-POSITIVE
  Reason: dashboard/auth.py and dashboard/house_blueprint.py both exist in the repo; the codex review ran in a worktree with permission issues that prevented git status from running correctly.

- [P1] Add the missing `portfolio.tickers` module — portfolio/journal.py:11
  Verdict: FALSE-POSITIVE
  Reason: portfolio/tickers.py exists and imports successfully. The codex worktree git command failed due to permission errors early in the review.

- [P1] Don't advance the Telegram offset past failed commands — portfolio/telegram_poller.py:160-161
  Verdict: FALSE-POSITIVE
  Reason: The code correctly avoids persisting offset on raised exceptions. In-memory offset advances unconditionally (line 161) but persistence only happens at line 265 if `should_persist` is True, which requires `drop_reason NOT in _SETTLED_DROP_REASONS`. Exception handlers set `drop_reason = f"raised:{type(exc).__name__}"` (line 252), which is explicitly not a "settled" reason, so persistence is skipped and the persisted offset stays at the last successful message. Restart re-fetches the failed update (see line 166 comment).

- [P1] Avoid replacing the symlinked `config.json` on `/mode` — portfolio/telegram_poller.py:361
  Verdict: CONFIRMED
  Reason: Line 361 calls `atomic_write_json(config_path, cfg)` which uses `os.replace(tmp, str(path))` on line 59 of file_utils.py. This breaks the symlink to C:\Users\Herc2\.config\finance-analyzer\config.json. First `/mode` command severs the link and writes a divergent local copy.

- [P2] Start `_cached()` TTL after the fetch completes — portfolio/shared_state.py:100
  Verdict: PARTIAL
  Reason: Line 48 captures `now = time.time()` before the function call (line 93), and line 100 stores this pre-call timestamp. For calls longer than TTL, the entry is stale on return. However, the consequence is not "dogpile protection fails"—the comment at line 99-100 explicitly documents this is intentional: avoid caching None results that indicate transient failures. The real issue is that a 10-minute LLM inference with TTL=300 will trigger recomputation by every subsequent caller within those 10 minutes, defeating cache value. This is a performance cliff (P2), not total dogpile failure.

## New findings (mine)

- [P1] Offset save sequence creates window for update loss — portfolio/telegram_poller.py:185-186, 264-265
  The code saves offset in two places: (1) if message has no text body (line 185-186, called from within the message handler before dispatch), and (2) at end of finally block (line 264-265). If a message passes stale/empty/parse checks but fails during dispatch (raising exception), the in-memory offset has already advanced (line 161) and will be persisted by the NEXT message that succeeds (line 264). This is actually correct per the comment, but the split persistence paths (line 186 vs 264-265) make the logic fragile and hard to audit. The intent would be clearer if all offset persistence went through a single guarded path.

- [P1] config.json symlink write — portfolio/telegram_poller.py:361 + portfolio/file_utils.py:59
  Symlink is clobbered by atomic_write_json. Fix: Before writing, check `os.path.islink(config_path)` and if true, either (a) update the target file directly instead of the symlink path, or (b) use `fcntl.flock` + direct write + fsync on Unix, or `msvcrt.locking` on Windows. See file_utils.py lines 12-19 for cross-platform lock examples.

## Summary

- Confirmed: 1
- Partial: 1  
- False-positive: 3
- New: 2
