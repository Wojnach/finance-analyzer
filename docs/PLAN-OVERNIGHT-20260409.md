# Overnight Work Plan — 2026-04-09 (Fleet v2 follow-ups + log migration)

**Branch**: `fgl/overnight-log-migration-20260409`
**Worktree**: `/mnt/q/finance-analyzer-overnight`
**Protocol**: `docs/GUIDELINES.md` (`/fgl` — plan → batches → test → codex → ship)
**Driver**: user's "im going to bed u got full permission" — autonomous
overnight session following the /fgl after-hours protocol.

## Context

Today's afternoon Fleet v1 shipped 7 commits fixing cash-sync dormancy +
noise suppression. Evening Fleet v2 shipped 3 more (get_buying_power
multi-shape, golddigger TNX → FRED fallback, get_cet_time dedup) plus a
read-only audit of the log()/_log() → logging migration (Agent D,
`docs/LOG_MIGRATION_AUDIT_20260409.md`).

Two concrete follow-ups were left on the table:

1. **Fleet v2 Agent A contract change** — `portfolio.avanza_session.get_buying_power()`
   now returns `dict | None` (was always `dict`). Two scripts call
   `get_buying_power().get('buying_power', 0)` without a None guard and
   will now crash with AttributeError on a failure path.
2. **Log migration stages 1-3** — Agent D recommended a 6-stage partial
   migration; stages 1-3 are zero-risk and high-value (shim + ERROR sites
   get free stack traces).

This plan ships both. Nothing else.

## Phase 0 result (daily review, already done)

- `health_state.json`: all 24 signals 100% healthy, 60K+ calls, 0
  failures, error_count=0. Heartbeat fresh.
- `layer2_journal.jsonl`: last entry 2026-04-02 (week ago — system is
  quiet, no recent trade triggers).
- `metals_trades.jsonl`: last trade 2026-03-05 (month ago — confirms
  today's 3h40m dormancy was long-standing).
- **No incidents to respond to.** The plan is purely engineering
  follow-ups.

## Scope

| # | Batch | Files | LOC delta | Risk |
|---|---|---|---|---|
| 1 | Fish script None-guards | `scripts/fish_straddle.py`, `scripts/fish_monitor_live.py` | ~30 | LOW |
| 2 | Log migration Stage 1 (shim) | `data/metals_loop.py`, `data/metals_swing_trader.py` | ~30 | LOW |
| 3 | Log migration Stage 2 (swing_trader ERROR) | `data/metals_swing_trader.py` | ~20 | LOW |
| 4 | Log migration Stage 3-A (metals_loop order+emergency ERROR) | `data/metals_loop.py` | ~50 | LOW-MED |
| 5 | Log migration Stage 3-B (metals_loop catch-all ERROR) | `data/metals_loop.py` | ~80 | LOW-MED |

Total: ~210 LOC across 4 files, 5 commits.

## Batch details

### Batch 1 — Fish script None-guards

**Problem**: Fleet v2 Agent A changed `get_buying_power()` return type
from `dict` (empty on failure) to `dict | None`. Two call sites were
left unguarded:

- `scripts/fish_straddle.py:174` inside a try/except block — the except
  would catch the AttributeError but silently eat the error.
- `scripts/fish_monitor_live.py:431` inside `_try_fish()` with no
  surrounding try/except — direct crash on AttributeError.

**Fix** — replace the pattern:
```python
bp = float(get_buying_power().get('buying_power', 0))
```
with:
```python
_bp = get_buying_power()
if _bp is None:
    log_msg("WARNING: get_buying_power() returned None — Avanza session may need refresh, aborting")
    return  # (or return None for fish_monitor_live's _try_fish)
bp = float(_bp.get('buying_power', 0))
```

**Rationale**: fail loud with a descriptive message, return early so we
don't trade off zero cash. The previous pattern would have either crashed
(fish_monitor_live) or silently continued with 0 budget (fish_straddle's
broad except). Either way the fish session dies silently — we want it to
die loud so the user sees the problem on Telegram or in the log tail.

**Test plan**: no existing tests touch these files per grep. Smoke-test
by importing each module and confirming no syntax/import errors.

### Batch 2 — Log migration Stage 1 (shim infrastructure)

**Goal**: zero call-site churn. `log()` and `_log()` become thin shims
that delegate to Python logging. All existing 290 `log()` sites in
metals_loop.py and 48 `_log()` sites in metals_swing_trader.py keep
working unchanged. Output format changes cosmetically from
`[HH:MM:SS] msg` to `[HH:MM:SS] [INFO] msg`. The `[SWING]` tag in
swing trader is preserved by embedding it in the message body.

**metals_loop.py changes**:
1. Add `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` +
   same for stderr, wrapped in `try/except (AttributeError, OSError)`
   for non-tty streams. Place at top, before any imports that might
   print at module load.
2. Add `logging.basicConfig(force=True, level=logging.INFO,
   format='[%(asctime)s] [%(levelname)s] %(message)s',
   datefmt='%H:%M:%S')` after the existing `logger = getLogger(...)`
   assignment at line 48.
3. Change `log(msg)` at line 717 from `_safe_print(f"[{ts}] {msg}")`
   to `logger.info(msg)`. The `[HH:MM:SS]` prefix is now added by
   the basicConfig datefmt.
4. Leave `_safe_print` direct calls at lines 714 and 757 alone —
   they're not going through `log()`, and touching them is Stage 2
   scope.

**metals_swing_trader.py changes**:
1. No basicConfig needed — same process as metals_loop.py, inherits
   the root handler.
2. Change `_log(msg)` at line 111 from `print(f"[{ts}] [SWING] {msg}",
   flush=True)` to `logger.info(f"[SWING] {msg}")`. The `[SWING]` tag
   stays in the message body for grep-compatibility.

**Smoke test**:
```
cd /mnt/q/finance-analyzer-overnight && .venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'data')
from metals_loop import log
log('Stage 1 shim smoke test')
"
```
Expected output: `[HH:MM:SS] [INFO] Stage 1 shim smoke test`

**Ruff**: check touched files, expect 0 new violations vs baseline.

**Risk control**: if the new format breaks any human log-tailing
workflow (fin-logs skill, etc.), the rollback is a single 2-line git
revert of the shim changes. The existing 43 `logger.warning/debug`
observability calls from Fleet v1 already produce the new format — so
the format shift is already partially visible. Full shift is a minor
cosmetic change.

### Batch 3 — Log migration Stage 2 (swing_trader ERROR sites)

**Target**: ~10 ERROR sites in `data/metals_swing_trader.py`. Convert
`except Exception as e: _log(f"<msg>: {e}")` patterns to
`except Exception: logger.exception("<msg>")` — same output plus a
free stack trace on the except path.

Per Agent D's audit, the sites are around:
- L160 (state save), L210 (Telegram send), L221 L228 (decision log)
- L452, L1352 (sell cancel failed)
- L924 (buy failed), L1329 (sell failed)
- L1033 (stop-loss failed)
- L1075 (corrupt position dropped)
- L1186 (exit optimizer error)

**Approach**: grep for `_log(.*error\|_log(.*FAIL\|_log(.*fail` at
current HEAD. Read each site's surrounding context to confirm it's
inside an except block. Convert one-by-one.

**Preserving existing behavior**:
- Keep the same log message TEXT so grep-based searches still work.
- `logger.exception()` adds `Traceback (most recent call last):` to the
  output — this is new, but is exactly the value of this batch.
- Keep `[SWING]` prefix in messages that had it (even though it's
  redundant after batch 2's shim adds it automatically) — because grep
  patterns may still rely on it.

**Test**: re-run the smoke test from Batch 2 + verify metals_swing_trader
imports cleanly after the edits.

### Batch 4 — Log migration Stage 3-A (metals_loop emergency + orders)

**Target**: ~15 high-value ERROR sites in metals_loop.py grouped by
subsystem: emergency sell (L2865, L2876, L2893, L2918) + order failures
(L1671, L2242, L3016, L3022, L3682, L3733, L3825, L3843, L3965).

**Approach**: same as Batch 3 — grep + read + convert. Target sites are
where a catch-all `except Exception as e: log(f"... FAILED: {e}")`
loses the stack trace.

**Safety**: metals_loop.py is the biggest file (6552 lines). I will:
- Edit one subsystem at a time (emergency block first, then orders)
- Re-grep after each subsystem to catch any sites I missed
- Smoke-import after every 5 edits

**Commit granularity**: one commit for the full batch (all ~15 edits
together). The sites are narrow and independently reviewable.

### Batch 5 — Log migration Stage 3-B (metals_loop catch-alls)

**Target**: remaining ~30 generic catch-all ERROR sites scattered
through metals_loop.py.

**Approach**: grep-based identification of `log\(f.*[Ff]ail|log\(f.*[Ee]rror|log\(f.*FATAL`
then manual filter to sites inside except blocks.

**Rules for deciding exception vs error**:
- Inside `except Exception as e:`: use `logger.exception("msg")` — adds
  stack trace automatically
- Standalone error without exception context: use `logger.error("msg")`
- Anything that could fire in steady state: demote to `logger.warning`
  or leave as `logger.info`

**Size control**: if this batch exceeds 100 edited lines, stop at a
clean checkpoint and defer the remainder. Quality over quantity.

### Verify & ship

1. Full test suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto --timeout=60`
2. Confirm no NEW failures vs baseline (26 pre-existing).
3. Ruff on all changed files: 0 new violations.
4. **Codex adversarial review** per /fgl step 5: `/codex:adversarial-review
   --wait --scope branch --effort xhigh` on the branch. Fix valid findings;
   document false positives.
5. FF-merge branch → main.
6. Push via `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`.
7. Restart loops: taskkill existing + `schtasks /run` PF-MetalsLoop + PF-DataLoop.
8. Tail metals_loop_out.txt for 60s post-restart. Confirm new format is
   readable and swing trader initializes.
9. Update `docs/SESSION_PROGRESS.md` with overnight handoff.
10. Clean up worktree + delete branch.

## Known risks + mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| New format breaks human log-reading | MED | Format adds `[INFO]` level prefix only. Rollback = 2-line revert of shim. |
| `_safe_print` bypass still fires | LOW | Leave `_safe_print` direct calls (lines 714, 757) alone in Stage 1. Continue working via stdout. |
| `logger.exception()` adds massive trace spam | LOW-MED | Only fires on except path (already an error condition). If noise, downgrade to `logger.error(msg, exc_info=False)`. |
| basicConfig conflicts with imported modules | LOW | `force=True` overrides. Must be applied early. |
| Sites from Agent D listed are off by N lines | LOW | Grep-based identification at current HEAD; don't trust numbers blindly. |
| Unicode stdout reconfigure fails on non-tty | LOW | Wrapped in try/except. Fallback is _safe_print which still exists. |
| Test suite surfaces a pre-existing failure I'll blame on this batch | LOW | Compare to baseline count before merging. |

## Out of scope

- **Stage 4-6 of Agent D's migration plan** (DEBUG sites, WARNING sites,
  shim retirement). Defer — those are cosmetic + ergonomic, less urgent.
- **`metals_llm.py` `[LLM]` print migration** — out of scope because
  health_check.py substring-matches on `[LLM] Chronos`/`[LLM] Ministral`.
- **XAU SHORT canary activation** — user-gated.
- **ARCH-18 metals_loop.py decomposition** — multi-session.
- **Duplicate python process mystery** — still not investigated.

## Success criteria

- All 5 batch commits land on main via FF-merge.
- Full test suite: zero new failures vs baseline.
- Metals loop restarts cleanly; first `SwingTrader init:` line appears
  in new `[HH:MM:SS] [INFO] [SWING] SwingTrader init:` format.
- Worktree + branch cleaned up.
- Session progress handoff written.
- Zero live-trading regressions.
