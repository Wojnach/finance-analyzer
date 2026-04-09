# log() / _log() / print() Migration Audit — 2026-04-09

Fleet v2 Agent D deliverable. Read-only audit of custom logging helpers in
the metals subsystem to inform a future `logging`-module migration.

## 1. Summary counts

| File | `log()` sites | `_log()` sites | `print()` sites | Notes |
|---|---|---|---|---|
| `data/metals_loop.py` | **290** (excl. the `def log` at line 718) | — | **22** (excl. `_safe_print`/`log` bodies) | ~6,600 LOC |
| `data/metals_swing_trader.py` | — | **48** (excl. `def _log` at line 111) | 0 | 1,513 LOC |

Both files already have a `logger = logging.getLogger(...)` instance added
2026-04-09 (`metals_loop.py:48`, `metals_swing_trader.py:23`) — currently
only used for the 43 bare-except observability sites from commits `fec1dde`
and `f557f9f`. Not the call sites in scope for this audit.

## 2. Existing helper signatures

**`metals_loop.py:718`** — `def log(msg)`
```python
def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _safe_print(f"[{ts}] {msg}")
```
Produces `[HH:MM:SS] <msg>` to stdout. Routes through `_safe_print()` (line
680) which catches `UnicodeEncodeError` on Windows non-UTF consoles and
falls back to ASCII-replace or `sys.stdout.buffer.write`. **This encoding
safety matters** — naive `logging.info()` with default `StreamHandler` will
raise on WinConsole.

**`metals_swing_trader.py:111`** — `def _log(msg)`
```python
def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [SWING] {msg}", flush=True)
```
Produces `[HH:MM:SS] [SWING] <msg>`. The hard-coded `[SWING]` tag makes
swing-trader output grep-distinguishable in the shared
`metals_loop_out.txt` (swing trader runs in-process inside metals_loop.py).
Does **not** use `_safe_print` — slightly less Unicode-safe than `log()`.

Both timestamps are local-tz wallclock (no explicit tz) via
`datetime.datetime.now()`.

## 3. Categorization by intended level

### INFO-level — routine operational output (~195 sites)
- **metals_loop.py (~170)**: startup banner (5791-5815, ~25), cycle
  heartbeat (6456), detect_holdings (458-510, 1511-1562, ~20), stop order
  placement (1669-1682, 3841-3872, 3962-3974, ~15), fish engine
  (1875-2425, ~25), spike catcher (4263-4504, 6263-6330, ~15), trade queue
  (3550-3825, ~15), session state (3429, 3820-3825, 5820, 5835-5845, ~15).
- **metals_swing_trader.py (~25)**: init (276), cash sync (316, 334),
  catalog refresh (362), position lifecycle (502 FILL VERIFIED, 880
  Selected, 896 BUY, 1015 Setting stop, 1031 Stop placed, 1288 SELL),
  dry-run stubs (918, 1019, 1323).

**Migration risk: LOW-MED.** Format shift from `[HH:MM:SS] msg` to stdlib
default would change human readability but not any machine parser.

### WARNING-level — degraded / retrying (~50 sites)
- **metals_loop.py (~35)**: session dead/stale (3438, 3459, 5824), price
  fetch fallbacks (485, 1124, 1161, 1173, 1182, 1224, 1261, 5845), stale
  microstructure/signal (1230, 1783-1849), fish/spike rollbacks
  (2258-2262, 4424-4493), daily range/seasonality missing (5907-5924),
  trade queue empty/stale (5989-6015, 3613, 3628), trailing stop skipped
  (1634, 3921), SHORT L1 warnings (5172), fish ORB/vol_scalar non-fatal
  (2348, 2379).
- **metals_swing_trader.py (~15)**: catalog refresh failure (291, 296,
  299, 364), reconciliation failure streak (417), phantom position
  cleared (441, 452), unfilled rollback (513, 521, 529), entries paused
  (581, 588), no valid warrant (626, 864), insufficient cash (663),
  barrier/spread gates (829, 832, 875), candidate scoring skip (875).

**Migration risk: MED.** Highest value — level routing + future alerting.

### ERROR-level — real failures (~55 sites)
- **metals_loop.py (~45)**: emergency sells (2865, 2876, 2893, 2918),
  order failures (1671, 2242, 3016, 3022, 3682, 3733, 3825, 3843, 3965),
  news fetch errors (1783, 1817), generic catch-all errors (418, 506,
  629, 845, 1399, 1562, 1616, 1722, 2178, 2249, 2852, 3403, 3471, 3520,
  4063, 4222, 4922, 4993, 5528, 5634, 6141, 6157, 6202, 6210, 6525, 6545,
  6552, 6558, 6565, 6572), fatal crash (6579), Claude CLI failures
  (5669-5770), force shutdown (5785).
- **metals_swing_trader.py (~10)**: state save (160), Telegram (210),
  decision log (221, 228), buy/sell failed (924, 1329), stop-loss failed
  (1033), sell cancel failed (452, 1352), corrupt position dropped
  (1075), exit optimizer error (1186).

**Migration risk: LOW**, **highest value**. Free stack traces via
`logger.exception()`. The `"!!!"` prefix convention is the closest thing
to a "severe" tag in the current system.

### DEBUG-level — verbose diagnostics (~20 sites)
- **metals_loop.py (~18)**: position verify inner loop (462, 466, 468,
  490, 495, 499, 504), warrant catalog dump (6004-6015), per-check cycle
  verbose (6135), signal tracker verbose (5940, 5943), fish Kelly
  verbose (2144, 2149, 2157).
- **metals_swing_trader.py (~2)**: candidate scoring diagnostic
  (829, 832).

**Migration risk: LOW.** Clearly DEBUG, fires unconditionally every
cycle, bloats the log file. Biggest silenceable-in-production win.

## 4. Downstream consumers

**Machine parsers:**
- `scripts/health_check.py:227` — tails last 200 lines of
  `metals_loop_out.txt` and substring-matches `[LLM] Chronos` and
  `[LLM] Ministral`. **CRITICAL**: emitted by `metals_llm.py:100` (an
  imported module), NOT by `metals_loop.py:log()` or
  `metals_swing_trader.py:_log()`. `metals_llm.py` has zero `log(` calls.
  **⇒ Migrating `log()`/`_log()` in the two in-scope files is transparent
  to health_check**, as long as metals_llm.py's `[LLM]` print is untouched.
- `scripts/fish_preflight.py:131` — reads `metals_signal_log.jsonl`
  (separate JSONL). Not affected.

**Human parsers (substring):**
- `.claude/commands/fin-logs.md` — Step 3 reads "last 50 lines of
  `data/metals_loop_out.txt`" for "errors", "failures", "Stop-loss API
  errors". Human reading, not positional. Would survive any format shift.
- `docs/STOP_LOSS_SETUP.md`, `docs/operational-runbook.md`,
  `docs/SYSTEM_HEALTH_CONTRACT.md`, `docs/GUIDELINES.md` — file path
  mentions for operators to `tail`. No parsing.

**Dashboard:**
- `dashboard/app.py` does **not** read or parse `metals_loop_out.txt`.
  Reads structured JSON files only.

**Conclusion:** only hard dependency is `health_check.py` on
`[LLM] Chronos`/`[LLM] Ministral` substrings, emitted by a *different*
file (`metals_llm.py`) which is out of scope. **Both in-scope files can
be migrated without breaking any machine consumer.**

## 5. Gotchas (high-friction migration points)

1. **`_safe_print` Unicode wrapper** (`metals_loop.py:680`) — current
   `log()` handles Windows non-UTF console crashes. Naive migration to
   `logger.info()` with default `StreamHandler` raises
   `UnicodeEncodeError`. **Mitigation**: custom `StreamHandler` subclass
   with the same fallback, OR `sys.stdout.reconfigure(encoding='utf-8',
   errors='replace')` at startup. Required for both files.

2. **`[SWING]` tag convention** (`metals_swing_trader.py:113`) — swing
   trader output currently distinguishable in combined stdout via the
   hard-coded `[SWING]` tag. Stdlib migration would need
   `%(name)s` in the format string to preserve this. Operators trained on
   `grep SWING metals_loop_out.txt` need to retrain.

3. **`!!!` prefix convention** for alerts (`metals_loop.py:2865, 4222,
   6378, 6423`) — triple-bang prefix stands out in stdout tailing. If
   migrated, the `WARNING`/`ERROR` keyword in the format string replaces
   it. Don't silently drop the `!!!` without a level-prefix replacement.

4. **Timestamp format difference** — current `[HH:MM:SS]` local-time;
   stdlib default is `2026-04-09 18:56:12,345`. Preserve compactness with
   `fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'`.

5. **Interleaving with `metals_llm.py` print()** — out-of-scope file but
   writes to the same stdout with format `[HH:MM:SS] [LLM] msg`. Post-
   migration the file will contain mixed formats. Not a breakage but
   visual inconsistency. Consider migrating metals_llm.py too.

6. **`.bat` redirect vs RotatingFileHandler** — currently `metals-loop.bat:11`
   does `> metals_loop_out.txt 2>&1`, overwriting each process start. If
   switching to `logging.handlers.RotatingFileHandler`, the .bat redirect
   interaction needs thought (two writers, or pure-Python file logging
   with no .bat redirect).

## 6. Recommendation — partial migrate in stages

**Zero machine parsers depend on `log()`/`_log()` format.** Both files
already have `logger` instances wired. Error-site migration is strictly
better (free stack traces, level routing, future alerting). DEBUG-level
sites currently fire unconditionally and would benefit hugely from being
silenceable in production.

### Suggested order (lowest risk → highest value)

1. **Stage 1 — Infrastructure** (~1 hour, ~30 LOC)
   - Add custom `UnicodeSafeStreamHandler` OR `sys.stdout.reconfigure`.
   - `logging.basicConfig(level=INFO,
     format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
     datefmt='%H:%M:%S')` at startup.
   - Keep `log()` and `_log()` as thin shims delegating to `logger.info()`.
   - **Zero call-site churn.** Ship, run 24h, verify log-readability.

2. **Stage 2 — ERROR sites in swing_trader** (~10 sites, ~30 min)
   - `_log(f"State save error: {e}")` → `logger.exception("State save error")`.
   - Highest value per edit.

3. **Stage 3 — ERROR sites in metals_loop** (~45 sites, ~2h)
   - Same transformation. Grouped by subsystem for clean PRs.

4. **Stage 4 — DEBUG sites** (~20 sites, ~1h)
   - Move to `logger.debug()`. Toggle via
     `logger.setLevel(os.getenv("METALS_LOG_LEVEL", "INFO"))`.
   - Potentially **30-40% fewer lines** in metals_loop_out.txt.

5. **Stage 5 — WARNING sites** (~50 sites, ~1.5h)
   - `log("WARNING: ...")` → `logger.warning("...")`.

6. **Stage 6 (optional) — Retire shims** (~30 min)
   - Bulk `log(` → `logger.info(`, `_log(` → `logger.info(`.
   - Only after stages 1-5 prove stable.

### Effort estimate
**~6-8 hours across multiple sessions, or one focused day.** Medium-large.

### Blockers to resolve before migration

1. **Confirm nothing in production tooling greps `metals_loop_out.txt`
   for `[fish]`/`[stops]`/`[spike]`/`[SWING]` tags.** Repo grep shows
   none, but external tools (cron, log-ship, ELK) could. Ask user before
   touching tags.

2. **Decide whether to migrate `metals_llm.py`'s `print()` in the same
   session.** Its `[LLM] Chronos`/`[LLM] Ministral` lines ARE parsed by
   `health_check.py`. If migrated, health_check needs update.

3. **Verify `metals-loop.bat` log rotation strategy** if adding
   `RotatingFileHandler`.

## Verdict

**Yes, migrate — in stages.** Start with Stage 1 (shim) for zero-risk
rollout, then Stage 2-3 (ERROR sites) for immediate value. Classic
maintenance win: no machine consumers threatened, infrastructure already
wired, payoff is free stack traces + level-based log size control. The
`_safe_print` Windows Unicode wrapper is the one real gotcha — handle in
Stage 1 or the migration reintroduces a crash class.

## Key file paths referenced
- `data/metals_loop.py` (helpers: `log` L718, `_safe_print` L680; `logger` L48)
- `data/metals_swing_trader.py` (helper: `_log` L111; `logger` L23)
- `data/metals_llm.py` (out-of-scope but emits `[LLM]` lines at L100)
- `scripts/health_check.py:227` (only machine parser — targets
  metals_llm.py substrings, not in-scope files)
- `scripts/win/metals-loop.bat:11` (stdout redirect)
- `.claude/commands/fin-logs.md` (human reader, substring-based)
