# FGL Self-Review Pass (2026-05-23)

Main-thread independent adversarial pass running in parallel with the 8
subagents. Goal: find issues the subagent partition missed, AND have an
independent voice for cross-critique.

Method: targeted greps for known dangerous patterns + spot reads of the
hottest dispatch / state-mutation paths.

---

## Independent Findings

### S-1 — Bare `except Exception: pass` swallowing `is_promoted` failure
File: `portfolio/signal_engine.py:3602-3606`

```python
_promoted_override = False
try:
    from portfolio.shadow_registry import is_promoted
    _promoted_override = is_promoted(sig_name)
except Exception:
    pass
```

If `shadow_registry.json` is corrupted, partially-written, or the import
fails (broken venv after pip update), `_promoted_override` silently
defaults to `False`. A signal that operator/user explicitly promoted goes
back to force-HOLD with zero telemetry. Same silent-failure class as the
3-week auth outage — exit 0 + no warning.

**Severity:** P1.
**Fix:** `except (ImportError, json.JSONDecodeError, OSError) as e:
logger.warning("is_promoted check failed for %s: %s", sig_name, e)` —
narrow + log.

### S-2 — `_streaming_max` rotation detection trusts byte size only
File: `portfolio/risk_management.py:60-83`

```python
file_size = history_path.stat().st_size
if file_size >= cached["offset"]:
    start_offset = cached["offset"]; peak = cached["peak"]
else:
    # File shrank (rotation) — full re-scan
```

The rotation check is `file_size >= cached_offset`. If
`portfolio_value_history.jsonl` is rotated to a new file that happens to
have **more bytes** than the previous file at the moment of rotation (e.g.
log-rotation rewrites the tail with 5000 entries while the cached offset
was at byte 4500), the cache's `start_offset=4500` reads halfway into a
DIFFERENT JSON object → `json.JSONDecodeError` → silently skipped, peak
underestimated. Very narrow race but it exists.

**Severity:** P3.
**Fix:** store an inode/file-id hash alongside offset; re-scan on
mismatch. Or use a sentinel sequence number.

### S-3 — `compute_stop_levels` ATR cap at 15% is asymmetric
File: `portfolio/risk_management.py:373`

```python
atr_pct = min(atr_pct, 15.0)
stop_price = entry_price * (1 - 2 * atr_pct / 100)
```

Cap means stops never wider than 30% of entry, regardless of true ATR.
For 5x leveraged certs the rule `.claude/rules/metals-avanza.md` says
"5x leverage certificates need -15%+ stops, not -8%". A 30% stop on a 5x
cert is fine; the cap matches user preference. But the same cap on a
non-leveraged underlying means stop never wider than 30% — too tight for
post-event recovery on small-cap stocks. Cross-asset symmetry issue.

**Severity:** P3 (system mostly trades the ones where 30% is right).
**Fix:** make cap asset-class-conditional, or move to per-ticker.

### S-4 — `MIN_VOTERS_METALS = 2` documented exception (verified, not a bug)
File: `portfolio/signal_engine.py:1014-1020`

Documented in `.claude/rules/signals.md` as "MIN_VOTERS = 3 for all asset
classes" but code has metals at 2. Looked at the code comment:

> 2026-05-11: metals run at noisier intraday horizon (1m-1h target)
> where the standard 3-voter floor almost never fires after persistence
> filter. Empirical: XAG sees 5 raw voters → 2 post-persistence;
> MIN_VOTERS=3 produced 0 trades in 20 days.

OK, deliberate carve-out documented + rationalized. The `.claude/rules`
file is stale — should reflect the metals exception.

**Severity:** P3 (docs drift).
**Fix:** update `.claude/rules/signals.md` to call out the metals exception.

### S-5 — `subprocess_utils.kill_orphaned_by_cmdline` PS-escape incomplete
File: `portfolio/subprocess_utils.py:213-220`

```python
safe_pattern = (
    pattern.replace("'", "''")
    .replace("[", "``[")
    .replace("]", "``]")
    .replace("*", "``*")
    .replace("?", "``?")
)
```

PS escape for `[` is `` `[ `` (one backtick). Python `"``[`"` is two
characters: backtick + `[`. That's what PS wants for a literal `[` inside
`-like` — but the replacement chain produces `` ` ` [ `` which is two
backticks. Two backticks is `` `` `` literal-backtick + literal-backtick.
Actually re-reading the Python: the source has `"``["` which IS one
backtick + `[`. (Python's `"``["` = backtick `+` backtick `+` `[`? No, in
Python `"```` ``"` is just literal chars — let me re-read the actual bytes.)

Verified by viewing: the source IS `"``["` which in Python is two
backticks followed by `[`. That's a PS escape of `` `[ `` = (one backtick
+ `[`) — but `"``["` has TWO backticks, so PS sees `` `` [ `` which is
literal-backtick + literal-`[`. The pattern then mis-matches command
lines that contain `[`.

Edge case, mostly benign — `kill_orphaned_by_cmdline` is called with
hard-coded patterns like `"python.exe.*metals_loop"` that don't contain
`[`. Latent bug if a future caller passes a glob.

**Severity:** P3.
**Fix:** use a raw string and ensure exactly one backtick per escape:
`pattern.replace("[", "`[").replace("]", "`]")` (in a raw f-string or
careful normal string).

### S-6 — `agent_invocation.py:1357` `_agent_proc.poll()` after `taskkill` race
File: `portfolio/agent_invocation.py:1345-1380`

Watchdog thread polls `_agent_proc.poll()` every 30s. After `taskkill /F`
succeeds but `wait(15)` times out (P0-12 above), `_agent_proc` is kept
non-None. Watchdog's next tick:
- `_agent_proc.poll()` may return an exit code now (process has reaped)
- _check_agent_completion_locked logs a `failed` row
- Then continues — BUT `_kill_overrun_agent` already logged a
  `timeout` row earlier

Double-log: same invocation appears twice in `invocations.jsonl` with
different statuses. Accuracy on Layer 2 health metrics is undermined.

**Severity:** P2.
**Fix:** mark `_agent_proc=None` even on `wait(15)` timeout in
`_kill_overrun_agent`, or have `_check_agent_completion_locked` check a
"already logged" flag before writing.

### S-7 — `valid_until=date.today()` in `place_order` has timezone semantics
File: `portfolio/avanza_client.py:352`, `portfolio/avanza/trading.py:82`

`date.today()` uses local time (Windows machine TZ). If the box is set
to UTC (server-ish setup) but Avanza interprets the date as CET, a 02:30
local order placed near midnight UTC might submit `valid_until=`
yesterday-CET = stale-day expiry. Edge case but for orders placed during
EOD sweep (~22:00 CET = 21:00 UTC) the offset matters.

**Severity:** P3.
**Fix:** use `datetime.now(ZoneInfo("Europe/Stockholm")).date()`.

---

## Cross-Verification of Subagent Claims

| Claim | Subagent | Verdict | Notes |
|---|---|---|---|
| connors_rsi2 ticker guard absorbs `context=` into `**kwargs` | signals-modules | **VERIFIED** | Read both files, signature confirmed, registry confirmed `requires_context=True`. |
| `signal_decay_alert.py` uses relative paths | signals-core | **VERIFIED** | Read line 27, confirmed `"data/accuracy_cache.json"` literal. |
| `accuracy_gate_threshold` config-overrideable below floor | signals-core | **VERIFIED** | Read line 4205, no clamp. |
| `multi_agent_layer2` synthesis runs on 0/3 success | orchestration | **VERIFIED** | Read agent_invocation.py:967-972, no `success_count` gate. |
| trigger.py price baseline stales | orchestration | **VERIFIED** | Read trigger.py:496, confirmed `if triggered:` gate around `state["last"]["prices"]`. |
| `api_utils.get_binance_config` reads `apiKey` not `key` | infrastructure | **VERIFIED** | Read api_utils.py:60 + config_validator.py:28. |
| `fin_snipe_manager` 1% stop violates 3% rule | metals-core | **VERIFIED** | Confirmed `.claude/rules/metals-avanza.md` says 3%. |
| `kelly_metals` ~50× over-sizing | portfolio-risk | **VERIFIED** | Math walkthrough independently reproduced. |
| `get_buying_power` returns zero on miss | avanza-api | **VERIFIED** | Read avanza/account.py:87-94. |
| `get_positions(None)` leaks pension | avanza-api | **VERIFIED** | Read avanza/account.py:27-61. |
| `_kill_overrun_agent` wedges when `wait(15)` times out | orchestration | **PARTIALLY VERIFIED** | Code reads as agent claims; effect (Popen.poll forever) requires test confirmation. |
| `_streaming_max` peak inflation under SEK weakening | portfolio-risk | **VERIFIED** | Logically follows from historical-FX storage + today-FX comparison. |
| `journal.load_recent` linear scan | infrastructure | **VERIFIED** | Read journal.py:23-40, no tail-read. |
| `data_collector` yfinance unlocked direct path | data-external | **NOT VERIFIED** | Did not deep-read; trusting agent. |
| `data/metals_loop.py:1978` raw `open() + json.load()` | metals-core | **NOT VERIFIED** | Did not read; metals_loop is 7880 lines, scoped to the agent. |
| grid_fisher `rotate_on_buy_fill` naked-position window | metals-core | **NOT VERIFIED** | Read enough to confirm `_safe_session_call` can return None. Verified plausible. |

---

## Adjustments to Subagent P0 Classifications

After cross-verification, my proposed re-classification (vs. raw subagent labels):

| Finding | Subagent | Re-class | Rationale |
|---|---|---|---|
| infrastructure P0-2 (dir fsync) | P0 | **P1** | NTFS metadata journaled separately, narrow window. |
| portfolio-risk P0-4 (trailing stop static) | P0 | **P1** | Missing feature, not broken safety. |
| signals-core P0-5 (SQLite check_same_thread) | P0 | **P2** | Latent: no module-level SignalDB exists today. |
| orchestration P0-3 (Popen exception leaves stale state) | P0 | **P1** | Reentrancy block at 761 catches the immediate next call. |
| signals-modules P0.1 (connors_rsi2) | P0 | **P0** (keep) | Promotion pipeline pollution is a real money risk, and the dispatcher-contract drift class is dangerous. |
| data-external P0-3 (Deribit silent kill) | P0 | **P0** (keep) | Active signal (crypto_evrp) depends on this. |
| metals-core P0-5 (legacy stop branch) | P0 | **P1** | Dead path under current config flags, but P1 because one flag flip exposes. |

Net P0 count after re-classification: 22 (down from 34 raw).

---

## What the Self-Review Missed

I did not deep-read:
- `data/metals_loop.py` (7880 lines) — trusted metals-core agent.
- All ~63 signal plugin modules — trusted signals-modules agent's table.
- `portfolio/avanza/streaming.py` (websocket reconnect logic) — trusted
  avanza-api agent.
- `multi_agent_layer2.py` end-to-end — verified only the specialist
  fan-in path the orchestration agent flagged.

These are the gaps where any P0 in the synthesis depends on the agent's
read, not on independent verification.

---

## Confidence Assessment

After cross-verification:
- High confidence on all P0s in the synthesis except orchestration P0-2
  (kill wedge needs runtime test).
- High confidence on the structural findings (god file, parallel order
  paths, bare except clusters).
- Medium confidence on the metals-core P1s — the agent has the deepest
  read of metals_loop, but I cannot independently verify the 7880-line
  context.
