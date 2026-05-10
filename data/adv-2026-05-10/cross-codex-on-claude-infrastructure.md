OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-infrastructure
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e127e-9fa8-7842-940b-a6d1272ba501
--------
user
You are an adversarial code reviewer cross-critiquing another reviewer's findings.

CONTEXT
- Codebase: Q:\finance-analyzer (autonomous trading system, Python).
- Subsystem: infrastructure
- The other reviewer (Claude Opus 4.7) audited this subsystem and produced
  the findings below.
- Your job: critique each finding by inspecting the actual source files and
  deciding whether the finding is correct.

PROCEDURE
For each finding in the list below:
1. Open the cited file(s) at the cited line(s) in the working tree.
2. Decide one of:
   - CONFIRMED: bug/issue is real, severity is reasonable.
   - PARTIAL: there is something there, but the analysis is wrong in a
     specific way (e.g., wrong line, wrong cause, wrong severity).
   - FALSE-POSITIVE: the finding is wrong; the code does not have the
     described problem. State why concretely (cite the actual code).
3. If you find a NEW issue while inspecting, list it under "New findings".

OUTPUT FORMAT (Markdown, no preamble)
## Verdicts

- [<orig severity>] <one-line restatement> — file:line
  Verdict: CONFIRMED | PARTIAL | FALSE-POSITIVE
  Reason: <one sentence citing actual code>
  (Adjustment: <if PARTIAL, what's actually wrong>)

## New findings (you, not Claude)

- [P1|P2|P3] <one-line> — file:line
  <one paragraph>

## Summary
- Confirmed: N
- Partial: N
- False-positive: N
- New from you: N

CLAUDE'S FINDINGS TO CRITIQUE:
=== BEGIN ===
# Adversarial Review: infrastructure subsystem (2026-05-08)

[P0] portfolio/http_retry.py:36
**Retries on all 4xx errors instead of only retryable 5xx/429.**
Problem: Auth failures (401/403) and bad-request (400) get retried, masking real
client-side bugs and bypassing the retry gate's intent.
Fix: Restrict retry policy to 5xx + 429 (and 408); never retry other 4xx.

[P0] dashboard/app.py:1039
**`POST /api/validate-portfolio` lacks CSRF token validation.**
Problem: Only the auth cookie is checked. Cross-origin form post can mutate state
without user consent. Same surface used for any other POST endpoint.
Fix: Require an explicit anti-CSRF header (SameSite=lax cookie is not enough), or a
short-lived token tied to the session cookie.

[P1] portfolio/shared_state.py:276
**Rate limiter thundering herd at `wait_time=0`.**
Problem: When wait time computes to zero, multiple threads see the same prior-call
timestamp and all skip the sleep — limit defeated.
Fix: Hold a lock around the read+update, or use a token-bucket primitive.

[P1] portfolio/file_utils.py:173-180
**`load_jsonl_tail` boundary detection assumes `\n` is always inside ASCII.**
Problem: Multibyte UTF-8 codepoints can include bytes that look like `\n`; boundary
search splits a codepoint, producing decode errors that look like data corruption.
Fix: `errors="replace"` or seek backward to a valid codepoint boundary before decode.

[P1] portfolio/gpu_gate.py:130-142
**Stale-lock break races with concurrent acquirer.**
Problem: `exists()` then `release()` is a TOCTOU; another process acquires after the
check but before the release; this process then releases the new live lock, allowing
two-process VRAM contention.
Fix: Use a real file lock (`msvcrt.locking` / `fcntl`) plus PID check inside the lock.

[P1] portfolio/process_lock.py:62-73
**`msvcrt.locking` unlock without `seek(0)` on the error path.**
Problem: Unlock targets the wrong byte range; lock is held permanently from Windows'
view, blocking subsequent process starts until reboot.
Fix: Always seek(0) before lock/unlock; wrap in try/finally that restores fp position.

[P1] portfolio/telegram_notifications.py:52, portfolio/message_store.py:119
**Token / chat_id leak via Markdown parse-error logs.**
Problem: When Telegram returns a parse error, the full request payload (including
chat_id and sometimes a token snippet) ends up in log output.
Fix: Redact tokens before logging; truncate chat_id; never include the bot token.

[P1] portfolio/subprocess_utils.py:140-143
**Timeout uses `proc.kill()` without Job Object guarantee on Windows.**
Problem: On Windows `proc.kill()` does not propagate to child processes; a claude CLI
sub-subprocess can keep running after the parent kill. This was the silent-auth outage
class.
Fix: Use a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; assign the
process tree to it on spawn.

[P1] portfolio/log_rotation.py:326-327
**`os.replace()` without `fsync()`.**
Problem: Write-back cache loss on crash leaves a half-written rotated log with
truncated tail; loss of recent error context exactly when needed.
Fix: `flush() + os.fsync(fileno())` before replace; consider directory fsync on POSIX.

[P1] portfolio/prophecy.py:70-73
**Concurrent `save_beliefs()` overwrites without merge.**
Problem: Layer 1 + Layer 2 writing prophecies in the same cycle clobber each other.
Fix: Read-modify-write under a per-file lock, or version optimistic concurrency.

[P2] portfolio/shared_state.py:94-123
**Cache exception handler mutates `_tool_cache` outside the lock.**
Problem: Lock released before cleanup; concurrent reader sees partially-mutated state.
Fix: Hold the lock through the cleanup branch.

[P2] dashboard/app.py:155-168
**Adaptive `tail_bytes` growth unbounded on corrupted JSONL.**
Problem: Bad line keeps doubling read size; up to 64MB allocation before fallback.
Fix: Cap growth at a sane max (8MB); on cap miss, fail-safe to "no records".

[P2] portfolio/alert_budget.py:42-44
**After-empty deque returns stale `last_call`.**
Problem: First message after deque pruned to empty passes; next caller sees stale
`last_call`, computes a huge wait. Alerts can be silenced for hours.
Fix: Reset `last_call` when deque empties.

[P2] portfolio/subprocess_utils.py:289-299
**PowerShell JSON parsing trusts stdout is valid JSON.**
Problem: A non-JSON PS error string raises in `json.loads`; caller gets crash instead
of context.
Fix: `try: json.loads ... except json.JSONDecodeError: log + return None`.

[P2] portfolio/journal.py:28-40
**TOCTOU between log_rotation.move and reader buffered read.**
Problem: Reader skipping lines (or hitting EOF early) when rotation runs mid-iteration.
Fix: Open with `O_RDONLY`, accept partial reads; or coordinate via shared rotation
lock.

[P3] dashboard/app.py:175-181
**Integer parsing silently truncates floats (`200.5 -> 200`).**
Problem: Bounds check is the only guard; sloppy fractional input slides through.
Fix: Validate digits-only before `int()`, return 400 otherwise.

[P3] portfolio/file_utils.py:56
**`ensure_ascii=True` bloats every JSON file with `\uXXXX` escapes.**
Problem: Bigger files, harder diffs.
Fix: `ensure_ascii=False` with explicit `encoding="utf-8"`.

[P3] portfolio/shared_state.py:68-75
**Eviction log warns "stuck key" on normal 2-min batch flushes.**
Problem: Alert fatigue.
Fix: Raise threshold to 5 min, or whitelist known batch keys.

[P3] dashboard/app.py:226-242
**GoldDigger payload duplicates fields via shallow copy.**
Problem: API surface confusion.
Fix: Pick one canonical field name; remove the duplicate.

[P3] portfolio/message_store.py:37-49
**Mojibake substitution incomplete; rare double-encoded cases survive.**
Fix: Use `ftfy` if available; document unsupported corners.

[P3] portfolio/gpu_gate.py:33
**Hardcoded `Q:/models` Windows-only path.**
Problem: Non-Windows installs never lock; concurrent VRAM loads possible.
Fix: Read path from config or compute relative to repo.

[P3] portfolio/prophecy.py:323
**Division-by-zero guard misses missing-field condition.**
Problem: Missing field falls through; consumer sees None where a number is expected.
Fix: `if field in data and data[field] != 0:`.

## Summary

2 P0 + 8 P1 + 5 P2 + 7 P3 = 22 findings. Themes: CSRF gap on dashboard POSTs, retry
policy retries 4xx, secrets in error logs, Windows process-tree kill not enforced,
concurrent writers without locks (prophecy, log rotation, GPU gate), rate-limiter
thundering herd at wait=0. Boundary handling weak in JSONL tail and adaptive
tail-bytes growth.
=== END ===

Use git/grep/read tools as needed. Be ruthless and cite code. Do not soften.

codex
Reviewing each cited finding against the working tree now. I’m starting by pulling the referenced snippets with line numbers so I can verify the exact behavior before I classify anything.
  → Collect cited code snippets for all 22 findings
  • Evaluate each finding against actual behavior
  • Draft verdicts, adjustments, and any new findings
codex
I’m gathering the cited line windows first, then I’ll trace outward into each function where the claim depends on surrounding control flow or locking. Some of these are likely line-accurate but causally wrong, so I’m checking more than the single line.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== $path:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/http_retry.py' 1 120
Show-Lines 'portfolio/shared_state.py' 1 340
Show-Lines 'portfolio/file_utils.py' 1 260
Show-Lines 'portfolio/gpu_gate.py' 1 220
Show-Lines 'portfolio/process_lock.py' 1 180" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== $path:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 120 260
Show-Lines 'dashboard/app.py' 1000 1085
Show-Lines 'portfolio/subprocess_utils.py' 110 330
Show-Lines 'portfolio/log_rotation.py' 300 360" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== $path:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/telegram_notifications.py' 1 120
Show-Lines 'portfolio/message_store.py' 1 180
Show-Lines 'portfolio/alert_budget.py' 1 120
Show-Lines 'portfolio/journal.py' 1 120
Show-Lines 'portfolio/prophecy.py' 1 380" in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:26:16.639188Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

 exited 1 in 657ms:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

2026-05-10T15:26:16.643007Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

 exited 1 in 661ms:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

2026-05-10T15:26:16.665176Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

 exited 1 in 682ms:
[31;1mParserError: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   5 | [0m   Write-Output "=== [36;1m$path:[0m$start-$end ==="[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                     ~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mVariable reference is not valid. ':' was not followed by a valid variable name character. Consider using ${} to[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m[31;1m[36;1m     | [31;1mdelimit the name.[0m

codex
The first pass tripped on a PowerShell string interpolation edge case. I’m rerunning with a cleaner reader so I can inspect the exact line ranges without losing time to shell noise.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/http_retry.py' 1 120
Show-Lines 'portfolio/shared_state.py' 1 340
Show-Lines 'portfolio/file_utils.py' 1 260
Show-Lines 'portfolio/gpu_gate.py' 1 220
Show-Lines 'portfolio/process_lock.py' 1 180" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 120 260
Show-Lines 'dashboard/app.py' 1000 1085
Show-Lines 'portfolio/subprocess_utils.py' 110 330
Show-Lines 'portfolio/log_rotation.py' 300 360" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  $abs = Resolve-Path $path
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content $abs | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/telegram_notifications.py' 1 120
Show-Lines 'portfolio/message_store.py' 1 180
Show-Lines 'portfolio/alert_budget.py' 1 120
Show-Lines 'portfolio/journal.py' 1 120
Show-Lines 'portfolio/prophecy.py' 1 380" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 742ms:
=== portfolio/telegram_notifications.py:1-120 ===
    1: """Telegram notification utilities — send messages, alerts, escape markdown."""
    2: 
    3: import logging
    4: import os
    5: import re
    6: 
    7: from portfolio.file_utils import load_json
    8: from portfolio.http_retry import fetch_with_retry
    9: from portfolio.message_store import send_or_store
   10: from portfolio.tickers import SYMBOLS
   11: 
   12: logger = logging.getLogger("portfolio.telegram")
   13: 
   14: _MD_V1_SPECIAL = re.compile(r'([_*`\[\]])')
   15: 
   16: from pathlib import Path
   17: 
   18: BOLD_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio_state_bold.json"
   19: _COOLDOWN_PREFIXES = ("cooldown", "crypto check-in", "startup")
   20: 
   21: 
   22: def escape_markdown_v1(text):
   23:     """Escape special Markdown v1 characters in dynamic content to prevent parse failures.
   24: 
   25:     Use this on user-facing dynamic strings (ticker names, error messages, reason text)
   26:     that are inserted into Markdown-formatted Telegram messages. Do NOT apply to the
   27:     entire message — it would break intentional formatting like *bold* and _italic_.
   28:     """
   29:     return _MD_V1_SPECIAL.sub(r'\\\1', str(text))
   30: 
   31: 
   32: _TELEGRAM_MAX_LENGTH = 4096  # Telegram API rejects messages exceeding this
   33: 
   34: 
   35: def send_telegram(msg, config):
   36:     if os.environ.get("NO_TELEGRAM"):
   37:         logger.info("[NO_TELEGRAM] Skipping send")
   38:         return True
   39:     # Global mute gate
   40:     if config.get("telegram", {}).get("mute_all", False):
   41:         logger.info("[mute_all] Skipping send_telegram")
   42:         return True
   43:     # Layer 1 messages disabled — only Layer 2 (Claude Code) sends Telegram
   44:     # via direct requests.post. To re-enable, set telegram.layer1_messages: true.
   45:     if not config.get("telegram", {}).get("layer1_messages", False):
   46:         logger.debug("[layer1_messages=false] Skipping Layer 1 send")
   47:         return True
   48:     # Truncate to Telegram's max message length to avoid silent 400 errors
   49:     if len(msg) > _TELEGRAM_MAX_LENGTH:
   50:         logger.warning("Telegram message truncated from %d to %d chars", len(msg), _TELEGRAM_MAX_LENGTH)
   51:         msg = msg[:_TELEGRAM_MAX_LENGTH - 20] + "\n...(truncated)"
   52:     token = config["telegram"]["token"]
   53:     chat_id = config["telegram"]["chat_id"]
   54:     r = fetch_with_retry(
   55:         f"https://api.telegram.org/bot{token}/sendMessage",
   56:         method="POST",
   57:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
   58:         timeout=30,
   59:     )
   60:     if r is None:
   61:         return False
   62:     if r.ok:
   63:         return True
   64:     # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
   65:     # still arrives (unformatted) rather than being silently lost.
   66:     if r.status_code == 400:
   67:         err_desc = ""
   68:         try:
   69:             err_desc = r.json().get("description", "")
   70:         except Exception:
   71:             logger.debug("Failed to parse Telegram error response", exc_info=True)
   72:         if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
   73:             logger.warning("Telegram Markdown parse failed (%s), resending without formatting", err_desc)
   74:             r2 = fetch_with_retry(
   75:                 f"https://api.telegram.org/bot{token}/sendMessage",
   76:                 method="POST",
   77:                 json_body={"chat_id": chat_id, "text": msg},
   78:                 timeout=30,
   79:             )
   80:             return r2 is not None and r2.ok
   81:     return False
   82: 
   83: 
   84: def _maybe_send_alert(config, signals, prices_usd, fx_rate, state, reasons, tf_data):
   85:     from portfolio.portfolio_mgr import portfolio_value
   86: 
   87:     significant = [r for r in reasons if not r.startswith(_COOLDOWN_PREFIXES)]
   88:     if not significant:
   89:         return
   90:     headline = escape_markdown_v1(significant[0])
   91:     lines = [f"*ALERT: {headline}*", ""]
   92:     # Actionable-only: show BUY/SELL tickers, compress HOLDs
   93:     hold_count = 0
   94:     for ticker in SYMBOLS:
   95:         sig = signals.get(ticker)
   96:         if not sig:
   97:             continue
   98:         action = sig["action"]
   99:         if action == "HOLD":
  100:             hold_count += 1
  101:             continue
  102:         price = prices_usd.get(ticker, 0)
  103:         extra = sig.get("extra", {})
  104:         b = extra.get("_buy_count", 0)
  105:         s = extra.get("_sell_count", 0)
  106:         total = extra.get("_total_applicable", 0)
  107:         h = max(0, total - b - s)
  108:         if price >= 1000:
  109:             p_str = f"${price:,.0f}"
  110:         else:
  111:             p_str = f"${price:,.2f}"
  112:         lines.append(f"`{ticker:<7} {p_str:>9}  {action:<4} {b}B/{s}S/{h}H`")
  113:     if hold_count > 0:
  114:         lines.append(f"_+ {hold_count} HOLD_")
  115:     fg_val = ""
  116:     for _ticker, sig in signals.items():
  117:         extra = sig.get("extra", {})
  118:         if "fear_greed" in extra:
  119:             fg_class = escape_markdown_v1(extra.get("fear_greed_class", ""))
  120:             fg_val = f"{extra['fear_greed']} ({fg_class})"
=== portfolio/message_store.py:1-180 ===
    1: """Central message routing — save all messages to JSONL, send only selected categories to Telegram.
    2: 
    3: Categories that are ALWAYS sent to Telegram:
    4:   - trade:   simulated BUY/SELL executions (Layer 2)
    5:   - iskbets: intraday entry/exit alerts
    6:   - bigbet:  mean-reversion BIG BET alerts
    7:   - digest:  4-hourly activity report
    8: 
    9: Categories that are SENT to Telegram:
   10:   - analysis:   HOLD analysis, market commentary (Layer 2 — sole Telegram sender)
   11: 
   12: Categories that are ALSO SENT to Telegram:
   13:   - invocation:  "Layer 2 Tx invoked" notifications
   14:   - regime:      regime shift alerts
   15:   - error:       loop crash notifications
   16: 
   17: Categories that are SAVED ONLY (viewable on dashboard / via file):
   18:   - fx_alert:    FX rate staleness warnings
   19: """
   20: 
   21: import logging
   22: import os
   23: import re
   24: from datetime import UTC, datetime
   25: from pathlib import Path
   26: 
   27: from portfolio.file_utils import atomic_append_jsonl
   28: from portfolio.http_retry import fetch_with_retry
   29: 
   30: logger = logging.getLogger("portfolio.message_store")
   31: 
   32: BASE_DIR = Path(__file__).resolve().parent.parent
   33: MESSAGES_FILE = BASE_DIR / "data" / "telegram_messages.jsonl"
   34: 
   35: _TELEGRAM_MAX_LENGTH = 4096
   36: _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
   37: _COMMON_MOJIBAKE_REPLACEMENTS = {
   38:     "Â·": "·",
   39:     "â": "—",
   40:     "â€“": "–",
   41:     "â": "'",
   42:     "â": "'",
   43:     'â': '"',
   44:     'â': '"',
   45:     "â": "→",
   46:     "â": "↑",
   47:     "â": "↓",
   48:     "Â": "",
   49: }
   50: 
   51: # Categories whose messages should be sent to Telegram in addition to being saved.
   52: SEND_CATEGORIES = {"trade", "iskbets", "bigbet", "digest", "daily_digest", "analysis", "invocation", "regime", "error", "elongir", "crypto_report"}
   53: 
   54: 
   55: def _repair_common_mojibake(text):
   56:     repaired = text
   57:     for bad, good in _COMMON_MOJIBAKE_REPLACEMENTS.items():
   58:         repaired = repaired.replace(bad, good)
   59:     return repaired
   60: 
   61: 
   62: def _normalize_message_whitespace(text):
   63:     lines = []
   64:     for raw_line in text.split("\n"):
   65:         if raw_line.startswith("`") and raw_line.endswith("`"):
   66:             lines.append(raw_line.rstrip())
   67:             continue
   68:         line = raw_line.replace("\t", " ")
   69:         line = re.sub(r" {2,}", " ", line).strip()
   70:         lines.append(line)
   71:     return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
   72: 
   73: 
   74: def sanitize_message_text(text):
   75:     """Normalize message text before saving/sending.
   76: 
   77:     Keeps intended Markdown structure while removing common control-byte and
   78:     mojibake artifacts that make Telegram messages unreadable.
   79:     """
   80:     cleaned = str(text or "")
   81:     cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
   82:     cleaned = _repair_common_mojibake(cleaned)
   83:     cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
   84:     return _normalize_message_whitespace(cleaned)
   85: 
   86: 
   87: def log_message(text, category="analysis", sent=False):
   88:     """Append a message to the JSONL message log.
   89: 
   90:     Args:
   91:         text: Message text (may contain Markdown).
   92:         category: Message category (see module docstring for valid values).
   93:         sent: Whether the message was actually sent to Telegram.
   94:     """
   95:     cleaned = sanitize_message_text(text)
   96:     entry = {
   97:         "ts": datetime.now(UTC).isoformat(),
   98:         "text": cleaned,
   99:         "category": category,
  100:         "sent": sent,
  101:     }
  102:     atomic_append_jsonl(MESSAGES_FILE, entry)
  103: 
  104: 
  105: def _do_send_telegram(msg, config):
  106:     """Actually send a message to Telegram. Returns True on success.
  107: 
  108:     This is the raw API call — no gating by layer1_messages or category.
  109:     Handles truncation, Markdown fallback on parse errors.
  110:     """
  111:     if os.environ.get("NO_TELEGRAM"):
  112:         logger.info("[NO_TELEGRAM] Skipping send")
  113:         return True
  114: 
  115:     msg = sanitize_message_text(msg)
  116: 
  117:     token = config.get("telegram", {}).get("token")
  118:     chat_id = config.get("telegram", {}).get("chat_id")
  119:     if not token or not chat_id:
  120:         logger.warning("Telegram token/chat_id not configured")
  121:         return False
  122: 
  123:     # Truncate to Telegram's max message length (BUG-131: truncate at line
  124:     # boundary to avoid breaking Markdown formatting mid-tag)
  125:     if len(msg) > _TELEGRAM_MAX_LENGTH:
  126:         logger.warning(
  127:             "Telegram message truncated from %d to %d chars",
  128:             len(msg), _TELEGRAM_MAX_LENGTH,
  129:         )
  130:         cut = _TELEGRAM_MAX_LENGTH - 20
  131:         # Find last newline before cut point to avoid splitting Markdown tags
  132:         nl_pos = msg.rfind("\n", 0, cut)
  133:         if nl_pos > cut // 2:
  134:             cut = nl_pos
  135:         msg = msg[:cut] + "\n...(truncated)"
  136: 
  137:     r = fetch_with_retry(
  138:         f"https://api.telegram.org/bot{token}/sendMessage",
  139:         method="POST",
  140:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
  141:         timeout=30,
  142:     )
  143:     if r is None:
  144:         return False
  145:     if r.ok:
  146:         return True
  147: 
  148:     # Markdown parse failure (HTTP 400) — retry without parse_mode
  149:     if r.status_code == 400:
  150:         err_desc = ""
  151:         try:
  152:             err_desc = r.json().get("description", "")
  153:         except Exception as e:
  154:             logger.debug("Failed to parse Telegram error response: %s", e)
  155:         if any(kw in err_desc.lower() for kw in ("parse", "markdown", "entity")):
  156:             logger.warning(
  157:                 "Telegram Markdown parse failed (%s), resending without formatting",
  158:                 err_desc,
  159:             )
  160:             r2 = fetch_with_retry(
  161:                 f"https://api.telegram.org/bot{token}/sendMessage",
  162:                 method="POST",
  163:                 json_body={"chat_id": chat_id, "text": msg},
  164:                 timeout=30,
  165:             )
  166:             return r2 is not None and r2.ok
  167:     return False
  168: 
  169: 
  170: def send_or_store(msg, config, category="analysis"):
  171:     """Central routing: save message to JSONL, optionally send to Telegram.
  172: 
  173:     If category is in SEND_CATEGORIES, the message is sent to Telegram AND logged.
  174:     Otherwise it is only logged (saved to JSONL for dashboard / file reading).
  175: 
  176:     This function bypasses the ``layer1_messages`` config gate — the category
  177:     determines whether to send, not the global flag.
  178: 
  179:     Args:
  180:         msg: Message text (may contain Markdown).
=== portfolio/alert_budget.py:1-120 ===
    1: """Telegram alert budgeting — prevents alert fatigue.
    2: 
    3: Priority levels:
    4:     1 = Normal (subject to budget)
    5:     2 = Important (subject to budget but gets priority in queue)
    6:     3 = Emergency (bypasses budget — stop-loss, circuit breaker, crash)
    7: """
    8: from __future__ import annotations
    9: 
   10: import logging
   11: import time
   12: from collections import deque
   13: 
   14: logger = logging.getLogger("portfolio.alert_budget")
   15: 
   16: PRIORITY_EMERGENCY = 3
   17: PRIORITY_IMPORTANT = 2
   18: PRIORITY_NORMAL = 1
   19: 
   20: 
   21: class AlertBudget:
   22:     """Token-bucket style alert rate limiter with priority bypass."""
   23: 
   24:     def __init__(self, max_per_hour: int = 3, window_seconds: int = 3600):
   25:         self.max_per_hour = max_per_hour
   26:         self.window_seconds = window_seconds
   27:         self._sent_timestamps: deque[float] = deque()
   28:         self._buffer: list[str] = []
   29: 
   30:     def _prune_old(self) -> None:
   31:         """Remove timestamps outside the current window."""
   32:         cutoff = time.time() - self.window_seconds
   33:         while self._sent_timestamps and self._sent_timestamps[0] < cutoff:
   34:             self._sent_timestamps.popleft()
   35: 
   36:     def should_send(self, message: str, priority: int = PRIORITY_NORMAL) -> bool:
   37:         """Check if an alert should be sent or buffered."""
   38:         if priority >= PRIORITY_EMERGENCY:
   39:             self._sent_timestamps.append(time.time())
   40:             return True
   41:         self._prune_old()
   42:         if len(self._sent_timestamps) < self.max_per_hour:
   43:             self._sent_timestamps.append(time.time())
   44:             return True
   45:         self._buffer.append(message)
   46:         return False
   47: 
   48:     def flush_buffer(self) -> list[str]:
   49:         """Return and clear buffered messages."""
   50:         buffered = self._buffer.copy()
   51:         self._buffer.clear()
   52:         return buffered
   53: 
   54:     @property
   55:     def remaining_budget(self) -> int:
   56:         self._prune_old()
   57:         return max(0, self.max_per_hour - len(self._sent_timestamps))
   58: 
   59:     @property
   60:     def buffer_size(self) -> int:
   61:         return len(self._buffer)
=== portfolio/journal.py:1-120 ===
    1: import json
    2: import logging
    3: import re
    4: from collections import defaultdict
    5: from datetime import UTC, datetime, timedelta
    6: from pathlib import Path
    7: 
    8: logger = logging.getLogger("portfolio.journal")
    9: 
   10: from portfolio.file_utils import atomic_write_text, load_json
   11: from portfolio.tickers import ALL_TICKERS
   12: 
   13: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   14: JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
   15: CONTEXT_FILE = DATA_DIR / "layer2_context.md"
   16: PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
   17: BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
   18: 
   19: TIER_FULL = 2
   20: TIER_COMPACT = 4
   21: 
   22: 
   23: def load_recent(max_entries=10, max_age_hours=8):
   24:     if not JOURNAL_FILE.exists():
   25:         return []
   26:     cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
   27:     entries = []
   28:     with open(JOURNAL_FILE, encoding="utf-8") as f:
   29:         for line in f:
   30:             line = line.strip()
   31:             if not line:
   32:                 continue
   33:             try:
   34:                 entry = json.loads(line)
   35:                 ts = datetime.fromisoformat(entry["ts"])
   36:                 if ts >= cutoff:
   37:                     entries.append(entry)
   38:             except (json.JSONDecodeError, KeyError, ValueError):
   39:                 continue
   40:     return entries[-max_entries:]
   41: 
   42: 
   43: def _is_all_hold(entry):
   44:     decisions = entry.get("decisions", {})
   45:     for strat in ("patient", "bold"):
   46:         d = decisions.get(strat, {})
   47:         if d.get("action", "HOLD") != "HOLD":
   48:             return False
   49:     return True
   50: 
   51: 
   52: def _non_neutral_tickers(entry):
   53:     tickers = entry.get("tickers", {})
   54:     return {
   55:         k: v for k, v in tickers.items() if v.get("outlook", "neutral") != "neutral"
   56:     }
   57: 
   58: 
   59: def _fmt_time(ts_str):
   60:     ts = datetime.fromisoformat(ts_str)
   61:     return ts.strftime("%H:%M UTC")
   62: 
   63: 
   64: def _fmt_time_range(ts_start, ts_end):
   65:     t0 = datetime.fromisoformat(ts_start).strftime("%H:%M")
   66:     t1 = datetime.fromisoformat(ts_end).strftime("%H:%M UTC")
   67:     return f"{t0}–{t1}"
   68: 
   69: 
   70: def _entry_age_hours(entry, now=None):
   71:     if now is None:
   72:         now = datetime.now(UTC)
   73:     ts = datetime.fromisoformat(entry["ts"])
   74:     return (now - ts).total_seconds() / 3600
   75: 
   76: 
   77: def _append_entry(lines, entry):
   78:     ts = _fmt_time(entry["ts"])
   79:     trigger = entry.get("trigger", "unknown")
   80:     regime = entry.get("regime", "unknown")
   81: 
   82:     lines.append(f"**{ts}** | trigger: {trigger}")
   83: 
   84:     reflection = entry.get("reflection")
   85:     if reflection:
   86:         lines.append(f"_Reflection: {reflection}_")
   87: 
   88:     lines.append(f"regime: {regime}")
   89: 
   90:     decisions = entry.get("decisions", {})
   91:     for strat in ("patient", "bold"):
   92:         d = decisions.get(strat, {})
   93:         action = d.get("action", "HOLD")
   94:         reasoning = d.get("reasoning", "")
   95:         lines.append(f"{strat}: {action} — {reasoning}")
   96: 
   97:     for ticker, info in _non_neutral_tickers(entry).items():
   98:         outlook = info.get("outlook", "neutral")
   99:         thesis = info.get("thesis", "")
  100:         levels = info.get("levels", [])
  101:         level_str = f" (S:{levels[0]} R:{levels[1]})" if len(levels) == 2 else ""
  102:         conviction = info.get("conviction")
  103:         conv_str = f" [{int(conviction * 100)}%]" if conviction else ""
  104:         lines.append(f"{ticker}: {outlook}{conv_str} — {thesis}{level_str}")
  105: 
  106:         debate = info.get("debate")
  107:         if debate and isinstance(debate, dict):
  108:             bull = debate.get("bull", "")
  109:             bear = debate.get("bear", "")
  110:             synthesis = debate.get("synthesis", "")
  111:             if bull:
  112:                 lines.append(f"  Bull: {bull}")
  113:             if bear:
  114:                 lines.append(f"  Bear: {bear}")
  115:             if synthesis:
  116:                 lines.append(f"  Synthesis: {synthesis}")
  117: 
  118:     lines.append("")
  119: 
  120: 
=== portfolio/prophecy.py:1-380 ===
    1: """Prophecy/Belief system — persistent macro convictions for Layer 2.
    2: 
    3: Manages a set of beliefs (macro convictions) that persist across invocations.
    4: Each belief has a thesis, conviction level, direction, target price, timeframe,
    5: supporting/opposing evidence, and checkpoints with dates/conditions that get
    6: auto-evaluated against live prices.
    7: 
    8: Layer 2 reads these beliefs every invocation to maintain strategic context
    9: and compare technical signals against fundamental convictions.
   10: """
   11: 
   12: import logging
   13: from datetime import UTC, datetime
   14: from pathlib import Path
   15: 
   16: from portfolio.file_utils import atomic_write_json, load_json
   17: 
   18: logger = logging.getLogger("portfolio.prophecy")
   19: 
   20: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   21: PROPHECY_FILE = DATA_DIR / "prophecy.json"
   22: 
   23: # Belief schema
   24: BELIEF_TEMPLATE = {
   25:     "id": "",                    # unique identifier (e.g., "silver_bull_2026")
   26:     "ticker": "",                # primary ticker (e.g., "XAG-USD")
   27:     "thesis": "",                # text description of the conviction
   28:     "direction": "neutral",      # "bullish", "bearish", "neutral"
   29:     "conviction": 0.5,           # 0.0-1.0 conviction level
   30:     "target_price": None,        # target price (USD)
   31:     "target_timeframe": "",      # e.g., "2026-Q4", "6 months"
   32:     "entry_price": None,         # price when belief was created
   33:     "created_at": "",            # ISO-8601
   34:     "updated_at": "",            # ISO-8601
   35:     "status": "active",          # "active", "paused", "expired", "confirmed", "invalidated"
   36:     "supporting_evidence": [],   # list of strings
   37:     "opposing_evidence": [],     # list of strings
   38:     "checkpoints": [],           # list of checkpoint dicts
   39:     "tags": [],                  # e.g., ["metals", "macro", "geopolitical"]
   40:     "notes": "",                 # free-form notes
   41: }
   42: 
   43: CHECKPOINT_TEMPLATE = {
   44:     "id": "",                    # unique checkpoint identifier
   45:     "condition": "",             # human-readable condition (e.g., "XAG breaks $35")
   46:     "target_value": None,        # numeric target (price level)
   47:     "comparison": "above",       # "above", "below", "between"
   48:     "deadline": None,            # ISO-8601 deadline (optional)
   49:     "status": "pending",         # "pending", "triggered", "expired", "missed"
   50:     "triggered_at": None,        # when condition was met
   51:     "created_at": "",            # ISO-8601
   52: }
   53: 
   54: 
   55: def load_beliefs():
   56:     """Load all beliefs from prophecy.json.
   57: 
   58:     Returns:
   59:         dict: {"beliefs": [...], "metadata": {...}}
   60:     """
   61:     data = load_json(PROPHECY_FILE)
   62:     if data is None:
   63:         return {"beliefs": [], "metadata": {"version": 1, "last_review": None}}
   64:     if isinstance(data, list):
   65:         # Legacy format — wrap in dict
   66:         return {"beliefs": data, "metadata": {"version": 1, "last_review": None}}
   67:     return data
   68: 
   69: 
   70: def save_beliefs(data):
   71:     """Save beliefs to prophecy.json."""
   72:     data["metadata"]["last_review"] = datetime.now(UTC).isoformat()
   73:     atomic_write_json(PROPHECY_FILE, data)
   74: 
   75: 
   76: def add_belief(belief_dict):
   77:     """Add a new belief. Fills in defaults from template.
   78: 
   79:     Args:
   80:         belief_dict: Dict with belief fields. Must include "id" and "ticker".
   81: 
   82:     Returns:
   83:         The added belief dict.
   84:     """
   85:     data = load_beliefs()
   86: 
   87:     # Check for duplicate ID
   88:     existing_ids = {b["id"] for b in data["beliefs"]}
   89:     if belief_dict.get("id") in existing_ids:
   90:         raise ValueError(f"Belief with id '{belief_dict['id']}' already exists")
   91: 
   92:     # Merge with template
   93:     belief = {**BELIEF_TEMPLATE, **belief_dict}
   94:     now = datetime.now(UTC).isoformat()
   95:     if not belief["created_at"]:
   96:         belief["created_at"] = now
   97:     if not belief["updated_at"]:
   98:         belief["updated_at"] = now
   99: 
  100:     data["beliefs"].append(belief)
  101:     save_beliefs(data)
  102:     return belief
  103: 
  104: 
  105: def update_belief(belief_id, updates):
  106:     """Update an existing belief.
  107: 
  108:     Args:
  109:         belief_id: The belief's unique ID.
  110:         updates: Dict of fields to update.
  111: 
  112:     Returns:
  113:         The updated belief dict, or None if not found.
  114:     """
  115:     data = load_beliefs()
  116: 
  117:     for i, belief in enumerate(data["beliefs"]):
  118:         if belief["id"] == belief_id:
  119:             belief.update(updates)
  120:             belief["updated_at"] = datetime.now(UTC).isoformat()
  121:             data["beliefs"][i] = belief
  122:             save_beliefs(data)
  123:             return belief
  124: 
  125:     return None
  126: 
  127: 
  128: def remove_belief(belief_id):
  129:     """Remove a belief by ID.
  130: 
  131:     Returns:
  132:         True if removed, False if not found.
  133:     """
  134:     data = load_beliefs()
  135:     original_len = len(data["beliefs"])
  136:     data["beliefs"] = [b for b in data["beliefs"] if b["id"] != belief_id]
  137: 
  138:     if len(data["beliefs"]) < original_len:
  139:         save_beliefs(data)
  140:         return True
  141:     return False
  142: 
  143: 
  144: def get_belief(belief_id):
  145:     """Get a single belief by ID.
  146: 
  147:     Returns:
  148:         Belief dict or None.
  149:     """
  150:     data = load_beliefs()
  151:     for belief in data["beliefs"]:
  152:         if belief["id"] == belief_id:
  153:             return belief
  154:     return None
  155: 
  156: 
  157: def get_active_beliefs(ticker=None):
  158:     """Get all active beliefs, optionally filtered by ticker.
  159: 
  160:     Returns:
  161:         List of active belief dicts.
  162:     """
  163:     data = load_beliefs()
  164:     beliefs = [b for b in data["beliefs"] if b.get("status") == "active"]
  165:     if ticker:
  166:         beliefs = [b for b in beliefs if b.get("ticker") == ticker]
  167:     return beliefs
  168: 
  169: 
  170: def add_checkpoint(belief_id, checkpoint_dict):
  171:     """Add a checkpoint to an existing belief.
  172: 
  173:     Args:
  174:         belief_id: The belief's unique ID.
  175:         checkpoint_dict: Dict with checkpoint fields.
  176: 
  177:     Returns:
  178:         The added checkpoint dict, or None if belief not found.
  179:     """
  180:     data = load_beliefs()
  181: 
  182:     for i, belief in enumerate(data["beliefs"]):
  183:         if belief["id"] == belief_id:
  184:             cp = {**CHECKPOINT_TEMPLATE, **checkpoint_dict}
  185:             if not cp["created_at"]:
  186:                 cp["created_at"] = datetime.now(UTC).isoformat()
  187:             if not cp["id"]:
  188:                 cp["id"] = f"cp_{len(belief.get('checkpoints', []))}"
  189: 
  190:             if "checkpoints" not in belief:
  191:                 belief["checkpoints"] = []
  192:             belief["checkpoints"].append(cp)
  193:             belief["updated_at"] = datetime.now(UTC).isoformat()
  194:             data["beliefs"][i] = belief
  195:             save_beliefs(data)
  196:             return cp
  197: 
  198:     return None
  199: 
  200: 
  201: def evaluate_checkpoints(prices_usd):
  202:     """Evaluate all pending checkpoints against current prices.
  203: 
  204:     Args:
  205:         prices_usd: Dict {ticker: price_usd} of current prices.
  206: 
  207:     Returns:
  208:         List of newly triggered checkpoint dicts (with belief_id added).
  209:     """
  210:     data = load_beliefs()
  211:     triggered = []
  212:     modified = False
  213:     now = datetime.now(UTC)
  214: 
  215:     for i, belief in enumerate(data["beliefs"]):
  216:         if belief.get("status") != "active":
  217:             continue
  218: 
  219:         ticker = belief.get("ticker", "")
  220:         current_price = prices_usd.get(ticker)
  221:         if current_price is None:
  222:             continue
  223: 
  224:         for j, cp in enumerate(belief.get("checkpoints", [])):
  225:             if cp.get("status") != "pending":
  226:                 continue
  227: 
  228:             # Check deadline expiry
  229:             deadline = cp.get("deadline")
  230:             if deadline:
  231:                 try:
  232:                     deadline_dt = datetime.fromisoformat(deadline)
  233:                     if now > deadline_dt:
  234:                         cp["status"] = "expired"
  235:                         data["beliefs"][i]["checkpoints"][j] = cp
  236:                         modified = True
  237:                         continue
  238:                 except (ValueError, TypeError):
  239:                     pass
  240: 
  241:             # Check condition
  242:             target = cp.get("target_value")
  243:             comparison = cp.get("comparison", "above")
  244: 
  245:             if target is None:
  246:                 continue
  247: 
  248:             met = (
  249:                 (comparison == "above" and current_price >= target)
  250:                 or (comparison == "below" and current_price <= target)
  251:                 or (
  252:                     comparison == "between"
  253:                     and isinstance(target, (list, tuple))
  254:                     and len(target) == 2
  255:                     and target[0] <= current_price <= target[1]
  256:                 )
  257:             )
  258: 
  259:             if met:
  260:                 cp["status"] = "triggered"
  261:                 cp["triggered_at"] = now.isoformat()
  262:                 data["beliefs"][i]["checkpoints"][j] = cp
  263:                 modified = True
  264:                 triggered.append({**cp, "belief_id": belief["id"], "ticker": ticker, "price": current_price})
  265: 
  266:     if modified:
  267:         save_beliefs(data)
  268: 
  269:     return triggered
  270: 
  271: 
  272: def get_context_for_layer2(prices_usd=None):
  273:     """Build compact belief context for Layer 2 consumption.
  274: 
  275:     Returns a dict suitable for inclusion in agent_summary_compact.json.
  276:     Only includes active beliefs with relevant context.
  277: 
  278:     Args:
  279:         prices_usd: Current prices for progress calculation.
  280: 
  281:     Returns:
  282:         dict: {
  283:             "beliefs": [
  284:                 {
  285:                     "id": "silver_bull_2026",
  286:                     "ticker": "XAG-USD",
  287:                     "direction": "bullish",
  288:                     "conviction": 0.8,
  289:                     "thesis": "Silver to $120...",
  290:                     "target_price": 120.0,
  291:                     "progress_pct": 15.2,  # % of way from entry to target
  292:                     "checkpoints_summary": "2/5 triggered",
  293:                     "tags": ["metals", "macro"],
  294:                 }
  295:             ],
  296:             "total_active": 3,
  297:         }
  298:     """
  299:     active = get_active_beliefs()
  300: 
  301:     if not active:
  302:         return {"beliefs": [], "total_active": 0}
  303: 
  304:     compact_beliefs = []
  305:     for belief in active:
  306:         entry = {
  307:             "id": belief["id"],
  308:             "ticker": belief.get("ticker", ""),
  309:             "direction": belief.get("direction", "neutral"),
  310:             "conviction": belief.get("conviction", 0.5),
  311:             "thesis": belief.get("thesis", "")[:200],  # truncate for compactness
  312:             "target_price": belief.get("target_price"),
  313:             "tags": belief.get("tags", []),
  314:         }
  315: 
  316:         # Compute progress toward target
  317:         if prices_usd and belief.get("ticker") in prices_usd and belief.get("target_price") and belief.get("entry_price"):
  318:             current = prices_usd[belief["ticker"]]
  319:             entry_price = belief["entry_price"]
  320:             target = belief["target_price"]
  321: 
  322:             if target != entry_price:
  323:                 progress = (current - entry_price) / (target - entry_price) * 100
  324:                 entry["progress_pct"] = round(progress, 1)
  325:                 entry["current_price"] = round(current, 2)
  326: 
  327:         # Checkpoint summary
  328:         checkpoints = belief.get("checkpoints", [])
  329:         if checkpoints:
  330:             triggered = sum(1 for cp in checkpoints if cp.get("status") == "triggered")
  331:             total = len(checkpoints)
  332:             entry["checkpoints_summary"] = f"{triggered}/{total} triggered"
  333: 
  334:         compact_beliefs.append(entry)
  335: 
  336:     return {
  337:         "beliefs": compact_beliefs,
  338:         "total_active": len(compact_beliefs),
  339:     }
  340: 
  341: 
  342: def print_prophecy_review():
  343:     """Print a human-readable review of all beliefs."""
  344:     data = load_beliefs()
  345:     beliefs = data.get("beliefs", [])
  346: 
  347:     if not beliefs:
  348:         print("No beliefs configured. Seed data/prophecy.json with macro convictions.")
  349:         return
  350: 
  351:     print("=== Prophecy / Belief Review ===\n")
  352: 
  353:     active = [b for b in beliefs if b.get("status") == "active"]
  354:     inactive = [b for b in beliefs if b.get("status") != "active"]
  355: 
  356:     for belief in active:
  357:         direction_symbol = "^" if belief.get("direction") == "bullish" else "v" if belief.get("direction") == "bearish" else ">"
  358:         conv = belief.get("conviction", 0)
  359:         conv_bar = "#" * int(conv * 10) + "." * (10 - int(conv * 10))
  360: 
  361:         print(f"  {direction_symbol} [{belief['id']}] {belief.get('ticker', '?')}")
  362:         print(f"    Thesis: {belief.get('thesis', 'N/A')}")
  363:         print(f"    Conviction: [{conv_bar}] {conv:.0%}")
  364:         if belief.get("target_price"):
  365:             print(f"    Target: ${belief['target_price']} ({belief.get('target_timeframe', 'N/A')})")
  366:         if belief.get("entry_price"):
  367:             print(f"    Entry: ${belief['entry_price']}")
  368: 
  369:         # Checkpoints
  370:         cps = belief.get("checkpoints", [])
  371:         if cps:
  372:             triggered_count = sum(1 for cp in cps if cp.get("status") == "triggered")
  373:             print(f"    Checkpoints: {triggered_count}/{len(cps)}")
  374:             for cp in cps:
  375:                 status_icon = "[x]" if cp.get("status") == "triggered" else "[ ]" if cp.get("status") == "pending" else "[!]"
  376:                 print(f"      {status_icon} {cp.get('condition', '?')} [{cp.get('status', '?')}]")
  377: 
  378:         # Evidence
  379:         supporting = belief.get("supporting_evidence", [])
  380:         opposing = belief.get("opposing_evidence", [])

 succeeded in 752ms:
=== portfolio/http_retry.py:1-120 ===
    1: """HTTP retry utility with exponential backoff for finance-analyzer API calls."""
    2: 
    3: import logging
    4: import random
    5: import time
    6: 
    7: import requests
    8: 
    9: logger = logging.getLogger(__name__)
   10: 
   11: DEFAULT_RETRIES = 3
   12: DEFAULT_BACKOFF = 1.0  # seconds
   13: DEFAULT_BACKOFF_FACTOR = 2.0
   14: RETRYABLE_STATUS = {429, 500, 502, 503, 504}
   15: 
   16: 
   17: def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
   18:                      backoff=DEFAULT_BACKOFF, backoff_factor=DEFAULT_BACKOFF_FACTOR,
   19:                      timeout=30, headers=None, params=None, json_body=None,
   20:                      session=None):
   21:     """Make an HTTP request with exponential backoff retry.
   22: 
   23:     Returns response object on success, None on all retries exhausted.
   24:     """
   25:     requester = session or requests
   26: 
   27:     for attempt in range(retries + 1):
   28:         try:
   29:             if method.upper() == "GET":
   30:                 resp = requester.get(url, headers=headers, params=params, timeout=timeout)
   31:             elif method.upper() == "POST":
   32:                 resp = requester.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
   33:             else:
   34:                 resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)
   35: 
   36:             if resp.status_code not in RETRYABLE_STATUS:
   37:                 return resp
   38: 
   39:             if attempt < retries:
   40:                 wait = backoff * (backoff_factor ** attempt)
   41:                 jitter = random.uniform(0, wait * 0.1)
   42:                 wait += jitter
   43:                 # H26/HTTP1: Honour Telegram retry_after parameter for 429 responses.
   44:                 if resp.status_code == 429:
   45:                     try:
   46:                         retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
   47:                     except Exception:
   48:                         retry_after = wait
   49:                     wait = retry_after
   50:                 logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
   51:                                resp.status_code, url, attempt + 1, retries, wait)
   52:                 time.sleep(wait)
   53:             else:
   54:                 logger.error("HTTP %s from %s after %d retries",
   55:                              resp.status_code, url, retries)
   56:                 return None
   57: 
   58:         except (requests.ConnectionError, requests.Timeout) as e:
   59:             if attempt < retries:
   60:                 wait = backoff * (backoff_factor ** attempt)
   61:                 jitter = random.uniform(0, wait * 0.1)
   62:                 wait += jitter
   63:                 logger.warning("%s from %s, retry %d/%d in %.1fs",
   64:                                e.__class__.__name__, url, attempt + 1, retries, wait)
   65:                 time.sleep(wait)
   66:             else:
   67:                 logger.error("Request failed after %d retries: %s - %s",
   68:                              retries, url, e)
   69:                 return None
   70: 
   71:     return None
   72: 
   73: 
   74: def fetch_json(url, *, method="GET", retries=DEFAULT_RETRIES, default=None,
   75:                label="", headers=None, params=None, timeout=30, session=None,
   76:                **kwargs):
   77:     """Fetch URL and return parsed JSON, or ``default`` on any failure.
   78: 
   79:     Combines fetch_with_retry() + raise_for_status() + .json() into one call.
   80:     """
   81:     resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
   82:                             headers=headers, params=params, session=session)
   83:     if resp is None:
   84:         if label:
   85:             logger.warning("[%s] request returned None", label)
   86:         return default
   87:     try:
   88:         resp.raise_for_status()
   89:         return resp.json()
   90:     except Exception as e:
   91:         if label:
   92:             logger.warning("[%s] HTTP %s or JSON parse error: %s", label,
   93:                            getattr(resp, 'status_code', '?'), e)
   94:         return default
=== portfolio/shared_state.py:1-340 ===
    1: """Shared mutable state for the portfolio system.
    2: 
    3: All modules that need caching or cross-cycle state import from here.
    4: This avoids circular imports and provides a single source of truth.
    5: """
    6: 
    7: import logging
    8: import threading
    9: import time
   10: from datetime import UTC
   11: 
   12: logger = logging.getLogger("portfolio.shared_state")
   13: 
   14: # Tool cache — avoid re-running expensive tools every cycle
   15: _tool_cache = {}
   16: 
   17: # Retry cooldown for _cached() errors
   18: _RETRY_COOLDOWN = 60
   19: 
   20: 
   21: _CACHE_MAX_SIZE = 512  # evict expired entries when cache exceeds this size
   22: _cache_lock = threading.Lock()
   23: 
   24: # BUG-166: Dogpile/thundering-herd prevention.
   25: # Tracks which keys are currently being refreshed. When a thread sees a cache
   26: # miss and the key is already loading, it returns stale data (if available)
   27: # instead of calling the function redundantly.
   28: _loading_keys: set[str] = set()
   29: _LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
   30: # C11/SS1: Track when each key was added to _loading_keys for eviction of
   31: # permanently stuck keys (batch flush crash before updating cache).
   32: _loading_timestamps: dict[str, float] = {}
   33: 
   34: _MAX_STALE_FACTOR = 3  # return None if cached data is older than TTL * this factor
   35: 
   36: 
   37: def _cached(key, ttl, func, *args):
   38:     """Cache-through helper: returns cached data if fresh, else calls func.
   39: 
   40:     Dogpile prevention (BUG-166): when multiple threads detect a cache miss
   41:     simultaneously, only one thread fetches the data. Others return stale
   42:     data if available, preventing redundant expensive calls (LLM inference,
   43:     API requests) and model swap contention.
   44: 
   45:     On error, returns stale data if it's less than TTL * _MAX_STALE_FACTOR old.
   46:     Beyond that, returns None to prevent trading on dangerously old data.
   47:     """
   48:     now = time.time()
   49:     with _cache_lock:
   50:         if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
   51:             return _tool_cache[key]["data"]
   52:         # Evict expired entries when cache grows too large
   53:         # Use TTL-aware eviction: entries expire after ttl * _MAX_STALE_FACTOR
   54:         if len(_tool_cache) > _CACHE_MAX_SIZE:
   55:             expired = [k for k, v in _tool_cache.items()
   56:                        if now - v["time"] > v.get("ttl", 3600) * _MAX_STALE_FACTOR]
   57:             for k in expired:
   58:                 del _tool_cache[k]
   59:             # LRU fallback: if still over limit (all entries fresh), evict oldest 25%
   60:             if len(_tool_cache) > _CACHE_MAX_SIZE:
   61:                 sorted_keys = sorted(
   62:                     _tool_cache, key=lambda k: _tool_cache[k]["time"]
   63:                 )
   64:                 evict_count = len(sorted_keys) // 4 or 1
   65:                 for k in sorted_keys[:evict_count]:
   66:                     del _tool_cache[k]
   67: 
   68:         # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
   69:         _now_evict = time.time()
   70:         stuck = [k for k, ts in _loading_timestamps.items()
   71:                  if _now_evict - ts > _LOADING_TIMEOUT]
   72:         for k in stuck:
   73:             _loading_keys.discard(k)
   74:             stuck_duration = _now_evict - _loading_timestamps.pop(k, _now_evict)
   75:             logger.warning("[%s] evicted stuck loading key after %.0fs (timeout %ds)", k, stuck_duration, _LOADING_TIMEOUT)
   76: 
   77:         # BUG-166: Dogpile prevention — if another thread is already loading
   78:         # this key, return stale data instead of calling func redundantly.
   79:         if key in _loading_keys:
   80:             if key in _tool_cache:
   81:                 age = now - _tool_cache[key]["time"]
   82:                 max_stale = ttl * _MAX_STALE_FACTOR
   83:                 if age <= max_stale:
   84:                     logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
   85:                     return _tool_cache[key]["data"]
   86:             # No stale data available — return None rather than pile on
   87:             logger.debug("[%s] no stale data, another thread loading — returning None", key)
   88:             return None
   89:         _loading_keys.add(key)
   90:         _loading_timestamps[key] = time.time()
   91: 
   92:     try:
   93:         data = func(*args)
   94:         with _cache_lock:
   95:             # Don't cache None results — they typically indicate transient
   96:             # API failures that returned None instead of raising. Caching
   97:             # None hides the failure for the entire TTL duration, preventing
   98:             # retry. Stale data (if any) is preserved for fallback.
   99:             if data is not None:
  100:                 _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
  101:             _loading_keys.discard(key)
  102:             _loading_timestamps.pop(key, None)  # BUG-213: clean up on success path
  103:         return data
  104:     except KeyboardInterrupt:
  105:         with _cache_lock:
  106:             _loading_keys.discard(key)
  107:         logger.warning("[%s] interrupted (KeyboardInterrupt), returning None", key)
  108:         return None
  109:     except Exception as e:
  110:         logger.warning("[%s] error: %s", key, e)
  111:         with _cache_lock:
  112:             _loading_keys.discard(key)
  113:             if key in _tool_cache:
  114:                 age = now - _tool_cache[key]["time"]
  115:                 max_stale = ttl * _MAX_STALE_FACTOR
  116:                 if age > max_stale:
  117:                     logger.warning(
  118:                         "[%s] stale data too old (%.0fs > %.0fs max), returning None",
  119:                         key, age, max_stale,
  120:                     )
  121:                     return None
  122:                 _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
  123:                 return _tool_cache[key]["data"]
  124:         return None
  125: 
  126: 
  127: def _cached_or_enqueue(key, ttl, enqueue_fn, context,
  128:                         should_enqueue_fn=None, max_stale_factor=None):
  129:     """Check cache — if fresh return it, if expired enqueue for batch and return stale.
  130: 
  131:     Unlike _cached(), this never calls the model directly. On miss, it adds
  132:     the request to the batch queue and returns stale data (or None).
  133: 
  134:     Dogpile prevention (Codex finding #5): uses _loading_keys to avoid
  135:     re-enqueuing the same key every cycle if the batch flush hasn't run yet.
  136: 
  137:     2026-04-10 (perf/llama-swap-reduction) — two new optional parameters to
  138:     support rotation scheduling of LLM signals:
  139: 
  140:     - should_enqueue_fn: callable returning bool. If provided and the cache
  141:       is stale-but-present, skip the enqueue when the callback says "no"
  142:       (rotation off-cycle). If stale data is NOT available, force-enqueue
  143:       regardless of the callback — we cannot leave the caller empty-handed
  144:       when no stale fallback exists. Default None means "always enqueue",
  145:       which preserves the pre-rotation behavior for every existing caller.
  146: 
  147:     - max_stale_factor: integer override for how stale data can be returned,
  148:       in multiples of ttl. Default None means use the module-level
  149:       _MAX_STALE_FACTOR. LLM rotation passes 5 here so each rotated vote
  150:       can stay valid across the full rotation cycle (3 * TTL) plus slippage.
  151:     """
  152:     now = time.time()
  153:     effective_stale_factor = (
  154:         max_stale_factor if max_stale_factor is not None else _MAX_STALE_FACTOR
  155:     )
  156:     with _cache_lock:
  157:         if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
  158:             return _tool_cache[key]["data"]
  159: 
  160:         # Check stale availability BEFORE deciding whether to enqueue, because
  161:         # the rotation gate can only safely skip enqueue when we have stale
  162:         # fallback to return. If stale is exhausted we must force-enqueue.
  163:         #
  164:         # 2026-04-10 code-review finding N1: also treat cached `data is None`
  165:         # as "stale NOT available". main.py writes _update_cache(key, None)
  166:         # when a flush fails as a short-lived retry cooldown; without this
  167:         # check, the rotation gate would skip enqueue on those None entries
  168:         # and the caller would see None for up to 3 rotation cycles
  169:         # (~3 minutes) before retry. The extra `is not None` guard force-
  170:         # enqueues on failed-cache entries matching legacy recovery behavior.
  171:         stale_data = None
  172:         stale_available = False
  173:         if key in _tool_cache:
  174:             age = now - _tool_cache[key]["time"]
  175:             cached_data = _tool_cache[key]["data"]
  176:             if age <= ttl * effective_stale_factor and cached_data is not None:
  177:                 stale_available = True
  178:                 stale_data = cached_data
  179: 
  180:         # Decide whether to enqueue:
  181:         # - Default (no should_enqueue_fn): always enqueue (legacy behavior)
  182:         # - Callback returns True: enqueue (rotation on-cycle, or force path)
  183:         # - Callback returns False AND stale available: skip (rotation off-cycle,
  184:         #   stale fallback carries us until next on-cycle)
  185:         # - Callback returns False AND stale NOT available: enqueue anyway
  186:         #   (fresh cold path; caller has no fallback, we must refresh)
  187:         if should_enqueue_fn is None:
  188:             should_enq = True
  189:         else:
  190:             try:
  191:                 should_enq = bool(should_enqueue_fn()) or not stale_available
  192:             except Exception as e:
  193:                 logger.warning(
  194:                     "[%s] should_enqueue_fn raised, defaulting to enqueue: %s",
  195:                     key, e,
  196:                 )
  197:                 should_enq = True
  198: 
  199:         if should_enq and enqueue_fn and context is not None and key not in _loading_keys:
  200:             _loading_keys.add(key)
  201:             # C11/SS1: Track enqueue time for stuck-key eviction.
  202:             _loading_timestamps[key] = time.time()
  203:             try:
  204:                 enqueue_fn(key, context)
  205:             except Exception as e:
  206:                 _loading_keys.discard(key)
  207:                 _loading_timestamps.pop(key, None)
  208:                 logger.warning("[%s] enqueue_fn raised, key released: %s", key, e)
  209: 
  210:         # Return stale if available
  211:         if stale_available:
  212:             return stale_data
  213:     return None
  214: 
  215: 
  216: # 2026-04-10 (perf/llama-swap-reduction): monotonic counter of full-LLM
  217: # batch flushes that actually processed work. Drives rotation scheduling in
  218: # portfolio.llm_batch.is_llm_on_cycle — incremented at the end of
  219: # flush_llm_batch() iff at least one phase had queued items. In-memory only,
  220: # resets to 0 on process start; on restart the rotation deterministically
  221: # restarts at ministral with a cold-start warmup cycle that runs all LLMs.
  222: _full_llm_cycle_count = 0
  223: 
  224: 
  225: def _update_cache(key, data, ttl=None):
  226:     """Update a cache entry directly (for batch flush results)."""
  227:     with _cache_lock:
  228:         _loading_keys.discard(key)
  229:         # C11/SS1: Clean up timestamp when key is resolved.
  230:         _loading_timestamps.pop(key, None)
  231:         _tool_cache[key] = {
  232:             "data": data,
  233:             "time": time.time(),
  234:             "ttl": ttl or 900,
  235:         }
  236: 
  237: 
  238: # Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
  239: _run_cycle_id = 0
  240: 
  241: # Current market state — updated each run() cycle, used by data_collector for yfinance fallback
  242: _current_market_state = "open"
  243: 
  244: # Regime detection cache (invalidated each cycle)
  245: # BUG-169: Protected by _regime_lock — accessed from 8 concurrent ThreadPoolExecutor threads
  246: _regime_cache = {}
  247: _regime_cache_cycle = 0
  248: _regime_lock = threading.Lock()
  249: 
  250: 
  251: # --- Rate limiters ---
  252: 
  253: class _RateLimiter:
  254:     """Token-bucket rate limiter. Sleeps when calls exceed rate."""
  255:     def __init__(self, max_per_minute, name=""):
  256:         self.interval = 60.0 / max_per_minute
  257:         self.last_call = 0.0
  258:         self.name = name
  259:         self._lock = threading.Lock()
  260: 
  261:     def wait(self):
  262:         # BUG-212: Sleep OUTSIDE the lock to avoid blocking all 8 worker
  263:         # threads. Calculate sleep duration under the lock, release it,
  264:         # then sleep.
  265:         # Fix: Reserve the next slot (last_call = last_call + interval)
  266:         # BEFORE releasing the lock, so parallel threads see the reserved
  267:         # time and calculate a longer wait instead of stampeding.
  268:         wait_time = 0.0
  269:         with self._lock:
  270:             now = time.time()
  271:             elapsed = now - self.last_call
  272:             if elapsed < self.interval:
  273:                 wait_time = self.interval - elapsed
  274:             # Reserve the next slot atomically — even if we haven't slept yet,
  275:             # the next thread to enter will see this and wait longer.
  276:             self.last_call = self.last_call + self.interval if wait_time > 0 else now
  277:         if wait_time > 0:
  278:             time.sleep(wait_time)
  279: 
  280: 
  281: # H11/DC-R3-4: yfinance is not thread-safe. This lock is shared across all
  282: # modules (fear_greed, golddigger/data_provider, data_collector) so that
  283: # concurrent calls from the 8-worker ThreadPoolExecutor are serialized.
  284: # data_collector.py imports this lock instead of defining its own.
  285: yfinance_lock = threading.Lock()
  286: 
  287: # Alpaca IEX: 200 req/min → target 150/min to leave headroom
  288: _alpaca_limiter = _RateLimiter(150, "alpaca")
  289: # Binance: 1200 weight/min → very generous, but space out slightly
  290: _binance_limiter = _RateLimiter(600, "binance")
  291: # Yahoo Finance (yfinance): no official limit, but be polite — 30/min
  292: _yfinance_limiter = _RateLimiter(30, "yfinance")
  293: 
  294: 
  295: # Alpha Vantage: 5 req/min free tier
  296: _alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")
  297: 
  298: 
  299: # NewsAPI: 100 req/day free tier — tiered priority system
  300: # Budget: metals (XAU, XAG) get 20-min refresh during active hours (~84/day)
  301: # All other tickers: Yahoo-only (0 NewsAPI calls)
  302: # BTC/ETH: already served by CryptoCompare, not NewsAPI
  303: _newsapi_daily_count = 0
  304: _newsapi_daily_reset = 0.0  # timestamp of last reset
  305: _NEWSAPI_DAILY_BUDGET = 90  # leave 10-call margin
  306: _newsapi_lock = threading.Lock()
  307: 
  308: # Tier 1 = 20-min TTL during active hours; Tier 2 = 3h; rest = Yahoo-only
  309: _NEWSAPI_PRIORITY = {"XAU": 1, "XAG": 1, "MSTR": 2}
  310: 
  311: # Better search queries — raw ticker symbols return sparse results on NewsAPI
  312: _NEWSAPI_SEARCH_QUERIES = {
  313:     "XAU": "gold AND (price OR market OR ounce OR bullion OR futures OR commodity)",
  314:     "XAG": "silver AND (price OR market OR ounce OR bullion OR futures OR commodity)",
  315:     "MSTR": "MicroStrategy OR MSTR",
  316: }
  317: 
  318: # Active monitoring: 08:00-22:00 CET = 07:00-21:00 UTC
  319: _NEWSAPI_ACTIVE_START_UTC = 7
  320: _NEWSAPI_ACTIVE_END_UTC = 21
  321: 
  322: 
  323: def newsapi_quota_ok() -> bool:
  324:     """Check if we still have NewsAPI quota today. Thread-safe."""
  325:     global _newsapi_daily_count, _newsapi_daily_reset
  326:     now = time.time()
  327:     with _newsapi_lock:
  328:         # Reset counter at midnight UTC
  329:         from datetime import datetime
  330:         today_start = datetime.now(UTC).replace(
  331:             hour=0, minute=0, second=0, microsecond=0
  332:         ).timestamp()
  333:         if _newsapi_daily_reset < today_start:
  334:             _newsapi_daily_count = 0
  335:             _newsapi_daily_reset = now
  336:         return _newsapi_daily_count < _NEWSAPI_DAILY_BUDGET
  337: 
  338: 
  339: def newsapi_track_call():
  340:     """Increment NewsAPI daily counter. Call after each successful API request."""
=== portfolio/file_utils.py:1-260 ===
    1: """Shared file I/O utilities."""
    2: import json
    3: import logging
    4: import os
    5: import tempfile
    6: from collections import deque
    7: from contextlib import suppress
    8: from pathlib import Path
    9: 
   10: # Cross-platform file-locking primitives for `atomic_append_jsonl`.
   11: # Same pattern as `portfolio/process_lock.py`.
   12: try:
   13:     import msvcrt as _msvcrt  # type: ignore[import-not-found]
   14: except ImportError:  # pragma: no cover - non-Windows
   15:     _msvcrt = None  # type: ignore[assignment]
   16: try:
   17:     import fcntl as _fcntl  # type: ignore[import-not-found]
   18: except ImportError:  # pragma: no cover - Windows
   19:     _fcntl = None  # type: ignore[assignment]
   20: 
   21: logger = logging.getLogger("portfolio.file_utils")
   22: 
   23: 
   24: def atomic_write_text(path, text, encoding="utf-8"):
   25:     """Atomically write text to a file using tempfile + os.replace.
   26: 
   27:     Same safety guarantees as atomic_write_json: fsync before replace,
   28:     no partial writes on crash.
   29:     """
   30:     path = Path(path)
   31:     path.parent.mkdir(parents=True, exist_ok=True)
   32:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   33:     try:
   34:         with os.fdopen(fd, "w", encoding=encoding) as f:
   35:             f.write(text)
   36:             f.flush()
   37:             os.fsync(f.fileno())
   38:         os.replace(tmp, str(path))
   39:     except BaseException:
   40:         with suppress(OSError):
   41:             os.unlink(tmp)
   42:         raise
   43: 
   44: 
   45: def atomic_write_json(path, data, indent=2, ensure_ascii=True):
   46:     """Atomically write JSON data to a file using tempfile + os.replace.
   47: 
   48:     Ensures the file is never left in a partially-written state.
   49:     Fsyncs before replace to guarantee durability on power loss (H34).
   50:     """
   51:     path = Path(path)
   52:     path.parent.mkdir(parents=True, exist_ok=True)
   53:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   54:     try:
   55:         with os.fdopen(fd, "w", encoding="utf-8") as f:
   56:             json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
   57:             f.flush()
   58:             os.fsync(f.fileno())
   59:         os.replace(tmp, str(path))
   60:     except BaseException:
   61:         with suppress(OSError):
   62:             os.unlink(tmp)
   63:         raise
   64: 
   65: 
   66: def load_json(path, default=None):
   67:     """Load a JSON file. Returns *default* if missing or unparseable.
   68: 
   69:     Uses try/except instead of exists() check to avoid TOCTOU race.
   70:     Handles OSError (permission denied, locked files) gracefully on Windows.
   71:     Logs WARNING on corrupt JSON so corruption is observable (H35).
   72:     """
   73:     path = Path(path)
   74:     try:
   75:         return json.loads(path.read_text(encoding="utf-8"))
   76:     except FileNotFoundError:
   77:         return default
   78:     except OSError:
   79:         # BUG-139: PermissionError (file locked by antivirus/another process)
   80:         # and other OS-level errors should degrade gracefully like missing files.
   81:         logger.debug("load_json: OS error reading %s, returning default", path.name)
   82:         return default
   83:     except (json.JSONDecodeError, ValueError):
   84:         # H35: Log corruption so it's observable — silent defaults hide data loss.
   85:         logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
   86:         return default
   87: 
   88: 
   89: def require_json(path):
   90:     """Load a JSON file, raising on corruption or missing file.
   91: 
   92:     Unlike load_json(), this function does NOT silently return defaults.
   93:     Use for critical files where corruption must be surfaced (H35).
   94: 
   95:     Raises:
   96:         FileNotFoundError: If the file does not exist.
   97:         json.JSONDecodeError: If the file contains invalid JSON.
   98:         OSError: If the file cannot be read.
   99:     """
  100:     path = Path(path)
  101:     return json.loads(path.read_text(encoding="utf-8"))
  102: 
  103: 
  104: def load_jsonl(path, limit=None):
  105:     """Load entries from a JSONL file.
  106: 
  107:     Args:
  108:         path: Path to the .jsonl file.
  109:         limit: If set, keep only the *last* N entries (uses a deque).
  110: 
  111:     Returns:
  112:         list of parsed dicts. Empty list if file missing or empty.
  113:     """
  114:     path = Path(path)
  115:     container = deque(maxlen=limit) if limit else []
  116:     try:
  117:         f = open(path, encoding="utf-8")
  118:     except FileNotFoundError:
  119:         return []
  120:     except OSError as e:
  121:         logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
  122:         return []
  123:     with f:
  124:         for line in f:
  125:             line = line.strip()
  126:             if not line:
  127:                 continue
  128:             try:
  129:                 container.append(json.loads(line))
  130:             except json.JSONDecodeError as e:
  131:                 logger.debug("Skipping malformed JSONL line in %s: %s", path.name, str(e)[:100])
  132:                 continue
  133:     return list(container)
  134: 
  135: 
  136: def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
  137:     """Load the last N entries from a JSONL file by reading from the end.
  138: 
  139:     Much more efficient than load_jsonl(limit=N) for large files because
  140:     it only reads the last `tail_bytes` bytes instead of the entire file.
  141: 
  142:     Args:
  143:         path: Path to the .jsonl file.
  144:         max_entries: Maximum entries to return.
  145:         tail_bytes: How many bytes to read from the end of the file.
  146:             Default 512KB ≈ ~1000 typical entries.
  147: 
  148:     Returns:
  149:         list of parsed dicts (chronological order). Empty list if missing.
  150:     """
  151:     path = Path(path)
  152:     try:
  153:         file_size = path.stat().st_size
  154:     except (FileNotFoundError, OSError):
  155:         return []
  156:     if file_size == 0:
  157:         return []
  158: 
  159:     entries = []
  160:     try:
  161:         with open(path, "rb") as f:
  162:             # Seek to near end of file
  163:             offset = max(0, file_size - tail_bytes)
  164:             # 2026-05-04 codex P3-1 follow-up: peek the byte just before
  165:             # the seek point. If it's a newline, the seek lands exactly
  166:             # at a line boundary and the first decoded line is intact.
  167:             # Without this check, a happy-coincidence boundary would
  168:             # cost us one valid entry on every read.
  169:             seek_on_boundary = False
  170:             if offset > 0:
  171:                 f.seek(offset - 1)
  172:                 prior = f.read(1)
  173:                 seek_on_boundary = prior == b"\n"
  174:             f.seek(offset)
  175:             data = f.read()
  176:         # Decode and split into lines
  177:         text = data.decode("utf-8", errors="replace")
  178:         lines = text.split("\n")
  179:         # Drop the first line only when we landed mid-line. When seek
  180:         # lands on a newline boundary, the first decoded line is
  181:         # complete and should be kept.
  182:         if offset > 0 and lines and not seek_on_boundary:
  183:             lines = lines[1:]
  184:         for line in lines:
  185:             line = line.strip()
  186:             if not line:
  187:                 continue
  188:             try:
  189:                 entries.append(json.loads(line))
  190:             except json.JSONDecodeError:
  191:                 continue
  192:     except (OSError, UnicodeDecodeError) as e:
  193:         logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
  194:         return []
  195: 
  196:     # Return last max_entries in chronological order
  197:     if len(entries) > max_entries:
  198:         entries = entries[-max_entries:]
  199:     return entries
  200: 
  201: 
  202: def atomic_append_jsonl(path, entry):
  203:     """Append a single JSON entry to a JSONL file with atomic semantics
  204:     across threads and processes.
  205: 
  206:     Implementation: binary-append (``"ab"``) to the target + an
  207:     exclusive lock on a *sidecar* lockfile held for the duration of
  208:     the ``write + flush + fsync`` sequence. Windows CRT does not
  209:     guarantee ``O_APPEND`` atomicity (unlike POSIX), so without a lock
  210:     heavy thread contention can produce torn lines (head bytes lost,
  211:     tail bytes survive).
  212: 
  213:     Sidecar-lockfile pattern (``<path>.lock``) — not the target file
  214:     itself — guarantees a non-empty, lockable byte-range exists even
  215:     when the target file is brand-new / size 0. This closes the race
  216:     window Codex flagged on 2026-04-17: two first-writers opening
  217:     the freshly-created target simultaneously could both have
  218:     failed the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` call and
  219:     interleaved their writes.
  220: 
  221:     This primitive is used by ~20 JSONL writers across the codebase
  222:     (signal_log, claude_invocations, critical_errors, telegram_messages,
  223:     accuracy_snapshots, etc.) so the fix eliminates torn-line risk
  224:     system-wide. Unxfails
  225:     ``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``.
  226:     """
  227:     path = Path(path)
  228:     path.parent.mkdir(parents=True, exist_ok=True)
  229:     data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
  230: 
  231:     # Sidecar lockfile — always non-empty so locking never fails on
  232:     # size-0 targets. Pre-create if missing; single byte is enough.
  233:     lock_path = path.parent / f".{path.name}.lock"
  234:     if not lock_path.exists():
  235:         try:
  236:             with open(lock_path, "ab") as lf:
  237:                 if lf.tell() == 0:
  238:                     lf.write(b"\0")
  239:         except OSError:
  240:             pass  # best-effort; lock open below will retry
  241: 
  242:     with open(lock_path, "rb+") as lock_f:
  243:         lfd = lock_f.fileno()
  244:         win_locked = False
  245:         try:
  246:             if _msvcrt is not None:
  247:                 os.lseek(lfd, 0, os.SEEK_SET)
  248:                 _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
  249:                 win_locked = True
  250:             elif _fcntl is not None:
  251:                 _fcntl.flock(lfd, _fcntl.LOCK_EX)
  252:             with open(path, "ab") as f:
  253:                 f.write(data)
  254:                 f.flush()
  255:                 os.fsync(f.fileno())
  256:         finally:
  257:             if win_locked and _msvcrt is not None:
  258:                 try:
  259:                     os.lseek(lfd, 0, os.SEEK_SET)
  260:                     _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
=== portfolio/gpu_gate.py:1-220 ===
    1: """GPU gating and VRAM monitoring for LLM model inference.
    2: 
    3: Ensures only one LLM model uses the GPU at a time. Logs VRAM usage
    4: before and after each model load for monitoring.
    5: 
    6: Uses a threading lock for in-process concurrency (ThreadPoolExecutor workers)
    7: plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.
    8: 
    9: Stale-lock recovery (2026-05-03):
   10: - Reactive: ``gpu_gate()`` calls ``_try_break_stale_lock()`` when another
   11:   caller blocks on the lock — same predicate as before BUG-182.
   12: - Background: a daemon thread (lazily spawned on first ``gpu_gate()`` call)
   13:   runs the same predicate every 30 s. This closes the liveness hole that
   14:   let the loop wedge for ~25 hours after chronos pid 13152 died holding
   15:   the lock 2026-05-02 02:14 (no other acquirer = no break = no recovery).
   16:   See ``docs/plans/2026-05-03-gpu-gate-sweeper.md``.
   17: """
   18: 
   19: import logging
   20: import os
   21: import subprocess
   22: import threading
   23: import time
   24: from contextlib import contextmanager, suppress
   25: from pathlib import Path
   26: 
   27: logger = logging.getLogger("portfolio.gpu_gate")
   28: 
   29: # In-process lock — prevents ThreadPoolExecutor workers from racing
   30: _THREAD_LOCK = threading.Lock()
   31: 
   32: # File-based lock for cross-process protection
   33: _GPU_LOCK_DIR = Path("Q:/models")
   34: _GPU_LOCK_FILE = _GPU_LOCK_DIR / ".gpu_lock"
   35: _STALE_SECONDS = 300  # 5 min
   36: 
   37: # Stale-lock sweeper daemon (2026-05-03). Module-level singleton so subprocess
   38: # workers that import this module only spawn one sweeper, not one per import.
   39: _SWEEPER_INTERVAL_SECONDS = 30
   40: _SWEEPER_LOCK = threading.Lock()
   41: _sweeper_thread: "threading.Thread | None" = None
   42: 
   43: 
   44: def get_vram_usage() -> dict:
   45:     """Query nvidia-smi for current VRAM usage. Returns dict or None on error."""
   46:     try:
   47:         proc = subprocess.run(
   48:             ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
   49:              "--format=csv,noheader,nounits"],
   50:             capture_output=True, text=True, timeout=5,
   51:         )
   52:         if proc.returncode == 0 and proc.stdout.strip():
   53:             parts = [p.strip() for p in proc.stdout.strip().split(",")]
   54:             if len(parts) >= 4:
   55:                 return {
   56:                     "used_mb": int(parts[0]),
   57:                     "free_mb": int(parts[1]),
   58:                     "total_mb": int(parts[2]),
   59:                     "gpu_util_pct": int(parts[3]),
   60:                 }
   61:     except Exception:
   62:         logger.debug("GPU info query failed", exc_info=True)
   63:     return None
   64: 
   65: 
   66: def _is_stale() -> bool:
   67:     try:
   68:         return (time.time() - _GPU_LOCK_FILE.stat().st_mtime) > _STALE_SECONDS
   69:     except OSError:
   70:         return True
   71: 
   72: 
   73: def _pid_alive(pid: int) -> bool:
   74:     """Check if a process is still running. BUG-182."""
   75:     if not pid or pid < 0:
   76:         return False
   77:     try:
   78:         import psutil
   79:         return psutil.pid_exists(pid)
   80:     except ImportError:
   81:         # Fallback: assume alive if we can't check
   82:         return True
   83: 
   84: 
   85: def _read_lock() -> dict:
   86:     try:
   87:         text = _GPU_LOCK_FILE.read_text(encoding="utf-8").strip()
   88:         parts = text.split("|")
   89:         return {
   90:             "model": parts[0] if len(parts) > 0 else "unknown",
   91:             "pid": int(parts[1]) if len(parts) > 1 else 0,
   92:             "ts": float(parts[2]) if len(parts) > 2 else 0,
   93:         }
   94:     except (OSError, ValueError):
   95:         return {}
   96: 
   97: 
   98: def _write_lock(model_name: str):
   99:     _GPU_LOCK_FILE.write_text(
  100:         f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}",
  101:         encoding="utf-8",
  102:     )
  103: 
  104: 
  105: def _release_lock():
  106:     with suppress(OSError):
  107:         _GPU_LOCK_FILE.unlink(missing_ok=True)
  108: 
  109: 
  110: def _try_break_stale_lock() -> bool:
  111:     """Reap the lock file iff stale-by-mtime AND owner pid is dead.
  112: 
  113:     Returns True if the lock was broken (caller can retry acquire), False
  114:     otherwise. Defensive: never raises — the sweeper daemon depends on this.
  115: 
  116:     Called from two paths:
  117:     - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
  118:     - Sweeper: the background daemon, when no one is waiting.
  119: 
  120:     Both paths must agree on the predicate so behaviour is identical
  121:     regardless of which path reaped the lock. Emits the same
  122:     ``Breaking stale GPU lock`` warning either way so log-grep tools and
  123:     postmortem audits work uniformly.
  124:     """
  125:     try:
  126:         if not _GPU_LOCK_FILE.exists():
  127:             return False
  128:         if not _is_stale():
  129:             return False
  130:         info = _read_lock()
  131:         pid = info.get("pid", 0)
  132:         if _pid_alive(pid):
  133:             return False
  134:         logger.warning("Breaking stale GPU lock: %s (pid=%s, dead)",
  135:                        info.get("model"), pid)
  136:         _release_lock()
  137:         return True
  138:     except Exception as exc:
  139:         # The sweeper must NEVER crash — a dead daemon stops sweeping forever.
  140:         logger.debug("Stale-lock sweep error: %s", exc)
  141:         return False
  142: 
  143: 
  144: def _sweeper_loop():
  145:     """Background daemon: reap stale-dead locks every 30 s.
  146: 
  147:     Wedge-recovery story (2026-05-02): chronos pid 13152 died holding the
  148:     lock at 02:14. No one tried to acquire while the loop was stuck inside
  149:     its LLM batch, so ``_is_stale()`` was never checked. Loop wedged for
  150:     ~25 hours until a system reboot. This daemon closes that hole.
  151:     """
  152:     while True:
  153:         try:
  154:             time.sleep(_SWEEPER_INTERVAL_SECONDS)
  155:             _try_break_stale_lock()
  156:         except Exception as exc:
  157:             # Defence-in-depth — _try_break_stale_lock already swallows but
  158:             # any future code added here must also keep the daemon alive.
  159:             logger.debug("Sweeper loop error: %s", exc)
  160: 
  161: 
  162: def _start_sweeper():
  163:     """Spawn the sweeper daemon (idempotent, thread-safe).
  164: 
  165:     Lazily called from ``gpu_gate()`` so:
  166:     - Subprocess workers that import this module but never call
  167:       ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
  168:       do NOT spawn a redundant daemon.
  169:     - Tests can reset ``_sweeper_thread = None`` and re-trigger spawn.
  170: 
  171:     If the daemon ever dies (it shouldn't — both layers swallow exceptions)
  172:     a future call will respawn it.
  173:     """
  174:     global _sweeper_thread
  175:     with _SWEEPER_LOCK:
  176:         if _sweeper_thread is None or not _sweeper_thread.is_alive():
  177:             t = threading.Thread(
  178:                 target=_sweeper_loop,
  179:                 name="gpu-gate-sweeper",
  180:                 daemon=True,
  181:             )
  182:             _sweeper_thread = t
  183:             t.start()
  184: 
  185: 
  186: @contextmanager
  187: def gpu_gate(model_name: str, timeout: float = 60):
  188:     """Acquire exclusive GPU access, log VRAM before/after.
  189: 
  190:     Uses a two-layer lock:
  191:     1. threading.Lock for in-process concurrency (ThreadPoolExecutor workers)
  192:     2. File-based lock for cross-process protection (metals loop, etc.)
  193: 
  194:     Args:
  195:         model_name: e.g. "ministral-3", "qwen3", "chronos"
  196:         timeout: max seconds to wait for lock
  197: 
  198:     Yields:
  199:         True if acquired, False if timed out.
  200:     """
  201:     # Lazy-spawn the stale-lock sweeper. Idempotent so no cost after the
  202:     # first call. See _start_sweeper() for the rationale.
  203:     _start_sweeper()
  204: 
  205:     deadline = time.time() + timeout
  206: 
  207:     # Layer 1: In-process thread lock (prevents ThreadPoolExecutor races)
  208:     remaining = deadline - time.time()
  209:     thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
  210:     if not thread_acquired:
  211:         logger.warning("GPU thread-lock timeout (%ss) for %s", timeout, model_name)
  212:         yield False
  213:         return
  214: 
  215:     try:
  216:         # Layer 2: File-based lock (cross-process)
  217:         file_acquired = False
  218:         while time.time() < deadline:
  219:             try:
  220:                 # Atomic create — fails if file already exists (no TOCTOU race)
=== portfolio/process_lock.py:1-180 ===
    1: """Cross-platform helpers for non-blocking singleton process locks."""
    2: 
    3: from __future__ import annotations
    4: 
    5: import contextlib
    6: import os
    7: from datetime import UTC, datetime
    8: from pathlib import Path
    9: from typing import IO
   10: 
   11: try:
   12:     import msvcrt  # type: ignore[attr-defined]
   13: except ImportError:  # pragma: no cover - non-Windows
   14:     msvcrt = None
   15: 
   16: try:
   17:     import fcntl  # type: ignore[import-not-found]
   18: except ImportError:  # pragma: no cover - Windows
   19:     fcntl = None
   20: 
   21: 
   22: def acquire_lock_file(
   23:     lock_path: str | Path,
   24:     *,
   25:     owner: str = "",
   26:     metadata: dict | None = None,
   27: ) -> IO[str] | None:
   28:     """Acquire a non-blocking file lock and return the open handle.
   29: 
   30:     Returns None if another process already holds the lock.
   31:     """
   32:     path = Path(lock_path)
   33:     if path.parent:
   34:         path.parent.mkdir(parents=True, exist_ok=True)
   35: 
   36:     fh = path.open("a+", encoding="utf-8")
   37:     try:
   38:         _lock_file(fh)
   39:     except OSError:
   40:         fh.close()
   41:         return None
   42: 
   43:     _write_lock_metadata(fh, owner=owner, metadata=metadata)
   44:     return fh
   45: 
   46: 
   47: def release_lock_file(fh: IO[str] | None) -> None:
   48:     """Release a previously acquired lock handle."""
   49:     if fh is None:
   50:         return
   51:     try:
   52:         _unlock_file(fh)
   53:     except OSError:
   54:         pass
   55:     finally:
   56:         with contextlib.suppress(Exception):
   57:             fh.close()
   58: 
   59: 
   60: def _lock_file(fh: IO[str]) -> None:
   61:     fh.seek(0)
   62:     if msvcrt is not None:
   63:         msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
   64:         return
   65:     if fcntl is not None:  # pragma: no branch - platform-specific
   66:         fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
   67: 
   68: 
   69: def _unlock_file(fh: IO[str]) -> None:
   70:     fh.seek(0)
   71:     if msvcrt is not None:
   72:         msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
   73:         return
   74:     if fcntl is not None:  # pragma: no branch - platform-specific
   75:         fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
   76: 
   77: 
   78: def _write_lock_metadata(
   79:     fh: IO[str],
   80:     *,
   81:     owner: str = "",
   82:     metadata: dict | None = None,
   83: ) -> None:
   84:     payload = {
   85:         "pid": os.getpid(),
   86:         "started": datetime.now(UTC).isoformat(),
   87:     }
   88:     if owner:
   89:         payload["owner"] = owner
   90:     if metadata:
   91:         payload.update({str(k): v for k, v in metadata.items() if v is not None})
   92: 
   93:     try:
   94:         fh.seek(0)
   95:         fh.truncate()
   96:         fh.write(" ".join(f"{key}={value}" for key, value in payload.items()) + "\n")
   97:         fh.flush()
   98:     except Exception:
   99:         pass

 succeeded in 794ms:
=== dashboard/app.py:120-260 ===
  120:     under-delivered) entries don't survive the deploy.
  121:     """
  122:     if limit and limit > 0:
  123:         return _cached_read(
  124:             f"jsonl_tail_v2:{path}:{limit}",
  125:             ttl,
  126:             lambda: _read_tail_with_growth(path, limit),
  127:         )
  128:     return _cached_read(
  129:         f"jsonl:{path}:{limit}", ttl, lambda: _load_jsonl_impl(path, limit=limit)
  130:     )
  131: 
  132: 
  133: def _read_tail_with_growth(path, limit):
  134:     """Read tail entries, doubling tail_bytes until we have `limit`
  135:     parsed rows or the whole file has been consumed.
  136: 
  137:     Falls back to the full-scan load_jsonl path if even reading the
  138:     full file via the tail helper still yields < limit entries —
  139:     that case implies the tail helper's first-line-drop heuristic is
  140:     chewing through real data and we should bypass it entirely.
  141:     """
  142:     try:
  143:         file_size = Path(path).stat().st_size
  144:     except (FileNotFoundError, OSError):
  145:         return []
  146:     if file_size == 0:
  147:         return []
  148: 
  149:     # Initial budget: ~1 KB per entry with a 512 KB floor.
  150:     tail_bytes = max(512_000, limit * 1024)
  151:     # Cap retry budget at 64 MB to avoid runaway reads on a corrupt or
  152:     # absurdly-sized file. Most logs in this codebase are < 100 MB and
  153:     # 64 MB will hold ~64 K typical-sized entries.
  154:     max_retry_bytes = 64 * 1024 * 1024
  155:     while True:
  156:         capped = min(tail_bytes, file_size, max_retry_bytes)
  157:         rows = _load_jsonl_tail_impl(path, max_entries=limit,
  158:                                        tail_bytes=capped)
  159:         if len(rows) >= limit or capped >= file_size or capped >= max_retry_bytes:
  160:             break
  161:         tail_bytes *= 2
  162: 
  163:     # Last-chance fallback: if even the full-file tail came up short,
  164:     # the issue isn't byte budget — it's the first-line-drop heuristic.
  165:     # Fall through to the canonical full-scan reader.
  166:     if len(rows) < limit and capped >= file_size:
  167:         rows = _load_jsonl_impl(path, limit=limit)
  168:     return rows
  169: 
  170: 
  171: def _get_config():
  172:     return _read_json(CONFIG_PATH, ttl=60) or {}
  173: 
  174: 
  175: def _parse_limit_arg(name, default, max_value):
  176:     """Parse integer query arg with sane bounds and fallback."""
  177:     try:
  178:         value = int(request.args.get(name, default))
  179:     except (ValueError, TypeError):
  180:         value = default
  181:     return max(1, min(value, max_value))
  182: 
  183: 
  184: def _iter_latest_dict_entries(path, read_limit):
  185:     """Yield JSONL entries newest-first, skipping non-dict shapes."""
  186:     raw = _read_jsonl(path, limit=read_limit)
  187:     for entry in reversed(raw):
  188:         if isinstance(entry, dict):
  189:             yield entry
  190: 
  191: 
  192: def _parse_iso8601(value):
  193:     """Parse an ISO-8601 timestamp into an aware datetime."""
  194:     if not value or not isinstance(value, str):
  195:         return None
  196:     try:
  197:         dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
  198:     except ValueError:
  199:         return None
  200:     if dt.tzinfo is None:
  201:         dt = dt.replace(tzinfo=UTC)
  202:     return dt
  203: 
  204: 
  205: def _stockholm_now():
  206:     return datetime.now(UTC).astimezone(STOCKHOLM_TZ)
  207: 
  208: 
  209: def _hours_until_stockholm_close(now=None, close_hour=21, close_minute=55):
  210:     """Return hours remaining until the Stockholm warrant close."""
  211:     now = (now or _stockholm_now()).astimezone(STOCKHOLM_TZ)
  212:     close_dt = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
  213:     if now >= close_dt:
  214:         return 0.0
  215:     return round((close_dt - now).total_seconds() / 3600.0, 2)
  216: 
  217: 
  218: def _is_number(value):
  219:     return isinstance(value, (int, float)) and math.isfinite(value)
  220: 
  221: 
  222: def _round_or_none(value, digits=2):
  223:     return round(float(value), digits) if _is_number(value) else None
  224: 
  225: 
  226: def _normalize_golddigger_position(raw_position, latest_log):
  227:     if not isinstance(raw_position, dict):
  228:         return None
  229: 
  230:     quantity = raw_position.get("quantity", raw_position.get("shares"))
  231:     entry_price = raw_position.get("avg_price", raw_position.get("entry_price"))
  232:     current_price = None
  233:     if isinstance(latest_log, dict):
  234:         current_price = latest_log.get("cert_bid", latest_log.get("cert_ask"))
  235:     if current_price is None:
  236:         current_price = raw_position.get("current_price")
  237:     pnl_pct = None
  238:     if _is_number(entry_price) and entry_price > 0 and _is_number(current_price):
  239:         pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
  240: 
  241:     position = dict(raw_position)
  242:     position["shares"] = quantity
  243:     position["quantity"] = quantity
  244:     position["side"] = raw_position.get("side") or raw_position.get("action") or "BUY"
  245:     position["entry_price"] = entry_price
  246:     position["avg_price"] = entry_price
  247:     position["stop_price"] = raw_position.get("stop_price", raw_position.get("stop"))
  248:     position["tp_price"] = raw_position.get("tp_price", raw_position.get("take_profit_price"))
  249:     position["take_profit_price"] = position["tp_price"]
  250:     position["current_price"] = current_price
  251:     position["pnl_pct"] = _round_or_none(
  252:         raw_position.get("pnl_pct") if raw_position.get("pnl_pct") is not None else pnl_pct,
  253:         2,
  254:     )
  255: 
  256:     has_position = any(
  257:         _is_number(value)
  258:         for value in (quantity, entry_price, current_price)
  259:     )
  260:     return position if has_position else None
=== dashboard/app.py:1000-1085 ===
 1000:             ca = get_or_compute_consensus_accuracy(horizon)
 1001:             ta = get_or_compute_per_ticker_accuracy(horizon)
 1002:             # ca/sa/ta may be None when the underlying cache miss returned
 1003:             # no data (cold cache + no signal-log entries yet); skip those
 1004:             # horizons entirely so the response stays well-formed.
 1005:             if ca and ca.get("total", 0) > 0:
 1006:                 result[horizon] = {
 1007:                     "signals": _enrich_signals(sa or {}),
 1008:                     "consensus": ca,
 1009:                     "per_ticker": ta or {},
 1010:                 }
 1011:         _API_ACCURACY_CACHE["data"] = result
 1012:         _API_ACCURACY_CACHE["ts"] = now
 1013:         return jsonify(result)
 1014:     except Exception:
 1015:         logger.exception("accuracy endpoint error")
 1016:         return jsonify({"error": "Internal server error"}), 500
 1017: 
 1018: 
 1019: @app.route("/api/iskbets")
 1020: @require_auth
 1021: def api_iskbets():
 1022:     config = _read_json(DATA_DIR / "iskbets_config.json")
 1023:     state = _read_json(DATA_DIR / "iskbets_state.json")
 1024:     return jsonify({"config": config, "state": state})
 1025: 
 1026: 
 1027: @app.route("/api/lora-status")
 1028: @require_auth
 1029: def api_lora_status():
 1030:     state = _read_json(TRAINING_DIR / "state.json")
 1031:     progress = _read_json(TRAINING_DIR / "training_progress.json")
 1032:     return jsonify({"state": state, "training_progress": progress})
 1033: 
 1034: 
 1035: # ---------------------------------------------------------------------------
 1036: # New: Portfolio validation
 1037: # ---------------------------------------------------------------------------
 1038: 
 1039: @app.route("/api/validate-portfolio", methods=["POST"])
 1040: @require_auth
 1041: def api_validate_portfolio():
 1042:     """Validate a portfolio JSON for integrity.
 1043: 
 1044:     Delegates to portfolio_validator.validate_portfolio() which performs
 1045:     comprehensive checks: cash, holdings, fees, transactions, avg_cost.
 1046:     """
 1047:     data = request.get_json(silent=True)
 1048:     if not data:
 1049:         return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400
 1050: 
 1051:     try:
 1052:         from portfolio.portfolio_validator import validate_portfolio
 1053:         errors = validate_portfolio(data)
 1054:     except Exception as e:
 1055:         return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500
 1056: 
 1057:     return jsonify({
 1058:         "valid": len(errors) == 0,
 1059:         "errors": errors,
 1060:     })
 1061: 
 1062: 
 1063: # ---------------------------------------------------------------------------
 1064: # New: Equity curve
 1065: # ---------------------------------------------------------------------------
 1066: 
 1067: @app.route("/api/equity-curve")
 1068: @require_auth
 1069: def api_equity_curve():
 1070:     """Return portfolio value history for charting.
 1071: 
 1072:     Reads data/portfolio_value_history.jsonl. Returns empty array if missing.
 1073:     """
 1074:     entries = _read_jsonl(DATA_DIR / "portfolio_value_history.jsonl", limit=5000)
 1075:     return jsonify(entries)
 1076: 
 1077: 
 1078: # ---------------------------------------------------------------------------
 1079: # New: Signal heatmap (30 signals x all tickers)
 1080: # ---------------------------------------------------------------------------
 1081: 
 1082: @app.route("/api/signal-heatmap")
 1083: @require_auth
 1084: def api_signal_heatmap():
 1085:     """Return the full 30-signal x all-tickers grid.
=== portfolio/subprocess_utils.py:110-330 ===
  110:     return job, kernel32
  111: 
  112: 
  113: def _run_with_job_object(cmd, **kwargs):
  114:     """Internal: run subprocess inside a Windows Job Object."""
  115:     job, kernel32 = _create_job_object()
  116: 
  117:     try:
  118:         popen_kwargs = dict(kwargs)
  119:         timeout = popen_kwargs.pop("timeout", None)
  120: 
  121:         if popen_kwargs.pop("capture_output", False):
  122:             popen_kwargs["stdout"] = subprocess.PIPE
  123:             popen_kwargs["stderr"] = subprocess.PIPE
  124: 
  125:         input_data = popen_kwargs.pop("input", None)
  126:         if input_data is not None and "stdin" not in popen_kwargs:
  127:             popen_kwargs["stdin"] = subprocess.PIPE
  128: 
  129:         proc = subprocess.Popen(cmd, **popen_kwargs)
  130: 
  131:         try:
  132:             kernel32.AssignProcessToJobObject(job, int(proc._handle))
  133:         except Exception as e:
  134:             logger.warning(
  135:                 "Job Object assignment failed for pid %d — child may orphan: %s",
  136:                 proc.pid, e,
  137:             )
  138: 
  139:         try:
  140:             stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
  141:         except subprocess.TimeoutExpired:
  142:             proc.kill()
  143:             proc.communicate()
  144:             raise
  145: 
  146:         return subprocess.CompletedProcess(
  147:             args=cmd,
  148:             returncode=proc.returncode,
  149:             stdout=stdout,
  150:             stderr=stderr,
  151:         )
  152:     finally:
  153:         kernel32.CloseHandle(job)
  154: 
  155: 
  156: def popen_in_job(cmd, **kwargs):
  157:     """Start a long-running subprocess inside a Windows Job Object.
  158: 
  159:     Like subprocess.Popen(), but assigns the child to a Job Object with
  160:     JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. If the parent process dies (crash,
  161:     kill -9, power loss), the OS automatically kills the child.
  162: 
  163:     Returns (proc, job_handle) on Windows. On non-Windows or if Job Object
  164:     creation fails, returns (proc, None).
  165: 
  166:     Caller must call close_job(job_handle) when explicitly stopping the child.
  167:     """
  168:     proc = subprocess.Popen(cmd, **kwargs)
  169: 
  170:     if sys.platform != "win32":
  171:         return proc, None
  172: 
  173:     try:
  174:         job, kernel32 = _create_job_object()
  175:         kernel32.AssignProcessToJobObject(job, int(proc._handle))
  176:         return proc, job
  177:     except Exception as exc:
  178:         logger.debug("Job Object creation failed for Popen (%s), no auto-cleanup", exc)
  179:         return proc, None
  180: 
  181: 
  182: def close_job(job_handle):
  183:     """Close a Job Object handle.
  184: 
  185:     Safe to call after the child has already been terminated — closing the
  186:     handle on a dead process is a no-op. Call this in your explicit stop
  187:     function after terminating the child.
  188:     """
  189:     if job_handle is None:
  190:         return
  191:     try:
  192:         import ctypes
  193:         ctypes.windll.kernel32.CloseHandle(job_handle)
  194:     except Exception:
  195:         pass
  196: 
  197: 
  198: def kill_orphaned_by_cmdline(pattern, exclude_pid=None):
  199:     """Find and kill processes whose command line contains *pattern*.
  200: 
  201:     Used at startup to sweep orphaned subprocesses from a previous crash.
  202:     Skips the current process and *exclude_pid* if given.
  203: 
  204:     Returns the number of processes killed. Returns 0 on non-Windows.
  205:     """
  206:     if sys.platform != "win32":
  207:         return 0
  208: 
  209:     my_pid = __import__("os").getpid()
  210:     skip = {my_pid}
  211:     if exclude_pid is not None:
  212:         skip.add(exclude_pid)
  213: 
  214:     try:
  215:         result = subprocess.run(
  216:             ["wmic", "process", "where",
  217:              f"CommandLine like '%{pattern}%'",
  218:              "get", "ProcessId", "/format:csv"],
  219:             capture_output=True, text=True, timeout=15,
  220:         )
  221:     except Exception as exc:
  222:         logger.debug("WMIC process query failed: %s", exc)
  223:         return 0
  224: 
  225:     killed = 0
  226:     for line in result.stdout.splitlines():
  227:         parts = line.strip().split(",")
  228:         if len(parts) < 2:
  229:             continue
  230:         try:
  231:             pid = int(parts[-1])
  232:         except ValueError:
  233:             continue
  234:         if pid in skip or pid == 0:
  235:             continue
  236: 
  237:         logger.info("Killing orphaned process (pattern=%r): PID %d", pattern, pid)
  238:         try:
  239:             subprocess.run(
  240:                 ["taskkill", "/F", "/PID", str(pid)],
  241:                 capture_output=True, timeout=10,
  242:             )
  243:             killed += 1
  244:         except Exception:
  245:             pass
  246: 
  247:     return killed
  248: 
  249: 
  250: def kill_orphaned_llama():
  251:     """Find and kill orphaned llama-completion.exe processes.
  252: 
  253:     An orphaned process is one whose parent PID no longer exists.
  254:     Uses PowerShell + Win32 API to enumerate and check processes.
  255: 
  256:     Returns the number of processes killed.  Returns 0 on non-Windows.
  257:     """
  258:     if sys.platform != "win32":
  259:         return 0
  260: 
  261:     import ctypes
  262: 
  263:     kernel32 = ctypes.windll.kernel32
  264:     PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
  265:     PROCESS_TERMINATE = 0x0001
  266: 
  267:     # --- Get llama-completion.exe processes via PowerShell --------------------
  268:     ps_cmd = (
  269:         'powershell.exe -NoProfile -Command "'
  270:         "Get-CimInstance Win32_Process -Filter \\\"Name='llama-completion.exe'\\\" "
  271:         '| Select-Object ProcessId,ParentProcessId | ConvertTo-Json"'
  272:     )
  273: 
  274:     try:
  275:         result = subprocess.run(
  276:             ps_cmd,
  277:             capture_output=True,
  278:             text=True,
  279:             timeout=15,
  280:             shell=True,
  281:         )
  282:     except Exception as exc:
  283:         logger.debug("PowerShell process query failed: %s", exc)
  284:         return 0
  285: 
  286:     if result.returncode != 0 or not result.stdout.strip():
  287:         return 0
  288: 
  289:     try:
  290:         data = json.loads(result.stdout.strip())
  291:     except json.JSONDecodeError:
  292:         logger.debug("Failed to parse PowerShell JSON output")
  293:         return 0
  294: 
  295:     # PowerShell returns a single object (not array) when there's only one match
  296:     if isinstance(data, dict):
  297:         data = [data]
  298:     if not isinstance(data, list):
  299:         return 0
  300: 
  301:     killed = 0
  302:     for entry in data:
  303:         pid = entry.get("ProcessId")
  304:         ppid = entry.get("ParentProcessId")
  305:         if pid is None or ppid is None:
  306:             continue
  307: 
  308:         # Check if parent is alive
  309:         parent_alive = False
  310:         handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(ppid))
  311:         if handle:
  312:             parent_alive = True
  313:             kernel32.CloseHandle(handle)
  314: 
  315:         if not parent_alive:
  316:             logger.info("Orphaned llama-completion.exe PID %d (parent %d dead) — killing", pid, ppid)
  317:             # Terminate the orphan
  318:             h_proc = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
  319:             if h_proc:
  320:                 kernel32.TerminateProcess(h_proc, 1)
  321:                 kernel32.CloseHandle(h_proc)
  322:                 killed += 1
  323:                 logger.info("Killed orphaned llama-completion.exe PID %d", pid)
  324:             else:
  325:                 logger.warning("Could not open llama-completion.exe PID %d for termination", pid)
  326: 
  327:     return killed
=== portfolio/log_rotation.py:300-360 ===
  300:             existing_lines = []
  301:             with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
  302:                 for existing_line in gf:
  303:                     existing_line = existing_line.rstrip("\n")
  304:                     if existing_line.strip():
  305:                         existing_lines.append(existing_line)
  306:             all_lines = existing_lines + lines
  307:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  308:                 for line in all_lines:
  309:                     gf.write(line + "\n")
  310:         elif policy.get("compress", True):
  311:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  312:                 for line in lines:
  313:                     gf.write(line + "\n")
  314:         else:
  315:             with open(archive_path, "a", encoding="utf-8") as af:
  316:                 for line in lines:
  317:                     af.write(line + "\n")
  318: 
  319:     # Rewrite the original file with only kept lines
  320:     tmp_path = filepath.with_suffix(".tmp")
  321:     with open(tmp_path, "w", encoding="utf-8") as f:
  322:         for line in keep_lines:
  323:             f.write(line + "\n")
  324: 
  325:     # Atomic-ish replace: remove original, rename tmp
  326:     # On Windows, os.replace is atomic within the same volume
  327:     os.replace(tmp_path, filepath)
  328: 
  329:     result["status"] = "rotated"
  330:     return result
  331: 
  332: 
  333: def rotate_text(filename, policy, dry_run=False):
  334:     """Rotate a plain text file by size.
  335: 
  336:     When file exceeds max_size_mb:
  337:     - Shift existing rotations: .2 -> .3, .1 -> .2, current -> .1
  338:     - Delete rotations beyond keep_rotations
  339:     - Compress old rotations if policy says so
  340: 
  341:     Returns dict with rotation stats.
  342:     """
  343:     filepath = DATA_DIR / filename
  344:     if not filepath.exists():
  345:         return {"file": filename, "status": "not_found"}
  346: 
  347:     size_mb = _file_size_mb(filepath)
  348:     max_size_mb = policy.get("max_size_mb", 10)
  349:     keep_rotations = policy.get("keep_rotations", 3)
  350:     compress = policy.get("compress", True)
  351: 
  352:     result = {
  353:         "file": filename,
  354:         "size_mb": round(size_mb, 2),
  355:         "max_size_mb": max_size_mb,
  356:     }
  357: 
  358:     if size_mb < max_size_mb:
  359:         result["status"] = "under_threshold"
  360:         return result

codex
The first sweep already knocked out a few obvious misses: the HTTP retry claim is flatly wrong, and the Windows file-lock seek issue is fixed in the current tree. I’m drilling into the authentication, concurrency, and subprocess findings now because those need surrounding code, not just the cited line.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 1 120
Show-Lines 'dashboard/app.py' 1085 1145
Show-Lines 'dashboard/app.py' 1145 1235" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/file_utils.py' 1 110
Show-Lines 'portfolio/file_utils.py' 136 205
Show-Lines 'portfolio/shared_state.py' 90 130
Show-Lines 'portfolio/shared_state.py' 253 280" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/gpu_gate.py' 110 260
Show-Lines 'portfolio/subprocess_utils.py' 1 120
Show-Lines 'portfolio/subprocess_utils.py' 130 170" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 952ms:
=== portfolio/gpu_gate.py:110-260 ===
  110: def _try_break_stale_lock() -> bool:
  111:     """Reap the lock file iff stale-by-mtime AND owner pid is dead.
  112: 
  113:     Returns True if the lock was broken (caller can retry acquire), False
  114:     otherwise. Defensive: never raises — the sweeper daemon depends on this.
  115: 
  116:     Called from two paths:
  117:     - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
  118:     - Sweeper: the background daemon, when no one is waiting.
  119: 
  120:     Both paths must agree on the predicate so behaviour is identical
  121:     regardless of which path reaped the lock. Emits the same
  122:     ``Breaking stale GPU lock`` warning either way so log-grep tools and
  123:     postmortem audits work uniformly.
  124:     """
  125:     try:
  126:         if not _GPU_LOCK_FILE.exists():
  127:             return False
  128:         if not _is_stale():
  129:             return False
  130:         info = _read_lock()
  131:         pid = info.get("pid", 0)
  132:         if _pid_alive(pid):
  133:             return False
  134:         logger.warning("Breaking stale GPU lock: %s (pid=%s, dead)",
  135:                        info.get("model"), pid)
  136:         _release_lock()
  137:         return True
  138:     except Exception as exc:
  139:         # The sweeper must NEVER crash — a dead daemon stops sweeping forever.
  140:         logger.debug("Stale-lock sweep error: %s", exc)
  141:         return False
  142: 
  143: 
  144: def _sweeper_loop():
  145:     """Background daemon: reap stale-dead locks every 30 s.
  146: 
  147:     Wedge-recovery story (2026-05-02): chronos pid 13152 died holding the
  148:     lock at 02:14. No one tried to acquire while the loop was stuck inside
  149:     its LLM batch, so ``_is_stale()`` was never checked. Loop wedged for
  150:     ~25 hours until a system reboot. This daemon closes that hole.
  151:     """
  152:     while True:
  153:         try:
  154:             time.sleep(_SWEEPER_INTERVAL_SECONDS)
  155:             _try_break_stale_lock()
  156:         except Exception as exc:
  157:             # Defence-in-depth — _try_break_stale_lock already swallows but
  158:             # any future code added here must also keep the daemon alive.
  159:             logger.debug("Sweeper loop error: %s", exc)
  160: 
  161: 
  162: def _start_sweeper():
  163:     """Spawn the sweeper daemon (idempotent, thread-safe).
  164: 
  165:     Lazily called from ``gpu_gate()`` so:
  166:     - Subprocess workers that import this module but never call
  167:       ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
  168:       do NOT spawn a redundant daemon.
  169:     - Tests can reset ``_sweeper_thread = None`` and re-trigger spawn.
  170: 
  171:     If the daemon ever dies (it shouldn't — both layers swallow exceptions)
  172:     a future call will respawn it.
  173:     """
  174:     global _sweeper_thread
  175:     with _SWEEPER_LOCK:
  176:         if _sweeper_thread is None or not _sweeper_thread.is_alive():
  177:             t = threading.Thread(
  178:                 target=_sweeper_loop,
  179:                 name="gpu-gate-sweeper",
  180:                 daemon=True,
  181:             )
  182:             _sweeper_thread = t
  183:             t.start()
  184: 
  185: 
  186: @contextmanager
  187: def gpu_gate(model_name: str, timeout: float = 60):
  188:     """Acquire exclusive GPU access, log VRAM before/after.
  189: 
  190:     Uses a two-layer lock:
  191:     1. threading.Lock for in-process concurrency (ThreadPoolExecutor workers)
  192:     2. File-based lock for cross-process protection (metals loop, etc.)
  193: 
  194:     Args:
  195:         model_name: e.g. "ministral-3", "qwen3", "chronos"
  196:         timeout: max seconds to wait for lock
  197: 
  198:     Yields:
  199:         True if acquired, False if timed out.
  200:     """
  201:     # Lazy-spawn the stale-lock sweeper. Idempotent so no cost after the
  202:     # first call. See _start_sweeper() for the rationale.
  203:     _start_sweeper()
  204: 
  205:     deadline = time.time() + timeout
  206: 
  207:     # Layer 1: In-process thread lock (prevents ThreadPoolExecutor races)
  208:     remaining = deadline - time.time()
  209:     thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
  210:     if not thread_acquired:
  211:         logger.warning("GPU thread-lock timeout (%ss) for %s", timeout, model_name)
  212:         yield False
  213:         return
  214: 
  215:     try:
  216:         # Layer 2: File-based lock (cross-process)
  217:         file_acquired = False
  218:         while time.time() < deadline:
  219:             try:
  220:                 # Atomic create — fails if file already exists (no TOCTOU race)
  221:                 fd = os.open(str(_GPU_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
  222:                 # H23/CI1: Always close fd in finally to prevent leak if write raises.
  223:                 try:
  224:                     os.write(fd, f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}".encode())
  225:                 finally:
  226:                     os.close(fd)
  227:                 file_acquired = True
  228:                 break
  229:             except FileExistsError:
  230:                 # Lock file exists — check if same process (re-entry) or stale.
  231:                 info = _read_lock()
  232:                 if info.get("pid") == os.getpid():
  233:                     # Re-entry from same process (shouldn't happen with thread lock, but safe)
  234:                     file_acquired = True
  235:                     break
  236:                 # BUG-182: Only break stale lock if owning process is dead.
  237:                 # Helper is shared with the sweeper daemon so the two paths
  238:                 # agree on the predicate.
  239:                 if _try_break_stale_lock():
  240:                     continue  # retry atomic create
  241:                 logger.debug("GPU file-locked by %s, waiting...", info.get("model", "?"))
  242:                 time.sleep(1.0)
  243: 
  244:         if not file_acquired:
  245:             info = _read_lock()
  246:             logger.warning("GPU file-lock timeout (%ss) — held by %s", timeout, info.get("model", "?"))
  247:             yield False
  248:             return
  249: 
  250:         # Log VRAM at acquire
  251:         vram = get_vram_usage()
  252:         if vram:
  253:             logger.info(
  254:                 "GPU gate ACQUIRED by %s — VRAM: %dMB used / %dMB free / %dMB total (GPU %d%%)",
  255:                 model_name, vram["used_mb"], vram["free_mb"], vram["total_mb"], vram["gpu_util_pct"],
  256:             )
  257: 
  258:         t0 = time.time()
  259:         try:
  260:             yield True
=== portfolio/subprocess_utils.py:1-120 ===
    1: """Subprocess utilities to prevent orphaned child processes on Windows.
    2: 
    3: Provides:
    4: - run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
    5:   with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so children die when the parent exits.
    6: - popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
    7:   to a Job Object so it's automatically killed if the parent dies.
    8: - kill_orphaned_by_cmdline(): Find and kill orphaned processes matching a command
    9:   line pattern (safety net for processes that escaped Job Object protection).
   10: - kill_orphaned_llama(): Safety-net reaper for orphaned llama-completion.exe processes.
   11: """
   12: 
   13: import json
   14: import logging
   15: import subprocess
   16: import sys
   17: 
   18: logger = logging.getLogger("portfolio.subprocess_utils")
   19: 
   20: 
   21: def run_safe(cmd, **kwargs):
   22:     """Run a subprocess with Windows Job Object protection.
   23: 
   24:     Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
   25:     with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so that the child process is
   26:     automatically killed if the parent Python process dies.
   27: 
   28:     Falls back to plain subprocess.run() on non-Windows or if Job Object
   29:     creation fails.
   30: 
   31:     Supported kwargs: capture_output, text, timeout, input, stdin (and any
   32:     others accepted by subprocess.Popen / subprocess.run).
   33:     """
   34:     if sys.platform != "win32":
   35:         return subprocess.run(cmd, **kwargs)
   36: 
   37:     try:
   38:         return _run_with_job_object(cmd, **kwargs)
   39:     except Exception as exc:
   40:         logger.debug("Job Object creation failed (%s), falling back to subprocess.run", exc)
   41:         return subprocess.run(cmd, **kwargs)
   42: 
   43: 
   44: def _create_job_object():
   45:     """Create a Windows Job Object with KILL_ON_JOB_CLOSE.
   46: 
   47:     Returns (job_handle, kernel32) or raises OSError.
   48:     """
   49:     import ctypes
   50:     from ctypes import wintypes
   51: 
   52:     kernel32 = ctypes.windll.kernel32
   53: 
   54:     job = kernel32.CreateJobObjectW(None, None)
   55:     if not job:
   56:         raise OSError("CreateJobObjectW failed")
   57: 
   58:     # JOBOBJECT_BASIC_LIMIT_INFORMATION (64-bit layout)
   59:     class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
   60:         _fields_ = [
   61:             ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
   62:             ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
   63:             ("LimitFlags", wintypes.DWORD),
   64:             ("_pad0", wintypes.DWORD),
   65:             ("MinimumWorkingSetSize", ctypes.c_size_t),
   66:             ("MaximumWorkingSetSize", ctypes.c_size_t),
   67:             ("ActiveProcessLimit", wintypes.DWORD),
   68:             ("_pad1", wintypes.DWORD),
   69:             ("Affinity", ctypes.c_size_t),
   70:             ("PriorityClass", wintypes.DWORD),
   71:             ("SchedulingClass", wintypes.DWORD),
   72:         ]
   73: 
   74:     class IO_COUNTERS(ctypes.Structure):
   75:         _fields_ = [
   76:             ("ReadOperationCount", ctypes.c_ulonglong),
   77:             ("WriteOperationCount", ctypes.c_ulonglong),
   78:             ("OtherOperationCount", ctypes.c_ulonglong),
   79:             ("ReadTransferCount", ctypes.c_ulonglong),
   80:             ("WriteTransferCount", ctypes.c_ulonglong),
   81:             ("OtherTransferCount", ctypes.c_ulonglong),
   82:         ]
   83: 
   84:     class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
   85:         _fields_ = [
   86:             ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
   87:             ("IoInfo", IO_COUNTERS),
   88:             ("ProcessMemoryLimit", ctypes.c_size_t),
   89:             ("JobMemoryLimit", ctypes.c_size_t),
   90:             ("PeakProcessMemoryUsed", ctypes.c_size_t),
   91:             ("PeakJobMemoryUsed", ctypes.c_size_t),
   92:         ]
   93: 
   94:     JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
   95:     JobObjectExtendedLimitInformation = 9
   96: 
   97:     info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
   98:     info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
   99: 
  100:     ok = kernel32.SetInformationJobObject(
  101:         job,
  102:         JobObjectExtendedLimitInformation,
  103:         ctypes.byref(info),
  104:         ctypes.sizeof(info),
  105:     )
  106:     if not ok:
  107:         kernel32.CloseHandle(job)
  108:         raise OSError("SetInformationJobObject failed")
  109: 
  110:     return job, kernel32
  111: 
  112: 
  113: def _run_with_job_object(cmd, **kwargs):
  114:     """Internal: run subprocess inside a Windows Job Object."""
  115:     job, kernel32 = _create_job_object()
  116: 
  117:     try:
  118:         popen_kwargs = dict(kwargs)
  119:         timeout = popen_kwargs.pop("timeout", None)
  120: 
=== portfolio/subprocess_utils.py:130-170 ===
  130: 
  131:         try:
  132:             kernel32.AssignProcessToJobObject(job, int(proc._handle))
  133:         except Exception as e:
  134:             logger.warning(
  135:                 "Job Object assignment failed for pid %d — child may orphan: %s",
  136:                 proc.pid, e,
  137:             )
  138: 
  139:         try:
  140:             stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
  141:         except subprocess.TimeoutExpired:
  142:             proc.kill()
  143:             proc.communicate()
  144:             raise
  145: 
  146:         return subprocess.CompletedProcess(
  147:             args=cmd,
  148:             returncode=proc.returncode,
  149:             stdout=stdout,
  150:             stderr=stderr,
  151:         )
  152:     finally:
  153:         kernel32.CloseHandle(job)
  154: 
  155: 
  156: def popen_in_job(cmd, **kwargs):
  157:     """Start a long-running subprocess inside a Windows Job Object.
  158: 
  159:     Like subprocess.Popen(), but assigns the child to a Job Object with
  160:     JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. If the parent process dies (crash,
  161:     kill -9, power loss), the OS automatically kills the child.
  162: 
  163:     Returns (proc, job_handle) on Windows. On non-Windows or if Job Object
  164:     creation fails, returns (proc, None).
  165: 
  166:     Caller must call close_job(job_handle) when explicitly stopping the child.
  167:     """
  168:     proc = subprocess.Popen(cmd, **kwargs)
  169: 
  170:     if sys.platform != "win32":

 succeeded in 957ms:
=== portfolio/file_utils.py:1-110 ===
    1: """Shared file I/O utilities."""
    2: import json
    3: import logging
    4: import os
    5: import tempfile
    6: from collections import deque
    7: from contextlib import suppress
    8: from pathlib import Path
    9: 
   10: # Cross-platform file-locking primitives for `atomic_append_jsonl`.
   11: # Same pattern as `portfolio/process_lock.py`.
   12: try:
   13:     import msvcrt as _msvcrt  # type: ignore[import-not-found]
   14: except ImportError:  # pragma: no cover - non-Windows
   15:     _msvcrt = None  # type: ignore[assignment]
   16: try:
   17:     import fcntl as _fcntl  # type: ignore[import-not-found]
   18: except ImportError:  # pragma: no cover - Windows
   19:     _fcntl = None  # type: ignore[assignment]
   20: 
   21: logger = logging.getLogger("portfolio.file_utils")
   22: 
   23: 
   24: def atomic_write_text(path, text, encoding="utf-8"):
   25:     """Atomically write text to a file using tempfile + os.replace.
   26: 
   27:     Same safety guarantees as atomic_write_json: fsync before replace,
   28:     no partial writes on crash.
   29:     """
   30:     path = Path(path)
   31:     path.parent.mkdir(parents=True, exist_ok=True)
   32:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   33:     try:
   34:         with os.fdopen(fd, "w", encoding=encoding) as f:
   35:             f.write(text)
   36:             f.flush()
   37:             os.fsync(f.fileno())
   38:         os.replace(tmp, str(path))
   39:     except BaseException:
   40:         with suppress(OSError):
   41:             os.unlink(tmp)
   42:         raise
   43: 
   44: 
   45: def atomic_write_json(path, data, indent=2, ensure_ascii=True):
   46:     """Atomically write JSON data to a file using tempfile + os.replace.
   47: 
   48:     Ensures the file is never left in a partially-written state.
   49:     Fsyncs before replace to guarantee durability on power loss (H34).
   50:     """
   51:     path = Path(path)
   52:     path.parent.mkdir(parents=True, exist_ok=True)
   53:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   54:     try:
   55:         with os.fdopen(fd, "w", encoding="utf-8") as f:
   56:             json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
   57:             f.flush()
   58:             os.fsync(f.fileno())
   59:         os.replace(tmp, str(path))
   60:     except BaseException:
   61:         with suppress(OSError):
   62:             os.unlink(tmp)
   63:         raise
   64: 
   65: 
   66: def load_json(path, default=None):
   67:     """Load a JSON file. Returns *default* if missing or unparseable.
   68: 
   69:     Uses try/except instead of exists() check to avoid TOCTOU race.
   70:     Handles OSError (permission denied, locked files) gracefully on Windows.
   71:     Logs WARNING on corrupt JSON so corruption is observable (H35).
   72:     """
   73:     path = Path(path)
   74:     try:
   75:         return json.loads(path.read_text(encoding="utf-8"))
   76:     except FileNotFoundError:
   77:         return default
   78:     except OSError:
   79:         # BUG-139: PermissionError (file locked by antivirus/another process)
   80:         # and other OS-level errors should degrade gracefully like missing files.
   81:         logger.debug("load_json: OS error reading %s, returning default", path.name)
   82:         return default
   83:     except (json.JSONDecodeError, ValueError):
   84:         # H35: Log corruption so it's observable — silent defaults hide data loss.
   85:         logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
   86:         return default
   87: 
   88: 
   89: def require_json(path):
   90:     """Load a JSON file, raising on corruption or missing file.
   91: 
   92:     Unlike load_json(), this function does NOT silently return defaults.
   93:     Use for critical files where corruption must be surfaced (H35).
   94: 
   95:     Raises:
   96:         FileNotFoundError: If the file does not exist.
   97:         json.JSONDecodeError: If the file contains invalid JSON.
   98:         OSError: If the file cannot be read.
   99:     """
  100:     path = Path(path)
  101:     return json.loads(path.read_text(encoding="utf-8"))
  102: 
  103: 
  104: def load_jsonl(path, limit=None):
  105:     """Load entries from a JSONL file.
  106: 
  107:     Args:
  108:         path: Path to the .jsonl file.
  109:         limit: If set, keep only the *last* N entries (uses a deque).
  110: 
=== portfolio/file_utils.py:136-205 ===
  136: def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
  137:     """Load the last N entries from a JSONL file by reading from the end.
  138: 
  139:     Much more efficient than load_jsonl(limit=N) for large files because
  140:     it only reads the last `tail_bytes` bytes instead of the entire file.
  141: 
  142:     Args:
  143:         path: Path to the .jsonl file.
  144:         max_entries: Maximum entries to return.
  145:         tail_bytes: How many bytes to read from the end of the file.
  146:             Default 512KB ≈ ~1000 typical entries.
  147: 
  148:     Returns:
  149:         list of parsed dicts (chronological order). Empty list if missing.
  150:     """
  151:     path = Path(path)
  152:     try:
  153:         file_size = path.stat().st_size
  154:     except (FileNotFoundError, OSError):
  155:         return []
  156:     if file_size == 0:
  157:         return []
  158: 
  159:     entries = []
  160:     try:
  161:         with open(path, "rb") as f:
  162:             # Seek to near end of file
  163:             offset = max(0, file_size - tail_bytes)
  164:             # 2026-05-04 codex P3-1 follow-up: peek the byte just before
  165:             # the seek point. If it's a newline, the seek lands exactly
  166:             # at a line boundary and the first decoded line is intact.
  167:             # Without this check, a happy-coincidence boundary would
  168:             # cost us one valid entry on every read.
  169:             seek_on_boundary = False
  170:             if offset > 0:
  171:                 f.seek(offset - 1)
  172:                 prior = f.read(1)
  173:                 seek_on_boundary = prior == b"\n"
  174:             f.seek(offset)
  175:             data = f.read()
  176:         # Decode and split into lines
  177:         text = data.decode("utf-8", errors="replace")
  178:         lines = text.split("\n")
  179:         # Drop the first line only when we landed mid-line. When seek
  180:         # lands on a newline boundary, the first decoded line is
  181:         # complete and should be kept.
  182:         if offset > 0 and lines and not seek_on_boundary:
  183:             lines = lines[1:]
  184:         for line in lines:
  185:             line = line.strip()
  186:             if not line:
  187:                 continue
  188:             try:
  189:                 entries.append(json.loads(line))
  190:             except json.JSONDecodeError:
  191:                 continue
  192:     except (OSError, UnicodeDecodeError) as e:
  193:         logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
  194:         return []
  195: 
  196:     # Return last max_entries in chronological order
  197:     if len(entries) > max_entries:
  198:         entries = entries[-max_entries:]
  199:     return entries
  200: 
  201: 
  202: def atomic_append_jsonl(path, entry):
  203:     """Append a single JSON entry to a JSONL file with atomic semantics
  204:     across threads and processes.
  205: 
=== portfolio/shared_state.py:90-130 ===
   90:         _loading_timestamps[key] = time.time()
   91: 
   92:     try:
   93:         data = func(*args)
   94:         with _cache_lock:
   95:             # Don't cache None results — they typically indicate transient
   96:             # API failures that returned None instead of raising. Caching
   97:             # None hides the failure for the entire TTL duration, preventing
   98:             # retry. Stale data (if any) is preserved for fallback.
   99:             if data is not None:
  100:                 _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
  101:             _loading_keys.discard(key)
  102:             _loading_timestamps.pop(key, None)  # BUG-213: clean up on success path
  103:         return data
  104:     except KeyboardInterrupt:
  105:         with _cache_lock:
  106:             _loading_keys.discard(key)
  107:         logger.warning("[%s] interrupted (KeyboardInterrupt), returning None", key)
  108:         return None
  109:     except Exception as e:
  110:         logger.warning("[%s] error: %s", key, e)
  111:         with _cache_lock:
  112:             _loading_keys.discard(key)
  113:             if key in _tool_cache:
  114:                 age = now - _tool_cache[key]["time"]
  115:                 max_stale = ttl * _MAX_STALE_FACTOR
  116:                 if age > max_stale:
  117:                     logger.warning(
  118:                         "[%s] stale data too old (%.0fs > %.0fs max), returning None",
  119:                         key, age, max_stale,
  120:                     )
  121:                     return None
  122:                 _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
  123:                 return _tool_cache[key]["data"]
  124:         return None
  125: 
  126: 
  127: def _cached_or_enqueue(key, ttl, enqueue_fn, context,
  128:                         should_enqueue_fn=None, max_stale_factor=None):
  129:     """Check cache — if fresh return it, if expired enqueue for batch and return stale.
  130: 
=== portfolio/shared_state.py:253-280 ===
  253: class _RateLimiter:
  254:     """Token-bucket rate limiter. Sleeps when calls exceed rate."""
  255:     def __init__(self, max_per_minute, name=""):
  256:         self.interval = 60.0 / max_per_minute
  257:         self.last_call = 0.0
  258:         self.name = name
  259:         self._lock = threading.Lock()
  260: 
  261:     def wait(self):
  262:         # BUG-212: Sleep OUTSIDE the lock to avoid blocking all 8 worker
  263:         # threads. Calculate sleep duration under the lock, release it,
  264:         # then sleep.
  265:         # Fix: Reserve the next slot (last_call = last_call + interval)
  266:         # BEFORE releasing the lock, so parallel threads see the reserved
  267:         # time and calculate a longer wait instead of stampeding.
  268:         wait_time = 0.0
  269:         with self._lock:
  270:             now = time.time()
  271:             elapsed = now - self.last_call
  272:             if elapsed < self.interval:
  273:                 wait_time = self.interval - elapsed
  274:             # Reserve the next slot atomically — even if we haven't slept yet,
  275:             # the next thread to enter will see this and wait longer.
  276:             self.last_call = self.last_call + self.interval if wait_time > 0 else now
  277:         if wait_time > 0:
  278:             time.sleep(wait_time)
  279: 
  280: 

 succeeded in 1107ms:
=== dashboard/app.py:1-120 ===
    1: """Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""
    2: 
    3: import functools
    4: import hmac
    5: import logging
    6: import math
    7: import threading
    8: import time
    9: from datetime import UTC, datetime
   10: from pathlib import Path
   11: from typing import Any
   12: from zoneinfo import ZoneInfo
   13: 
   14: from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
   15: from flask.json.provider import DefaultJSONProvider
   16: 
   17: logger = logging.getLogger(__name__)
   18: 
   19: 
   20: def _json_safe(value):
   21:     """Convert NaN/Infinity to JSON-safe null recursively."""
   22:     if isinstance(value, float):
   23:         return value if math.isfinite(value) else None
   24:     if isinstance(value, dict):
   25:         return {key: _json_safe(item) for key, item in value.items()}
   26:     if isinstance(value, list):
   27:         return [_json_safe(item) for item in value]
   28:     if isinstance(value, tuple):
   29:         return [_json_safe(item) for item in value]
   30:     return value
   31: 
   32: 
   33: class SafeJSONProvider(DefaultJSONProvider):
   34:     """Flask JSON provider that strips non-finite floats."""
   35: 
   36:     def dumps(self, obj, **kwargs):
   37:         return super().dumps(_json_safe(obj), **kwargs)
   38: 
   39: 
   40: app = Flask(__name__, static_folder="static")
   41: app.json = SafeJSONProvider(app)
   42: 
   43: 
   44: _ALLOWED_ORIGINS = {
   45:     "http://localhost:5055",
   46:     "http://127.0.0.1:5055",
   47:     "http://localhost:3000",
   48:     "http://127.0.0.1:3000",
   49: }
   50: 
   51: 
   52: @app.after_request
   53: def add_cors_headers(response):
   54:     """Allow same-network browser access from known origins only (BUG-230)."""
   55:     origin = request.headers.get("Origin", "")
   56:     if origin in _ALLOWED_ORIGINS:
   57:         response.headers["Access-Control-Allow-Origin"] = origin
   58:     response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
   59:     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
   60:     response.headers["Access-Control-Allow-Credentials"] = "false"
   61:     return response
   62: 
   63: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   64: TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
   65: CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
   66: STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")
   67: 
   68: import sys
   69: 
   70: sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
   71: from portfolio.file_utils import load_json as _load_json_impl
   72: from portfolio.file_utils import load_jsonl as _load_jsonl_impl
   73: from portfolio.file_utils import load_jsonl_tail as _load_jsonl_tail_impl
   74: 
   75: # ---------------------------------------------------------------------------
   76: # TTL Cache (BUG-130: avoid re-reading files on every API request)
   77: # ---------------------------------------------------------------------------
   78: 
   79: _cache = {}
   80: _cache_lock = threading.Lock()
   81: _DEFAULT_TTL = 5  # seconds
   82: 
   83: 
   84: def _cached_read(key, ttl, read_fn):
   85:     """Return cached result if fresh, otherwise call read_fn and cache."""
   86:     now = time.monotonic()
   87:     with _cache_lock:
   88:         entry = _cache.get(key)
   89:         if entry and (now - entry[1]) < ttl:
   90:             return entry[0]
   91:     result = read_fn()
   92:     with _cache_lock:
   93:         _cache[key] = (result, now)
   94:     return result
   95: 
   96: 
   97: # ---------------------------------------------------------------------------
   98: # Helpers
   99: # ---------------------------------------------------------------------------
  100: 
  101: def _read_json(path, ttl=_DEFAULT_TTL):
  102:     return _cached_read(f"json:{path}", ttl, lambda: _load_json_impl(path))
  103: 
  104: 
  105: def _read_jsonl(path, limit=100, ttl=_DEFAULT_TTL):
  106:     """Cached JSONL read returning the last `limit` entries.
  107: 
  108:     Switched from load_jsonl(limit=) (full scan + deque) to
  109:     load_jsonl_tail (seek from end). For an 80MB log the difference is
  110:     ~880ms vs ~5ms.
  111: 
  112:     2026-05-04 codex P2-1 follow-up: the original 4 MB tail-bytes
  113:     ceiling could silently under-deliver entries when callers ask for
  114:     a large window AND individual rows are large (e.g. /api/telegrams
  115:     requests 5000 entries × up to 4 KB each ≈ 20 MB needed). The
  116:     fetcher now grows tail_bytes adaptively — doubling on each retry
  117:     until either `limit` rows are parsed or the whole file has been
  118:     pulled — and falls through to the full-scan path as a final
  119:     safety net. Cache key bumped to v2 so old (potentially
  120:     under-delivered) entries don't survive the deploy.
=== dashboard/app.py:1085-1145 ===
 1085:     """Return the full 30-signal x all-tickers grid.
 1086: 
 1087:     Each cell is BUY/SELL/HOLD. Built from agent_summary.json signals + enhanced_signals.
 1088:     """
 1089:     summary = _read_json(DATA_DIR / "agent_summary.json")
 1090:     if not summary:
 1091:         return jsonify({"error": "no data"}), 404
 1092: 
 1093:     signals_data = summary.get("signals", {})
 1094: 
 1095:     # Core signal names (11 total: 8 active + 3 disabled)
 1096:     core_signals = [
 1097:         "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
 1098:         "ministral", "volume", "ml", "funding", "custom_lora"
 1099:     ]
 1100:     # Enhanced composite signal names (19 modules, signals #12-#30)
 1101:     enhanced_signals = [
 1102:         "trend", "momentum", "volume_flow", "volatility_sig",
 1103:         "candlestick", "structure", "fibonacci", "smart_money",
 1104:         "oscillators", "heikin_ashi", "mean_reversion", "calendar",
 1105:         "macro_regime", "momentum_factors", "news_event", "econ_calendar",
 1106:         "forecast", "claude_fundamental", "futures_flow"
 1107:     ]
 1108:     all_signals = core_signals + enhanced_signals
 1109: 
 1110:     heatmap = {}
 1111:     tickers = list(signals_data.keys())
 1112: 
 1113:     for ticker in tickers:
 1114:         sig = signals_data[ticker]
 1115:         extra = sig.get("extra", {})
 1116:         votes = extra.get("_votes", {})
 1117: 
 1118:         # _votes contains all 30 signal keys (core + enhanced)
 1119:         row = {}
 1120:         for s in all_signals:
 1121:             row[s] = (votes.get(s, "HOLD") or "HOLD").upper()
 1122:         heatmap[ticker] = row
 1123: 
 1124:     # Per-(ticker, signal) state-change timestamps for the "time-in-state" badge.
 1125:     # Written by portfolio.reporting._update_signal_state_since each loop cycle.
 1126:     # Missing or malformed payload degrades to an empty map: frontend renders
 1127:     # cells without the badge — never 500.
 1128:     #
 1129:     # Codex P2 (2026-05-05): the since-file is written *before* agent_summary
 1130:     # in the same cycle, and a swallowed write-failure can also leave the two
 1131:     # out of sync. Guard against showing a stale duration on a freshly-flipped
 1132:     # vote by only emitting `since` when the recorded vote matches the current
 1133:     # heatmap value. Mismatched cells fall back to colour-only until the next
 1134:     # cycle re-syncs both files.
 1135:     state_since_payload = _read_json(DATA_DIR / "signal_state_since.json") or {}
 1136:     state_since_votes = state_since_payload.get("votes") if isinstance(state_since_payload, dict) else None
 1137:     since: dict[str, dict[str, str]] = {}
 1138:     if isinstance(state_since_votes, dict):
 1139:         for ticker in tickers:
 1140:             tk_state = state_since_votes.get(ticker)
 1141:             if not isinstance(tk_state, dict):
 1142:                 continue
 1143:             row_since: dict[str, str] = {}
 1144:             current_row = heatmap.get(ticker, {})
 1145:             for s in all_signals:
=== dashboard/app.py:1145-1235 ===
 1145:             for s in all_signals:
 1146:                 entry = tk_state.get(s)
 1147:                 if not isinstance(entry, dict):
 1148:                     continue
 1149:                 since_ts = entry.get("since")
 1150:                 if not isinstance(since_ts, str):
 1151:                     continue
 1152:                 if entry.get("vote") != current_row.get(s):
 1153:                     continue  # stale: vote in since-file disagrees with heatmap
 1154:                 row_since[s] = since_ts
 1155:             if row_since:
 1156:                 since[ticker] = row_since
 1157: 
 1158:     # 2026-05-05: ship the disabled set so the heatmap can render
 1159:     # disabled cells with the muted style + tap-to-show reason. The
 1160:     # frontend already reads `data.disabled_signals` (signals.js:137).
 1161:     try:
 1162:         from portfolio.tickers import DISABLED_SIGNALS
 1163:         disabled = sorted(DISABLED_SIGNALS)
 1164:     except Exception:
 1165:         disabled = []
 1166:     return jsonify({
 1167:         "tickers": tickers,
 1168:         "signals": all_signals,
 1169:         "core_signals": core_signals,
 1170:         "enhanced_signals": enhanced_signals,
 1171:         "heatmap": heatmap,
 1172:         "since": since,
 1173:         "disabled_signals": disabled,
 1174:     })
 1175: 
 1176: 
 1177: # ---------------------------------------------------------------------------
 1178: # New: Trigger activity timeline
 1179: # ---------------------------------------------------------------------------
 1180: 
 1181: @app.route("/api/triggers")
 1182: @require_auth
 1183: def api_triggers():
 1184:     """Return last 50 trigger/invocation events from invocations.jsonl."""
 1185:     entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
 1186:     return jsonify(entries)
 1187: 
 1188: 
 1189: @app.route("/api/accuracy-history")
 1190: @require_auth
 1191: def api_accuracy_history():
 1192:     """Return accuracy snapshots over time for charting trend lines.
 1193: 
 1194:     2026-05-05: tag each per-signal slice with `enabled` so the chart
 1195:     can dim/exclude force-HOLD'd signals. Tag is derived at response
 1196:     time from DISABLED_SIGNALS so historical snapshots written before
 1197:     the flag existed are also tagged correctly.
 1198:     """
 1199:     entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
 1200:     try:
 1201:         from portfolio.tickers import DISABLED_SIGNALS
 1202:         for snap in entries:
 1203:             sigs = snap.get("signals") if isinstance(snap, dict) else None
 1204:             if not isinstance(sigs, dict):
 1205:                 continue
 1206:             for sig_name, info in sigs.items():
 1207:                 if isinstance(info, dict):
 1208:                     # Overwrite (not setdefault) — see /api/accuracy comment.
 1209:                     info["enabled"] = sig_name not in DISABLED_SIGNALS
 1210:     except Exception:
 1211:         logger.exception("accuracy-history enrichment failed; serving raw")
 1212:     return jsonify(entries)
 1213: 
 1214: 
 1215: @app.route("/api/local-llm-trends")
 1216: @require_auth
 1217: def api_local_llm_trends():
 1218:     """Return local-LLM report trend data for dashboard charts.
 1219: 
 1220:     Query params:
 1221:       - limit: number of history points to return (default 90, max 366)
 1222:       - ticker: optional ticker filter for Ministral per-ticker series
 1223:     """
 1224:     limit = _parse_limit_arg("limit", default=90, max_value=366)
 1225:     ticker = request.args.get("ticker", "").strip().upper() or None
 1226:     latest = _read_json(DATA_DIR / "local_llm_report_latest.json")
 1227:     history = _read_jsonl(DATA_DIR / "local_llm_report_history.jsonl", limit=limit)
 1228: 
 1229:     return jsonify({
 1230:         "ticker": ticker,
 1231:         "latest": latest,
 1232:         "series": [
 1233:             _build_local_llm_trend_point(entry, ticker=ticker)
 1234:             for entry in history
 1235:             if isinstance(entry, dict)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
rg -n \"def require_auth|@app\\.route\\(|set_cookie|login|logout|SECRET|csrf|SameSite|samesite\" dashboard/app.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
rg -n \"run_safe\\(|popen_in_job\\(|subprocess\\.run\\(|subprocess\\.Popen\\(\" -g \"*.py\" portfolio" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
rg -n \"rotate_jsonl|move\\(|rotate_text|load_recent\\(|layer2_journal|JOURNAL_FILE|atomic_write_json\\(|save_beliefs\\(|load_beliefs\\(\" -g \"*.py\" portfolio dashboard" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 698ms:
750:@app.route("/")
762:@app.route("/legacy")
772:@app.route("/logout")
773:def logout():
782:    No `require_auth`: an unauthenticated visitor hitting /logout still gets
788:    response.set_cookie(
795:        samesite="Lax",
804:@app.route("/api/summary")
820:@app.route("/api/signals")
829:@app.route("/api/portfolio")
838:@app.route("/api/portfolio-bold")
847:@app.route("/api/mstr_loop")
894:@app.route("/api/invocations")
901:@app.route("/api/telegrams")
928:@app.route("/api/signal-log")
939:@app.route("/api/accuracy")
1019:@app.route("/api/iskbets")
1027:@app.route("/api/lora-status")
1039:@app.route("/api/validate-portfolio", methods=["POST"])
1067:@app.route("/api/equity-curve")
1082:@app.route("/api/signal-heatmap")
1181:@app.route("/api/triggers")
1189:@app.route("/api/accuracy-history")
1215:@app.route("/api/local-llm-trends")
1240:@app.route("/api/metals-accuracy")
1250:@app.route("/api/trades")
1281:@app.route("/api/decisions")
1324:@app.route("/api/health")
1340:@app.route("/api/warrants")
1357:@app.route("/api/risk")
1377:@app.route("/api/metals")
1434:@app.route("/api/crypto")
1465:@app.route("/api/btc")
1485:@app.route("/api/eth")
1505:@app.route("/api/loop_health")
1523:@app.route("/api/oil")
1561:@app.route("/api/mstr")
1587:@app.route("/api/golddigger")
1611:@app.route("/api/market-health")
1864:@app.route("/api/avanza_account")
1901:@app.route("/api/tradeable_assets")
1953:@app.route("/api/system_status")
1987:@app.route("/api/trading_status")

 succeeded in 712ms:
portfolio\subprocess_utils.py:4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
portfolio\subprocess_utils.py:6:- popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
portfolio\subprocess_utils.py:21:def run_safe(cmd, **kwargs):
portfolio\subprocess_utils.py:24:    Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
portfolio\subprocess_utils.py:28:    Falls back to plain subprocess.run() on non-Windows or if Job Object
portfolio\subprocess_utils.py:35:        return subprocess.run(cmd, **kwargs)
portfolio\subprocess_utils.py:41:        return subprocess.run(cmd, **kwargs)
portfolio\subprocess_utils.py:129:        proc = subprocess.Popen(cmd, **popen_kwargs)
portfolio\subprocess_utils.py:156:def popen_in_job(cmd, **kwargs):
portfolio\subprocess_utils.py:159:    Like subprocess.Popen(), but assigns the child to a Job Object with
portfolio\subprocess_utils.py:168:    proc = subprocess.Popen(cmd, **kwargs)
portfolio\subprocess_utils.py:215:        result = subprocess.run(
portfolio\subprocess_utils.py:239:            subprocess.run(
portfolio\subprocess_utils.py:275:        result = subprocess.run(
portfolio\gpu_gate.py:47:        proc = subprocess.run(

 succeeded in 702ms:
dashboard\app.py:1298:    for entry in _iter_latest_dict_entries(DATA_DIR / "layer2_journal.jsonl", read_limit=1000):
portfolio\file_utils.py:45:def atomic_write_json(path, data, indent=2, ensure_ascii=True):
portfolio\journal.py:14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\journal.py:23:def load_recent(max_entries=10, max_age_hours=8):
portfolio\journal.py:24:    if not JOURNAL_FILE.exists():
portfolio\journal.py:28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio\journal.py:577:    entries = load_recent()
portfolio\journal_index.py:19:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\journal_index.py:367:    if not JOURNAL_FILE.exists():
portfolio\journal_index.py:373:        with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio\log_rotation.py:95:    "layer2_journal.jsonl": {
portfolio\log_rotation.py:219:def rotate_jsonl(filename, policy, dry_run=False):
portfolio\log_rotation.py:333:def rotate_text(filename, policy, dry_run=False):
portfolio\log_rotation.py:414:        return rotate_jsonl(filename, policy, dry_run=dry_run)
portfolio\log_rotation.py:416:        return rotate_text(filename, policy, dry_run=dry_run)
portfolio\message_throttle.py:64:    atomic_write_json(PENDING_FILE, state)
portfolio\message_throttle.py:111:    atomic_write_json(PENDING_FILE, state)
portfolio\prophecy.py:55:def load_beliefs():
portfolio\prophecy.py:70:def save_beliefs(data):
portfolio\prophecy.py:73:    atomic_write_json(PROPHECY_FILE, data)
portfolio\prophecy.py:85:    data = load_beliefs()
portfolio\prophecy.py:101:    save_beliefs(data)
portfolio\prophecy.py:115:    data = load_beliefs()
portfolio\prophecy.py:122:            save_beliefs(data)
portfolio\prophecy.py:134:    data = load_beliefs()
portfolio\prophecy.py:139:        save_beliefs(data)
portfolio\prophecy.py:150:    data = load_beliefs()
portfolio\prophecy.py:163:    data = load_beliefs()
portfolio\prophecy.py:180:    data = load_beliefs()
portfolio\prophecy.py:195:            save_beliefs(data)
portfolio\prophecy.py:210:    data = load_beliefs()
portfolio\prophecy.py:267:        save_beliefs(data)
portfolio\prophecy.py:344:    data = load_beliefs()
portfolio\telegram_poller.py:98:            atomic_write_json(
portfolio\telegram_poller.py:361:        atomic_write_json(config_path, cfg)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 700 810" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 637ms:
=== dashboard/app.py:700-810 ===
  700:         "ticker": ticker,
  701:         "ministral_accuracy": ministral.get("accuracy"),
  702:         "ministral_samples": ministral.get("samples", 0),
  703:         "ministral_ticker_accuracy": ticker_stats.get("accuracy"),
  704:         "ministral_ticker_samples": ticker_stats.get("samples", 0),
  705:         "chronos_success_rate": (health.get("chronos") or {}).get("success_rate"),
  706:         "chronos_total": (health.get("chronos") or {}).get("total", 0),
  707:         "kronos_success_rate": (health.get("kronos") or {}).get("success_rate"),
  708:         "kronos_total": (health.get("kronos") or {}).get("total", 0),
  709:         "forecast_raw_1h_accuracy": raw_1h["accuracy"],
  710:         "forecast_raw_1h_total": raw_1h["total"],
  711:         "forecast_raw_24h_accuracy": raw_24h["accuracy"],
  712:         "forecast_raw_24h_total": raw_24h["total"],
  713:         "forecast_effective_1h_accuracy": effective_1h["accuracy"],
  714:         "forecast_effective_1h_total": effective_1h["total"],
  715:         "forecast_effective_24h_accuracy": effective_24h["accuracy"],
  716:         "forecast_effective_24h_total": effective_24h["total"],
  717:         "forecast_gating_raw": gating.get("raw", 0),
  718:         "forecast_gating_held": gating.get("held", 0),
  719:         "forecast_gating_insufficient_data": gating.get("insufficient_data", 0),
  720:         "forecast_gating_vol_gated": gating.get("vol_gated", 0),
  721:     }
  722: 
  723: 
  724: # ---------------------------------------------------------------------------
  725: # Token authentication middleware
  726: # ---------------------------------------------------------------------------
  727: 
  728: # Auth + cookie machinery moved to dashboard/auth.py on 2026-05-02 to break
  729: # the circular import with dashboard/house_blueprint.py. We re-import here
  730: # so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
  731: # inside this module's body, and so any lingering external code that does
  732: # `from dashboard.app import require_auth` still resolves. Tests should
  733: # patch `dashboard.auth.*` directly — patches on `dashboard.app.*` will not
  734: # take effect since require_auth resolves names via dashboard.auth's
  735: # module globals.
  736: from dashboard.auth import (  # noqa: E402
  737:     COOKIE_MAX_AGE,
  738:     COOKIE_NAME,
  739:     _get_config as _auth_get_config,  # noqa: F401 — kept for compat
  740:     _get_dashboard_token,
  741:     _refresh_cookie,
  742:     require_auth,
  743: )
  744: 
  745: 
  746: # ---------------------------------------------------------------------------
  747: # Routes — Static
  748: # ---------------------------------------------------------------------------
  749: 
  750: @app.route("/")
  751: @require_auth
  752: def index():
  753:     # If the user arrived via ?token=XXX, the cookie was just set in
  754:     # require_auth. Redirect to a token-less URL so the address bar (and
  755:     # whatever the user bookmarks next) stays clean. The redirect inherits
  756:     # the Set-Cookie from require_auth's wrapped response.
  757:     if request.args.get("token"):
  758:         return redirect("/", code=302)
  759:     return send_from_directory("static", "index.html")
  760: 
  761: 
  762: @app.route("/legacy")
  763: @require_auth
  764: def index_legacy():
  765:     # Pre-redesign single-file dashboard preserved as a fallback during the
  766:     # 2026-05-03 mobile-first rollout. See docs/PLAN.md.
  767:     if request.args.get("token"):
  768:         return redirect("/legacy", code=302)
  769:     return send_from_directory("static", "index_legacy.html")
  770: 
  771: 
  772: @app.route("/logout")
  773: def logout():
  774:     """Clear the pf_dashboard_token cookie and redirect to /.
  775: 
  776:     The auth cookie is HttpOnly, so client JS cannot expire it via
  777:     document.cookie — the browser ignores any attempt to write a name that
  778:     Set-Cookie marked HttpOnly. The mobile Settings → Sign out button
  779:     therefore has to navigate here so the server can emit the matching
  780:     Set-Cookie with Max-Age=0. (Codex P2 finding 2026-05-03.)
  781: 
  782:     No `require_auth`: an unauthenticated visitor hitting /logout still gets
  783:     the cookie wiped (harmless — they had no valid cookie anyway) and
  784:     Cloudflare Access still gates the redirected destination.
  785:     """
  786:     response = redirect("/", code=302)
  787:     # Match every flag we set on the original cookie except expiry.
  788:     response.set_cookie(
  789:         "pf_dashboard_token",
  790:         "",
  791:         max_age=0,
  792:         expires=0,
  793:         httponly=True,
  794:         secure=True,
  795:         samesite="Lax",
  796:     )
  797:     return response
  798: 
  799: 
  800: # ---------------------------------------------------------------------------
  801: # Routes — API (all require auth)
  802: # ---------------------------------------------------------------------------
  803: 
  804: @app.route("/api/summary")
  805: @require_auth
  806: def api_summary():
  807:     """Combined endpoint for auto-refresh: signals + both portfolios + telegrams."""
  808:     sig = _read_json(DATA_DIR / "agent_summary.json")
  809:     port = _read_json(DATA_DIR / "portfolio_state.json")
  810:     port_bold = _read_json(DATA_DIR / "portfolio_state_bold.json")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/auth.py' 1 220" in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:27:08.876644Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
=== dashboard/auth.py:1-220 ===
[31;1mResolve-Path: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   6 | [0m   Get-Content ([36;1mResolve-Path $path[0m) | ForEach-Object { $i++; if($i -ge …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                ~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\auth.py' because it does not exist.[0m

 exited 1 in 819ms:
=== dashboard/auth.py:1-220 ===
[31;1mResolve-Path: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   6 | [0m   Get-Content ([36;1mResolve-Path $path[0m) | ForEach-Object { $i++; if($i -ge …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                ~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\auth.py' because it does not exist.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files -g "auth.py" dashboard' in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:27:11.802860Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:

 exited 1 in 745ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "COOKIE_NAME|def require_auth|_refresh_cookie|_get_dashboard_token" -g "*.py"' in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:27:15.073701Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 648ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -Depth 2 dashboard | Select-Object FullName' in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 567ms:

[32;1mFullName[0m
[32;1m--------[0m
Q:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\app.py


codex
I hit something unexpected in the dashboard auth path: `app.py` imports `dashboard.auth`, but that file does not exist in this worktree. I’m checking whether that’s a real break in the branch or just an odd namespace/package layout before I treat any dashboard-auth conclusion as solid.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"dashboard\\.auth|from dashboard\\.auth|import dashboard\\.auth|def require_auth\" -g \"*.py\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force dashboard | Format-List -Property Name,FullName,Mode,Length' in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git status --short' in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:27:29.082657Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-infrastructure'
'Q:/finance-analyzer/.worktrees/adv-infrastructure/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-infrastructure

 exited 1 in 541ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-infrastructure'
'Q:/finance-analyzer/.worktrees/adv-infrastructure/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-infrastructure

 succeeded in 571ms:
.\dashboard\app.py:733:# patch `dashboard.auth.*` directly — patches on `dashboard.app.*` will not
.\dashboard\app.py:734:# take effect since require_auth resolves names via dashboard.auth's
.\dashboard\app.py:736:from dashboard.auth import (  # noqa: E402
.\dashboard\app.py:2017:# auth via dashboard.auth.require_auth. Path roots come from
.\dashboard\app.py:2021:# dashboard.auth (NOT dashboard.app), so importing it here at module-init

 succeeded in 564ms:

[32;1mName     : [0mapp.py
[32;1mFullName : [0mQ:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\app.py
[32;1mMode     : [0m-a---
[32;1mLength   : [0m80722


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 1988 2050" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 640ms:
=== dashboard/app.py:1988-2050 ===
 1988: @require_auth
 1989: def api_trading_status():
 1990:     """Per-bot Avanza trading state with reason.
 1991: 
 1992:     See dashboard/trading_status.py. Each bot resolves to one of
 1993:     SCANNING / TRADING / HALTED / COOLDOWN / OUTSIDE_HOURS / UNKNOWN.
 1994:     Same lock discipline as ``/api/system_status``.
 1995:     """
 1996:     from dashboard import trading_status as _trading_status
 1997: 
 1998:     force = request.args.get("force", "").strip() in {"1", "true", "yes"}
 1999:     if not force:
 2000:         with _TRADING_STATUS_LOCK:
 2001:             cached = _TRADING_STATUS_CACHE.get("value")
 2002:             if cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
 2003:                 return jsonify(cached)
 2004:     with _TRADING_STATUS_LOCK:
 2005:         cached = _TRADING_STATUS_CACHE.get("value")
 2006:         if not force and cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
 2007:             return jsonify(cached)
 2008:         payload = _trading_status.compute()
 2009:         _TRADING_STATUS_CACHE["value"] = payload
 2010:         _TRADING_STATUS_CACHE["at"] = time.monotonic()
 2011:         return jsonify(payload)
 2012: 
 2013: 
 2014: # ---------------------------------------------------------------------------
 2015: # Blueprint: /house — read-only viewer over the househunting project
 2016: # (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
 2017: # auth via dashboard.auth.require_auth. Path roots come from
 2018: # config.json[house_root]. See dashboard/house_blueprint.py for routes.
 2019: #
 2020: # House_blueprint imports `_get_config` and `require_auth` from
 2021: # dashboard.auth (NOT dashboard.app), so importing it here at module-init
 2022: # time no longer causes a circular import — auth.py has no back-reference
 2023: # to app.py. The sys.modules alias hack added 2026-05-02 has been removed.
 2024: # ---------------------------------------------------------------------------
 2025: from dashboard.house_blueprint import bp as _house_bp  # noqa: E402
 2026: 
 2027: app.register_blueprint(_house_bp)
 2028: 
 2029: 
 2030: def _serve_dual_stack(port: int = 5055) -> None:
 2031:     """Run the Flask app on a dual-stack IPv4+IPv6 socket.
 2032: 
 2033:     2026-05-04: previously used `app.run(host="0.0.0.0", ...)` which is
 2034:     IPv4-only. Local Python tooling (urllib, requests) on Windows that
 2035:     resolves "localhost" to ::1 first then waits ~2s for the IPv6
 2036:     connection to fail before falling back to IPv4 — perceived as a
 2037:     universal "2s auth floor" but actually a client-side Happy Eyeballs
 2038:     timeout. Real users (Cloudflare tunnel, LAN browsers) never see it.
 2039: 
 2040:     Switching to `host="::"` would fix localhost on Linux but on
 2041:     Windows the default `IPV6_V6ONLY=True` socket option means IPv4
 2042:     clients can no longer connect. So we bind manually with
 2043:     `IPV6_V6ONLY=0`, which works on every modern Windows / Linux /
 2044:     macOS host.
 2045:     """
 2046:     import socket
 2047:     from werkzeug.serving import ThreadedWSGIServer
 2048: 
 2049:     # Build the dual-stack listening socket explicitly. IPV6_V6ONLY=0
 2050:     # enables IPv4 mapping (::ffff:127.0.0.1 etc.), so a single AF_INET6

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files -g "dashboard.py" -g "house_blueprint.py" -g "trading_status.py" -g "auth.py" .' in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:27:40.620128Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:

 exited 1 in 514ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 1560 1620
Show-Lines 'dashboard/app.py' 170 182" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/log_rotation.py' 219 340
Show-Lines 'portfolio/journal_index.py' 350 420" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/prophecy.py' 314 328
Show-Lines 'portfolio/telegram_notifications.py' 52 80
Show-Lines 'portfolio/message_store.py' 137 166" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 606ms:
=== portfolio/prophecy.py:314-328 ===
  314:         }
  315: 
  316:         # Compute progress toward target
  317:         if prices_usd and belief.get("ticker") in prices_usd and belief.get("target_price") and belief.get("entry_price"):
  318:             current = prices_usd[belief["ticker"]]
  319:             entry_price = belief["entry_price"]
  320:             target = belief["target_price"]
  321: 
  322:             if target != entry_price:
  323:                 progress = (current - entry_price) / (target - entry_price) * 100
  324:                 entry["progress_pct"] = round(progress, 1)
  325:                 entry["current_price"] = round(current, 2)
  326: 
  327:         # Checkpoint summary
  328:         checkpoints = belief.get("checkpoints", [])
=== portfolio/telegram_notifications.py:52-80 ===
   52:     token = config["telegram"]["token"]
   53:     chat_id = config["telegram"]["chat_id"]
   54:     r = fetch_with_retry(
   55:         f"https://api.telegram.org/bot{token}/sendMessage",
   56:         method="POST",
   57:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
   58:         timeout=30,
   59:     )
   60:     if r is None:
   61:         return False
   62:     if r.ok:
   63:         return True
   64:     # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
   65:     # still arrives (unformatted) rather than being silently lost.
   66:     if r.status_code == 400:
   67:         err_desc = ""
   68:         try:
   69:             err_desc = r.json().get("description", "")
   70:         except Exception:
   71:             logger.debug("Failed to parse Telegram error response", exc_info=True)
   72:         if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
   73:             logger.warning("Telegram Markdown parse failed (%s), resending without formatting", err_desc)
   74:             r2 = fetch_with_retry(
   75:                 f"https://api.telegram.org/bot{token}/sendMessage",
   76:                 method="POST",
   77:                 json_body={"chat_id": chat_id, "text": msg},
   78:                 timeout=30,
   79:             )
   80:             return r2 is not None and r2.ok
=== portfolio/message_store.py:137-166 ===
  137:     r = fetch_with_retry(
  138:         f"https://api.telegram.org/bot{token}/sendMessage",
  139:         method="POST",
  140:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
  141:         timeout=30,
  142:     )
  143:     if r is None:
  144:         return False
  145:     if r.ok:
  146:         return True
  147: 
  148:     # Markdown parse failure (HTTP 400) — retry without parse_mode
  149:     if r.status_code == 400:
  150:         err_desc = ""
  151:         try:
  152:             err_desc = r.json().get("description", "")
  153:         except Exception as e:
  154:             logger.debug("Failed to parse Telegram error response: %s", e)
  155:         if any(kw in err_desc.lower() for kw in ("parse", "markdown", "entity")):
  156:             logger.warning(
  157:                 "Telegram Markdown parse failed (%s), resending without formatting",
  158:                 err_desc,
  159:             )
  160:             r2 = fetch_with_retry(
  161:                 f"https://api.telegram.org/bot{token}/sendMessage",
  162:                 method="POST",
  163:                 json_body={"chat_id": chat_id, "text": msg},
  164:                 timeout=30,
  165:             )
  166:             return r2 is not None and r2.ok

 succeeded in 638ms:
=== portfolio/log_rotation.py:219-340 ===
  219: def rotate_jsonl(filename, policy, dry_run=False):
  220:     """Rotate a JSONL file by age: archive old entries, keep recent ones.
  221: 
  222:     Old entries are grouped by year-month and written to
  223:     data/archive/FILENAME.YYYY-MM.jsonl.gz
  224: 
  225:     Returns dict with rotation stats.
  226:     """
  227:     filepath = DATA_DIR / filename
  228:     if not filepath.exists():
  229:         return {"file": filename, "status": "not_found"}
  230: 
  231:     size_mb = _file_size_mb(filepath)
  232:     ts_field = policy.get("ts_field", "ts")
  233:     max_age_days = policy.get("max_age_days", 30)
  234:     cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=max_age_days)
  235: 
  236:     # Read all lines and classify as keep vs archive
  237:     keep_lines = []
  238:     archive_buckets = {}  # "YYYY-MM" -> list of raw lines
  239:     parse_failures = 0
  240:     total_lines = 0
  241: 
  242:     with open(filepath, encoding="utf-8") as f:
  243:         for line in f:
  244:             line = line.rstrip("\n")
  245:             if not line.strip():
  246:                 continue
  247:             total_lines += 1
  248:             try:
  249:                 entry = json.loads(line)
  250:                 ts = _parse_ts(entry.get(ts_field))
  251:                 if ts is None:
  252:                     # Can't determine age -- keep the entry to be safe
  253:                     keep_lines.append(line)
  254:                     parse_failures += 1
  255:                 elif ts >= cutoff:
  256:                     keep_lines.append(line)
  257:                 else:
  258:                     # Archive this entry, grouped by month
  259:                     month_key = ts.strftime("%Y-%m")
  260:                     archive_buckets.setdefault(month_key, []).append(line)
  261:             except json.JSONDecodeError:
  262:                 # Malformed line -- keep it to avoid data loss
  263:                 keep_lines.append(line)
  264:                 parse_failures += 1
  265: 
  266:     archived_count = sum(len(v) for v in archive_buckets.values())
  267:     result = {
  268:         "file": filename,
  269:         "size_mb": round(size_mb, 2),
  270:         "total_lines": total_lines,
  271:         "kept": len(keep_lines),
  272:         "archived": archived_count,
  273:         "archive_months": sorted(archive_buckets.keys()),
  274:         "parse_failures": parse_failures,
  275:     }
  276: 
  277:     if archived_count == 0:
  278:         result["status"] = "nothing_to_archive"
  279:         return result
  280: 
  281:     if dry_run:
  282:         result["status"] = "dry_run"
  283:         return result
  284: 
  285:     _ensure_archive_dir()
  286: 
  287:     # Write archived entries to monthly files
  288:     stem = pathlib.Path(filename).stem  # e.g. "signal_log"
  289:     suffix = pathlib.Path(filename).suffix  # e.g. ".jsonl"
  290: 
  291:     for month_key, lines in sorted(archive_buckets.items()):
  292:         archive_name = f"{stem}.{month_key}{suffix}"
  293:         archive_path = ARCHIVE_DIR / archive_name
  294:         gz_path = ARCHIVE_DIR / f"{archive_name}.gz"
  295: 
  296:         # Append to existing archive for this month (may already have entries
  297:         # from a previous rotation)
  298:         if gz_path.exists() and policy.get("compress", True):
  299:             # Decompress existing, append, re-compress
  300:             existing_lines = []
  301:             with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
  302:                 for existing_line in gf:
  303:                     existing_line = existing_line.rstrip("\n")
  304:                     if existing_line.strip():
  305:                         existing_lines.append(existing_line)
  306:             all_lines = existing_lines + lines
  307:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  308:                 for line in all_lines:
  309:                     gf.write(line + "\n")
  310:         elif policy.get("compress", True):
  311:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  312:                 for line in lines:
  313:                     gf.write(line + "\n")
  314:         else:
  315:             with open(archive_path, "a", encoding="utf-8") as af:
  316:                 for line in lines:
  317:                     af.write(line + "\n")
  318: 
  319:     # Rewrite the original file with only kept lines
  320:     tmp_path = filepath.with_suffix(".tmp")
  321:     with open(tmp_path, "w", encoding="utf-8") as f:
  322:         for line in keep_lines:
  323:             f.write(line + "\n")
  324: 
  325:     # Atomic-ish replace: remove original, rename tmp
  326:     # On Windows, os.replace is atomic within the same volume
  327:     os.replace(tmp_path, filepath)
  328: 
  329:     result["status"] = "rotated"
  330:     return result
  331: 
  332: 
  333: def rotate_text(filename, policy, dry_run=False):
  334:     """Rotate a plain text file by size.
  335: 
  336:     When file exceeds max_size_mb:
  337:     - Shift existing rotations: .2 -> .3, .1 -> .2, current -> .1
  338:     - Delete rotations beyond keep_rotations
  339:     - Compress old rotations if policy says so
  340: 
=== portfolio/journal_index.py:350-420 ===
  350: 
  351: def retrieve_relevant_entries(signals, held_tickers, regime, prices, k=8):
  352:     """Retrieve the most relevant journal entries for the current market state.
  353: 
  354:     This is the main entry point called by journal.py.
  355: 
  356:     Args:
  357:         signals: dict of ticker -> signal data.
  358:         held_tickers: list of currently held ticker symbols.
  359:         regime: str (current market regime).
  360:         prices: dict of ticker -> current USD price.
  361:         k: number of entries to return.
  362: 
  363:     Returns:
  364:         list of journal entry dicts, ranked by relevance.
  365:         Falls back to chronological (most recent) on any error.
  366:     """
  367:     if not JOURNAL_FILE.exists():
  368:         return []
  369: 
  370:     # Load all entries
  371:     entries = []
  372:     try:
  373:         with open(JOURNAL_FILE, encoding="utf-8") as f:
  374:             for line in f:
  375:                 line = line.strip()
  376:                 if not line:
  377:                     continue
  378:                 try:
  379:                     entries.append(json.loads(line))
  380:                 except json.JSONDecodeError:
  381:                     continue
  382:     except OSError:
  383:         return []
  384: 
  385:     if not entries:
  386:         return []
  387: 
  388:     # Build index and query
  389:     index = JournalIndex()
  390:     index.build(entries)
  391: 
  392:     market_state = {
  393:         "held_tickers": held_tickers or [],
  394:         "regime": regime or "",
  395:         "prices": prices or {},
  396:         "signals": signals or {},
  397:     }
  398: 
  399:     return index.query(market_state, k=k)

 succeeded in 698ms:
=== dashboard/app.py:1560-1620 ===
 1560: 
 1561: @app.route("/api/mstr")
 1562: @require_auth
 1563: def api_mstr():
 1564:     """MSTR deep-context endpoint.
 1565: 
 1566:     The pre-existing `/api/mstr_loop` returns the strategy-loop state
 1567:     (positions, scorecard, last poll). This new endpoint returns the deep
 1568:     context (NAV premium, BTC correlation, options skew, analyst consensus)
 1569:     written by `portfolio/mstr_precompute.py`. Together they parallel
 1570:     `/api/metals` (decisions+context) for the metals subsystem.
 1571:     """
 1572:     deep = _read_json(DATA_DIR / "mstr_deep_context.json") or {}
 1573:     loop_state = _read_json(DATA_DIR / "mstr_loop_state.json") or {}
 1574:     scorecard = _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {}
 1575:     return jsonify({
 1576:         "ticker": "MSTR",
 1577:         "deep_context": deep,
 1578:         "loop_state": loop_state,
 1579:         "scorecard": scorecard,
 1580:     })
 1581: 
 1582: 
 1583: # ---------------------------------------------------------------------------
 1584: # New: GoldDigger monitoring
 1585: # ---------------------------------------------------------------------------
 1586: 
 1587: @app.route("/api/golddigger")
 1588: @require_auth
 1589: def api_golddigger():
 1590:     """Return GoldDigger signal data normalized for the dashboard.
 1591: 
 1592:     The bot persists a lean state snapshot plus compact JSONL logs. This route
 1593:     reshapes those records into the richer schema expected by the dashboard UI.
 1594:     """
 1595:     raw_log = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_log.jsonl", read_limit=100))
 1596:     raw_trades = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_trades.jsonl", read_limit=50))
 1597:     state = _normalize_golddigger_state(_read_json(DATA_DIR / "golddigger_state.json"), raw_log)
 1598:     log = [entry for entry in (_normalize_golddigger_log_entry(item) for item in raw_log) if entry]
 1599:     trades = [entry for entry in (_normalize_golddigger_trade_entry(item) for item in raw_trades) if entry]
 1600:     return jsonify({
 1601:         "state": state if state or log or trades else None,
 1602:         "log": log,
 1603:         "trades": trades,
 1604:     })
 1605: 
 1606: 
 1607: # ---------------------------------------------------------------------------
 1608: # Market health
 1609: # ---------------------------------------------------------------------------
 1610: 
 1611: @app.route("/api/market-health")
 1612: @require_auth
 1613: def api_market_health():
 1614:     """Return market health snapshot (distribution days, FTD, breadth score).
 1615: 
 1616:     Also includes exposure recommendation and earnings proximity data.
 1617:     """
 1618:     try:
 1619:         result = {}
 1620:         # Market health from agent_summary (pre-computed) or live
=== dashboard/app.py:170-182 ===
  170: 
  171: def _get_config():
  172:     return _read_json(CONFIG_PATH, ttl=60) or {}
  173: 
  174: 
  175: def _parse_limit_arg(name, default, max_value):
  176:     """Parse integer query arg with sane bounds and fallback."""
  177:     try:
  178:         value = int(request.args.get(name, default))
  179:     except (ValueError, TypeError):
  180:         value = default
  181:     return max(1, min(value, max_value))
  182: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/gpu_gate.py' 258 320" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 575ms:
=== portfolio/gpu_gate.py:258-320 ===
  258:         t0 = time.time()
  259:         try:
  260:             yield True
  261:         finally:
  262:             elapsed = time.time() - t0
  263:             vram = get_vram_usage()
  264:             if vram:
  265:                 logger.info(
  266:                     "GPU gate RELEASED by %s after %.1fs — VRAM: %dMB used / %dMB free",
  267:                     model_name, elapsed, vram["used_mb"], vram["free_mb"],
  268:                 )
  269:             _release_lock()
  270:     finally:
  271:         _THREAD_LOCK.release()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"add_belief\\(|update_belief\\(|remove_belief\\(|add_checkpoint\\(|evaluate_checkpoints\\(|save_beliefs\\(\" -g \"*.py\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 506ms:
.\portfolio\prophecy.py:70:def save_beliefs(data):
.\portfolio\prophecy.py:76:def add_belief(belief_dict):
.\portfolio\prophecy.py:101:    save_beliefs(data)
.\portfolio\prophecy.py:105:def update_belief(belief_id, updates):
.\portfolio\prophecy.py:122:            save_beliefs(data)
.\portfolio\prophecy.py:128:def remove_belief(belief_id):
.\portfolio\prophecy.py:139:        save_beliefs(data)
.\portfolio\prophecy.py:170:def add_checkpoint(belief_id, checkpoint_dict):
.\portfolio\prophecy.py:195:            save_beliefs(data)
.\portfolio\prophecy.py:201:def evaluate_checkpoints(prices_usd):
.\portfolio\prophecy.py:267:        save_beliefs(data)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"import .*prophecy|from portfolio\\.prophecy|load_beliefs\\(|get_context_for_layer2\\(|evaluate_checkpoints\\(\" -g \"*.py\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 510ms:
.\portfolio\prophecy.py:55:def load_beliefs():
.\portfolio\prophecy.py:85:    data = load_beliefs()
.\portfolio\prophecy.py:115:    data = load_beliefs()
.\portfolio\prophecy.py:134:    data = load_beliefs()
.\portfolio\prophecy.py:150:    data = load_beliefs()
.\portfolio\prophecy.py:163:    data = load_beliefs()
.\portfolio\prophecy.py:180:    data = load_beliefs()
.\portfolio\prophecy.py:201:def evaluate_checkpoints(prices_usd):
.\portfolio\prophecy.py:210:    data = load_beliefs()
.\portfolio\prophecy.py:272:def get_context_for_layer2(prices_usd=None):
.\portfolio\prophecy.py:344:    data = load_beliefs()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_GPU_LOCK_DIR|mkdir\\(|Q:/models|gpu_gate\\(\" portfolio/gpu_gate.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 622ms:
7:plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.
10:- Reactive: ``gpu_gate()`` calls ``_try_break_stale_lock()`` when another
12:- Background: a daemon thread (lazily spawned on first ``gpu_gate()`` call)
33:_GPU_LOCK_DIR = Path("Q:/models")
34:_GPU_LOCK_FILE = _GPU_LOCK_DIR / ".gpu_lock"
117:    - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
165:    Lazily called from ``gpu_gate()`` so:
167:      ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
187:def gpu_gate(model_name: str, timeout: float = 60):

codex
I’ve got enough code context to classify most of Claude’s list. One notable side issue surfaced while doing that: this worktree’s `dashboard` package is missing the modules that `dashboard/app.py` imports (`dashboard.auth`, `dashboard.house_blueprint`, `dashboard.trading_status`), so I’m verifying whether the app actually fails to import before I decide whether that belongs as a new finding.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -c \"import importlib.util, sys; spec = importlib.util.spec_from_file_location('dashboard.app', r'Q:\\finance-analyzer\\.worktrees\\adv-infrastructure\\dashboard\\app.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')\"" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"atomic_append_jsonl\\(|load_jsonl_tail\\(|rotate_jsonl\\(|rotate_text\\(|require_auth|_cached\\(|_cached_or_enqueue\\(|_update_cache\\(\" -g \"*.py\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 527ms:
.\portfolio\telegram_poller.py:289:            atomic_append_jsonl(INBOUND_LOG, entry)
.\portfolio\shared_state.py:17:# Retry cooldown for _cached() errors
.\portfolio\shared_state.py:37:def _cached(key, ttl, func, *args):
.\portfolio\shared_state.py:127:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
.\portfolio\shared_state.py:131:    Unlike _cached(), this never calls the model directly. On miss, it adds
.\portfolio\shared_state.py:165:        # as "stale NOT available". main.py writes _update_cache(key, None)
.\portfolio\shared_state.py:225:def _update_cache(key, data, ttl=None):
.\portfolio\message_store.py:102:    atomic_append_jsonl(MESSAGES_FILE, entry)
.\portfolio\log_rotation.py:219:def rotate_jsonl(filename, policy, dry_run=False):
.\portfolio\log_rotation.py:333:def rotate_text(filename, policy, dry_run=False):
.\portfolio\log_rotation.py:414:        return rotate_jsonl(filename, policy, dry_run=dry_run)
.\portfolio\log_rotation.py:416:        return rotate_text(filename, policy, dry_run=dry_run)
.\portfolio\file_utils.py:136:def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
.\portfolio\file_utils.py:202:def atomic_append_jsonl(path, entry):
.\dashboard\app.py:730:# so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
.\dashboard\app.py:732:# `from dashboard.app import require_auth` still resolves. Tests should
.\dashboard\app.py:734:# take effect since require_auth resolves names via dashboard.auth's
.\dashboard\app.py:742:    require_auth,
.\dashboard\app.py:751:@require_auth
.\dashboard\app.py:754:    # require_auth. Redirect to a token-less URL so the address bar (and
.\dashboard\app.py:756:    # the Set-Cookie from require_auth's wrapped response.
.\dashboard\app.py:763:@require_auth
.\dashboard\app.py:782:    No `require_auth`: an unauthenticated visitor hitting /logout still gets
.\dashboard\app.py:805:@require_auth
.\dashboard\app.py:821:@require_auth
.\dashboard\app.py:830:@require_auth
.\dashboard\app.py:839:@require_auth
.\dashboard\app.py:848:@require_auth
.\dashboard\app.py:895:@require_auth
.\dashboard\app.py:902:@require_auth
.\dashboard\app.py:929:@require_auth
.\dashboard\app.py:940:@require_auth
.\dashboard\app.py:1020:@require_auth
.\dashboard\app.py:1028:@require_auth
.\dashboard\app.py:1040:@require_auth
.\dashboard\app.py:1068:@require_auth
.\dashboard\app.py:1083:@require_auth
.\dashboard\app.py:1182:@require_auth
.\dashboard\app.py:1190:@require_auth
.\dashboard\app.py:1216:@require_auth
.\dashboard\app.py:1241:@require_auth
.\dashboard\app.py:1251:@require_auth
.\dashboard\app.py:1282:@require_auth
.\dashboard\app.py:1325:@require_auth
.\dashboard\app.py:1341:@require_auth
.\dashboard\app.py:1358:@require_auth
.\dashboard\app.py:1378:@require_auth
.\dashboard\app.py:1435:@require_auth
.\dashboard\app.py:1466:@require_auth
.\dashboard\app.py:1486:@require_auth
.\dashboard\app.py:1506:@require_auth
.\dashboard\app.py:1524:@require_auth
.\dashboard\app.py:1562:@require_auth
.\dashboard\app.py:1588:@require_auth
.\dashboard\app.py:1612:@require_auth
.\dashboard\app.py:1865:@require_auth
.\dashboard\app.py:1902:@require_auth
.\dashboard\app.py:1954:@require_auth
.\dashboard\app.py:1988:@require_auth
.\dashboard\app.py:2017:# auth via dashboard.auth.require_auth. Path roots come from
.\dashboard\app.py:2020:# House_blueprint imports `_get_config` and `require_auth` from

2026-05-10T15:30:00.627303Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\app.py", line 14, in <module>
    from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
ModuleNotFoundError: No module named 'flask'

 exited 1 in 610ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-infrastructure\dashboard\app.py", line 14, in <module>
    from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
ModuleNotFoundError: No module named 'flask'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"methods=\\[\\\"POST\\\"\\]|methods=\\['POST'\\]|POST\\]\" dashboard/app.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:30:07.528936Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
rg: regex parse error:
    (?:methods=\[\)
    ^
error: unclosed group

 exited 1 in 608ms:
rg: regex parse error:
    (?:methods=\[\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "methods=.*POST" dashboard/app.py' in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 528ms:
1039:@app.route("/api/validate-portfolio", methods=["POST"])

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"layer2_journal\\.jsonl|JOURNAL_FILE|atomic_append_jsonl\\(\" portfolio/journal.py portfolio/*.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:30:14.471581Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
portfolio/journal.py:14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/journal.py:24:    if not JOURNAL_FILE.exists():
portfolio/journal.py:28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 513ms:
portfolio/journal.py:14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/journal.py:24:    if not JOURNAL_FILE.exists():
portfolio/journal.py:28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"JOURNAL_FILE|layer2_journal\\.jsonl|atomic_append_jsonl\\(\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 524ms:
.\portfolio\log_rotation.py:95:    "layer2_journal.jsonl": {
.\portfolio\message_store.py:102:    atomic_append_jsonl(MESSAGES_FILE, entry)
.\portfolio\telegram_poller.py:289:            atomic_append_jsonl(INBOUND_LOG, entry)
.\portfolio\file_utils.py:202:def atomic_append_jsonl(path, entry):
.\portfolio\journal_index.py:19:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
.\portfolio\journal_index.py:367:    if not JOURNAL_FILE.exists():
.\portfolio\journal_index.py:373:        with open(JOURNAL_FILE, encoding="utf-8") as f:
.\portfolio\journal.py:14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
.\portfolio\journal.py:24:    if not JOURNAL_FILE.exists():
.\portfolio\journal.py:28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
.\dashboard\app.py:1298:    for entry in _iter_latest_dict_entries(DATA_DIR / "layer2_journal.jsonl", read_limit=1000):

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_loading_timestamps\\.pop|_loading_timestamps\\[|_loading_keys\\.discard\\(key\\)\" portfolio/shared_state.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 522ms:
74:            stuck_duration = _now_evict - _loading_timestamps.pop(k, _now_evict)
90:        _loading_timestamps[key] = time.time()
101:            _loading_keys.discard(key)
102:            _loading_timestamps.pop(key, None)  # BUG-213: clean up on success path
106:            _loading_keys.discard(key)
112:            _loading_keys.discard(key)
202:            _loading_timestamps[key] = time.time()
206:                _loading_keys.discard(key)
207:                _loading_timestamps.pop(key, None)
228:        _loading_keys.discard(key)
230:        _loading_timestamps.pop(key, None)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-infrastructure ls-tree -r --name-only HEAD dashboard' in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 540ms:
dashboard/app.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_LOADING_TIMEOUT|stale-while-revalidate|stuck loading key|_update_cache\\(|_cached_or_enqueue\\(\" portfolio/shared_state.py portfolio/*.py" in Q:\finance-analyzer\.worktrees\adv-infrastructure
2026-05-10T15:30:55.674264Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
portfolio/shared_state.py:29:_LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
portfolio/shared_state.py:68:        # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
portfolio/shared_state.py:71:                 if _now_evict - ts > _LOADING_TIMEOUT]
portfolio/shared_state.py:75:            logger.warning("[%s] evicted stuck loading key after %.0fs (timeout %ds)", k, stuck_duration, _LOADING_TIMEOUT)
portfolio/shared_state.py:84:                    logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
portfolio/shared_state.py:127:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
portfolio/shared_state.py:165:        # as "stale NOT available". main.py writes _update_cache(key, None)
portfolio/shared_state.py:225:def _update_cache(key, data, ttl=None):
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 509ms:
portfolio/shared_state.py:29:_LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
portfolio/shared_state.py:68:        # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
portfolio/shared_state.py:71:                 if _now_evict - ts > _LOADING_TIMEOUT]
portfolio/shared_state.py:75:            logger.warning("[%s] evicted stuck loading key after %.0fs (timeout %ds)", k, stuck_duration, _LOADING_TIMEOUT)
portfolio/shared_state.py:84:                    logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
portfolio/shared_state.py:127:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
portfolio/shared_state.py:165:        # as "stale NOT available". main.py writes _update_cache(key, None)
portfolio/shared_state.py:225:def _update_cache(key, data, ttl=None):
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_cached_or_enqueue\\(\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 525ms:
.\portfolio\shared_state.py:127:def _cached_or_enqueue(key, ttl, enqueue_fn, context,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_update_cache\\(\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 553ms:
.\portfolio\shared_state.py:165:        # as "stale NOT available". main.py writes _update_cache(key, None)
.\portfolio\shared_state.py:225:def _update_cache(key, data, ttl=None):

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"\\.wait\\(\\)|_RateLimiter|_alpaca_limiter|_binance_limiter|_yfinance_limiter|_alpha_vantage_limiter\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 508ms:
.\portfolio\shared_state.py:253:class _RateLimiter:
.\portfolio\shared_state.py:288:_alpaca_limiter = _RateLimiter(150, "alpaca")
.\portfolio\shared_state.py:290:_binance_limiter = _RateLimiter(600, "binance")
.\portfolio\shared_state.py:292:_yfinance_limiter = _RateLimiter(30, "yfinance")
.\portfolio\shared_state.py:296:_alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"AlertBudget|should_send\\(|remaining_budget|buffer_size|PRIORITY_\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 555ms:
.\portfolio\alert_budget.py:16:PRIORITY_EMERGENCY = 3
.\portfolio\alert_budget.py:17:PRIORITY_IMPORTANT = 2
.\portfolio\alert_budget.py:18:PRIORITY_NORMAL = 1
.\portfolio\alert_budget.py:21:class AlertBudget:
.\portfolio\alert_budget.py:36:    def should_send(self, message: str, priority: int = PRIORITY_NORMAL) -> bool:
.\portfolio\alert_budget.py:38:        if priority >= PRIORITY_EMERGENCY:
.\portfolio\alert_budget.py:55:    def remaining_budget(self) -> int:
.\portfolio\alert_budget.py:60:    def buffer_size(self) -> int:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"run_safe\\(|popen_in_job\\(|close_job\\(\" ." in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 502ms:
.\portfolio\subprocess_utils.py:4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
.\portfolio\subprocess_utils.py:6:- popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
.\portfolio\subprocess_utils.py:21:def run_safe(cmd, **kwargs):
.\portfolio\subprocess_utils.py:156:def popen_in_job(cmd, **kwargs):
.\portfolio\subprocess_utils.py:166:    Caller must call close_job(job_handle) when explicitly stopping the child.
.\portfolio\subprocess_utils.py:182:def close_job(job_handle):

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"api\\.telegram\\.org/bot\\{token\\}|fetch_with_retry\\(\" portfolio" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 625ms:
portfolio\telegram_poller.py:129:        r = fetch_with_retry(
portfolio\telegram_poller.py:374:            r = fetch_with_retry(
portfolio\telegram_notifications.py:54:    r = fetch_with_retry(
portfolio\telegram_notifications.py:55:        f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\telegram_notifications.py:74:            r2 = fetch_with_retry(
portfolio\telegram_notifications.py:75:                f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\message_store.py:137:    r = fetch_with_retry(
portfolio\message_store.py:138:        f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\message_store.py:160:            r2 = fetch_with_retry(
portfolio\message_store.py:161:                f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\http_retry.py:17:def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
portfolio\http_retry.py:79:    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
portfolio\http_retry.py:81:    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/telegram_poller.py' 118 145
Show-Lines 'portfolio/telegram_poller.py' 362 385" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 620ms:
=== portfolio/telegram_poller.py:118-145 ===
  118:                     self._handle_update(update)
  119:             except Exception as e:
  120:                 logger.warning("Poller error: %s", e)
  121:             time.sleep(5)
  122: 
  123:     def _get_updates(self):
  124:         """Fetch new updates from Telegram."""
  125:         params = {"timeout": 3, "allowed_updates": ["message"]}
  126:         if self.offset:
  127:             params["offset"] = self.offset
  128: 
  129:         r = fetch_with_retry(
  130:             f"https://api.telegram.org/bot{self.token}/getUpdates",
  131:             params=params,
  132:             timeout=10,
  133:         )
  134:         if r is None or not r.ok:
  135:             return []
  136: 
  137:         data = r.json()
  138:         if not data.get("ok"):
  139:             return []
  140: 
  141:         return data.get("result", [])
  142: 
  143:     # Drop reasons that represent a *settled* outcome — the message was
  144:     # examined and intentionally not acted on (stale, empty, unrecognized,
  145:     # or no message body / wrong chat). Re-fetching these on a restart
=== portfolio/telegram_poller.py:362-385 ===
  362: 
  363:         # Update in-memory config
  364:         if "notification" not in self.config:
  365:             self.config["notification"] = {}
  366:         self.config["notification"]["mode"] = mode_arg
  367: 
  368:         logger.info("Notification mode changed to: %s", mode_arg)
  369:         return f"Notification mode set to *{mode_arg}*"
  370: 
  371:     def _send_reply(self, text):
  372:         """Send a reply to the user."""
  373:         try:
  374:             r = fetch_with_retry(
  375:                 f"https://api.telegram.org/bot{self.token}/sendMessage",
  376:                 method="POST",
  377:                 json_body={
  378:                     "chat_id": self.chat_id,
  379:                     "text": text,
  380:                     "parse_mode": "Markdown",
  381:                 },
  382:                 timeout=30,
  383:             )
  384:             if r is not None and not r.ok:
  385:                 logger.warning("Poller reply error: %s %s", r.status_code, r.text[:200])

codex
I’ve finished the verdict calls. I’m pulling three extra snippets now for issues Claude missed: a real Telegram token leak in retry logs, uncoordinated log rotation vs atomic appenders, and a broken dashboard import graph in this worktree.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/http_retry.py' 39 69
Show-Lines 'portfolio/message_store.py' 137 165
Show-Lines 'portfolio/telegram_notifications.py' 54 78
Show-Lines 'portfolio/telegram_poller.py' 129 133" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'portfolio/file_utils.py' 227 260
Show-Lines 'portfolio/log_rotation.py' 236 327" in Q:\finance-analyzer\.worktrees\adv-infrastructure
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$ErrorActionPreference='"'Stop'
function Show-Lines("'$path, $start, $end){
  Write-Output "=== ${path}:$start-$end ==="
  $i=0
  Get-Content (Resolve-Path $path) | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }
}
Show-Lines '"'dashboard/app.py' 728 742
Show-Lines 'dashboard/app.py' 2015 2027
Write-Output '=== git ls-tree dashboard ==='
git -c safe.directory=Q:/finance-analyzer/.worktrees/adv-infrastructure ls-tree -r --name-only HEAD dashboard" in Q:\finance-analyzer\.worktrees\adv-infrastructure
 succeeded in 597ms:
=== portfolio/http_retry.py:39-69 ===
   39:             if attempt < retries:
   40:                 wait = backoff * (backoff_factor ** attempt)
   41:                 jitter = random.uniform(0, wait * 0.1)
   42:                 wait += jitter
   43:                 # H26/HTTP1: Honour Telegram retry_after parameter for 429 responses.
   44:                 if resp.status_code == 429:
   45:                     try:
   46:                         retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
   47:                     except Exception:
   48:                         retry_after = wait
   49:                     wait = retry_after
   50:                 logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
   51:                                resp.status_code, url, attempt + 1, retries, wait)
   52:                 time.sleep(wait)
   53:             else:
   54:                 logger.error("HTTP %s from %s after %d retries",
   55:                              resp.status_code, url, retries)
   56:                 return None
   57: 
   58:         except (requests.ConnectionError, requests.Timeout) as e:
   59:             if attempt < retries:
   60:                 wait = backoff * (backoff_factor ** attempt)
   61:                 jitter = random.uniform(0, wait * 0.1)
   62:                 wait += jitter
   63:                 logger.warning("%s from %s, retry %d/%d in %.1fs",
   64:                                e.__class__.__name__, url, attempt + 1, retries, wait)
   65:                 time.sleep(wait)
   66:             else:
   67:                 logger.error("Request failed after %d retries: %s - %s",
   68:                              retries, url, e)
   69:                 return None
=== portfolio/message_store.py:137-165 ===
  137:     r = fetch_with_retry(
  138:         f"https://api.telegram.org/bot{token}/sendMessage",
  139:         method="POST",
  140:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
  141:         timeout=30,
  142:     )
  143:     if r is None:
  144:         return False
  145:     if r.ok:
  146:         return True
  147: 
  148:     # Markdown parse failure (HTTP 400) — retry without parse_mode
  149:     if r.status_code == 400:
  150:         err_desc = ""
  151:         try:
  152:             err_desc = r.json().get("description", "")
  153:         except Exception as e:
  154:             logger.debug("Failed to parse Telegram error response: %s", e)
  155:         if any(kw in err_desc.lower() for kw in ("parse", "markdown", "entity")):
  156:             logger.warning(
  157:                 "Telegram Markdown parse failed (%s), resending without formatting",
  158:                 err_desc,
  159:             )
  160:             r2 = fetch_with_retry(
  161:                 f"https://api.telegram.org/bot{token}/sendMessage",
  162:                 method="POST",
  163:                 json_body={"chat_id": chat_id, "text": msg},
  164:                 timeout=30,
  165:             )
=== portfolio/telegram_notifications.py:54-78 ===
   54:     r = fetch_with_retry(
   55:         f"https://api.telegram.org/bot{token}/sendMessage",
   56:         method="POST",
   57:         json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
   58:         timeout=30,
   59:     )
   60:     if r is None:
   61:         return False
   62:     if r.ok:
   63:         return True
   64:     # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
   65:     # still arrives (unformatted) rather than being silently lost.
   66:     if r.status_code == 400:
   67:         err_desc = ""
   68:         try:
   69:             err_desc = r.json().get("description", "")
   70:         except Exception:
   71:             logger.debug("Failed to parse Telegram error response", exc_info=True)
   72:         if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
   73:             logger.warning("Telegram Markdown parse failed (%s), resending without formatting", err_desc)
   74:             r2 = fetch_with_retry(
   75:                 f"https://api.telegram.org/bot{token}/sendMessage",
   76:                 method="POST",
   77:                 json_body={"chat_id": chat_id, "text": msg},
   78:                 timeout=30,
=== portfolio/telegram_poller.py:129-133 ===
  129:         r = fetch_with_retry(
  130:             f"https://api.telegram.org/bot{self.token}/getUpdates",
  131:             params=params,
  132:             timeout=10,
  133:         )

 succeeded in 623ms:
=== portfolio/file_utils.py:227-260 ===
  227:     path = Path(path)
  228:     path.parent.mkdir(parents=True, exist_ok=True)
  229:     data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
  230: 
  231:     # Sidecar lockfile — always non-empty so locking never fails on
  232:     # size-0 targets. Pre-create if missing; single byte is enough.
  233:     lock_path = path.parent / f".{path.name}.lock"
  234:     if not lock_path.exists():
  235:         try:
  236:             with open(lock_path, "ab") as lf:
  237:                 if lf.tell() == 0:
  238:                     lf.write(b"\0")
  239:         except OSError:
  240:             pass  # best-effort; lock open below will retry
  241: 
  242:     with open(lock_path, "rb+") as lock_f:
  243:         lfd = lock_f.fileno()
  244:         win_locked = False
  245:         try:
  246:             if _msvcrt is not None:
  247:                 os.lseek(lfd, 0, os.SEEK_SET)
  248:                 _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
  249:                 win_locked = True
  250:             elif _fcntl is not None:
  251:                 _fcntl.flock(lfd, _fcntl.LOCK_EX)
  252:             with open(path, "ab") as f:
  253:                 f.write(data)
  254:                 f.flush()
  255:                 os.fsync(f.fileno())
  256:         finally:
  257:             if win_locked and _msvcrt is not None:
  258:                 try:
  259:                     os.lseek(lfd, 0, os.SEEK_SET)
  260:                     _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
=== portfolio/log_rotation.py:236-327 ===
  236:     # Read all lines and classify as keep vs archive
  237:     keep_lines = []
  238:     archive_buckets = {}  # "YYYY-MM" -> list of raw lines
  239:     parse_failures = 0
  240:     total_lines = 0
  241: 
  242:     with open(filepath, encoding="utf-8") as f:
  243:         for line in f:
  244:             line = line.rstrip("\n")
  245:             if not line.strip():
  246:                 continue
  247:             total_lines += 1
  248:             try:
  249:                 entry = json.loads(line)
  250:                 ts = _parse_ts(entry.get(ts_field))
  251:                 if ts is None:
  252:                     # Can't determine age -- keep the entry to be safe
  253:                     keep_lines.append(line)
  254:                     parse_failures += 1
  255:                 elif ts >= cutoff:
  256:                     keep_lines.append(line)
  257:                 else:
  258:                     # Archive this entry, grouped by month
  259:                     month_key = ts.strftime("%Y-%m")
  260:                     archive_buckets.setdefault(month_key, []).append(line)
  261:             except json.JSONDecodeError:
  262:                 # Malformed line -- keep it to avoid data loss
  263:                 keep_lines.append(line)
  264:                 parse_failures += 1
  265: 
  266:     archived_count = sum(len(v) for v in archive_buckets.values())
  267:     result = {
  268:         "file": filename,
  269:         "size_mb": round(size_mb, 2),
  270:         "total_lines": total_lines,
  271:         "kept": len(keep_lines),
  272:         "archived": archived_count,
  273:         "archive_months": sorted(archive_buckets.keys()),
  274:         "parse_failures": parse_failures,
  275:     }
  276: 
  277:     if archived_count == 0:
  278:         result["status"] = "nothing_to_archive"
  279:         return result
  280: 
  281:     if dry_run:
  282:         result["status"] = "dry_run"
  283:         return result
  284: 
  285:     _ensure_archive_dir()
  286: 
  287:     # Write archived entries to monthly files
  288:     stem = pathlib.Path(filename).stem  # e.g. "signal_log"
  289:     suffix = pathlib.Path(filename).suffix  # e.g. ".jsonl"
  290: 
  291:     for month_key, lines in sorted(archive_buckets.items()):
  292:         archive_name = f"{stem}.{month_key}{suffix}"
  293:         archive_path = ARCHIVE_DIR / archive_name
  294:         gz_path = ARCHIVE_DIR / f"{archive_name}.gz"
  295: 
  296:         # Append to existing archive for this month (may already have entries
  297:         # from a previous rotation)
  298:         if gz_path.exists() and policy.get("compress", True):
  299:             # Decompress existing, append, re-compress
  300:             existing_lines = []
  301:             with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
  302:                 for existing_line in gf:
  303:                     existing_line = existing_line.rstrip("\n")
  304:                     if existing_line.strip():
  305:                         existing_lines.append(existing_line)
  306:             all_lines = existing_lines + lines
  307:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  308:                 for line in all_lines:
  309:                     gf.write(line + "\n")
  310:         elif policy.get("compress", True):
  311:             with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
  312:                 for line in lines:
  313:                     gf.write(line + "\n")
  314:         else:
  315:             with open(archive_path, "a", encoding="utf-8") as af:
  316:                 for line in lines:
  317:                     af.write(line + "\n")
  318: 
  319:     # Rewrite the original file with only kept lines
  320:     tmp_path = filepath.with_suffix(".tmp")
  321:     with open(tmp_path, "w", encoding="utf-8") as f:
  322:         for line in keep_lines:
  323:             f.write(line + "\n")
  324: 
  325:     # Atomic-ish replace: remove original, rename tmp
  326:     # On Windows, os.replace is atomic within the same volume
  327:     os.replace(tmp_path, filepath)

 succeeded in 662ms:
=== dashboard/app.py:728-742 ===
  728: # Auth + cookie machinery moved to dashboard/auth.py on 2026-05-02 to break
  729: # the circular import with dashboard/house_blueprint.py. We re-import here
  730: # so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
  731: # inside this module's body, and so any lingering external code that does
  732: # `from dashboard.app import require_auth` still resolves. Tests should
  733: # patch `dashboard.auth.*` directly — patches on `dashboard.app.*` will not
  734: # take effect since require_auth resolves names via dashboard.auth's
  735: # module globals.
  736: from dashboard.auth import (  # noqa: E402
  737:     COOKIE_MAX_AGE,
  738:     COOKIE_NAME,
  739:     _get_config as _auth_get_config,  # noqa: F401 — kept for compat
  740:     _get_dashboard_token,
  741:     _refresh_cookie,
  742:     require_auth,
=== dashboard/app.py:2015-2027 ===
 2015: # Blueprint: /house — read-only viewer over the househunting project
 2016: # (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
 2017: # auth via dashboard.auth.require_auth. Path roots come from
 2018: # config.json[house_root]. See dashboard/house_blueprint.py for routes.
 2019: #
 2020: # House_blueprint imports `_get_config` and `require_auth` from
 2021: # dashboard.auth (NOT dashboard.app), so importing it here at module-init
 2022: # time no longer causes a circular import — auth.py has no back-reference
 2023: # to app.py. The sys.modules alias hack added 2026-05-02 has been removed.
 2024: # ---------------------------------------------------------------------------
 2025: from dashboard.house_blueprint import bp as _house_bp  # noqa: E402
 2026: 
 2027: app.register_blueprint(_house_bp)
=== git ls-tree dashboard ===
dashboard/app.py

codex
## Verdicts

- [P0] `fetch_with_retry()` retries all 4xx statuses — portfolio/http_retry.py:36
  Verdict: FALSE-POSITIVE
  Reason: Line 36 returns immediately unless the status is in `RETRYABLE_STATUS`, and that set at line 14 is only `{429, 500, 502, 503, 504}`.

- [P0] `POST /api/validate-portfolio` lacks CSRF validation — dashboard/app.py:1039
  Verdict: FALSE-POSITIVE
  Reason: The handler only reads `request.get_json(silent=True)` and returns validation output at lines 1047-1059, while the app also sets `Access-Control-Allow-Credentials=false` at lines 52-60, so the claimed state-changing CSRF surface is not present.

- [P1] Rate limiter stampedes when `wait_time=0` — portfolio/shared_state.py:276
  Verdict: FALSE-POSITIVE
  Reason: The code computes wait and reserves `last_call` under `self._lock` at lines 269-276, so once one thread passes, the next thread sees the reserved timestamp and waits.

- [P1] `load_jsonl_tail()` can split UTF-8 on fake newline bytes — portfolio/file_utils.py:173-180
  Verdict: FALSE-POSITIVE
  Reason: UTF-8 multibyte sequences cannot contain byte `0x0A`, and the function already decodes with `errors="replace"` at line 177 and drops the partial first line at lines 182-183.

- [P1] Stale-lock breaker can delete a new live GPU lock — portfolio/gpu_gate.py:130-142
  Verdict: CONFIRMED
  Reason: `_try_break_stale_lock()` decides on the current path state at lines 126-135 and then blindly unlinks it at line 136, so a concurrent breaker/acquirer can swap in a fresh lock before the unlink.

- [P1] Windows unlock path skips `seek(0)` — portfolio/process_lock.py:62-73
  Verdict: FALSE-POSITIVE
  Reason: `_unlock_file()` explicitly does `fh.seek(0)` before `msvcrt.locking(..., LK_UNLCK, 1)` at lines 69-73.

- [P1] Telegram parse-error logs leak token/chat_id payloads — portfolio/telegram_notifications.py:52, portfolio/message_store.py:119
  Verdict: FALSE-POSITIVE
  Reason: The 400 parse-error branches only log Telegram's `description` string at `telegram_notifications.py:66-79` and `message_store.py:149-165`, not the request payload, token, or `chat_id`.

- [P1] Timeout kill can orphan grandchildren on Windows — portfolio/subprocess_utils.py:140-143
  Verdict: PARTIAL
  Reason: On the normal path the process is put in a Job Object and the handle is closed in `finally` at lines 131-153, so assigned descendants do die with the job. (Adjustment: the real gap is lines 131-137 tolerating `AssignProcessToJobObject` failure and continuing unprotected, not `proc.kill()` itself.)

- [P1] Log rotation replaces without `fsync()` — portfolio/log_rotation.py:326-327
  Verdict: PARTIAL
  Reason: The temp file is written and replaced at lines 321-327 without an explicit `flush()/fsync()`, so crash durability is not guaranteed. (Adjustment: `os.replace()` still makes the switch atomic in normal operation, so this is a crash-window durability issue, not a routine torn-file bug.)

- [P1] Concurrent `save_beliefs()` calls clobber each other — portfolio/prophecy.py:70-73
  Verdict: PARTIAL
  Reason: The mutators all do unsynchronized `load_beliefs()` then `save_beliefs()` cycles at lines 85-101, 115-123, 134-140, 180-195, and 210-267, so concurrent writers can lose updates. (Adjustment: the code shows a generic read-modify-write race, not the specific "Layer 1 + Layer 2 same cycle" path Claude asserted.)

- [P2] Cache exception handler mutates `_tool_cache` outside the lock — portfolio/shared_state.py:94-123
  Verdict: FALSE-POSITIVE
  Reason: The cleanup and timestamp mutation are inside `with _cache_lock:` at lines 111-123.

- [P2] Adaptive tail growth is unbounded on bad JSONL — dashboard/app.py:155-168
  Verdict: FALSE-POSITIVE
  Reason: The loop is explicitly capped by `max_retry_bytes = 64 * 1024 * 1024` and stops once `capped >= max_retry_bytes` at lines 151-160.

- [P2] Empty deque leaves a stale `last_call` and suppresses alerts — portfolio/alert_budget.py:42-44
  Verdict: FALSE-POSITIVE
  Reason: This limiter has no `last_call` state at all; it only prunes and counts `self._sent_timestamps` at lines 27-44.

- [P2] PowerShell JSON parsing can crash callers — portfolio/subprocess_utils.py:289-299
  Verdict: FALSE-POSITIVE
  Reason: `kill_orphaned_llama()` already catches `json.JSONDecodeError` at lines 289-293 and returns `0` instead of propagating.

- [P2] Reader races with log rotation and skips journal lines — portfolio/journal.py:28-40
  Verdict: FALSE-POSITIVE
  Reason: `load_recent()` opens `JOURNAL_FILE` once and iterates that handle at lines 28-39, while rotation rewrites via a temp file plus `os.replace()` at `log_rotation.py:320-327`, so the cited buffered-read/move race is not what this code does.

- [P3] Integer parsing truncates floats like `200.5 -> 200` — dashboard/app.py:175-181
  Verdict: FALSE-POSITIVE
  Reason: `int(request.args.get(...))` on `"200.5"` raises `ValueError`, which the function catches at lines 177-180 and replaces with `default`.

- [P3] `ensure_ascii=True` is a bug in JSON writes — portfolio/file_utils.py:56
  Verdict: FALSE-POSITIVE
  Reason: `atomic_write_json()` exposes `ensure_ascii` as a configurable parameter at lines 45-56, so this is a format choice rather than a correctness defect, and `atomic_append_jsonl()` already uses `ensure_ascii=False` at line 229.

- [P3] Stuck-key eviction warning fires on normal batch flushes — portfolio/shared_state.py:68-75
  Verdict: FALSE-POSITIVE
  Reason: The warning only fires after `_LOADING_TIMEOUT` at lines 68-75, and successful resolutions clear timestamps at lines 102 and 230, so the claimed "normal 2-min flush" behavior is not shown in this tree.

- [P3] GoldDigger payload duplicates fields because of shallow copy — dashboard/app.py:226-242
  Verdict: FALSE-POSITIVE
  Reason: The normalizer intentionally writes synchronized aliases such as `shares`/`quantity` and `tp_price`/`take_profit_price` at lines 241-249.

- [P3] Mojibake cleanup is incomplete — portfolio/message_store.py:37-49
  Verdict: FALSE-POSITIVE
  Reason: The cited block is a bounded best-effort sanitizer used by `sanitize_message_text()` at lines 55-84, and Claude did not identify a concrete incorrect behavior at those lines.

- [P3] Hardcoded `Q:/models` means non-Windows installs never lock — portfolio/gpu_gate.py:33
  Verdict: PARTIAL
  Reason: The hardcoded path is real at lines 33-34, but `gpu_gate()` does not quietly skip locking on a missing directory; `os.open(..., O_CREAT | O_EXCL | O_WRONLY)` at line 221 will raise instead. (Adjustment: portability bug exists, but the actual failure mode is an acquire-time exception, not silent lock bypass.)

- [P3] Division-by-zero guard misses missing-field cases — portfolio/prophecy.py:323
  Verdict: FALSE-POSITIVE
  Reason: The progress calculation only runs when `prices_usd` has the ticker and both `target_price` and `entry_price` are present/truthy at line 317, and line 322 separately guards `target != entry_price`.

## New findings (you, not Claude)

- [P2] Telegram bot token is logged verbatim on retries and terminal HTTP failures — portfolio/http_retry.py:50
  `fetch_with_retry()` logs the full `url` on retry and failure at lines 50-68, and Telegram callers build that URL as `https://api.telegram.org/bot{token}/...` in `message_store.py:138`, `telegram_notifications.py:55`, and `telegram_poller.py:130`; any 429/5xx/timeout therefore writes the bot token to logs.

- [P1] JSONL rotation can silently drop concurrent appends — portfolio/log_rotation.py:242
  `atomic_append_jsonl()` serializes writers with a sidecar lock at `file_utils.py:233-260`, but `rotate_jsonl()` reads the live file, builds `keep_lines`, and `os.replace()`s a temp copy at lines 242-327 without taking that lock; any append that lands after the initial read and before the replace is lost.

- [P1] The dashboard import graph is broken in this worktree — dashboard/app.py:736
  `dashboard/app.py` has module-level imports of `dashboard.auth` at line 736 and `dashboard.house_blueprint` at line 2025, but the tracked `dashboard/` tree in this worktree contains only `dashboard/app.py`, so once import reaches those lines the dashboard cannot start.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 17
- New from you: 3
tokens used
214,465
## Verdicts

- [P0] `fetch_with_retry()` retries all 4xx statuses — portfolio/http_retry.py:36
  Verdict: FALSE-POSITIVE
  Reason: Line 36 returns immediately unless the status is in `RETRYABLE_STATUS`, and that set at line 14 is only `{429, 500, 502, 503, 504}`.

- [P0] `POST /api/validate-portfolio` lacks CSRF validation — dashboard/app.py:1039
  Verdict: FALSE-POSITIVE
  Reason: The handler only reads `request.get_json(silent=True)` and returns validation output at lines 1047-1059, while the app also sets `Access-Control-Allow-Credentials=false` at lines 52-60, so the claimed state-changing CSRF surface is not present.

- [P1] Rate limiter stampedes when `wait_time=0` — portfolio/shared_state.py:276
  Verdict: FALSE-POSITIVE
  Reason: The code computes wait and reserves `last_call` under `self._lock` at lines 269-276, so once one thread passes, the next thread sees the reserved timestamp and waits.

- [P1] `load_jsonl_tail()` can split UTF-8 on fake newline bytes — portfolio/file_utils.py:173-180
  Verdict: FALSE-POSITIVE
  Reason: UTF-8 multibyte sequences cannot contain byte `0x0A`, and the function already decodes with `errors="replace"` at line 177 and drops the partial first line at lines 182-183.

- [P1] Stale-lock breaker can delete a new live GPU lock — portfolio/gpu_gate.py:130-142
  Verdict: CONFIRMED
  Reason: `_try_break_stale_lock()` decides on the current path state at lines 126-135 and then blindly unlinks it at line 136, so a concurrent breaker/acquirer can swap in a fresh lock before the unlink.

- [P1] Windows unlock path skips `seek(0)` — portfolio/process_lock.py:62-73
  Verdict: FALSE-POSITIVE
  Reason: `_unlock_file()` explicitly does `fh.seek(0)` before `msvcrt.locking(..., LK_UNLCK, 1)` at lines 69-73.

- [P1] Telegram parse-error logs leak token/chat_id payloads — portfolio/telegram_notifications.py:52, portfolio/message_store.py:119
  Verdict: FALSE-POSITIVE
  Reason: The 400 parse-error branches only log Telegram's `description` string at `telegram_notifications.py:66-79` and `message_store.py:149-165`, not the request payload, token, or `chat_id`.

- [P1] Timeout kill can orphan grandchildren on Windows — portfolio/subprocess_utils.py:140-143
  Verdict: PARTIAL
  Reason: On the normal path the process is put in a Job Object and the handle is closed in `finally` at lines 131-153, so assigned descendants do die with the job. (Adjustment: the real gap is lines 131-137 tolerating `AssignProcessToJobObject` failure and continuing unprotected, not `proc.kill()` itself.)

- [P1] Log rotation replaces without `fsync()` — portfolio/log_rotation.py:326-327
  Verdict: PARTIAL
  Reason: The temp file is written and replaced at lines 321-327 without an explicit `flush()/fsync()`, so crash durability is not guaranteed. (Adjustment: `os.replace()` still makes the switch atomic in normal operation, so this is a crash-window durability issue, not a routine torn-file bug.)

- [P1] Concurrent `save_beliefs()` calls clobber each other — portfolio/prophecy.py:70-73
  Verdict: PARTIAL
  Reason: The mutators all do unsynchronized `load_beliefs()` then `save_beliefs()` cycles at lines 85-101, 115-123, 134-140, 180-195, and 210-267, so concurrent writers can lose updates. (Adjustment: the code shows a generic read-modify-write race, not the specific "Layer 1 + Layer 2 same cycle" path Claude asserted.)

- [P2] Cache exception handler mutates `_tool_cache` outside the lock — portfolio/shared_state.py:94-123
  Verdict: FALSE-POSITIVE
  Reason: The cleanup and timestamp mutation are inside `with _cache_lock:` at lines 111-123.

- [P2] Adaptive tail growth is unbounded on bad JSONL — dashboard/app.py:155-168
  Verdict: FALSE-POSITIVE
  Reason: The loop is explicitly capped by `max_retry_bytes = 64 * 1024 * 1024` and stops once `capped >= max_retry_bytes` at lines 151-160.

- [P2] Empty deque leaves a stale `last_call` and suppresses alerts — portfolio/alert_budget.py:42-44
  Verdict: FALSE-POSITIVE
  Reason: This limiter has no `last_call` state at all; it only prunes and counts `self._sent_timestamps` at lines 27-44.

- [P2] PowerShell JSON parsing can crash callers — portfolio/subprocess_utils.py:289-299
  Verdict: FALSE-POSITIVE
  Reason: `kill_orphaned_llama()` already catches `json.JSONDecodeError` at lines 289-293 and returns `0` instead of propagating.

- [P2] Reader races with log rotation and skips journal lines — portfolio/journal.py:28-40
  Verdict: FALSE-POSITIVE
  Reason: `load_recent()` opens `JOURNAL_FILE` once and iterates that handle at lines 28-39, while rotation rewrites via a temp file plus `os.replace()` at `log_rotation.py:320-327`, so the cited buffered-read/move race is not what this code does.

- [P3] Integer parsing truncates floats like `200.5 -> 200` — dashboard/app.py:175-181
  Verdict: FALSE-POSITIVE
  Reason: `int(request.args.get(...))` on `"200.5"` raises `ValueError`, which the function catches at lines 177-180 and replaces with `default`.

- [P3] `ensure_ascii=True` is a bug in JSON writes — portfolio/file_utils.py:56
  Verdict: FALSE-POSITIVE
  Reason: `atomic_write_json()` exposes `ensure_ascii` as a configurable parameter at lines 45-56, so this is a format choice rather than a correctness defect, and `atomic_append_jsonl()` already uses `ensure_ascii=False` at line 229.

- [P3] Stuck-key eviction warning fires on normal batch flushes — portfolio/shared_state.py:68-75
  Verdict: FALSE-POSITIVE
  Reason: The warning only fires after `_LOADING_TIMEOUT` at lines 68-75, and successful resolutions clear timestamps at lines 102 and 230, so the claimed "normal 2-min flush" behavior is not shown in this tree.

- [P3] GoldDigger payload duplicates fields because of shallow copy — dashboard/app.py:226-242
  Verdict: FALSE-POSITIVE
  Reason: The normalizer intentionally writes synchronized aliases such as `shares`/`quantity` and `tp_price`/`take_profit_price` at lines 241-249.

- [P3] Mojibake cleanup is incomplete — portfolio/message_store.py:37-49
  Verdict: FALSE-POSITIVE
  Reason: The cited block is a bounded best-effort sanitizer used by `sanitize_message_text()` at lines 55-84, and Claude did not identify a concrete incorrect behavior at those lines.

- [P3] Hardcoded `Q:/models` means non-Windows installs never lock — portfolio/gpu_gate.py:33
  Verdict: PARTIAL
  Reason: The hardcoded path is real at lines 33-34, but `gpu_gate()` does not quietly skip locking on a missing directory; `os.open(..., O_CREAT | O_EXCL | O_WRONLY)` at line 221 will raise instead. (Adjustment: portability bug exists, but the actual failure mode is an acquire-time exception, not silent lock bypass.)

- [P3] Division-by-zero guard misses missing-field cases — portfolio/prophecy.py:323
  Verdict: FALSE-POSITIVE
  Reason: The progress calculation only runs when `prices_usd` has the ticker and both `target_price` and `entry_price` are present/truthy at line 317, and line 322 separately guards `target != entry_price`.

## New findings (you, not Claude)

- [P2] Telegram bot token is logged verbatim on retries and terminal HTTP failures — portfolio/http_retry.py:50
  `fetch_with_retry()` logs the full `url` on retry and failure at lines 50-68, and Telegram callers build that URL as `https://api.telegram.org/bot{token}/...` in `message_store.py:138`, `telegram_notifications.py:55`, and `telegram_poller.py:130`; any 429/5xx/timeout therefore writes the bot token to logs.

- [P1] JSONL rotation can silently drop concurrent appends — portfolio/log_rotation.py:242
  `atomic_append_jsonl()` serializes writers with a sidecar lock at `file_utils.py:233-260`, but `rotate_jsonl()` reads the live file, builds `keep_lines`, and `os.replace()`s a temp copy at lines 242-327 without taking that lock; any append that lands after the initial read and before the replace is lost.

- [P1] The dashboard import graph is broken in this worktree — dashboard/app.py:736
  `dashboard/app.py` has module-level imports of `dashboard.auth` at line 736 and `dashboard.house_blueprint` at line 2025, but the tracked `dashboard/` tree in this worktree contains only `dashboard/app.py`, so once import reaches those lines the dashboard cannot start.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 17
- New from you: 3
