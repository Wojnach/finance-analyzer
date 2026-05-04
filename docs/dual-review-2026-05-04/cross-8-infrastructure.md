# Cross-critique — infrastructure

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `telegram_poller.py:361` — `/mode` command writes via `atomic_write_json()` to `config_path`. **`config.json` is the external symlink** (CLAUDE.md "NEVER commit config.json — symlink"). `atomic_write_json` does `os.replace(tmp, target)` which **replaces the symlink with a regular file**. After first `/mode` command, external secrets/config updates stop propagating. | **Critical — Claude knew the project rule but didn't trace the `/mode` write path.** Codex caught the rare interaction between symlink target identity and `os.replace` semantics. The rule says "never commit"; it should also say "never overwrite via atomic_write." |
| `gpu_gate.py:33-34` — `Path("Q:/models")` is just a relative path on Linux. First `gpu_gate()` call from `ministral_signal`/`qwen3_signal` tries to create `./Q:/models/.gpu_lock` → `FileNotFoundError`. Surrounding LLM code already supports `/home/deck/...` paths. | Claude assumed Windows-only environment (consistent with CLAUDE.md) and didn't probe Linux portability. Codex caught the cross-platform inconsistency. |
| `llama_server.py:179-180` — Malformed `llama_server.pid` file: `int(content.split(":")[0])` raises ValueError → handler tries to log `pid` which was never assigned → `UnboundLocalError`. Crashes `_stop_server()`. | Claude reviewed `llama_server.py` for race conditions but didn't probe malformed-state recovery. Edge case. |
| `vector_memory.py:41-42` — `_get_collection()` returns cached `_collection` without checking `collection_name`. Test/per-strategy namespaces silently reuse the first collection → embeddings mixed across collections. | Claude reviewed `vector_memory.py` for raw open() (P1) but missed the singleton-vs-parameter bug. |
| `log_rotation.py:319-327` — Race between `rotate_jsonl()` reading old file and `atomic_append_jsonl()` appender. Lines appended after rotate finishes reading but before `os.replace` lands in **the old file** and are silently dropped from the new one. **Live records lost.** | Partial overlap with Claude's findings on the same function (P0 no fsync, P1 gz truncate-in-place). Codex's angle is the data-loss race; Claude's was durability + gz truncate. **All three are real and complementary.** |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `log_rotation.py:320-327` — `rotate_jsonl` writes temp file without `f.flush(); os.fsync()` before `os.replace`. Power-loss can leave torn rotate. No cleanup of `.tmp` if `os.replace` raises. **`signal_log.jsonl` 68MB rotation can be unrecoverable.** | Codex flagged the race (above) but didn't audit the durability/cleanup. Compare to `atomic_write_json` which does fsync + `BaseException` cleanup. |
| `local_llm_report.py:37` — `path.read_text(encoding="utf-8").splitlines()` violates atomic I/O rule on actively-written `forecast_predictions.jsonl`. Truncated last line silently dropped. | Codex didn't audit local_llm_report. Same rule violation class as Codex's `vector_memory.py` chroma collection bug. |
| `claude_gate.py:271-279` — `_count_today_invocations()` scans entire JSONL on every call, outside `_invoke_lock`. Latency bomb after months. | Codex didn't probe the scaling behavior. |
| `vector_memory.py:272-281` — Raw `open()` on `JOURNAL_FILE` violates atomic I/O rule. ChromaDB embeddings silently miss truncated last entries. | Different from Codex's collection bug in same module. Both real. |
| `gpu_gate.py:98-102` — `_write_lock` function defined but never called (dead code). Future dev calling it would bypass atomic-create. | Codex caught a different gpu_gate bug (Linux path). Both real. |
| `log_rotation.py:298-309` — `rotate_jsonl` decompresses + recompresses gz archive in-place with truncate. Crash mid-write **permanently destroys historical archive**. | Codex flagged the live-records-dropped race; Claude flagged the historical-archive-destroyed crash. **Both terrifying, both real, both in the same function.** |

## Disagreements

### `log_rotation.py:319-327` — Same function, three independent bugs (no actual disagreement)
- **Codex** P2: race with appenders (live record loss).
- **Claude** P0: no fsync, no cleanup on replace failure.
- **Claude** P1: gz truncate-in-place (historical destruction on crash).

All three are real. Reconciled to **P0 (Claude's durability)** + **P1 (Codex's race)** + **P1 (Claude's gz truncate)**. The function needs a complete rewrite using the `atomic_write_json` pattern + an append lock during rotation.

### `vector_memory.py` — Different bugs
- **Codex** P3: `_get_collection()` returns cached collection regardless of `collection_name`.
- **Claude** P1: raw `open()` on `JOURNAL_FILE`.

Different root causes, both real.

## What both missed (likely)

- **`telegram_notifications.py` Markdown escaping** — neither reviewer flagged anything. User text containing `_` or `*` in a ticker name breaks Telegram parsing.
- **`telegram_poller.py` command authorization** — neither asked whether `/mode` requires sender ID validation. Codex flagged the symlink bug but didn't ask: "could anyone with the bot token send /mode?"
- **`process_lock.py` cross-process semantics on the same physical machine but different worktrees** — both worktrees would compete for the same lock file path. Neither reviewer asked.
- **`subprocess_utils.py`** — neither flagged. Wrappers around subprocess often hide signal-propagation issues.
- **`memory_consolidation.py`** — neither flagged. Consolidation can rewrite memory files atomically but if interrupted, lose data.

## Reconciled verdict

**P0 (must fix — durable corruption + project-rule violation):**
1. **(Codex)** `telegram_poller.py:361` `/mode` writes through symlink → first `/mode` breaks the external config.json link. **Direct project-rule violation; first user `/mode` command silently severs config sync.**
2. **(Claude)** `log_rotation.py:320-327` no fsync on `rotate_jsonl` + no `.tmp` cleanup → `signal_log.jsonl` rotation can be unrecoverable on power loss.
3. **(Claude)** `local_llm_report.py:37` raw `path.read_text()` on actively-written JSONL → truncated lines silently dropped → corrupted accuracy stats.

**P1:**
4. (Claude) `log_rotation.py:298-309` gz archive decompress+truncate in-place → crash destroys historical archive.
5. (Codex) `log_rotation.py:319-327` race with appenders → live records dropped during rotation.
6. (Codex) `gpu_gate.py:33-34` `Path("Q:/models")` Linux portability → silent FileNotFoundError on non-Windows.
7. (Claude) `claude_gate.py:271-279` full-file scan + outside lock.
8. (Claude) `vector_memory.py:272-281` raw `open()` on journal.
9. (Codex) `llama_server.py:179-180` `UnboundLocalError` on malformed PID file → no graceful recovery.
10. (Claude) `gpu_gate.py:98-102` dead `_write_lock` function — maintenance trap.

**P2:**
11. (Claude) `config_validator.py:58-60` raw `open()` on config.json.
12. (Claude) `shared_state.py:104-108` `KeyboardInterrupt` cleanup misses `_loading_timestamps.pop`.
13. (Claude) `http_retry.py:44-49` Telegram 429 `retry_after` no cap → 24h thread block possible.
14. (Codex) `vector_memory.py:41-42` `_get_collection()` ignores `collection_name` after first call.
15. (Claude) `ministral_trader.py:45-46` fixed-path temp file race.
