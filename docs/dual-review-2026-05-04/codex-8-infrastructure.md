# Codex Review — 8-infrastructure

## Summary

The patch introduces several runtime bugs in supported scenarios: `/mode` can sever the external config symlink, Linux GPU-gated inference can fail immediately, live JSONL rotation can drop records, and malformed PID state can crash server cleanup. Those issues make the patch unsafe to treat as correct as-is.

Full review comments:

- [P1] Preserve the config.json symlink when saving /mode — Q:\fa-review\portfolio\telegram_poller.py:361-361
  This breaks installations where `config.json` is the external symlink described in the surrounding comment. `atomic_write_json()` writes a temp file and `os.replace()`s it onto `config_path`, which replaces the symlink itself with a regular file in the repo; after the first `/mode` command, external config/secrets updates stop propagating and this process starts editing a forked local copy instead.

- [P1] Make the GPU lock path platform-aware — Q:\fa-review\portfolio\gpu_gate.py:33-34
  On non-Windows deployments, which the surrounding LLM code already supports via `/home/deck/...` paths, `Path("Q:/models")` is just a relative path. The first `gpu_gate()` call from `ministral_signal` or `qwen3_signal` then tries to create `./Q:/models/.gpu_lock` and raises `FileNotFoundError` because that parent directory is never created, so GPU-gated inference fails before the lock can be acquired.

- [P2] Lock JSONL files before rotating them — Q:\fa-review\portfolio\log_rotation.py:319-327
  This rewrite path races with the active `atomic_append_jsonl()` writers used by files in this same rotation set, notably `telegram_inbound.jsonl` and `telegram_messages.jsonl`. Any line appended after `rotate_jsonl()` finishes reading the old file but before `os.replace()` lands in the old file and is silently dropped from the new one, so the scheduled rotation can lose live records on a running process.

- [P2] Handle malformed llama_server.pid without crashing — Q:\fa-review\portfolio\llama_server.py:179-180
  If `llama_server.pid` is truncated or corrupt, `int(content.split(":")[0])` jumps to this handler before `pid` has been assigned. The debug log then raises `UnboundLocalError` while formatting `pid`, which means `_stop_server()` stops recovering gracefully and model swaps/startup can fail on the stale PID-file path.

- [P3] Respect the requested Chroma collection after initialization — Q:\fa-review\portfolio\vector_memory.py:41-42
  After the first successful call, `_get_collection()` returns `_collection` without checking `collection_name`. Any later call that asks for a different collection (for example a test namespace or a per-strategy namespace) silently reuses the first one, mixing embeddings and query results across collections despite the public `collection_name=` parameter.
The patch introduces several runtime bugs in supported scenarios: `/mode` can sever the external config symlink, Linux GPU-gated inference can fail immediately, live JSONL rotation can drop records, and malformed PID state can crash server cleanup. Those issues make the patch unsafe to treat as correct as-is.

## Full review comments

- [P1] Preserve the config.json symlink when saving /mode — Q:\fa-review\portfolio\telegram_poller.py:361-361
  This breaks installations where `config.json` is the external symlink described in the surrounding comment. `atomic_write_json()` writes a temp file and `os.replace()`s it onto `config_path`, which replaces the symlink itself with a regular file in the repo; after the first `/mode` command, external config/secrets updates stop propagating and this process starts editing a forked local copy instead.

- [P1] Make the GPU lock path platform-aware — Q:\fa-review\portfolio\gpu_gate.py:33-34
  On non-Windows deployments, which the surrounding LLM code already supports via `/home/deck/...` paths, `Path("Q:/models")` is just a relative path. The first `gpu_gate()` call from `ministral_signal` or `qwen3_signal` then tries to create `./Q:/models/.gpu_lock` and raises `FileNotFoundError` because that parent directory is never created, so GPU-gated inference fails before the lock can be acquired.

- [P2] Lock JSONL files before rotating them — Q:\fa-review\portfolio\log_rotation.py:319-327
  This rewrite path races with the active `atomic_append_jsonl()` writers used by files in this same rotation set, notably `telegram_inbound.jsonl` and `telegram_messages.jsonl`. Any line appended after `rotate_jsonl()` finishes reading the old file but before `os.replace()` lands in the old file and is silently dropped from the new one, so the scheduled rotation can lose live records on a running process.

- [P2] Handle malformed llama_server.pid without crashing — Q:\fa-review\portfolio\llama_server.py:179-180
  If `llama_server.pid` is truncated or corrupt, `int(content.split(":")[0])` jumps to this handler before `pid` has been assigned. The debug log then raises `UnboundLocalError` while formatting `pid`, which means `_stop_server()` stops recovering gracefully and model swaps/startup can fail on the stale PID-file path.

- [P3] Respect the requested Chroma collection after initialization — Q:\fa-review\portfolio\vector_memory.py:41-42
  After the first successful call, `_get_collection()` returns `_collection` without checking `collection_name`. Any later call that asks for a different collection (for example a test namespace or a per-strategy namespace) silently reuses the first one, mixing embeddings and query results across collections despite the public `collection_name=` parameter.
