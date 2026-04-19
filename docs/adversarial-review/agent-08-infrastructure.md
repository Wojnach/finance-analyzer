# Agent Review: infrastructure

## P1 Findings
1. **journal.py non-atomic write** of layer2_context.md — Layer 2 can read truncated context (journal.py:568,580)
2. **prune_jsonl races with atomic_append_jsonl** — drops most recent entries (file_utils.py:304-330)

## P2 Findings
1. **claude_gate unbounded JSONL read** — O(N) on every invocation (claude_gate.py:263-271)
2. **shared_state cache timestamp stale** — TTL underestimated by func duration (shared_state.py:48,95)
3. **Dashboard cache double-checked locking race** — burst requests bypass dedup (dashboard/app.py:70-80)
4. **Dashboard token timing oracle** — str == comparison on CORS * API (dashboard/app.py:675,682)

## P3 Findings
1. dashboard api_mstr_loop raw file read, bypasses cache
2. atomic_append_jsonl sidecar lock TOCTOU (benign)
3. config_validator raw open() instead of load_json
4. journal.py load_recent O(N) file scan
