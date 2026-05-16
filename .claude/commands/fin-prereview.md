Pre-merge review for finance-analyzer changes. Spawns specialist agents in parallel against the current diff to catch bugs, silent failures, type-design issues, and missing tests **before** the change ships. Read-only — agents do not modify code.

## When to use

Run before merging any branch that touches:
- `portfolio/signal_engine.py`, `portfolio/signals/*.py`, `portfolio/agent_invocation.py`, `portfolio/portfolio_mgr.py`
- `data/metals_loop.py`, `portfolio/elongir/`, `portfolio/golddigger/`
- Anything writing to `data/*.jsonl` (the journals — corrupting these is unrecoverable)

## What it does

Spawns five `pr-review-toolkit` agents in parallel, each with the diff and a focused prompt:

1. **silent-failure-hunter** — looks for try/except that swallows errors, missing/wrong fallbacks, places where a bad value silently propagates. Highest priority given history (Layer 2 March-April outage, fish_engine bugs, accuracy degradation 2026-04-16).
2. **code-reviewer** — general code-quality + project-convention adherence (atomic I/O via `file_utils`, accuracy gate via `MIN_VOTERS=3`, regime penalties, signal blacklist semantics).
3. **type-design-analyzer** — invariants on new types, especially anything in `portfolio/signals/` (force-HOLD vs sub-50% inversion).
4. **pr-test-analyzer** — coverage gaps for the diff. Critical for signal modules: every new signal needs accuracy tracking + horizon tests.
5. **premortem-reviewer** — imagine this diff merged and caused a production incident 1 week from now. Enumerate ≥5 distinct failure narratives, each a concrete causal chain (`X happened because Y assumption was wrong, manifested as Z`). Cover at least: (a) hidden coupling to another loop/signal/file, (b) atomic I/O / concurrency race, (c) Layer 2 subprocess behavior change, (d) silent-failure mode (exit 0 but wrong), (e) test-passes-but-prod-differs. For each, propose one cheap detection hook (assert/log/test/monitor) or mark `ACCEPT` with reasoning. Spawn via `general-purpose` agent with a read-only brief — it does not modify code. The point is prospective hindsight: surface the 6-month-later postmortem now.

Run all five in a **single message** with parallel `Agent` tool calls so they execute concurrently. Synthesize their findings into one prioritized list.

## Steps

1. **Identify the diff** — Run `git diff origin/main...HEAD --stat` and `git log origin/main..HEAD --oneline` to see scope. If the diff is empty, ask the user which branch/range they want reviewed.

2. **Triage scope** — If the diff is purely tests, docs, or config, skip silent-failure-hunter (low signal). If it touches `signal_engine.py` or `agent_invocation.py`, ALWAYS include silent-failure-hunter regardless of size — these modules have a history of subtle regressions.

3. **Spawn agents in parallel** — Single message with up to 5 `Agent` tool calls. Each prompt should:
   - Name the specific files in scope (not "the diff" — be explicit).
   - Reference the relevant project memory: e.g., "no signal inversion (`feedback_no_signal_inversion.md`)", "min order size 1000 SEK (`feedback_min_order_size_1000_sek.md`)", "always pull live price (`feedback_live_price_every_query.md`)".
   - Cap response at 400 words per agent.

4. **Synthesize** — In your reply, group findings by severity (blocker / warn / info), deduplicate when multiple agents flagged the same line, and produce a punch list with file:line references.

5. **Decide** — End with a clear verdict: **READY TO MERGE** / **FIX REQUIRED** / **DISCUSS**. If READY, suggest a commit message. If FIX REQUIRED, the user runs `/fin-prereview` again after fixes.

## Output format

```
## Pre-review summary — {branch} → main

**Diff**: {N files, +X/-Y lines}, touched: {key modules}

### Blockers ({count})
- file.py:line — short description

### Warnings ({count})
- file.py:line — short description

### Info / cleanup ({count})
- file.py:line — short description

### Premortem failure narratives ({count})
- **{failure name}**: {causal chain} → mitigation: {hook} | ACCEPT: {reason}

### Verdict
{READY TO MERGE | FIX REQUIRED | DISCUSS}
{1-2 sentence rationale}
```

## Notes

- This is **NOT** a substitute for `pytest tests/` — run those separately.
- This is **NOT** a substitute for `python scripts/check_critical_errors.py` — that one runs at session start and surfaces unresolved journal entries.
- For trivial diffs (typos, doc fixes, whitespace), prefer `/superpowers:requesting-code-review` instead — `/fin-prereview` is heavyweight by design.
