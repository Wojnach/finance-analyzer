# Autonomous System Improvement Session

You are about to perform a deep, autonomous improvement session on this codebase. This is a long-running task — spend your entire output context working systematically. Do not stop early due to token budget concerns. Save progress and state to files before your context window refreshes.

## PROGRESS TRACKING (MANDATORY)

You MUST update the progress file at **every phase transition** so external monitoring can
see where you are. Use this exact pattern at the START of each phase:

```python
import json, datetime, pathlib
progress_file = pathlib.Path("data/auto-improve-progress.json")
progress = json.loads(progress_file.read_text()) if progress_file.exists() else {"phases_completed": []}
progress.update({
    "current_phase": "PHASE N: NAME",
    "phase_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "last_update": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running",
    "notes": "Brief description of what you are doing"
})
progress_file.write_text(json.dumps(progress, indent=2))
```

And at the END of each phase (before moving to the next):

```python
progress = json.loads(progress_file.read_text())
if "PHASE N: NAME" not in progress.get("phases_completed", []):
    progress["phases_completed"].append("PHASE N: NAME")
progress["last_update"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
progress["notes"] = "Phase N complete. Moving to Phase N+1."
progress_file.write_text(json.dumps(progress, indent=2))
```

If the session ends (success or failure), set `"status": "done"` or `"status": "failed"`.
If you encounter an error or blocker, update `"status": "blocked"` with details in `"notes"`.

This file is the **only** way to monitor your progress externally. Do not skip this.

---

## PHASE 0: SETUP — Isolate from live environment

- Create a new git worktree branch named `improve/auto-session-<date>` so the live environment is never disturbed.
- Work entirely inside this worktree until all phases are complete.
- If a worktree already exists from a previous session, resume from it.

---

## PHASE 1: EXPLORE — Understand the system deeply (DO NOT write any code yet)

- Read every relevant file. Trace how modules connect. Map dependencies, entry points, data flow, and configuration.
- Look for an existing design document or architecture doc (e.g., `DESIGN.md`, `ARCHITECTURE.md`, `docs/`). If one exists, read it carefully and note any drift from actual implementation.
- Think deeply about the system before proposing any changes. Use extended thinking.

**Deliverable:** Write your findings to `docs/SYSTEM_OVERVIEW.md` — a design doc covering: architecture, module responsibilities, data flow, external dependencies, configuration, and any discrepancies you found vs existing documentation.

---

## PHASE 2: PLAN — Propose improvements before implementing anything

Based on your exploration, create `docs/IMPROVEMENT_PLAN.md` containing:

1. **Bugs & Problems Found** — Actual bugs, race conditions, error handling gaps, security concerns. Include file paths and line references.
2. **Architecture Improvements** — Modularity issues, tight coupling, missing abstractions. Every improvement must explain *why* it matters and what it enables.
3. **Useful Features** — Features you believe would genuinely improve the system. Justify each one.
4. **Refactoring TODOs** — Code quality, naming, dead code, duplication.
5. **Dependency/Ordering** — Which changes must happen before others. Group into ordered batches of 5-10 files max.

**Rules for the plan:**
- Prioritize: critical bugs > architecture > features > polish.
- Every proposed change must include an impact assessment: what other parts of the system could break.
- The system must remain modular — all changes should make it easier to swap, remove, or add parts independently.
- Do NOT plan changes that aren't justified. Less is more if the system is solid.

**Stop and save.** Commit the two docs before proceeding.

---

## PHASE 3: IMPLEMENT — Execute the plan in ordered batches

For each batch in the plan:

1. **Before touching code:** Ensure tests exist for the area you're about to change. If tests are missing, write failing tests FIRST, run them to confirm they fail, then commit the tests separately.
2. **Implement the batch.** Only modify the files listed in that batch. Do not scope-creep into other areas.
3. **After each batch:**
   - Run the full test suite. Fix any failures before moving on.
   - Update existing tests to compensate for your changes.
   - Think deeply: did this change affect other parts of the system? If yes, trace the impact and update those parts too.
   - Commit with a descriptive conventional commit message (e.g., `fix(auth): close race condition in token refresh`).
4. **After every 2-3 batches:** Write current progress to `docs/SESSION_PROGRESS.md` (what's done, what's next, any blockers or decisions made). This is your checkpoint in case of context refresh.

**Rules during implementation:**
- Never delete or overwrite existing tests without understanding why they exist.
- Keep all changes reversible — prefer additive changes over destructive rewrites.
- If you're unsure whether a change is safe, leave a `// TODO: REVIEW` comment and document it in the progress file.
- Run linters/formatters if the project has them configured.

---

## PHASE 4: DOCUMENTATION — Update everything to reflect reality

- Update `docs/SYSTEM_OVERVIEW.md` to reflect the system as it now exists post-changes.
- Update any existing READMEs, API docs, or config docs if your changes affected them.
- Update or create a CHANGELOG entry summarizing what changed and why.
- Remove the `docs/SESSION_PROGRESS.md` file (it was a working doc, not permanent).

---

## PHASE 5: VERIFY & SHIP

1. Run the full test suite one final time. All tests must pass.
2. Run any linters, type checkers, or build steps the project uses. Everything must be clean.
3. Review the git log for this branch — ensure commits are clean and tell a coherent story.
4. Merge the worktree branch into main (or the primary branch).
5. Push.
6. Restart the system/services if applicable.
7. Clean up the worktree.

---

## EXECUTION GUIDELINES

- **Use subagents and agent teams** when parallelism helps: e.g., one subagent runs tests while you implement, one explores a module while another explores a different one, or spin up an agent team when multiple independent modules need changes simultaneously. Use worktree isolation for agents working on overlapping files. Don't force parallelism — only use it when it genuinely speeds things up or improves quality (e.g., separate test-writing agent to prevent context pollution).
- **If your context gets long**, save state to `docs/SESSION_PROGRESS.md`, commit, and reference it after a context refresh. Prefer fresh context over degraded context.
- **Do not ask me for approval** during this session. Make your best judgment calls, document your reasoning in commits and docs, and keep moving.
- **If something feels too risky** to change autonomously, skip it and document it as a `// TODO: MANUAL REVIEW` with full explanation in the improvement plan.
