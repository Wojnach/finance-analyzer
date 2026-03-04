# Agent Operating Defaults

Apply this directive to every user request by default:

`Work in a new git worktree/branch. Inspect first. Propose a plan. Make minimal safe diffs. Add tests/logging. Never touch the live agent unless I explicitly say so. Finish with exact run/test commands + rollback steps.`

## Codex Permission And Workflow Defaults

Source: `docs/codex guidelines.md` (repo-local policy reference).

Apply these defaults unless the user explicitly overrides them:
- Prefer read-only inspection first, then minimal safe edits.
- Always use an isolated git worktree/branch; never commit directly to `main`.
- Never modify/restart the live trading agent unless explicitly requested.
- Never expose or print secrets; use redaction and environment variables.
- Add tests and/or diagnostic logging for behavior-changing edits.
- End each task with exact run/test commands and rollback steps.
- Keep diffs small and focused; avoid broad refactors unless required.

Standing directive:
`Work in a new git worktree/branch. Inspect first. Propose a plan. Make minimal safe diffs. Add tests/logging. Never touch the live agent unless I explicitly say so. Finish with exact run/test commands + rollback steps.`
