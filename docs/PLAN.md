# Session Plan: CLAUDE.md Restructure + Project Optimization

Updated: 2026-03-23
Worktree: `bridge-cse_0132tqmY1gBYsC1KbybtNE55`

## Goal

Make every new Claude Code session immediately productive by:
1. Restructuring CLAUDE.md from Layer-2-only playbook → project-wide orientation
2. Splitting trading playbook into dedicated doc for Layer 2
3. Cleaning up overloaded MEMORY.md (208+ lines, truncated at 200)
4. Auditing and trimming MCP plugins
5. Verifying nothing breaks

## What Could Break

- Layer 2 agent may not find the trading playbook if prompt wording is wrong
- Tests in `test_agent_completion.py` or similar may assert on old prompt text
- MEMORY.md cleanup could remove info that future sessions need

## Execution Order

### Batch 1: CLAUDE.md Split (DONE)
- [x] Deep explore: 3 parallel agents (project structure, plugins, memory files)
- [x] New CLAUDE.md (267 lines): project overview, architecture, modules, signals, instruments, CLI, testing, environment, critical rules
- [x] New docs/TRADING_PLAYBOOK.md (385 lines): full Layer 2 trading instructions
- [x] Updated agent_invocation.py: all 3 tier prompts reference playbook
- [x] Updated pf-agent.bat: prompt references playbook
- [x] Committed: f66cc6a, merged to main, pushed

### Batch 2: Test Verification
- [ ] Run tests related to agent_invocation.py
- [ ] Run tests related to trigger/prompt building
- [ ] Verify no test asserts on old "CLAUDE.md instructions" text

### Batch 3: MEMORY.md Cleanup
- [ ] Audit all 18 memory files for staleness
- [ ] Remove info now covered by new CLAUDE.md (architecture, modules, environment)
- [ ] Consolidate duplicated entries
- [ ] Get MEMORY.md under 200-line limit (currently 208+, truncated)

### Batch 4: Plugin Cleanup
- [x] Disabled frontend-design plugin in settings.json
- [ ] Document cloud plugin recommendations for user to action in claude.ai UI

### Batch 5: Verify & Ship
- [ ] Full test suite (parallel)
- [ ] Review git log
- [ ] Update SESSION_PROGRESS.md
