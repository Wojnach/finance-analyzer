# Repo-Root Untracked Cleanup — 2026-05-02

**Branch:** `cleanup/repo-root-20260502`
**Worktree:** `/mnt/q/finance-analyzer-repo-cleanup`
**Trigger:** `git status` has been polluted for weeks with untracked cruft in the repo root.

## Why

The repo root accumulated three categories of debris:
1. **Zero-byte shell-redirect artifacts** (`+3.4%`, `Bold`, `SELL`, etc.) — created accidentally
   when something parsed quoted strings as filenames and shell-redirected to them.
2. **Wrong-project leakage** — `booli_*.json/.md` are scrape outputs from the househunting
   project at `/mnt/q/househunting/`. They were never moved to that repo.
3. **Local-only scaffolding** — `.openclaw/`, `.playwright-*/`, `.venv-unsloth/`, plus
   `AGENTS.md`/`SOUL.md`/`USER.md`/etc. (a different agent framework's template stubs).
   These should be gitignored, not committed.

Long-untracked files are noise that hides real WIP in `git status`. Cleaning them up
removes 20+ lines of clutter from every status check and prevents accidental commits.

## Per-File Decisions

### `definitely_cruft` — DELETE
| File | Size | Verified |
|---|---|---|
| `+3.4%` | 0 bytes | Zero-byte shell artifact (Apr 22) |
| `+5.2%` | 0 bytes | Zero-byte shell artifact (Apr 22) |
| `Bold` | 0 bytes | Zero-byte shell artifact (Apr 24) |
| `Bold:` | 0 bytes | Zero-byte shell artifact (Apr 22) |
| `No` | 0 bytes | Zero-byte shell artifact (Apr 22) |
| `SELL` | 0 bytes | Zero-byte shell artifact (Apr 22) |
| `Tied` | 0 bytes | Zero-byte shell artifact (Apr 1) |

All seven confirmed zero-byte. Names match shell-redirect-of-output-token pattern (likely
`echo "+3.4%" > +3.4%` or similar from Telegram message rendering / decision parsing).

### `wrong_project` — DELETE
| File | Size | Content |
|---|---|---|
| `booli_grev_check.json` | 67 B | Booli sold-listings count for "Grev Turegatan" |
| `booli_page1_check.json` | 5,340 B | Booli first-page paginated results |
| `booli_raw_results.json` | 86,057 B | Booli raw scrape blob |
| `booli_search_api.json` | 74 B | Booli search API error response |
| `booli_search_suggestions.md` | 4,795 B | Playwright snapshot of Booli search page |
| `booli_slutpriser_snapshot.md` | 20,324 B | Playwright snapshot of Booli sold-listings page |

Confirmed: NOT replicated in `/mnt/q/househunting/` (already-purged scratch). Safe to delete.

### `dev_tools` — KEEP, ADD GITIGNORE PATTERN
| Directory | Reason |
|---|---|
| `.openclaw/` | OpenClaw workspace state (1 file, 74 bytes) |
| `.playwright-cli/` | Playwright CLI session logs (March, ~108 KB) |
| `.playwright-mcp/` | Playwright MCP session logs (April, ~1.9 MB) |
| `.venv-unsloth/` | Python venv for Unsloth LoRA training (existing `.venv/` pattern doesn't match) |

These are legitimate runtime/dev outputs. They should never be committed.

### `local_config` — KEEP, ADD GITIGNORE PATTERN
| File | Purpose |
|---|---|
| `CLAUDE.local.md` | Per-machine instructions referenced explicitly in CLAUDE.md as not-checked-in |
| `AGENTS.md` | "AGENTS.md" framework workspace doc (template stubs) |

`CLAUDE.local.md` is **already documented as not-committed** in CLAUDE.md itself
("`CLAUDE.local.md` (user's private project instructions, not checked in)"). Add to gitignore.

`AGENTS.md` is a template from a different agent framework (references `SOUL.md`/`USER.md`/etc.).
The user has chosen to keep these in the repo as scaffolding for a personal-agent setup —
they're paired with the personal_notes group below. Gitignore.

### `personal_notes` — KEEP, ADD GITIGNORE PATTERN
| File | Purpose |
|---|---|
| `HEARTBEAT.md` | "Keep this file empty to skip heartbeat API calls" — heartbeat config stub |
| `IDENTITY.md` | "Who Am I?" template (placeholder text) |
| `SOUL.md` | "Who You Are" — agent personality/values template |
| `TOOLS.md` | "Local Notes" — environment-specific notes template |
| `USER.md` | "About Your Human" template |

All 5 files are template stubs from the same external agent framework as `AGENTS.md`.
First-line read of each confirms they are personal/local — none reference finance-analyzer
code. They've been sitting untracked for 4-6 weeks. Safe to gitignore.

## Suggested .gitignore Patterns (FOR git-tracking-hygiene AGENT TO MERGE)

The following block should be appended to `.gitignore`. Grouped logically and commented.

```gitignore
# Personal agent framework scaffolding (not finance-analyzer code)
# These are template stubs from a separate AGENTS.md framework. Local-only.
AGENTS.md
HEARTBEAT.md
IDENTITY.md
SOUL.md
TOOLS.md
USER.md
CLAUDE.local.md

# Dev tooling state (Playwright recordings, OpenClaw workspace, Unsloth venv)
.openclaw/
.playwright-cli/
.playwright-mcp/
.venv-*/

# Stray scrape artifacts from the househunting sibling project (/mnt/q/househunting/)
booli_*.json
booli_*.md
```

**Rationale notes for the merger:**
- The existing `.venv/` line stays; we add `.venv-*/` to also catch `.venv-unsloth/`,
  `.venv-train/` (also already in gitignore as a separate line — the new pattern makes
  that line redundant but not wrong), and any future `.venv-foo/`. Order matters here:
  put `.venv-*/` ABOVE the line `.venv-train/` so dedup is obvious in a future cleanup.
- The `booli_*` patterns are project-specific. If the user later adopts a strict
  "no project-leakage" policy, this won't be needed — but deleting current files +
  adding the pattern is cheap insurance against the next leak.
- `CLAUDE.local.md` is already DOCUMENTED in `CLAUDE.md` line ~283 as "user's private
  project instructions, not checked in" — gitignoring it just enforces what the docs say.

## Risk

Near-zero. All deletions are:
- Either zero-byte (no content lost) or
- Confirmed scrape blobs from another project (regenerable by re-running the scraper).

No tracked files are touched. No `.gitignore` is modified by this agent (per scope).

## Test Plan

Before: 27 untracked entries in repo root (per snapshot).
After: ~0 untracked entries in repo root once gitignore patterns merged by the
hygiene agent. Until then, the dev_tools/personal_notes/local_config files will
still appear as untracked — but the cruft and wrong-project files will be gone.

`git status` should show ~0 untracked-root-file lines after both this branch and
the hygiene branch merge.

## Execution Steps

1. Delete `definitely_cruft` files (7 files, all 0 bytes).
2. Delete `wrong_project` files (6 files, ~117 KB total).
3. Commit deletions on `cleanup/repo-root-20260502` (deletions of untracked-by-git
   files don't show in `git diff`; the commit is documentation-only — this plan doc
   itself is the artifact).
4. Hand off the gitignore-pattern block above to the git-tracking-hygiene agent.

Note: untracked files don't appear in `git status` once deleted via `rm`, and
they were never tracked, so there's nothing for git to record beyond this plan
doc. The branch's only file change is this plan doc.
