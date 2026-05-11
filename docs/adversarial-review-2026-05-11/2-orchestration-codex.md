Reading additional input from stdin...
OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\fa-adv-2026-05-11
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019e17a7-29a9-75f1-b8d7-a988f953d4c4
--------
user
You are doing an ADVERSARIAL code review of the orchestration subsystem of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files (read these and ONLY these):
- portfolio/main.py
- portfolio/agent_invocation.py
- portfolio/autonomous.py
- portfolio/trigger.py
- portfolio/market_timing.py
- portfolio/claude_gate.py
- portfolio/gpu_gate.py
- portfolio/health.py
- portfolio/alert_budget.py
- portfolio/llm_prewarmer.py
- portfolio/llm_calibration.py
- portfolio/llm_batch.py
- portfolio/llm_outcome_backfill.py
- portfolio/llm_probability_log.py
- portfolio/llama_server.py
- portfolio/multi_agent_layer2.py
- portfolio/perception_gate.py
- portfolio/focus_analysis.py
- portfolio/reporting.py
- portfolio/journal.py
- portfolio/journal_index.py
- portfolio/telegram_notifications.py
- portfolio/telegram_poller.py
- portfolio/digest.py
- portfolio/daily_digest.py
- portfolio/weekly_digest.py
- portfolio/reflection.py
- portfolio/regime_alerts.py
- portfolio/analyze.py
- portfolio/bigbet.py
- portfolio/prophecy.py
- portfolio/qwen3_signal.py
- portfolio/circuit_breaker.py
- portfolio/cumulative_tracker.py
- portfolio/decision_outcome_tracker.py

Project rules:
- 60s loop must not block — parallel ticker processing via ThreadPoolExecutor, 8 workers.
- Crash recovery: exponential backoff 10s→5min, Telegram alert suppression after 5 crashes.
- Layer 2 (Claude CLI subprocess): T1 120s/15 turns, T2 600s/40 turns, T3 900s/40 turns. Layer 2 March-April outage was caused by `claude -p --bare` exiting 0 while printing "Not logged in" — this kind of silent failure mode is the #1 concern.
- CLAUDECODE env var must NOT be inherited by subprocess or `claude -p` fails with "nested session".
- Atomic I/O via file_utils only. JSONL append-only journals.
- Telegram alerts: cap 5 to avoid spam.
- Multi-agent layer2: do not block forever; always set timeout.

Adversarial focus:
1. Subprocess deadlocks / leaks: missing timeout, missing stdout/stderr drains causing pipe full, missing kill on parent exit.
2. Silent-failure modes: exit-0 paths that don't mean success (the `claude -p` "Not logged in" trap).
3. Race conditions on shared state, locks held across IO, GIL-aware concurrency bugs.
4. Trigger detection: hysteresis missing, debounce missing, false-positive cascades.
5. Layer 2 invocation: wrong tier escalation, cost-blow-up paths (T1 → T2 → T3 every loop).
6. Telegram: rate-limit handling, error-on-error infinite loops, secret leakage in messages.
7. Journal corruption: torn writes mid-rotation, mixing keys across schema versions.
8. LLM prewarmer: GPU lock not released on exception, VRAM leak, runaway prewarm spawning.
9. Health/heartbeat: false-green when loop alive but no work done; heartbeat written from wrong thread.
10. Circuit breaker / drawdown: math off-by-one, persistent stuck state, NEVER reset.

Output format — strict markdown:
# Codex adversarial review: orchestration
## Summary
(2-3 sentence overall verdict)
## P0 — Blockers
- path:line — short description. Why it bites: ... Fix: ...
## P1 — High
- ...
## P2 — Medium
- ...
## P3 — Low
- ...
## Tests missing
- ...

Be concrete. Cite line numbers. No false-positive padding.
codex
Reviewing the orchestration files for deadlocks, silent failure paths, concurrency hazards, trigger behavior, and alert/journal reliability. I’m reading the scoped files with line numbers first, then I’ll cross-check the failure modes against the project rules before I write findings.
  → Read all in-scope orchestration files with line numbers
  • Trace subprocess, concurrency, trigger, alerting, and journaling failure modes
  • Draft adversarial findings with exact file:line references
2026-05-11T15:28:32.930387Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/main.py',
'portfolio/agent_invocation.py',
'portfolio/autonomous.py',
'portfolio/trigger.py',
'portfolio/market_timing.py',
'portfolio/claude_gate.py',
'portfolio/gpu_gate.py',
'portfolio/health.py',
'portfolio/alert_budget.py',
'portfolio/llm_prewarmer.py',
'portfolio/llm_calibration.py',
'portfolio/llm_batch.py',
'portfolio/llm_outcome_backfill.py',
'portfolio/llm_probability_log.py',
'portfolio/llama_server.py',
'portfolio/multi_agent_layer2.py',
'portfolio/perception_gate.py',
'portfolio/focus_analysis.py',
'portfolio/reporting.py',
'portfolio/journal.py',
'portfolio/journal_index.py',
'portfolio/telegram_notifications.py',
'portfolio/telegram_poller.py',
'portfolio/digest.py',
'portfolio/daily_digest.py',
'portfolio/weekly_digest.py',
'portfolio/reflection.py',
'portfolio/regime_alerts.py',
'portfolio/analyze.py',
'portfolio/bigbet.py',
'portfolio/prophecy.py',
'portfolio/qwen3_signal.py',
'portfolio/circuit_breaker.py',
'portfolio/cumulative_tracker.py',
'portfolio/decision_outcome_tracker.py'
)
foreach ("'$f in $files) {
  $lines = (Get-Content -LiteralPath $f | Measure-Object -Line).Lines
  Write-Output "$f`t$lines"
}'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/main.py',
'portfolio/agent_invocation.py',
'portfolio/autonomous.py',
'portfolio/trigger.py',
'portfolio/market_timing.py',
'portfolio/claude_gate.py',
'portfolio/gpu_gate.py',
'portfolio/health.py',
'portfolio/alert_budget.py',
'portfolio/llm_prewarmer.py',
'portfolio/llm_calibration.py',
'portfolio/llm_batch.py',
'portfolio/llm_outcome_backfill.py',
'portfolio/llm_probability_log.py',
'portfolio/llama_server.py',
'portfolio/multi_agent_layer2.py',
'portfolio/perception_gate.py',
'portfolio/focus_analysis.py',
'portfolio/reporting.py',
'portfolio/journal.py',
'portfolio/journal_index.py',
'portfolio/telegram_notifications.py',
'portfolio/telegram_poller.py',
'portfolio/digest.py',
'portfolio/daily_digest.py',
'portfolio/weekly_digest.py',
'portfolio/reflection.py',
'portfolio/regime_alerts.py',
'portfolio/analyze.py',
'portfolio/bigbet.py',
'portfolio/prophecy.py',
'portfolio/qwen3_signal.py',
'portfolio/circuit_breaker.py',
'portfolio/cumulative_tracker.py',
'portfolio/decision_outcome_tracker.py'
)
foreach ("'$f in $files) {
  $lines = (Get-Content -LiteralPath $f | Measure-Object -Line).Lines
  Write-Output "$f`t$lines"
}' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
'"'portfolio/main.py',
'portfolio/agent_invocation.py',
'portfolio/autonomous.py',
'portfolio/trigger.py',
'portfolio/market_timing.py',
'portfolio/claude_gate.py',
'portfolio/gpu_gate.py',
'portfolio/health.py',
'portfolio/alert_budget.py',
'portfolio/llm_prewarmer.py',
'portfolio/llm_calibration.py',
'portfolio/llm_batch.py',
'portfolio/llm_outcome_backfill.py',
'portfolio/llm_probability_log.py',
'portfolio/llama_server.py',
'portfolio/multi_agent_layer2.py',
'portfolio/perception_gate.py',
'portfolio/focus_analysis.py',
'portfolio/reporting.py',
'portfolio/journal.py',
'portfolio/journal_index.py',
'portfolio/telegram_notifications.py',
'portfolio/telegram_poller.py',
'portfolio/digest.py',
'portfolio/daily_digest.py',
'portfolio/weekly_digest.py',
'portfolio/reflection.py',
'portfolio/regime_alerts.py',
'portfolio/analyze.py',
'portfolio/bigbet.py',
'portfolio/prophecy.py',
'portfolio/qwen3_signal.py',
'portfolio/circuit_breaker.py',
'portfolio/cumulative_tracker.py',
'portfolio/decision_outcome_tracker.py'
)
foreach ("'$f in $files) {
  $lines = (Get-Content -LiteralPath $f | Measure-Object -Line).Lines
  Write-Output "$f`t$lines"
}'` rejected: blocked by policy
codex
The repo path may not be mounted as the current workspace, so I’m checking what the sandbox can actually read before I continue. If `Q:\finance-analyzer` is outside the allowed root, I’ll need to tell you that directly rather than invent findings.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-Location
Get-ChildItem -Force' in Q:\fa-adv-2026-05-11
 succeeded in 725ms:

[32;1mPath[0m
[32;1m----[0m
Q:\fa-adv-2026-05-11

[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\.claude
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0m.claude
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0m.claude
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\.claude
[32;1mExtension           : [0m.claude
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0m.claude
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\.claude
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\dashboard
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mdashboard
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mdashboard
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\dashboard
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mdashboard
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\dashboard
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\data
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mdata
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mdata
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\data
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mdata
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\data
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\docs
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mdocs
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mdocs
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\docs
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mdocs
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\docs
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\models
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mmodels
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mmodels
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\models
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mmodels
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\models
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\portfolio
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mportfolio
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mportfolio
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\portfolio
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mportfolio
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\portfolio
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\scripts
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mscripts
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mscripts
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\scripts
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mscripts
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\scripts
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\tests
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mtests
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mtests
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\tests
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mtests
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\tests
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\training
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0mtraining
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0mtraining
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\training
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0mtraining
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\training
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mPSPath              : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11\user_data
[32;1mPSParentPath        : [0mMicrosoft.PowerShell.Core\FileSystem::Q:\fa-adv-2026-05-11
[32;1mPSChildName         : [0muser_data
[32;1mPSDrive             : [0mQ
[32;1mPSProvider          : [0mMicrosoft.PowerShell.Core\FileSystem
[32;1mPSIsContainer       : [0mTrue
[32;1mName                : [0muser_data
[32;1mParent              : [0mQ:\fa-adv-2026-05-11
[32;1mRoot                : [0mQ:\
[32;1mExists              : [0mTrue
[32;1mFullName            : [0mQ:\fa-adv-2026-05-11\user_data
[32;1mExtension           : [0m
[32;1mCreationTime        : [0m2026-05-11 17:24:03
[32;1mCreationTimeUtc     : [0m2026-05-11 15:24:03
[32;1mLastAccessTime      : [0m2026-05-11 17:27:04
[32;1mLastAccessTimeUtc   : [0m2026-05-11 15:27:04
[32;1mLastWriteTime       : [0m2026-05-11 17:24:03
[32;1mLastWriteTimeUtc    : [0m2026-05-11 15:24:03
[32;1mLinkTarget          : [0m
[32;1mUnixFileMode        : [0m-1
[32;1mAttributes          : [0mDirectory
[32;1mMode                : [0md----
[32;1mModeWithoutHardLink : [0md----
[32;1mBaseName            : [0muser_data
[32;1mResolvedTarget      : [0mQ:\fa-adv-2026-05-11\user_data
[32;1mTarget              : [0m
[32;1mLinkType            : [0m


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m61
[32;1mName          : [0m.git


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m257
[32;1mName          : [0m.git_commit_msg.txt


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m285
[32;1mName          : [0m.gitattributes


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m5052
[32;1mName          : [0m.gitignore


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m682
[32;1mName          : [0m.mcp.example.json


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m19195
[32;1mName          : [0mCLAUDE.md


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m4964
[32;1mName          : [0mconfig.example.json


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m2660
[32;1mName          : [0mmypy.ini


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m1975
[32;1mName          : [0mpyproject.toml


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m6736
[32;1mName          : [0mREADME.md


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m694
[32;1mName          : [0mrequirements-dev.txt


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m297
[32;1mName          : [0mrequirements.txt


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m4653
[32;1mName          : [0mSESSION_PLAN.md


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m169
[32;1mName          : [0mstart-loop.bat


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m3216
[32;1mName          : [0mtest_digest.py


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m11392
[32;1mName          : [0mTODO.md


[32;1mLastWriteTime : [0m2026-05-11 17:24:03
[32;1mLength        : [0m1432
[32;1mName          : [0mWORKTREE_README.md


[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:28:47.032950Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "(Get-Content -LiteralPath 'portfolio/main.py' | Measure-Object -Line).Lines"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "(Get-Content -LiteralPath 'portfolio/main.py' | Measure-Object -Line).Lines" in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "(Get-Content -LiteralPath 'portfolio/main.py' | Measure-Object -Line).Lines"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/main.py"' in Q:\fa-adv-2026-05-11
 succeeded in 664ms:
1:#!/usr/bin/env python3
2:"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data
4:This is the orchestrator module. All logic has been extracted to:
5:- shared_state.py — mutable globals, caching, rate limiters
6:- market_timing.py — DST-aware market hours, agent window
7:- fx_rates.py — USD/SEK exchange rate fetching
8:- indicators.py — compute_indicators, detect_regime, technical_signal
9:- data_collector.py — Binance/Alpaca/yfinance kline fetchers
10:- signal_engine.py — 30-signal voting system, generate_signal
11:- portfolio_mgr.py — portfolio state load/save/value
12:- reporting.py — agent_summary.json builder
13:- telegram_notifications.py — Telegram send/escape/alert
14:- digest.py — 4-hour digest builder
15:- daily_digest.py — morning daily digest (focus instruments + movers)
16:- message_throttle.py — analysis message rate limiting
17:- agent_invocation.py — Layer 2 Claude Code subprocess
18:- logging_config.py — structured logging setup
19:"""
21:import atexit
22:import logging
23:import os
24:import random
25:import sys
26:import time
27:from contextlib import suppress
28:from datetime import UTC, datetime
29:from pathlib import Path
31:from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json
33:BASE_DIR = Path(__file__).resolve().parent.parent
34:sys.path.insert(0, str(BASE_DIR))
35:DATA_DIR = BASE_DIR / "data"
37:logger = logging.getLogger("portfolio.loop")
39:# --- Singleton guard (same pattern as metals_loop.py) ---
40:# C5: Support both Windows (msvcrt) and Linux/WSL (fcntl) locking.
41:# Previously, non-Windows silently returned True — no protection.
42:try:
43:    import msvcrt
44:except ImportError:
45:    msvcrt = None
47:try:
48:    import fcntl
49:except ImportError:
50:    fcntl = None
52:_SINGLETON_LOCK_FILE = str(DATA_DIR / "main_loop.singleton.lock")
53:_DUPLICATE_EXIT_CODE = 11
54:_singleton_lock_fh = None
57:def _acquire_singleton_lock():
58:    """Acquire single-instance lock for main loop (non-blocking).
60:    C5: Now supports both Windows (msvcrt) and Linux/WSL (fcntl).
61:    Raises RuntimeError if neither locking mechanism is available.
62:    """
63:    global _singleton_lock_fh
64:    if _singleton_lock_fh is not None:
65:        return True
67:    if msvcrt is None and fcntl is None:
68:        logger.error("C5: No file locking available (neither msvcrt nor fcntl)")
69:        raise RuntimeError(
70:            "Singleton lock requires msvcrt (Windows) or fcntl (Linux/WSL)"
71:        )
73:    os.makedirs(os.path.dirname(_SINGLETON_LOCK_FILE), exist_ok=True)
74:    fh = open(_SINGLETON_LOCK_FILE, "a+", encoding="utf-8")
75:    try:
76:        # Always lock byte 0 — "a+" mode positions at EOF for existing files,
77:        # so without seek(0) two processes could lock different byte ranges.
78:        fh.seek(0)
79:        if msvcrt is not None:
80:            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
81:        else:
82:            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
83:    except OSError:
84:        fh.close()
85:        return False
87:    fh.seek(0)
88:    fh.truncate()
89:    fh.write(f"{os.getpid()}\n")
90:    fh.flush()
91:    _singleton_lock_fh = fh
92:    return True
95:def _release_singleton_lock():
96:    """Release single-instance lock if held."""
97:    global _singleton_lock_fh
98:    if _singleton_lock_fh is None:
99:        return
100:    try:
101:        if msvcrt is not None:
102:            _singleton_lock_fh.seek(0)
103:            msvcrt.locking(_singleton_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
104:    except OSError:
105:        pass
106:    finally:
107:        with suppress(Exception):
108:            _singleton_lock_fh.close()
109:        _singleton_lock_fh = None
112:# --- Re-exports for backwards compatibility ---
113:# External code (tests, trigger.py, etc.) that does `from portfolio.main import X`
114:# continues to work via these re-exports.
116:import portfolio.shared_state as _ss
118:# Agent invocation re-exports
119:from portfolio.agent_invocation import (  # noqa: E402, F401
120:    INVOCATIONS_FILE,
121:    TIER_CONFIG,
122:    _log_trigger,
123:    check_agent_completion,
124:    invoke_agent,
125:)
126:from portfolio.api_utils import load_config as _load_config  # noqa: E402, F401
128:# Data collector re-exports
129:from portfolio.data_collector import (  # noqa: E402, F401
130:    ALPACA_BASE,
131:    ALPACA_INTERVAL_MAP,
132:    BINANCE_BASE,
133:    BINANCE_FAPI_BASE,
134:    STOCK_TIMEFRAMES,
135:    TIMEFRAMES,
136:    _fetch_klines,
137:    alpaca_klines,
138:    binance_fapi_klines,
139:    binance_klines,
140:    collect_timeframes,
141:    yfinance_klines,
142:)
144:# Digest re-exports
145:from portfolio.digest import _maybe_send_digest  # noqa: E402, F401
147:# FX rates re-exports
148:from portfolio.fx_rates import _fx_cache, fetch_usd_sek  # noqa: E402, F401
149:from portfolio.http_retry import fetch_with_retry  # noqa: E402, F401
151:# Indicators re-exports
152:from portfolio.indicators import (  # noqa: E402, F401
153:    compute_indicators,
154:    detect_regime,
155:    technical_signal,
156:)
158:# Market timing re-exports
159:from portfolio.market_timing import (  # noqa: E402, F401
160:    INTERVAL_MARKET_CLOSED,
161:    INTERVAL_MARKET_OPEN,
162:    INTERVAL_WEEKEND,
163:    MARKET_OPEN_HOUR,
164:    _is_agent_window,
165:    _is_us_dst,
166:    _market_close_hour_utc,
167:    get_market_state,
168:)
170:# Portfolio manager re-exports
171:from portfolio.portfolio_mgr import (  # noqa: E402, F401
172:    INITIAL_CASH_SEK,
173:    STATE_FILE,
174:    _atomic_write_json,
175:    load_state,
176:    portfolio_value,
177:    save_state,
178:)
180:# Reporting re-exports
181:from portfolio.reporting import (  # noqa: E402, F401
182:    AGENT_SUMMARY_FILE,
183:    COMPACT_SUMMARY_FILE,
184:    _cross_asset_signals,
185:    _write_compact_summary,
186:    write_agent_summary,
187:)
189:# Shared state re-exports
190:from portfolio.shared_state import (  # noqa: E402, F401
191:    _RETRY_COOLDOWN,
192:    FEAR_GREED_TTL,
193:    FUNDAMENTALS_TTL,
194:    FUNDING_RATE_TTL,
195:    MINISTRAL_TTL,
196:    ML_SIGNAL_TTL,
197:    SENTIMENT_TTL,
198:    VOLUME_TTL,
199:    _alpaca_limiter,
200:    _alpha_vantage_limiter,
201:    _binance_limiter,
202:    _cached,
203:    _current_market_state,
204:    _RateLimiter,
205:    _regime_cache,
206:    _regime_cache_cycle,
207:    _run_cycle_id,
208:    _tool_cache,
209:    _yfinance_limiter,
210:)
212:# Signal engine re-exports
213:from portfolio.signal_engine import (  # noqa: E402, F401
214:    MIN_VOTERS_CRYPTO,
215:    MIN_VOTERS_STOCK,
216:    REGIME_WEIGHTS,
217:    _confluence_score,
218:    _get_prev_sentiment,
219:    _load_prev_sentiments,
220:    _prev_sentiment,
221:    _prev_sentiment_loaded,
222:    _set_prev_sentiment,
223:    _time_of_day_factor,
224:    _weighted_consensus,
225:    generate_signal,
226:)
228:# Telegram re-exports
229:from portfolio.telegram_notifications import (  # noqa: E402, F401
230:    BOLD_STATE_FILE,
231:    _maybe_send_alert,
232:    escape_markdown_v1,
233:    send_telegram,
234:)
235:from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS  # noqa: E402, F401
237:CONFIG_FILE = BASE_DIR / "config.json"
240:# --- Helpers ---
242:import re as _re
244:_TICKER_PAT = _re.compile(r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+(?:consensus|moved|flipped)')
247:def _extract_triggered_tickers(reasons):
248:    """Parse ticker names from trigger reason strings.
250:    Examples:
251:        "MU consensus BUY (79%)" -> "MU"
252:        "BTC-USD moved 3.1% up" -> "BTC-USD"
253:        "ETH-USD flipped SELL->BUY (sustained)" -> "ETH-USD"
254:    """
255:    tickers = set()
256:    for reason in reasons:
257:        m = _TICKER_PAT.match(reason)
258:        if m:
259:            tickers.add(m.group(1))
260:    return tickers
263:def _run_post_cycle(config, report=None):
264:    """Post-cycle housekeeping: digest, daily digest, message throttle flush, AV refresh.
266:    Args:
267:        config: Application config dict.
268:        report: Optional CycleReport to track task success/failure.
269:    """
270:    def _track(name, func, *args):
271:        """Run a post-cycle task, tracking success on report if provided."""
272:        try:
273:            func(*args)
274:            if report is not None:
275:                report.post_cycle_results[name] = True
276:        except Exception as e:
277:            logger.warning("%s failed: %s", name, e)
278:            if report is not None:
279:                report.post_cycle_results[name] = False
280:                report.errors.append((name, str(e)))
282:    _maybe_send_digest(config)
283:    # Market health refresh (hourly via internal cache, self-checking)
284:    try:
285:        from portfolio.market_health import maybe_refresh_market_health
286:        _track("market_health", maybe_refresh_market_health)
287:    except Exception as e_mh:
288:        logger.warning("market health import failed: %s", e_mh)
289:    try:
290:        from portfolio.daily_digest import maybe_send_daily_digest
291:        _track("daily_digest", maybe_send_daily_digest, config)
292:    except Exception as e_dd:
293:        logger.warning("daily digest import failed: %s", e_dd)
294:    # BUG-178/W15-W16 follow-up (2026-04-16): daily snapshot writer + the
295:    # daily Telegram summary. Both internally guard once-per-day at the
296:    # configured UTC hour (notification.accuracy_snapshot_hour_utc, default
297:    # 6) and no-op every other call. The hourly degradation check itself
298:    # is wired into loop_contract.verify_contract().
299:    try:
300:        from portfolio.accuracy_degradation import (
301:            maybe_save_daily_snapshot,
302:            maybe_send_degradation_summary,
303:        )
304:        _track("accuracy_snapshot", maybe_save_daily_snapshot, config)
305:        _track("accuracy_degradation_summary",
306:               maybe_send_degradation_summary, config)
307:    except Exception as e_deg:
308:        logger.warning("accuracy degradation import failed: %s", e_deg)
309:    # 2026-05-04: pre-warm the four dashboard horizons (1d/3d/5d/10d)
310:    # for /api/accuracy so the first request after a dashboard restart
311:    # doesn't spend seconds scanning the signal log. Self-gates to once
312:    # per hour internally — cheap on every other cycle.
313:    try:
314:        from portfolio.accuracy_stats import maybe_prewarm_dashboard_accuracy
315:        _track("dashboard_accuracy_prewarm", maybe_prewarm_dashboard_accuracy)
316:    except Exception as e_pw:
317:        logger.warning("dashboard accuracy prewarm import failed: %s", e_pw)
318:    try:
319:        from portfolio.message_throttle import flush_and_send
320:        _track("message_throttle", flush_and_send, config)
321:    except Exception as e_mt:
322:        logger.warning("message throttle import failed: %s", e_mt)
323:    try:
324:        from portfolio.alpha_vantage import refresh_fundamentals_batch, should_batch_refresh
325:        if should_batch_refresh(config):
326:            _track("alpha_vantage", refresh_fundamentals_batch, config)
327:    except Exception as e_av:
328:        logger.warning("Alpha Vantage import failed: %s", e_av)
329:    try:
330:        from portfolio.local_llm_report import maybe_export_local_llm_report
331:        export = maybe_export_local_llm_report(config=config)
332:        if export:
333:            logger.info(
334:                "local LLM report exported: %s (%sd window)",
335:                export["date"],
336:                export["days"],
337:            )
338:        if report is not None:
339:            report.post_cycle_results["local_llm_report"] = True
340:    except Exception as e_report:
341:        logger.warning("local LLM report export failed: %s", e_report)
342:        if report is not None:
343:            report.post_cycle_results["local_llm_report"] = False
344:    # Metals deep context precompute (every 4h, self-checking)
345:    try:
346:        from portfolio.metals_precompute import maybe_precompute_metals
347:        _track("metals_precompute", maybe_precompute_metals, config)
348:    except Exception as e_metals:
349:        logger.warning("Metals precompute import failed: %s", e_metals)
350:    # Oil deep context precompute (every 2h, self-checking)
351:    try:
352:        from portfolio.oil_precompute import maybe_precompute_oil
353:        _track("oil_precompute", maybe_precompute_oil, config)
354:    except Exception as e_oil:
355:        logger.warning("Oil precompute import failed: %s", e_oil)
356:    # Prune unbounded JSONL files to prevent disk exhaustion (BUG-59).
357:    # Per-file isolation: a locked file doesn't block the others.
358:    from portfolio.file_utils import prune_jsonl
359:    _prune_failures = []
360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl", "claude_invocations.jsonl"):
361:        try:
362:            prune_jsonl(DATA_DIR / name, max_entries=5000)
363:        except Exception as e_prune:
364:            _prune_failures.append(name)
365:            logger.warning("JSONL prune failed for %s: %s", name, e_prune)
366:    if report is not None:
367:        report.post_cycle_results["jsonl_prune"] = len(_prune_failures) == 0
368:    # Fin command self-improvement: backfill outcomes + evolve lessons (daily)
369:    try:
370:        from portfolio.fin_evolve import maybe_evolve
371:        _track("fin_evolve", maybe_evolve, config)
372:    except Exception as e_evolve:
373:        logger.warning("Fin evolve import failed: %s", e_evolve)
374:    # Scheduled crypto analysis report (08:00, 13:00, 18:00 CET)
375:    try:
376:        from portfolio.crypto_scheduler import maybe_run_crypto_report
377:        _track("crypto_scheduler", maybe_run_crypto_report, config)
378:    except Exception as e_crypto:
379:        logger.warning("Crypto scheduler import failed: %s", e_crypto)
380:    # Signal postmortem (daily — uses accuracy cache, generates once per day)
381:    try:
382:        from portfolio.file_utils import load_json as _lj
383:        from portfolio.signal_postmortem import POSTMORTEM_FILE, generate_postmortem
384:        pm = _lj(POSTMORTEM_FILE)
385:        # Regenerate if missing or stale (>20 hours old)
386:        if not pm or (time.time() - pm.get("_epoch", 0)) > 72000:
387:            result = generate_postmortem()
388:            if result:
389:                result["_epoch"] = time.time()
390:                from portfolio.file_utils import atomic_write_json as _awj
391:                _awj(POSTMORTEM_FILE, result)
392:        if report is not None:
393:            report.post_cycle_results["signal_postmortem"] = True
394:    except Exception as e_pm:
395:        logger.warning("Signal postmortem failed: %s", e_pm)
396:        if report is not None:
397:            report.post_cycle_results["signal_postmortem"] = False
398:    # H25/L3: Rotate unbounded JSONL files approximately once per hour.
399:    # Was cycle-count-based (every 60 cycles assumed 60s cadence = 1h). After
400:    # the 2026-04-09 cadence bump to 600s that would become once per 10h, so
401:    # this is now driven by wall-clock via a monotonic timestamp tracked on
402:    # shared_state so it survives cross-module access within one process.
403:    _now_rot_ts = time.monotonic()
404:    _last_rot_ts = getattr(_ss, "_last_log_rotation_ts", None)
405:    should_rotate_logs = _last_rot_ts is None or (_now_rot_ts - _last_rot_ts) >= 3600
406:    if should_rotate_logs:
407:        try:
408:            from portfolio.log_rotation import rotate_all
409:            rotation_results = rotate_all()
410:            rotated = [r for r in rotation_results if r.get("status") == "rotated"]
411:            if rotated:
412:                logger.info("Log rotation: %d file(s) rotated: %s",
413:                            len(rotated), [r["file"] for r in rotated])
414:            if report is not None:
415:                report.post_cycle_results["log_rotation"] = True
416:            _ss._last_log_rotation_ts = _now_rot_ts
417:        except Exception as e_rot:
418:            logger.warning("Log rotation failed: %s", e_rot)
419:            if report is not None:
420:                report.post_cycle_results["log_rotation"] = False
423:# --- Main orchestrator ---
426:def run(force_report=False, active_symbols=None):
427:    _ss._run_cycle_id += 1
429:    # Check if a previously-spawned agent has completed (BUG-39).
430:    # 2026-05-05: this call is now lock-protected and shares the
431:    # _completion_lock with portfolio.agent_invocation._completion_watchdog,
432:    # which polls every 30s independent of run()'s cadence. So when this
433:    # cycle bloats (cycle_duration violations 2026-05-01..04), the watchdog
434:    # still observes subprocess completion and enforces the per-tier
435:    # wall-clock timeout — see docs/plans/2026-05-05-l2-completion-watchdog.md.
436:    try:
437:        completion = check_agent_completion()
438:        if completion:
439:            logger.info(
440:                "Agent completed: status=%s tier=%s duration=%.1fs",
441:                completion.get("status"), completion.get("tier"),
442:                completion.get("duration_s", 0),
443:            )
444:    except Exception as e:
445:        logger.warning("check_agent_completion failed: %s", e)
447:    config = _load_config()
448:    state = load_state()
449:    fx_rate = fetch_usd_sek()
451:    market_state, default_symbols, _ = get_market_state()
452:    _ss._current_market_state = market_state
453:    active = active_symbols or default_symbols
455:    skipped = set(SYMBOLS.keys()) - active
456:    skip_note = f" (skipped: {', '.join(sorted(skipped))})" if skipped else ""
457:    logger.info("USD/SEK: %.2f | market: %s%s", fx_rate, market_state, skip_note)
459:    signals_ok = 0
460:    signals_failed = 0
461:    signals = {}
462:    prices_usd = {}
463:    tf_data = {}
465:    _run_start = time.monotonic()
467:    # Loop contract: track what actually happens this cycle
468:    from portfolio.loop_contract import CycleReport
469:    report = CycleReport(cycle_id=_ss._run_cycle_id, active_tickers=set(active))
470:    report.cycle_start = _run_start
472:    # --- Fully parallel: data collection + signal generation per ticker ---
473:    # Each ticker: fetch timeframes, compute indicators, generate signals — all threaded.
474:    # Rate limiters, cache locks, and GPU gate are already thread-safe.
475:    from concurrent.futures import ThreadPoolExecutor, as_completed
477:    active_items = [(name, source) for name, source in SYMBOLS.items() if name in active]
479:    def _process_ticker(name, source):
480:        """Fetch data + generate signals for one ticker. Fully thread-safe."""
481:        try:
482:            t0 = time.monotonic()
483:            tfs = collect_timeframes(source)
484:            tf_elapsed = time.monotonic() - t0
486:            now_entry = tfs[0][1] if tfs else None
487:            now_df = None
488:            if now_entry and "indicators" in now_entry:
489:                ind = now_entry["indicators"]
490:                now_df = now_entry.get("_df")
491:            else:
492:                now_df = _fetch_klines(source, interval="15m", limit=100)
493:                ind = compute_indicators(now_df)
495:            if ind is None:
496:                logger.info("%s: insufficient data, skipping", name)
497:                return name, None
499:            price = ind["close"]
501:            sig_start = time.monotonic()
502:            action, conf, extra = generate_signal(
503:                ind, ticker=name, config=config, timeframes=tfs, df=now_df
504:            )
505:            sig_elapsed = time.monotonic() - sig_start
506:            total_elapsed = time.monotonic() - t0
507:            logger.info(
508:                "%s: timing: tf=%.1fs sig=%.1fs total=%.1fs",
509:                name, tf_elapsed, sig_elapsed, total_elapsed,
510:            )
512:            extra_str = ""
513:            if extra:
514:                parts = []
515:                if extra.get("_gpu_signals_skipped"):
516:                    parts.append("GPU:skip")
517:                if "fear_greed" in extra:
518:                    parts.append(f"F&G:{extra['fear_greed']}")
519:                if "sentiment" in extra:
520:                    parts.append(f"News:{extra['sentiment']}")
521:                if "ministral_action" in extra:
522:                    parts.append(f"8B:{extra['ministral_action']}")
523:                if "ml_action" in extra:
524:                    parts.append(f"ML:{extra['ml_action']}")
525:                if "volume_action" in extra and extra["volume_action"] != "HOLD":
526:                    parts.append(f"Vol:{extra['volume_ratio']}x")
527:                if parts:
528:                    extra_str = f" | {' '.join(parts)}"
529:            enh_parts = []
530:            for esig in ("trend", "momentum", "volume_flow", "volatility_sig",
531:                         "candlestick", "structure", "fibonacci", "smart_money",
532:                         "oscillators", "heikin_ashi", "mean_reversion", "calendar",
533:                         "macro_regime", "momentum_factors", "futures_flow"):
534:                ea = extra.get(f"{esig}_action", "HOLD")
535:                if ea != "HOLD":
536:                    enh_parts.append(f"{esig[:4].title()}:{ea[0]}")
537:            enh_str = f" | Enh: {' '.join(enh_parts)}" if enh_parts else ""
539:            logger.info(
540:                "%s: $%s | RSI %.0f | MACD %+.1f%s%s | %s (%.0f%%)",
541:                name, f"{price:,.2f}", ind['rsi'], ind['macd_hist'], extra_str, enh_str, action, conf * 100
542:            )
544:            for label, entry in tfs[1:]:
545:                if "error" in entry:
546:                    logger.warning("%s: %s", label, entry['error'])
547:                else:
548:                    ei = entry["indicators"]
549:                    logger.info(
550:                        "%s: %s %.0f%% | RSI %.0f | MACD %+.1f",
551:                        label, entry['action'], entry['confidence'] * 100, ei['rsi'], ei['macd_hist']
552:                    )
554:            return name, {
555:                "tfs": tfs, "ind": ind, "now_df": now_df, "price": price,
556:                "action": action, "confidence": conf, "extra": extra,
557:            }
558:        except Exception as e:
559:            logger.error("%s: %s", name, e, exc_info=True)
560:            return name, None
562:    max_workers = max(1, min(len(active_items), 8))
564:    # BUG-178: Add timeout to prevent indefinite hangs from stuck tickers.
565:    #
566:    # Timeline:
567:    # - original: 120s (assumed 60s cycle cadence)
568:    # - 2026-04-09 (CPU fingpt daemon era): 500s — bumped because the CPU
569:    #   fingpt daemon was serializing every ticker's sentiment behind its
570:    #   own global lock, stretching per-ticker latency to ~75s × 5 tickers
571:    #   = ~375s tail. 500s was 2x that max.
572:    # - 2026-04-09 (post feat/fingpt-in-llmbatch): 180s. The fingpt daemon
573:    #   was retired; fingpt moved to portfolio.llm_batch as a post-cycle
574:    #   phase via llama_server full GPU offload. Per-ticker work no longer
575:    #   serialized on fingpt. Live measurement after the merge showed
576:    #   cycles dropping from ~472s to ~226s with 45s/ticker average.
577:    #   180s = 4x the observed per-ticker average and 2x the target "slow"
578:    #   cycle of 90s, a comfortable safety margin for genuinely stuck
579:    #   tickers (network timeouts, yfinance blocking).
580:    # - 2026-04-15: 360s. Telegram alerts at 10:34 showed recurring BUG-178
581:    #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
582:    #   completing 330-525s into the cycle, all 5 within ~10s of each
583:    #   other — the signature of a shared-resource wait rather than truly
584:    #   stuck work. Since 2026-04-09 the ticker path has grown (vix_term_-
585:    #   structure, DXY intraday cross-asset, per-ticker signal gating,
586:    #   fundamental correlation cluster, per-ticker directional accuracy,
587:    #   ETH qwen3 gate) and the llama_server rotation (2026-04-10) means
588:    #   signals occasionally pull stale/miss data under contention bursts.
589:    #   The old 180s was measured when the system had 12 tickers; with 5
590:    #   tickers and more per-ticker work the cost moved legitimately, not
591:    #   because something is "stuck". 360s is 2.8x the observed p50-slow
592:    #   (~130s) and 0.7x the observed p95-slow (~525s), leaving 240s of
593:    #   margin inside the 600s cadence for post-cycle LLM batch, trigger
594:    #   detection, journal, and telegram. Loop contract's own cycle_dur
595:    #   check at 600s remains the catch-all for genuine hangs. Batch 1 of
596:    #   this fix (phase-level instrumentation in signal_engine) and batch
597:    #   2 (signal_utility TTL cache) ship together so we can see per-phase
598:    #   timing in future slow cycles and the next bump decision is
599:    #   grounded in data, not guesswork. See docs/plans/2026-04-15-bug178-
600:    #   instrumentation-timeout.md for the full rationale.
601:    #
602:    # If cycles start creeping above ~360s again, the first place to look
603:    # is the BUG-178 phase log dumped by the slow-cycle diagnostic below —
604:    # acc_load, utility_overlay, weighted_consensus, penalties, linear_-
605:    # factor, and consensus_gate are each tagged in portfolio.log so a
606:    # real bottleneck is identifiable without guessing.
607:    _TICKER_POOL_TIMEOUT = 360
608:    # OR-I-001: avoid context manager — __exit__ calls shutdown(wait=True)
609:    # which blocks the loop when threads hang past the timeout.
610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
611:    futures = {
612:        pool.submit(_process_ticker, name, source): name
613:        for name, source in active_items
614:    }
615:    try:
616:        for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
617:            name, result = future.result()
618:            if result is not None:
619:                tf_data[name] = result["tfs"]
620:                prices_usd[name] = result["price"]
621:                signals[name] = {
622:                    "action": result["action"],
623:                    "confidence": result["confidence"],
624:                    "indicators": result["ind"],
625:                    "extra": result["extra"],
626:                }
627:                signals_ok += 1
628:            else:
629:                signals_failed += 1
630:    except TimeoutError:
631:        timed_out = [n for f, n in futures.items() if not f.done()]
632:        try:
633:            from portfolio.signal_engine import get_last_signal as _get_last
634:            from portfolio.signal_engine import get_phase_log as _get_phase_log
635:            last_sigs = {n: _get_last(n) for n in timed_out}
636:            # 2026-04-15: also dump per-ticker phase breakdown when the pool
637:            # times out. This tells us WHICH post-dispatch phase
638:            # (acc_load / utility_overlay / weighted_consensus / penalties /
639:            # linear_factor / consensus_gate / regime_gate) burned the time,
640:            # so we can target the real bottleneck instead of coarsely blaming
641:            # __post_dispatch__.
642:            phase_logs = {n: _get_phase_log(n) for n in timed_out}
643:        except Exception:
644:            last_sigs = {}
645:            phase_logs = {}
646:        logger.error(
647:            "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
648:            _TICKER_POOL_TIMEOUT, timed_out, last_sigs,
649:        )
650:        for name, phases in phase_logs.items():
651:            if phases:
652:                # Format as 'phase=dur_s' pairs, one ticker per line. Keep on
653:                # one log line per ticker so Windows Event Log / tail -f stays
654:                # readable when 5 tickers time out simultaneously.
655:                phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
656:                logger.error("BUG-178 phases [%s]: %s", name, phase_str)
657:        for f in futures:
658:            f.cancel()
659:        signals_failed += len(timed_out)
660:    finally:
661:        pool.shutdown(wait=False, cancel_futures=True)
663:    # --- Post-cycle LLM batch flush ---
664:    # Ministral/Qwen3/fingpt cache misses were enqueued during parallel
665:    # ticker processing. Now flush them sequentially, grouped by model
666:    # (max 2 swaps: ministral → qwen3 → fingpt). Fingpt phase added
667:    # 2026-04-09 as part of feat/fingpt-in-llmbatch which retired the
668:    # bespoke scripts/fingpt_daemon.py. The sentiment A/B log write is
669:    # also deferred: flush_ab_log() below walks sentiment._pending_ab_entries
670:    # and assembles the final rows once Phase 3 has stashed fingpt results.
671:    try:
672:        from portfolio.llm_batch import _lock as _llm_lock
673:        from portfolio.llm_batch import _ministral_queue, _qwen3_queue, flush_llm_batch
674:        from portfolio.shared_state import MINISTRAL_TTL, _update_cache
675:        # H24/SS2: Capture queued keys before flush to clear stuck loading keys.
676:        with _llm_lock:
677:            _queued_keys = {k for k, _ in _ministral_queue} | {k for k, _ in _qwen3_queue}
678:        batch_results = flush_llm_batch()
679:        for cache_key, result in batch_results.items():
680:            _update_cache(cache_key, result, ttl=MINISTRAL_TTL)
681:        # Clear loading keys for items that didn't return results (retry next cycle).
682:        for key in _queued_keys:
683:            if key not in batch_results:
684:                _update_cache(key, None, ttl=60)
685:        # Now that Phase 3 (fingpt) has stashed its results into
686:        # sentiment._pending_ab_entries via _stash_fingpt_result, write out
687:        # the sentiment_ab_log.jsonl rows for this cycle. Must run AFTER
688:        # flush_llm_batch so the fingpt shadow data is available.
689:        from portfolio.sentiment import flush_ab_log
690:        flush_ab_log()
691:        report.llm_batch_flushed = True
692:    except Exception as e_batch:
693:        logger.warning("LLM batch flush failed: %s", e_batch)
694:        report.errors.append(("llm_batch_flush", str(e_batch)))
696:    _run_elapsed = time.monotonic() - _run_start
697:    logger.info(
698:        "Signal loop done: %d OK, %d failed in %.1fs (%.1fs/ticker avg)",
699:        signals_ok, signals_failed, _run_elapsed,
700:        _run_elapsed / max(signals_ok + signals_failed, 1),
701:    )
703:    # BUG-178 slow-cycle diagnostic (added 2026-04-10, diag/bug178-end-of-
704:    # cycle-snapshot). The ticker pool BUG-178 handler already logs per-ticker
705:    # last_signal state on its 180 s timeout, but cycles that stay under 180 s
706:    # never fire the handler — so slow paths in the 120-180 s range hide from
707:    # us. Fire a warning-level diagnostic when a cycle exceeds 120 s so we
708:    # capture per-ticker phase state retrospectively.
709:    #
710:    # Each value in last_sigs is (sig_name, elapsed_since_set) where sig_name
711:    # is one of: __pre_dispatch__ (hung in sentiment/fear_greed/LLM enqueue),
712:    # a concrete enhanced signal name (hung in the dispatch loop on that one),
713:    # or __post_dispatch__ (hung in accuracy_stats / consensus / per-ticker
714:    # gating). The `elapsed_since_set` value is how long ago the tracker was
715:    # updated — if the cycle total is 150 s but elapsed_since_set for a
716:    # ticker is only 2 s, the slow code is AFTER the last-tracked marker;
717:    # if elapsed_since_set is ~150 s, the thread was stuck at that marker.
718:    if _run_elapsed > 120:
719:        try:
720:            from portfolio.signal_engine import get_last_signal as _get_last
721:            from portfolio.signal_engine import get_phase_log as _get_phase_log
722:            # Use signals.keys() because those are the tickers that successfully
723:            # returned from the pool. Timed-out tickers are already named by the
724:            # BUG-178 handler's Last signals log line.
725:            last_sigs = {n: _get_last(n) for n in signals}
726:            logger.warning(
727:                "Slow cycle diagnostic: %.1fs total, last signals tracked: %s",
728:                _run_elapsed, last_sigs,
729:            )
730:            # 2026-04-15: also dump the post-dispatch phase breakdown for each
731:            # ticker that returned successfully. On a slow cycle the phase log
732:            # reveals which named phase (acc_load, utility_overlay, weighted_-
733:            # consensus, penalties, linear_factor, consensus_gate, regime_gate)
734:            # burned the budget — otherwise we only see the aggregate and can't
735:            # target the fix.
736:            for name in signals:
737:                phases = _get_phase_log(name)
738:                if phases:
739:                    phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
740:                    logger.warning("Slow cycle phases [%s]: %s", name, phase_str)
741:        except Exception as e:
742:            logger.debug("Slow cycle diagnostic failed: %s", e)
744:    # BUG-85: Flush batched sentiment state to disk once per cycle (not per-ticker)
745:    try:
746:        from portfolio.signal_engine import flush_sentiment_state
747:        flush_sentiment_state()
748:    except Exception:
749:        logger.warning("Failed to flush sentiment state", exc_info=True)
751:    # --- Cycle failure alert via Telegram ---
752:    # Collect per-ticker signal failures from this cycle
753:    _cycle_signal_failures = {}
754:    for _tk, _sig in signals.items():
755:        _sf = _sig.get("extra", {}).get("_signal_failures", [])
756:        if _sf:
757:            _cycle_signal_failures[_tk] = _sf
759:    if signals_failed > 0 or _cycle_signal_failures:
760:        _parts = []
761:        if signals_failed > 0:
762:            _parts.append(f"{signals_failed} ticker(s) failed entirely")
763:        if _cycle_signal_failures:
764:            _sf_total = sum(len(v) for v in _cycle_signal_failures.values())
765:            _sf_tickers = ", ".join(
766:                f"{tk}({len(sigs)})" for tk, sigs in _cycle_signal_failures.items()
767:            )
768:            _parts.append(f"{_sf_total} signal failures: {_sf_tickers}")
769:        _fail_msg = f"*LOOP ERRORS* ({int(_run_elapsed)}s cycle)\n" + "\n".join(_parts)
770:        try:
771:            from portfolio.message_store import send_or_store
772:            send_or_store(_fail_msg, config, category="error")
773:        except Exception as _e:
774:            logger.warning("Failed to send cycle error alert: %s", _e)
776:    total = portfolio_value(state, prices_usd, fx_rate)
777:    # BUG-103: Guard against zero/missing initial_value_sek to prevent ZeroDivisionError
778:    initial_val = state.get("initial_value_sek") or INITIAL_CASH_SEK
779:    pnl_pct = ((total - initial_val) / initial_val) * 100
780:    logger.info("Portfolio: %s SEK (%+.2f%%) | Cash: %s SEK", f"{total:,.0f}", pnl_pct, "{:,.0f}".format(state['cash_sek']))
782:    if not STATE_FILE.exists():
783:        save_state(state)
785:    # Log hourly price snapshot for cumulative tracking
786:    try:
787:        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
788:        maybe_log_hourly_snapshot(prices_usd)
789:    except Exception as e:
790:        logger.warning("hourly snapshot failed: %s", e)
792:    # Smart trigger
793:    from portfolio.trigger import check_triggers
795:    fear_greeds = {}
796:    sentiments = {}
797:    for name, sig in signals.items():
798:        extra = sig.get("extra", {})
799:        if "fear_greed" in extra:
800:            fear_greeds[name] = {
801:                "value": extra["fear_greed"],
802:                "classification": extra.get("fear_greed_class", ""),
803:            }
804:        if "sentiment" in extra:
805:            sentiments[name] = extra["sentiment"]
807:    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)
809:    if triggered or force_report:
810:        reasons_list = reasons if reasons else ["startup"]
811:        summary = write_agent_summary(signals, prices_usd, fx_rate, state, tf_data, reasons_list)
812:        report.summary_written = True
813:        logger.info("Trigger: %s", ', '.join(reasons_list))
815:        # Classify tier and write tier-specific context
816:        from portfolio.reporting import write_tiered_summary
817:        from portfolio.trigger import classify_tier, update_tier_state
818:        tier = classify_tier(reasons_list)
819:        triggered_tickers = _extract_triggered_tickers(reasons_list)
820:        write_tiered_summary(summary, tier, triggered_tickers)
821:        update_tier_state(tier)
822:        logger.info("Tier: T%d (%s)", tier, TIER_CONFIG.get(tier, {}).get('label', 'UNKNOWN'))
824:        try:
825:            from portfolio.outcome_tracker import log_signal_snapshot
826:            log_signal_snapshot(signals, prices_usd, fx_rate, reasons_list)
827:        except Exception as e:
828:            logger.warning("signal logging failed: %s", e)
830:        # 2026-05-04: Wrap long-blocking work (Layer 2 T2/T3 = 600-900s
831:        # subprocess; autonomous fallback = bounded but not instant) in a
832:        # heartbeat keepalive. update_health() (the normal heartbeat write)
833:        # only runs at end-of-cycle, so without periodic ticks the
834:        # dashboard /api/health flips stale 300s into any triggering cycle
835:        # even though the loop is alive and waiting on Claude CLI.
836:        # The context manager's __exit__ runs on exceptions too, so the
837:        # daemon thread is always cleaned up. Skip-paths (NO_TELEGRAM,
838:        # outside agent window) are NOT wrapped — they don't block.
839:        from portfolio.health import heartbeat_keepalive
841:        layer2_cfg = config.get("layer2", {})
842:        if os.environ.get("NO_TELEGRAM"):
843:            logger.info("[NO_TELEGRAM] Skipping agent invocation")
844:            _log_trigger(reasons_list, "skipped_test", tier=tier)
845:        elif layer2_cfg.get("enabled", True):
846:            if _is_agent_window():
847:                with heartbeat_keepalive():
848:                    result = invoke_agent(reasons_list, tier=tier)
849:                _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
850:            else:
851:                logger.info("Layer 2: outside market window, skipping")
852:                _log_trigger(reasons_list, "skipped_offhours", tier=tier)
853:        else:
854:            logger.info("Layer 2 disabled — autonomous mode")
855:            from portfolio.autonomous import autonomous_decision
856:            with heartbeat_keepalive():
857:                autonomous_decision(
858:                    config, signals, prices_usd, fx_rate, state,
859:                    reasons_list, tf_data, tier, triggered_tickers,
860:                )
861:            _log_trigger(reasons_list, "autonomous", tier=tier)
862:    else:
863:        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
864:        report.summary_written = True
865:        logger.info("No trigger.")
867:    # Big Bet detection — can invoke a 30s Claude subprocess per qualifying
868:    # candidate (portfolio/bigbet.py:invoke_layer2_eval), with no per-cycle
869:    # cap. Wrapped in keepalive so heartbeat stays fresh across multi-minute
870:    # bigbet sweeps that would otherwise re-trip the dashboard stale gate.
871:    bigbet_cfg = config.get("bigbet", {})
872:    if bigbet_cfg.get("enabled", False):
873:        try:
874:            from portfolio.bigbet import check_bigbet
875:            from portfolio.health import heartbeat_keepalive
876:            with heartbeat_keepalive():
877:                check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
878:        except Exception as e:
879:            logger.warning("Big Bet check failed: %s", e)
881:    # ISKBETS monitoring — same shape: each qualifying ticker can fire a 30s
882:    # Claude gate subprocess (portfolio/iskbets.py:invoke_layer2_gate). With
883:    # 5 Tier-1 tickers configured the worst case is ~150s of subprocess work,
884:    # well past the 300s heartbeat threshold when stacked with bigbet+L2.
885:    try:
886:        from portfolio.health import heartbeat_keepalive
887:        from portfolio.iskbets import check_iskbets
888:        with heartbeat_keepalive():
889:            check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
890:    except Exception as e:
891:        logger.warning("ISKBETS check failed: %s", e)
893:    # Avanza pending order confirmations
894:    try:
895:        from portfolio.avanza_orders import check_pending_orders
896:        check_pending_orders(config)
897:    except Exception as e:
898:        logger.warning("Avanza order check failed: %s", e)
900:    # Periodic trade reflection
901:    try:
902:        from portfolio.reflection import maybe_reflect
903:        maybe_reflect(config)
904:    except Exception as e:
905:        logger.warning("reflection check failed: %s", e)
907:    # Health update
908:    try:
909:        from portfolio.health import update_health
910:        trigger_reason = reasons[0] if (triggered or force_report) and reasons else None
911:        update_health(
912:            cycle_count=_ss._run_cycle_id,
913:            signals_ok=signals_ok,
914:            signals_failed=signals_failed,
915:            last_trigger_reason=trigger_reason,
916:        )
917:        report.health_updated = True
918:    except Exception as e:
919:        logger.warning("health update failed: %s", e)
920:        report.errors.append(("health_update", str(e)))
922:    # Periodic safeguard checks (every 100 cycles ≈ 100 min)
923:    if _ss._run_cycle_id % 100 == 0 and _ss._run_cycle_id > 0:
924:        try:
925:            from portfolio.health import check_dead_signals, check_outcome_staleness
926:            outcome_status = check_outcome_staleness()
927:            if outcome_status["stale"]:
928:                age = outcome_status["newest_outcome_age_hours"]
929:                msg = (f"⚠️ SAFEGUARD: Outcome backfill stale! "
930:                       f"Newest outcome: {age:.0f}h ago. "
931:                       f"Entries missing outcomes: {outcome_status['entries_without_outcomes']}/50. "
932:                       f"Accuracy data is degrading.")
933:                logger.warning(msg)
934:                try:
935:                    from portfolio.message_store import send_or_store
936:                    send_or_store(msg, config, category="error")
937:                except Exception:
938:                    logger.warning("Failed to send outcome staleness alert", exc_info=True)
940:            dead_signals = check_dead_signals()
941:            if dead_signals:
942:                msg = (f"⚠️ SAFEGUARD: Dead signals (100% HOLD in last 20 entries): "
943:                       f"{', '.join(dead_signals)}. "
944:                       f"These signals contribute nothing to consensus.")
945:                logger.warning(msg)
946:                try:
947:                    from portfolio.message_store import send_or_store
948:                    send_or_store(msg, config, category="error")
949:                except Exception:
950:                    logger.warning("Failed to send dead signals alert", exc_info=True)
951:        except Exception as e:
952:            logger.warning("safeguard checks failed: %s", e)
954:    # Log portfolio equity snapshot for dashboard chart
955:    try:
956:        from portfolio.risk_management import log_portfolio_value
957:        log_portfolio_value()
958:    except Exception as e:
959:        logger.warning("equity snapshot failed: %s", e)
961:    # Finalize cycle report
962:    report.signals_ok = signals_ok
963:    report.signals_failed = signals_failed
964:    report.signals = signals
965:    report.cycle_end = time.monotonic()
966:    return report
969:_MAX_CRASH_ALERTS = 5  # stop sending alerts after this many consecutive crashes
970:_MAX_CRASH_BACKOFF = 300  # max sleep between crashes (5 min)
971:_CRASH_SUMMARY_INTERVAL = 100  # send a summary every N crashes after suppression
972:_CRASH_COUNTER_FILE = DATA_DIR / "crash_counter.json"
975:def _load_crash_counter() -> int:
976:    """Load persisted crash counter. Returns 0 if missing/corrupt."""
977:    data = load_json(_CRASH_COUNTER_FILE)
978:    if data and isinstance(data.get("count"), int):
979:        return data["count"]
980:    return 0
983:def _save_crash_counter(count: int) -> None:
984:    """Persist crash counter to survive process restarts."""
985:    atomic_write_json(_CRASH_COUNTER_FILE, {
986:        "count": count,
987:        "updated": datetime.now(UTC).isoformat(),
988:    })
991:_consecutive_crashes = _load_crash_counter()
994:def _crash_alert(error_msg):
995:    """Save crash alert to message log with crash-loop protection.
997:    After _MAX_CRASH_ALERTS consecutive crashes, stops sending alerts
998:    to prevent Telegram spam — except for periodic summaries every
999:    _CRASH_SUMMARY_INTERVAL crashes so operators retain visibility.
1000:    Sleep backoff is handled by the caller.
1001:    """
1002:    global _consecutive_crashes
1003:    _consecutive_crashes += 1
1004:    _save_crash_counter(_consecutive_crashes)
1006:    if _consecutive_crashes > _MAX_CRASH_ALERTS:
1007:        logger.error(
1008:            "Crash #%d (alerts suppressed after %d): %s",
1009:            _consecutive_crashes, _MAX_CRASH_ALERTS, error_msg[:200],
1010:        )
1011:        # Periodic summary so operators don't lose visibility
1012:        if _consecutive_crashes % _CRASH_SUMMARY_INTERVAL == 0:
1013:            try:
1014:                config_path = Path(__file__).resolve().parent.parent / "config.json"
1015:                config = load_json(config_path, default={})
1016:                text = (
1017:                    f"CRASH LOOP SUMMARY: {_consecutive_crashes} consecutive crashes\n\n"
1018:                    f"Latest: {error_msg[:2000]}"
1019:                )
1020:                from portfolio.message_store import send_or_store
1021:                send_or_store(text, config, category="error")
1022:            except Exception as e:
1023:                logger.debug("Crash summary send failed: %s", e)
1024:        return
1026:    try:
1027:        config_path = Path(__file__).resolve().parent.parent / "config.json"
1028:        config = load_json(config_path, default={})
1029:        text = f"LOOP CRASH #{_consecutive_crashes}\n\n{error_msg[:3000]}"
1030:        if _consecutive_crashes == _MAX_CRASH_ALERTS:
1031:            text += "\n\n_Further crash alerts suppressed until recovery._"
1032:        from portfolio.message_store import send_or_store
1033:        send_or_store(text, config, category="error")
1034:    except Exception as e:
1035:        logger.debug("Crash alert send failed: %s", e)
1038:def _crash_sleep():
1039:    """Exponential backoff sleep with jitter for consecutive crashes.
1041:    Jitter prevents synchronized retry storms when multiple loops
1042:    (main + metals) crash simultaneously.
1043:    """
1044:    base_delay = min(10 * (2 ** (_consecutive_crashes - 1)), _MAX_CRASH_BACKOFF)
1045:    delay = base_delay * (0.5 + random.random())  # jitter: 50-150% of base
1046:    logger.info("Crash backoff: sleeping %.0fs (crash #%d)", delay, _consecutive_crashes)
1047:    time.sleep(delay)
1050:# Adversarial review 04-29 OR-P1-2 (2026-05-02): the original loop did
1051:#     _crash_alert(traceback)
1052:#     _crash_sleep()
1053:# inline in the except handler. If _crash_alert raised (disk full on
1054:# _save_crash_counter, load_json IO error before the inner try guard, etc.)
1055:# the loop process either died entirely or — under any future refactor
1056:# that tried to be defensive — proceeded without backoff. Even today
1057:# `_crash_sleep` would fire 10 * 2^(-1) ≈ 5s but only if
1058:# _consecutive_crashes was incremented by a non-failing _crash_alert call.
1059:#
1060:# This wrapper guarantees the loop ALWAYS sleeps before continuing,
1061:# regardless of what fails inside _crash_alert/_crash_sleep — using the
1062:# plan's recommended `time.sleep(min(2 ** n_failures, 30))` floor as the
1063:# last line of defense.
1064:_CRASH_FLOOR_SLEEP_CAP = 30  # seconds — plan-spec ceiling
1067:def _safe_crash_recovery(traceback_text: str) -> None:
1068:    """Crash-recovery sequence with a guaranteed minimum sleep floor.
1070:    Sequence:
1071:      1. _crash_alert (Telegram + counter persistence) — best-effort,
1072:         exceptions logged but never re-raised.
1073:      2. _crash_sleep (exponential backoff with jitter) — best-effort,
1074:         exceptions logged but never re-raised.
1075:      3. Floor sleep `min(2 ** n_failures, 30)` IF _crash_sleep didn't
1076:         actually sleep (raised before time.sleep, or both attempts above
1077:         died). Always fires when both helpers raise; skipped when
1078:         _crash_sleep ran cleanly (it already paused the cycle).
1080:    The floor exists to prevent the loop from spinning tight on persistent
1081:    failure when the alerting machinery itself is broken (e.g. disk full
1082:    is what's crashing the loop in the first place — same disk hosts the
1083:    crash counter file).
1084:    """
1085:    crash_sleep_succeeded = False
1086:    try:
1087:        _crash_alert(traceback_text)
1088:    except Exception as e:  # noqa: BLE001
1089:        logger.warning("Crash alert helper raised: %s — proceeding to floor sleep", e)
1090:    try:
1091:        _crash_sleep()
1092:        crash_sleep_succeeded = True
1093:    except Exception as e:  # noqa: BLE001
1094:        logger.warning("Crash sleep helper raised: %s — using floor sleep", e)
1096:    if not crash_sleep_succeeded:
1097:        n = max(_consecutive_crashes, 1)
1098:        # Cap exponent at log2(_CRASH_FLOOR_SLEEP_CAP) to avoid
1099:        # arithmetic overflow on n=50 — 2 ** 50 is fine in Python but
1100:        # the cap math happens regardless.
1101:        floor = min(2 ** min(n, 16), _CRASH_FLOOR_SLEEP_CAP)
1102:        logger.warning("Crash floor sleep: %.0fs (crash #%d)", floor, n)
1103:        time.sleep(floor)
1106:def _reset_crash_counter():
1107:    """Reset crash counter after a successful run cycle."""
1108:    global _consecutive_crashes
1109:    if _consecutive_crashes > 0:
1110:        logger.info("Recovered after %d consecutive crashes", _consecutive_crashes)
1111:        _consecutive_crashes = 0
1112:        _save_crash_counter(0)
1115:def _sleep_for_next_cycle(previous_cycle_started, interval_s):
1116:    """Sleep until the next scheduled cycle start.
1118:    Anchors cadence to cycle start time so the loop period remains close to the
1119:    configured interval instead of drifting by the work duration each cycle.
1120:    """
1121:    elapsed = time.monotonic() - previous_cycle_started
1122:    remaining = interval_s - elapsed
1123:    if remaining > 0:
1124:        time.sleep(remaining)
1125:        return
1126:    logger.warning("Loop overran target cadence by %.1fs", abs(remaining))
1129:def loop(interval=None):
1130:    from portfolio.logging_config import setup_logging
1131:    setup_logging()
1133:    # Prevent duplicate loop instances
1134:    if not _acquire_singleton_lock():
1135:        logger.warning("Duplicate main loop instance detected; exiting.")
1136:        sys.exit(_DUPLICATE_EXIT_CODE)
1137:    atexit.register(_release_singleton_lock)
1139:    # Validate config on startup (fail fast if misconfigured)
1140:    from portfolio.config_validator import validate_config_file
1141:    validate_config_file()
1143:    # Check if previous loop crashed (stale heartbeat)
1144:    heartbeat_file = DATA_DIR / "heartbeat.txt"
1145:    if heartbeat_file.exists():
1146:        try:
1147:            last_beat = datetime.fromisoformat(heartbeat_file.read_text().strip())
1148:            age_seconds = (datetime.now(UTC) - last_beat).total_seconds()
1149:            if age_seconds > 300:  # 5 minutes — previous loop likely crashed
1150:                age_min = int(age_seconds // 60)
1151:                msg = f"_LOOP RESTARTED_ — previous heartbeat was {age_min}m ago. Possible crash."
1152:                logger.warning(msg)
1153:                try:
1154:                    config = _load_config()
1155:                    from portfolio.message_store import send_or_store
1156:                    send_or_store(msg, config, category="error")
1157:                except Exception as e2:
1158:                    logger.debug("Restart notification failed: %s", e2)
1159:        except Exception as e:
1160:            logger.warning("Failed to check heartbeat staleness: %s", e)
1162:    # Reset session start_time so uptime_seconds is accurate for this session
1163:    from portfolio.health import reset_session_start
1164:    reset_session_start()
1166:    logger.info("Loop started")
1168:    # Load Alpha Vantage fundamentals cache from disk
1169:    try:
1170:        from portfolio.alpha_vantage import load_persistent_cache
1171:        load_persistent_cache()
1172:    except Exception as e:
1173:        logger.warning("Failed to load fundamentals cache: %s", e)
1175:    config = _load_config()
1176:    logger.info("Starting loop with market-aware scheduling. Ctrl+C to stop.")
1178:    try:
1179:        from portfolio.iskbets import handle_command
1180:        from portfolio.telegram_poller import TelegramPoller
1181:        poller = TelegramPoller(config, on_command=handle_command)
1182:        poller.start()
1183:        logger.info("ISKBETS Telegram poller started")
1184:    except Exception as e:
1185:        logger.warning("ISKBETS poller failed to start: %s", e)
1187:    try:
1188:        initial_report = run(force_report=True)
1189:        _run_post_cycle(config, report=initial_report)
1190:        _reset_crash_counter()
1191:        try:
1192:            atomic_write_text(DATA_DIR / "heartbeat.txt", datetime.now(UTC).isoformat())
1193:            if initial_report is not None:
1194:                initial_report.heartbeat_updated = True
1195:        except Exception as e:
1196:            logger.warning("Heartbeat write after initial run failed: %s", e)
1197:    except KeyboardInterrupt:
1198:        logger.info("Loop interrupted during initial run, shutting down cleanly")
1199:        return
1200:    except Exception as e:
1201:        import traceback
1202:        # OR-P1-2 (2026-05-02): wrap alert+sleep in _safe_crash_recovery so
1203:        # an alert helper failure (disk full on crash counter, etc.) still
1204:        # leaves a minimum backoff before the next try.
1205:        logger.error("in initial run: %s", e)
1206:        _safe_crash_recovery(traceback.format_exc())
1208:    last_state = None
1209:    last_cycle_started = time.monotonic()
1210:    while True:
1211:        market_state, active_symbols, sleep_interval = get_market_state()
1212:        if interval:
1213:            sleep_interval = interval
1214:        if market_state != last_state:
1215:            logger.info(
1216:                "Schedule: %s — %d instruments, %ds interval",
1217:                market_state, len(active_symbols), sleep_interval
1218:            )
1219:            last_state = market_state
1220:        _sleep_for_next_cycle(last_cycle_started, sleep_interval)
1221:        cycle_started = time.monotonic()
1222:        try:
1223:            report = run(force_report=False, active_symbols=active_symbols)
1224:            _run_post_cycle(config, report=report)
1225:            _reset_crash_counter()
1226:        except KeyboardInterrupt:
1227:            logger.info("Loop interrupted, shutting down cleanly")
1228:            break
1229:        except Exception as e:
1230:            import traceback
1231:            tb_text = traceback.format_exc()
1232:            logger.error("in run: %s", e)
1233:            try:
1234:                from portfolio.health import update_health
1235:                update_health(cycle_count=_ss._run_cycle_id, signals_ok=0, signals_failed=0,
1236:                              error=str(e))
1237:            except Exception as e2:
1238:                logger.warning("Health update after crash failed: %s", e2)
1239:            # OR-P1-2 (2026-05-02): _safe_crash_recovery guarantees a
1240:            # minimum sleep even if both _crash_alert and _crash_sleep fail
1241:            # (e.g. disk full breaking the counter file). Without this floor,
1242:            # the loop could spin tight on persistent failure since
1243:            # _sleep_for_next_cycle takes 0s when elapsed > interval.
1244:            _safe_crash_recovery(tb_text)
1245:            report = None
1246:        last_cycle_started = cycle_started
1247:        try:
1248:            atomic_write_text(DATA_DIR / "heartbeat.txt", datetime.now(UTC).isoformat())
1249:            if report is not None:
1250:                report.heartbeat_updated = True
1251:        except Exception as e:
1252:            logger.warning("Heartbeat write failed: %s", e)
1253:        # Loop contract verification
1254:        if report is not None:
1255:            try:
1256:                from portfolio.loop_contract import verify_and_act
1257:                verify_and_act(report, config)
1258:            except Exception as e_contract:
1259:                logger.warning("Loop contract check failed: %s", e_contract)
1262:if __name__ == "__main__":
1263:    args = sys.argv[1:]
1264:    if "--check-outcomes" in args:
1265:        print("=== Outcome Backfill ===")
1266:        from portfolio.outcome_tracker import backfill_outcomes
1267:        updated = backfill_outcomes()
1268:        print(f"Updated {updated} entries")
1269:        # Also backfill forecast outcomes
1270:        print("\n=== Forecast Outcome Backfill ===")
1271:        try:
1272:            from portfolio.forecast_accuracy import backfill_forecast_outcomes
1273:            fc_updated = backfill_forecast_outcomes()
1274:            print(f"Updated {fc_updated} forecast entries")
1275:        except Exception as e:
1276:            print(f"Forecast backfill failed: {e}")
1277:        # Backfill Layer 2 decision outcomes
1278:        print("\n=== Decision Outcome Backfill ===")
1279:        try:
1280:            from portfolio.decision_outcome_tracker import backfill_decision_outcomes
1281:            dec_updated = backfill_decision_outcomes()
1282:            print(f"Updated {dec_updated} decision outcome entries")
1283:        except Exception as e:
1284:            print(f"Decision outcome backfill failed: {e}")
1285:        # Signal decay detection
1286:        print("\n=== Signal Decay Check ===")
1287:        try:
1288:            from portfolio.signal_decay_alert import run_decay_check
1289:            decay_alerts = run_decay_check()
1290:            if decay_alerts:
1291:                print(f"Found {len(decay_alerts)} decaying signals")
1292:            else:
1293:                print("No signal decay detected")
1294:        except Exception as e:
1295:            print(f"Signal decay check failed: {e}")
1296:    elif "--accuracy" in args:
1297:        from portfolio.accuracy_stats import print_accuracy_report
1298:        print_accuracy_report()
1299:    elif "--forecast-accuracy" in args:
1300:        from portfolio.forecast_accuracy import print_forecast_accuracy_report
1301:        print_forecast_accuracy_report()
1302:    elif "--local-llm-report" in args:
1303:        from portfolio.local_llm_report import print_local_llm_report
1304:        idx = args.index("--local-llm-report")
1305:        days = int(args[idx + 1]) if idx + 1 < len(args) and not args[idx + 1].startswith("--") else 30
1306:        print_local_llm_report(days=days)
1307:    elif "--export-local-llm-report" in args:
1308:        from portfolio.local_llm_report import HISTORY_FILE, LATEST_REPORT_FILE, export_local_llm_report
1309:        idx = args.index("--export-local-llm-report")
1310:        days = int(args[idx + 1]) if idx + 1 < len(args) and not args[idx + 1].startswith("--") else 30
1311:        export = export_local_llm_report(days=days)
1312:        print(f"Exported local LLM report for {export['date']} ({days}d window)")
1313:        print(f"Latest: {LATEST_REPORT_FILE}")
1314:        print(f"History: {HISTORY_FILE}")
1315:    elif "--prophecy-review" in args:
1316:        from portfolio.prophecy import print_prophecy_review
1317:        print_prophecy_review()
1318:    elif "--forecast-outcomes" in args:
1319:        print("=== Forecast Outcome Backfill ===")
1320:        from portfolio.forecast_accuracy import backfill_forecast_outcomes
1321:        updated = backfill_forecast_outcomes()
1322:        print(f"Updated {updated} forecast entries with actual outcomes")
1323:    elif "--retrain" in args:
1324:        print("=== ML Retraining ===")
1325:        print("Refreshing data from Binance API...")
1326:        from portfolio.data_refresh import refresh_all
1327:        refresh_all(days=365)
1328:        print("\nTraining model...")
1329:        from portfolio.ml_trainer import load_data, train_final
1330:        data = load_data()
1331:        feature_cols = [c for c in data.columns if c not in ("target", "month")]
1332:        print(f"Dataset: {len(data):,} rows, {len(feature_cols)} features")
1333:        train_final(data, feature_cols)
1334:        print("Done.")
1335:    elif "--analyze" in args:
1336:        idx = args.index("--analyze")
1337:        if idx + 1 >= len(args):
1338:            print("Usage: --analyze TICKER (e.g. --analyze ETH-USD)")
1339:            sys.exit(1)
1340:        ticker = args[idx + 1].upper()
1341:        from portfolio.analyze import run_analysis
1342:        run_analysis(ticker)
1343:    elif "--analyze-focus" in args:
1344:        idx = args.index("--analyze-focus")
1345:        raw = args[idx + 1] if idx + 1 < len(args) and not args[idx + 1].startswith("--") else ""
1346:        tickers = [t.strip() for t in raw.split(",") if t.strip()] if raw else None
1347:        from portfolio.focus_analysis import run_focus_analysis
1348:        msg = run_focus_analysis(tickers=tickers)
1349:        print(msg)
1350:    elif "--watch" in args:
1351:        idx = args.index("--watch")
1352:        pos_args = args[idx + 1:]
1353:        if not pos_args:
1354:            print("Usage: --watch TICKER:ENTRY [TICKER:ENTRY ...]")
1355:            print("Example: --watch BTC:66500 ETH:1920 AMD:150")
1356:            sys.exit(1)
1357:        from portfolio.analyze import watch_positions
1358:        watch_positions(pos_args)
1359:    elif "--price-targets" in args:
1360:        from pathlib import Path as _Path
1361:        _data = _Path("data")
1362:        _summary = load_json(_data / "agent_summary.json", default={})
1363:        _patient = load_json(_data / "portfolio_state.json", default={})
1364:        _bold = load_json(_data / "portfolio_state_bold.json", default={})
1365:        from portfolio.api_utils import load_config as _pt_load
1366:        from portfolio.price_targets import compute_all_targets
1367:        _pt_cfg = _pt_load().get("price_targets", {})
1368:        results = compute_all_targets(_summary, {"patient": _patient, "bold": _bold}, _pt_cfg)
1369:        for ticker, data in (results or {}).items():
1370:            side = data["side"].upper()
1371:            print(f"\n=== {ticker} {side} @ ${data['price_usd']:.2f} ({data['hours_remaining']:.1f}h left) ===")
1372:            ext = data.get("extremes", {})
1373:            label = "Running max" if data["side"] == "sell" else "Running min"
1374:            print(f"{label}: p25=${ext.get('p25',0):.2f}  p50=${ext.get('p50',0):.2f}  p75=${ext.get('p75',0):.2f}  p90=${ext.get('p90',0):.2f}")
1375:            for t in data.get("targets", [])[:5]:
1376:                print(f"  {t['label']:<14} ${t['price']:.2f}  fill={t['fill_prob']:.0%}  EV={t['ev_sek']:+,.0f} SEK")
1377:            rec = data.get("recommended")
1378:            if rec:
1379:                print(f"  >>> RECOMMENDED: ${rec['price']:.2f}  fill={rec['fill_prob']:.0%}  EV={rec['ev_sek']:+,.0f} SEK")
1380:    elif "--avanza-status" in args:
1381:        from portfolio.avanza_client import get_portfolio_value, get_positions
1382:        positions = get_positions()
1383:        value = get_portfolio_value()
1384:        print(f"Portfolio value: {value:,.0f} SEK")
1385:        if positions:
1386:            for p in positions:
1387:                print(f"  {p}")
1388:    elif "--loop" in args:
1389:        idx = args.index("--loop")
1390:        override = int(args[idx + 1]) if idx + 1 < len(args) else None
1391:        loop(interval=override)
1392:    else:
1393:        run(force_report="--report" in args)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --line-number --no-heading -e \"subprocess|Popen|ThreadPoolExecutor|timeout|communicate\\(|wait\\(|kill\\(|terminate\\(|CLAUDECODE|telegram|atomic_write|append_jsonl|jsonl|Lock\\(|RLock\\(|Event\\(|sleep\\(|drawdown|circuit|heartbeat|prewarm|gpu|stderr|stdout|poll\\(|webhook|429|rate limit|Not logged in|nested session\" portfolio/main.py portfolio/agent_invocation.py portfolio/autonomous.py portfolio/trigger.py portfolio/market_timing.py portfolio/claude_gate.py portfolio/gpu_gate.py portfolio/health.py portfolio/alert_budget.py portfolio/llm_prewarmer.py portfolio/llm_calibration.py portfolio/llm_batch.py portfolio/llm_outcome_backfill.py portfolio/llm_probability_log.py portfolio/llama_server.py portfolio/multi_agent_layer2.py portfolio/perception_gate.py portfolio/focus_analysis.py portfolio/reporting.py portfolio/journal.py portfolio/journal_index.py portfolio/telegram_notifications.py portfolio/telegram_poller.py portfolio/digest.py portfolio/daily_digest.py portfolio/weekly_digest.py portfolio/reflection.py portfolio/regime_alerts.py portfolio/analyze.py portfolio/bigbet.py portfolio/prophecy.py portfolio/qwen3_signal.py portfolio/circuit_breaker.py portfolio/cumulative_tracker.py portfolio/decision_outcome_tracker.py" in Q:\fa-adv-2026-05-11
 succeeded in 742ms:
portfolio/llm_calibration.py:46:_DEFAULT_LOG = _BASE_DIR / "data" / "llm_probability_log.jsonl"
portfolio/llm_prewarmer.py:4:2026-05-11 Stage 3 Phase 1: minimal in-process prewarmer. Called from
portfolio/llm_prewarmer.py:7:``llama_server`` to swap to it synchronously inside the prewarm call —
portfolio/llm_prewarmer.py:14:  prewarmer cannot regress the working ministral/qwen3/fingpt path
portfolio/llm_prewarmer.py:22:- Chronos (``gpu_gate("chronos", timeout=30)``) is unaffected: this
portfolio/llm_prewarmer.py:23:  module never acquires ``gpu_gate`` itself. The win is that by the
portfolio/llm_prewarmer.py:25:  done — so Chronos's 30 s timeout no longer races a mid-flight swap.
portfolio/llm_prewarmer.py:33:``data/llm_rotation_state.jsonl`` after each prewarm attempt. On
portfolio/llm_prewarmer.py:35:prewarmed slot S at counter C doesn't redundantly prewarm S again
portfolio/llm_prewarmer.py:48:logger = logging.getLogger("portfolio.llm_prewarmer")
portfolio/llm_prewarmer.py:62:# State file: one JSONL line per prewarm attempt. Used to short-circuit
portfolio/llm_prewarmer.py:63:# duplicate prewarms across process restarts at the same counter.
portfolio/llm_prewarmer.py:66:STATE_FILE = DATA_DIR / "llm_rotation_state.jsonl"
portfolio/llm_prewarmer.py:93:    Therefore: prewarm slot index = ``(current_counter - 1) % 3``.
portfolio/llm_prewarmer.py:107:    file size — every prewarm pays this. We seek to the end and read
portfolio/llm_prewarmer.py:141:        # atomic_append_jsonl. We try it first.
portfolio/llm_prewarmer.py:154:        logger.warning("llm_prewarmer state read failed: %s", e)
portfolio/llm_prewarmer.py:158:def _write_state(counter: int, prewarmed_slot: str, server_slot: str,
portfolio/llm_prewarmer.py:162:        from portfolio.file_utils import atomic_append_jsonl
portfolio/llm_prewarmer.py:167:            "prewarmed_slot": prewarmed_slot,
portfolio/llm_prewarmer.py:173:        atomic_append_jsonl(STATE_FILE, entry)
portfolio/llm_prewarmer.py:175:        logger.warning("llm_prewarmer state write failed: %s", e)
portfolio/llm_prewarmer.py:182:    Used both to short-circuit prewarm when the target is already loaded
portfolio/llm_prewarmer.py:184:    see Fix A 2026-05-11 in ``prewarm_next_model``.
portfolio/llm_prewarmer.py:191:        logger.debug("llm_prewarmer load-check failed: %s", e)
portfolio/llm_prewarmer.py:198:    rather prewarm an already-loaded model than skip a needed prewarm).
portfolio/llm_prewarmer.py:203:def prewarm_next_model(current_counter: int) -> bool:
portfolio/llm_prewarmer.py:210:            yet, no useful prewarm possible.
portfolio/llm_prewarmer.py:213:        True if a prewarm query was actually dispatched.
portfolio/llm_prewarmer.py:214:        False if the prewarm was a no-op (already loaded / duplicate /
portfolio/llm_prewarmer.py:218:    A broken prewarmer cannot be allowed to regress the working LLM
portfolio/llm_prewarmer.py:223:        # function. We auto-skip when that's present so the real prewarmer
portfolio/llm_prewarmer.py:227:        # exercise the prewarmer directly call prewarm_next_model from
portfolio/llm_prewarmer.py:234:            logger.debug("llm_prewarmer skip: pytest detected, no-op")
portfolio/llm_prewarmer.py:241:            logger.debug("llm_prewarmer skip: counter=%d not positive", counter)
portfolio/llm_prewarmer.py:248:                "llm_prewarmer skip: no server mapping for slot=%s", next_slot,
portfolio/llm_prewarmer.py:262:        # the expected slot is still loaded. Any mismatch → force prewarm.
portfolio/llm_prewarmer.py:268:            and last.get("prewarmed_slot") == next_slot
portfolio/llm_prewarmer.py:273:                "llm_prewarmer skip: counter=%d slot=%s already prewarmed "
portfolio/llm_prewarmer.py:284:                "llm_prewarmer noop: slot=%s server=%s already loaded",
portfolio/llm_prewarmer.py:296:            "llm_prewarmer start: counter=%d slot=%s server=%s",
portfolio/llm_prewarmer.py:307:            # outcome is logged so we can see prewarm failures in the
portfolio/llm_prewarmer.py:310:                "llm_prewarmer query returned None: counter=%d slot=%s in %.1fs",
portfolio/llm_prewarmer.py:318:            "llm_prewarmer warmed: counter=%d slot=%s server=%s in %.1fs",
portfolio/llm_prewarmer.py:327:        # that the prewarmer NEVER raises.
portfolio/llm_prewarmer.py:329:            "llm_prewarmer unexpected failure: counter=%s err=%s",
portfolio/alert_budget.py:6:    3 = Emergency (bypasses budget — stop-loss, circuit breaker, crash)
portfolio/alert_budget.py:22:    """Token-bucket style alert rate limiter with priority bypass."""
portfolio/telegram_poller.py:14:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
portfolio/telegram_poller.py:17:logger = logging.getLogger("portfolio.telegram_poller")
portfolio/telegram_poller.py:19:INBOUND_LOG = Path(__file__).resolve().parent.parent / "data" / "telegram_inbound.jsonl"
portfolio/telegram_poller.py:29:POLLER_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "telegram_poller_state.json"
portfolio/telegram_poller.py:45:        config: full app config dict (with telegram.token, telegram.chat_id)
portfolio/telegram_poller.py:48:        self.token = config["telegram"]["token"]
portfolio/telegram_poller.py:49:        self.chat_id = str(config["telegram"]["chat_id"])
portfolio/telegram_poller.py:98:            atomic_write_json(
portfolio/telegram_poller.py:121:            time.sleep(5)
portfolio/telegram_poller.py:125:        params = {"timeout": 3, "allowed_updates": ["message"]}
portfolio/telegram_poller.py:130:            f"https://api.telegram.org/bot{self.token}/getUpdates",
portfolio/telegram_poller.py:132:            timeout=10,
portfolio/telegram_poller.py:182:                msg = None  # short-circuit out of the rest of the body
portfolio/telegram_poller.py:268:        """Persist one inbound message to data/telegram_inbound.jsonl.
portfolio/telegram_poller.py:289:            atomic_append_jsonl(INBOUND_LOG, entry)
portfolio/telegram_poller.py:324:        # file_utils helpers (load_json + atomic_write_json) rather than
portfolio/telegram_poller.py:328:        #      race against an external atomic_write_json rename mid-read on
portfolio/telegram_poller.py:361:        atomic_write_json(config_path, cfg)
portfolio/telegram_poller.py:375:                f"https://api.telegram.org/bot{self.token}/sendMessage",
portfolio/telegram_poller.py:382:                timeout=30,
portfolio/health.py:9:from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail
portfolio/health.py:17:_health_lock = threading.Lock()
portfolio/health.py:25:        state["last_heartbeat"] = datetime.now(UTC).isoformat()
portfolio/health.py:34:            # re-parsing invocations.jsonl on every call.
portfolio/health.py:41:        atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:61:        atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:64:def heartbeat() -> None:
portfolio/health.py:65:    """Touch only the last_heartbeat timestamp.
portfolio/health.py:72:    is misleading: the loop is alive, just waiting on the subprocess.
portfolio/health.py:85:        state["last_heartbeat"] = datetime.now(UTC).isoformat()
portfolio/health.py:86:        atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:92:# atomic_write_json on a heavily-loaded disk).
portfolio/health.py:96:class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
portfolio/health.py:97:    """Context manager that ticks heartbeat() every interval seconds.
portfolio/health.py:99:    Wraps long-blocking work (Layer 2 T2/T3 subprocess, autonomous decision
portfolio/health.py:102:    is auto-stopped on context exit; a 2s join timeout prevents shutdown
portfolio/health.py:106:        with heartbeat_keepalive():
portfolio/health.py:109:    The first beat is synchronous (so a fast-returning subprocess gets at
portfolio/health.py:110:    least one heartbeat even if it finishes before the first interval).
portfolio/health.py:120:        self._stop = threading.Event()
portfolio/health.py:123:    def __enter__(self) -> "heartbeat_keepalive":
portfolio/health.py:127:            heartbeat()
portfolio/health.py:129:            logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)
portfolio/health.py:132:            target=self._run, daemon=True, name="heartbeat-keepalive",
portfolio/health.py:140:            self._thread.join(timeout=2.0)
portfolio/health.py:143:        # Event.wait returns True when set (stop signaled), False on timeout.
portfolio/health.py:144:        # So we tick on each timeout and exit on the first True.
portfolio/health.py:145:        while not self._stop.wait(self._interval):
portfolio/health.py:147:                heartbeat()
portfolio/health.py:149:                logger.warning("heartbeat_keepalive tick failed", exc_info=True)
portfolio/health.py:153:    """Check if the loop heartbeat is stale.
portfolio/health.py:157:    hb = state.get("last_heartbeat")
portfolio/health.py:163:        logger.warning("check_staleness: corrupt last_heartbeat=%r", hb)
portfolio/health.py:180:    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
portfolio/health.py:185:    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
portfolio/health.py:187:        invocations_file = DATA_DIR / "invocations.jsonl"
portfolio/health.py:188:        last_ts = last_jsonl_entry(invocations_file, field="ts")
portfolio/health.py:196:            atomic_write_json(HEALTH_FILE, wb_state)
portfolio/health.py:247:            atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:251:            atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:298:        atomic_write_json(HEALTH_FILE, state)
portfolio/health.py:345:        "heartbeat_age_seconds": round(age, 1),
portfolio/health.py:358:    # Include circuit breaker status if data_collector has been imported
portfolio/health.py:361:        summary["circuit_breakers"] = {
portfolio/health.py:377:    signal_log = DATA_DIR / "signal_log.jsonl"
portfolio/health.py:382:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
portfolio/health.py:383:    entries = load_jsonl_tail(signal_log, max_entries=50)
portfolio/health.py:429:    signal_log = DATA_DIR / "signal_log.jsonl"
portfolio/health.py:431:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
portfolio/health.py:432:    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
portfolio/journal_index.py:19:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/gpu_gate.py:6:Uses a threading lock for in-process concurrency (ThreadPoolExecutor workers)
portfolio/gpu_gate.py:7:plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.
portfolio/gpu_gate.py:10:- Reactive: ``gpu_gate()`` calls ``_try_break_stale_lock()`` when another
portfolio/gpu_gate.py:12:- Background: a daemon thread (lazily spawned on first ``gpu_gate()`` call)
portfolio/gpu_gate.py:16:  See ``docs/plans/2026-05-03-gpu-gate-sweeper.md``.
portfolio/gpu_gate.py:21:import subprocess
portfolio/gpu_gate.py:27:logger = logging.getLogger("portfolio.gpu_gate")
portfolio/gpu_gate.py:29:# In-process lock — prevents ThreadPoolExecutor workers from racing
portfolio/gpu_gate.py:30:_THREAD_LOCK = threading.Lock()
portfolio/gpu_gate.py:34:_GPU_LOCK_FILE = _GPU_LOCK_DIR / ".gpu_lock"
portfolio/gpu_gate.py:37:# Stale-lock sweeper daemon (2026-05-03). Module-level singleton so subprocess
portfolio/gpu_gate.py:40:_SWEEPER_LOCK = threading.Lock()
portfolio/gpu_gate.py:47:        proc = subprocess.run(
portfolio/gpu_gate.py:48:            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
portfolio/gpu_gate.py:50:            capture_output=True, text=True, timeout=5,
portfolio/gpu_gate.py:52:        if proc.returncode == 0 and proc.stdout.strip():
portfolio/gpu_gate.py:53:            parts = [p.strip() for p in proc.stdout.strip().split(",")]
portfolio/gpu_gate.py:59:                    "gpu_util_pct": int(parts[3]),
portfolio/gpu_gate.py:110:    - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
portfolio/gpu_gate.py:147:            time.sleep(_SWEEPER_INTERVAL_SECONDS)
portfolio/gpu_gate.py:158:    Lazily called from ``gpu_gate()`` so:
portfolio/gpu_gate.py:160:      ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
portfolio/gpu_gate.py:172:                name="gpu-gate-sweeper",
portfolio/gpu_gate.py:180:def gpu_gate(model_name: str, timeout: float = 60):
portfolio/gpu_gate.py:184:    1. threading.Lock for in-process concurrency (ThreadPoolExecutor workers)
portfolio/gpu_gate.py:189:        timeout: max seconds to wait for lock
portfolio/gpu_gate.py:198:    deadline = time.time() + timeout
portfolio/gpu_gate.py:200:    # Layer 1: In-process thread lock (prevents ThreadPoolExecutor races)
portfolio/gpu_gate.py:202:    thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
portfolio/gpu_gate.py:204:        logger.warning("GPU thread-lock timeout (%ss) for %s", timeout, model_name)
portfolio/gpu_gate.py:235:                time.sleep(1.0)
portfolio/gpu_gate.py:239:            logger.warning("GPU file-lock timeout (%ss) — held by %s", timeout, info.get("model", "?"))
portfolio/gpu_gate.py:248:                model_name, vram["used_mb"], vram["free_mb"], vram["total_mb"], vram["gpu_util_pct"],
portfolio/telegram_notifications.py:12:logger = logging.getLogger("portfolio.telegram")
portfolio/telegram_notifications.py:35:def send_telegram(msg, config):
portfolio/telegram_notifications.py:40:    if config.get("telegram", {}).get("mute_all", False):
portfolio/telegram_notifications.py:41:        logger.info("[mute_all] Skipping send_telegram")
portfolio/telegram_notifications.py:44:    # via direct requests.post. To re-enable, set telegram.layer1_messages: true.
portfolio/telegram_notifications.py:45:    if not config.get("telegram", {}).get("layer1_messages", False):
portfolio/telegram_notifications.py:52:    token = config["telegram"]["token"]
portfolio/telegram_notifications.py:53:    chat_id = config["telegram"]["chat_id"]
portfolio/telegram_notifications.py:55:        f"https://api.telegram.org/bot{token}/sendMessage",
portfolio/telegram_notifications.py:58:        timeout=30,
portfolio/telegram_notifications.py:75:                f"https://api.telegram.org/bot{token}/sendMessage",
portfolio/telegram_notifications.py:78:                timeout=30,
portfolio/decision_outcome_tracker.py:3:Reads data/layer2_decisions.jsonl, checks if enough time has elapsed for
portfolio/decision_outcome_tracker.py:5:records to data/layer2_decision_outcomes.jsonl.
portfolio/decision_outcome_tracker.py:14:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
portfolio/decision_outcome_tracker.py:19:DECISIONS_FILE = BASE_DIR / "data" / "layer2_decisions.jsonl"
portfolio/decision_outcome_tracker.py:20:OUTCOMES_FILE = BASE_DIR / "data" / "layer2_decision_outcomes.jsonl"
portfolio/decision_outcome_tracker.py:29:    decisions = load_jsonl(DECISIONS_FILE)
portfolio/decision_outcome_tracker.py:35:    existing_outcomes = load_jsonl(OUTCOMES_FILE)
portfolio/decision_outcome_tracker.py:99:                atomic_append_jsonl(OUTCOMES_FILE, outcome)
portfolio/circuit_breaker.py:5:  OPEN    — API is failing, requests blocked until recovery timeout
portfolio/circuit_breaker.py:14:logger = logging.getLogger("portfolio.circuit_breaker")
portfolio/circuit_breaker.py:24:    """Thread-safe circuit breaker for a single data source."""
portfolio/circuit_breaker.py:26:    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
portfolio/circuit_breaker.py:27:                 max_recovery_timeout: int = 300):
portfolio/circuit_breaker.py:30:        self.recovery_timeout = recovery_timeout
portfolio/circuit_breaker.py:31:        self._base_recovery_timeout = recovery_timeout
portfolio/circuit_breaker.py:32:        self._max_recovery_timeout = max_recovery_timeout
portfolio/circuit_breaker.py:36:        self._lock = threading.Lock()
portfolio/circuit_breaker.py:51:                self.recovery_timeout = self._base_recovery_timeout
portfolio/circuit_breaker.py:61:                # BUG-245: Exponential backoff — double timeout on each failed
portfolio/circuit_breaker.py:64:                prev_timeout = self.recovery_timeout
portfolio/circuit_breaker.py:65:                self.recovery_timeout = min(
portfolio/circuit_breaker.py:66:                    self.recovery_timeout * 2, self._max_recovery_timeout
portfolio/circuit_breaker.py:71:                    self.name, self._failure_count, self.recovery_timeout, prev_timeout,
portfolio/circuit_breaker.py:92:                if elapsed >= self.recovery_timeout:
portfolio/circuit_breaker.py:109:        """Return current circuit breaker status."""
portfolio/circuit_breaker.py:134:            self.recovery_timeout = self._base_recovery_timeout
portfolio/cumulative_tracker.py:3:Logs hourly price snapshots to data/price_snapshots_hourly.jsonl and computes
portfolio/cumulative_tracker.py:13:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
portfolio/cumulative_tracker.py:20:SNAPSHOTS_FILE = DATA_DIR / "price_snapshots_hourly.jsonl"
portfolio/cumulative_tracker.py:52:    atomic_append_jsonl(SNAPSHOTS_FILE, entry)
portfolio/cumulative_tracker.py:105:        snapshots = load_jsonl(SNAPSHOTS_FILE)
portfolio/qwen3_signal.py:4:Falls back to subprocess if server unavailable.
portfolio/qwen3_signal.py:13:import subprocess
portfolio/qwen3_signal.py:17:from portfolio.gpu_gate import gpu_gate
portfolio/qwen3_signal.py:19:from portfolio.subprocess_utils import kill_orphaned_llama, run_safe
portfolio/qwen3_signal.py:28:def _extract_json_from_stdout(stdout):
portfolio/qwen3_signal.py:29:    """Extract JSON (object or array) from subprocess stdout."""
portfolio/qwen3_signal.py:30:    if not stdout:
portfolio/qwen3_signal.py:32:    text = stdout.strip()
portfolio/qwen3_signal.py:61:    """Call Qwen3-8B, preferring persistent llama-server, with subprocess fallback."""
portfolio/qwen3_signal.py:76:    # the latter case, the subprocess fallback below would cold-start an 8B
portfolio/qwen3_signal.py:87:        logger.warning("qwen3: abstaining — Plex transcoding and VRAM <7168MB; skipping subprocess fallback")
portfolio/qwen3_signal.py:90:    # Fallback: subprocess (cold start)
portfolio/qwen3_signal.py:91:    logger.info("llama-server unavailable for qwen3, falling back to subprocess")
portfolio/qwen3_signal.py:107:            timeout=240,
portfolio/qwen3_signal.py:109:    except subprocess.TimeoutExpired as e:
portfolio/qwen3_signal.py:110:        stderr_text = e.stderr[-500:] if e.stderr else "(no stderr)"
portfolio/qwen3_signal.py:111:        logger.error("Qwen3 subprocess timed out after 240s — stderr: %s", stderr_text)
portfolio/qwen3_signal.py:114:        raise RuntimeError(f"Qwen3 failed: {result.stderr[-500:]}")
portfolio/qwen3_signal.py:115:    payload = _extract_json_from_stdout(result.stdout)
portfolio/qwen3_signal.py:117:        raise RuntimeError(f"Qwen3 returned invalid JSON: {result.stdout[-500:]}")
portfolio/qwen3_signal.py:122:    """Call Qwen3-8B inference subprocess in batch mode.
portfolio/qwen3_signal.py:146:        timeout=60 + 30 * len(contexts),  # 60s base + 30s per ticker (extended for deeper reasoning)
portfolio/qwen3_signal.py:153:        raise RuntimeError(f"Qwen3 batch failed: {result.stderr[-500:]}")
portfolio/qwen3_signal.py:154:    payload = _extract_json_from_stdout(result.stdout)
portfolio/qwen3_signal.py:173:    with gpu_gate("qwen3", timeout=300) as acquired:
portfolio/qwen3_signal.py:175:            logger.warning("GPU gate timeout — returning HOLD")
portfolio/qwen3_signal.py:178:        from portfolio.gpu_gate import get_vram_usage
portfolio/journal.py:10:from portfolio.file_utils import atomic_write_text, load_json
portfolio/journal.py:14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/journal.py:570:                    atomic_write_text(CONTEXT_FILE, md)
portfolio/journal.py:582:    atomic_write_text(CONTEXT_FILE, md)
portfolio/prophecy.py:16:from portfolio.file_utils import atomic_write_json, load_json
portfolio/prophecy.py:73:    atomic_write_json(PROPHECY_FILE, data)
portfolio/reporting.py:11:from portfolio.file_utils import atomic_write_json
portfolio/reporting.py:27:# 2026-04-22: escalate persistent silent failures to critical_errors.jsonl.
portfolio/reporting.py:808:    atomic_write_json(AGENT_SUMMARY_FILE, summary)
portfolio/reporting.py:840:        atomic_write_json(SIGNAL_STATE_SINCE_FILE, payload)
portfolio/reporting.py:1018:    atomic_write_json(COMPACT_SUMMARY_FILE, compact)
portfolio/reporting.py:1201:    atomic_write_json(TIER1_FILE, t1)
portfolio/reporting.py:1314:    atomic_write_json(TIER2_FILE, t2)
portfolio/claude_gate.py:8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
portfolio/claude_gate.py:9:Doing so bypasses the kill switch, rate limiter, and invocation tracking.
portfolio/claude_gate.py:20:        timeout=180,
portfolio/claude_gate.py:31:import subprocess
portfolio/claude_gate.py:36:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
portfolio/claude_gate.py:51:INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"
portfolio/claude_gate.py:53:# session must see. Intentionally separate from claude_invocations.jsonl so
portfolio/claude_gate.py:57:CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
portfolio/claude_gate.py:61:# subprocesses can all call invoke_claude in parallel. The Claude CLI is
portfolio/claude_gate.py:62:# expensive (sonnet ~30s, opus ~3-5min) and the rate limiter is per-day,
portfolio/claude_gate.py:70:_invoke_lock = threading.Lock()
portfolio/claude_gate.py:99:    Prevents the "nested session" error when invoking ``claude -p`` from a
portfolio/claude_gate.py:103:    env.pop("CLAUDECODE", None)
portfolio/claude_gate.py:112:# invocation between 2026-03-27 and 2026-04-13 to print "Not logged in —
portfolio/claude_gate.py:113:# Please run /login" on stdout and exit 0. Nothing surfaced the failure
portfolio/claude_gate.py:118:_AUTH_ERROR_MARKERS = ("Not logged in", "Please run /login", "Invalid API key")
portfolio/claude_gate.py:121:# unresolved critical_errors.jsonl entries verbatim at session start. Those
portfolio/claude_gate.py:122:# entries CONTAIN the literal string "Not logged in", so the substring scan
portfolio/claude_gate.py:137:# log entries (`["ts": ..., "message": "...Not logged in..."]`); whitespace
portfolio/claude_gate.py:158:    # like `- Not logged in` that tests pre-empt by checking line[0]).
portfolio/claude_gate.py:168:    """Append a critical error to ``data/critical_errors.jsonl``.
portfolio/claude_gate.py:195:        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
portfolio/claude_gate.py:198:        logger.error("Failed to write critical_errors.jsonl: %s", e)
portfolio/claude_gate.py:203:    """Scan subprocess output for claude-CLI auth errors and escalate.
portfolio/claude_gate.py:206:    CRITICAL level AND records the failure to ``critical_errors.jsonl`` so
portfolio/claude_gate.py:213:    critical-level log + critical_errors.jsonl entry + invocation-log
portfolio/claude_gate.py:248:                    f"claude CLI subprocess printed {marker!r} — OAuth session "
portfolio/claude_gate.py:266:        atomic_append_jsonl(INVOCATIONS_LOG, entry)
portfolio/claude_gate.py:275:    for entry in load_jsonl(INVOCATIONS_LOG):
portfolio/claude_gate.py:282:# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
portfolio/claude_gate.py:290:# Fix: explicitly Popen with a new process group/session so we can kill the
portfolio/claude_gate.py:294:def _popen_kwargs_for_tree_kill() -> dict:
portfolio/claude_gate.py:295:    """Return Popen kwargs that allow tree-killing the spawned process."""
portfolio/claude_gate.py:297:        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
portfolio/claude_gate.py:301:def _kill_process_tree(proc: subprocess.Popen, *, label: str = "claude") -> None:
portfolio/claude_gate.py:302:    """Kill a Popen process and all of its descendants. Best-effort:
portfolio/claude_gate.py:303:    falls back to proc.kill() if the platform-specific path fails.
portfolio/claude_gate.py:305:    if proc.poll() is not None:
portfolio/claude_gate.py:311:            # /F = force, /PID = the parent PID. Capture stderr to keep
portfolio/claude_gate.py:312:            # logs clean if the process already exited between poll() and here.
portfolio/claude_gate.py:313:            res = subprocess.run(
portfolio/claude_gate.py:315:                capture_output=True, timeout=5,
portfolio/claude_gate.py:319:                    "%s tree kill via taskkill returned %d (stderr=%r) — "
portfolio/claude_gate.py:320:                    "falling back to proc.kill()",
portfolio/claude_gate.py:321:                    label, res.returncode, res.stderr.decode("utf-8", "replace")[:200],
portfolio/claude_gate.py:323:                proc.kill()
portfolio/claude_gate.py:329:                logger.warning("%s killpg(%d) failed: %s — falling back to proc.kill()", label, pid, e)
portfolio/claude_gate.py:330:                proc.kill()
portfolio/claude_gate.py:334:            "%s tree kill encountered unexpected error: %s — proc.kill()",
portfolio/claude_gate.py:338:            proc.kill()
portfolio/claude_gate.py:341:                "%s proc.kill() also failed after tree-kill error: %s — "
portfolio/claude_gate.py:347:def _run_with_tree_kill(
portfolio/claude_gate.py:350:    timeout: float,
portfolio/claude_gate.py:355:    """Run a subprocess with proper timeout + tree-kill cleanup.
portfolio/claude_gate.py:358:        (returncode, stdout, stderr, timed_out)
portfolio/claude_gate.py:360:    On timeout, kills the entire process tree (not just the direct child)
portfolio/claude_gate.py:364:    proc = subprocess.Popen(
portfolio/claude_gate.py:366:        stdout=subprocess.PIPE,
portfolio/claude_gate.py:367:        stderr=subprocess.PIPE,
portfolio/claude_gate.py:368:        stdin=subprocess.DEVNULL,
portfolio/claude_gate.py:372:        **_popen_kwargs_for_tree_kill(),
portfolio/claude_gate.py:375:        stdout, stderr = proc.communicate(timeout=timeout)
portfolio/claude_gate.py:376:        return proc.returncode, stdout or "", stderr or "", False
portfolio/claude_gate.py:377:    except subprocess.TimeoutExpired:
portfolio/claude_gate.py:379:                       label, timeout, proc.pid)
portfolio/claude_gate.py:383:            stdout, stderr = proc.communicate(timeout=5)
portfolio/claude_gate.py:384:        except subprocess.TimeoutExpired:
portfolio/claude_gate.py:387:                proc.kill()
portfolio/claude_gate.py:388:            stdout, stderr = "", ""
portfolio/claude_gate.py:389:        return -1, stdout or "", stderr or "", True
portfolio/claude_gate.py:402:    timeout: int = 180,
portfolio/claude_gate.py:413:        timeout: Subprocess timeout in seconds.
portfolio/claude_gate.py:414:        cwd: Working directory for the subprocess.  Defaults to the repo root.
portfolio/claude_gate.py:493:        # 8-worker ticker pool / metals fast-tick / signal subprocesses
portfolio/claude_gate.py:495:        # A-IN-2: tree-killing helper for grandchild cleanup on timeout.
portfolio/claude_gate.py:497:            rc, _stdout, _stderr, timed_out = _run_with_tree_kill(
portfolio/claude_gate.py:499:                timeout=timeout,
portfolio/claude_gate.py:505:            status = "timeout"
portfolio/claude_gate.py:510:            # printing "Not logged in" when OAuth/keychain auth can't be read
portfolio/claude_gate.py:513:            # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan stdout
portfolio/claude_gate.py:514:            # and stderr SEPARATELY rather than concatenating without a
portfolio/claude_gate.py:516:            # the last stdout line ("...stdoutNot logged in"), defeating
portfolio/claude_gate.py:519:            stdout_hit = detect_auth_failure(
portfolio/claude_gate.py:520:                _stdout or "", caller,
portfolio/claude_gate.py:523:            stderr_hit = detect_auth_failure(
portfolio/claude_gate.py:524:                _stderr or "", caller,
portfolio/claude_gate.py:526:            ) if not stdout_hit else False
portfolio/claude_gate.py:527:            if stdout_hit or stderr_hit:
portfolio/claude_gate.py:558:    timeout: int = 60,
portfolio/claude_gate.py:562:    Unlike ``invoke_claude()``, this captures stdout and returns the text
portfolio/claude_gate.py:600:            rc, stdout, _stderr, timed_out = _run_with_tree_kill(
portfolio/claude_gate.py:602:                timeout=timeout,
portfolio/claude_gate.py:608:            status = "timeout"
portfolio/claude_gate.py:611:            text = stdout
portfolio/claude_gate.py:615:            # stdout and stderr because the CLI can write "Not logged in"
portfolio/claude_gate.py:619:            # the marker into the last stdout line. See invoke_claude for
portfolio/claude_gate.py:621:            stdout_hit = detect_auth_failure(
portfolio/claude_gate.py:622:                stdout or "", caller,
portfolio/claude_gate.py:625:            stderr_hit = detect_auth_failure(
portfolio/claude_gate.py:626:                _stderr or "", caller,
portfolio/claude_gate.py:628:            ) if not stdout_hit else False
portfolio/claude_gate.py:629:            if stdout_hit or stderr_hit:
portfolio/claude_gate.py:658:    entries = load_jsonl(INVOCATIONS_LOG)
portfolio/bigbet.py:8:import subprocess
portfolio/bigbet.py:15:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
portfolio/bigbet.py:44:    atomic_write_json(STATE_FILE, state)
portfolio/bigbet.py:47:EVAL_LOG_FILE = DATA_DIR / "bigbet_gate_log.jsonl"
portfolio/bigbet.py:169:        # P2 (2026-04-17): PF_HEADLESS_AGENT=1 so the Claude subprocess skips
portfolio/bigbet.py:175:        result = subprocess.run(
portfolio/bigbet.py:179:            timeout=30,
portfolio/bigbet.py:183:        output = result.stdout.strip()
portfolio/bigbet.py:186:        # bypasses claude_gate's invoke_claude wrapper, so a "Not logged in"
portfolio/bigbet.py:187:        # stdout with exit 0 would otherwise be passed to the response parser
portfolio/bigbet.py:189:        # critical_errors.jsonl. Escalate first, return safe default if hit.
portfolio/bigbet.py:191:        scan = f"{output}\n{result.stderr or ''}"
portfolio/bigbet.py:200:    except subprocess.TimeoutExpired:
portfolio/bigbet.py:202:        logger.warning("BIG BET L2: timeout after %.1fs", elapsed)
portfolio/bigbet.py:212:        atomic_append_jsonl(EVAL_LOG_FILE, {
portfolio/bigbet.py:456:                _send_telegram(msg, config)
portfolio/bigbet.py:458:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:481:                _send_telegram(msg, config)
portfolio/bigbet.py:483:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:504:                _send_telegram(msg, config)
portfolio/bigbet.py:506:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:599:                _send_telegram(msg, config)
portfolio/bigbet.py:601:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:618:def _send_telegram(msg, config):
portfolio/focus_analysis.py:16:from portfolio.file_utils import load_json, load_jsonl
portfolio/focus_analysis.py:41:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/focus_analysis.py:86:    entries = load_jsonl(JOURNAL_FILE, limit=400)
portfolio/regime_alerts.py:11:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/regime_alerts.py:16:REGIME_HISTORY_FILE = DATA_DIR / "regime_history.jsonl"
portfolio/regime_alerts.py:39:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:97:    atomic_append_jsonl(REGIME_HISTORY_FILE, entry)
portfolio/regime_alerts.py:111:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:144:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:152:    Also logs the message to telegram_messages.jsonl.
portfolio/analyze.py:10:import subprocess
portfolio/analyze.py:15:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/analyze.py:18:from portfolio.telegram_notifications import send_telegram as _shared_send_telegram
portfolio/analyze.py:24:    """Return env dict without CLAUDECODE to avoid nested-session errors.
portfolio/analyze.py:26:    P2 (2026-04-17): sets PF_HEADLESS_AGENT=1 so the Claude subprocess
portfolio/analyze.py:31:    errors — it just doesn't block the subprocess path on a fake-user
portfolio/analyze.py:35:    env.pop("CLAUDECODE", None)
portfolio/analyze.py:40:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/analyze.py:44:ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log.jsonl"
portfolio/analyze.py:45:WATCH_LOG_FILE = DATA_DIR / "watch_log.jsonl"
portfolio/analyze.py:64:    all_entries = load_jsonl(JOURNAL_FILE)
portfolio/analyze.py:231:    """Append to analysis_log.jsonl."""
portfolio/analyze.py:233:        atomic_append_jsonl(ANALYSIS_LOG_FILE, {
portfolio/analyze.py:243:def _send_telegram(msg, config):
portfolio/analyze.py:244:    _shared_send_telegram(msg, config)
portfolio/analyze.py:282:        result = subprocess.run(
portfolio/analyze.py:286:            timeout=120,
portfolio/analyze.py:288:            stdin=subprocess.DEVNULL,
portfolio/analyze.py:291:        output = result.stdout.strip()
portfolio/analyze.py:297:        scan = f"{output}\n{result.stderr or ''}"
portfolio/analyze.py:304:            if result.stderr:
portfolio/analyze.py:305:                print(f"stderr: {result.stderr[:500]}")
portfolio/analyze.py:318:            _send_telegram(tg_msg, config)
portfolio/analyze.py:323:    except subprocess.TimeoutExpired:
portfolio/analyze.py:405:        atomic_append_jsonl(WATCH_LOG_FILE, event)
portfolio/analyze.py:639:        _send_telegram(
portfolio/analyze.py:654:                time.sleep(interval)
portfolio/analyze.py:686:                        _send_telegram(msg, config)
portfolio/analyze.py:702:                        _send_telegram(msg, config)
portfolio/analyze.py:718:                        _send_telegram(msg, config)
portfolio/analyze.py:746:                    result = subprocess.run(
portfolio/analyze.py:750:                        timeout=60,
portfolio/analyze.py:752:                        stdin=subprocess.DEVNULL,
portfolio/analyze.py:755:                    output = result.stdout.strip()
portfolio/analyze.py:787:                                    _send_telegram(tg_msg, config)
portfolio/analyze.py:804:                        if result.stderr:
portfolio/analyze.py:805:                            print(f"  stderr: {result.stderr[:200]}")
portfolio/analyze.py:807:                except subprocess.TimeoutExpired:
portfolio/analyze.py:831:            time.sleep(interval)
portfolio/analyze.py:844:            _send_telegram(
portfolio/multi_agent_layer2.py:7:    2. Risk Agent: portfolio state, exposure, drawdown, stops
portfolio/multi_agent_layer2.py:24:import subprocess
portfolio/multi_agent_layer2.py:43:        "timeout": 120,
portfolio/multi_agent_layer2.py:47:        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
portfolio/multi_agent_layer2.py:54:        "timeout": 90,
portfolio/multi_agent_layer2.py:64:        "timeout": 90,
portfolio/multi_agent_layer2.py:130:) -> list[subprocess.Popen]:
portfolio/multi_agent_layer2.py:133:    Returns list of Popen processes. Caller must wait for them.
portfolio/multi_agent_layer2.py:143:    agent_env.pop("CLAUDECODE", None)
portfolio/multi_agent_layer2.py:147:    # subprocesses with no interactive stdin, same as invoke_agent. Without
portfolio/multi_agent_layer2.py:150:    # hang asking "How would you like to proceed?" until specialist_timeout_s.
portfolio/multi_agent_layer2.py:168:            proc = subprocess.Popen(
portfolio/multi_agent_layer2.py:171:                stdout=log_fh,
portfolio/multi_agent_layer2.py:172:                stderr=subprocess.STDOUT,
portfolio/multi_agent_layer2.py:186:    procs: list[subprocess.Popen],
portfolio/multi_agent_layer2.py:187:    timeout: int = 150,
portfolio/multi_agent_layer2.py:194:    deadline = time.time() + timeout
portfolio/multi_agent_layer2.py:200:            proc.wait(timeout=remaining)
portfolio/multi_agent_layer2.py:205:        except subprocess.TimeoutExpired:
portfolio/multi_agent_layer2.py:207:            proc.kill()
portfolio/multi_agent_layer2.py:208:            proc.wait(timeout=5)
portfolio/reflection.py:4:summary stored in data/reflections.jsonl. Layer 2 reads the latest
portfolio/reflection.py:19:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/reflection.py:24:REFLECTIONS_FILE = DATA_DIR / "reflections.jsonl"
portfolio/reflection.py:27:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/reflection.py:93:    entries = load_jsonl(JOURNAL_FILE, limit=100)
portfolio/reflection.py:154:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio/reflection.py:204:    atomic_append_jsonl(REFLECTIONS_FILE, reflection)
portfolio/reflection.py:228:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio/market_timing.py:304:def should_skip_gpu(ticker, config=None, now=None):
portfolio/market_timing.py:313:    gpu_cfg = (config or {}).get("gpu_signals", {})
portfolio/market_timing.py:314:    if not gpu_cfg.get("skip_stocks_offhours", True):
portfolio/market_timing.py:317:    pre_buffer = gpu_cfg.get("pre_market_buffer_min", 30)
portfolio/market_timing.py:318:    post_buffer = gpu_cfg.get("post_market_buffer_min", 15)
portfolio/autonomous.py:18:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/autonomous.py:28:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/autonomous.py:34:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/autonomous.py:35:DECISIONS_FILE = DATA_DIR / "layer2_decisions.jsonl"
portfolio/autonomous.py:100:    prev_entries = load_jsonl(JOURNAL_FILE, limit=5)
portfolio/autonomous.py:151:    atomic_append_jsonl(JOURNAL_FILE, journal_entry)
portfolio/autonomous.py:166:    atomic_append_jsonl(DECISIONS_FILE, decision_log)
portfolio/autonomous.py:170:        msg = _build_telegram(
portfolio/autonomous.py:180:            logger.exception("Autonomous telegram send failed")
portfolio/autonomous.py:484:def _build_telegram(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:492:        return _build_telegram_mode_b(
portfolio/autonomous.py:498:    return _build_telegram_mode_a(
portfolio/autonomous.py:505:def _build_telegram_mode_a(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:658:def _build_telegram_mode_b(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:825:        from portfolio.file_utils import atomic_write_json
portfolio/autonomous.py:826:        atomic_write_json(THROTTLE_FILE, data)
portfolio/llm_probability_log.py:45:from portfolio.file_utils import atomic_append_jsonl
portfolio/llm_probability_log.py:50:_PROB_LOG = _DATA_DIR / "llm_probability_log.jsonl"
portfolio/llm_probability_log.py:159:        atomic_append_jsonl(log_path or _PROB_LOG, entry)
portfolio/llm_outcome_backfill.py:1:"""Outcome backfill for `data/llm_probability_log.jsonl`.
portfolio/llm_outcome_backfill.py:5:and writes the outcome into a companion `data/llm_probability_outcomes.jsonl`
portfolio/llm_outcome_backfill.py:25:hourly price snapshot jsonl — the same path the existing outcome_tracker
portfolio/llm_outcome_backfill.py:35:from portfolio.file_utils import atomic_append_jsonl
portfolio/llm_outcome_backfill.py:42:_PROB_LOG = _BASE_DIR / "data" / "llm_probability_log.jsonl"
portfolio/llm_outcome_backfill.py:43:_OUTCOMES = _BASE_DIR / "data" / "llm_probability_outcomes.jsonl"
portfolio/llm_outcome_backfill.py:145:      log_path: override for the input jsonl (tests).
portfolio/llm_outcome_backfill.py:146:      outcomes_path: override for the output jsonl.
portfolio/llm_outcome_backfill.py:250:            atomic_append_jsonl(outcomes_path, outcome_row)
portfolio/llama_server.py:21:import subprocess
portfolio/llama_server.py:91:_thread_lock = threading.Lock()
portfolio/llama_server.py:92:_local_proc = None       # Popen if this process started the server
portfolio/llama_server.py:101:            result = subprocess.run(
portfolio/llama_server.py:103:                capture_output=True, text=True, timeout=10,
portfolio/llama_server.py:106:            for line in result.stdout.splitlines():
portfolio/llama_server.py:115:                    subprocess.run(
portfolio/llama_server.py:117:                        capture_output=True, timeout=10,
portfolio/llama_server.py:120:            result = subprocess.run(
portfolio/llama_server.py:122:                capture_output=True, text=True, timeout=10,
portfolio/llama_server.py:124:            for pid_str in result.stdout.split():
portfolio/llama_server.py:128:                        os.kill(pid, 9)
portfolio/llama_server.py:141:            result = subprocess.run(
portfolio/llama_server.py:143:                capture_output=True, text=True, timeout=5,
portfolio/llama_server.py:147:            return "llama-server" in result.stdout.lower()
portfolio/llama_server.py:172:                    subprocess.run(
portfolio/llama_server.py:174:                        capture_output=True, timeout=10,
portfolio/llama_server.py:177:                    os.kill(pid, 9)
portfolio/llama_server.py:178:                time.sleep(1)
portfolio/llama_server.py:212:        r = _requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=2)
portfolio/llama_server.py:229:            _local_proc.terminate()
portfolio/llama_server.py:230:            _local_proc.wait(timeout=10)
portfolio/llama_server.py:233:                _local_proc.kill()
portfolio/llama_server.py:255:        result = subprocess.run(
portfolio/llama_server.py:256:            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
portfolio/llama_server.py:259:            timeout=2,
portfolio/llama_server.py:264:        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
portfolio/llama_server.py:294:        result = subprocess.run(
portfolio/llama_server.py:298:            timeout=2,
portfolio/llama_server.py:300:        if result.returncode == 0 and "plex transcoder" in result.stdout.lower():
portfolio/llama_server.py:311:# bypass query_llama_server (subprocess fallbacks) get identical protection.
portfolio/llama_server.py:319:    (subprocess fallbacks in `qwen3_signal._call_qwen3`, `ministral_signal._call_model`,
portfolio/llama_server.py:326:    subprocess that could evict Plex's NVENC context.
portfolio/llama_server.py:346:    Replaces the hardcoded `time.sleep(4)` that used to follow _stop_server()
portfolio/llama_server.py:362:    free-VRAM floor to 7168 MB (>=7 GB free) and extends the timeout to 30 s.
portfolio/llama_server.py:380:        time.sleep(max_wait)
portfolio/llama_server.py:385:        time.sleep(0.1)
portfolio/llama_server.py:390:            time.sleep(remaining)
portfolio/llama_server.py:422:    # 2026-04-10 (perf/llama-swap-reduction): replaced `time.sleep(4)` with an
portfolio/llama_server.py:435:    # its subprocess inference path. Slower than HTTP but never racing.
portfolio/llama_server.py:456:        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
portfolio/llama_server.py:460:            if proc.poll() is not None:
portfolio/llama_server.py:469:            time.sleep(1)
portfolio/llama_server.py:471:        proc.kill()
portfolio/llama_server.py:488:def _acquire_file_lock(timeout=300):
portfolio/llama_server.py:491:    Timeout must exceed the HTTP query timeout (240s) to prevent callers
portfolio/llama_server.py:492:    from falling back to subprocess while the server is still handling a
portfolio/llama_server.py:496:    deadline = time.time() + timeout
portfolio/llama_server.py:511:                    result = subprocess.run(
portfolio/llama_server.py:513:                        capture_output=True, text=True, timeout=5,
portfolio/llama_server.py:515:                    if str(lock_pid) not in result.stdout:
portfolio/llama_server.py:519:                    os.kill(lock_pid, 0)  # raises if dead
portfolio/llama_server.py:524:            time.sleep(1)
portfolio/llama_server.py:525:    logger.warning("llama-server file lock timeout (%ds)", timeout)
portfolio/llama_server.py:543:    Returns completion text or None (caller should fall back to subprocess).
portfolio/llama_server.py:554:        fh = _acquire_file_lock(timeout=300)
portfolio/llama_server.py:599:        timeout=240,
portfolio/llama_server.py:624:        fh = _acquire_file_lock(timeout=300)
portfolio/digest.py:14:from portfolio.file_utils import atomic_write_json as _atomic_write_json
portfolio/digest.py:18:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/digest.py:24:INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio/digest.py:27:SIGNAL_LOG_FILE = DATA_DIR / "signal_log.jsonl"
portfolio/digest.py:52:    _atomic_write_json(_DIGEST_STATE_FILE, state)
portfolio/digest.py:55:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/digest.py:59:    from portfolio.file_utils import load_jsonl_tail
portfolio/digest.py:65:    # BUG-190: Use tail read for efficiency (invocations.jsonl grows unbounded)
portfolio/digest.py:66:    entries = load_jsonl_tail(INVOCATIONS_FILE, max_entries=500)
portfolio/digest.py:100:    journal = load_jsonl_tail(JOURNAL_FILE, max_entries=500)
portfolio/digest.py:122:    signal_entries = load_jsonl_tail(SIGNAL_LOG_FILE, max_entries=500)
portfolio/digest.py:239:        heartbeat_age = health.get("heartbeat_age_seconds", 0)
portfolio/digest.py:240:        if heartbeat_age < 300:
portfolio/digest.py:242:        elif heartbeat_age < 3600:
portfolio/digest.py:243:            uptime_str = f"{heartbeat_age / 60:.0f}m stale"
portfolio/digest.py:245:            uptime_str = f"{heartbeat_age / 3600:.1f}h stale"
portfolio/daily_digest.py:19:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/daily_digest.py:42:    from portfolio.file_utils import atomic_write_json
portfolio/daily_digest.py:47:    atomic_write_json(_DAILY_DIGEST_STATE_FILE, state)
portfolio/weekly_digest.py:16:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/weekly_digest.py:23:SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
portfolio/weekly_digest.py:24:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/weekly_digest.py:27:def _load_jsonl(path, since=None):
portfolio/weekly_digest.py:29:    entries = load_jsonl(path)
portfolio/weekly_digest.py:154:    signal_entries = _load_jsonl(SIGNAL_LOG, since=week_ago)
portfolio/weekly_digest.py:168:    journal_entries = _load_jsonl(JOURNAL_FILE, since=week_ago)
portfolio/weekly_digest.py:272:    token = config.get("telegram", {}).get("token")
portfolio/weekly_digest.py:273:    chat_id = config.get("telegram", {}).get("chat_id")
portfolio/weekly_digest.py:280:    log_file = DATA_DIR / "telegram_messages.jsonl"
portfolio/weekly_digest.py:286:    atomic_append_jsonl(log_file, entry)
portfolio/weekly_digest.py:290:        from portfolio.telegram_notifications import send_telegram
portfolio/weekly_digest.py:291:        result = send_telegram(msg, config)
portfolio/llm_batch.py:29:_lock = threading.Lock()
portfolio/llm_batch.py:36:# sentiment_ab_log.jsonl entry by sentiment.flush_ab_log() post-cycle.
portfolio/llm_batch.py:167:    Called once after ThreadPoolExecutor completes in main.py.
portfolio/llm_batch.py:302:    # 2026-05-11 (feat/llm-prewarmer Stage 3 Phase 1): pre-warm the NEXT
portfolio/llm_batch.py:303:    # LLM in rotation right now, while we still hold no Chronos/gpu_gate.
portfolio/llm_batch.py:305:    # required model is already resident — Chronos's gpu_gate("chronos",
portfolio/llm_batch.py:306:    # timeout=30) no longer races a mid-flight cold swap. The prewarmer
portfolio/llm_batch.py:308:    # outer try/except as a second backstop because a broken prewarmer
portfolio/llm_batch.py:311:        from portfolio.llm_prewarmer import prewarm_next_model
portfolio/llm_batch.py:312:        prewarm_next_model(_ss._full_llm_cycle_count)
portfolio/llm_batch.py:314:        logger.warning("llm prewarmer dispatch failed (non-fatal): %s", e)
portfolio/llm_batch.py:501:        # to scan in tail/grep contexts (loop_out tail, telegram digests).
portfolio/trigger.py:14:off-hours) provides the "heartbeat" via classify_tier(), but only when
portfolio/trigger.py:25:from portfolio.file_utils import atomic_write_json, load_json
portfolio/trigger.py:126:    atomic_write_json(STATE_FILE, state)
portfolio/agent_invocation.py:1:"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""
portfolio/agent_invocation.py:7:import subprocess
portfolio/agent_invocation.py:16:from portfolio.file_utils import atomic_append_jsonl, last_jsonl_entry, load_jsonl
portfolio/agent_invocation.py:18:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/agent_invocation.py:24:INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio/agent_invocation.py:25:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio/agent_invocation.py:26:TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
portfolio/agent_invocation.py:30:# BUG-214: Drawdown circuit breaker thresholds.
portfolio/agent_invocation.py:41:# for timeout enforcement when `_agent_start` (monotonic) gets poisoned.
portfolio/agent_invocation.py:42:# The clamp alone could silently disable the P1B T1 timeout check; this
portfolio/agent_invocation.py:47:_agent_timeout = 900  # per-invocation timeout (set from tier config)
portfolio/agent_invocation.py:51:_telegram_ts_before = None  # last telegram timestamp before agent started
portfolio/agent_invocation.py:67:# subprocess.poll() and enforces the per-tier wall-clock timeout, but it
portfolio/agent_invocation.py:69:# (333-918s violations 2026-05-01..04), a T1 subprocess that finishes
portfolio/agent_invocation.py:71:# inflating ``duration_s`` in invocations.jsonl and delaying the kill
portfolio/agent_invocation.py:77:_completion_lock = threading.Lock()
portfolio/agent_invocation.py:79:_watchdog_stop = threading.Event()
portfolio/agent_invocation.py:94:        # Event.wait(timeout) returns True if the event was set during
portfolio/agent_invocation.py:95:        # the wait — ie shutdown. Returning False means the timeout
portfolio/agent_invocation.py:97:        if _watchdog_stop.wait(_COMPLETION_WATCHDOG_INTERVAL_S):
portfolio/agent_invocation.py:136:def _stop_completion_watchdog(timeout_s: float = 1.0) -> None:
portfolio/agent_invocation.py:141:    exit. ``timeout_s`` is intentionally short — the worst case is a
portfolio/agent_invocation.py:149:        _watchdog_thread.join(timeout=timeout_s)
portfolio/agent_invocation.py:164:    from portfolio.file_utils import atomic_write_json
portfolio/agent_invocation.py:165:    atomic_write_json(_STACK_OVERFLOW_FILE, {
portfolio/agent_invocation.py:175:    1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
portfolio/agent_invocation.py:176:    2: {"max_turns": 40, "timeout": 600, "label": "SIGNAL ANALYSIS"},
portfolio/agent_invocation.py:177:    3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
portfolio/agent_invocation.py:251:    atomic_append_jsonl(INVOCATIONS_FILE, entry)
portfolio/agent_invocation.py:285:    Scans layer2_journal.jsonl (most-recent-first) for entries mentioning
portfolio/agent_invocation.py:293:        entries = load_jsonl(JOURNAL_FILE)
portfolio/agent_invocation.py:333:def _last_jsonl_ts(path):
portfolio/agent_invocation.py:336:    Uses efficient tail-read via last_jsonl_entry() (reads last 4KB only).
portfolio/agent_invocation.py:338:    return last_jsonl_entry(path, field="ts")
portfolio/agent_invocation.py:341:def _safe_last_jsonl_ts(path, label):
portfolio/agent_invocation.py:344:        return _last_jsonl_ts(path)
portfolio/agent_invocation.py:353:    P2B (2026-04-17): yesterday's 2026-04-16T13:45:45 critical_errors.jsonl
portfolio/agent_invocation.py:361:    disabled the P1B timeout path — `elapsed > _agent_timeout` can never
portfolio/agent_invocation.py:396:    P1-3 (2026-05-02 last-followups): the timeout-kill path
portfolio/agent_invocation.py:397:    (``_kill_overrun_agent``) used to forget the dead subprocess without
portfolio/agent_invocation.py:400:    printed "Not logged in" before getting stuck on a network retry
portfolio/agent_invocation.py:401:    would surface as ``timeout`` (not ``auth_error``) and never land in
portfolio/agent_invocation.py:402:    ``critical_errors.jsonl``. That asymmetry is the same class of silent
portfolio/agent_invocation.py:416:            ``"layer2_t2_timeout"``). Tier and trigger context are pulled
portfolio/agent_invocation.py:450:    be called from ``check_agent_completion``. Previously the timeout
portfolio/agent_invocation.py:453:    invoked 16:04:58 with timeout=120s completed at 16:15:01 = 603s).
portfolio/agent_invocation.py:455:    Logs the trigger with status="timeout" and clears ``_agent_proc`` /
portfolio/agent_invocation.py:460:    so the silent-auth-failure detector covers the timeout path too — not
portfolio/agent_invocation.py:489:        result = subprocess.run(
portfolio/agent_invocation.py:497:                result.stderr.decode(errors="replace").strip(),
portfolio/agent_invocation.py:503:        _agent_proc.kill()
portfolio/agent_invocation.py:505:        _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
portfolio/agent_invocation.py:506:    except subprocess.TimeoutExpired:
portfolio/agent_invocation.py:519:    # for auth-error markers before forgetting the dead subprocess. Done
portfolio/agent_invocation.py:525:    auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
portfolio/agent_invocation.py:531:        "timeout",
portfolio/agent_invocation.py:540:    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
portfolio/agent_invocation.py:541:    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
portfolio/agent_invocation.py:563:    timeout = tier_cfg["timeout"]
portfolio/agent_invocation.py:568:    # observe a freshly-killed _agent_proc.poll() exit code and write a
portfolio/agent_invocation.py:570:    # writing its "timeout" row — exactly the double-log the lock was added
portfolio/agent_invocation.py:575:        if _agent_proc and _agent_proc.poll() is None:
portfolio/agent_invocation.py:578:            # can't cause a negative elapsed that silently skips the timeout.
portfolio/agent_invocation.py:580:            if elapsed > _agent_timeout:
portfolio/agent_invocation.py:623:    # BUG-214: Drawdown circuit breaker — first-ever automated risk gate on
portfolio/agent_invocation.py:629:    # 50%+ drawdown could continue trading if anything in the check threw
portfolio/agent_invocation.py:631:    # on a missing dd dict key). The fail-safe direction for a circuit
portfolio/agent_invocation.py:642:    _drawdown_context = ""
portfolio/agent_invocation.py:644:        from portfolio.risk_management import check_drawdown
portfolio/agent_invocation.py:646:        logger.error("DRAWDOWN BLOCK: check_drawdown unavailable (%s) — fail-safe block", e)
portfolio/agent_invocation.py:647:        _log_trigger(reasons, "blocked_drawdown_unavailable", tier=tier)
portfolio/agent_invocation.py:654:            dd = check_drawdown(str(pf_path), max_drawdown_pct=_DRAWDOWN_WARN_PCT)
portfolio/agent_invocation.py:655:            if dd["current_drawdown_pct"] > _DRAWDOWN_BLOCK_PCT:
portfolio/agent_invocation.py:657:                    "DRAWDOWN BLOCK: %s portfolio at %.1f%% drawdown (>%.0f%%) — skipping invocation",
portfolio/agent_invocation.py:658:                    label, dd["current_drawdown_pct"], _DRAWDOWN_BLOCK_PCT,
portfolio/agent_invocation.py:660:                _log_trigger(reasons, f"blocked_drawdown_{label.lower()}", tier=tier)
portfolio/agent_invocation.py:662:            if dd["current_drawdown_pct"] > _DRAWDOWN_WARN_PCT:
portfolio/agent_invocation.py:664:                    "DRAWDOWN WARNING: %s portfolio at %.1f%% drawdown (peak %.0f, current %.0f SEK)",
portfolio/agent_invocation.py:665:                    label, dd["current_drawdown_pct"], dd["peak_value"], dd["current_value"],
portfolio/agent_invocation.py:667:            _drawdown_context += (
portfolio/agent_invocation.py:668:                f"\n[DRAWDOWN {label}] {dd['current_drawdown_pct']:.1f}% from peak "
portfolio/agent_invocation.py:687:    # and Bold gets short-circuited before the multi-agent / subprocess spawn
portfolio/agent_invocation.py:688:    # (saves ~600s of T2 subprocess + Claude tokens for a decision that
portfolio/agent_invocation.py:700:    #      fail-OPEN — unlike drawdown, cooldowns are soft constraints and a
portfolio/agent_invocation.py:767:                # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
portfolio/agent_invocation.py:768:                # layer2.specialist_timeout_s) to avoid blocking the main loop.
portfolio/agent_invocation.py:770:                specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
portfolio/agent_invocation.py:771:                results = wait_for_specialists(procs, timeout=specialist_timeout)
portfolio/agent_invocation.py:786:    # BUG-214: Append drawdown context so Layer 2 sees current risk levels.
portfolio/agent_invocation.py:787:    if _drawdown_context:
portfolio/agent_invocation.py:788:        prompt += "\n\n[RISK DATA]" + _drawdown_context
portfolio/agent_invocation.py:814:        # invocation ("Not logged in" to stdout, exit 0). Commit b4bb57d
portfolio/agent_invocation.py:841:        # Strip Claude Code session markers to avoid "nested session" error
portfolio/agent_invocation.py:844:        agent_env.pop("CLAUDECODE", None)
portfolio/agent_invocation.py:848:        # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
portfolio/agent_invocation.py:850:        # when it finds unresolved critical_errors.jsonl entries. The agent
portfolio/agent_invocation.py:852:        # makes it hit the tier timeout with zero work done. The CLAUDE.md
portfolio/agent_invocation.py:856:        _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
portfolio/agent_invocation.py:857:        _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
portfolio/agent_invocation.py:858:        _agent_proc = subprocess.Popen(
portfolio/agent_invocation.py:861:            stdout=log_fh,
portfolio/agent_invocation.py:862:            stderr=subprocess.STDOUT,
portfolio/agent_invocation.py:869:        _agent_timeout = timeout
portfolio/agent_invocation.py:893:        # unwriteable (atomic_write_json handles the happy path; any
portfolio/agent_invocation.py:896:            from portfolio.file_utils import atomic_write_json, load_json
portfolio/agent_invocation.py:907:            atomic_write_json(health_path, health)
portfolio/agent_invocation.py:911:            "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
portfolio/agent_invocation.py:912:            tier, _agent_proc.pid, max_turns, timeout,
portfolio/agent_invocation.py:916:        # timeout fires within ~30 s of the real budget even when the
portfolio/agent_invocation.py:1030:        from portfolio.file_utils import atomic_write_json
portfolio/agent_invocation.py:1033:        atomic_write_json(DATA_DIR / 'fishing_context.json', context)
portfolio/agent_invocation.py:1041:            from portfolio.file_utils import atomic_write_json
portfolio/agent_invocation.py:1042:            atomic_write_json(DATA_DIR / 'fishing_context.json', {
portfolio/agent_invocation.py:1101:    invocations.jsonl row, the other returns ``None`` because
portfolio/agent_invocation.py:1106:        agent is still in progress and under its timeout):
portfolio/agent_invocation.py:1109:          "timeout" (P1B, 2026-04-17), or "stack_overflow"
portfolio/agent_invocation.py:1110:        * ``exit_code`` — int or None (None on timeout-kill path)
portfolio/agent_invocation.py:1115:        * ``telegram_sent`` — bool
portfolio/agent_invocation.py:1128:    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
portfolio/agent_invocation.py:1133:    exit_code = _agent_proc.poll()
portfolio/agent_invocation.py:1135:        # Still running. P1B (2026-04-17): enforce the wall-clock timeout
portfolio/agent_invocation.py:1138:        # no new triggers came through (yesterday: T1 timeout=120s ran
portfolio/agent_invocation.py:1142:        if _agent_timeout and elapsed > _agent_timeout:
portfolio/agent_invocation.py:1147:                "status": "timeout",
portfolio/agent_invocation.py:1153:                "telegram_sent": False,
portfolio/agent_invocation.py:1165:    # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
portfolio/agent_invocation.py:1167:        journal_ts_after = _last_jsonl_ts(JOURNAL_FILE)
portfolio/agent_invocation.py:1177:    # BUG-97: Same protection for telegram file
portfolio/agent_invocation.py:1179:        telegram_ts_after = _last_jsonl_ts(TELEGRAM_FILE)
portfolio/agent_invocation.py:1181:        logger.warning("Failed to read telegram timestamp after agent completion")
portfolio/agent_invocation.py:1182:        telegram_ts_after = None
portfolio/agent_invocation.py:1183:    telegram_sent = (
portfolio/agent_invocation.py:1184:        _telegram_ts_before is not None
portfolio/agent_invocation.py:1185:        and telegram_ts_after is not None
portfolio/agent_invocation.py:1186:        and telegram_ts_after != _telegram_ts_before
portfolio/agent_invocation.py:1193:    if _telegram_ts_before is None:
portfolio/agent_invocation.py:1194:        telegram_sent = False
portfolio/agent_invocation.py:1198:    # in" to stdout — that's exactly the 3-week silent Layer 2 outage that
portfolio/agent_invocation.py:1200:    # spawning the subprocess, so we only scan output from this invocation.
portfolio/agent_invocation.py:1203:    # ``_scan_agent_log_for_auth_failure`` so the timeout-kill path
portfolio/agent_invocation.py:1206:    # the same asymmetry the timeout path used to have.
portfolio/agent_invocation.py:1217:    elif journal_written and telegram_sent:
portfolio/agent_invocation.py:1228:        # completion-path and timeout-path dicts have symmetric shape.
portfolio/agent_invocation.py:1234:        "telegram_sent": telegram_sent,
portfolio/agent_invocation.py:1246:        "telegram_sent": telegram_sent,
portfolio/agent_invocation.py:1249:        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
portfolio/agent_invocation.py:1256:            new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
portfolio/agent_invocation.py:1263:    # (cooldowns, loss escalation, position rate limits).
portfolio/agent_invocation.py:1267:        "Agent completed: status=%s exit=%d duration=%.1fs tier=%s journal=%s telegram=%s",
portfolio/agent_invocation.py:1268:        status, exit_code, duration_s, _agent_tier, journal_written, telegram_sent,
portfolio/agent_invocation.py:1276:                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
portfolio/agent_invocation.py:1286:                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
portfolio/agent_invocation.py:1337:    _telegram_ts_before = None
portfolio/agent_invocation.py:1351:        dict with keys: total, success, incomplete, failed, timeout,
portfolio/agent_invocation.py:1355:    Codex P2 #4 follow-up (2026-04-17): "timeout" and "auth_error" were
portfolio/agent_invocation.py:1356:    being dropped entirely by the status filter. Before P1B, timeouts
portfolio/agent_invocation.py:1358:    P1B check_agent_completion enforces timeout every cycle — these
portfolio/agent_invocation.py:1361:    completion_rate honest (timeouts count as failures for rate).
portfolio/agent_invocation.py:1363:    entries = load_jsonl(INVOCATIONS_FILE)
portfolio/agent_invocation.py:1370:    timeout = 0
portfolio/agent_invocation.py:1373:    tracked_statuses = ("success", "incomplete", "failed", "timeout", "auth_error")
portfolio/agent_invocation.py:1401:        elif entry_status == "timeout":
portfolio/agent_invocation.py:1402:            timeout += 1
portfolio/agent_invocation.py:1413:        "timeout": timeout,
portfolio/main.py:5:- shared_state.py — mutable globals, caching, rate limiters
portfolio/main.py:13:- telegram_notifications.py — Telegram send/escape/alert
portfolio/main.py:16:- message_throttle.py — analysis message rate limiting
portfolio/main.py:17:- agent_invocation.py — Layer 2 Claude Code subprocess
portfolio/main.py:31:from portfolio.file_utils import atomic_write_json, atomic_write_text, load_json
portfolio/main.py:174:    _atomic_write_json,
portfolio/main.py:229:from portfolio.telegram_notifications import (  # noqa: E402, F401
portfolio/main.py:233:    send_telegram,
portfolio/main.py:314:        from portfolio.accuracy_stats import maybe_prewarm_dashboard_accuracy
portfolio/main.py:315:        _track("dashboard_accuracy_prewarm", maybe_prewarm_dashboard_accuracy)
portfolio/main.py:317:        logger.warning("dashboard accuracy prewarm import failed: %s", e_pw)
portfolio/main.py:358:    from portfolio.file_utils import prune_jsonl
portfolio/main.py:360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl", "claude_invocations.jsonl"):
portfolio/main.py:362:            prune_jsonl(DATA_DIR / name, max_entries=5000)
portfolio/main.py:367:        report.post_cycle_results["jsonl_prune"] = len(_prune_failures) == 0
portfolio/main.py:390:                from portfolio.file_utils import atomic_write_json as _awj
portfolio/main.py:434:    # still observes subprocess completion and enforces the per-tier
portfolio/main.py:435:    # wall-clock timeout — see docs/plans/2026-05-05-l2-completion-watchdog.md.
portfolio/main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed
portfolio/main.py:515:                if extra.get("_gpu_signals_skipped"):
portfolio/main.py:564:    # BUG-178: Add timeout to prevent indefinite hangs from stuck tickers.
portfolio/main.py:579:    #   tickers (network timeouts, yfinance blocking).
portfolio/main.py:581:    #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
portfolio/main.py:594:    #   detection, journal, and telegram. Loop contract's own cycle_dur
portfolio/main.py:600:    #   instrumentation-timeout.md for the full rationale.
portfolio/main.py:609:    # which blocks the loop when threads hang past the timeout.
portfolio/main.py:610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
portfolio/main.py:616:        for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
portfolio/main.py:647:            "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
portfolio/main.py:687:        # the sentiment_ab_log.jsonl rows for this cycle. Must run AFTER
portfolio/main.py:705:    # last_signal state on its 180 s timeout, but cycles that stay under 180 s
portfolio/main.py:831:        # subprocess; autonomous fallback = bounded but not instant) in a
portfolio/main.py:832:        # heartbeat keepalive. update_health() (the normal heartbeat write)
portfolio/main.py:839:        from portfolio.health import heartbeat_keepalive
portfolio/main.py:847:                with heartbeat_keepalive():
portfolio/main.py:856:            with heartbeat_keepalive():
portfolio/main.py:867:    # Big Bet detection — can invoke a 30s Claude subprocess per qualifying
portfolio/main.py:869:    # cap. Wrapped in keepalive so heartbeat stays fresh across multi-minute
portfolio/main.py:875:            from portfolio.health import heartbeat_keepalive
portfolio/main.py:876:            with heartbeat_keepalive():
portfolio/main.py:882:    # Claude gate subprocess (portfolio/iskbets.py:invoke_layer2_gate). With
portfolio/main.py:883:    # 5 Tier-1 tickers configured the worst case is ~150s of subprocess work,
portfolio/main.py:884:    # well past the 300s heartbeat threshold when stacked with bigbet+L2.
portfolio/main.py:886:        from portfolio.health import heartbeat_keepalive
portfolio/main.py:888:        with heartbeat_keepalive():
portfolio/main.py:985:    atomic_write_json(_CRASH_COUNTER_FILE, {
portfolio/main.py:1038:def _crash_sleep():
portfolio/main.py:1047:    time.sleep(delay)
portfolio/main.py:1052:#     _crash_sleep()
portfolio/main.py:1062:# plan's recommended `time.sleep(min(2 ** n_failures, 30))` floor as the
portfolio/main.py:1091:        _crash_sleep()
portfolio/main.py:1103:        time.sleep(floor)
portfolio/main.py:1124:        time.sleep(remaining)
portfolio/main.py:1143:    # Check if previous loop crashed (stale heartbeat)
portfolio/main.py:1144:    heartbeat_file = DATA_DIR / "heartbeat.txt"
portfolio/main.py:1145:    if heartbeat_file.exists():
portfolio/main.py:1147:            last_beat = datetime.fromisoformat(heartbeat_file.read_text().strip())
portfolio/main.py:1151:                msg = f"_LOOP RESTARTED_ — previous heartbeat was {age_min}m ago. Possible crash."
portfolio/main.py:1160:            logger.warning("Failed to check heartbeat staleness: %s", e)
portfolio/main.py:1180:        from portfolio.telegram_poller import TelegramPoller
portfolio/main.py:1192:            atomic_write_text(DATA_DIR / "heartbeat.txt", datetime.now(UTC).isoformat())
portfolio/main.py:1194:                initial_report.heartbeat_updated = True
portfolio/main.py:1248:            atomic_write_text(DATA_DIR / "heartbeat.txt", datetime.now(UTC).isoformat())
portfolio/main.py:1250:                report.heartbeat_updated = True
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve confirmed the repo is available locally and started tracing the hot paths. Next I’m drilling into the Layer 2 subprocess code, GPU/prewarm flow, Telegram paths, and health/journal logic line-by-line to separate real blockers from noise.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/agent_invocation.py" | Select-Object -First 260' in Q:\fa-adv-2026-05-11
 succeeded in 1238ms:
1:"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""
3:import logging
4:import os
5:import platform
6:import shutil
7:import subprocess
8:import threading
9:import time
10:from contextlib import suppress
11:from datetime import UTC, datetime
12:from pathlib import Path
14:from portfolio.api_utils import load_config as _load_config
15:from portfolio.claude_gate import detect_auth_failure
16:from portfolio.file_utils import atomic_append_jsonl, last_jsonl_entry, load_jsonl
17:from portfolio.message_store import send_or_store
18:from portfolio.telegram_notifications import escape_markdown_v1
20:logger = logging.getLogger("portfolio.agent")
22:BASE_DIR = Path(__file__).resolve().parent.parent
23:DATA_DIR = BASE_DIR / "data"
24:INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
25:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
26:TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
27:PATIENT_PORTFOLIO = DATA_DIR / "portfolio_state.json"
28:BOLD_PORTFOLIO = DATA_DIR / "portfolio_state_bold.json"
30:# BUG-214: Drawdown circuit breaker thresholds.
31:# Advisory at WARN level, hard-block at BLOCK level.
32:# User accepts 10-20% knockout risk; only de-risk at 50%+.
33:_DRAWDOWN_WARN_PCT = 20.0
34:_DRAWDOWN_BLOCK_PCT = 50.0
36:_agent_proc = None
37:_agent_log = None
38:_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
39:_agent_start = 0
40:# P2B follow-up (Codex P2 #2, 2026-04-17): fallback wall-clock timestamp
41:# for timeout enforcement when `_agent_start` (monotonic) gets poisoned.
42:# The clamp alone could silently disable the P1B T1 timeout check; this
43:# fallback lets _safe_elapsed_s() recover a plausible elapsed from wall
44:# clock so the hung agent still gets killed. Always set alongside
45:# _agent_start so the pair are in sync.
46:_agent_start_wall = 0.0
47:_agent_timeout = 900  # per-invocation timeout (set from tier config)
48:_agent_tier = None  # tier of the currently running agent
49:_agent_reasons = None  # trigger reasons for the current invocation
50:_journal_ts_before = None  # last journal timestamp before agent started
51:_telegram_ts_before = None  # last telegram timestamp before agent started
53:# BUG-219: Transaction counts at invoke time — used by check_agent_completion()
54:# to detect new trades and call record_trade() for overtrading prevention.
55:# PR-R4-4: record_trade() was never called from production code; this wires it.
56:_patient_txn_count_before = 0
57:_bold_txn_count_before = 0
59:# Stack overflow detection — exit code 3221225794 = Windows STATUS_STACK_OVERFLOW (0xC00000FD)
60:_STACK_OVERFLOW_EXIT_CODE = 3221225794
61:_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
62:_STACK_OVERFLOW_FILE = DATA_DIR / "stack_overflow_counter.json"
64:# 2026-05-05 (item 3a of dashboard-noise-followups, see
65:# docs/plans/2026-05-05-l2-completion-watchdog.md): completion-detection
66:# watchdog. ``check_agent_completion`` is the only path that observes
67:# subprocess.poll() and enforces the per-tier wall-clock timeout, but it
68:# was called only once per ``main.run()`` cycle. When the cycle bloats
69:# (333-918s violations 2026-05-01..04), a T1 subprocess that finishes
70:# at its real 120s budget is not noticed for up to 6 more minutes —
71:# inflating ``duration_s`` in invocations.jsonl and delaying the kill
72:# of a hung agent past its real budget. The daemon thread below runs
73:# the same check every 30s independent of ``run()``'s cadence; the
74:# lock serialises with the main-thread call site so the two cannot
75:# race on ``_agent_proc`` / ``_agent_start`` state.
76:_COMPLETION_WATCHDOG_INTERVAL_S = 30
77:_completion_lock = threading.Lock()
78:_watchdog_thread: threading.Thread | None = None
79:_watchdog_stop = threading.Event()
82:def _completion_watchdog() -> None:
83:    """Daemon thread body: poll completion every 30 s while not stopped.
85:    Each tick takes ``_completion_lock`` and calls
86:    ``_check_agent_completion_locked`` directly so the main-thread call
87:    via ``check_agent_completion`` and this watchdog tick share one
88:    critical section. Failures inside the tick are logged and swallowed
89:    — the watchdog must never die from a transient I/O error or it
90:    silently regresses to the pre-fix state where the main loop is the
91:    only completion observer.
92:    """
93:    while not _watchdog_stop.is_set():
94:        # Event.wait(timeout) returns True if the event was set during
95:        # the wait — ie shutdown. Returning False means the timeout
96:        # elapsed normally, so we tick.
97:        if _watchdog_stop.wait(_COMPLETION_WATCHDOG_INTERVAL_S):
98:            return
99:        try:
100:            with _completion_lock:
101:                _check_agent_completion_locked()
102:        except Exception as e:  # noqa: BLE001 — never let the watchdog die
103:            logger.warning("completion watchdog tick failed: %s", e)
106:def _ensure_completion_watchdog() -> None:
107:    """Start the daemon watchdog if it is not already running.
109:    Idempotent: the spawn happens at most once per process under normal
110:    operation. If the previous thread died (uncaught exception escaping
111:    the ``except Exception`` above is impossible, but a thread.start
112:    failure or interpreter restart between calls could leave the global
113:    pointing at a dead thread), spawn a fresh one. Resets the stop
114:    event so a successor process that imports this module after a
115:    SIGTERM can still arm a new watchdog.
117:    Uses ``_completion_lock`` to make the is-alive-check + spawn atomic
118:    so concurrent callers cannot both pass the check and both spawn (a
119:    race exposed by tests that drive start/stop concurrently — in
120:    production the lazy-start happens once at the end of try_invoke_agent,
121:    which is itself serialised by the main loop).
122:    """
123:    global _watchdog_thread
124:    with _completion_lock:
125:        if _watchdog_thread is not None and _watchdog_thread.is_alive():
126:            return
127:        _watchdog_stop.clear()
128:        _watchdog_thread = threading.Thread(
129:            target=_completion_watchdog,
130:            name="L2CompletionWatchdog",
131:            daemon=True,
132:        )
133:        _watchdog_thread.start()
136:def _stop_completion_watchdog(timeout_s: float = 1.0) -> None:
137:    """Signal the watchdog to exit and wait briefly for it.
139:    Used by tests to keep xdist parallel runs hermetic; production code
140:    relies on ``daemon=True`` to terminate the thread at interpreter
141:    exit. ``timeout_s`` is intentionally short — the worst case is a
142:    sleeping thread that wakes within ``_COMPLETION_WATCHDOG_INTERVAL_S``
143:    ticks, but ``_watchdog_stop.set()`` interrupts that wait
144:    immediately.
145:    """
146:    global _watchdog_thread
147:    _watchdog_stop.set()
148:    if _watchdog_thread is not None:
149:        _watchdog_thread.join(timeout=timeout_s)
150:    _watchdog_thread = None
153:def _load_stack_overflow_counter() -> int:
154:    """Load persisted stack overflow counter. Returns 0 if missing/corrupt."""
155:    from portfolio.file_utils import load_json
156:    data = load_json(_STACK_OVERFLOW_FILE)
157:    if data and isinstance(data.get("count"), int):
158:        return data["count"]
159:    return 0
162:def _save_stack_overflow_counter(count: int) -> None:
163:    """Persist stack overflow counter to survive loop restarts."""
164:    from portfolio.file_utils import atomic_write_json
165:    atomic_write_json(_STACK_OVERFLOW_FILE, {
166:        "count": count,
167:        "updated": datetime.now(UTC).isoformat(),
168:    })
171:_consecutive_stack_overflows = _load_stack_overflow_counter()
173:# Per-tier configuration
174:TIER_CONFIG = {
175:    1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
176:    2: {"max_turns": 40, "timeout": 600, "label": "SIGNAL ANALYSIS"},
177:    3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
178:}
181:def _build_tier_prompt(tier, reasons):
182:    """Build a tier-specific prompt for the Claude Code agent."""
183:    reason_str = ", ".join(reasons[:5])
185:    playbook = "docs/TRADING_PLAYBOOK.md"
187:    if tier == 1:
188:        return (
189:            "You are the Layer 2 trading agent (QUICK CHECK). "
190:            f"Trigger: {reason_str}. "
191:            f"Read {playbook} for trading rules, then data/layer2_context.md "
192:            "then data/agent_context_t1.json. "
193:            "This is a routine check. Confirm held positions are OK (check ATR stops). "
194:            "If no positions are held, briefly assess macro state. "
195:            "Write a brief journal entry and send a short Telegram message. "
196:            "Do NOT analyze all tickers — focus only on held positions and macro headline."
197:        )
198:    elif tier == 2:
199:        return (
200:            "You are the Layer 2 trading agent (SIGNAL ANALYSIS). "
201:            f"Trigger: {reason_str}. "
202:            "If data/trading_insights.md exists, read it first for recent signal performance context. "
203:            f"Read {playbook} for trading rules, then data/layer2_context.md, "
204:            "then data/agent_context_t2.json, "
205:            "data/portfolio_state.json, and data/portfolio_state_bold.json. "
206:            "Analyze triggered tickers and held positions. Decide for BOTH strategies. "
207:            "Write journal entry and send Telegram per the playbook instructions."
208:        )
209:    else:
210:        # Tier 3 — full review
211:        return (
212:            "You are the Layer 2 trading agent. "
213:            "If data/trading_insights.md exists, read it first for recent signal performance context. "
214:            f"FIRST read {playbook} for trading rules. "
215:            "Then read data/layer2_context.md (your memory from previous invocations). "
216:            "Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), "
217:            "data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json "
218:            "(Bold portfolio). Follow the playbook to analyze, decide, and act "
219:            "for BOTH strategies independently. Compare your previous theses and prices with "
220:            "current data — were you right? Always write a journal entry and send a Telegram message."
221:        )
224:def _extract_ticker(reasons):
225:    """Extract the primary ticker from trigger reasons.
227:    Looks for common ticker patterns like 'XAG-USD', 'BTC-USD', 'NVDA'.
228:    Falls back to 'XAG-USD' if no ticker found.
229:    """
230:    import re
231:    for r in reasons:
232:        # Match patterns like XAG-USD, BTC-USD, ETH-USD
233:        m = re.search(r'\b([A-Z]{2,5}-USD)\b', r)
234:        if m:
235:            return m.group(1)
236:        # Match stock tickers like NVDA, PLTR
237:        m = re.search(r'\b([A-Z]{2,5})\b(?:\s+flipped|\s+crossed|\s+broke)', r)
238:        if m:
239:            return m.group(1)
240:    return "XAG-USD"  # default to silver
243:def _log_trigger(reasons, status, tier=None):
244:    entry = {
245:        "ts": datetime.now(UTC).isoformat(),
246:        "reasons": reasons,
247:        "status": status,
248:    }
249:    if tier is not None:
250:        entry["tier"] = tier
251:    atomic_append_jsonl(INVOCATIONS_FILE, entry)
254:def _load_guard_warnings():
255:    """Read trade_guard_warnings from agent_summary.json.
257:    P1-12 (2026-05-02): the trade-guards pre-execution gate consumes the
258:    warnings already computed by reporting.py and stored in agent_summary.
259:    Reading them here (rather than recomputing) keeps the gate consistent
260:    with what Layer 2 sees in its prompt context, and is much cheaper.
262:    Returns a dict shaped like trade_guards.get_all_guard_warnings():
263:        {"warnings": [...], "summary": "..."}
264:    Defaults to empty/no_summary when agent_summary is missing or has
265:    no trade_guard_warnings field — caller treats that as "no blocks".
266:    """
267:    from portfolio.file_utils import load_json
268:    summary_path = DATA_DIR / "agent_summary.json"
269:    summary = load_json(summary_path, default=None)
270:    if not isinstance(summary, dict):
271:        return {"warnings": [], "summary": "no_summary"}
272:    guard_block = summary.get("trade_guard_warnings")
273:    if not isinstance(guard_block, dict):
274:        return {"warnings": [], "summary": "no_warnings"}
275:    # Normalize: ensure required keys exist so callers don't have to .get()
276:    return {
277:        "warnings": guard_block.get("warnings", []) or [],
278:        "summary": guard_block.get("summary", ""),
279:    }
282:def _build_decision_feedback(ticker, max_entries=5):
283:    """Build recent-decision feedback for the trigger ticker.
285:    Scans layer2_journal.jsonl (most-recent-first) for entries mentioning
286:    *ticker* in the trigger string or the tickers dict.  Returns a formatted
287:    block that Layer 2 can use to calibrate against its own prior calls, or
288:    an empty string when no relevant history exists.
290:    Token budget: ≤15 lines.  Never fails the invocation on error.
291:    """
292:    try:
293:        entries = load_jsonl(JOURNAL_FILE)
294:    except Exception:
295:        return ""
296:    if not entries:
297:        return ""
299:    relevant = []
300:    for e in reversed(entries):  # most recent first
301:        trigger = e.get("trigger", "")
302:        tickers = e.get("tickers", {})
303:        if ticker in trigger or ticker in tickers:
304:            relevant.append(e)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/agent_invocation.py" | Select-Object -Skip 260 -First 260' in Q:\fa-adv-2026-05-11
 succeeded in 670ms:
305:            if len(relevant) >= max_entries:
306:                break
308:    if not relevant:
309:        return ""
311:    lines = [f"[RECENT DECISIONS FOR {ticker}]"]
312:    for e in relevant:
313:        ts = e.get("ts", "?")[:16]
314:        decisions = e.get("decisions", {})
315:        prices = e.get("prices", {})
316:        price = prices.get(ticker)
318:        parts = []
319:        for strat, d in decisions.items():
320:            action = d.get("action", "HOLD")
321:            parts.append(f"{strat}={action}")
322:        action_str = ", ".join(parts) if parts else "?"
323:        price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "?"
324:        lines.append(f"  - {ts}: {action_str} @ {price_str}")
326:    lines.append(
327:        "  Review: were these decisions correct given current price? "
328:        "Has the thesis changed?"
329:    )
330:    return "\n".join(lines)
333:def _last_jsonl_ts(path):
334:    """Return the 'ts' value from the last entry of a JSONL file, or None.
336:    Uses efficient tail-read via last_jsonl_entry() (reads last 4KB only).
337:    """
338:    return last_jsonl_entry(path, field="ts")
341:def _safe_last_jsonl_ts(path, label):
342:    """Return the last JSONL timestamp without failing the invocation flow."""
343:    try:
344:        return _last_jsonl_ts(path)
345:    except Exception as e:
346:        logger.warning("%s baseline read failed: %s", label, e)
347:        return None
350:def _safe_elapsed_s():
351:    """Return elapsed-since-invoke seconds, robust to a poisoned _agent_start.
353:    P2B (2026-04-17): yesterday's 2026-04-16T13:45:45 critical_errors.jsonl
354:    entry had duration_s=-1776254571.5 (matches time.monotonic() - time.time()).
355:    Indicates some historical path seeded _agent_start with an epoch
356:    timestamp instead of a monotonic value. Clamping at the source +
357:    logging a diagnostic keeps downstream consumers trustworthy and
358:    surfaces the bug if it recurs.
360:    Codex P2 #2 follow-up (2026-04-17): a naive clamp-to-0 silently
361:    disabled the P1B timeout path — `elapsed > _agent_timeout` can never
362:    be true when elapsed is always 0. Fall back to `_agent_start_wall`
363:    (set alongside `_agent_start` at spawn) so we still recover a
364:    plausible elapsed and the hung-agent kill still fires. If both
365:    clocks are corrupted, return 0 — that's the pre-existing failure
366:    mode, not a worse state.
367:    """
368:    raw = time.monotonic() - _agent_start
369:    if raw >= 0:
370:        return raw
371:    # Monotonic is poisoned — try the wall-clock fallback.
372:    if _agent_start_wall > 0:
373:        wall_elapsed = time.time() - _agent_start_wall
374:        if wall_elapsed >= 0:
375:            logger.warning(
376:                "BUG-P2B: monotonic elapsed negative (raw=%.1fs, "
377:                "_agent_start=%.1f); falling back to wall-clock "
378:                "(%.1fs since _agent_start_wall=%.1f). "
379:                "Indicates _agent_start was seeded with a non-monotonic value.",
380:                raw, _agent_start, wall_elapsed, _agent_start_wall,
381:            )
382:            return wall_elapsed
383:    # Both clocks bad — clamp to 0 and warn loudly.
384:    logger.warning(
385:        "BUG-P2B: negative elapsed AND no wall-clock fallback "
386:        "(raw=%.1fs, _agent_start=%.1f, _agent_start_wall=%.1f) — "
387:        "clamping to 0. Timeout enforcement will not fire this cycle.",
388:        raw, _agent_start, _agent_start_wall,
389:    )
390:    return 0.0
393:def _scan_agent_log_for_auth_failure(label: str, extra_context: dict | None = None) -> bool:
394:    """Scan the captured agent.log slice for claude-CLI auth-error markers.
396:    P1-3 (2026-05-02 last-followups): the timeout-kill path
397:    (``_kill_overrun_agent``) used to forget the dead subprocess without
398:    inspecting what it had printed. ``check_agent_completion()`` already
399:    runs this scan on the happy path (line 956), so a hung agent that
400:    printed "Not logged in" before getting stuck on a network retry
401:    would surface as ``timeout`` (not ``auth_error``) and never land in
402:    ``critical_errors.jsonl``. That asymmetry is the same class of silent
403:    auth outage that the March-April 2026 incident exposed — the whole
404:    point of the journal is to make that failure mode impossible to miss.
406:    Helper exists at module level so both call sites
407:    (``check_agent_completion`` and ``_kill_overrun_agent``) stay in sync
408:    if the scan logic ever needs to evolve.
410:    Returns True iff an auth-error marker was detected in the new slice.
411:    Never raises — IO or decode failures are swallowed and logged so a
412:    transient log-read problem cannot break the kill / completion paths.
414:    Args:
415:        label: Caller identifier used in the auth-failure record (e.g.
416:            ``"layer2_t2_timeout"``). Tier and trigger context are pulled
417:            from the module-level ``_agent_tier`` / ``_agent_reasons``.
418:        extra_context: Optional dict merged into the auth-failure record's
419:            ``context`` field (e.g. ``{"exit_code": 0, "duration_s": 12.3}``
420:            on the completion path). Tier/reasons are always included; this
421:            is for caller-specific extras.
422:    """
423:    try:
424:        agent_log_path = DATA_DIR / "agent.log"
425:        if not agent_log_path.exists():
426:            return False
427:        with open(agent_log_path, "rb") as f:
428:            f.seek(_agent_log_start_offset)
429:            new_output = f.read().decode("utf-8", errors="replace")
430:        ctx = {
431:            "tier": _agent_tier,
432:            "reasons": (_agent_reasons or [])[:5],
433:        }
434:        if extra_context:
435:            ctx.update(extra_context)
436:        return detect_auth_failure(
437:            new_output,
438:            caller=label,
439:            context=ctx,
440:        )
441:    except Exception as e:
442:        logger.warning("Auth-error scan of agent.log failed (%s): %s", label, e)
443:        return False
446:def _kill_overrun_agent(fallback_reasons=None, fallback_tier=None):
447:    """Kill the running _agent_proc and clear module state.
449:    P1B (2026-04-17): extracted from ``try_invoke_agent`` so it can also
450:    be called from ``check_agent_completion``. Previously the timeout
451:    check lived only inside try_invoke_agent, meaning a hung agent could
452:    run indefinitely if no new triggers fired (yesterday evidence: T1
453:    invoked 16:04:58 with timeout=120s completed at 16:15:01 = 603s).
455:    Logs the trigger with status="timeout" and clears ``_agent_proc`` /
456:    ``_agent_log`` on the way out.
458:    P1-3 (2026-05-02 last-followups): also scans the captured agent.log
459:    slice for claude-CLI auth-error markers BEFORE clearing module state,
460:    so the silent-auth-failure detector covers the timeout path too — not
461:    just the happy completion path. See ``_scan_agent_log_for_auth_failure``
462:    for full rationale.
464:    Args:
465:        fallback_reasons: Reason list to use for the trigger log entry if
466:            ``_agent_reasons`` is empty (caller context for the missing
467:            _reasons.).
468:        fallback_tier: Tier to log if ``_agent_tier`` is None.
470:    Returns:
471:        bool: True if the kill succeeded (or the process had already
472:        exited). False if the kill command itself failed — caller must
473:        NOT spawn a replacement in that case because the old process
474:        may still be holding resources.
475:    """
476:    global _agent_proc, _agent_log
478:    if _agent_proc is None:
479:        return True
481:    pid = _agent_proc.pid
482:    elapsed = _safe_elapsed_s()
483:    logger.info("Agent pid=%s timed out (%.0fs), killing", pid, elapsed)
485:    kill_ok = True
486:    if platform.system() == "Windows":
487:        # BUG-92: Check taskkill return code to detect kill failure
488:        # BUG-189: rc=128 means process already exited — treat as success
489:        result = subprocess.run(
490:            ["taskkill", "/F", "/T", "/PID", str(pid)],
491:            capture_output=True,
492:        )
493:        if result.returncode not in (0, 128):
494:            logger.error(
495:                "taskkill failed (rc=%d): %s",
496:                result.returncode,
497:                result.stderr.decode(errors="replace").strip(),
498:            )
499:            kill_ok = False
500:        elif result.returncode == 128:
501:            logger.info("Agent pid=%s already exited (rc=128)", pid)
502:    else:
503:        _agent_proc.kill()
504:    try:
505:        _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
506:    except subprocess.TimeoutExpired:
507:        if kill_ok:
508:            logger.error("Agent pid=%s did not exit after kill+15s wait", pid)
509:        kill_ok = False
511:    if _agent_log:
512:        try:
513:            _agent_log.close()
514:        except Exception as e:
515:            logger.warning("Agent log close failed: %s", e)
516:        _agent_log = None
518:    # P1-3 (2026-05-02 last-followups): scan the captured agent.log slice
519:    # for auth-error markers before forgetting the dead subprocess. Done
520:    # AFTER closing _agent_log (so any buffered output is flushed) but
521:    # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so
522:    # the auth-failure record carries the right tier + trigger context).
523:    # Best-effort: failures are swallowed inside the helper so a busted
524:    # log read can never break the kill path.
525:    auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
526:    _scan_agent_log_for_auth_failure(auth_label)
528:    # BUG-91: Log the timed-out invocation before returning
529:    _log_trigger(
530:        _agent_reasons or fallback_reasons or [],
531:        "timeout",
532:        tier=_agent_tier or fallback_tier,
533:    )
535:    _agent_proc = None
536:    return kill_ok
539:def invoke_agent(reasons, tier=3):
540:    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
541:    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
543:    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
544:    if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
545:        logger.info(
546:            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
547:            _consecutive_stack_overflows,
548:        )
549:        _log_trigger(reasons, "skipped_stack_overflow", tier=tier)
550:        return False
552:    # Check if Layer 2 is enabled — allows running data loop without Claude quota
553:    try:
554:        config = _load_config()
555:        l2_cfg = config.get("layer2", {})
556:        if not l2_cfg.get("enabled", True):
557:            logger.info("Layer 2 disabled (config.layer2.enabled=false), skipping")
558:            return False
559:    except Exception as e:
560:        logger.warning("Failed to load config for layer2 check: %s", e)
562:    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
563:    timeout = tier_cfg["timeout"]
565:    # 2026-05-05: this reentrancy block reads/mutates the same _agent_proc /
566:    # _agent_log / _agent_start state that the watchdog tick observes via
567:    # _check_agent_completion_locked. Without the lock, the watchdog could
568:    # observe a freshly-killed _agent_proc.poll() exit code and write a
569:    # "failed"/"incomplete" row at the same time _kill_overrun_agent is
570:    # writing its "timeout" row — exactly the double-log the lock was added
571:    # to prevent. Hold _completion_lock for the entire read-decide-kill
572:    # path; _kill_overrun_agent itself does NOT take the lock so this is
573:    # safe (no reentrant acquire).
574:    with _completion_lock:
575:        if _agent_proc and _agent_proc.poll() is None:
576:            # BUG-203: use monotonic clock for elapsed — wall clock is NTP-jump-prone.
577:            # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
578:            # can't cause a negative elapsed that silently skips the timeout.
579:            elapsed = _safe_elapsed_s()
580:            if elapsed > _agent_timeout:
581:                # P1B (2026-04-17): helper so check_agent_completion can share
582:                # the kill path — see _kill_overrun_agent docstring.
583:                kill_ok = _kill_overrun_agent(
584:                    fallback_reasons=reasons, fallback_tier=tier,
585:                )
586:                # BUG-92: If kill failed, don't spawn new agent (old one may
587:                # still be running)
588:                if not kill_ok:
589:                    logger.error(
590:                        "Not spawning new agent — old process may still be running"
591:                    )
592:                    return False
593:            else:
594:                logger.info(
595:                    "Agent still running (pid %s, %.0fs), skipping",
596:                    _agent_proc.pid, elapsed,
597:                )
598:                return False
600:    if _agent_log:
601:        _agent_log.close()
602:        _agent_log = None
604:    try:
605:        from portfolio.journal import write_context
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/agent_invocation.py" | Select-Object -Skip 520 -First 260' in Q:\fa-adv-2026-05-11
 succeeded in 723ms:
607:        n = write_context()
608:        logger.info("Layer 2 context: %d journal entries", n)
609:    except Exception as e:
610:        logger.warning("journal context failed: %s", e)
612:    # Perception gate: skip low-value invocations
613:    try:
614:        from portfolio.perception_gate import should_invoke as _should_invoke
615:        should, gate_reason = _should_invoke(reasons, tier)
616:        if not should:
617:            logger.info("Perception gate skipped: %s", gate_reason)
618:            _log_trigger(reasons, "skipped_gate", tier=tier)
619:            return False
620:    except Exception as e:
621:        logger.warning("perception gate error (passing through): %s", e)
623:    # BUG-214: Drawdown circuit breaker — first-ever automated risk gate on
624:    # the primary trading path. Advisory below _DRAWDOWN_BLOCK_PCT, hard-block
625:    # above it. Respects user's high risk tolerance (memory/feedback_risk_tolerance.md).
626:    #
627:    # 2026-05-02 (adversarial review 05-01 P0-5): the bare `except Exception`
628:    # used to swallow all errors and proceed. That meant a portfolio in
629:    # 50%+ drawdown could continue trading if anything in the check threw
630:    # (ImportError, IO error on portfolio_state.json mid-rename, KeyError
631:    # on a missing dd dict key). The fail-safe direction for a circuit
632:    # breaker is BLOCK on failure, not pass.
633:    #
634:    # New behavior:
635:    # - Per-portfolio errors (file read, dict access) are tolerated for THAT
636:    #   portfolio only — we still check the other portfolio.
637:    # - A complete failure to even load the check (ImportError) is logged
638:    #   ERROR + treated as block (fail-safe).
639:    # - The narrow per-portfolio try/except still tolerates transient I/O,
640:    #   so a missing portfolio_state.json mid-rename doesn't take the loop
641:    #   down.
642:    _drawdown_context = ""
643:    try:
644:        from portfolio.risk_management import check_drawdown
645:    except Exception as e:
646:        logger.error("DRAWDOWN BLOCK: check_drawdown unavailable (%s) — fail-safe block", e)
647:        _log_trigger(reasons, "blocked_drawdown_unavailable", tier=tier)
648:        return False
650:    for label, pf_path in [("Patient", PATIENT_PORTFOLIO), ("Bold", BOLD_PORTFOLIO)]:
651:        if not pf_path.exists():
652:            continue
653:        try:
654:            dd = check_drawdown(str(pf_path), max_drawdown_pct=_DRAWDOWN_WARN_PCT)
655:            if dd["current_drawdown_pct"] > _DRAWDOWN_BLOCK_PCT:
656:                logger.error(
657:                    "DRAWDOWN BLOCK: %s portfolio at %.1f%% drawdown (>%.0f%%) — skipping invocation",
658:                    label, dd["current_drawdown_pct"], _DRAWDOWN_BLOCK_PCT,
659:                )
660:                _log_trigger(reasons, f"blocked_drawdown_{label.lower()}", tier=tier)
661:                return False
662:            if dd["current_drawdown_pct"] > _DRAWDOWN_WARN_PCT:
663:                logger.warning(
664:                    "DRAWDOWN WARNING: %s portfolio at %.1f%% drawdown (peak %.0f, current %.0f SEK)",
665:                    label, dd["current_drawdown_pct"], dd["peak_value"], dd["current_value"],
666:                )
667:            _drawdown_context += (
668:                f"\n[DRAWDOWN {label}] {dd['current_drawdown_pct']:.1f}% from peak "
669:                f"(peak={dd['peak_value']:.0f}, current={dd['current_value']:.0f} SEK)"
670:            )
671:        except Exception as e:
672:            # Per-portfolio failure: log ERROR (not WARNING), but tolerate so
673:            # the OTHER portfolio still gets checked. This keeps a transient IO
674:            # error on one file from disabling the gate entirely. If BOTH
675:            # portfolios fail, neither will set the block flag, and the
676:            # invocation proceeds — by design, since blocking trading on a pure
677:            # IO race that the next cycle will re-check is too aggressive.
678:            logger.error(
679:                "DRAWDOWN check failed for %s portfolio (proceeding for this portfolio only): %s",
680:                label, e,
681:            )
683:    # Adversarial review 05-01 P1-12 (2026-05-02): trade-guards pre-execution gate.
684:    # `should_block_trade` was implemented in trade_guards.py for ARCH-29 but
685:    # never imported by production code — only by tests. Wire it here so an
686:    # invocation triggered by a ticker that is in cooldown for BOTH Patient
687:    # and Bold gets short-circuited before the multi-agent / subprocess spawn
688:    # (saves ~600s of T2 subprocess + Claude tokens for a decision that
689:    # cannot be acted on).
690:    #
691:    # Semantics:
692:    #   1. Pull the trade_guard_warnings already computed by reporting.py and
693:    #      stored in agent_summary.json.
694:    #   2. Build _guard_context for the prompt (advisory) — Layer 2 should
695:    #      see active cooldowns/loss-streaks regardless of the gate decision.
696:    #   3. Block ONLY when should_block_trade(...) is True AND the trigger
697:    #      ticker is blocked for BOTH strategies. Single-strategy block
698:    #      proceeds (the unblocked strategy can still trade).
699:    #   4. Failure to load warnings (missing agent_summary, IO race) is
700:    #      fail-OPEN — unlike drawdown, cooldowns are soft constraints and a
701:    #      single missed gate cycle is not a safety risk.
702:    _guard_context = ""
703:    try:
704:        guard_result = _load_guard_warnings()
705:    except Exception as e:
706:        logger.warning("trade-guards load failed (proceeding): %s", e)
707:        guard_result = {"warnings": [], "summary": "load_failed"}
709:    if guard_result.get("warnings"):
710:        _guard_context += f"\n[TRADE GUARDS] {guard_result.get('summary', '')}"
711:        for w in guard_result["warnings"][:10]:  # cap context size
712:            sev = w.get("severity", "?")
713:            tkr = w.get("ticker") or w.get("details", {}).get("ticker", "?")
714:            strat = w.get("strategy") or w.get("details", {}).get("strategy", "?")
715:            msg = w.get("message", w.get("guard", "?"))
716:            _guard_context += f"\n  [{sev.upper()}] {tkr}/{strat}: {msg}"
718:    try:
719:        from portfolio.trade_guards import should_block_trade
720:        if should_block_trade(guard_result):
721:            # Determine the trigger ticker and check whether BOTH strategies
722:            # are blocked on it. Anything else (single-strategy block, or
723:            # block on a different ticker than the trigger) is advisory.
724:            trigger_ticker = _extract_ticker(reasons)
725:            blocked_strategies = {
726:                w.get("strategy") or w.get("details", {}).get("strategy")
727:                for w in guard_result["warnings"]
728:                if w.get("severity") == "block"
729:                and (
730:                    w.get("ticker") == trigger_ticker
731:                    or w.get("details", {}).get("ticker") == trigger_ticker
732:                )
733:            }
734:            blocked_strategies.discard(None)
735:            if {"patient", "bold"}.issubset(blocked_strategies):
736:                logger.error(
737:                    "TRADE GUARDS BLOCK: %s in cooldown for BOTH strategies — "
738:                    "skipping invocation",
739:                    trigger_ticker,
740:                )
741:                _log_trigger(reasons, "blocked_trade_guards", tier=tier)
742:                return False
743:    except Exception as e:
744:        # Import failures or shape mismatches must not derail the invocation.
745:        logger.warning("trade-guards gate failed (proceeding): %s", e)
747:    # Multi-agent mode: parallel specialists + synthesis (Coordinator Mode pattern)
748:    # Enabled via config.layer2.multi_agent = true, only for T2/T3
749:    try:
750:        config = _load_config()
751:        multi_agent = config.get("layer2", {}).get("multi_agent", False)
752:    except Exception:
753:        multi_agent = False
755:    if multi_agent and tier >= 2:
756:        try:
757:            from portfolio.multi_agent_layer2 import (
758:                build_synthesis_prompt,
759:                launch_specialists,
760:                wait_for_specialists,
761:            )
762:            # Extract primary ticker from reasons
763:            ticker = _extract_ticker(reasons)
764:            logger.info("Multi-agent T%d: launching 3 specialists for %s", tier, ticker)
765:            procs = launch_specialists(ticker, reasons)
766:            if procs:
767:                # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
768:                # layer2.specialist_timeout_s) to avoid blocking the main loop.
769:                # TODO: run specialists in background thread, collect results async.
770:                specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
771:                results = wait_for_specialists(procs, timeout=specialist_timeout)
772:                success_count = sum(1 for v in results.values() if v)
773:                logger.info("Specialists complete: %d/%d succeeded", success_count, len(results))
774:                # Even if some fail, proceed with synthesis using available reports
775:                prompt = build_synthesis_prompt(ticker, reasons)
776:                # Fall through to normal agent launch with synthesis prompt
777:            else:
778:                logger.warning("No specialists launched, falling back to single-agent")
779:                prompt = _build_tier_prompt(tier, reasons)
780:        except Exception as e:
781:            logger.warning("Multi-agent failed (%s), falling back to single-agent", e)
782:            prompt = _build_tier_prompt(tier, reasons)
783:    else:
784:        prompt = _build_tier_prompt(tier, reasons)
786:    # BUG-214: Append drawdown context so Layer 2 sees current risk levels.
787:    if _drawdown_context:
788:        prompt += "\n\n[RISK DATA]" + _drawdown_context
789:    # P1-12 (2026-05-02): also surface trade-guard warnings to Layer 2 so
790:    # it can avoid suggesting actions that the guards would just block in
791:    # check_overtrading_guards anyway.
792:    if _guard_context:
793:        prompt += "\n\n[TRADE GUARDS]" + _guard_context
795:    # Decision feedback loop (2026-05-02 research): inject recent decisions
796:    # for the trigger ticker so Layer 2 can see its own track record and
797:    # calibrate (e.g., "I said SELL at $73 — price is now $75, was I wrong?").
798:    try:
799:        feedback_ticker = _extract_ticker(reasons)
800:        _feedback = _build_decision_feedback(feedback_ticker)
801:        if _feedback:
802:            prompt += "\n\n" + _feedback
803:    except Exception as e:
804:        logger.debug("decision feedback failed (non-fatal): %s", e)
806:    max_turns = tier_cfg["max_turns"]
808:    # Try direct claude invocation first; fall back to bat file for T3
809:    claude_cmd = shutil.which("claude")
810:    if claude_cmd:
811:        # 2026-04-13: DO NOT add `--bare`. It disables OAuth/keychain auth
812:        # and only accepts ANTHROPIC_API_KEY. This user runs on a Max
813:        # subscription with no API key, so `--bare` silently breaks every
814:        # invocation ("Not logged in" to stdout, exit 0). Commit b4bb57d
815:        # added it on 2026-03-27; removed on 2026-04-13 after 3 weeks of
816:        # silent Layer 2 failures. See portfolio/claude_gate.py
817:        # (detect_auth_failure) for the runtime guard.
818:        cmd = [
819:            claude_cmd, "-p", prompt,
820:            "--allowedTools", "Edit,Read,Bash,Write",
821:            "--max-turns", str(max_turns),
822:        ]
823:    else:
824:        # Fallback: use pf-agent.bat (always Tier 3)
825:        agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
826:        if not agent_bat.exists():
827:            logger.warning("Agent script not found at %s", agent_bat)
828:            return False
829:        cmd = ["cmd", "/c", str(agent_bat)]
830:        logger.info("claude not on PATH, falling back to pf-agent.bat (T3)")
832:    log_fh = None
833:    try:
834:        agent_log_path = DATA_DIR / "agent.log"
835:        # Capture the current file size BEFORE opening in append mode, so
836:        # check_agent_completion() can read only this invocation's output
837:        # (for auth-error detection) and not the entire log history.
838:        global _agent_log_start_offset
839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
840:        log_fh = open(agent_log_path, "a", encoding="utf-8")
841:        # Strip Claude Code session markers to avoid "nested session" error
842:        # when the parent process tree has Claude Code running
843:        agent_env = os.environ.copy()
844:        agent_env.pop("CLAUDECODE", None)
845:        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
846:        # Increase Node.js stack size to prevent stack overflow in Claude CLI
847:        agent_env["NODE_OPTIONS"] = "--stack-size=16384"
848:        # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
849:        # STARTUP CHECK protocol doesn't ask "How would you like to proceed?"
850:        # when it finds unresolved critical_errors.jsonl entries. The agent
851:        # has no stdin (pipe only), so any prompt that blocks on user input
852:        # makes it hit the tier timeout with zero work done. The CLAUDE.md
853:        # conditional turns that into "log the unresolved entries in your
854:        # journal entry and proceed with the trigger task".
855:        agent_env["PF_HEADLESS_AGENT"] = "1"
856:        _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
857:        _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
858:        _agent_proc = subprocess.Popen(
859:            cmd,
860:            cwd=str(BASE_DIR),
861:            stdout=log_fh,
862:            stderr=subprocess.STDOUT,
863:            env=agent_env,
864:        )
865:        _agent_log = log_fh  # transfer ownership on success
866:        log_fh = None  # prevent cleanup below from closing it
867:        _agent_start = time.monotonic()
868:        _agent_start_wall = time.time()  # wall-clock fallback for P2B
869:        _agent_timeout = timeout
870:        _agent_tier = tier
871:        _agent_reasons = list(reasons)
872:        # BUG-219: Snapshot transaction counts so check_agent_completion()
873:        # can detect new trades and call record_trade().
874:        global _patient_txn_count_before, _bold_txn_count_before
875:        try:
876:            from portfolio.file_utils import load_json
877:            _patient_txn_count_before = len(
878:                (load_json(PATIENT_PORTFOLIO, default={}) or {}).get("transactions", [])
879:            )
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/agent_invocation.py" | Select-Object -Skip 780 -First 260' in Q:\fa-adv-2026-05-11
 succeeded in 714ms:
880:            _bold_txn_count_before = len(
881:                (load_json(BOLD_PORTFOLIO, default={}) or {}).get("transactions", [])
882:            )
883:        except Exception:
884:            _patient_txn_count_before = 0
885:            _bold_txn_count_before = 0
886:        # 2026-04-17: Publish the tier into health_state so loop_contract
887:        # can pick the right per-tier grace window for the journal-activity
888:        # check. Without this, the contract defaults to T3 grace (20m),
889:        # which is conservative but can delay detection when an all-T1
890:        # cadence runs silent. See loop_contract._get_layer2_grace_s() for
891:        # the consumer and LAYER2_JOURNAL_GRACE_S_BY_TIER for the table.
892:        # Best-effort: never fail the invocation because health_state is
893:        # unwriteable (atomic_write_json handles the happy path; any
894:        # exception is logged and swallowed).
895:        try:
896:            from portfolio.file_utils import atomic_write_json, load_json
897:            # 2026-04-17 Codex P2: when claude is missing from PATH we fall
898:            # back to pf-agent.bat which is unconditionally T3 regardless of
899:            # the requested tier. Record the *effective* tier so the
900:            # per-tier grace window in loop_contract reflects what's
901:            # actually running.
902:            effective_tier = 3 if not claude_cmd else tier
903:            health_path = DATA_DIR / "health_state.json"
904:            health = load_json(health_path, default={}) or {}
905:            health["last_invocation_tier"] = effective_tier
906:            health["last_invocation_tier_ts"] = datetime.now(UTC).isoformat()
907:            atomic_write_json(health_path, health)
908:        except Exception as e:
909:            logger.warning("health_state tier publish failed: %s", e)
910:        logger.info(
911:            "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
912:            tier, _agent_proc.pid, max_turns, timeout,
913:            ", ".join(reasons[:3]),
914:        )
915:        # 2026-05-05: arm the completion watchdog so the wall-clock
916:        # timeout fires within ~30 s of the real budget even when the
917:        # main loop's run() cycle bloats. See module-level note at
918:        # _COMPLETION_WATCHDOG_INTERVAL_S.
919:        _ensure_completion_watchdog()
920:        # Save Layer 2 invocation notification (save-only, not sent to Telegram)
921:        try:
922:            config = _load_config()
923:            reason_str = escape_markdown_v1(", ".join(reasons[:3]))
924:            if len(reasons) > 3:
925:                reason_str += f" (+{len(reasons) - 3} more)"
926:            tier_label = tier_cfg["label"]
927:            notify_msg = f"_Layer 2 T{tier} ({tier_label}): {reason_str}_"
928:            send_or_store(notify_msg, config, category="invocation")
929:        except Exception as e:
930:            logger.warning("invocation notification failed: %s", e)
931:        return True
932:    except Exception as e:
933:        logger.error("invoking agent: %s", e)
934:        if log_fh is not None:
935:            log_fh.close()
936:        return False
939:def _write_fishing_context(journal_entry):
940:    """Extract fishing context from Layer 2 journal entry.
942:    Called after Layer 2 completes. Creates a structured context file
943:    that the fish engine reads as its strongest tactic vote.
944:    """
945:    try:
946:        tickers = journal_entry.get('tickers', {})
947:        xag = tickers.get('XAG-USD')
948:        if not xag:
949:            return
951:        outlook = xag.get('outlook', '')
952:        conviction = float(xag.get('conviction', 0))
953:        levels = xag.get('levels', [])
954:        thesis = xag.get('thesis', '')
956:        # Determine direction bias
957:        if outlook == 'bullish' and conviction >= 0.4:
958:            direction_bias = 'bullish'
959:            tactic_vote = 'LONG'
960:            allow_long = True
961:            allow_short = conviction < 0.6  # block short only if very bullish
962:        elif outlook == 'bearish' and conviction >= 0.4:
963:            direction_bias = 'bearish'
964:            tactic_vote = 'SHORT'
965:            allow_long = conviction < 0.6
966:            allow_short = True
967:        else:
968:            direction_bias = 'neutral'
969:            tactic_vote = None
970:            allow_long = True
971:            allow_short = True
973:        # Check for event context from watchlist
974:        watchlist = journal_entry.get('watchlist', [])
975:        event_context = ''
976:        for item in watchlist:
977:            if isinstance(item, str) and any(
978:                w in item.lower() for w in ['event', 'fomc', 'cpi', 'tariff', 'opec']
979:            ):
980:                event_context = item[:100]
981:                break
983:        # Determine position size multiplier from regime
984:        regime = journal_entry.get('regime', 'ranging')
985:        if regime == 'high-vol':
986:            position_size_multiplier = 0.5
987:        elif regime in ('trending-up', 'trending-down'):
988:            position_size_multiplier = 1.0
989:        else:
990:            position_size_multiplier = 0.75  # ranging = slightly reduced
992:        context = {
993:            'timestamp': journal_entry.get('ts', ''),
994:            'valid_until': '',  # fish engine uses 4h staleness check
995:            'ticker': 'XAG-USD',
996:            'direction_bias': direction_bias,
997:            'bias_confidence': conviction,
998:            'bias_reasoning': thesis[:200] if thesis else '',
999:            'allow_long': allow_long,
1000:            'allow_short': allow_short,
1001:            'max_hold_minutes': 120,
1002:            'position_size_multiplier': position_size_multiplier,
1003:            'allow_overnight': conviction >= 0.6 and outlook == 'bullish',
1004:            'event_context': event_context,
1005:            'bull_case': '',
1006:            'bear_case': '',
1007:            'journal_action': '',
1008:            'journal_confidence': conviction,
1009:            'tactic_vote': tactic_vote,
1010:            'tactic_weight': 2.0,
1011:            'levels': levels,
1012:        }
1014:        # Extract bull/bear cases from decisions
1015:        decisions = journal_entry.get('decisions', {})
1016:        for strategy in ('patient', 'bold'):
1017:            dec = decisions.get(strategy, {})
1018:            reasoning = dec.get('reasoning', '')
1019:            action = dec.get('action', 'HOLD')
1020:            if action != 'HOLD':
1021:                context['journal_action'] = action
1022:            if reasoning:
1023:                if not context['bull_case'] and 'bullish' in reasoning.lower():
1024:                    context['bull_case'] = reasoning[:150]
1025:                elif not context['bear_case'] and (
1026:                    'bearish' in reasoning.lower() or 'sell' in reasoning.lower()
1027:                ):
1028:                    context['bear_case'] = reasoning[:150]
1030:        from portfolio.file_utils import atomic_write_json
1032:        # H22/NEW-3: use DATA_DIR absolute path instead of relative 'data/...'
1033:        atomic_write_json(DATA_DIR / 'fishing_context.json', context)
1035:    except Exception as e:
1036:        logger.warning('Fishing context error: %s', e)
1037:        # BUG-181: Write neutral context on failure to prevent stale bias
1038:        try:
1039:            from datetime import UTC, datetime
1041:            from portfolio.file_utils import atomic_write_json
1042:            atomic_write_json(DATA_DIR / 'fishing_context.json', {
1043:                'timestamp': datetime.now(UTC).isoformat(),
1044:                'ticker': 'XAG-USD',
1045:                'direction_bias': 'neutral',
1046:                'bias_confidence': 0.0,
1047:                'bias_reasoning': f'Context extraction failed: {e}',
1048:                'allow_long': True,
1049:                'allow_short': True,
1050:                'tactic_vote': None,
1051:                'tactic_weight': 0.0,
1052:            })
1053:        except Exception:
1054:            logger.warning("Failed to write neutral journal entry", exc_info=True)
1057:def _record_new_trades():
1058:    """BUG-219 / PR-R4-4: Check for new transactions since invoke_agent()
1059:    and call record_trade() for each, activating overtrading prevention.
1061:    Never raises — all errors are logged and swallowed so the completion
1062:    path is never broken by guard bookkeeping failures.
1063:    """
1064:    try:
1065:        from portfolio.file_utils import load_json
1066:        from portfolio.trade_guards import record_trade
1068:        for strategy, pf_path, count_before in [
1069:            ("patient", PATIENT_PORTFOLIO, _patient_txn_count_before),
1070:            ("bold", BOLD_PORTFOLIO, _bold_txn_count_before),
1071:        ]:
1072:            state = load_json(pf_path, default={}) or {}
1073:            txns = state.get("transactions", [])
1074:            if len(txns) <= count_before:
1075:                continue
1076:            # New transactions appeared — record each for guard tracking
1077:            new_txns = txns[count_before:]
1078:            for txn in new_txns:
1079:                ticker = txn.get("ticker")
1080:                direction = txn.get("action")
1081:                if not ticker or direction not in ("BUY", "SELL"):
1082:                    continue
1083:                pnl_pct = txn.get("pnl_pct")
1084:                record_trade(ticker, direction, strategy, pnl_pct=pnl_pct)
1085:                logger.info(
1086:                    "BUG-219: recorded %s %s %s pnl=%.2f%% for overtrading guards",
1087:                    strategy, direction, ticker, pnl_pct or 0.0,
1088:                )
1089:    except Exception as e:
1090:        logger.warning("BUG-219: record_trade wiring failed: %s", e)
1093:def check_agent_completion():
1094:    """Check if a running agent has completed and log completion info.
1096:    Thread-safe: serialised by ``_completion_lock`` so the main-loop call
1097:    site (``portfolio.main.run``) and the 30 s daemon watchdog
1098:    (``_completion_watchdog``) cannot race on ``_agent_proc`` /
1099:    ``_agent_start`` state. Both call paths share the same lock; whichever
1100:    reaches the lock first observes the completion and writes the
1101:    invocations.jsonl row, the other returns ``None`` because
1102:    ``_agent_proc`` is cleared at the end of the handler.
1104:    Returns:
1105:        dict with the following keys (None if no agent is running or the
1106:        agent is still in progress and under its timeout):
1108:        * ``status`` — "success", "incomplete", "failed", "auth_error",
1109:          "timeout" (P1B, 2026-04-17), or "stack_overflow"
1110:        * ``exit_code`` — int or None (None on timeout-kill path)
1111:        * ``duration_s`` — float, always >= 0 (P2B clamp)
1112:        * ``tier`` — int, the tier of the completed agent
1113:        * ``reasons`` — list[str], the triggers for this invocation
1114:        * ``journal_written`` — bool
1115:        * ``telegram_sent`` — bool
1116:        * ``completed_at`` — ISO-8601 UTC timestamp
1117:    """
1118:    with _completion_lock:
1119:        return _check_agent_completion_locked()
1122:def _check_agent_completion_locked():
1123:    """Body of ``check_agent_completion``. The caller MUST hold
1124:    ``_completion_lock``. Split out so the watchdog tick can call into
1125:    the same code path without re-acquiring the lock recursively.
1126:    """
1127:    global _agent_proc, _agent_log, _agent_start, _agent_start_wall
1128:    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
1130:    if _agent_proc is None:
1131:        return None
1133:    exit_code = _agent_proc.poll()
1134:    if exit_code is None:
1135:        # Still running. P1B (2026-04-17): enforce the wall-clock timeout
1136:        # here too — the lazy check in try_invoke_agent only fires when a
1137:        # new trigger arrives, so a hung agent could run indefinitely if
1138:        # no new triggers came through (yesterday: T1 timeout=120s ran
1139:        # 603s). Share the same kill helper used by try_invoke_agent to
1140:        # keep kill semantics identical.
1141:        elapsed = _safe_elapsed_s()
1142:        if _agent_timeout and elapsed > _agent_timeout:
1143:            killed_tier = _agent_tier
1144:            killed_reasons = list(_agent_reasons or [])
1145:            _kill_overrun_agent()
1146:            return {
1147:                "status": "timeout",
1148:                "exit_code": None,
1149:                "duration_s": round(elapsed, 1),
1150:                "tier": killed_tier,
1151:                "reasons": killed_reasons,
1152:                "journal_written": False,
1153:                "telegram_sent": False,
1154:                "completed_at": datetime.now(UTC).isoformat(),
1155:            }
1156:        return None
1158:    # Process has finished — collect completion info.
1159:    # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
1160:    # can't produce the negative duration_s seen in yesterday's 13:45:45
1161:    # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).
1162:    duration_s = round(_safe_elapsed_s(), 1)
1163:    completed_at = datetime.now(UTC).isoformat()
1165:    # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
1166:    try:
1167:        journal_ts_after = _last_jsonl_ts(JOURNAL_FILE)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --line-number --no-heading "." "portfolio/agent_invocation.py" | Select-Object -Skip 1040 -First 240' in Q:\fa-adv-2026-05-11
 succeeded in 705ms:
1168:    except Exception:
1169:        logger.warning("Failed to read journal timestamp after agent completion")
1170:        journal_ts_after = None
1171:    journal_written = (
1172:        _journal_ts_before is not None
1173:        and journal_ts_after is not None
1174:        and journal_ts_after != _journal_ts_before
1175:    )
1177:    # BUG-97: Same protection for telegram file
1178:    try:
1179:        telegram_ts_after = _last_jsonl_ts(TELEGRAM_FILE)
1180:    except Exception:
1181:        logger.warning("Failed to read telegram timestamp after agent completion")
1182:        telegram_ts_after = None
1183:    telegram_sent = (
1184:        _telegram_ts_before is not None
1185:        and telegram_ts_after is not None
1186:        and telegram_ts_after != _telegram_ts_before
1187:    )
1189:    # Without a baseline from invoke_agent(), stay conservative and do not infer
1190:    # success from pre-existing files in the workspace.
1191:    if _journal_ts_before is None:
1192:        journal_written = False
1193:    if _telegram_ts_before is None:
1194:        telegram_sent = False
1196:    # 2026-04-13: Scan agent.log for auth-error markers (see claude_gate.py
1197:    # detect_auth_failure). Claude CLI can exit 0 while printing "Not logged
1198:    # in" to stdout — that's exactly the 3-week silent Layer 2 outage that
1199:    # motivated this detection. We captured _agent_log_start_offset before
1200:    # spawning the subprocess, so we only scan output from this invocation.
1201:    #
1202:    # P1-3 (2026-05-02 last-followups): scan logic extracted to
1203:    # ``_scan_agent_log_for_auth_failure`` so the timeout-kill path
1204:    # (``_kill_overrun_agent``) can share the exact same semantics. Without
1205:    # the helper, fixing one path and forgetting the other would re-open
1206:    # the same asymmetry the timeout path used to have.
1207:    auth_error_detected = _scan_agent_log_for_auth_failure(
1208:        f"layer2_t{_agent_tier}",
1209:        extra_context={"exit_code": exit_code, "duration_s": duration_s},
1210:    )
1212:    # Determine status
1213:    if auth_error_detected:
1214:        status = "auth_error"
1215:    elif exit_code != 0:
1216:        status = "failed"
1217:    elif journal_written and telegram_sent:
1218:        status = "success"
1219:    else:
1220:        status = "incomplete"
1222:    result = {
1223:        "status": status,
1224:        "exit_code": exit_code,
1225:        "duration_s": duration_s,
1226:        "tier": _agent_tier,
1227:        # Codex P2 #3 follow-up (2026-04-17): include `reasons` so the
1228:        # completion-path and timeout-path dicts have symmetric shape.
1229:        # Callers that dispatch on reasons shouldn't need to know which
1230:        # path produced the dict.
1231:        "reasons": list(_agent_reasons or []),
1232:        "completed_at": completed_at,
1233:        "journal_written": journal_written,
1234:        "telegram_sent": telegram_sent,
1235:    }
1237:    # Log to invocations file
1238:    log_entry = {
1239:        "ts": completed_at,
1240:        "reasons": _agent_reasons or [],
1241:        "status": status,
1242:        "tier": _agent_tier,
1243:        "exit_code": exit_code,
1244:        "duration_s": duration_s,
1245:        "journal_written": journal_written,
1246:        "telegram_sent": telegram_sent,
1247:    }
1248:    try:
1249:        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
1250:    except Exception as e:
1251:        logger.warning("Failed to log agent completion: %s", e)
1253:    # Post-process: extract fishing context from journal for metals fish engine
1254:    if journal_written:
1255:        with suppress(Exception):
1256:            new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
1257:            if new_journal_entry:
1258:                _write_fishing_context(new_journal_entry)
1260:    # BUG-219 / PR-R4-4: Wire record_trade() into production.
1261:    # After a successful agent run, check if new transactions appeared in
1262:    # either portfolio and record them for overtrading prevention guards
1263:    # (cooldowns, loss escalation, position rate limits).
1264:    _record_new_trades()
1266:    logger.info(
1267:        "Agent completed: status=%s exit=%d duration=%.1fs tier=%s journal=%s telegram=%s",
1268:        status, exit_code, duration_s, _agent_tier, journal_written, telegram_sent,
1269:    )
1271:    if status == "failed":
1272:        try:
1273:            config = _load_config()
1274:            send_or_store(
1275:                f"*L2 FAILED* T{_agent_tier} exit={exit_code} "
1276:                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
1277:                config, category="error",
1278:            )
1279:        except Exception as e:
1280:            logger.warning("Agent failure alert failed: %s", e)
1281:    elif status == "incomplete":
1282:        try:
1283:            config = _load_config()
1284:            send_or_store(
1285:                f"*L2 INCOMPLETE* T{_agent_tier} exit={exit_code} "
1286:                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
1287:                config, category="error",
1288:            )
1289:        except Exception as e:
1290:            logger.warning("Agent incomplete alert failed: %s", e)
1292:    # Track consecutive stack overflow crashes
1293:    global _consecutive_stack_overflows
1294:    if exit_code == _STACK_OVERFLOW_EXIT_CODE:
1295:        _consecutive_stack_overflows += 1
1296:        _save_stack_overflow_counter(_consecutive_stack_overflows)
1297:        logger.error(
1298:            "Claude CLI stack overflow (exit %d), %d consecutive. "
1299:            "Check project root for problematic files or update Claude Code.",
1300:            exit_code, _consecutive_stack_overflows,
1301:        )
1302:        if _consecutive_stack_overflows == _MAX_STACK_OVERFLOWS:
1303:            logger.error(
1304:                "Layer 2 auto-disabled after %d consecutive stack overflows",
1305:                _MAX_STACK_OVERFLOWS,
1306:            )
1307:            try:
1308:                config = _load_config()
1309:                send_or_store(
1310:                    f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
1311:                    f"consecutive stack overflows (exit {exit_code}). "
1312:                    "Claude CLI is crashing — investigate project root.",
1313:                    config, category="alert",
1314:                )
1315:            except Exception as e:
1316:                logger.warning("Stack overflow alert failed: %s", e)
1317:    else:
1318:        # BUG-95: Reset counter on any non-stack-overflow completion (success or otherwise).
1319:        # This prevents false positive auto-disable when the consecutive chain is broken.
1320:        if _consecutive_stack_overflows > 0:
1321:            _consecutive_stack_overflows = 0
1322:            _save_stack_overflow_counter(0)
1324:    # Clean up
1325:    if _agent_log:
1326:        try:
1327:            _agent_log.close()
1328:        except Exception as e:
1329:            logger.warning("Agent log close failed: %s", e)
1330:    _agent_proc = None
1331:    _agent_log = None
1332:    _agent_start = 0
1333:    _agent_start_wall = 0.0
1334:    _agent_tier = None
1335:    _agent_reasons = None
1336:    _journal_ts_before = None
1337:    _telegram_ts_before = None
1338:    _patient_txn_count_before = 0
1339:    _bold_txn_count_before = 0
1341:    return result
1344:def get_completion_stats(hours=24):
1345:    """Compute rolling completion stats from the invocations log.
1347:    Args:
1348:        hours: Number of hours to look back (default 24).
1350:    Returns:
1351:        dict with keys: total, success, incomplete, failed, timeout,
1352:        auth_error, completion_rate.  Returns zeroed stats if no data is
1353:        available.
1355:    Codex P2 #4 follow-up (2026-04-17): "timeout" and "auth_error" were
1356:    being dropped entirely by the status filter. Before P1B, timeouts
1357:    only fired when a new trigger arrived, so they were rare. After
1358:    P1B check_agent_completion enforces timeout every cycle — these
1359:    are now meaningful failure categories that belong in the health
1360:    rollup. Added as distinct buckets to preserve history and keep
1361:    completion_rate honest (timeouts count as failures for rate).
1362:    """
1363:    entries = load_jsonl(INVOCATIONS_FILE)
1364:    cutoff = datetime.now(UTC).timestamp() - (hours * 3600)
1366:    total = 0
1367:    success = 0
1368:    incomplete = 0
1369:    failed = 0
1370:    timeout = 0
1371:    auth_error = 0
1373:    tracked_statuses = ("success", "incomplete", "failed", "timeout", "auth_error")
1374:    for entry in entries:
1375:        entry_status = entry.get("status", "")
1376:        if entry_status not in tracked_statuses:
1377:            continue
1379:        ts_str = entry.get("ts", "")
1380:        if not ts_str:
1381:            continue
1383:        try:
1384:            dt = datetime.fromisoformat(ts_str)
1385:            if dt.tzinfo is None:
1386:                dt = dt.replace(tzinfo=UTC)
1387:            entry_ts = dt.timestamp()
1388:        except (ValueError, TypeError):
1389:            continue
1391:        if entry_ts < cutoff:
1392:            continue
1394:        total += 1
1395:        if entry_status == "success":
1396:            success += 1
1397:        elif entry_status == "incomplete":
1398:            incomplete += 1
1399:        elif entry_status == "failed":
1400:            failed += 1
1401:        elif entry_status == "timeout":
1402:            timeout += 1
1403:        elif entry_status == "auth_error":
1404:            auth_error += 1
1406:    completion_rate = (success / total * 100) if total > 0 else 0.0
1408:    return {
1409:        "total": total,
1410:        "success": success,
1411:        "incomplete": incomplete,
1412:        "failed": failed,
1413:        "timeout": timeout,
1414:        "auth_error": auth_error,
1415:        "completion_rate": round(completion_rate, 1),
1416:    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve read the core Layer 2 lifecycle and confirmed the watchdog, auth-scan, and timeout paths are central to the review. I’m switching to structured local inspection for the rest so I can cross-reference line numbers accurately without drifting outside the scoped file set.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 100ms:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
{
  "portfolio/main.py": 1394,
  "portfolio/agent_invocation.py": 1417,
  "portfolio/autonomous.py": 847,
  "portfolio/trigger.py": 495,
  "portfolio/market_timing.py": 343,
  "portfolio/claude_gate.py": 685,
  "portfolio/gpu_gate.py": 265,
  "portfolio/health.py": 453,
  "portfolio/alert_budget.py": 62,
  "portfolio/llm_prewarmer.py": 333,
  "portfolio/llm_calibration.py": 228,
  "portfolio/llm_batch.py": 554,
  "portfolio/llm_outcome_backfill.py": 292,
  "portfolio/llm_probability_log.py": 281,
  "portfolio/llama_server.py": 659,
  "portfolio/multi_agent_layer2.py": 243,
  "portfolio/perception_gate.py": 96,
  "portfolio/focus_analysis.py": 237,
  "portfolio/reporting.py": 1315,
  "portfolio/journal.py": 584,
  "portfolio/journal_index.py": 400,
  "portfolio/telegram_notifications.py": 143,
  "portfolio/telegram_poller.py": 388,
  "portfolio/digest.py": 272,
  "portfolio/daily_digest.py": 279,
  "portfolio/weekly_digest.py": 310,
  "portfolio/reflection.py": 242,
  "portfolio/regime_alerts.py": 218,
  "portfolio/analyze.py": 850,
  "portfolio/bigbet.py": 620,
  "portfolio/prophecy.py": 393,
  "portfolio/qwen3_signal.py": 216,
  "portfolio/circuit_breaker.py": 135,
  "portfolio/cumulative_tracker.py": 218,
  "portfolio/decision_outcome_tracker.py": 106
}
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
helpers ready
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
560:            return name, None
561:
562:    max_workers = max(1, min(len(active_items), 8))
563:
564:    # BUG-178: Add timeout to prevent indefinite hangs from stuck tickers.
565:    #
566:    # Timeline:
567:    # - original: 120s (assumed 60s cycle cadence)
568:    # - 2026-04-09 (CPU fingpt daemon era): 500s — bumped because the CPU
569:    #   fingpt daemon was serializing every ticker's sentiment behind its
570:    #   own global lock, stretching per-ticker latency to ~75s × 5 tickers
571:    #   = ~375s tail. 500s was 2x that max.
572:    # - 2026-04-09 (post feat/fingpt-in-llmbatch): 180s. The fingpt daemon
573:    #   was retired; fingpt moved to portfolio.llm_batch as a post-cycle
574:    #   phase via llama_server full GPU offload. Per-ticker work no longer
575:    #   serialized on fingpt. Live measurement after the merge showed
576:    #   cycles dropping from ~472s to ~226s with 45s/ticker average.
577:    #   180s = 4x the observed per-ticker average and 2x the target "slow"
578:    #   cycle of 90s, a comfortable safety margin for genuinely stuck
579:    #   tickers (network timeouts, yfinance blocking).
580:    # - 2026-04-15: 360s. Telegram alerts at 10:34 showed recurring BUG-178
581:    #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
582:    #   completing 330-525s into the cycle, all 5 within ~10s of each
583:    #   other — the signature of a shared-resource wait rather than truly
584:    #   stuck work. Since 2026-04-09 the ticker path has grown (vix_term_-
585:    #   structure, DXY intraday cross-asset, per-ticker signal gating,
586:    #   fundamental correlation cluster, per-ticker directional accuracy,
587:    #   ETH qwen3 gate) and the llama_server rotation (2026-04-10) means
588:    #   signals occasionally pull stale/miss data under contention bursts.
589:    #   The old 180s was measured when the system had 12 tickers; with 5
590:    #   tickers and more per-ticker work the cost moved legitimately, not
591:    #   because something is "stuck". 360s is 2.8x the observed p50-slow
592:    #   (~130s) and 0.7x the observed p95-slow (~525s), leaving 240s of
593:    #   margin inside the 600s cadence for post-cycle LLM batch, trigger
594:    #   detection, journal, and telegram. Loop contract's own cycle_dur
595:    #   check at 600s remains the catch-all for genuine hangs. Batch 1 of
596:    #   this fix (phase-level instrumentation in signal_engine) and batch
597:    #   2 (signal_utility TTL cache) ship together so we can see per-phase
598:    #   timing in future slow cycles and the next bump decision is
599:    #   grounded in data, not guesswork. See docs/plans/2026-04-15-bug178-
600:    #   instrumentation-timeout.md for the full rationale.
601:    #
602:    # If cycles start creeping above ~360s again, the first place to look
603:    # is the BUG-178 phase log dumped by the slow-cycle diagnostic below —
604:    # acc_load, utility_overlay, weighted_consensus, penalties, linear_-
605:    # factor, and consensus_gate are each tagged in portfolio.log so a
606:    # real bottleneck is identifiable without guessing.
607:    _TICKER_POOL_TIMEOUT = 360
608:    # OR-I-001: avoid context manager — __exit__ calls shutdown(wait=True)
609:    # which blocks the loop when threads hang past the timeout.
610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
611:    futures = {
612:        pool.submit(_process_ticker, name, source): name
613:        for name, source in active_items
614:    }
615:    try:
616:        for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
617:            name, result = future.result()
618:            if result is not None:
619:                tf_data[name] = result["tfs"]
620:                prices_usd[name] = result["price"]
621:                signals[name] = {
622:                    "action": result["action"],
623:                    "confidence": result["confidence"],
624:                    "indicators": result["ind"],
625:                    "extra": result["extra"],
626:                }
627:                signals_ok += 1
628:            else:
629:                signals_failed += 1
630:    except TimeoutError:
631:        timed_out = [n for f, n in futures.items() if not f.done()]
632:        try:
633:            from portfolio.signal_engine import get_last_signal as _get_last
634:            from portfolio.signal_engine import get_phase_log as _get_phase_log
635:            last_sigs = {n: _get_last(n) for n in timed_out}
636:            # 2026-04-15: also dump per-ticker phase breakdown when the pool
637:            # times out. This tells us WHICH post-dispatch phase
638:            # (acc_load / utility_overlay / weighted_consensus / penalties /
639:            # linear_factor / consensus_gate / regime_gate) burned the time,
640:            # so we can target the real bottleneck instead of coarsely blaming
641:            # __post_dispatch__.
642:            phase_logs = {n: _get_phase_log(n) for n in timed_out}
643:        except Exception:
644:            last_sigs = {}
645:            phase_logs = {}
646:        logger.error(
647:            "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
648:            _TICKER_POOL_TIMEOUT, timed_out, last_sigs,
649:        )
650:        for name, phases in phase_logs.items():
651:            if phases:
652:                # Format as 'phase=dur_s' pairs, one ticker per line. Keep on
653:                # one log line per ticker so Windows Event Log / tail -f stays
654:                # readable when 5 tickers time out simultaneously.
655:                phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
656:                logger.error("BUG-178 phases [%s]: %s", name, phase_str)
657:        for f in futures:
658:            f.cancel()
659:        signals_failed += len(timed_out)
660:    finally:
661:        pool.shutdown(wait=False, cancel_futures=True)
662:
663:    # --- Post-cycle LLM batch flush ---
664:    # Ministral/Qwen3/fingpt cache misses were enqueued during parallel
665:    # ticker processing. Now flush them sequentially, grouped by model
666:    # (max 2 swaps: ministral → qwen3 → fingpt). Fingpt phase added
667:    # 2026-04-09 as part of feat/fingpt-in-llmbatch which retired the
668:    # bespoke scripts/fingpt_daemon.py. The sentiment A/B log write is
669:    # also deferred: flush_ab_log() below walks sentiment._pending_ab_entries
670:    # and assembles the final rows once Phase 3 has stashed fingpt results.
671:    try:
672:        from portfolio.llm_batch import _lock as _llm_lock
673:        from portfolio.llm_batch import _ministral_queue, _qwen3_queue, flush_llm_batch
674:        from portfolio.shared_state import MINISTRAL_TTL, _update_cache
675:        # H24/SS2: Capture queued keys before flush to clear stuck loading keys.
676:        with _llm_lock:
677:            _queued_keys = {k for k, _ in _ministral_queue} | {k for k, _ in _qwen3_queue}
678:        batch_results = flush_llm_batch()
679:        for cache_key, result in batch_results.items():
680:            _update_cache(cache_key, result, ttl=MINISTRAL_TTL)
681:        # Clear loading keys for items that didn't return results (retry next cycle).
682:        for key in _queued_keys:
683:            if key not in batch_results:
684:                _update_cache(key, None, ttl=60)
685:        # Now that Phase 3 (fingpt) has stashed its results into
686:        # sentiment._pending_ab_entries via _stash_fingpt_result, write out
687:        # the sentiment_ab_log.jsonl rows for this cycle. Must run AFTER
688:        # flush_llm_batch so the fingpt shadow data is available.
689:        from portfolio.sentiment import flush_ab_log
690:        flush_ab_log()
691:        report.llm_batch_flushed = True
692:    except Exception as e_batch:
693:        logger.warning("LLM batch flush failed: %s", e_batch)
694:        report.errors.append(("llm_batch_flush", str(e_batch)))
695:
696:    _run_elapsed = time.monotonic() - _run_start
697:    logger.info(
698:        "Signal loop done: %d OK, %d failed in %.1fs (%.1fs/ticker avg)",
699:        signals_ok, signals_failed, _run_elapsed,
700:        _run_elapsed / max(signals_ok + signals_failed, 1),
701:    )
702:
703:    # BUG-178 slow-cycle diagnostic (added 2026-04-10, diag/bug178-end-of-
704:    # cycle-snapshot). The ticker pool BUG-178 handler already logs per-ticker
705:    # last_signal state on its 180 s timeout, but cycles that stay under 180 s
706:    # never fire the handler — so slow paths in the 120-180 s range hide from
707:    # us. Fire a warning-level diagnostic when a cycle exceeds 120 s so we
708:    # capture per-ticker phase state retrospectively.
709:    #
710:    # Each value in last_sigs is (sig_name, elapsed_since_set) where sig_name
711:    # is one of: __pre_dispatch__ (hung in sentiment/fear_greed/LLM enqueue),
712:    # a concrete enhanced signal name (hung in the dispatch loop on that one),
713:    # or __post_dispatch__ (hung in accuracy_stats / consensus / per-ticker
714:    # gating). The `elapsed_since_set` value is how long ago the tracker was
715:    # updated — if the cycle total is 150 s but elapsed_since_set for a
716:    # ticker is only 2 s, the slow code is AFTER the last-tracked marker;
717:    # if elapsed_since_set is ~150 s, the thread was stuck at that marker.
718:    if _run_elapsed > 120:
719:        try:
720:            from portfolio.signal_engine import get_last_signal as _get_last
721:            from portfolio.signal_engine import get_phase_log as _get_phase_log
722:            # Use signals.keys() because those are the tickers that successfully
723:            # returned from the pool. Timed-out tickers are already named by the
724:            # BUG-178 handler's Last signals log line.
725:            last_sigs = {n: _get_last(n) for n in signals}
726:            logger.warning(
727:                "Slow cycle diagnostic: %.1fs total, last signals tracked: %s",
728:                _run_elapsed, last_sigs,
729:            )
730:            # 2026-04-15: also dump the post-dispatch phase breakdown for each
731:            # ticker that returned successfully. On a slow cycle the phase log
732:            # reveals which named phase (acc_load, utility_overlay, weighted_-
733:            # consensus, penalties, linear_factor, consensus_gate, regime_gate)
734:            # burned the budget — otherwise we only see the aggregate and can't
735:            # target the fix.
736:            for name in signals:
737:                phases = _get_phase_log(name)
738:                if phases:
739:                    phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
740:                    logger.warning("Slow cycle phases [%s]: %s", name, phase_str)
741:        except Exception as e:
742:            logger.debug("Slow cycle diagnostic failed: %s", e)
743:
744:    # BUG-85: Flush batched sentiment state to disk once per cycle (not per-ticker)
745:    try:
746:        from portfolio.signal_engine import flush_sentiment_state
747:        flush_sentiment_state()
748:    except Exception:
749:        logger.warning("Failed to flush sentiment state", exc_info=True)
750:
751:    # --- Cycle failure alert via Telegram ---
752:    # Collect per-ticker signal failures from this cycle
753:    _cycle_signal_failures = {}
754:    for _tk, _sig in signals.items():
755:        _sf = _sig.get("extra", {}).get("_signal_failures", [])
756:        if _sf:
757:            _cycle_signal_failures[_tk] = _sf
758:
759:    if signals_failed > 0 or _cycle_signal_failures:
760:        _parts = []
761:        if signals_failed > 0:
762:            _parts.append(f"{signals_failed} ticker(s) failed entirely")
763:        if _cycle_signal_failures:
764:            _sf_total = sum(len(v) for v in _cycle_signal_failures.values())
765:            _sf_tickers = ", ".join(
766:                f"{tk}({len(sigs)})" for tk, sigs in _cycle_signal_failures.items()
767:            )
768:            _parts.append(f"{_sf_total} signal failures: {_sf_tickers}")
769:        _fail_msg = f"*LOOP ERRORS* ({int(_run_elapsed)}s cycle)\n" + "\n".join(_parts)
770:        try:
771:            from portfolio.message_store import send_or_store
772:            send_or_store(_fail_msg, config, category="error")
773:        except Exception as _e:
774:            logger.warning("Failed to send cycle error alert: %s", _e)
775:
776:    total = portfolio_value(state, prices_usd, fx_rate)
777:    # BUG-103: Guard against zero/missing initial_value_sek to prevent ZeroDivisionError
778:    initial_val = state.get("initial_value_sek") or INITIAL_CASH_SEK
779:    pnl_pct = ((total - initial_val) / initial_val) * 100
780:    logger.info("Portfolio: %s SEK (%+.2f%%) | Cash: %s SEK", f"{total:,.0f}", pnl_pct, "{:,.0f}".format(state['cash_sek']))
781:
782:    if not STATE_FILE.exists():
783:        save_state(state)
784:
785:    # Log hourly price snapshot for cumulative tracking
786:    try:
787:        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
788:        maybe_log_hourly_snapshot(prices_usd)
789:    except Exception as e:
790:        logger.warning("hourly snapshot failed: %s", e)
791:
792:    # Smart trigger
793:    from portfolio.trigger import check_triggers
794:
795:    fear_greeds = {}
796:    sentiments = {}
797:    for name, sig in signals.items():
798:        extra = sig.get("extra", {})
799:        if "fear_greed" in extra:
800:            fear_greeds[name] = {
801:                "value": extra["fear_greed"],
802:                "classification": extra.get("fear_greed_class", ""),
803:            }
804:        if "sentiment" in extra:
805:            sentiments[name] = extra["sentiment"]
806:
807:    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)
808:
809:    if triggered or force_report:
810:        reasons_list = reasons if reasons else ["startup"]
811:        summary = write_agent_summary(signals, prices_usd, fx_rate, state, tf_data, reasons_list)
812:        report.summary_written = True
813:        logger.info("Trigger: %s", ', '.join(reasons_list))
814:
815:        # Classify tier and write tier-specific context
816:        from portfolio.reporting import write_tiered_summary
817:        from portfolio.trigger import classify_tier, update_tier_state
818:        tier = classify_tier(reasons_list)
819:        triggered_tickers = _extract_triggered_tickers(reasons_list)
820:        write_tiered_summary(summary, tier, triggered_tickers)
821:        update_tier_state(tier)
822:        logger.info("Tier: T%d (%s)", tier, TIER_CONFIG.get(tier, {}).get('label', 'UNKNOWN'))
823:
824:        try:
825:            from portfolio.outcome_tracker import log_signal_snapshot
826:            log_signal_snapshot(signals, prices_usd, fx_rate, reasons_list)
827:        except Exception as e:
828:            logger.warning("signal logging failed: %s", e)
829:
830:        # 2026-05-04: Wrap long-blocking work (Layer 2 T2/T3 = 600-900s
831:        # subprocess; autonomous fallback = bounded but not instant) in a
832:        # heartbeat keepalive. update_health() (the normal heartbeat write)
833:        # only runs at end-of-cycle, so without periodic ticks the
834:        # dashboard /api/health flips stale 300s into any triggering cycle
835:        # even though the loop is alive and waiting on Claude CLI.
836:        # The context manager's __exit__ runs on exceptions too, so the
837:        # daemon thread is always cleaned up. Skip-paths (NO_TELEGRAM,
838:        # outside agent window) are NOT wrapped — they don't block.
839:        from portfolio.health import heartbeat_keepalive
840:
841:        layer2_cfg = config.get("layer2", {})
842:        if os.environ.get("NO_TELEGRAM"):
843:            logger.info("[NO_TELEGRAM] Skipping agent invocation")
844:            _log_trigger(reasons_list, "skipped_test", tier=tier)
845:        elif layer2_cfg.get("enabled", True):
846:            if _is_agent_window():
847:                with heartbeat_keepalive():
848:                    result = invoke_agent(reasons_list, tier=tier)
849:                _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
850:            else:
851:                logger.info("Layer 2: outside market window, skipping")
852:                _log_trigger(reasons_list, "skipped_offhours", tier=tier)
853:        else:
854:            logger.info("Layer 2 disabled — autonomous mode")
855:            from portfolio.autonomous import autonomous_decision
856:            with heartbeat_keepalive():
857:                autonomous_decision(
858:                    config, signals, prices_usd, fx_rate, state,
859:                    reasons_list, tf_data, tier, triggered_tickers,
860:                )
861:            _log_trigger(reasons_list, "autonomous", tier=tier)
862:    else:
863:        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
864:        report.summary_written = True
865:        logger.info("No trigger.")
866:
867:    # Big Bet detection — can invoke a 30s Claude subprocess per qualifying
868:    # candidate (portfolio/bigbet.py:invoke_layer2_eval), with no per-cycle
869:    # cap. Wrapped in keepalive so heartbeat stays fresh across multi-minute
870:    # bigbet sweeps that would otherwise re-trip the dashboard stale gate.
871:    bigbet_cfg = config.get("bigbet", {})
872:    if bigbet_cfg.get("enabled", False):
873:        try:
874:            from portfolio.bigbet import check_bigbet
875:            from portfolio.health import heartbeat_keepalive
876:            with heartbeat_keepalive():
877:                check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
878:        except Exception as e:
879:            logger.warning("Big Bet check failed: %s", e)
880:
881:    # ISKBETS monitoring — same shape: each qualifying ticker can fire a 30s
882:    # Claude gate subprocess (portfolio/iskbets.py:invoke_layer2_gate). With
883:    # 5 Tier-1 tickers configured the worst case is ~150s of subprocess work,
884:    # well past the 300s heartbeat threshold when stacked with bigbet+L2.
885:    try:
886:        from portfolio.health import heartbeat_keepalive
887:        from portfolio.iskbets import check_iskbets
888:        with heartbeat_keepalive():
889:            check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
890:    except Exception as e:
891:        logger.warning("ISKBETS check failed: %s", e)
892:
893:    # Avanza pending order confirmations
894:    try:
895:        from portfolio.avanza_orders import check_pending_orders
896:        check_pending_orders(config)
897:    except Exception as e:
898:        logger.warning("Avanza order check failed: %s", e)
899:
900:    # Periodic trade reflection
901:    try:
902:        from portfolio.reflection import maybe_reflect
903:        maybe_reflect(config)
904:    except Exception as e:
905:        logger.warning("reflection check failed: %s", e)
906:
907:    # Health update
908:    try:
909:        from portfolio.health import update_health
910:        trigger_reason = reasons[0] if (triggered or force_report) and reasons else None
911:        update_health(
912:            cycle_count=_ss._run_cycle_id,
913:            signals_ok=signals_ok,
914:            signals_failed=signals_failed,
915:            last_trigger_reason=trigger_reason,
916:        )
917:        report.health_updated = True
918:    except Exception as e:
919:        logger.warning("health update failed: %s", e)
920:        report.errors.append(("health_update", str(e)))
921:
922:    # Periodic safeguard checks (every 100 cycles ≈ 100 min)
923:    if _ss._run_cycle_id % 100 == 0 and _ss._run_cycle_id > 0:
924:        try:
925:            from portfolio.health import check_dead_signals, check_outcome_staleness
926:            outcome_status = check_outcome_staleness()
927:            if outcome_status["stale"]:
928:                age = outcome_status["newest_outcome_age_hours"]
929:                msg = (f"⚠️ SAFEGUARD: Outcome backfill stale! "
930:                       f"Newest outcome: {age:.0f}h ago. "
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
1:"""Smart trigger system — detects meaningful market changes to reduce noise.
2:
3:Layer 1 runs on a 10-minute cadence during every market state (see
4:``portfolio/market_timing.py:INTERVAL_MARKET_OPEN``). Layer 2 is invoked when:
5:- Signal consensus: any ticker NEWLY reaches BUY or SELL from HOLD
6:- Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (see below)
7:- Price moved >2% since last trigger
8:- Fear & Greed crossed extreme threshold (20 or 80)
9:- Sentiment reversal: sustained for SUSTAINED_CHECKS cycles (filters oscillation)
10:- Post-trade reassessment: after a BUY/SELL trade
11:
12:No periodic cooldown — Layer 2 is only invoked when Layer 1 detects a
13:meaningful change. The Tier 3 periodic full review (every 2h market / 4h
14:off-hours) provides the "heartbeat" via classify_tier(), but only when
15:another trigger has already fired.
16:"""
17:
18:import logging
19:import os
20:import re
21:import time
22:from datetime import UTC, datetime
23:from pathlib import Path
24:
25:from portfolio.file_utils import atomic_write_json, load_json
26:
27:logger = logging.getLogger("portfolio.trigger")
28:
29:BASE_DIR = Path(__file__).resolve().parent.parent
30:STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
31:PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
32:PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"
33:
34:PRICE_THRESHOLD = 0.02  # 2% move
35:FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
36:# A signal flip triggers Layer 2 when EITHER of these holds:
37:#   - SUSTAINED_CHECKS consecutive cycles show the new action, OR
38:#   - SUSTAINED_DURATION_S seconds of wall-clock time have elapsed since
39:#     the flip first appeared.
40:# The count path is the original behavior (unchanged at the 60s cadence).
41:# The duration path is new (added 2026-04-09 with the cadence bump to 600s);
42:# at 600s cadence the count path would require ≥30 min of sustained flip
43:# before triggering, which effectively disables the trigger for fast-moving
44:# events. The duration gate bounds the worst case to ~1 cycle after flip
45:# (≈10 min at 600s cadence, ≈2 min at 60s cadence — both unchanged or better
46:# than the old count-only behavior).
47:SUSTAINED_CHECKS = 3
48:SUSTAINED_DURATION_S = 120
49:
50:# Per-ticker flip cooldown (2026-05-08): after a sustained flip fires a Layer 2
51:# trigger, suppress further sustained flip triggers for the SAME ticker for
52:# FLIP_COOLDOWN_S seconds.  Prevents whiplash where volatile tickers (e.g. MSTR)
53:# produce 3+ sustained flips in under an hour, each invoking Layer 2 for a HOLD.
54:# Does NOT suppress consensus triggers (section 1), price moves (section 3), or
55:# F&G crossings (section 4) — only section-2 sustained flips.
56:FLIP_COOLDOWN_S = 1800  # 30 min
57:
58:# Ranging regime dampening (2026-04-22): when a ticker's regime is "ranging",
59:# require a minimum consensus confidence before triggering Layer 2. In ranging
60:# markets, consensus oscillates between HOLD and weak BUY/SELL, producing 20+
61:# Layer 2 invocations per day that all return HOLD — wasting compute and token
62:# budget. Setting this to 0.0 disables dampening without code change.
63:RANGING_CONSENSUS_MIN_CONFIDENCE = 0.35
64:
65:# Startup grace period — after a restart, the first loop iteration updates the
66:# baseline without triggering Layer 2. This prevents spurious T3 full reviews
67:# every time the loop is restarted for a code update.
68:_GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
69:_startup_grace_active = True  # True until first check_triggers call completes
70:
71:
72:def _update_sustained(
73:    state_dict: dict, key: str, value, now_ts: float
74:) -> tuple[bool, bool]:
75:    """Update sustained-debounce state for a key and return gate results.
76:
77:    Shared by signal flip (section 2) and sentiment reversal (section 5).
78:    Increments count if value unchanged, resets if changed. Returns
79:    (count_ok, duration_ok) indicating whether either debounce gate passed.
80:
81:    Duration tracking uses time.monotonic() internally to avoid NTP-jump
82:    false negatives. On process restart, monotonic origin resets and the
83:    duration gate conservatively starts fresh (correct behavior — a
84:    restart already resets the sustained counter).
85:    """
86:    mono_now = time.monotonic()
87:    prev = state_dict.get(key, {})
88:    if prev.get("value") == value:
89:        state_dict[key] = {
90:            "value": value,
91:            "count": prev["count"] + 1,
92:            "_mono_start": prev.get("_mono_start", mono_now),
93:        }
94:    else:
95:        state_dict[key] = {
96:            "value": value,
97:            "count": 1,
98:            "_mono_start": mono_now,
99:        }
100:    entry = state_dict[key]
101:    count_ok = entry["count"] >= SUSTAINED_CHECKS
102:    duration_ok = (mono_now - entry["_mono_start"]) >= SUSTAINED_DURATION_S
103:    return count_ok, duration_ok
104:
105:
106:def _today_str():
107:    return datetime.now(UTC).strftime("%Y-%m-%d")
108:
109:
110:def _load_state():
111:    return load_json(STATE_FILE, default={})
112:
113:
114:def _save_state(state):
115:    # Prune triggered_consensus entries for tickers not in current signals
116:    # to prevent unbounded growth when tickers are removed from tracking
117:    tc = state.get("triggered_consensus", {})
118:    current_tickers = state.get("_current_tickers")
119:    if current_tickers is not None:
120:        removed = {k for k in tc if k not in current_tickers}
121:        if removed:
122:            logger.info("trigger: pruning %d stale ticker(s) from baseline: %s", len(removed), ", ".join(sorted(removed)))
123:        pruned = {k: v for k, v in tc.items() if k in current_tickers}
124:        state["triggered_consensus"] = pruned
125:    state.pop("_current_tickers", None)  # don't persist internal field
126:    atomic_write_json(STATE_FILE, state)
127:
128:
129:def _check_recent_trade(state):
130:    """Check if Layer 2 executed a trade since our last trigger.
131:
132:    Returns True if a recent trade was detected.
133:    """
134:    last_checked_tx = state.get("last_checked_tx_count", {})
135:
136:    trade_detected = False
137:    new_tx_counts = {}
138:
139:    for label, pf_file in [("patient", PORTFOLIO_FILE), ("bold", PORTFOLIO_BOLD_FILE)]:
140:        try:
141:            pf = load_json(pf_file, default=None)
142:            if pf is None:
143:                continue
144:            txs = pf.get("transactions", [])
145:            current_count = len(txs)
146:            prev_count = last_checked_tx.get(label, current_count)
147:            new_tx_counts[label] = current_count
148:
149:            if current_count > prev_count:
150:                trade_detected = True
151:        except (KeyError, AttributeError) as exc:
152:            logger.warning("Failed to parse portfolio file %s: %s", pf_file, exc)
153:
154:    if new_tx_counts:
155:        state["last_checked_tx_count"] = new_tx_counts
156:
157:    return trade_detected
158:
159:
160:def check_triggers(signals, prices_usd, fear_greeds, sentiments):
161:    global _startup_grace_active
162:    state = _load_state()
163:    state["_current_tickers"] = set(signals.keys())  # for pruning in _save_state
164:
165:    # Startup grace period: on the first iteration after a restart, update the
166:    # baseline (prices, signals, consensus) WITHOUT triggering Layer 2.
167:    # This lets the loop restart for code updates without spurious T3 reviews.
168:    current_pid = os.getpid()
169:    saved_pid = state.get(_GRACE_PERIOD_KEY)
170:    if _startup_grace_active and saved_pid != current_pid:
171:        import logging
172:        _logger = logging.getLogger("portfolio.trigger")
173:        _logger.info(
174:            "Startup grace period: updating baseline without triggering "
175:            "(pid %s -> %s)", saved_pid, current_pid,
176:        )
177:        state[_GRACE_PERIOD_KEY] = current_pid
178:        # Update baselines so next iteration compares from NOW
179:        state["last"] = {
180:            "signals": {
181:                t: {"action": s["action"], "confidence": s["confidence"]}
182:                for t, s in signals.items()
183:            },
184:            "prices": dict(prices_usd),
185:            "fear_greeds": {
186:                t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
187:            },
188:            "sentiments": dict(sentiments),
189:            "time": time.time(),
190:        }
191:        # Update triggered_consensus baseline to current state
192:        tc = state.get("triggered_consensus", {})
193:        for ticker, sig in signals.items():
194:            tc[ticker] = sig["action"]
195:        state["triggered_consensus"] = tc
196:        state["today_date"] = _today_str()
197:        _startup_grace_active = False
198:        _save_state(state)
199:        return False, []
200:
201:    _startup_grace_active = False
202:    prev = state.get("last", {})
203:    sustained = state.get("sustained_counts", {})
204:    reasons = []
205:
206:    # 0. Trade reset — if Layer 2 made a trade, trigger reassessment
207:    if _check_recent_trade(state):
208:        state["last_trigger_time"] = 0
209:        reasons.append("post-trade reassessment")
210:
211:    # 1. Signal consensus — trigger ONLY when a ticker first reaches BUY/SELL
212:    #    from HOLD. BUY↔SELL direction flips are handled by the sustained flip
213:    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
214:    #    when unrelated triggers (sentiment, etc.) fire.
215:    #
216:    #    Ranging regime dampening (2026-04-22): in ranging regime, low-confidence
217:    #    consensus crossings are noise — require RANGING_CONSENSUS_MIN_CONFIDENCE
218:    #    to actually fire the trigger. Prevents 20+ HOLD invocations per day.
219:    triggered_consensus = state.get("triggered_consensus", {})
220:    for ticker, sig in signals.items():
221:        action = sig["action"]
222:        last_tc = triggered_consensus.get(ticker, "HOLD")
223:        if action in ("BUY", "SELL") and last_tc == "HOLD":
224:            conf = sig.get("confidence", 0)
225:            # Ranging regime dampening: skip low-confidence consensus triggers
226:            ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
227:            if (
228:                ticker_regime == "ranging"
229:                and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
230:                and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
231:            ):
232:                logger.info(
233:                    "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
234:                    "(min %.0f%%)",
235:                    ticker, action, conf * 100,
236:                    RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
237:                )
238:                # Still update baseline so we don't re-trigger next cycle
239:                triggered_consensus[ticker] = action
240:                continue
241:            # New consensus from HOLD — trigger
242:            reasons.append(f"{ticker} consensus {action} ({conf:.0%})")
243:            triggered_consensus[ticker] = action
244:        elif action == "HOLD" and last_tc != "HOLD":
245:            # Consensus cleared — reset so next BUY/SELL is "new"
246:            triggered_consensus[ticker] = "HOLD"
247:        elif action in ("BUY", "SELL") and action != last_tc:
248:            # Direction flip (BUY↔SELL) — update baseline silently,
249:            # let sustained flip trigger (#2) handle it
250:            triggered_consensus[ticker] = action
251:    state["triggered_consensus"] = triggered_consensus
252:
253:    # 2. Signal flip — triggers when the new action has been seen for
254:    #    SUSTAINED_CHECKS consecutive cycles OR for SUSTAINED_DURATION_S
255:    #    wall-clock seconds, whichever comes first. The duration gate was
256:    #    added 2026-04-09 so the trigger still fires within ~1 cycle at
257:    #    long cadences (e.g. 600s); at the historical 60s cadence the count
258:    #    gate still dominates and behavior is unchanged.
259:    prev_triggered = prev.get("signals", {})
260:    flip_cooldowns = state.get("flip_cooldowns", {})
261:    _flip_now_ts = time.time()
262:    for ticker, sig in signals.items():
263:        current_action = sig["action"]
264:        count_ok, duration_ok = _update_sustained(
265:            sustained, ticker, current_action, _flip_now_ts,
266:        )
267:
268:        triggered_action = prev_triggered.get(ticker, {}).get("action")
269:        if triggered_action and current_action != triggered_action and (count_ok or duration_ok):
270:            last_flip_ts = flip_cooldowns.get(ticker, 0)
271:            if (_flip_now_ts - last_flip_ts) < FLIP_COOLDOWN_S:
272:                logger.info(
273:                    "Flip cooldown: %s %s->%s suppressed (%.0fs remaining)",
274:                    ticker, triggered_action, current_action,
275:                    FLIP_COOLDOWN_S - (_flip_now_ts - last_flip_ts),
276:                )
277:                continue
278:            flip_cooldowns[ticker] = _flip_now_ts
279:            reasons.append(
280:                f"{ticker} flipped {triggered_action}->{current_action} (sustained)"
281:            )
282:    state["flip_cooldowns"] = flip_cooldowns
283:
284:    # 3. Price move >2% since last trigger
285:    prev_prices = prev.get("prices", {})
286:    for ticker, price in prices_usd.items():
287:        old_price = prev_prices.get(ticker)
288:        if old_price and old_price > 0:
289:            pct = abs(price - old_price) / old_price
290:            if pct >= PRICE_THRESHOLD:
291:                direction = "up" if price > old_price else "down"
292:                reasons.append(f"{ticker} moved {pct:.1%} {direction}")
293:
294:    # 4. Fear & Greed crossed threshold
295:    prev_fg = prev.get("fear_greeds", {})
296:    for ticker, fg in fear_greeds.items():
297:        val = fg.get("value", 50) if isinstance(fg, dict) else 50
298:        old_val = (
299:            prev_fg.get(ticker, {}).get("value", 50)
300:            if isinstance(prev_fg.get(ticker), dict)
301:            else 50
302:        )
303:        for threshold in FG_THRESHOLDS:
304:            if (old_val > threshold) != (val > threshold):
305:                reasons.append(f"F&G crossed {threshold} ({old_val}->{val})")
306:                break
307:
308:    # 5. Sentiment reversal — same OR-debounce as section 2.
309:    sustained_sent = state.get("sustained_sentiment", {})
310:    stable_sent = state.get("stable_sentiment", {})
311:    _sent_now_ts = time.time()
312:    for ticker, sent in sentiments.items():
313:        count_ok, duration_ok = _update_sustained(
314:            sustained_sent, ticker, sent, _sent_now_ts,
315:        )
316:        if count_ok or duration_ok:
317:            last_stable = stable_sent.get(ticker)
318:            if (
319:                last_stable
320:                and last_stable != sent
321:                and sent != "neutral"
322:                and last_stable != "neutral"
323:            ):
324:                reasons.append(
325:                    f"{ticker} sentiment {last_stable}->{sent} (sustained)"
326:                )
327:            stable_sent[ticker] = sent
328:    state["sustained_sentiment"] = sustained_sent
329:    state["stable_sentiment"] = stable_sent
330:
331:    triggered = len(reasons) > 0
332:
333:    if triggered:
334:        state["last_trigger_time"] = time.time()
335:        state["last"] = {
336:            "signals": {
337:                t: {"action": s["action"], "confidence": s["confidence"]}
338:                for t, s in signals.items()
339:            },
340:            "prices": dict(prices_usd),
341:            "fear_greeds": {
342:                t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
343:            },
344:            "sentiments": dict(sentiments),
345:            "time": time.time(),
346:        }
347:        # C4/NEW-2: only update last_trigger_date when a real trigger fires, so that
348:        # classify_tier() can correctly detect the first real trigger of the day.
349:        state["last_trigger_date"] = _today_str()
350:
351:    # Track today_date for other purposes
352:    state["today_date"] = _today_str()
353:
354:    state["sustained_counts"] = sustained
355:    _save_state(state)
356:
357:    return triggered, reasons
358:
359:
360:# ---------------------------------------------------------------------------
361:# Tier classification
362:# ---------------------------------------------------------------------------
363:
364:# Full review interval: 4h during market hours, 4h off-hours (T1 only)
365:_FULL_REVIEW_MARKET_HOURS = 4
366:_FULL_REVIEW_OFF_HOURS = 4  # Off-hours caps at T1, not T3
367:
368:# Option P (2026-04-17): confidence-aware tier downshift.
369:# When every reason in a T2 trigger is either a low-conviction consensus
370:# crossing (<TIER_DOWNSHIFT_CONFIDENCE) or a fade flip (*->HOLD sustained),
371:# downshift T2 -> T1 to save Claude token budget. T3 triggers (first-of-day,
372:# F&G extreme, periodic full review) are NEVER downshifted. Sustained
373:# direction flips (BUY<->SELL) and non-consensus triggers (post-trade, price
374:# move, sentiment) block downshift. Setting this to 0.0 disables downshift
375:# without code change.
376:TIER_DOWNSHIFT_CONFIDENCE = 0.40
377:
378:# Precompiled patterns for downshift eligibility analysis on reason strings
379:# produced by check_triggers(). Reason shape stays stable across releases;
380:# if the format ever changes, these miss -> downshift fails open (tier
381:# stays T2, safe over-invocation rather than under-invocation).
382:#
383:# Word boundaries (\b) on "consensus" and "flipped" prevent substring
384:# collisions — e.g. a hypothetical future reason containing "nonconsensus"
385:# or "preflipped" would NOT accidentally match and trigger a downshift.
386:# Current check_triggers has no such reasons, but anchoring is cheap
387:# insurance against future regressions. Added 2026-04-17 after an
388:# adversarial self-review surfaced the issue.
389:_CONSENSUS_CONF_RE = re.compile(r'\bconsensus (?:BUY|SELL) \((\d+)%\)')
390:_FADE_FLIP_RE = re.compile(r'\bflipped (?:BUY|SELL)->HOLD \(sustained\)')
391:
392:
393:def _reason_is_downshiftable(reason: str, threshold: float) -> bool:
394:    """Return True if this reason is low-conviction enough to allow T2->T1.
395:
396:    A reason qualifies if it is either:
397:      - A consensus crossing with confidence < threshold, or
398:      - A fade flip (*->HOLD sustained).
399:
400:    Any other reason type (direction flip, post-trade, price move, F&G,
401:    sentiment, startup) returns False and blocks downshift for the whole
402:    reason list.
403:    """
404:    m = _CONSENSUS_CONF_RE.search(reason)
405:    if m:
406:        conf_pct = int(m.group(1))
407:        return conf_pct < threshold * 100
408:    return bool(_FADE_FLIP_RE.search(reason))
409:
410:
411:def _should_downshift_to_t1(reasons, threshold: float | None = None) -> bool:
412:    """Decide whether a T2 tier should be downshifted to T1.
413:
414:    Returns True only when every reason is either a low-conviction consensus
415:    crossing or a fade flip — i.e. all reasons are individually downshiftable.
416:    A single high-conviction or non-consensus reason blocks downshift.
417:
418:    Empty reason list returns False (no downshift). Called only after
419:    classify_tier() has already chosen T2 — T1 and T3 are never affected.
420:
421:    threshold=None (default) looks up TIER_DOWNSHIFT_CONFIDENCE at call time,
422:    allowing runtime overrides via mock.patch or module-attribute reassignment
423:    (the module-level constant is the single config knob). Passing an explicit
424:    float overrides for testing.
425:    """
426:    if not reasons:
427:        return False
428:    effective = TIER_DOWNSHIFT_CONFIDENCE if threshold is None else threshold
429:    return all(_reason_is_downshiftable(r, effective) for r in reasons)
430:
431:
432:def classify_tier(reasons, state=None):
433:    """Classify trigger reasons into invocation tier (1=quick, 2=signal, 3=full).
434:
435:    Tier 3 (Full Review): periodic review, F&G extreme, first of day.
436:    Tier 2 (Signal Analysis): new consensus, price moves, post-trade, signal flips.
437:    Tier 1 (Quick Check): sentiment noise, repeated triggers.
438:
439:    M10/NEW-4: pass state=<dict> to avoid a redundant disk read when the caller
440:    already has the trigger state loaded. Falls back to loading from file if None.
441:    """
442:    if state is None:
443:        state = _load_state()
444:
445:    # Tier 3: periodic full review
446:    last_full = state.get("last_full_review_time", 0)
447:    hours_since = (time.time() - last_full) / 3600
448:
449:    now_utc = datetime.now(UTC)
450:    from portfolio.market_timing import _eu_market_open_hour_utc, _market_close_hour_utc
451:    close_hour = _market_close_hour_utc(now_utc)
452:    eu_open = _eu_market_open_hour_utc(now_utc)
453:    market_open = now_utc.weekday() < 5 and eu_open <= now_utc.hour < close_hour
454:
455:    # C4/NEW-2: first-of-day T3 check must precede the off-hours periodic cap.
456:    # An off-hours trigger 4+ hours after the last full review would otherwise
457:    # return T1 early (line below), skipping the first-of-day T3 entirely.
458:    if state.get("last_trigger_date") != _today_str():
459:        return 3  # first real trigger of the day
460:
461:    if any("F&G crossed" in r for r in reasons):
462:        return 3
463:
464:    if market_open and hours_since >= _FULL_REVIEW_MARKET_HOURS:
465:        return 3
466:    if not market_open and hours_since >= _FULL_REVIEW_OFF_HOURS:
467:        return 1  # T1 quick check only — save T3 budget for market hours
468:
469:    # Tier 2: new actionable signals
470:    tier2_patterns = ["consensus", "moved", "post-trade", "flipped"]
471:    if any(p in r for r in reasons for p in tier2_patterns):
472:        # Option P (2026-04-17): downshift T2 -> T1 when every reason is
473:        # low-conviction (consensus <40% confidence or *->HOLD fade flip).
474:        # Preserves trigger firing + signal/accuracy data; only cuts Claude
475:        # analysis depth on signals that reliably return HOLD anyway.
476:        if _should_downshift_to_t1(reasons):
477:            return 1
478:        return 2
479:
480:    # Tier 1: cooldowns, sentiment noise, repeated triggers
481:    return 1
482:
483:
484:def update_tier_state(tier, state=None):
485:    """Update trigger state after a tier classification.
486:
487:    Called by the main loop after classify_tier() to persist tier-specific state.
488:    M10/NEW-4: accepts an optional state dict to avoid re-reading trigger_state.json.
489:    """
490:    if state is None:
491:        state = _load_state()
492:    if tier == 3:
493:        state["last_full_review_time"] = time.time()
494:    _save_state(state)
495:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Multi-agent Layer 2 orchestration — parallel specialists + synthesis.
2:
3:Inspired by Claude Code's Coordinator Mode. Instead of one monolithic
4:agent reading everything, splits analysis into parallel specialists:
5:
6:    1. Technical Agent: signals, regime, momentum, trend
7:    2. Risk Agent: portfolio state, exposure, drawdown, stops
8:    3. Microstructure Agent: order flow, depth, cross-asset context
9:
10:Each specialist writes a brief report to a temp file. A synthesis agent
11:reads all three and makes the final BUY/SELL/HOLD decision.
12:
13:Key design principles (from Claude Code's Agent architecture):
14:    - Fresh context per agent (no context pollution)
15:    - 5-word task description forces clarity
16:    - Standardized report format for mechanical parsing
17:    - Parent owns the gate — synthesis agent makes final call
18:"""
19:from __future__ import annotations
20:
21:import logging
22:import os
23:import shutil
24:import subprocess
25:import time
26:from pathlib import Path
27:
28:from portfolio.claude_gate import detect_auth_failure
29:
30:logger = logging.getLogger("portfolio.multi_agent_layer2")
31:
32:BASE_DIR = Path(__file__).resolve().parents[1]
33:DATA_DIR = BASE_DIR / "data"
34:
35:SPECIALISTS = {
36:    "technical": {
37:        "focus": "Technical analysis: signals, regime, momentum, trend direction",
38:        "data_files": [
39:            "data/agent_context_t2.json",
40:            "data/accuracy_cache.json",
41:        ],
42:        "output_file": "data/_specialist_technical.md",
43:        "timeout": 120,
44:        "max_turns": 10,
45:    },
46:    "risk": {
47:        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
48:        "data_files": [
49:            "data/portfolio_state.json",
50:            "data/portfolio_state_bold.json",
51:            "data/portfolio_state_warrants.json",
52:        ],
53:        "output_file": "data/_specialist_risk.md",
54:        "timeout": 90,
55:        "max_turns": 8,
56:    },
57:    "microstructure": {
58:        "focus": "Order flow and cross-asset: depth imbalance, trade flow, VPIN, copper, GVZ, gold/silver ratio",
59:        "data_files": [
60:            "data/microstructure_state.json",
61:            "data/seasonality_profiles.json",
62:        ],
63:        "output_file": "data/_specialist_microstructure.md",
64:        "timeout": 90,
65:        "max_turns": 8,
66:    },
67:}
68:
69:
70:def build_specialist_prompts(
71:    ticker: str,
72:    trigger_reasons: list[str],
73:) -> dict[str, str]:
74:    """Build prompts for each specialist agent.
75:
76:    Returns dict keyed by specialist name with prompt strings.
77:    """
78:    reason_str = ", ".join(trigger_reasons[:5])
79:    prompts = {}
80:
81:    for name, spec in SPECIALISTS.items():
82:        data_reads = " ".join(f"Read {f}." for f in spec["data_files"])
83:        prompts[name] = (
84:            f"You are a {name} specialist for the trading system. "
85:            f"Ticker: {ticker}. Trigger: {reason_str}. "
86:            f"Focus: {spec['focus']}. "
87:            f"{data_reads} "
88:            f"Write a brief analysis (max 500 words) to {spec['output_file']}. "
89:            "Include: current state, key signals, recommendation "
90:            "(bullish/bearish/neutral), and confidence (low/medium/high). "
91:            "Be concise and data-driven. Do NOT make trade decisions."
92:        )
93:
94:    return prompts
95:
96:
97:def build_synthesis_prompt(
98:    ticker: str,
99:    trigger_reasons: list[str],
100:    report_paths: list[str] | None = None,
101:) -> str:
102:    """Build the synthesis agent prompt that reads all specialist reports."""
103:    reason_str = ", ".join(trigger_reasons[:5])
104:    if report_paths is None:
105:        report_paths = get_report_paths()
106:    reads = " ".join(f"Read {p}." for p in report_paths)
107:
108:    return (
109:        "You are the Layer 2 synthesis agent. "
110:        f"Ticker: {ticker}. Trigger: {reason_str}. "
111:        "Read docs/TRADING_PLAYBOOK.md for trading rules. "
112:        "If data/trading_insights.md exists, read it for recent performance context. "
113:        f"{reads} "
114:        "These are reports from 3 specialist agents (technical, risk, microstructure). "
115:        "Synthesize their findings into a trading decision for BOTH Patient and Bold strategies. "
116:        "If specialists disagree, explain why you sided with one over the other. "
117:        "Read data/portfolio_state.json and data/portfolio_state_bold.json for current positions. "
118:        "Write journal entry and send Telegram per the playbook."
119:    )
120:
121:
122:def get_report_paths() -> list[str]:
123:    """Get output file paths for all specialists."""
124:    return [spec["output_file"] for spec in SPECIALISTS.values()]
125:
126:
127:def launch_specialists(
128:    ticker: str,
129:    trigger_reasons: list[str],
130:) -> list[subprocess.Popen]:
131:    """Launch all specialist agents in parallel.
132:
133:    Returns list of Popen processes. Caller must wait for them.
134:    """
135:    prompts = build_specialist_prompts(ticker, trigger_reasons)
136:    claude_cmd = shutil.which("claude")
137:    if not claude_cmd:
138:        logger.warning("claude not on PATH, cannot launch specialists")
139:        return []
140:
141:    procs = []
142:    agent_env = os.environ.copy()
143:    agent_env.pop("CLAUDECODE", None)
144:    agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
145:    agent_env["NODE_OPTIONS"] = "--stack-size=16384"
146:    # P2 follow-up (Codex P1 #1, 2026-04-17): specialists spawn as headless
147:    # subprocesses with no interactive stdin, same as invoke_agent. Without
148:    # this, when multi_agent=true fires, three specialist Claude sessions
149:    # hit CLAUDE.md's STARTUP CHECK, see unresolved critical errors, and
150:    # hang asking "How would you like to proceed?" until specialist_timeout_s.
151:    agent_env["PF_HEADLESS_AGENT"] = "1"
152:
153:    for name, prompt in prompts.items():
154:        spec = SPECIALISTS[name]
155:        # 2026-04-13: DO NOT add `--bare` here either. Same reason as
156:        # agent_invocation.py: `--bare` disables OAuth/keychain auth and
157:        # requires ANTHROPIC_API_KEY. User runs Max-subscription OAuth only.
158:        # Commit 857fd45 (2026-04-01) added `--bare` to specialist launches;
159:        # removed 2026-04-13 after confirming it broke all specialist runs.
160:        cmd = [
161:            claude_cmd, "-p", prompt,
162:            "--allowedTools", "Read,Write",
163:            "--max-turns", str(spec["max_turns"]),
164:        ]
165:        try:
166:            log_path = DATA_DIR / f"_specialist_{name}.log"
167:            log_fh = open(log_path, "w", encoding="utf-8")
168:            proc = subprocess.Popen(
169:                cmd,
170:                cwd=str(BASE_DIR),
171:                stdout=log_fh,
172:                stderr=subprocess.STDOUT,
173:                env=agent_env,
174:            )
175:            proc._log_fh = log_fh  # attach for cleanup
176:            proc._name = name
177:            procs.append(proc)
178:            logger.info("Specialist %s launched pid=%s", name, proc.pid)
179:        except Exception as e:
180:            logger.error("Failed to launch specialist %s: %s", name, e)
181:
182:    return procs
183:
184:
185:def wait_for_specialists(
186:    procs: list[subprocess.Popen],
187:    timeout: int = 150,
188:) -> dict[str, bool]:
189:    """Wait for all specialist agents to complete.
190:
191:    Returns dict of specialist_name -> success (True/False).
192:    """
193:    results = {}
194:    deadline = time.time() + timeout
195:
196:    for proc in procs:
197:        remaining = max(1, deadline - time.time())
198:        name = getattr(proc, "_name", "unknown")
199:        try:
200:            proc.wait(timeout=remaining)
201:            success = proc.returncode == 0
202:            results[name] = success
203:            if not success:
204:                logger.warning("Specialist %s exited with code %d", name, proc.returncode)
205:        except subprocess.TimeoutExpired:
206:            logger.warning("Specialist %s timed out, killing", name)
207:            proc.kill()
208:            proc.wait(timeout=5)
209:            results[name] = False
210:        finally:
211:            log_fh = getattr(proc, "_log_fh", None)
212:            if log_fh:
213:                log_fh.close()
214:
215:        # 2026-04-13: Auth-error scan — specialist log is truncated per run
216:        # ("w" mode in launch_specialists), so reading the whole file is safe.
217:        # Override success to False if auth failure detected so synthesis
218:        # doesn't proceed with an empty specialist report masquerading as OK.
219:        try:
220:            log_path = DATA_DIR / f"_specialist_{name}.log"
221:            if log_path.exists():
222:                text = log_path.read_text(encoding="utf-8", errors="replace")
223:                if detect_auth_failure(text, caller=f"layer2_specialist_{name}",
224:                                       context={"specialist": name}):
225:                    results[name] = False
226:        except Exception as e:
227:            logger.warning("Auth-error scan of specialist %s log failed: %s", name, e)
228:
229:    return results
230:
231:
232:def cleanup_reports() -> None:
233:    """Remove specialist report files after synthesis."""
234:    for spec in SPECIALISTS.values():
235:        path = BASE_DIR / spec["output_file"]
236:        if path.exists():
237:            path.unlink()
238:    # Also clean up log files
239:    for name in SPECIALISTS:
240:        log_path = DATA_DIR / f"_specialist_{name}.log"
241:        if log_path.exists():
242:            log_path.unlink()
243:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Telegram notification utilities — send messages, alerts, escape markdown."""
2:
3:import logging
4:import os
5:import re
6:
7:from portfolio.file_utils import load_json
8:from portfolio.http_retry import fetch_with_retry
9:from portfolio.message_store import send_or_store
10:from portfolio.tickers import SYMBOLS
11:
12:logger = logging.getLogger("portfolio.telegram")
13:
14:_MD_V1_SPECIAL = re.compile(r'([_*`\[\]])')
15:
16:from pathlib import Path
17:
18:BOLD_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio_state_bold.json"
19:_COOLDOWN_PREFIXES = ("cooldown", "crypto check-in", "startup")
20:
21:
22:def escape_markdown_v1(text):
23:    """Escape special Markdown v1 characters in dynamic content to prevent parse failures.
24:
25:    Use this on user-facing dynamic strings (ticker names, error messages, reason text)
26:    that are inserted into Markdown-formatted Telegram messages. Do NOT apply to the
27:    entire message — it would break intentional formatting like *bold* and _italic_.
28:    """
29:    return _MD_V1_SPECIAL.sub(r'\\\1', str(text))
30:
31:
32:_TELEGRAM_MAX_LENGTH = 4096  # Telegram API rejects messages exceeding this
33:
34:
35:def send_telegram(msg, config):
36:    if os.environ.get("NO_TELEGRAM"):
37:        logger.info("[NO_TELEGRAM] Skipping send")
38:        return True
39:    # Global mute gate
40:    if config.get("telegram", {}).get("mute_all", False):
41:        logger.info("[mute_all] Skipping send_telegram")
42:        return True
43:    # Layer 1 messages disabled — only Layer 2 (Claude Code) sends Telegram
44:    # via direct requests.post. To re-enable, set telegram.layer1_messages: true.
45:    if not config.get("telegram", {}).get("layer1_messages", False):
46:        logger.debug("[layer1_messages=false] Skipping Layer 1 send")
47:        return True
48:    # Truncate to Telegram's max message length to avoid silent 400 errors
49:    if len(msg) > _TELEGRAM_MAX_LENGTH:
50:        logger.warning("Telegram message truncated from %d to %d chars", len(msg), _TELEGRAM_MAX_LENGTH)
51:        msg = msg[:_TELEGRAM_MAX_LENGTH - 20] + "\n...(truncated)"
52:    token = config["telegram"]["token"]
53:    chat_id = config["telegram"]["chat_id"]
54:    r = fetch_with_retry(
55:        f"https://api.telegram.org/bot{token}/sendMessage",
56:        method="POST",
57:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
58:        timeout=30,
59:    )
60:    if r is None:
61:        return False
62:    if r.ok:
63:        return True
64:    # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
65:    # still arrives (unformatted) rather than being silently lost.
66:    if r.status_code == 400:
67:        err_desc = ""
68:        try:
69:            err_desc = r.json().get("description", "")
70:        except Exception:
71:            logger.debug("Failed to parse Telegram error response", exc_info=True)
72:        if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
73:            logger.warning("Telegram Markdown parse failed (%s), resending without formatting", err_desc)
74:            r2 = fetch_with_retry(
75:                f"https://api.telegram.org/bot{token}/sendMessage",
76:                method="POST",
77:                json_body={"chat_id": chat_id, "text": msg},
78:                timeout=30,
79:            )
80:            return r2 is not None and r2.ok
81:    return False
82:
83:
84:def _maybe_send_alert(config, signals, prices_usd, fx_rate, state, reasons, tf_data):
85:    from portfolio.portfolio_mgr import portfolio_value
86:
87:    significant = [r for r in reasons if not r.startswith(_COOLDOWN_PREFIXES)]
88:    if not significant:
89:        return
90:    headline = escape_markdown_v1(significant[0])
91:    lines = [f"*ALERT: {headline}*", ""]
92:    # Actionable-only: show BUY/SELL tickers, compress HOLDs
93:    hold_count = 0
94:    for ticker in SYMBOLS:
95:        sig = signals.get(ticker)
96:        if not sig:
97:            continue
98:        action = sig["action"]
99:        if action == "HOLD":
100:            hold_count += 1
101:            continue
102:        price = prices_usd.get(ticker, 0)
103:        extra = sig.get("extra", {})
104:        b = extra.get("_buy_count", 0)
105:        s = extra.get("_sell_count", 0)
106:        total = extra.get("_total_applicable", 0)
107:        h = max(0, total - b - s)
108:        if price >= 1000:
109:            p_str = f"${price:,.0f}"
110:        else:
111:            p_str = f"${price:,.2f}"
112:        lines.append(f"`{ticker:<7} {p_str:>9}  {action:<4} {b}B/{s}S/{h}H`")
113:    if hold_count > 0:
114:        lines.append(f"_+ {hold_count} HOLD_")
115:    fg_val = ""
116:    for _ticker, sig in signals.items():
117:        extra = sig.get("extra", {})
118:        if "fear_greed" in extra:
119:            fg_class = escape_markdown_v1(extra.get("fear_greed_class", ""))
120:            fg_val = f"{extra['fear_greed']} ({fg_class})"
121:            break
122:    patient_total = portfolio_value(state, prices_usd, fx_rate)
123:    patient_pnl = (
124:        (patient_total - state["initial_value_sek"]) / state["initial_value_sek"]
125:    ) * 100
126:    lines.append("")
127:    if fg_val:
128:        lines.append(f"_F&G: {fg_val}_")
129:    lines.append(f"_Patient: {patient_total:,.0f} SEK ({patient_pnl:+.1f}%)_")
130:    bold = load_json(BOLD_STATE_FILE)
131:    if bold is not None:
132:        bold_total = portfolio_value(bold, prices_usd, fx_rate)
133:        bold_pnl = (
134:            (bold_total - bold["initial_value_sek"]) / bold["initial_value_sek"]
135:        ) * 100
136:        lines.append(f"_Bold: {bold_total:,.0f} SEK ({bold_pnl:+.1f}%)_")
137:    msg = "\n".join(lines)
138:    try:
139:        send_or_store(msg, config, category="analysis")
140:        logger.info("Alert sent: %s", headline)
141:    except Exception as e:
142:        logger.warning("alert send failed: %s", e)
143:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Telegram Poller — Background thread for ISKBETS + system commands.
2:
3:Polls getUpdates every 5 seconds. Parses bought/sold/cancel/status commands
4:and delegates to iskbets.handle_command(). Also handles /mode command for
5:switching notification format (signals vs probability).
6:"""
7:
8:import logging
9:import threading
10:import time
11:from datetime import UTC, datetime
12:from pathlib import Path
13:
14:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
15:from portfolio.http_retry import fetch_with_retry
16:
17:logger = logging.getLogger("portfolio.telegram_poller")
18:
19:INBOUND_LOG = Path(__file__).resolve().parent.parent / "data" / "telegram_inbound.jsonl"
20:# 2026-04-28: persisted offset across loop restarts. Without this, every
21:# `schtasks /run PF-DataLoop` resets self.offset to 0, re-fetches every
22:# pending getUpdates, and then the stale filter (msg_date < startup-60s)
23:# silently drops anything the user sent during the restart window. With
24:# the file present, init reloads the last-acknowledged update_id, and
25:# _handle_update bypasses the stale filter for post-restart pending
26:# updates (those the user expects to execute, e.g. a ``bought MSTR …``
27:# confirmation sent while the loop was bouncing) UP TO a bounded age:
28:# see RESTART_BYPASS_MAX_AGE_S below.
29:POLLER_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "telegram_poller_state.json"
30:
31:# Codex P1 round-4 (2026-04-28): cap the post-restart bypass to 1 hour.
32:# A bot that was down for days could otherwise execute every queued
33:# 'bought MSTR …' confirmation on next start, even though the user has
34:# since traded manually. 1 h is generous enough to cover any realistic
35:# restart window (schtasks rerun + loop boot < 5 min in practice) while
36:# still rejecting commands that are old enough that the user almost
37:# certainly resolved them out-of-band. Beyond this window the original
38:# 60 s stale filter applies.
39:RESTART_BYPASS_MAX_AGE_S = 60 * 60
40:
41:
42:class TelegramPoller:
43:    def __init__(self, config, on_command):
44:        """
45:        config: full app config dict (with telegram.token, telegram.chat_id)
46:        on_command: callback(cmd, args, config) -> response_text or None
47:        """
48:        self.token = config["telegram"]["token"]
49:        self.chat_id = str(config["telegram"]["chat_id"])
50:        self.config = config
51:        self.on_command = on_command
52:        # Restore offset from disk so updates acknowledged in a previous
53:        # process don't get re-fetched (and re-stale-filtered) on restart.
54:        # ``_initial_offset`` is the value we loaded from disk — the stale
55:        # filter uses it to recognize "this update arrived during downtime,
56:        # process don't drop". A fresh install with no state file yields 0,
57:        # which preserves the original cold-start behavior.
58:        self._initial_offset = self._load_persisted_offset()
59:        self.offset = self._initial_offset
60:        self._has_persisted_offset = self._initial_offset > 0
61:        self._startup_time = time.time()
62:        self._thread = None
63:
64:    @staticmethod
65:    def _load_persisted_offset() -> int:
66:        """Read offset from POLLER_STATE_FILE. Returns 0 on any failure
67:        (missing file, malformed JSON, non-int value, or negative
68:        integer) — fail-soft so a corrupted state file never prevents
69:        the loop from polling. Negative values are explicitly rejected
70:        because Telegram's getUpdates treats negative offsets as a
71:        backward count from the latest update, not as cold-start
72:        behavior (Codex P3 round-3 2026-04-28)."""
73:        try:
74:            state = load_json(POLLER_STATE_FILE, default=None)
75:        except Exception as e:
76:            logger.warning("poller offset load failed: %s", e)
77:            return 0
78:        if not isinstance(state, dict):
79:            return 0
80:        try:
81:            offset = int(state.get("offset", 0) or 0)
82:        except (TypeError, ValueError):
83:            return 0
84:        if offset < 0:
85:            logger.warning(
86:                "poller offset state had negative value %d; clamping to 0",
87:                offset,
88:            )
89:            return 0
90:        return offset
91:
92:    def _save_offset(self) -> None:
93:        """Persist current offset atomically. Best-effort: a write failure
94:        means the next restart re-fetches updates we already acked, but
95:        that's recoverable (Telegram dedups via the same update_id) so we
96:        don't crash the poll loop on disk errors."""
97:        try:
98:            atomic_write_json(
99:                POLLER_STATE_FILE,
100:                {
101:                    "offset": int(self.offset),
102:                    "updated_ts": datetime.now(UTC).isoformat(),
103:                },
104:            )
105:        except Exception as e:
106:            logger.warning("poller offset persist failed: %s", e)
107:
108:    def start(self):
109:        """Start the poller in a daemon thread."""
110:        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
111:        self._thread.start()
112:
113:    def _poll_loop(self):
114:        while True:
115:            try:
116:                updates = self._get_updates()
117:                for update in updates:
118:                    self._handle_update(update)
119:            except Exception as e:
120:                logger.warning("Poller error: %s", e)
121:            time.sleep(5)
122:
123:    def _get_updates(self):
124:        """Fetch new updates from Telegram."""
125:        params = {"timeout": 3, "allowed_updates": ["message"]}
126:        if self.offset:
127:            params["offset"] = self.offset
128:
129:        r = fetch_with_retry(
130:            f"https://api.telegram.org/bot{self.token}/getUpdates",
131:            params=params,
132:            timeout=10,
133:        )
134:        if r is None or not r.ok:
135:            return []
136:
137:        data = r.json()
138:        if not data.get("ok"):
139:            return []
140:
141:        return data.get("result", [])
142:
143:    # Drop reasons that represent a *settled* outcome — the message was
144:    # examined and intentionally not acted on (stale, empty, unrecognized,
145:    # or no message body / wrong chat). Re-fetching these on a restart
146:    # would just settle them the same way, so we ack the offset.
147:    # Excluded: ``raised:*`` outcomes — those represent a transient
148:    # dispatch failure where the user's command is genuinely at risk of
149:    # being lost if we ack the offset before it succeeds (Codex P1
150:    # round-7 2026-04-28).
151:    _SETTLED_DROP_REASONS = frozenset({
152:        "stale_at_startup",
153:        "empty_text",
154:        "unrecognized",
155:    })
156:
157:    def _handle_update(self, update):
158:        """Process a single update."""
159:        update_id = update.get("update_id", 0)
160:        prev_offset = self.offset
161:        self.offset = max(self.offset, update_id + 1)
162:        # In-memory offset advances unconditionally so a single poison
163:        # update doesn't loop the in-process poll, but persistence is
164:        # delayed until we know the message is settled — successful
165:        # dispatch, intentional drop, or non-message frame. If the
166:        # handler raises, we leave the persisted offset where it was so
167:        # restart re-fetches and retries (Codex P1 round-7 2026-04-28).
168:        offset_settled = False
169:
170:        msg = update.get("message")
171:        if not msg:
172:            offset_settled = True
173:
174:        # Only process messages from our chat_id. Drop others without logging —
175:        # no point persisting spam from strangers who can't affect state.
176:        # We DO still ack the offset on chat-mismatch so the bot's
177:        # getUpdates queue doesn't accumulate stranger spam over time.
178:        if msg is not None:
179:            chat = msg.get("chat", {})
180:            if str(chat.get("id")) != self.chat_id:
181:                offset_settled = True
182:                msg = None  # short-circuit out of the rest of the body
183:
184:        if msg is None:
185:            if offset_settled and self.offset > prev_offset:
186:                self._save_offset()
187:            return
188:
189:        # Accumulate log outcome; single append in finally so we log every
190:        # inbound message exactly once, even if parse/dispatch raises.
191:        outcome = {"cmd": None, "processed": False, "drop_reason": None}
192:        try:
193:            # Stale filter: ignore messages older than 60s at startup so we
194:            # don't re-execute commands after a loop restart. Still log them
195:            # — useful for reconstructing what the user sent during downtime.
196:            #
197:            # Bypass when (a) we have a persisted offset and (b) this
198:            # update_id is past it. Those are post-restart pending updates
199:            # — by definition arrived during downtime, the user expects
200:            # them to execute, and the persisted offset proves we're not
201:            # accidentally re-running a stale getUpdates queue from a long
202:            # outage. Cold-start (no persisted offset) keeps the original
203:            # protection because we can't distinguish "user sent during
204:            # restart" from "Telegram re-delivering 2-week-old updates"
205:            # without that prior.
206:            msg_date = msg.get("date", 0)
207:            # update_id can EQUAL self._initial_offset legitimately: the
208:            # persisted value uses next-offset semantics (last_acked + 1)
209:            # so the first genuinely-new update after restart has
210:            # update_id == self._initial_offset, not strictly greater.
211:            # `>=` covers the single-message-during-restart case that was
212:            # the whole reason for adding persistence (Codex P1
213:            # 2026-04-28).
214:            #
215:            # Codex P1 round-4 (2026-04-28): bound the bypass to
216:            # RESTART_BYPASS_MAX_AGE_S so a multi-day outage doesn't
217:            # execute every queued command on next start.
218:            is_post_restart_pending = (
219:                self._has_persisted_offset
220:                and update_id >= self._initial_offset
221:                and msg_date >= self._startup_time - RESTART_BYPASS_MAX_AGE_S
222:            )
223:            if msg_date < self._startup_time - 60 and not is_post_restart_pending:
224:                outcome["drop_reason"] = "stale_at_startup"
225:                return
226:
227:            text = (msg.get("text") or "").strip()
228:            if not text:
229:                outcome["drop_reason"] = "empty_text"
230:                return
231:
232:            cmd, args = self._parse_command(text)
233:            outcome["cmd"] = cmd
234:            if cmd is None:
235:                outcome["drop_reason"] = "unrecognized"
236:                return
237:
238:            # Dispatch can raise (Avanza session, volume math, network) — we
239:            # want processed=True to mean "dispatch completed", not "dispatch
240:            # was attempted". On raise, tag drop_reason with the exception
241:            # type so the audit log reflects the actual outcome, then re-raise
242:            # to preserve the old error-propagation behavior.
243:            try:
244:                if cmd == "mode":
245:                    response = self._handle_mode_command(args)
246:                else:
247:                    response = self.on_command(cmd, args, self.config)
248:                if response:
249:                    self._send_reply(response)
250:                outcome["processed"] = True
251:            except Exception as exc:
252:                outcome["drop_reason"] = f"raised:{type(exc).__name__}"
253:                raise
254:        finally:
255:            self._log_inbound(update, msg, **outcome)
256:            # Persist offset only when the message has *settled* —
257:            # successful dispatch or an intentional drop. A raised
258:            # dispatch leaves persistence un-claimed so a restart can
259:            # retry; otherwise a transient handler crash silently
260:            # consumes the user's command (Codex P1 round-7 2026-04-28).
261:            should_persist = outcome["processed"] or (
262:                outcome["drop_reason"] in self._SETTLED_DROP_REASONS
263:            )
264:            if should_persist and self.offset > prev_offset:
265:                self._save_offset()
266:
267:    def _log_inbound(self, update, msg, cmd, processed, drop_reason):
268:        """Persist one inbound message to data/telegram_inbound.jsonl.
269:
270:        Rotation registered in portfolio/log_rotation.py (90d / 20 MB).
271:        """
272:        try:
273:            sender = msg.get("from") or {}
274:            entry = {
275:                "ts": datetime.now(UTC).isoformat(),
276:                "direction": "inbound",
277:                "update_id": update.get("update_id"),
278:                "message_id": msg.get("message_id"),
279:                "msg_date": msg.get("date"),
280:                "from": {
281:                    "id": sender.get("id"),
282:                    "username": sender.get("username"),
283:                },
284:                "text": msg.get("text") or "",
285:                "cmd": cmd,
286:                "processed": processed,
287:                "drop_reason": drop_reason,
288:            }
289:            atomic_append_jsonl(INBOUND_LOG, entry)
290:        except Exception as e:
291:            logger.warning("Inbound log write failed: %s", e)
292:
293:    def _parse_command(self, text):
294:        """Parse ISKBETS and system commands from message text.
295:
296:        Returns (cmd, args) or (None, None) for non-commands.
297:        Recognized: bought, sold, cancel, status, /mode
298:        """
299:        parts = text.split(None, 1)
300:        first_word = parts[0].lower() if parts else ""
301:        rest = parts[1] if len(parts) > 1 else ""
302:
303:        if first_word in ("bought", "sold", "cancel", "status"):
304:            return first_word, rest
305:
306:        # /mode command — switch notification format
307:        if first_word in ("/mode", "mode"):
308:            return "mode", rest.strip().lower()
309:
310:        return None, None
311:
312:    def _handle_mode_command(self, mode_arg):
313:        """Handle /mode command — switch notification format.
314:
315:        Args:
316:            mode_arg: "signals" or "probability" (or empty to query current mode)
317:
318:        Returns:
319:            Reply text for the user.
320:        """
321:        from pathlib import Path
322:
323:        # Adversarial review 04-29 IN-P1-3 (2026-05-02): use the
324:        # file_utils helpers (load_json + atomic_write_json) rather than
325:        # raw open()/json.load(). Two reasons:
326:        #   1. CLAUDE.md rule 4: "Atomic I/O only".
327:        #   2. config.json is a symlink to an external file; raw open() can
328:        #      race against an external atomic_write_json rename mid-read on
329:        #      Windows (we've seen partial-byte reads in agent.log). load_json
330:        #      handles the same edge cases (missing/corrupt → default) as
331:        #      every other consumer in the codebase.
332:        config_path = Path(__file__).resolve().parent.parent / "config.json"
333:
334:        if not mode_arg:
335:            # Query current mode
336:            current = self.config.get("notification", {}).get("mode", "signals")
337:            return f"Current notification mode: *{current}*"
338:
339:        if mode_arg not in ("signals", "probability"):
340:            return "Usage: `/mode signals` or `/mode probability`"
341:
342:        # Update config.json — load_json returns {} for missing/corrupt files
343:        # without raising, so the BUG-210 size guard below catches both the
344:        # genuine-corrupt case and the transient-unreadable case.
345:        cfg = load_json(config_path, default={})
346:
347:        # BUG-210: Guard against writing suspiciously small config.
348:        # If config.json was momentarily unreadable (symlink, AV lock, fs
349:        # glitch), cfg={} and the write below would destroy all API keys.
350:        if len(cfg) < 5:
351:            logger.error(
352:                "Refusing to write config — loaded config has only %d keys "
353:                "(expected 5+, possible transient read failure)", len(cfg)
354:            )
355:            return "Error: config file appears corrupt or unreadable. Try again."
356:
357:        if "notification" not in cfg:
358:            cfg["notification"] = {}
359:        cfg["notification"]["mode"] = mode_arg
360:
361:        atomic_write_json(config_path, cfg)
362:
363:        # Update in-memory config
364:        if "notification" not in self.config:
365:            self.config["notification"] = {}
366:        self.config["notification"]["mode"] = mode_arg
367:
368:        logger.info("Notification mode changed to: %s", mode_arg)
369:        return f"Notification mode set to *{mode_arg}*"
370:
371:    def _send_reply(self, text):
372:        """Send a reply to the user."""
373:        try:
374:            r = fetch_with_retry(
375:                f"https://api.telegram.org/bot{self.token}/sendMessage",
376:                method="POST",
377:                json_body={
378:                    "chat_id": self.chat_id,
379:                    "text": text,
380:                    "parse_mode": "Markdown",
381:                },
382:                timeout=30,
383:            )
384:            if r is not None and not r.ok:
385:                logger.warning("Poller reply error: %s %s", r.status_code, r.text[:200])
386:        except Exception as e:
387:            logger.warning("Poller reply failed: %s", e)
388:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Health monitoring for the finance-analyzer Layer 1 loop."""
2:
3:import logging
4:import threading
5:import time
6:from datetime import UTC, datetime
7:from pathlib import Path
8:
9:from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail
10:
11:logger = logging.getLogger(__name__)
12:
13:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
14:HEALTH_FILE = DATA_DIR / "health_state.json"
15:
16:# C10/H17: Protect all read-modify-write sequences in health.py.
17:_health_lock = threading.Lock()
18:
19:
20:def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
21:                  last_trigger_reason: str = None, error: str = None):
22:    """Called at end of each Layer 1 cycle to update health state."""
23:    with _health_lock:
24:        state = load_health()
25:        state["last_heartbeat"] = datetime.now(UTC).isoformat()
26:        state["cycle_count"] = cycle_count
27:        state["signals_ok"] = signals_ok
28:        state["signals_failed"] = signals_failed
29:        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
30:        if last_trigger_reason:
31:            state["last_trigger_reason"] = last_trigger_reason
32:            state["last_trigger_time"] = datetime.now(UTC).isoformat()
33:            # Cache the invocation timestamp so check_agent_silence() can avoid
34:            # re-parsing invocations.jsonl on every call.
35:            state["last_invocation_ts"] = state["last_trigger_time"]
36:        if error:
37:            state["errors"] = state.get("errors", [])[-19:] + [
38:                {"ts": datetime.now(UTC).isoformat(), "error": error}
39:            ]
40:            state["error_count"] = state.get("error_count", 0) + 1
41:        atomic_write_json(HEALTH_FILE, state)
42:
43:
44:def load_health() -> dict:
45:    """Load current health state. Returns defaults if missing or corrupt."""
46:    state = load_json(HEALTH_FILE)
47:    if state is not None:
48:        return state
49:    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}
50:
51:
52:def reset_session_start():
53:    """Reset start_time to current time — call at loop startup.
54:
55:    Prevents uptime_seconds from inheriting a stale start_time
56:    from a previous session's health_state.json.
57:    """
58:    with _health_lock:
59:        state = load_health()
60:        state["start_time"] = time.time()
61:        atomic_write_json(HEALTH_FILE, state)
62:
63:
64:def heartbeat() -> None:
65:    """Touch only the last_heartbeat timestamp.
66:
67:    Called as a one-shot or periodically from a keepalive thread while
68:    long-blocking work is in flight. Layer 2 invocation can block up to
69:    600s (T2) or 900s (T3), but update_health() only runs at end-of-cycle
70:    (AFTER Layer 2 returns). Without periodic touches the dashboard
71:    /api/health endpoint flips fresh→stale every triggering cycle, which
72:    is misleading: the loop is alive, just waiting on the subprocess.
73:
74:    Other state (cycle_count, signals_ok/failed, errors) is left untouched
75:    — those reflect the previously-completed cycle, still the most recent
76:    ground truth. update_health() at end-of-cycle overwrites them with
77:    this cycle's results.
78:
79:    Failure-tolerant: callers wrap in try/except since this is a "nice to
80:    have" hint and must never crash the loop. The atomic write means a
81:    partial run leaves the prior file intact.
82:    """
83:    with _health_lock:
84:        state = load_health()
85:        state["last_heartbeat"] = datetime.now(UTC).isoformat()
86:        atomic_write_json(HEALTH_FILE, state)
87:
88:
89:# Keepalive default interval. The dashboard's stale gate fires at 300s
90:# (check_staleness max_age_seconds=300), so 60s gives 5x headroom while
91:# leaving plenty of margin for missed ticks (e.g. GIL contention, slow
92:# atomic_write_json on a heavily-loaded disk).
93:_HEARTBEAT_KEEPALIVE_INTERVAL_S = 60.0
94:
95:
96:class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
97:    """Context manager that ticks heartbeat() every interval seconds.
98:
99:    Wraps long-blocking work (Layer 2 T2/T3 subprocess, autonomous decision
100:    paths, anything that can block longer than the 300s stale threshold)
101:    so /api/health stays fresh for the duration. Background daemon thread
102:    is auto-stopped on context exit; a 2s join timeout prevents shutdown
103:    deadlocks (the thread only sleeps + writes, both bounded).
104:
105:    Usage:
106:        with heartbeat_keepalive():
107:            result = invoke_agent(reasons_list, tier=tier)
108:
109:    The first beat is synchronous (so a fast-returning subprocess gets at
110:    least one heartbeat even if it finishes before the first interval).
111:    Subsequent beats run on the daemon thread until __exit__.
112:
113:    Failure-tolerant by design: tick exceptions are swallowed at WARNING
114:    level — a Disk-full or permission-denied during keepalive must never
115:    abort an in-flight Layer 2 trade decision.
116:    """
117:
118:    def __init__(self, interval: float = _HEARTBEAT_KEEPALIVE_INTERVAL_S) -> None:
119:        self._interval = interval
120:        self._stop = threading.Event()
121:        self._thread: threading.Thread | None = None
122:
123:    def __enter__(self) -> "heartbeat_keepalive":
124:        # Synchronous first beat — covers the case where the wrapped call
125:        # returns before the keepalive thread's first tick.
126:        try:
127:            heartbeat()
128:        except Exception:
129:            logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)
130:
131:        self._thread = threading.Thread(
132:            target=self._run, daemon=True, name="heartbeat-keepalive",
133:        )
134:        self._thread.start()
135:        return self
136:
137:    def __exit__(self, *_exc: object) -> None:
138:        self._stop.set()
139:        if self._thread is not None:
140:            self._thread.join(timeout=2.0)
141:
142:    def _run(self) -> None:
143:        # Event.wait returns True when set (stop signaled), False on timeout.
144:        # So we tick on each timeout and exit on the first True.
145:        while not self._stop.wait(self._interval):
146:            try:
147:                heartbeat()
148:            except Exception:
149:                logger.warning("heartbeat_keepalive tick failed", exc_info=True)
150:
151:
152:def check_staleness(max_age_seconds: int = 300) -> tuple:
153:    """Check if the loop heartbeat is stale.
154:    Returns (is_stale: bool, age_seconds: float, state: dict)
155:    """
156:    state = load_health()
157:    hb = state.get("last_heartbeat")
158:    if not hb:
159:        return True, float("inf"), state
160:    try:
161:        last = datetime.fromisoformat(hb)
162:    except (ValueError, TypeError):
163:        logger.warning("check_staleness: corrupt last_heartbeat=%r", hb)
164:        return True, float("inf"), state
165:    age = (datetime.now(UTC) - last).total_seconds()
166:    return age > max_age_seconds, age, state
167:
168:
169:def check_agent_silence(max_market_seconds: int = 7200,
170:                        max_offhours_seconds: int = 14400) -> dict:
171:    """Detect silent Layer 2 agent (no invocation for too long).
172:
173:    Args:
174:        max_market_seconds: Max allowed silence during market hours (default 2h).
175:        max_offhours_seconds: Max allowed silence outside market hours (default 4h).
176:
177:    Returns:
178:        dict with keys: silent (bool), age_seconds (float), threshold (int), market_open (bool)
179:    """
180:    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
181:    last_ts = None
182:    state = load_health()
183:    last_ts = state.get("last_invocation_ts")
184:
185:    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
186:    if not last_ts:
187:        invocations_file = DATA_DIR / "invocations.jsonl"
188:        last_ts = last_jsonl_entry(invocations_file, field="ts")
189:        if last_ts is None:
190:            return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
191:        # Write back to health state so subsequent calls hit the cache
192:        # instead of re-parsing the JSONL file every time.
193:        with _health_lock:
194:            wb_state = load_health()
195:            wb_state["last_invocation_ts"] = last_ts
196:            atomic_write_json(HEALTH_FILE, wb_state)
197:
198:    if not last_ts:
199:        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
200:
201:    try:
202:        last = datetime.fromisoformat(last_ts)
203:    except (ValueError, TypeError):
204:        logger.warning("Corrupt last_invocation_ts in health state: %r", last_ts)
205:        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
206:    now = datetime.now(UTC)
207:    age = (now - last).total_seconds()
208:
209:    # DST-aware market hours check
210:    from portfolio.market_timing import get_market_state
211:    market_state, _, _ = get_market_state()
212:    market_open = (market_state == "open")
213:    threshold = max_market_seconds if market_open else max_offhours_seconds
214:
215:    return {
216:        "silent": age > threshold,
217:        "age_seconds": round(age, 1),
218:        "threshold": threshold,
219:        "market_open": market_open,
220:    }
221:
222:
223:def update_module_failures(failures: list):
224:    """Record which reporting modules failed in the current cycle.
225:
226:    Called by reporting.py after generating the agent summary.
227:    Persists module names + timestamp in health_state.json so the dashboard
228:    and monitoring scripts can see per-module status without parsing logs.
229:
230:    Recovery semantics (2026-05-03): when called with an empty list AND a
231:    prior failure record exists, this clears the record so the dashboard
232:    reflects the *current* state, not a stale "last known failure" — the
233:    bug surfaced after a 2026-05-03 cycle-0 transient that left dashboard
234:    /api/health falsely flagging monte_carlo / price_targets / equity_curve
235:    as failed for hours after the modules had recovered.
236:
237:    The clean-no-prior-failure case still skips the write to avoid
238:    spamming the disk every 60s for the common case.
239:    """
240:    with _health_lock:
241:        state = load_health()
242:        if failures:
243:            state["last_module_failures"] = {
244:                "ts": datetime.now(UTC).isoformat(),
245:                "modules": list(failures),
246:            }
247:            atomic_write_json(HEALTH_FILE, state)
248:        elif state.get("last_module_failures") is not None:
249:            # Recovery: prior failure cleared on a clean cycle. One-shot write.
250:            state.pop("last_module_failures", None)
251:            atomic_write_json(HEALTH_FILE, state)
252:
253:
254:def update_signal_health(signal_name: str, success: bool):
255:    """Record a single signal execution result.
256:
257:    For batch updates (multiple signals per cycle), prefer
258:    update_signal_health_batch() to avoid repeated disk writes.
259:    """
260:    update_signal_health_batch({signal_name: success})
261:
262:
263:def update_signal_health_batch(results: dict):
264:    """Record multiple signal execution results in a single disk write.
265:
266:    Args:
267:        results: dict of {signal_name: bool} where True=success, False=failure.
268:    """
269:    if not results:
270:        return
271:    with _health_lock:
272:        state = load_health()
273:        sh = state.setdefault("signal_health", {})
274:        now = datetime.now(UTC).isoformat()
275:
276:        for signal_name, success in results.items():
277:            entry = sh.setdefault(signal_name, {
278:                "total_calls": 0,
279:                "total_failures": 0,
280:                "last_success": None,
281:                "last_failure": None,
282:                "recent_results": [],
283:            })
284:            entry["total_calls"] = entry.get("total_calls", 0) + 1
285:            if success:
286:                entry["last_success"] = now
287:            else:
288:                entry["total_failures"] = entry.get("total_failures", 0) + 1
289:                entry["last_failure"] = now
290:
291:            # Rolling window: keep last 50 results for recent success rate
292:            recent = entry.get("recent_results", [])
293:            recent.append(success)
294:            if len(recent) > 50:
295:                recent = recent[-50:]
296:            entry["recent_results"] = recent
297:
298:        atomic_write_json(HEALTH_FILE, state)
299:
300:
301:def get_signal_health(signal_name: str = None) -> dict:
302:    """Get signal health data.
303:
304:    If signal_name is given, returns that signal's health dict.
305:    Otherwise returns the full signal_health dict for all signals.
306:    """
307:    state = load_health()
308:    sh = state.get("signal_health", {})
309:    if signal_name:
310:        return sh.get(signal_name, {})
311:    return sh
312:
313:
314:def get_signal_health_summary() -> dict:
315:    """Compact signal health summary for reporting.
316:
317:    Returns dict of signal_name -> {success_rate, total_calls, total_failures,
318:    last_failure} for signals with at least 1 call.
319:    """
320:    sh = get_signal_health()
321:    summary = {}
322:    for sig_name, data in sh.items():
323:        total = data.get("total_calls", 0)
324:        if total == 0:
325:            continue
326:        failures = data.get("total_failures", 0)
327:        recent = data.get("recent_results", [])
328:        recent_rate = (sum(1 for r in recent if r) / len(recent) * 100) if recent else 0
329:        summary[sig_name] = {
330:            "success_rate_pct": round(recent_rate, 1),
331:            "total_calls": total,
332:            "total_failures": failures,
333:            "last_failure": data.get("last_failure"),
334:        }
335:    return summary
336:
337:
338:def get_health_summary() -> dict:
339:    """Return a summary dict suitable for API/dashboard consumption."""
340:    state = load_health()
341:    is_stale, age, _ = check_staleness()
342:    agent_silence = check_agent_silence()
343:    summary = {
344:        "status": "stale" if is_stale else "healthy",
345:        "heartbeat_age_seconds": round(age, 1),
346:        "cycle_count": state.get("cycle_count", 0),
347:        "error_count": state.get("error_count", 0),
348:        "last_trigger": state.get("last_trigger_reason"),
349:        "last_trigger_time": state.get("last_trigger_time"),
350:        "recent_errors": state.get("errors", [])[-5:],
351:        "signals_ok": state.get("signals_ok", 0),
352:        "signals_failed": state.get("signals_failed", 0),
353:        "agent_silent": agent_silence["silent"],
354:        "agent_silence_seconds": agent_silence["age_seconds"],
355:        "module_failures": state.get("last_module_failures"),
356:        "signal_health": get_signal_health_summary(),
357:    }
358:    # Include circuit breaker status if data_collector has been imported
359:    try:
360:        from portfolio.data_collector import alpaca_cb, binance_fapi_cb, binance_spot_cb
361:        summary["circuit_breakers"] = {
362:            "binance_spot": binance_spot_cb.get_status(),
363:            "binance_fapi": binance_fapi_cb.get_status(),
364:            "alpaca": alpaca_cb.get_status(),
365:        }
366:    except Exception as e:
367:        logger.warning("Circuit breaker status unavailable: %s", e)
368:    return summary
369:
370:
371:def check_outcome_staleness(max_age_hours: int = 36) -> dict:
372:    """Check if outcome backfill is stale (no recent outcomes in signal_log).
373:
374:    Returns dict with: stale (bool), newest_outcome_age_hours (float),
375:    entries_without_outcomes (int).
376:    """
377:    signal_log = DATA_DIR / "signal_log.jsonl"
378:
379:    now = time.time()
380:    newest_outcome_ts = 0
381:    missing_count = 0
382:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
383:    entries = load_jsonl_tail(signal_log, max_entries=50)
384:    if not entries:
385:        return {"stale": True, "newest_outcome_age_hours": float("inf"),
386:                "entries_without_outcomes": 0}
387:
388:    try:
389:        for entry in entries:
390:            outcomes = entry.get("outcomes", {})
391:            has_any = any(
392:                outcomes.get(t, {}).get("1d") is not None
393:                for t in outcomes
394:            )
395:            if has_any:
396:                # Parse outcome timestamps to find newest
397:                for t_outcomes in outcomes.values():
398:                    for h_data in t_outcomes.values():
399:                        if isinstance(h_data, dict) and h_data.get("ts"):
400:                            try:
401:                                ots = datetime.fromisoformat(h_data["ts"]).timestamp()
402:                                newest_outcome_ts = max(newest_outcome_ts, ots)
403:                            except (ValueError, TypeError):
404:                                pass
405:            else:
406:                missing_count += 1
407:    except Exception as exc:
408:        logger.warning("check_outcome_staleness error: %s", exc)
409:        return {"stale": True, "newest_outcome_age_hours": float("inf"),
410:                "entries_without_outcomes": 0}
411:
412:    if newest_outcome_ts == 0:
413:        age_hours = float("inf")
414:    else:
415:        age_hours = (now - newest_outcome_ts) / 3600
416:
417:    return {
418:        "stale": age_hours > max_age_hours,
419:        "newest_outcome_age_hours": round(age_hours, 1),
420:        "entries_without_outcomes": missing_count,
421:    }
422:
423:
424:def check_dead_signals(recent_entries: int = 20) -> list[str]:
425:    """Detect signals that voted HOLD on every ticker in the last N entries.
426:
427:    Returns list of signal names that are effectively dead (100% HOLD).
428:    """
429:    signal_log = DATA_DIR / "signal_log.jsonl"
430:
431:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
432:    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
433:    if not entries:
434:        return []
435:
436:    # Collect vote counts per signal
437:    from collections import defaultdict
438:    vote_counts = defaultdict(lambda: {"total": 0, "non_hold": 0})
439:
440:    for entry in entries:
441:        for _ticker, tdata in entry.get("tickers", {}).items():
442:            for sig_name, vote in tdata.get("signals", {}).items():
443:                vote_counts[sig_name]["total"] += 1
444:                if vote in ("BUY", "SELL"):
445:                    vote_counts[sig_name]["non_hold"] += 1
446:
447:    # Signals with >0 total votes but 0 non-HOLD votes are dead
448:    dead = []
449:    for sig_name, counts in vote_counts.items():
450:        if counts["total"] >= recent_entries and counts["non_hold"] == 0:
451:            dead.append(sig_name)
452:    return sorted(dead)
453:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/health.py:35:            state["last_invocation_ts"] = state["last_trigger_time"]
portfolio/health.py:183:    last_ts = state.get("last_invocation_ts")
portfolio/health.py:195:            wb_state["last_invocation_ts"] = last_ts
portfolio/health.py:204:        logger.warning("Corrupt last_invocation_ts in health state: %r", last_ts)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Pre-warm the NEXT LLM in the rotation so the upcoming loop cycle
2:doesn't pay the cold-swap cost.
3:
4:2026-05-11 Stage 3 Phase 1: minimal in-process prewarmer. Called from
5:``flush_llm_batch()`` right after the rotation counter increments. Issues
6:a 1-token dummy prompt to the next-slot model, which forces
7:``llama_server`` to swap to it synchronously inside the prewarm call —
8:the next real cycle (60 s away) then finds the model already loaded and
9:skips the swap entirely.
10:
11:Why this is safe:
12:
13:- All exceptions are swallowed and logged at WARNING. A broken
14:  prewarmer cannot regress the working ministral/qwen3/fingpt path
15:  because the call site in ``flush_llm_batch()`` wraps this in its own
16:  try/except as a belt-and-braces backstop.
17:- We rely solely on the public ``query_llama_server`` contract — we
18:  never touch ``llama_server`` internals or the file lock directly.
19:  The 1-token dummy prompt holds the same locks the real swap path
20:  holds; concurrent metals_loop swaps will simply queue behind us
21:  (~10-30 s typical).
22:- Chronos (``gpu_gate("chronos", timeout=30)``) is unaffected: this
23:  module never acquires ``gpu_gate`` itself. The win is that by the
24:  time Chronos runs in the *next* cycle, the LLM swap is already
25:  done — so Chronos's 30 s timeout no longer races a mid-flight swap.
26:
27:Rotation order (must match ``llm_batch._LLM_ROTATION``):
28:``ministral → qwen3 → fingpt``. The llama_server slot names are
29:``ministral3`` / ``qwen3`` / ``finance-llama-8b`` respectively — see
30:``ROTATION_SLOTS`` below for the mapping.
31:
32:State persistence: writes a single line to
33:``data/llm_rotation_state.jsonl`` after each prewarm attempt. On
34:restart, the most recent line is consulted so a process that just
35:prewarmed slot S at counter C doesn't redundantly prewarm S again
36:when restarted at the same counter (which would happen if a crash
37:left the rotation counter at the same value).
38:"""
39:
40:from __future__ import annotations
41:
42:import logging
43:import os
44:import time
45:from datetime import datetime, timezone
46:from pathlib import Path
47:
48:logger = logging.getLogger("portfolio.llm_prewarmer")
49:
50:# Rotation order pinned to llm_batch._LLM_ROTATION. Each entry is the
51:# abstract rotation name; ROTATION_SLOT_TO_SERVER maps it to the actual
52:# llama_server slot name (which is what _read_pid_model() returns and
53:# what query_llama_server expects as `name`).
54:ROTATION_SLOTS: tuple[str, ...] = ("ministral", "qwen3", "fingpt")
55:
56:ROTATION_SLOT_TO_SERVER: dict[str, str] = {
57:    "ministral": "ministral3",
58:    "qwen3": "qwen3",
59:    "fingpt": "finance-llama-8b",
60:}
61:
62:# State file: one JSONL line per prewarm attempt. Used to short-circuit
63:# duplicate prewarms across process restarts at the same counter.
64:BASE_DIR = Path(__file__).resolve().parent.parent
65:DATA_DIR = BASE_DIR / "data"
66:STATE_FILE = DATA_DIR / "llm_rotation_state.jsonl"
67:
68:
69:def _next_slot(current_counter: int) -> str:
70:    """Compute the next rotation slot name. Mirrors the rotation logic in
71:    ``llm_batch.is_llm_on_cycle``: after a flush at counter C completes,
72:    the counter is incremented so the *upcoming* cycle's slot index is
73:    ``(new_counter - 1) % 3``. The next cycle after that is
74:    ``new_counter % 3`` — and that's what we want to pre-warm.
75:
76:    But we're called with ``current_counter`` being the *just-incremented*
77:    counter (i.e. the counter that flush_llm_batch leaves in shared_state
78:    after the bump). The next cycle's slot is ``(current_counter - 1) % 3``;
79:    we want the slot AFTER that — i.e. ``current_counter % 3``.
80:
81:    Worked example with ROTATION = (ministral, qwen3, fingpt):
82:
83:        flush 1 finishes, counter=1, ran ministral.
84:          next cycle will run qwen3 (slot = (1-1) % 3 = 0 → ministral).
85:          Wait, that's wrong direction.
86:
87:    Re-read is_llm_on_cycle: at enqueue time counter==1 means slot 0
88:    (ministral) is the active LLM for that cycle. The counter is
89:    incremented AFTER the flush. So when we're called by flush_llm_batch
90:    POST-increment with counter=2, the next real cycle has counter=2
91:    and slot = (2-1) % 3 = 1 → qwen3. We pre-warm qwen3.
92:
93:    Therefore: prewarm slot index = ``(current_counter - 1) % 3``.
94:    """
95:    if not ROTATION_SLOTS:
96:        raise ValueError("ROTATION_SLOTS is empty")
97:    idx = (int(current_counter) - 1) % len(ROTATION_SLOTS)
98:    return ROTATION_SLOTS[idx]
99:
100:
101:def _read_last_state() -> dict | None:
102:    """Return the most recent entry from STATE_FILE, or None.
103:
104:    Fix C 2026-05-11 (codex review): true tail read instead of
105:    ``f.readlines()``. The state file is bounded by log_rotation now
106:    (Fix B), but even pre-rotation we don't want to grow O(n) with
107:    file size — every prewarm pays this. We seek to the end and read
108:    a small trailing block, then split on newlines and take the last
109:    complete JSON line. If the block doesn't contain a complete line
110:    (extremely long single record, unlikely), fall back to a full read
111:    once.
112:    """
113:    import json
114:
115:    TAIL_BLOCK = 8192
116:
117:    try:
118:        if not STATE_FILE.exists():
119:            return None
120:        size = STATE_FILE.stat().st_size
121:        if size == 0:
122:            return None
123:
124:        with open(STATE_FILE, "rb") as f:
125:            if size <= TAIL_BLOCK:
126:                f.seek(0)
127:                block = f.read()
128:            else:
129:                f.seek(size - TAIL_BLOCK)
130:                block = f.read()
131:
132:        text = block.decode("utf-8", errors="replace")
133:        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
134:        if not lines:
135:            return None
136:
137:        # If we tail-read and the first kept line came from a partial
138:        # block (i.e. we started mid-line because there was no newline
139:        # at the seek point), it may parse garbage. The LAST line is
140:        # always safe because the writer terminates with \n via
141:        # atomic_append_jsonl. We try it first.
142:        try:
143:            return json.loads(lines[-1])
144:        except json.JSONDecodeError:
145:            # Fallback: pathological case where the tail block does not
146:            # contain a complete JSON record. Read the whole file once.
147:            if size > TAIL_BLOCK:
148:                with open(STATE_FILE, encoding="utf-8") as f:
149:                    all_lines = [ln.strip() for ln in f if ln.strip()]
150:                if all_lines:
151:                    return json.loads(all_lines[-1])
152:            return None
153:    except Exception as e:
154:        logger.warning("llm_prewarmer state read failed: %s", e)
155:        return None
156:
157:
158:def _write_state(counter: int, prewarmed_slot: str, server_slot: str,
159:                 outcome: str, duration_s: float | None = None) -> None:
160:    """Append a single state record. Best-effort; swallows errors."""
161:    try:
162:        from portfolio.file_utils import atomic_append_jsonl
163:        DATA_DIR.mkdir(parents=True, exist_ok=True)
164:        entry = {
165:            "ts": datetime.now(timezone.utc).isoformat(),
166:            "counter": int(counter),
167:            "prewarmed_slot": prewarmed_slot,
168:            "server_slot": server_slot,
169:            "outcome": outcome,
170:        }
171:        if duration_s is not None:
172:            entry["duration_s"] = round(float(duration_s), 3)
173:        atomic_append_jsonl(STATE_FILE, entry)
174:    except Exception as e:
175:        logger.warning("llm_prewarmer state write failed: %s", e)
176:
177:
178:def _current_loaded_server_slot() -> str | None:
179:    """Return the llama_server slot name currently loaded according to the
180:    PID file, or None on any error/missing file.
181:
182:    Used both to short-circuit prewarm when the target is already loaded
183:    AND to validate the JSONL idempotency record against ground truth —
184:    see Fix A 2026-05-11 in ``prewarm_next_model``.
185:    """
186:    try:
187:        from portfolio.llama_server import _read_pid_model
188:        _, current_model = _read_pid_model()
189:        return current_model
190:    except Exception as e:
191:        logger.debug("llm_prewarmer load-check failed: %s", e)
192:        return None
193:
194:
195:def _is_slot_already_loaded(server_slot: str) -> bool:
196:    """Check llama_server's PID file to see if the target slot is already
197:    the active model. Returns False on any error (safe default — we'd
198:    rather prewarm an already-loaded model than skip a needed prewarm).
199:    """
200:    return _current_loaded_server_slot() == server_slot
201:
202:
203:def prewarm_next_model(current_counter: int) -> bool:
204:    """Issue a dummy 1-token prompt to the next-slot model.
205:
206:    Args:
207:        current_counter: the value of ``shared_state._full_llm_cycle_count``
208:            *after* ``flush_llm_batch`` has incremented it. Must be a
209:            positive int — counter==0 means the warmup hasn't happened
210:            yet, no useful prewarm possible.
211:
212:    Returns:
213:        True if a prewarm query was actually dispatched.
214:        False if the prewarm was a no-op (already loaded / duplicate /
215:        invalid counter / error). Never raises.
216:
217:    Contract: this function MUST NOT propagate exceptions to the caller.
218:    A broken prewarmer cannot be allowed to regress the working LLM
219:    rotation path.
220:    """
221:    try:
222:        # Test-suite safety: pytest sets PYTEST_CURRENT_TEST for every test
223:        # function. We auto-skip when that's present so the real prewarmer
224:        # never fires during test collection of the broader suite (which
225:        # would issue real swap requests against a running llama-server
226:        # if one happens to be up on the dev box). Tests that DO want to
227:        # exercise the prewarmer directly call prewarm_next_model from
228:        # their own fixtures with mocked dependencies — those tests bypass
229:        # this guard by setting PF_PREWARM_FORCE_RUN=1.
230:        if (
231:            os.environ.get("PYTEST_CURRENT_TEST")
232:            and os.environ.get("PF_PREWARM_FORCE_RUN") != "1"
233:        ):
234:            logger.debug("llm_prewarmer skip: pytest detected, no-op")
235:            return False
236:
237:        counter = int(current_counter)
238:        if counter <= 0:
239:            # Counter==0 means no flush has run yet, so the rotation hasn't
240:            # started; counter<0 is nonsense. Either way: no-op.
241:            logger.debug("llm_prewarmer skip: counter=%d not positive", counter)
242:            return False
243:
244:        next_slot = _next_slot(counter)
245:        server_slot = ROTATION_SLOT_TO_SERVER.get(next_slot)
246:        if server_slot is None:
247:            logger.warning(
248:                "llm_prewarmer skip: no server mapping for slot=%s", next_slot,
249:            )
250:            return False
251:
252:        # Fix A 2026-05-11 (codex review): reconcile JSONL idempotency
253:        # against the *currently loaded* slot. The rotation counter is
254:        # in-memory only and resets to 0 on process restart, which means a
255:        # fresh process will re-hit counter=1, counter=2, ... — and the
256:        # state JSONL from the previous process lifetime still has a
257:        # matching "warmed" line for those counters. Trusting it blindly
258:        # would let a restart skip a swap that is actually still needed.
259:        #
260:        # Rule: skip-by-state only if BOTH the JSONL record matches the
261:        # current (counter, slot) AND the llama_server PID file confirms
262:        # the expected slot is still loaded. Any mismatch → force prewarm.
263:        currently_loaded = _current_loaded_server_slot()
264:        last = _read_last_state()
265:        if (
266:            last is not None
267:            and int(last.get("counter", -1)) == counter
268:            and last.get("prewarmed_slot") == next_slot
269:            and last.get("outcome") == "warmed"
270:            and currently_loaded == server_slot
271:        ):
272:            logger.debug(
273:                "llm_prewarmer skip: counter=%d slot=%s already prewarmed "
274:                "and still loaded",
275:                counter, next_slot,
276:            )
277:            return False
278:
279:        # If the target model is already the active llama-server model
280:        # (e.g. metals_loop happened to swap to it for an unrelated
281:        # reason), there is nothing to do.
282:        if currently_loaded == server_slot:
283:            logger.info(
284:                "llm_prewarmer noop: slot=%s server=%s already loaded",
285:                next_slot, server_slot,
286:            )
287:            _write_state(counter, next_slot, server_slot, outcome="already_loaded")
288:            return False
289:
290:        # Fire the dummy query. n_predict=1 keeps the prompt-completion
291:        # work minimal — the load + KV-cache-prime cost dominates and
292:        # that's what we actually want to pay before the next loop cycle.
293:        from portfolio.llama_server import query_llama_server
294:        t0 = time.monotonic()
295:        logger.info(
296:            "llm_prewarmer start: counter=%d slot=%s server=%s",
297:            counter, next_slot, server_slot,
298:        )
299:        text = query_llama_server(
300:            server_slot, "test", n_predict=1, temperature=0.0,
301:        )
302:        duration = time.monotonic() - t0
303:
304:        if text is None:
305:            # query_llama_server returns None on failure but does not
306:            # raise. Treat that as a soft failure: state is recorded but
307:            # outcome is logged so we can see prewarm failures in the
308:            # JSONL.
309:            logger.warning(
310:                "llm_prewarmer query returned None: counter=%d slot=%s in %.1fs",
311:                counter, next_slot, duration,
312:            )
313:            _write_state(counter, next_slot, server_slot,
314:                         outcome="query_none", duration_s=duration)
315:            return False
316:
317:        logger.info(
318:            "llm_prewarmer warmed: counter=%d slot=%s server=%s in %.1fs",
319:            counter, next_slot, server_slot, duration,
320:        )
321:        _write_state(counter, next_slot, server_slot,
322:                     outcome="warmed", duration_s=duration)
323:        return True
324:    except Exception as e:
325:        # Defensive backstop. Anything else in this function should have
326:        # already caught its own errors; this is the contract guarantee
327:        # that the prewarmer NEVER raises.
328:        logger.warning(
329:            "llm_prewarmer unexpected failure: counter=%s err=%s",
330:            current_counter, e,
331:        )
332:        return False
333:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""GPU gating and VRAM monitoring for LLM model inference.
2:
3:Ensures only one LLM model uses the GPU at a time. Logs VRAM usage
4:before and after each model load for monitoring.
5:
6:Uses a threading lock for in-process concurrency (ThreadPoolExecutor workers)
7:plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.
8:
9:Stale-lock recovery (2026-05-03):
10:- Reactive: ``gpu_gate()`` calls ``_try_break_stale_lock()`` when another
11:  caller blocks on the lock — same predicate as before BUG-182.
12:- Background: a daemon thread (lazily spawned on first ``gpu_gate()`` call)
13:  runs the same predicate every 30 s. This closes the liveness hole that
14:  let the loop wedge for ~25 hours after chronos pid 13152 died holding
15:  the lock 2026-05-02 02:14 (no other acquirer = no break = no recovery).
16:  See ``docs/plans/2026-05-03-gpu-gate-sweeper.md``.
17:"""
18:
19:import logging
20:import os
21:import subprocess
22:import threading
23:import time
24:from contextlib import contextmanager, suppress
25:from pathlib import Path
26:
27:logger = logging.getLogger("portfolio.gpu_gate")
28:
29:# In-process lock — prevents ThreadPoolExecutor workers from racing
30:_THREAD_LOCK = threading.Lock()
31:
32:# File-based lock for cross-process protection
33:_GPU_LOCK_DIR = Path("Q:/models")
34:_GPU_LOCK_FILE = _GPU_LOCK_DIR / ".gpu_lock"
35:_STALE_SECONDS = 300  # 5 min
36:
37:# Stale-lock sweeper daemon (2026-05-03). Module-level singleton so subprocess
38:# workers that import this module only spawn one sweeper, not one per import.
39:_SWEEPER_INTERVAL_SECONDS = 30
40:_SWEEPER_LOCK = threading.Lock()
41:_sweeper_thread: "threading.Thread | None" = None
42:
43:
44:def get_vram_usage() -> dict:
45:    """Query nvidia-smi for current VRAM usage. Returns dict or None on error."""
46:    try:
47:        proc = subprocess.run(
48:            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
49:             "--format=csv,noheader,nounits"],
50:            capture_output=True, text=True, timeout=5,
51:        )
52:        if proc.returncode == 0 and proc.stdout.strip():
53:            parts = [p.strip() for p in proc.stdout.strip().split(",")]
54:            if len(parts) >= 4:
55:                return {
56:                    "used_mb": int(parts[0]),
57:                    "free_mb": int(parts[1]),
58:                    "total_mb": int(parts[2]),
59:                    "gpu_util_pct": int(parts[3]),
60:                }
61:    except Exception:
62:        logger.debug("GPU info query failed", exc_info=True)
63:    return None
64:
65:
66:def _is_stale() -> bool:
67:    try:
68:        return (time.time() - _GPU_LOCK_FILE.stat().st_mtime) > _STALE_SECONDS
69:    except OSError:
70:        return True
71:
72:
73:def _pid_alive(pid: int) -> bool:
74:    """Check if a process is still running. BUG-182."""
75:    if not pid or pid < 0:
76:        return False
77:    try:
78:        import psutil
79:        return psutil.pid_exists(pid)
80:    except ImportError:
81:        # Fallback: assume alive if we can't check
82:        return True
83:
84:
85:def _read_lock() -> dict:
86:    try:
87:        text = _GPU_LOCK_FILE.read_text(encoding="utf-8").strip()
88:        parts = text.split("|")
89:        return {
90:            "model": parts[0] if len(parts) > 0 else "unknown",
91:            "pid": int(parts[1]) if len(parts) > 1 else 0,
92:            "ts": float(parts[2]) if len(parts) > 2 else 0,
93:        }
94:    except (OSError, ValueError):
95:        return {}
96:
97:
98:def _release_lock():
99:    with suppress(OSError):
100:        _GPU_LOCK_FILE.unlink(missing_ok=True)
101:
102:
103:def _try_break_stale_lock() -> bool:
104:    """Reap the lock file iff stale-by-mtime AND owner pid is dead.
105:
106:    Returns True if the lock was broken (caller can retry acquire), False
107:    otherwise. Defensive: never raises — the sweeper daemon depends on this.
108:
109:    Called from two paths:
110:    - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
111:    - Sweeper: the background daemon, when no one is waiting.
112:
113:    Both paths must agree on the predicate so behaviour is identical
114:    regardless of which path reaped the lock. Emits the same
115:    ``Breaking stale GPU lock`` warning either way so log-grep tools and
116:    postmortem audits work uniformly.
117:    """
118:    try:
119:        if not _GPU_LOCK_FILE.exists():
120:            return False
121:        if not _is_stale():
122:            return False
123:        info = _read_lock()
124:        pid = info.get("pid", 0)
125:        if _pid_alive(pid):
126:            return False
127:        logger.warning("Breaking stale GPU lock: %s (pid=%s, dead)",
128:                       info.get("model"), pid)
129:        _release_lock()
130:        return True
131:    except Exception as exc:
132:        # The sweeper must NEVER crash — a dead daemon stops sweeping forever.
133:        logger.debug("Stale-lock sweep error: %s", exc)
134:        return False
135:
136:
137:def _sweeper_loop():
138:    """Background daemon: reap stale-dead locks every 30 s.
139:
140:    Wedge-recovery story (2026-05-02): chronos pid 13152 died holding the
141:    lock at 02:14. No one tried to acquire while the loop was stuck inside
142:    its LLM batch, so ``_is_stale()`` was never checked. Loop wedged for
143:    ~25 hours until a system reboot. This daemon closes that hole.
144:    """
145:    while True:
146:        try:
147:            time.sleep(_SWEEPER_INTERVAL_SECONDS)
148:            _try_break_stale_lock()
149:        except Exception as exc:
150:            # Defence-in-depth — _try_break_stale_lock already swallows but
151:            # any future code added here must also keep the daemon alive.
152:            logger.debug("Sweeper loop error: %s", exc)
153:
154:
155:def _start_sweeper():
156:    """Spawn the sweeper daemon (idempotent, thread-safe).
157:
158:    Lazily called from ``gpu_gate()`` so:
159:    - Subprocess workers that import this module but never call
160:      ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
161:      do NOT spawn a redundant daemon.
162:    - Tests can reset ``_sweeper_thread = None`` and re-trigger spawn.
163:
164:    If the daemon ever dies (it shouldn't — both layers swallow exceptions)
165:    a future call will respawn it.
166:    """
167:    global _sweeper_thread
168:    with _SWEEPER_LOCK:
169:        if _sweeper_thread is None or not _sweeper_thread.is_alive():
170:            t = threading.Thread(
171:                target=_sweeper_loop,
172:                name="gpu-gate-sweeper",
173:                daemon=True,
174:            )
175:            _sweeper_thread = t
176:            t.start()
177:
178:
179:@contextmanager
180:def gpu_gate(model_name: str, timeout: float = 60):
181:    """Acquire exclusive GPU access, log VRAM before/after.
182:
183:    Uses a two-layer lock:
184:    1. threading.Lock for in-process concurrency (ThreadPoolExecutor workers)
185:    2. File-based lock for cross-process protection (metals loop, etc.)
186:
187:    Args:
188:        model_name: e.g. "ministral-3", "qwen3", "chronos"
189:        timeout: max seconds to wait for lock
190:
191:    Yields:
192:        True if acquired, False if timed out.
193:    """
194:    # Lazy-spawn the stale-lock sweeper. Idempotent so no cost after the
195:    # first call. See _start_sweeper() for the rationale.
196:    _start_sweeper()
197:
198:    deadline = time.time() + timeout
199:
200:    # Layer 1: In-process thread lock (prevents ThreadPoolExecutor races)
201:    remaining = deadline - time.time()
202:    thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
203:    if not thread_acquired:
204:        logger.warning("GPU thread-lock timeout (%ss) for %s", timeout, model_name)
205:        yield False
206:        return
207:
208:    try:
209:        # Layer 2: File-based lock (cross-process)
210:        file_acquired = False
211:        while time.time() < deadline:
212:            try:
213:                # Atomic create — fails if file already exists (no TOCTOU race)
214:                fd = os.open(str(_GPU_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
215:                # H23/CI1: Always close fd in finally to prevent leak if write raises.
216:                try:
217:                    os.write(fd, f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}".encode())
218:                finally:
219:                    os.close(fd)
220:                file_acquired = True
221:                break
222:            except FileExistsError:
223:                # Lock file exists — check if same process (re-entry) or stale.
224:                info = _read_lock()
225:                if info.get("pid") == os.getpid():
226:                    # Re-entry from same process (shouldn't happen with thread lock, but safe)
227:                    file_acquired = True
228:                    break
229:                # BUG-182: Only break stale lock if owning process is dead.
230:                # Helper is shared with the sweeper daemon so the two paths
231:                # agree on the predicate.
232:                if _try_break_stale_lock():
233:                    continue  # retry atomic create
234:                logger.debug("GPU file-locked by %s, waiting...", info.get("model", "?"))
235:                time.sleep(1.0)
236:
237:        if not file_acquired:
238:            info = _read_lock()
239:            logger.warning("GPU file-lock timeout (%ss) — held by %s", timeout, info.get("model", "?"))
240:            yield False
241:            return
242:
243:        # Log VRAM at acquire
244:        vram = get_vram_usage()
245:        if vram:
246:            logger.info(
247:                "GPU gate ACQUIRED by %s — VRAM: %dMB used / %dMB free / %dMB total (GPU %d%%)",
248:                model_name, vram["used_mb"], vram["free_mb"], vram["total_mb"], vram["gpu_util_pct"],
249:            )
250:
251:        t0 = time.time()
252:        try:
253:            yield True
254:        finally:
255:            elapsed = time.time() - t0
256:            vram = get_vram_usage()
257:            if vram:
258:                logger.info(
259:                    "GPU gate RELEASED by %s after %.1fs — VRAM: %dMB used / %dMB free",
260:                    model_name, elapsed, vram["used_mb"], vram["free_mb"],
261:                )
262:            _release_lock()
263:    finally:
264:        _THREAD_LOCK.release()
265:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/llm_batch.py:11:Uses query_llama_server_batch() which holds the file lock for the entire
portfolio/llm_batch.py:15:sentiment. Previously fingpt ran in its own bespoke NDJSON daemon
portfolio/llm_batch.py:16:(scripts/fingpt_daemon.py) on CPU (~60-150s/cycle). Moving it into this
portfolio/llm_batch.py:17:batched rotation lets it use full GPU offload (-ngl 99) in its own
portfolio/llm_batch.py:24:import threading
portfolio/llm_batch.py:29:_lock = threading.Lock()
portfolio/llm_batch.py:55:#    a baseline before the rotation begins.
portfolio/llm_batch.py:61:#    would also poison _loading_keys when the rotation skips.
portfolio/llm_batch.py:64:#    That way rotation is driven by actual LLM invocations, not by idle cache-
portfolio/llm_batch.py:68:#    LLMs run so we have a full baseline before rotation kicks in. Subsequent
portfolio/llm_batch.py:105:    with _lock:
portfolio/llm_batch.py:112:    with _lock:
portfolio/llm_batch.py:130:    with _lock:
portfolio/llm_batch.py:139:    """Flush a batch using query_llama_server_batch (atomic, lock held for entire phase)."""
portfolio/llm_batch.py:141:        from portfolio.llama_server import query_llama_server_batch
portfolio/llm_batch.py:153:    texts = query_llama_server_batch(model_name, prompts_and_params)
portfolio/llm_batch.py:167:    Called once after ThreadPoolExecutor completes in main.py.
portfolio/llm_batch.py:173:    rotation counter in shared_state._full_llm_cycle_count if at least one
portfolio/llm_batch.py:177:    with _lock:
portfolio/llm_batch.py:188:    # Log which LLMs actually ran this cycle vs which were rotation-gated
portfolio/llm_batch.py:189:    # out at the call site. Useful for debugging rotation behaviour in logs.
portfolio/llm_batch.py:191:    rotation_slot = (
portfolio/llm_batch.py:196:        "LLM batch start: rotation_slot=%s counter=%d queues M=%d Q=%d F=%d",
portfolio/llm_batch.py:197:        rotation_slot, _ss._full_llm_cycle_count, len(m_batch), len(q_batch), len(f_batch),
portfolio/llm_batch.py:206:    # Phase 1: All Ministral queries (lock held for entire phase)
portfolio/llm_batch.py:228:    # Phase 2: All Qwen3 queries (lock held for entire phase)
portfolio/llm_batch.py:251:    # Phase 3: All fingpt sentiment queries (lock held for entire phase)
portfolio/llm_batch.py:253:    # scripts/fingpt_daemon.py NDJSON daemon with the shared llama_server
portfolio/llm_batch.py:254:    # rotation, trading ~1 extra swap for a ~70-120s reduction in fingpt
portfolio/llm_batch.py:296:    # Advance rotation counter — next flush will target the next LLM in rotation.
portfolio/llm_batch.py:302:    # 2026-05-11 (feat/llm-prewarmer Stage 3 Phase 1): pre-warm the NEXT
portfolio/llm_batch.py:303:    # LLM in rotation right now, while we still hold no Chronos/gpu_gate.
portfolio/llm_batch.py:305:    # required model is already resident — Chronos's gpu_gate("chronos",
portfolio/llm_batch.py:306:    # timeout=30) no longer races a mid-flight cold swap. The prewarmer
portfolio/llm_batch.py:308:    # outer try/except as a second backstop because a broken prewarmer
portfolio/llm_batch.py:311:        from portfolio.llm_prewarmer import prewarm_next_model
portfolio/llm_batch.py:312:        prewarm_next_model(_ss._full_llm_cycle_count)
portfolio/llm_batch.py:314:        logger.warning("llm prewarmer dispatch failed (non-fatal): %s", e)
portfolio/llm_batch.py:358:        # response parsers that were originally used by the retired daemon.
portfolio/llm_batch.py:379:                headlines_block = "\n".join(f"- {h}" for h in texts[:20])
portfolio/llm_batch.py:382:                    headlines_block=headlines_block,
portfolio/llm_batch.py:400:                # Headlines mode: one prompt per headline. The daemon used
portfolio/llm_batch.py:426:        # Single HTTP batch — llama_server holds its own file lock for the
portfolio/llm_batch.py:428:        from portfolio.llama_server import query_llama_server_batch
portfolio/llm_batch.py:429:        texts_out = query_llama_server_batch("finance-llama-8b", prompts_and_params)
portfolio/llm_batch.py:434:        # broke, llama-server crashed mid-batch, file-lock starvation).
portfolio/llm_batch.py:480:                # Cumulative: daemon applied a +0.1 confidence boost when
portfolio/llama_server.py:7:Cross-process coordination via file lock: both main.py and metals_loop.py
portfolio/llama_server.py:8:can call query_llama_server(), and the lock prevents simultaneous swaps.
portfolio/llama_server.py:11:    from portfolio.llama_server import query_llama_server, stop_all_servers
portfolio/llama_server.py:13:    text = query_llama_server("ministral3", prompt)
portfolio/llama_server.py:14:    text = query_llama_server("qwen3", prompt)
portfolio/llama_server.py:15:    text = query_llama_server("ministral8_lora", prompt)
portfolio/llama_server.py:21:import subprocess
portfolio/llama_server.py:22:import threading
portfolio/llama_server.py:36:_LOCK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "llama_server.lock")
portfolio/llama_server.py:70:    # fingpt → llm_batch rotation migration. Previously this GGUF was loaded by a
portfolio/llama_server.py:71:    # bespoke NDJSON daemon (scripts/fingpt_daemon.py), first on GPU all-layers
portfolio/llama_server.py:74:    # into the shared llama_server rotation lets fingpt use full GPU offload
portfolio/llama_server.py:91:_thread_lock = threading.Lock()
portfolio/llama_server.py:92:_local_proc = None       # Popen if this process started the server
portfolio/llama_server.py:96:def _kill_by_port():
portfolio/llama_server.py:101:            result = subprocess.run(
portfolio/llama_server.py:103:                capture_output=True, text=True, timeout=10,
portfolio/llama_server.py:105:            pids_to_kill = set()
portfolio/llama_server.py:111:                            pids_to_kill.add(int(parts[-1]))
portfolio/llama_server.py:112:            for pid in pids_to_kill:
portfolio/llama_server.py:115:                    subprocess.run(
portfolio/llama_server.py:116:                        ["taskkill", "/F", "/PID", str(pid)],
portfolio/llama_server.py:117:                        capture_output=True, timeout=10,
portfolio/llama_server.py:120:            result = subprocess.run(
portfolio/llama_server.py:122:                capture_output=True, text=True, timeout=10,
portfolio/llama_server.py:128:                        os.kill(pid, 9)
portfolio/llama_server.py:130:        logger.debug("Port kill check failed: %s", e)
portfolio/llama_server.py:134:    """Verify a PID is actually a llama-server process before killing it.
portfolio/llama_server.py:141:            result = subprocess.run(
portfolio/llama_server.py:143:                capture_output=True, text=True, timeout=5,
portfolio/llama_server.py:156:def _kill_server_by_pid():
portfolio/llama_server.py:159:    Validates the process is actually llama-server before killing to
portfolio/llama_server.py:169:                    logger.warning("PID %d from pid file is not llama-server, skipping kill", pid)
portfolio/llama_server.py:172:                    subprocess.run(
portfolio/llama_server.py:173:                        ["taskkill", "/F", "/PID", str(pid)],
portfolio/llama_server.py:174:                        capture_output=True, timeout=10,
portfolio/llama_server.py:177:                    os.kill(pid, 9)
portfolio/llama_server.py:180:        logger.debug("Failed to kill server pid=%s", pid)
portfolio/llama_server.py:212:        r = _requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=2)
portfolio/llama_server.py:224:    _kill_server_by_pid()
portfolio/llama_server.py:226:    # Also kill our local ref if we started it
portfolio/llama_server.py:230:            _local_proc.wait(timeout=10)
portfolio/llama_server.py:233:                _local_proc.kill()
portfolio/llama_server.py:237:    # Safety net: kill anything still on port 8787 (catches orphaned servers
portfolio/llama_server.py:239:    _kill_by_port()
portfolio/llama_server.py:249:    in _wait_for_vram_reclaim below. pynvml is not installed in the main venv
portfolio/llama_server.py:255:        result = subprocess.run(
portfolio/llama_server.py:259:            timeout=2,
portfolio/llama_server.py:274:# so a 5 s TTL is plenty and keeps the polling loop in _wait_for_vram_reclaim
portfolio/llama_server.py:285:    5 s so the active-poll loop in _wait_for_vram_reclaim doesn't hammer
portfolio/llama_server.py:286:    nvidia-smi. Returns False on any error — never blocks the finance loop.
portfolio/llama_server.py:294:        result = subprocess.run(
portfolio/llama_server.py:298:            timeout=2,
portfolio/llama_server.py:309:# Matches the threshold used by _wait_for_vram_reclaim's plex_safe mode and
portfolio/llama_server.py:311:# bypass query_llama_server (subprocess fallbacks) get identical protection.
portfolio/llama_server.py:319:    (subprocess fallbacks in `qwen3_signal._call_qwen3`, `ministral_signal._call_model`,
portfolio/llama_server.py:322:    The check is intentionally conservative: only blocks when *both* conditions
portfolio/llama_server.py:326:    subprocess that could evict Plex's NVENC context.
portfolio/llama_server.py:329:    block the finance loop on a tooling failure — a missed signal is recoverable,
portfolio/llama_server.py:330:    a permanent block is not.
portfolio/llama_server.py:340:def _wait_for_vram_reclaim(min_free_mb: int = 5632, max_wait: float = 4.0,
portfolio/llama_server.py:342:    """Poll nvidia-smi until at least `min_free_mb` is free, up to `max_wait` seconds.
portfolio/llama_server.py:344:    Returns the wall-clock seconds spent waiting, for logging/observability.
portfolio/llama_server.py:362:    free-VRAM floor to 7168 MB (>=7 GB free) and extends the timeout to 30 s.
portfolio/llama_server.py:374:        max_wait = max(max_wait, 30.0)
portfolio/llama_server.py:376:    deadline = start + max_wait
portfolio/llama_server.py:380:        time.sleep(max_wait)
portfolio/llama_server.py:381:        return max_wait
portfolio/llama_server.py:407:    # 2026-05-11 (plex-vram-coord): detect Plex transcoding before the kill so
portfolio/llama_server.py:412:    # reclaim wait + into the abort decision below.
portfolio/llama_server.py:417:            "(>=7 GB free, <=30 s wait) before loading %s",
portfolio/llama_server.py:424:    # asynchronous — see _wait_for_vram_reclaim docstring for the full history
portfolio/llama_server.py:428:    waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0, plex_safe=plex_active)
portfolio/llama_server.py:429:    logger.debug("VRAM reclaim poll: %.2fs before launching %s", waited, name)
portfolio/llama_server.py:432:    # safe headroom within max_wait, loading the new model is more likely to
portfolio/llama_server.py:434:    # query_llama_server) returns None on False, and the caller falls back to
portfolio/llama_server.py:435:    # its subprocess inference path. Slower than HTTP but never racing.
portfolio/llama_server.py:456:        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
portfolio/llama_server.py:471:        proc.kill()
portfolio/llama_server.py:488:def _acquire_file_lock(timeout=300):
portfolio/llama_server.py:489:    """Acquire cross-process file lock. Returns lock file handle or None.
portfolio/llama_server.py:491:    Timeout must exceed the HTTP query timeout (240s) to prevent callers
portfolio/llama_server.py:492:    from falling back to subprocess while the server is still handling a
portfolio/llama_server.py:496:    deadline = time.time() + timeout
portfolio/llama_server.py:505:            # Check if lock is stale (owner dead)
portfolio/llama_server.py:508:                    lock_pid = int(f.read().strip())
portfolio/llama_server.py:511:                    result = subprocess.run(
portfolio/llama_server.py:512:                        ["tasklist", "/FI", f"PID eq {lock_pid}"],
portfolio/llama_server.py:513:                        capture_output=True, text=True, timeout=5,
portfolio/llama_server.py:515:                    if str(lock_pid) not in result.stdout:
portfolio/llama_server.py:519:                    os.kill(lock_pid, 0)  # raises if dead
portfolio/llama_server.py:525:    logger.warning("llama-server file lock timeout (%ds)", timeout)
portfolio/llama_server.py:529:def _release_file_lock(fh):
portfolio/llama_server.py:530:    """Release cross-process file lock."""
portfolio/llama_server.py:538:def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
portfolio/llama_server.py:542:    Thread-safe and cross-process-safe via file lock.
portfolio/llama_server.py:543:    Returns completion text or None (caller should fall back to subprocess).
portfolio/llama_server.py:549:    # BUG-165: Hold both locks for the entire model-swap + query operation.
portfolio/llama_server.py:550:    # Releasing locks between swap and query allowed another thread/process to
portfolio/llama_server.py:551:    # swap the model mid-query, killing the server and causing silent failures.
portfolio/llama_server.py:553:    with _thread_lock:
portfolio/llama_server.py:554:        fh = _acquire_file_lock(timeout=300)
portfolio/llama_server.py:568:            _release_file_lock(fh)
portfolio/llama_server.py:572:    """Send an HTTP completion request. No locking — caller must hold locks.
portfolio/llama_server.py:581:    only in the per-ticker Market Data / Sentiment / Headlines block. On
portfolio/llama_server.py:599:        timeout=240,
portfolio/llama_server.py:606:def query_llama_server_batch(name, prompts_and_params):
portfolio/llama_server.py:607:    """Query the server for multiple prompts, holding the lock for the entire batch.
portfolio/llama_server.py:623:    with _thread_lock:
portfolio/llama_server.py:624:        fh = _acquire_file_lock(timeout=300)
portfolio/llama_server.py:644:            _release_file_lock(fh)
portfolio/llama_server.py:650:    with _thread_lock:
portfolio/llama_server.py:657:    with _thread_lock:
portfolio/qwen3_signal.py:4:Falls back to subprocess if server unavailable.
portfolio/qwen3_signal.py:5:Uses GPU lock to coordinate with Ministral.
portfolio/qwen3_signal.py:13:import subprocess
portfolio/qwen3_signal.py:17:from portfolio.gpu_gate import gpu_gate
portfolio/qwen3_signal.py:18:from portfolio.llama_server import query_llama_server
portfolio/qwen3_signal.py:19:from portfolio.subprocess_utils import kill_orphaned_llama, run_safe
portfolio/qwen3_signal.py:29:    """Extract JSON (object or array) from subprocess stdout."""
portfolio/qwen3_signal.py:61:    """Call Qwen3-8B, preferring persistent llama-server, with subprocess fallback."""
portfolio/qwen3_signal.py:65:    text = query_llama_server("qwen3", prompt, n_predict=1024, temperature=0.6,
portfolio/qwen3_signal.py:74:    # 2026-05-11 (plex-vram-coord): query_llama_server returning None can mean
portfolio/qwen3_signal.py:76:    # the latter case, the subprocess fallback below would cold-start an 8B
portfolio/qwen3_signal.py:87:        logger.warning("qwen3: abstaining — Plex transcoding and VRAM <7168MB; skipping subprocess fallback")
portfolio/qwen3_signal.py:90:    # Fallback: subprocess (cold start)
portfolio/qwen3_signal.py:91:    logger.info("llama-server unavailable for qwen3, falling back to subprocess")
portfolio/qwen3_signal.py:107:            timeout=240,
portfolio/qwen3_signal.py:109:    except subprocess.TimeoutExpired as e:
portfolio/qwen3_signal.py:111:        logger.error("Qwen3 subprocess timed out after 240s — stderr: %s", stderr_text)
portfolio/qwen3_signal.py:122:    """Call Qwen3-8B inference subprocess in batch mode.
portfolio/qwen3_signal.py:146:        timeout=60 + 30 * len(contexts),  # 60s base + 30s per ticker (extended for deeper reasoning)
portfolio/qwen3_signal.py:166:        killed = kill_orphaned_llama()
portfolio/qwen3_signal.py:167:        if killed:
portfolio/qwen3_signal.py:168:            logger.warning("Reaped %d orphaned llama process(es)", killed)
portfolio/qwen3_signal.py:172:        logger.debug("kill_orphaned_llama failed", exc_info=True)
portfolio/qwen3_signal.py:173:    with gpu_gate("qwen3", timeout=300) as acquired:
portfolio/qwen3_signal.py:175:            logger.warning("GPU gate timeout — returning HOLD")
portfolio/qwen3_signal.py:178:        from portfolio.gpu_gate import get_vram_usage
portfolio/llm_calibration.py:6:accuracy logic with thread-locks, TTL caches, and horizon fanout. Mixing
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Batch queue for LLM signals — eliminates model swap overhead.
2:
3:During parallel ticker processing, expired LLM cache entries are enqueued
4:instead of triggering immediate model loads. After all tickers finish,
5:flush_llm_batch() processes them grouped by model: load Ministral once →
6:query all tickers → swap to Qwen3 once → query all tickers → swap to fingpt
7:once → run all sentiment prompts.
8:
9:Result: max 2 model swaps per cycle instead of N swaps.
10:
11:Uses query_llama_server_batch() which holds the file lock for the entire
12:model phase, preventing the metals loop from swapping mid-batch (Codex #4).
13:
14:2026-04-09 update (feat/fingpt-in-llmbatch): added Phase 3 for fingpt
15:sentiment. Previously fingpt ran in its own bespoke NDJSON daemon
16:(scripts/fingpt_daemon.py) on CPU (~60-150s/cycle). Moving it into this
17:batched rotation lets it use full GPU offload (-ngl 99) in its own
18:llama-server phase, trading one extra swap (~10-25s) for a ~70-120s
19:reduction in fingpt inference time. See project_fingpt_llmbatch_session
20:memory entry for the full design rationale.
21:"""
22:
23:import logging
24:import threading
25:import time
26:
27:logger = logging.getLogger("portfolio.llm_batch")
28:
29:_lock = threading.Lock()
30:_ministral_queue: list[tuple[str, dict]] = []   # (cache_key, context)
31:_qwen3_queue: list[tuple[str, dict]] = []       # (cache_key, context)
32:# _fingpt_queue entries are (ab_key, sub_key, context) — sub_key is
33:# "headlines" for per-headline inference or "cumul:<N>" for a cumulative
34:# cluster. The ab_key is shared by all fingpt calls for a single ticker's
35:# get_sentiment() invocation so the results can be stitched back into one
36:# sentiment_ab_log.jsonl entry by sentiment.flush_ab_log() post-cycle.
37:_fingpt_queue: list[tuple[str, str, dict]] = []
38:
39:
40:# 2026-04-10 (perf/llama-swap-reduction) — ROTATION SCHEDULING
41:#
42:# Rotation across the three llama-server LLMs reduces the LLM batch phase
43:# from running all 3 models every cycle (~85 s: 40 s Ministral + 19 s Qwen3
44:# + 9 s fingpt + 15-18 s of swaps) to running ONE model per cycle (~25-40 s
45:# depending on which one). Each LLM still gets a fresh vote every 3rd full-
46:# LLM batch, and _cached_or_enqueue returns stale data on the off-cycle 2
47:# of 3 cycles (max staleness is bounded by max_stale_factor=5 passed at the
48:# call site in signal_engine.py / sentiment.py).
49:#
50:# Design decisions (see docs/PLAN.md / plan file for full rationale):
51:#
52:# 1. Counter lives in shared_state._full_llm_cycle_count, increments AFTER
53:#    flush when the batch actually had work. In-memory only — restart resets
54:#    to 0 and triggers a warmup cycle that runs all three models to establish
55:#    a baseline before the rotation begins.
56:#
57:# 2. Rotation gate sits at the _cached_or_enqueue caller via should_enqueue_fn,
58:#    NOT inside enqueue_ministral/qwen3 themselves, because the enqueue helpers
59:#    also need to be callable directly (from sentiment.py for fingpt) without
60:#    going through _cached_or_enqueue. Gating inside the enqueue functions
61:#    would also poison _loading_keys when the rotation skips.
62:#
63:# 3. Counter advances once per flush-with-work, not once per loop iteration.
64:#    That way rotation is driven by actual LLM invocations, not by idle cache-
65:#    hit cycles where nothing needs to run.
66:#
67:# 4. Warmup: on the very first flush after process start (counter == 0), ALL
68:#    LLMs run so we have a full baseline before rotation kicks in. Subsequent
69:#    flushes rotate.
70:_LLM_ROTATION: tuple[str, ...] = ("ministral", "qwen3", "fingpt")
71:
72:
73:def is_llm_on_cycle(llm_name: str) -> bool:
74:    """Return True if `llm_name` is scheduled to run during the current cycle.
75:
76:    Called at enqueue time to decide whether to skip the enqueue. The current
77:    cycle's slot is `(shared_state._full_llm_cycle_count - 1) % 3` because
78:    the counter advances AFTER the flush — at enqueue time, the counter
79:    represents "how many flushes have already completed" and the next slot
80:    is `counter % 3`, but we want to treat "counter == 0" as a warmup in
81:    which everything runs. So:
82:
83:        counter == 0  → warmup → every LLM returns True
84:        counter == 1  → slot 0 → ministral only
85:        counter == 2  → slot 1 → qwen3 only
86:        counter == 3  → slot 2 → fingpt only
87:        counter == 4  → slot 0 → ministral again
88:        ...
89:
90:    Unknown llm_name raises ValueError (from tuple.index) — that's a
91:    programming error we want to catch in tests rather than silently
92:    return False.
93:    """
94:    from portfolio import shared_state as _ss
95:    count = _ss._full_llm_cycle_count
96:    if count == 0:
97:        return True  # warmup — run everything the first time through
98:    idx = _LLM_ROTATION.index(llm_name)  # raises ValueError for bad names
99:    slot = (count - 1) % len(_LLM_ROTATION)
100:    return slot == idx
101:
102:
103:def enqueue_ministral(cache_key, context):
104:    """Add a Ministral cache miss to the batch queue."""
105:    with _lock:
106:        if not any(k == cache_key for k, _ in _ministral_queue):
107:            _ministral_queue.append((cache_key, context))
108:
109:
110:def enqueue_qwen3(cache_key, context):
111:    """Add a Qwen3 cache miss to the batch queue."""
112:    with _lock:
113:        if not any(k == cache_key for k, _ in _qwen3_queue):
114:            _qwen3_queue.append((cache_key, context))
115:
116:
117:def enqueue_fingpt(ab_key: str, sub_key: str, context: dict) -> None:
118:    """Add a fingpt sentiment request to the batch queue.
119:
120:    Args:
121:        ab_key: Shared key identifying the parent get_sentiment() call
122:            (e.g. "BTC:2026-04-09T18:04:00+00:00"). All fingpt calls for
123:            the same get_sentiment() invocation share this key so their
124:            results can be merged into one A/B log entry.
125:        sub_key: "headlines" for per-headline inference, or "cumul:<N>"
126:            for the N-th cumulative cluster.
127:        context: {"mode": "headlines"|"cumulative", "texts": [...],
128:                  "ticker": "BTC"}
129:    """
130:    with _lock:
131:        # Deduplicate on (ab_key, sub_key). Unlike ministral/qwen3 we use a
132:        # composite key because one ticker may enqueue both headlines and
133:        # multiple cumulative clusters in the same get_sentiment() call.
134:        if not any(k == ab_key and s == sub_key for k, s, _ in _fingpt_queue):
135:            _fingpt_queue.append((ab_key, sub_key, context))
136:
137:
138:def _flush_via_server(model_name, batch, build_prompt_fn, parse_response_fn, stop_tokens):
139:    """Flush a batch using query_llama_server_batch (atomic, lock held for entire phase)."""
140:    try:
141:        from portfolio.llama_server import query_llama_server_batch
142:    except ImportError:
143:        return {}
144:
145:    prompts_and_params = []
146:    for _cache_key, ctx in batch:
147:        prompt = build_prompt_fn(ctx)
148:        prompts_and_params.append({
149:            "prompt": prompt,
150:            "stop": stop_tokens,
151:        })
152:
153:    texts = query_llama_server_batch(model_name, prompts_and_params)
154:
155:    results = {}
156:    for (cache_key, _ctx), text in zip(batch, texts):
157:        if text is not None:
158:            parsed = parse_response_fn(text)
159:            if parsed:
160:                results[cache_key] = parsed
161:    return results
162:
163:
164:def flush_llm_batch():
165:    """Process all queued LLM requests, batched by model.
166:
167:    Called once after ThreadPoolExecutor completes in main.py.
168:    Returns dict of {cache_key: result} for cache updates — ministral and
169:    qwen3 results only. Fingpt results are stashed into sentiment._pending_ab_entries
170:    via _stash_fingpt_result() and emitted later by sentiment.flush_ab_log().
171:
172:    2026-04-10 (perf/llama-swap-reduction): after processing, advances the
173:    rotation counter in shared_state._full_llm_cycle_count if at least one
174:    phase had queued work. This is what makes is_llm_on_cycle() rotate through
175:    ministral → qwen3 → fingpt across successive flushes.
176:    """
177:    with _lock:
178:        m_batch = list(_ministral_queue)
179:        q_batch = list(_qwen3_queue)
180:        f_batch = list(_fingpt_queue)
181:        _ministral_queue.clear()
182:        _qwen3_queue.clear()
183:        _fingpt_queue.clear()
184:
185:    if not m_batch and not q_batch and not f_batch:
186:        return {}
187:
188:    # Log which LLMs actually ran this cycle vs which were rotation-gated
189:    # out at the call site. Useful for debugging rotation behaviour in logs.
190:    from portfolio import shared_state as _ss
191:    rotation_slot = (
192:        "warmup" if _ss._full_llm_cycle_count == 0
193:        else _LLM_ROTATION[(_ss._full_llm_cycle_count - 1) % len(_LLM_ROTATION)]
194:    )
195:    logger.info(
196:        "LLM batch start: rotation_slot=%s counter=%d queues M=%d Q=%d F=%d",
197:        rotation_slot, _ss._full_llm_cycle_count, len(m_batch), len(q_batch), len(f_batch),
198:    )
199:
200:    results = {}
201:    t0 = time.monotonic()
202:    m_parsed = 0  # 2026-05-03 (fix/fingpt-batch-observability): track per-phase
203:    q_parsed = 0  # parsed counts so the summary log shows results-vs-queue
204:    f_parsed = 0  # rather than the misleading "%d results" of the M+Q dict.
205:
206:    # Phase 1: All Ministral queries (lock held for entire phase)
207:    if m_batch:
208:        logger.info("LLM batch: %d Ministral queries", len(m_batch))
209:        try:
210:            from portfolio.ministral_trader import _build_prompt, _parse_response
211:
212:            def _parse_ministral(text):
213:                decision, reasoning, confidence = _parse_response(text)
214:                result = {
215:                    "original": {"action": decision, "reasoning": reasoning, "model": "Ministral-3-8B"},
216:                    "custom": None,
217:                }
218:                if confidence is not None:
219:                    result["original"]["confidence"] = confidence
220:                return result
221:
222:            phase = _flush_via_server("ministral3", m_batch, _build_prompt, _parse_ministral, ["[INST]"])
223:            m_parsed = len(phase)
224:            results.update(phase)
225:        except Exception as e:
226:            logger.warning("LLM batch Ministral failed: %s", e)
227:
228:    # Phase 2: All Qwen3 queries (lock held for entire phase)
229:    if q_batch:
230:        logger.info("LLM batch: %d Qwen3 queries", len(q_batch))
231:        try:
232:            from portfolio.qwen3_trader import _build_prompt as _qwen_build
233:            from portfolio.qwen3_trader import _parse_response as _qwen_parse_raw
234:
235:            def _parse_qwen3(text):
236:                decision, reasoning, confidence = _qwen_parse_raw(text)
237:                result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
238:                if confidence is not None:
239:                    result["confidence"] = confidence
240:                return result
241:
242:            phase = _flush_via_server(
243:                "qwen3", q_batch, _qwen_build, _parse_qwen3,
244:                ["<|endoftext|>", "<|im_end|>"],
245:            )
246:            q_parsed = len(phase)
247:            results.update(phase)
248:        except Exception as e:
249:            logger.warning("LLM batch Qwen3 failed: %s", e)
250:
251:    # Phase 3: All fingpt sentiment queries (lock held for entire phase)
252:    # Added 2026-04-09 (feat/fingpt-in-llmbatch). Replaces the bespoke
253:    # scripts/fingpt_daemon.py NDJSON daemon with the shared llama_server
254:    # rotation, trading ~1 extra swap for a ~70-120s reduction in fingpt
255:    # inference time per cycle. Fingpt is a SHADOW sentiment signal (does
256:    # not vote) so its failures are log-only — primary sentiment (CryptoBERT
257:    # / Trading-Hero-LLM) is unaffected if this phase breaks.
258:    f_queries = 0  # 2026-05-03: prompts sent (1 per cumulative entry, N per
259:                   # headlines entry). Used as the F denominator so the unit
260:                   # is "prompts" not "queue groups" — apples-to-apples with
261:                   # the fingpt parser stage that produces one parsed dict
262:                   # per prompt.
263:    if f_batch:
264:        # 2026-05-03 (fix/fingpt-batch-observability codex P3): renamed from
265:        # "%d fingpt queries" because each f_batch entry can fan out to many
266:        # per-headline prompts in _flush_fingpt_phase. The old wording showed
267:        # "1 fingpt queries" right next to a summary line claiming "F=10/10",
268:        # which was confusing — "groups" matches what's being counted.
269:        logger.info("LLM batch: %d fingpt groups", len(f_batch))
270:        # 2026-05-03 (fix/fingpt-batch-observability): _flush_fingpt_phase now
271:        # returns a metrics dict on every code path. Used in the summary log
272:        # below so a fingpt-only cycle no longer reports "0 results" when
273:        # fingpt actually stashed its outputs to sentiment._pending_ab_entries.
274:        f_metrics = _flush_fingpt_phase(f_batch)
275:        f_parsed = f_metrics.get("parsed", 0)
276:        f_queries = f_metrics.get("queries", 0)
277:
278:    elapsed = time.monotonic() - t0
279:    # 2026-05-03 (fix/fingpt-batch-observability): replaced the old line
280:    # `"LLM batch: %d results in %.1fs (M:%d Q:%d F:%d)"` which counted
281:    # only Phase 1+2 entries in the local `results` dict — fingpt-only
282:    # cycles always logged "0 results" regardless of actual outcome.
283:    # New format shows parsed/queued for each phase so silent fingpt
284:    # failures (e.g. F=0/6) are visible at a glance. Note: M and Q use
285:    # `len(_batch)` because those phases run 1 prompt per queue entry;
286:    # fingpt uses the metrics-tracked query count because one queue entry
287:    # can fan out to multiple per-headline prompts.
288:    logger.info(
289:        "LLM batch: M=%d/%d Q=%d/%d F=%d/%d in %.1fs",
290:        m_parsed, len(m_batch),
291:        q_parsed, len(q_batch),
292:        f_parsed, f_queries,
293:        elapsed,
294:    )
295:
296:    # Advance rotation counter — next flush will target the next LLM in rotation.
297:    # Only bumped when at least one phase had work (we already returned early
298:    # for an empty flush above). Wrapping behaviour in is_llm_on_cycle handles
299:    # arbitrary large counters.
300:    _ss._full_llm_cycle_count += 1
301:
302:    # 2026-05-11 (feat/llm-prewarmer Stage 3 Phase 1): pre-warm the NEXT
303:    # LLM in rotation right now, while we still hold no Chronos/gpu_gate.
304:    # Goal: by the time the *next* loop cycle hits (~60 s away), the
305:    # required model is already resident — Chronos's gpu_gate("chronos",
306:    # timeout=30) no longer races a mid-flight cold swap. The prewarmer
307:    # contract is exception-safe (NEVER raises), but we wrap it in an
308:    # outer try/except as a second backstop because a broken prewarmer
309:    # must not cascade into the flush path.
310:    try:
311:        from portfolio.llm_prewarmer import prewarm_next_model
312:        prewarm_next_model(_ss._full_llm_cycle_count)
313:    except Exception as e:
314:        logger.warning("llm prewarmer dispatch failed (non-fatal): %s", e)
315:
316:    return results
317:
318:
319:def _flush_fingpt_phase(f_batch: list[tuple[str, str, dict]]) -> dict:
320:    """Execute Phase 3: load finance-llama-8b once, run all queued sentiment
321:    prompts, stash results in sentiment._pending_ab_entries.
322:
323:    Returns a metrics dict on EVERY code path (success, partial, exception):
324:
325:        {
326:          "queries": int,         # prompts sent to llama-server
327:          "received": int,        # non-None text completions back
328:          "parsed": int,          # parsed dicts (non-None) handed to _stash_fingpt_result
329:          "stashed_groups": int,  # distinct (ab_key, sub_key) groups stashed
330:          "exception": str|None,  # exception class name if the bare except fired
331:        }
332:
333:    Per-item failure (None text from the server) bubbles up to the A/B
334:    logger which writes a tagged fingpt:error entry instead of silently
335:    dropping the sample.
336:
337:    The whole phase is wrapped in try/except so fingpt errors never leak
338:    out into the main loop. Shadow signals must not crash anything above
339:    them. The metrics dict is the observability hook so a silent failure
340:    becomes loud — see callers in flush_llm_batch().
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
340:    becomes loud — see callers in flush_llm_batch().
341:
342:    2026-05-03 (fix/fingpt-batch-observability): added metrics return +
343:    specific failure-mode warnings (server-returned-all-None, parser-
344:    failed-majority). Previously the phase returned None and a single
345:    bare `except` swallowed every error class with one generic warning,
346:    making silent regressions invisible until the A/B log was inspected
347:    by hand.
348:    """
349:    metrics = {
350:        "queries": 0,
351:        "received": 0,
352:        "parsed": 0,
353:        "stashed_groups": 0,
354:        "exception": None,
355:    }
356:    try:
357:        # fingpt_infer provides the prompt templates, stop tokens, and
358:        # response parsers that were originally used by the retired daemon.
359:        # Imported here (lazy) so a missing Q:\models path degrades gracefully
360:        # — if the import fails, fingpt just doesn't run this cycle.
361:        import platform
362:        import sys
363:        if platform.system() == "Windows":
364:            _models_dir = r"Q:\models"
365:        else:
366:            _models_dir = "/home/deck/models"
367:        if _models_dir not in sys.path:
368:            sys.path.insert(0, _models_dir)
369:        import fingpt_infer  # noqa: E402  (path injection above)
370:
371:        # Flatten the batch into per-prompt requests and keep a parallel meta
372:        # list so we can group results back by (ab_key, sub_key) afterward.
373:        prompts_and_params: list[dict] = []
374:        meta: list[tuple[str, str, dict, int]] = []  # (ab_key, sub_key, ctx, prompt_idx_within_call)
375:        for ab_key, sub_key, ctx in f_batch:
376:            mode = ctx.get("mode", "headlines")
377:            texts = ctx.get("texts") or []
378:            if mode == "cumulative":
379:                headlines_block = "\n".join(f"- {h}" for h in texts[:20])
380:                prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
381:                    count=len(texts),
382:                    headlines_block=headlines_block,
383:                )
384:                prompts_and_params.append({
385:                    "prompt": prompt,
386:                    "n_predict": 30,
387:                    "temperature": 0.1,
388:                    # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
389:                    # Old stop ["\n", "<|eot_id|>"] was designed for the Llama-3
390:                    # chat-format prompt that wiroai-finance-llama-8b doesn't
391:                    # recognize. New CUMULATIVE_PROMPT is a plain-text one-shot
392:                    # template that ends each section with a blank line, so
393:                    # "\n\n" is the natural stop. The <|eot_id|> token was never
394:                    # emitted by this model (it's not chat-tuned) so removing
395:                    # it is a no-op.
396:                    "stop": ["\n\n"],
397:                })
398:                meta.append((ab_key, sub_key, ctx, 0))
399:            else:
400:                # Headlines mode: one prompt per headline. The daemon used
401:                # PROMPT_TEMPLATES[name] for the loaded model; llama_server
402:                # loads finance-llama-8b so we index into that entry directly.
403:                template = fingpt_infer.PROMPT_TEMPLATES.get(
404:                    "finance-llama-8b",
405:                    next(iter(fingpt_infer.PROMPT_TEMPLATES.values())),
406:                )
407:                for i, headline in enumerate(texts):
408:                    prompts_and_params.append({
409:                        "prompt": template.format(headline=headline),
410:                        "n_predict": 20,
411:                        "temperature": 0.1,
412:                        # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
413:                        # Same reason as the cumulative case above. The old
414:                        # stop ["\n", "<|eot_id|>", "[INST]"] cut the few-shot
415:                        # prompt apart at the first newline, which is exactly
416:                        # where the expected answer word appears — so even a
417:                        # correctly-answering model would have been silenced.
418:                        "stop": ["\n\n"],
419:                    })
420:                    meta.append((ab_key, sub_key, ctx, i))
421:
422:        metrics["queries"] = len(prompts_and_params)
423:        if not prompts_and_params:
424:            return metrics
425:
426:        # Single HTTP batch — llama_server holds its own file lock for the
427:        # duration so no other process can swap the model mid-phase.
428:        from portfolio.llama_server import query_llama_server_batch
429:        texts_out = query_llama_server_batch("finance-llama-8b", prompts_and_params)
430:        metrics["received"] = sum(1 for t in texts_out if t is not None)
431:
432:        # 2026-05-03: explicit warning when the server returned None for
433:        # every prompt — this is the "silent failure" mode (model swap
434:        # broke, llama-server crashed mid-batch, file-lock starvation).
435:        # Without this line operators see only the summary "F=0/N" and
436:        # have to dig through agent.log to figure out which layer broke.
437:        if metrics["queries"] > 0 and metrics["received"] == 0:
438:            logger.warning(
439:                "fingpt: server returned None for all %d prompts "
440:                "(likely llama_server unavailable or swap failed)",
441:                metrics["queries"],
442:            )
443:
444:        # Group results back by (ab_key, sub_key) → list of per-prompt parsed dicts.
445:        grouped: dict[tuple[str, str], list[tuple[int, dict | None, dict]]] = {}
446:        for (ab_key, sub_key, ctx, prompt_idx), text in zip(meta, texts_out):
447:            parsed = _parse_fingpt_completion(text, fingpt_infer)
448:            if parsed is not None:
449:                metrics["parsed"] += 1
450:            grouped.setdefault((ab_key, sub_key), []).append((prompt_idx, parsed, ctx))
451:
452:        # 2026-05-03: parser-regression warning. If the server gave us text
453:        # but the parser produced None for >50% of completions, something
454:        # broke upstream in fingpt_infer (template change, model swap to a
455:        # chat-tuned variant, prompt format drift). The 50% threshold is
456:        # generous — the parser always returns SOME label for non-empty
457:        # text, so a high None rate means the completions themselves are
458:        # garbage (empty / truncated / wrong-language).
459:        if (
460:            metrics["received"] > 0
461:            and metrics["parsed"] * 2 < metrics["received"]
462:        ):
463:            logger.warning(
464:                "fingpt: parser returned None for %d/%d completions "
465:                "(>50%%; possible parser or prompt regression)",
466:                metrics["received"] - metrics["parsed"],
467:                metrics["received"],
468:            )
469:
470:        # Stash each (ab_key, sub_key) result into the sentiment buffer. The
471:        # buffer is consumed by sentiment.flush_ab_log() which runs right
472:        # after flush_llm_batch() in main.py and writes the final A/B log
473:        # entries.
474:        from portfolio.sentiment import _stash_fingpt_result
475:        for (ab_key, sub_key), items in grouped.items():
476:            items.sort(key=lambda t: t[0])
477:            mode = items[0][2].get("mode", "headlines")
478:            if mode == "cumulative":
479:                parsed = items[0][1]
480:                # Cumulative: daemon applied a +0.1 confidence boost when
481:                # len(headlines) >= 5. Replicate here so the A/B log shows
482:                # the same numbers it did pre-migration.
483:                if parsed is not None:
484:                    texts = items[0][2].get("texts") or []
485:                    if len(texts) >= 5:
486:                        parsed = dict(parsed)
487:                        parsed["confidence"] = min(
488:                            parsed.get("confidence", 0.0) + 0.1, 0.95,
489:                        )
490:                    parsed["headline_count"] = len(texts)
491:                    parsed["model"] = "fingpt:cumulative"
492:                _stash_fingpt_result(ab_key, sub_key, parsed)
493:            else:
494:                per_headline = [p for (_idx, p, _c) in items]
495:                _stash_fingpt_result(ab_key, sub_key, per_headline)
496:            metrics["stashed_groups"] += 1
497:    except Exception as e:
498:        # 2026-05-03: log the exception class name + repr so a single grep
499:        # tells you what blew up. The previous bare `except: warning(...,
500:        # exc_info=True)` produced a multi-line traceback that was harder
501:        # to scan in tail/grep contexts (loop_out tail, telegram digests).
502:        metrics["exception"] = type(e).__name__
503:        logger.warning("LLM batch fingpt phase failed: %s", repr(e), exc_info=True)
504:    return metrics
505:
506:
507:def _parse_fingpt_completion(text: str | None, fingpt_infer) -> dict | None:
508:    """Parse one llama-server completion into the dict shape sentiment.py
509:    expects. Returns None on hard failure (the None bubbles up to the A/B
510:    logger which writes a tagged fingpt:error entry).
511:
512:    2026-04-09 (fix/fingpt-parser-prompt): the original fingpt migration
513:    left this wrapper untouched because the parser bug was upstream in
514:    fingpt_infer._parse_sentiment / _estimate_confidence + the Llama-3
515:    chat template in PROMPT_TEMPLATES["finance-llama-8b"]. That was all
516:    fixed in the same commit as this comment update — wiroai-finance-llama-8b
517:    is a completion model and the new few-shot plain-text templates make
518:    it emit clean sentiment words. See /mnt/q/models/fingpt_infer.py for
519:    the parser + template changes.
520:
521:    2026-05-03 (fix/fingpt-batch-observability codex P2): also reject
522:    empty / whitespace-only text. The production fingpt_infer._parse_sentiment
523:    falls back to "neutral" for any unparseable input rather than raising,
524:    AND llama_server._query_http returns "" (not None) for HTTP 200 with
525:    empty body — so without this guard, an empty completion silently scores
526:    as a neutral-with-low-confidence parse and the new "parsed" metric in
527:    _flush_fingpt_phase looks healthy (parsed == queries) when in fact
528:    nothing was produced. Treating empty-text as a parse failure makes the
529:    silent-failure path show up as F=0/N in the summary log, which is the
530:    whole point of this PR.
531:    """
532:    if text is None or not text.strip():
533:        return None
534:    try:
535:        sentiment = fingpt_infer._parse_sentiment(text)
536:        confidence = fingpt_infer._estimate_confidence(text, sentiment)
537:        scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.1}
538:        scores[sentiment] = confidence
539:        remaining = 1.0 - confidence
540:        other_labels = [lb for lb in fingpt_infer.SENTIMENT_LABELS if lb != sentiment]
541:        if other_labels:
542:            share = remaining / len(other_labels)
543:            for ol in other_labels:
544:                scores[ol] = share
545:        return {
546:            "sentiment": sentiment,
547:            "confidence": round(confidence, 4),
548:            "scores": {k: round(v, 4) for k, v in scores.items()},
549:            "model": "fingpt:finance-llama-8b",
550:        }
551:    except Exception:
552:        logger.debug("fingpt completion parse failed for text=%r", text, exc_info=True)
553:        return None
554:
codex
I’ve reached the parts that can silently degrade the whole loop without crashing: the health cache, the prewarmer, and the trigger/tier logic. I’m validating those against the stated operational contracts now, especially anything that can look “healthy” while doing the wrong thing.
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
480:    # Check if correct model is already running (possibly started by another process)
481:    _, current_model = _read_pid_model()
482:    if current_model == name and _is_server_alive():
483:        return True
484:    # Need to swap
485:    return _start_server(name)
486:
487:
488:def _acquire_file_lock(timeout=300):
489:    """Acquire cross-process file lock. Returns lock file handle or None.
490:
491:    Timeout must exceed the HTTP query timeout (240s) to prevent callers
492:    from falling back to subprocess while the server is still handling a
493:    legitimate query (Codex review finding #1).
494:    """
495:    os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
496:    deadline = time.time() + timeout
497:    while time.time() < deadline:
498:        try:
499:            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
500:            fh = os.fdopen(fd, "w")
501:            fh.write(f"{os.getpid()}\n")
502:            fh.flush()
503:            return fh
504:        except FileExistsError:
505:            # Check if lock is stale (owner dead)
506:            try:
507:                with open(_LOCK_FILE, encoding="utf-8") as f:
508:                    lock_pid = int(f.read().strip())
509:                # Check if PID is alive
510:                if platform.system() == "Windows":
511:                    result = subprocess.run(
512:                        ["tasklist", "/FI", f"PID eq {lock_pid}"],
513:                        capture_output=True, text=True, timeout=5,
514:                    )
515:                    if str(lock_pid) not in result.stdout:
516:                        os.remove(_LOCK_FILE)
517:                        continue
518:                else:
519:                    os.kill(lock_pid, 0)  # raises if dead
520:            except (ProcessLookupError, OSError, ValueError):
521:                with suppress(OSError):
522:                    os.remove(_LOCK_FILE)
523:                continue
524:            time.sleep(1)
525:    logger.warning("llama-server file lock timeout (%ds)", timeout)
526:    return None
527:
528:
529:def _release_file_lock(fh):
530:    """Release cross-process file lock."""
531:    if fh is not None:
532:        with suppress(Exception):
533:            fh.close()
534:        with suppress(OSError):
535:            os.remove(_LOCK_FILE)
536:
537:
538:def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
539:                       top_p=0.2, stop=None):
540:    """Query the shared llama-server. Swaps model if needed.
541:
542:    Thread-safe and cross-process-safe via file lock.
543:    Returns completion text or None (caller should fall back to subprocess).
544:    """
545:    cfg = _MODEL_CONFIGS.get(name)
546:    if cfg is None:
547:        return None
548:
549:    # BUG-165: Hold both locks for the entire model-swap + query operation.
550:    # Releasing locks between swap and query allowed another thread/process to
551:    # swap the model mid-query, killing the server and causing silent failures.
552:    # Serialization is correct here — only one 8B model fits in VRAM at a time.
553:    with _thread_lock:
554:        fh = _acquire_file_lock(timeout=300)
555:        if fh is None:
556:            return None
557:        try:
558:            if not _ensure_model(name):
559:                return None
560:            text = _query_http(prompt, n_predict, temperature, top_p, stop)
561:            if text is None:
562:                logger.warning("llama-server %s returned empty response", name)
563:            return text
564:        except Exception as e:
565:            logger.warning("llama-server %s query failed: %s", name, e)
566:            return None
567:        finally:
568:            _release_file_lock(fh)
569:
570:
571:def _query_http(prompt, n_predict=1024, temperature=0.0, top_p=0.2, stop=None):
572:    """Send an HTTP completion request. No locking — caller must hold locks.
573:
574:    2026-04-10 (perf/llama-swap-reduction): added `cache_prompt: true`. This
575:    is a llama.cpp server feature (stable since b2000, our binary is the
576:    cuda13 build from 2025 which supports it) that tells the server to
577:    reuse the KV cache across successive requests whenever the new prompt
578:    shares a token prefix with the previous one. Our Ministral prompts in
579:    _build_prompt share ~300 tokens of fixed boilerplate (the [INST] header,
580:    the analysis questions 1-5, the JSON schema at the bottom) and differ
581:    only in the per-ticker Market Data / Sentiment / Headlines block. On
582:    the 4 sequential per-ticker Ministral queries per cycle, this saves
583:    ~300 prefill tokens × 3 queries = ~900 tokens of prefill work, which
584:    on an 8B Q5_K_M is roughly ~5-15 s per batch. If the server build
585:    doesn't recognize the field, it is silently ignored — no breakage.
586:    """
587:    body = {
588:        "prompt": prompt,
589:        "n_predict": n_predict,
590:        "temperature": temperature,
591:        "top_p": top_p,
592:        "cache_prompt": True,
593:    }
594:    if stop:
595:        body["stop"] = stop
596:    r = _requests.post(
597:        f"http://127.0.0.1:{_PORT}/completion",
598:        json=body,
599:        timeout=240,
600:    )
601:    if r.status_code == 200:
602:        return r.json().get("content", "").strip()
603:    return None
604:
605:
606:def query_llama_server_batch(name, prompts_and_params):
607:    """Query the server for multiple prompts, holding the lock for the entire batch.
608:
609:    Prevents other processes from swapping the model between items (Codex finding #4).
610:
611:    Args:
612:        name: model name (e.g. "ministral3", "qwen3")
613:        prompts_and_params: list of dicts with keys: prompt, n_predict, temperature, top_p, stop
614:
615:    Returns:
616:        list of (completion_text_or_None) in same order as input.
617:    """
618:    cfg = _MODEL_CONFIGS.get(name)
619:    if cfg is None:
620:        return [None] * len(prompts_and_params)
621:
622:    results = []
623:    with _thread_lock:
624:        fh = _acquire_file_lock(timeout=300)
625:        if fh is None:
626:            return [None] * len(prompts_and_params)
627:        try:
628:            if not _ensure_model(name):
629:                return [None] * len(prompts_and_params)
630:            for params in prompts_and_params:
631:                try:
632:                    text = _query_http(
633:                        params["prompt"],
634:                        n_predict=params.get("n_predict", 1024),
635:                        temperature=params.get("temperature", 0.0),
636:                        top_p=params.get("top_p", 0.2),
637:                        stop=params.get("stop"),
638:                    )
639:                    results.append(text)
640:                except Exception as e:
641:                    logger.warning("llama-server batch query failed: %s", e)
642:                    results.append(None)
643:        finally:
644:            _release_file_lock(fh)
645:    return results
646:
647:
648:def stop_server(name=None):
649:    """Stop the llama-server (optionally only if a specific model is loaded)."""
650:    with _thread_lock:
651:        if name is None or _local_model == name:
652:            _stop_server()
653:
654:
655:def stop_all_servers():
656:    """Stop the llama-server regardless of which model is loaded."""
657:    with _thread_lock:
658:        _stop_server()
659:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
340:def _wait_for_vram_reclaim(min_free_mb: int = 5632, max_wait: float = 4.0,
341:                           plex_safe: bool = False) -> float:
342:    """Poll nvidia-smi until at least `min_free_mb` is free, up to `max_wait` seconds.
343:
344:    Returns the wall-clock seconds spent waiting, for logging/observability.
345:
346:    Replaces the hardcoded `time.sleep(4)` that used to follow _stop_server()
347:    in _start_server(). The 4 s sleep was added because Windows VRAM release
348:    is asynchronous — 1 s was insufficient when Ministral (5 GB) was torn
349:    down and Qwen3 (5 GB) needed the memory: Qwen3 would start up seeing only
350:    136 MB free, fail to allocate, and retry for ~90 s. 4 s fixed it but was
351:    conservative: most swaps actually reclaim in 0.5-2 s.
352:
353:    This helper polls every 100 ms and exits as soon as `min_free_mb` is
354:    available. Defaults: 5632 MB (~5.5 GB, enough for an 8B Q4_K_M model +
355:    KV cache) and 4 s ceiling (the original hardcoded sleep). If nvidia-smi
356:    is unavailable (headless VM, non-NVIDIA, permission error), we fall back
357:    to the full 4 s sleep — never faster than the original, always at least
358:    as safe. The feedback memory note on PowerShell quoting applies equally
359:    here: single string, no variable interpolation, nothing bash can eat.
360:
361:    2026-05-11 (plex-vram-coord): added `plex_safe`. When True, raises the
362:    free-VRAM floor to 7168 MB (>=7 GB free) and extends the timeout to 30 s.
363:    Rationale: when Plex is hardware-transcoding it holds a CUDA NVENC
364:    encoder context of ~0.5 GB and is constantly working it. If the swap
365:    proceeds while VRAM is tight, the new model's working-set allocation
366:    forces CUDA to evict Plex's context, which crashes/hangs Plex. The 7 GB
367:    floor covers 8B Q4 weights (~4.5 GB) + KV cache (~0.8 GB) + transient
368:    load peaks (~1 GB) + Plex headroom (~0.5 GB). Caller passes the
369:    plex-active flag through from `_start_server` so we never need to query
370:    nvidia-smi twice for the same swap.
371:    """
372:    if plex_safe:
373:        min_free_mb = max(min_free_mb, 7168)
374:        max_wait = max(max_wait, 30.0)
375:    start = time.time()
376:    deadline = start + max_wait
377:    first_probe = _query_free_vram_mb()
378:    if first_probe is None:
379:        # nvidia-smi unavailable — fall back to the original conservative sleep.
380:        time.sleep(max_wait)
381:        return max_wait
382:    if first_probe >= min_free_mb:
383:        return 0.0  # already enough, no sleep at all
384:    while time.time() < deadline:
385:        time.sleep(0.1)
386:        free = _query_free_vram_mb()
387:        if free is None:
388:            # nvidia-smi broke mid-poll; fall back to full sleep.
389:            remaining = max(0.0, deadline - time.time())
390:            time.sleep(remaining)
391:            return time.time() - start
392:        if free >= min_free_mb:
393:            return time.time() - start
394:    return time.time() - start
395:
396:
397:def _start_server(name):
398:    """Launch llama-server with the given model. Returns True if ready."""
399:    global _local_proc, _local_model
400:    cfg = _MODEL_CONFIGS.get(name)
401:    if cfg is None:
402:        return False
403:    if not os.path.exists(_LLAMA_SERVER) or not os.path.exists(cfg["model"]):
404:        logger.info("llama-server or model %s not found", name)
405:        return False
406:
407:    # 2026-05-11 (plex-vram-coord): detect Plex transcoding before the kill so
408:    # we know whether to require extra VRAM headroom. Plex's NVENC encoder
409:    # context (~0.5 GB) gets evicted by CUDA when the swap pushes total
410:    # allocation past 10 GB on the RTX 3080, hard-crashing Plex (confirmed
411:    # twice 2026-05-10). We probe once per swap and feed the flag through the
412:    # reclaim wait + into the abort decision below.
413:    plex_active = _plex_transcode_active()
414:    if plex_active:
415:        logger.info(
416:            "llama-server: Plex transcoding active, using safe VRAM reclaim "
417:            "(>=7 GB free, <=30 s wait) before loading %s",
418:            name,
419:        )
420:
421:    _stop_server()
422:    # 2026-04-10 (perf/llama-swap-reduction): replaced `time.sleep(4)` with an
423:    # active poll. The old sleep was there because Windows VRAM release is
424:    # asynchronous — see _wait_for_vram_reclaim docstring for the full history
425:    # on why 4 s exists. In steady state most swaps only need 0.5-2 s, so the
426:    # poll saves ~2-3 s per swap (×3 swaps = ~6-10 s/cycle) while preserving
427:    # the 4 s ceiling as a hard fallback.
428:    waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0, plex_safe=plex_active)
429:    logger.debug("VRAM reclaim poll: %.2fs before launching %s", waited, name)
430:
431:    # Plex-aware abort: if Plex is still transcoding AND we couldn't reach the
432:    # safe headroom within max_wait, loading the new model is more likely to
433:    # crash Plex than to succeed cleanly. Caller (_ensure_model →
434:    # query_llama_server) returns None on False, and the caller falls back to
435:    # its subprocess inference path. Slower than HTTP but never racing.
436:    if plex_active:
437:        free_now = _query_free_vram_mb() or 0
438:        if free_now < 7168:
439:            logger.warning(
440:                "llama-server: aborting %s swap — Plex transcoding and only %d MB free (<7168)",
441:                name, free_now,
442:            )
443:            return False
444:
445:    try:
446:        cmd = [
447:            _LLAMA_SERVER,
448:            "-m", cfg["model"],
449:            "--port", str(_PORT),
450:            "--host", "127.0.0.1",
451:            "-ngl", "99",
452:            "-t", "4",
453:            "-c", "4096",
454:        ] + cfg.get("extra_args", [])
455:
456:        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
457:
458:        deadline = time.time() + 90
459:        while time.time() < deadline:
460:            if proc.poll() is not None:
461:                logger.warning("llama-server %s exited during startup (code %s)", name, proc.returncode)
462:                return False
463:            if _is_server_alive():
464:                logger.info("llama-server %s ready on port %d", name, _PORT)
465:                _local_proc = proc
466:                _local_model = name
467:                _write_pid(proc, name)
468:                return True
469:            time.sleep(1)
470:        logger.warning("llama-server %s startup timed out", name)
471:        proc.kill()
472:        return False
473:    except Exception as e:
474:        logger.warning("llama-server %s launch failed: %s", name, e)
475:        return False
476:
477:
478:def _ensure_model(name):
479:    """Ensure the requested model is loaded. Swap if needed. Returns True if ready."""
480:    # Check if correct model is already running (possibly started by another process)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Wrapper to call Qwen3-8B trading model.
2:
3:Prefers persistent llama-server (HTTP), swapping models as needed.
4:Falls back to subprocess if server unavailable.
5:Uses GPU lock to coordinate with Ministral.
6:
7:Supports batch mode: multiple tickers processed in one model-load cycle.
8:"""
9:
10:import json
11:import logging
12:import platform
13:import subprocess
14:import time
15:from pathlib import Path
16:
17:from portfolio.gpu_gate import gpu_gate
18:from portfolio.llama_server import query_llama_server
19:from portfolio.subprocess_utils import kill_orphaned_llama, run_safe
20:
21:logger = logging.getLogger("portfolio.qwen3_signal")
22:
23:# Batch queue — accumulates contexts, flushed when get_qwen3_batch() is called
24:_batch_queue: list[dict] = []
25:_batch_results: dict[str, dict] = {}  # ticker -> result, populated by flush
26:
27:
28:def _extract_json_from_stdout(stdout):
29:    """Extract JSON (object or array) from subprocess stdout."""
30:    if not stdout:
31:        return None
32:    text = stdout.strip()
33:    if not text:
34:        return None
35:    # Try parsing as-is (could be array for batch mode)
36:    if text.startswith("[") or text.startswith("{"):
37:        try:
38:            return json.loads(text)
39:        except json.JSONDecodeError:
40:            pass
41:    # Find first [ or { and parse from there
42:    for start_char in ("[", "{"):
43:        idx = text.find(start_char)
44:        if idx >= 0:
45:            try:
46:                return json.loads(text[idx:])
47:            except json.JSONDecodeError:
48:                pass
49:    # Last resort: scan lines in reverse
50:    for line in reversed(text.splitlines()):
51:        line = line.strip()
52:        if line.startswith(("{", "[")):
53:            try:
54:                return json.loads(line)
55:            except json.JSONDecodeError:
56:                continue
57:    return None
58:
59:
60:def _call_qwen3(context):
61:    """Call Qwen3-8B, preferring persistent llama-server, with subprocess fallback."""
62:    from portfolio.qwen3_trader import _build_prompt, _parse_response
63:    prompt = _build_prompt(context)
64:
65:    text = query_llama_server("qwen3", prompt, n_predict=1024, temperature=0.6,
66:                              top_p=0.95, stop=["<|endoftext|>", "<|im_end|>"])
67:    if text is not None:
68:        decision, reasoning, confidence = _parse_response(text)
69:        result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
70:        if confidence is not None:
71:            result["confidence"] = confidence
72:        return result
73:
74:    # 2026-05-11 (plex-vram-coord): query_llama_server returning None can mean
75:    # the server died OR the swap was aborted because Plex is transcoding. In
76:    # the latter case, the subprocess fallback below would cold-start an 8B
77:    # model with -ngl 99 — exactly the VRAM allocation that crashes Plex.
78:    #
79:    # If unsafe, signal abstention via the existing "model": "skipped"
80:    # sentinel (matches the GPU-busy convention in ministral_signal.py:110)
81:    # so the vote isn't recorded as a real Qwen3 prediction and the operator
82:    # can grep the warning in logs to know finance loop is being throttled
83:    # by Plex. WARNING level — this is an externally-caused signal loss,
84:    # not normal flow.
85:    from portfolio.llama_server import model_load_safe
86:    if not model_load_safe():
87:        logger.warning("qwen3: abstaining — Plex transcoding and VRAM <7168MB; skipping subprocess fallback")
88:        return {"action": "HOLD", "reasoning": "skipped: Plex transcode active, VRAM tight", "model": "skipped"}
89:
90:    # Fallback: subprocess (cold start)
91:    logger.info("llama-server unavailable for qwen3, falling back to subprocess")
92:    repo_root = Path(__file__).resolve().parent.parent
93:    if platform.system() == "Windows":
94:        python = str(repo_root / ".venv" / "Scripts" / "python.exe")
95:    else:
96:        python = str(repo_root / ".venv" / "bin" / "python")
97:
98:    script = repo_root / "portfolio" / "qwen3_trader.py"
99:    cmd = [python, str(script)]
100:
101:    try:
102:        result = run_safe(
103:            cmd,
104:            input=json.dumps(context),
105:            capture_output=True,
106:            text=True,
107:            timeout=240,
108:        )
109:    except subprocess.TimeoutExpired as e:
110:        stderr_text = e.stderr[-500:] if e.stderr else "(no stderr)"
111:        logger.error("Qwen3 subprocess timed out after 240s — stderr: %s", stderr_text)
112:        raise
113:    if result.returncode != 0:
114:        raise RuntimeError(f"Qwen3 failed: {result.stderr[-500:]}")
115:    payload = _extract_json_from_stdout(result.stdout)
116:    if payload is None:
117:        raise RuntimeError(f"Qwen3 returned invalid JSON: {result.stdout[-500:]}")
118:    return payload
119:
120:
121:def _call_qwen3_batch(contexts):
122:    """Call Qwen3-8B inference subprocess in batch mode.
123:
124:    Loads model once, processes all tickers, returns list of results.
125:    Saves ~5s model load per additional ticker vs single-ticker mode.
126:    """
127:    if not contexts:
128:        return []
129:
130:    repo_root = Path(__file__).resolve().parent.parent
131:    if platform.system() == "Windows":
132:        python = str(repo_root / ".venv" / "Scripts" / "python.exe")
133:    else:
134:        python = str(repo_root / ".venv" / "bin" / "python")
135:
136:    script = repo_root / "portfolio" / "qwen3_trader.py"
137:    cmd = [python, str(script)]
138:
139:    t0 = time.time()
140:    # Send as JSON array to trigger batch mode in qwen3_trader.py
141:    result = run_safe(
142:        cmd,
143:        input=json.dumps(contexts),
144:        capture_output=True,
145:        text=True,
146:        timeout=60 + 30 * len(contexts),  # 60s base + 30s per ticker (extended for deeper reasoning)
147:    )
148:    elapsed = time.time() - t0
149:    logger.info("Qwen3 batch: %d tickers in %.1fs (%.1fs/ticker)",
150:                len(contexts), elapsed, elapsed / len(contexts) if contexts else 0)
151:
152:    if result.returncode != 0:
153:        raise RuntimeError(f"Qwen3 batch failed: {result.stderr[-500:]}")
154:    payload = _extract_json_from_stdout(result.stdout)
155:    if not isinstance(payload, list):
156:        raise RuntimeError(f"Qwen3 batch returned non-list: {type(payload)}")
157:    return payload
158:
159:
160:def get_qwen3_signal(context):
161:    """Get trading signal from Qwen3-8B with GPU gating.
162:
163:    Returns dict with 'action', 'reasoning', 'model' keys.
164:    """
165:    try:
166:        killed = kill_orphaned_llama()
167:        if killed:
168:            logger.warning("Reaped %d orphaned llama process(es)", killed)
169:    except Exception:
170:        # BUG-204: log rather than swallow — if the reaper itself breaks,
171:        # VRAM leaks will be invisible otherwise.
172:        logger.debug("kill_orphaned_llama failed", exc_info=True)
173:    with gpu_gate("qwen3", timeout=300) as acquired:
174:        if not acquired:
175:            logger.warning("GPU gate timeout — returning HOLD")
176:            return {"action": "HOLD", "reasoning": "GPU busy", "model": "Qwen3-8B"}
177:        # Check VRAM before loading model — Qwen3 Q4_K_M needs ~5GB
178:        from portfolio.gpu_gate import get_vram_usage
179:        vram = get_vram_usage()
180:        if vram and vram["free_mb"] < 5500:
181:            logger.warning("Insufficient VRAM for Qwen3: %dMB free (need 5500MB) — returning HOLD", vram["free_mb"])
182:            return {"action": "HOLD", "reasoning": f"low VRAM ({vram['free_mb']}MB free)", "model": "Qwen3-8B"}
183:        return _call_qwen3(context)
184:
185:
186:def get_qwen3_signal_batch(contexts):
187:    """Get trading signals for multiple tickers in one model-load cycle.
188:
189:    Args:
190:        contexts: list of context dicts, each with 'ticker' key.
191:
192:    Returns:
193:        dict mapping ticker -> result dict.
194:    """
195:    if not contexts:
196:        return {}
197:
198:    try:
199:        results = _call_qwen3_batch(contexts)
200:        # Map results back to tickers
201:        mapped = {}
202:        for ctx, res in zip(contexts, results):
203:            ticker = ctx.get("ticker", "UNKNOWN")
204:            mapped[ticker] = res
205:        return mapped
206:    except Exception as e:
207:        logger.warning("Qwen3 batch failed (%s), returning HOLD for all", e)
208:        return {
209:            ctx.get("ticker", "UNKNOWN"): {
210:                "action": "HOLD",
211:                "reasoning": f"batch error: {e}",
212:                "model": "Qwen3-8B",
213:            }
214:            for ctx in contexts
215:        }
216:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 5ms:
1:"""Centralized Claude Code invocation gatekeeper.
2:
3:This module is the ONLY approved way to invoke Claude Code (``claude -p``)
4:from anywhere in the codebase.  All callers — agent_invocation, metals_loop,
5:silver_monitor, claude_fundamental, analyze, bigbet, iskbets, etc. — MUST
6:route through ``invoke_claude()`` defined here.
7:
8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
9:Doing so bypasses the kill switch, rate limiter, and invocation tracking.
10:
11:Usage::
12:
13:    from portfolio.claude_gate import invoke_claude
14:
15:    success, exit_code = invoke_claude(
16:        prompt="Analyze BTC-USD",
17:        caller="silver_monitor",
18:        model="sonnet",
19:        max_turns=20,
20:        timeout=180,
21:    )
22:"""
23:
24:import contextlib
25:import json
26:import logging
27:import os
28:import platform
29:import shutil
30:import signal
31:import subprocess
32:import time
33:from datetime import UTC, datetime
34:from pathlib import Path
35:
36:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
37:
38:logger = logging.getLogger("portfolio.claude_gate")
39:
40:import threading
41:
42:# ---------------------------------------------------------------------------
43:# Master kill switch.  Set to False to block ALL Claude Code invocations
44:# across the entire codebase — no exceptions.
45:# ---------------------------------------------------------------------------
46:CLAUDE_ENABLED = True
47:
48:BASE_DIR = Path(__file__).resolve().parent.parent
49:DATA_DIR = BASE_DIR / "data"
50:CONFIG_FILE = BASE_DIR / "config.json"
51:INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"
52:# 2026-04-13: Append-only journal of failures that EVERY future Claude Code
53:# session must see. Intentionally separate from claude_invocations.jsonl so
54:# hooks and startup scripts can cheaply poll it without parsing routine
55:# invocation noise. Consumed by scripts/check_critical_errors.py, which is
56:# referenced from CLAUDE.md to guarantee surfacing at session start.
57:CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
58:
59:# A-IN-3 (2026-04-11): In-process concurrency lock. Without this, the main
60:# loop's 8-worker ticker pool + the metals loop's fast-tick + signal
61:# subprocesses can all call invoke_claude in parallel. The Claude CLI is
62:# expensive (sonnet ~30s, opus ~3-5min) and the rate limiter is per-day,
63:# not per-second — uncoordinated parallel invocations can:
64:#   1. Race past the kill switch (CLAUDE_ENABLED check is non-atomic)
65:#   2. Spawn 5+ concurrent Claude processes, each holding ~500MB RAM
66:#   3. Confuse the invocation log (timestamps interleave)
67:# Serializing in-process invocations is the simplest robust fix. For
68:# cross-process coordination (multiple Python processes), see the file
69:# lock TODO below.
70:_invoke_lock = threading.Lock()
71:
72:# Rate-limit threshold: warn when daily invocations exceed this count.
73:_DAILY_WARN_THRESHOLD = 50
74:
75:
76:# ---------------------------------------------------------------------------
77:# Internal helpers
78:# ---------------------------------------------------------------------------
79:
80:def _load_config_layer2_enabled() -> bool:
81:    """Check ``config.json -> layer2.enabled``.
82:
83:    Returns True if the key is missing or the file cannot be read (fail-open
84:    for the config check — the module-level CLAUDE_ENABLED flag is the hard
85:    gate).
86:    """
87:    try:
88:        with open(CONFIG_FILE, encoding="utf-8") as f:
89:            cfg = json.load(f)
90:        return cfg.get("layer2", {}).get("enabled", True)
91:    except Exception:
92:        # Config unreadable — don't block on that alone.
93:        return True
94:
95:
96:def _clean_env() -> dict:
97:    """Return a copy of ``os.environ`` with Claude session markers removed.
98:
99:    Prevents the "nested session" error when invoking ``claude -p`` from a
100:    process tree that already has a Claude Code session active.
101:    """
102:    env = os.environ.copy()
103:    env.pop("CLAUDECODE", None)
104:    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
105:    return env
106:
107:
108:# 2026-04-13: Detector for silent auth failures. The `--bare` flag (removed
109:# from agent_invocation.py and multi_agent_layer2.py on 2026-04-13) disables
110:# OAuth/keychain auth and requires ANTHROPIC_API_KEY. Since this user runs
111:# on a Max subscription with no API key, `--bare` caused every Layer 2
112:# invocation between 2026-03-27 and 2026-04-13 to print "Not logged in —
113:# Please run /login" on stdout and exit 0. Nothing surfaced the failure
114:# because exit_code=0 was treated as success across all three invocation
115:# paths. Do not re-add `--bare`. If a new CLI flag or env tweak
116:# re-introduces this class of silent auth error, this detector should
117:# catch it.
118:_AUTH_ERROR_MARKERS = ("Not logged in", "Please run /login", "Invalid API key")
119:
120:# 2026-04-16: feedback-loop fix. CLAUDE.md tells every agent to surface
121:# unresolved critical_errors.jsonl entries verbatim at session start. Those
122:# entries CONTAIN the literal string "Not logged in", so the substring scan
123:# below was treating every echo as a new auth failure, journaling it,
124:# triggering the next agent to re-surface, ad infinitum (today's entries
125:# 13:45:45 + 14:15:01 are both echoes, not real failures).
126:#
127:# The fix narrows the match: a marker only counts when it's the START of a
128:# line, NOT preceded by quote/backtick/paren/blockquote, with no leading
129:# indentation, AND within the first _AUTH_SCAN_LINE_LIMIT lines of output.
130:# Real Claude CLI auth errors print as standalone preamble — they never
131:# appear deep in agent chat. Echoes always appear quoted, indented, in
132:# code blocks, or wrapped in conversational context.
133:_AUTH_SCAN_LINE_LIMIT = 16
134:# Characters that, when they precede the marker, mean "this is quoted, not
135:# CLI output". `'` `"` and `` ` `` cover plain quotes; `(` covers
136:# parentheticals; `>` covers Markdown blockquotes; `[` covers JSON-style
137:# log entries (`["ts": ..., "message": "...Not logged in..."]`); whitespace
138:# at line start covers code-block indentation.
139:_AUTH_MARKER_PREFIX_REJECT = ("'", '"', "`", "(", ">", "[", " ", "\t")
140:
141:
142:def _is_real_auth_marker_line(line: str, marker: str) -> bool:
143:    """Return True if `line` looks like an actual CLI auth-error line.
144:
145:    The CLI prints markers as standalone lines without quoting. Anything
146:    quoted, indented, blockquoted, or embedded in conversational text is
147:    almost certainly an echo of a previously-journaled error.
148:    """
149:    if not line:
150:        return False
151:    # Reject lines that begin with a wrapper character before the marker.
152:    if line[0] in _AUTH_MARKER_PREFIX_REJECT:
153:        return False
154:    # The marker must appear at the very start (after any leading wrapper
155:    # check above has already passed — i.e. no leading whitespace).
156:    # Defense in depth: even if startswith matches, reject if any wrapper
157:    # char appears in the slice BEFORE the marker (handles bullet lists
158:    # like `- Not logged in` that tests pre-empt by checking line[0]).
159:    return line.startswith(marker)
160:
161:
162:def record_critical_error(
163:    category: str,
164:    caller: str,
165:    message: str,
166:    context: dict | None = None,
167:) -> bool:
168:    """Append a critical error to ``data/critical_errors.jsonl``.
169:
170:    The journal is the single source of truth consulted by
171:    ``scripts/check_critical_errors.py`` at Claude session start (via
172:    CLAUDE.md). Writing here guarantees the failure is visible to every
173:    future Claude session until it's resolved with a follow-up entry.
174:
175:    Never raises — logging failures here must not cascade into the caller.
176:
177:    Returns ``True`` when the append landed, ``False`` when it failed.
178:    The boolean lets dedup-aware callers (e.g. loop_contract's
179:    ``_dispatch_critical_errors_for_degradation``) avoid claiming a
180:    dedup slot for a row that never made it to disk — otherwise a
181:    transient IO problem would silence 6+ h of unrecorded incidents
182:    (Codex P2 2026-04-28). Callers that don't need the signal can
183:    safely ignore the return.
184:    """
185:    try:
186:        entry = {
187:            "ts": datetime.now(UTC).isoformat(),
188:            "level": "critical",
189:            "category": category,
190:            "caller": caller,
191:            "resolution": None,
192:            "message": message,
193:            "context": context or {},
194:        }
195:        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
196:        return True
197:    except Exception as e:
198:        logger.error("Failed to write critical_errors.jsonl: %s", e)
199:        return False
200:
201:
202:def detect_auth_failure(output: str, caller: str, context: dict | None = None) -> bool:
203:    """Scan subprocess output for claude-CLI auth errors and escalate.
204:
205:    Returns True if an auth failure pattern is detected. On match, logs at
206:    CRITICAL level AND records the failure to ``critical_errors.jsonl`` so
207:    future Claude sessions see it via the CLAUDE.md startup check. Callers
208:    should downgrade ``success`` to False and mark the invocation status as
209:    ``auth_error`` so the failure also shows up in the invocation log.
210:
211:    Deliberately logger.critical rather than an exception — the finance
212:    loop runs 24/7 and raising here would tear down a tick. The
213:    critical-level log + critical_errors.jsonl entry + invocation-log
214:    status="auth_error" together make the failure impossible to miss.
215:    """
216:    if not output:
217:        return False
218:
219:    # Scan only the top of the output. Real CLI auth errors print as
220:    # preamble before any agent turn output; echoes always appear later
221:    # in conversational chat. See _AUTH_SCAN_LINE_LIMIT comment above
222:    # for the full feedback-loop rationale (BUG-ECHO 2026-04-16).
223:    candidate_lines = output.splitlines()[:_AUTH_SCAN_LINE_LIMIT]
224:    in_fenced_code_block = False
225:    for line in candidate_lines:
226:        # Track Markdown fenced code blocks (```). Lines inside the block
227:        # are quoted content even if they don't have leading whitespace.
228:        if line.startswith("```"):
229:            in_fenced_code_block = not in_fenced_code_block
230:            continue
231:        if in_fenced_code_block:
232:            continue
233:        for marker in _AUTH_ERROR_MARKERS:
234:            if not _is_real_auth_marker_line(line, marker):
235:                continue
236:            logger.critical(
237:                "[AUTH_FAILURE] caller=%s — claude CLI printed %r. "
238:                "OAuth session not being read. Likely causes: "
239:                "--bare flag re-added, ANTHROPIC_API_KEY set to an invalid "
240:                "value, or ~/.claude/.credentials.json expired/missing. "
241:                "Run `claude` interactively to re-login.",
242:                caller, marker,
243:            )
244:            record_critical_error(
245:                category="auth_failure",
246:                caller=caller,
247:                message=(
248:                    f"claude CLI subprocess printed {marker!r} — OAuth session "
249:                    f"not being read. Check for --bare flag, invalid "
250:                    f"ANTHROPIC_API_KEY, or expired ~/.claude/.credentials.json."
251:                ),
252:                context={**(context or {}), "marker": marker},
253:            )
254:            return True
255:    return False
256:
257:
258:def _find_claude_cmd() -> str | None:
259:    """Locate the ``claude`` CLI executable on PATH."""
260:    return shutil.which("claude")
261:
262:
263:def _log_invocation(entry: dict) -> None:
264:    """Append an invocation record to the JSONL log."""
265:    try:
266:        atomic_append_jsonl(INVOCATIONS_LOG, entry)
267:    except Exception as e:
268:        logger.warning("Failed to write invocation log: %s", e)
269:
270:
271:def _count_today_invocations() -> int:
272:    """Count invocation records from today (UTC)."""
273:    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
274:    count = 0
275:    for entry in load_jsonl(INVOCATIONS_LOG):
276:        ts = entry.get("timestamp", "")
277:        if ts.startswith(today_str):
278:            count += 1
279:    return count
280:
281:
282:# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
283:# CPython's run() does kill the *direct* child on TimeoutExpired, but the
284:# Claude CLI is a Node.js process that spawns its own helpers (MCP servers,
285:# the actual claude API client process, etc.). Killing the direct child
286:# leaves all of its descendants running as zombies on Windows. Over a long
287:# session this leaks file handles, sockets, and (worst) GPU VRAM held by
288:# any local-LLM helpers Claude may have spawned.
289:#
290:# Fix: explicitly Popen with a new process group/session so we can kill the
291:# entire tree, not just the direct child. On Windows we use taskkill /T /F
292:# (kills the whole tree by PID); on Unix we use os.killpg(SIGKILL) on the
293:# process group started via start_new_session=True.
294:def _popen_kwargs_for_tree_kill() -> dict:
295:    """Return Popen kwargs that allow tree-killing the spawned process."""
296:    if platform.system() == "Windows":
297:        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
298:    return {"start_new_session": True}
299:
300:
301:def _kill_process_tree(proc: subprocess.Popen, *, label: str = "claude") -> None:
302:    """Kill a Popen process and all of its descendants. Best-effort:
303:    falls back to proc.kill() if the platform-specific path fails.
304:    Always returns; never raises."""
305:    if proc.poll() is not None:
306:        return  # already exited
307:    pid = proc.pid
308:    try:
309:        if platform.system() == "Windows":
310:            # taskkill /T = terminate this PID and all child processes,
311:            # /F = force, /PID = the parent PID. Capture stderr to keep
312:            # logs clean if the process already exited between poll() and here.
313:            res = subprocess.run(
314:                ["taskkill", "/T", "/F", "/PID", str(pid)],
315:                capture_output=True, timeout=5,
316:            )
317:            if res.returncode not in (0, 128):  # 128 = "process not found"
318:                logger.warning(
319:                    "%s tree kill via taskkill returned %d (stderr=%r) — "
320:                    "falling back to proc.kill()",
321:                    label, res.returncode, res.stderr.decode("utf-8", "replace")[:200],
322:                )
323:                proc.kill()
324:        else:
325:            try:
326:                pgid = os.getpgid(pid)
327:                os.killpg(pgid, signal.SIGKILL)
328:            except (ProcessLookupError, OSError) as e:
329:                logger.warning("%s killpg(%d) failed: %s — falling back to proc.kill()", label, pid, e)
330:                proc.kill()
331:    except Exception as e:
332:        # Last-ditch fallback so a kill failure never propagates.
333:        logger.error(
334:            "%s tree kill encountered unexpected error: %s — proc.kill()",
335:            label, e, exc_info=True,
336:        )
337:        try:
338:            proc.kill()
339:        except Exception as kill_err:  # 2026-04-17: surface orphan risk
340:            logger.error(
341:                "%s proc.kill() also failed after tree-kill error: %s — "
342:                "process pid=%s may be orphaned",
343:                label, kill_err, getattr(proc, "pid", "?"),
344:            )
345:
346:
347:def _run_with_tree_kill(
348:    cmd: list[str],
349:    *,
350:    timeout: float,
351:    env: dict | None,
352:    cwd: str,
353:    label: str,
354:) -> tuple[int, str, str, bool]:
355:    """Run a subprocess with proper timeout + tree-kill cleanup.
356:
357:    Returns:
358:        (returncode, stdout, stderr, timed_out)
359:
360:    On timeout, kills the entire process tree (not just the direct child)
361:    and waits up to 5s for the tree to actually exit before returning.
362:    Logs an error if the tree refused to exit.
363:    """
364:    proc = subprocess.Popen(
365:        cmd,
366:        stdout=subprocess.PIPE,
367:        stderr=subprocess.PIPE,
368:        stdin=subprocess.DEVNULL,
369:        text=True,
370:        env=env,
371:        cwd=cwd,
372:        **_popen_kwargs_for_tree_kill(),
373:    )
374:    try:
375:        stdout, stderr = proc.communicate(timeout=timeout)
376:        return proc.returncode, stdout or "", stderr or "", False
377:    except subprocess.TimeoutExpired:
378:        logger.warning("%s timed out after %ds — killing process tree (pid=%d)",
379:                       label, timeout, proc.pid)
380:        _kill_process_tree(proc, label=label)
381:        # Drain pipes after kill so the OS can release them.
382:        try:
383:            stdout, stderr = proc.communicate(timeout=5)
384:        except subprocess.TimeoutExpired:
385:            logger.error("%s process tree did not exit within 5s of kill — possible zombie", label)
386:            with contextlib.suppress(Exception):
387:                proc.kill()
388:            stdout, stderr = "", ""
389:        return -1, stdout or "", stderr or "", True
390:
391:
392:# ---------------------------------------------------------------------------
393:# Public API
394:# ---------------------------------------------------------------------------
395:
396:def invoke_claude(
397:    prompt: str,
398:    caller: str,
399:    model: str = "sonnet",
400:    max_turns: int = 20,
401:    allowed_tools: str = "Read,Edit,Bash,Write",
402:    timeout: int = 180,
403:    cwd: str | None = None,
404:) -> tuple[bool, int]:
405:    """Invoke Claude Code via ``claude -p`` and wait for completion.
406:
407:    Args:
408:        prompt: The prompt text to send.
409:        caller: Identifier of the calling module (e.g. ``"silver_monitor"``).
410:        model: Claude model to use (``"sonnet"``, ``"haiku"``, ``"opus"``).
411:        max_turns: Maximum agentic turns.
412:        allowed_tools: Comma-separated tool names for ``--allowedTools``.
413:        timeout: Subprocess timeout in seconds.
414:        cwd: Working directory for the subprocess.  Defaults to the repo root.
415:
416:    Returns:
417:        ``(success, exit_code)`` where *success* is True when exit_code == 0.
418:        If the invocation is blocked, returns ``(False, -1)``.
419:    """
420:    now_iso = datetime.now(UTC).isoformat()
421:    working_dir = cwd or str(BASE_DIR)
422:
423:    # --- Gate 1: module-level kill switch ---
424:    if not CLAUDE_ENABLED:
425:        logger.info("Claude invocation BLOCKED (CLAUDE_ENABLED=False) caller=%s", caller)
426:        _log_invocation({
427:            "timestamp": now_iso,
428:            "caller": caller,
429:            "status": "blocked",
430:            "reason": "CLAUDE_ENABLED=False",
431:            "model": model,
432:            "max_turns": max_turns,
433:            "duration_seconds": 0,
434:            "exit_code": -1,
435:        })
436:        return False, -1
437:
438:    # --- Gate 2: config.json layer2.enabled ---
439:    if not _load_config_layer2_enabled():
440:        logger.info("Claude invocation BLOCKED (config layer2.enabled=false) caller=%s", caller)
441:        _log_invocation({
442:            "timestamp": now_iso,
443:            "caller": caller,
444:            "status": "blocked",
445:            "reason": "config.layer2.enabled=false",
446:            "model": model,
447:            "max_turns": max_turns,
448:            "duration_seconds": 0,
449:            "exit_code": -1,
450:        })
451:        return False, -1
452:
453:    # --- Rate-limit warning ---
454:    today_count = _count_today_invocations()
455:    if today_count >= _DAILY_WARN_THRESHOLD:
456:        logger.warning(
457:            "Daily invocation count (%d) exceeds threshold (%d) — caller=%s",
458:            today_count, _DAILY_WARN_THRESHOLD, caller,
459:        )
460:
461:    # --- Locate claude CLI ---
462:    claude_cmd = _find_claude_cmd()
463:    if not claude_cmd:
464:        logger.error("claude CLI not found on PATH — caller=%s", caller)
465:        _log_invocation({
466:            "timestamp": now_iso,
467:            "caller": caller,
468:            "status": "error",
469:            "reason": "claude not on PATH",
470:            "model": model,
471:            "max_turns": max_turns,
472:            "duration_seconds": 0,
473:            "exit_code": -1,
474:        })
475:        return False, -1
476:
477:    # --- Build command ---
478:    cmd = [
479:        claude_cmd, "-p", prompt,
480:        "--allowedTools", allowed_tools,
481:        "--max-turns", str(max_turns),
482:        "--model", model,
483:        "--output-format", "text",
484:    ]
485:
486:    # --- Execute ---
487:    t0 = time.time()
488:    exit_code = -1
489:    status = "error"
490:
491:    try:
492:        # A-IN-3: serialize all in-process Claude invocations so the
493:        # 8-worker ticker pool / metals fast-tick / signal subprocesses
494:        # don't spawn 5 concurrent expensive Claude processes.
495:        # A-IN-2: tree-killing helper for grandchild cleanup on timeout.
496:        with _invoke_lock:
497:            rc, _stdout, _stderr, timed_out = _run_with_tree_kill(
498:                cmd,
499:                timeout=timeout,
500:                env=_clean_env(),
501:                cwd=working_dir,
502:                label=f"claude({caller})",
503:            )
504:        if timed_out:
505:            status = "timeout"
506:        else:
507:            exit_code = rc
508:            status = "invoked" if exit_code == 0 else "error"
509:            # 2026-04-13: Silent-failure detector. claude CLI can exit 0 while
510:            # printing "Not logged in" when OAuth/keychain auth can't be read
511:            # (e.g. --bare flag, missing ANTHROPIC_API_KEY). Override status
512:            # so the failure surfaces instead of being lost to exit_code=0.
513:            # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan stdout
514:            # and stderr SEPARATELY rather than concatenating without a
515:            # newline. Concat-without-newline could merge the marker into
516:            # the last stdout line ("...stdoutNot logged in"), defeating
517:            # the start-of-line check shipped today. Scanning each stream
518:            # independently preserves both streams' line-1 position.
519:            stdout_hit = detect_auth_failure(
520:                _stdout or "", caller,
521:                context={"model": model, "max_turns": max_turns, "exit_code": exit_code},
522:            )
523:            stderr_hit = detect_auth_failure(
524:                _stderr or "", caller,
525:                context={"model": model, "max_turns": max_turns, "exit_code": exit_code},
526:            ) if not stdout_hit else False
527:            if stdout_hit or stderr_hit:
528:                status = "auth_error"
529:                exit_code = exit_code or 1
530:    except Exception as e:
531:        status = "error"
532:        logger.error("Claude invocation failed — caller=%s: %s", caller, e)
533:
534:    duration = round(time.time() - t0, 2)
535:
536:    _log_invocation({
537:        "timestamp": now_iso,
538:        "caller": caller,
539:        "status": status,
540:        "model": model,
541:        "max_turns": max_turns,
542:        "duration_seconds": duration,
543:        "exit_code": exit_code,
544:    })
545:
546:    logger.info(
547:        "Claude invocation: caller=%s model=%s status=%s exit=%d duration=%.1fs",
548:        caller, model, status, exit_code, duration,
549:    )
550:
551:    return status == "invoked", exit_code
552:
553:
554:def invoke_claude_text(
555:    prompt: str,
556:    caller: str,
557:    model: str = "sonnet",
558:    timeout: int = 60,
559:) -> tuple[str, bool, int]:
560:    """Invoke Claude CLI for text-only Q&A (no tools, single turn).
561:
562:    Unlike ``invoke_claude()``, this captures stdout and returns the text
563:    response.  Used by signals that need Claude's analysis as structured
564:    text (e.g., claude_fundamental).
565:
566:    Returns:
567:        ``(text, success, exit_code)``
568:    """
569:    now_iso = datetime.now(UTC).isoformat()
570:
571:    if not CLAUDE_ENABLED or not _load_config_layer2_enabled():
572:        _log_invocation({
573:            "timestamp": now_iso, "caller": caller, "status": "blocked",
574:            "reason": "disabled", "model": model, "max_turns": 1,
575:            "duration_seconds": 0, "exit_code": -1,
576:        })
577:        return "", False, -1
578:
579:    claude_cmd = _find_claude_cmd()
580:    if not claude_cmd:
581:        logger.error("claude CLI not found — caller=%s", caller)
582:        return "", False, -1
583:
584:    cmd = [
585:        claude_cmd, "-p", prompt,
586:        "--model", model,
587:        "--output-format", "text",
588:        "--max-turns", "1",
589:        "--allowedTools", "",
590:    ]
591:
592:    t0 = time.time()
593:    text = ""
594:    exit_code = -1
595:    status = "error"
596:
597:    try:
598:        # A-IN-3 + A-IN-2: serialized + tree-killing.
599:        with _invoke_lock:
600:            rc, stdout, _stderr, timed_out = _run_with_tree_kill(
601:                cmd,
602:                timeout=timeout,
603:                env=_clean_env(),
604:                cwd=str(BASE_DIR),
605:                label=f"claude_text({caller})",
606:            )
607:        if timed_out:
608:            status = "timeout"
609:        else:
610:            exit_code = rc
611:            text = stdout
612:            status = "invoked" if exit_code == 0 else "error"
613:            # 2026-04-13: Same auth-failure detection as invoke_claude — see
614:            # the comment there for the full context. Need to scan both
615:            # stdout and stderr because the CLI can write "Not logged in"
616:            # to either depending on version.
617:            # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan each
618:            # stream independently so concat-without-newline can't merge
619:            # the marker into the last stdout line. See invoke_claude for
620:            # the full rationale.
621:            stdout_hit = detect_auth_failure(
622:                stdout or "", caller,
623:                context={"model": model, "max_turns": 1, "exit_code": exit_code},
624:            )
625:            stderr_hit = detect_auth_failure(
626:                _stderr or "", caller,
627:                context={"model": model, "max_turns": 1, "exit_code": exit_code},
628:            ) if not stdout_hit else False
629:            if stdout_hit or stderr_hit:
630:                status = "auth_error"
631:                exit_code = exit_code or 1
632:                text = ""  # don't let the error message leak into the caller's "text"
633:    except Exception as e:
634:        logger.error("Claude text invocation failed — caller=%s: %s", caller, e)
635:
636:    duration = round(time.time() - t0, 2)
637:    _log_invocation({
638:        "timestamp": now_iso, "caller": caller, "status": status,
639:        "model": model, "max_turns": 1,
640:        "duration_seconds": duration, "exit_code": exit_code,
641:    })
642:
643:    logger.info(
644:        "Claude text: caller=%s model=%s status=%s exit=%d duration=%.1fs len=%d",
645:        caller, model, status, exit_code, duration, len(text),
646:    )
647:
648:    return text, status == "invoked", exit_code
649:
650:
651:def get_invocation_stats() -> dict:
652:    """Return summary statistics from the invocation log.
653:
654:    Returns:
655:        Dict with keys: ``total_invocations``, ``today_invocations``,
656:        ``last_invocation_ts``, ``last_caller``, ``enabled``.
657:    """
658:    entries = load_jsonl(INVOCATIONS_LOG)
659:    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
660:
661:    total = len(entries)
662:    today_count = 0
663:    last_ts = None
664:    last_caller = None
665:
666:    for entry in entries:
667:        ts = entry.get("timestamp", "")
668:        if ts.startswith(today_str):
669:            today_count += 1
670:        # Track the latest entry by position (JSONL is append-only,
671:        # so the last entry in the list is the most recent).
672:        last_ts = ts or last_ts
673:        last_caller = entry.get("caller") or last_caller
674:
675:    # Combine both gates for the overall enabled status.
676:    enabled = CLAUDE_ENABLED and _load_config_layer2_enabled()
677:
678:    return {
679:        "total_invocations": total,
680:        "today_invocations": today_count,
681:        "last_invocation_ts": last_ts,
682:        "last_caller": last_caller,
683:        "enabled": enabled,
684:    }
685:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/autonomous.py:18:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/autonomous.py:28:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/autonomous.py:100:    prev_entries = load_jsonl(JOURNAL_FILE, limit=5)
portfolio/autonomous.py:151:    atomic_append_jsonl(JOURNAL_FILE, journal_entry)
portfolio/autonomous.py:166:    atomic_append_jsonl(DECISIONS_FILE, decision_log)
portfolio/autonomous.py:170:        msg = _build_telegram(
portfolio/autonomous.py:180:            logger.exception("Autonomous telegram send failed")
portfolio/autonomous.py:484:def _build_telegram(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:492:        return _build_telegram_mode_b(
portfolio/autonomous.py:498:    return _build_telegram_mode_a(
portfolio/autonomous.py:505:def _build_telegram_mode_a(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:658:def _build_telegram_mode_b(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:825:        from portfolio.file_utils import atomic_write_json
portfolio/autonomous.py:826:        atomic_write_json(THROTTLE_FILE, data)
portfolio/bigbet.py:8:import subprocess
portfolio/bigbet.py:15:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
portfolio/bigbet.py:44:    atomic_write_json(STATE_FILE, state)
portfolio/bigbet.py:169:        # P2 (2026-04-17): PF_HEADLESS_AGENT=1 so the Claude subprocess skips
portfolio/bigbet.py:175:        result = subprocess.run(
portfolio/bigbet.py:179:            timeout=30,
portfolio/bigbet.py:186:        # bypasses claude_gate's invoke_claude wrapper, so a "Not logged in"
portfolio/bigbet.py:200:    except subprocess.TimeoutExpired:
portfolio/bigbet.py:202:        logger.warning("BIG BET L2: timeout after %.1fs", elapsed)
portfolio/bigbet.py:212:        atomic_append_jsonl(EVAL_LOG_FILE, {
portfolio/bigbet.py:456:                _send_telegram(msg, config)
portfolio/bigbet.py:458:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:481:                _send_telegram(msg, config)
portfolio/bigbet.py:483:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:504:                _send_telegram(msg, config)
portfolio/bigbet.py:506:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:599:                _send_telegram(msg, config)
portfolio/bigbet.py:601:                logger.warning("Big Bet telegram failed: %s", e)
portfolio/bigbet.py:618:def _send_telegram(msg, config):
portfolio/analyze.py:10:import subprocess
portfolio/analyze.py:15:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/analyze.py:18:from portfolio.telegram_notifications import send_telegram as _shared_send_telegram
portfolio/analyze.py:24:    """Return env dict without CLAUDECODE to avoid nested-session errors.
portfolio/analyze.py:26:    P2 (2026-04-17): sets PF_HEADLESS_AGENT=1 so the Claude subprocess
portfolio/analyze.py:31:    errors — it just doesn't block the subprocess path on a fake-user
portfolio/analyze.py:35:    env.pop("CLAUDECODE", None)
portfolio/analyze.py:64:    all_entries = load_jsonl(JOURNAL_FILE)
portfolio/analyze.py:233:        atomic_append_jsonl(ANALYSIS_LOG_FILE, {
portfolio/analyze.py:243:def _send_telegram(msg, config):
portfolio/analyze.py:244:    _shared_send_telegram(msg, config)
portfolio/analyze.py:282:        result = subprocess.run(
portfolio/analyze.py:286:            timeout=120,
portfolio/analyze.py:288:            stdin=subprocess.DEVNULL,
portfolio/analyze.py:318:            _send_telegram(tg_msg, config)
portfolio/analyze.py:323:    except subprocess.TimeoutExpired:
portfolio/analyze.py:405:        atomic_append_jsonl(WATCH_LOG_FILE, event)
portfolio/analyze.py:639:        _send_telegram(
portfolio/analyze.py:653:                print(f"  [{ts_str}] No agent_summary.json — waiting...")
portfolio/analyze.py:654:                time.sleep(interval)
portfolio/analyze.py:686:                        _send_telegram(msg, config)
portfolio/analyze.py:702:                        _send_telegram(msg, config)
portfolio/analyze.py:718:                        _send_telegram(msg, config)
portfolio/analyze.py:746:                    result = subprocess.run(
portfolio/analyze.py:750:                        timeout=60,
portfolio/analyze.py:752:                        stdin=subprocess.DEVNULL,
portfolio/analyze.py:787:                                    _send_telegram(tg_msg, config)
portfolio/analyze.py:807:                except subprocess.TimeoutExpired:
portfolio/analyze.py:831:            time.sleep(interval)
portfolio/analyze.py:844:            _send_telegram(
portfolio/focus_analysis.py:16:from portfolio.file_utils import load_json, load_jsonl
portfolio/focus_analysis.py:86:    entries = load_jsonl(JOURNAL_FILE, limit=400)
portfolio/reflection.py:19:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/reflection.py:93:    entries = load_jsonl(JOURNAL_FILE, limit=100)
portfolio/reflection.py:154:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio/reflection.py:204:    atomic_append_jsonl(REFLECTIONS_FILE, reflection)
portfolio/reflection.py:228:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio/regime_alerts.py:11:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/regime_alerts.py:39:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:97:    atomic_append_jsonl(REGIME_HISTORY_FILE, entry)
portfolio/regime_alerts.py:111:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:144:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio/regime_alerts.py:152:    Also logs the message to telegram_messages.jsonl.
portfolio/reporting.py:11:from portfolio.file_utils import atomic_write_json
portfolio/reporting.py:808:    atomic_write_json(AGENT_SUMMARY_FILE, summary)
portfolio/reporting.py:840:        atomic_write_json(SIGNAL_STATE_SINCE_FILE, payload)
portfolio/reporting.py:1018:    atomic_write_json(COMPACT_SUMMARY_FILE, compact)
portfolio/reporting.py:1201:    atomic_write_json(TIER1_FILE, t1)
portfolio/reporting.py:1314:    atomic_write_json(TIER2_FILE, t2)
portfolio/digest.py:14:from portfolio.file_utils import atomic_write_json as _atomic_write_json
portfolio/digest.py:18:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/digest.py:52:    _atomic_write_json(_DIGEST_STATE_FILE, state)
portfolio/digest.py:59:    from portfolio.file_utils import load_jsonl_tail
portfolio/digest.py:66:    entries = load_jsonl_tail(INVOCATIONS_FILE, max_entries=500)
portfolio/digest.py:100:    journal = load_jsonl_tail(JOURNAL_FILE, max_entries=500)
portfolio/digest.py:122:    signal_entries = load_jsonl_tail(SIGNAL_LOG_FILE, max_entries=500)
portfolio/digest.py:239:        heartbeat_age = health.get("heartbeat_age_seconds", 0)
portfolio/digest.py:240:        if heartbeat_age < 300:
portfolio/digest.py:242:        elif heartbeat_age < 3600:
portfolio/digest.py:243:            uptime_str = f"{heartbeat_age / 60:.0f}m stale"
portfolio/digest.py:245:            uptime_str = f"{heartbeat_age / 3600:.1f}h stale"
portfolio/daily_digest.py:19:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/daily_digest.py:42:    from portfolio.file_utils import atomic_write_json
portfolio/daily_digest.py:47:    atomic_write_json(_DAILY_DIGEST_STATE_FILE, state)
portfolio/weekly_digest.py:16:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
portfolio/weekly_digest.py:27:def _load_jsonl(path, since=None):
portfolio/weekly_digest.py:29:    entries = load_jsonl(path)
portfolio/weekly_digest.py:154:    signal_entries = _load_jsonl(SIGNAL_LOG, since=week_ago)
portfolio/weekly_digest.py:168:    journal_entries = _load_jsonl(JOURNAL_FILE, since=week_ago)
portfolio/weekly_digest.py:272:    token = config.get("telegram", {}).get("token")
portfolio/weekly_digest.py:273:    chat_id = config.get("telegram", {}).get("chat_id")
portfolio/weekly_digest.py:280:    log_file = DATA_DIR / "telegram_messages.jsonl"
portfolio/weekly_digest.py:286:    atomic_append_jsonl(log_file, entry)
portfolio/weekly_digest.py:290:        from portfolio.telegram_notifications import send_telegram
portfolio/weekly_digest.py:291:        result = send_telegram(msg, config)
portfolio/decision_outcome_tracker.py:14:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
portfolio/decision_outcome_tracker.py:29:    decisions = load_jsonl(DECISIONS_FILE)
portfolio/decision_outcome_tracker.py:35:    existing_outcomes = load_jsonl(OUTCOMES_FILE)
portfolio/decision_outcome_tracker.py:99:                atomic_append_jsonl(OUTCOMES_FILE, outcome)
portfolio/cumulative_tracker.py:13:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
portfolio/cumulative_tracker.py:52:    atomic_append_jsonl(SNAPSHOTS_FILE, entry)
portfolio/cumulative_tracker.py:105:        snapshots = load_jsonl(SNAPSHOTS_FILE)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
130:    probability = None
131:    reasoning = ""
132:
133:    for line in output.strip().splitlines():
134:        line = line.strip()
135:        upper = line.upper()
136:        if upper.startswith("PROBABILITY:"):
137:            val = line.split(":", 1)[1].strip()
138:            # Extract number from "X/10" or just "X"
139:            num_str = val.split("/")[0].strip()
140:            try:
141:                probability = int(num_str)
142:                probability = max(1, min(10, probability))
143:            except (ValueError, IndexError):
144:                pass
145:        elif upper.startswith("REASONING:"):
146:            reasoning = line.split(":", 1)[1].strip()
147:
148:    return probability, reasoning
149:
150:
151:def invoke_layer2_eval(ticker, direction, conditions, signals, tf_data, prices_usd, config):
152:    """Invoke Claude CLI to evaluate a big bet setup.
153:
154:    Returns (probability: int|None, reasoning: str).
155:    Never blocks — returns (None, "") on any failure.
156:    """
157:    import os
158:
159:    if os.environ.get("NO_TELEGRAM"):
160:        return None, ""
161:
162:    prompt = _build_eval_prompt(ticker, direction, conditions, signals, tf_data, prices_usd)
163:
164:    t0 = time.time()
165:    probability = None
166:    reasoning = ""
167:
168:    try:
169:        # P2 (2026-04-17): PF_HEADLESS_AGENT=1 so the Claude subprocess skips
170:        # the "ask user about unresolved critical errors" step in CLAUDE.md's
171:        # STARTUP CHECK. This path has no interactive stdin.
172:        import os
173:        bigbet_env = os.environ.copy()
174:        bigbet_env["PF_HEADLESS_AGENT"] = "1"
175:        result = subprocess.run(
176:            ["claude", "-p", prompt, "--max-turns", "1"],
177:            capture_output=True,
178:            text=True,
179:            timeout=30,
180:            env=bigbet_env,
181:        )
182:        elapsed = time.time() - t0
183:        output = result.stdout.strip()
184:
185:        # BUG-200 (2026-04-16): Route through detect_auth_failure. This site
186:        # bypasses claude_gate's invoke_claude wrapper, so a "Not logged in"
187:        # stdout with exit 0 would otherwise be passed to the response parser
188:        # (which returns None) without ever escalating the auth failure to
189:        # critical_errors.jsonl. Escalate first, return safe default if hit.
190:        from portfolio.claude_gate import detect_auth_failure
191:        scan = f"{output}\n{result.stderr or ''}"
192:        if detect_auth_failure(scan, caller="bigbet_layer2", context={"ticker": ticker, "direction": direction}):
193:            logger.warning("BIG BET L2: auth failure detected for %s %s — returning None", ticker, direction)
194:            probability, reasoning = None, ""
195:        elif result.returncode == 0 and output:
196:            probability, reasoning = _parse_eval_response(output)
197:            logger.info("BIG BET L2: %s %s — %s/10 (%.1fs)", ticker, direction, probability, elapsed)
198:        else:
199:            logger.warning("BIG BET L2: claude returned code %s", result.returncode)
200:    except subprocess.TimeoutExpired:
201:        elapsed = time.time() - t0
202:        logger.warning("BIG BET L2: timeout after %.1fs", elapsed)
203:    except FileNotFoundError:
204:        elapsed = time.time() - t0
205:        logger.warning("BIG BET L2: claude not found in PATH")
206:    except Exception as e:
207:        elapsed = time.time() - t0
208:        logger.warning("BIG BET L2: error — %s", e)
209:
210:    # Log evaluation
211:    try:
212:        atomic_append_jsonl(EVAL_LOG_FILE, {
213:            "ts": datetime.now(UTC).isoformat(),
214:            "ticker": ticker,
215:            "direction": direction,
216:            "probability": probability,
217:            "reasoning": reasoning,
218:            "elapsed_s": round(time.time() - t0, 2),
219:        })
220:    except Exception:
221:        logger.warning("Failed to append bigbet gate log", exc_info=True)
222:
223:    return probability, reasoning
224:
225:
226:def _evaluate_conditions(ticker, signals, prices_usd, tf_data):
227:    sig = signals.get(ticker)
228:    if not sig:
229:        return [], [], {}
230:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Manual instrument analysis and position watchdog.
2:
3:Usage:
4:  python main.py --analyze ETH-USD
5:  python main.py --watch BTC-USD:66500 ETH-USD:1920 AMD:150
6:"""
7:
8:import logging
9:import os
10:import subprocess
11:import time
12:from datetime import UTC, datetime
13:from pathlib import Path
14:
15:from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
16:
17:logger = logging.getLogger("portfolio.analyze")
18:from portfolio.telegram_notifications import send_telegram as _shared_send_telegram
19:
20:BASE_DIR = Path(__file__).resolve().parent.parent
21:
22:
23:def _clean_env():
24:    """Return env dict without CLAUDECODE to avoid nested-session errors.
25:
26:    P2 (2026-04-17): sets PF_HEADLESS_AGENT=1 so the Claude subprocess
27:    skips the "ask the user about unresolved critical errors" branch in
28:    CLAUDE.md's STARTUP CHECK. /fin-analyze is invoked both from the
29:    main loop (no stdin) AND interactively by the user via the CLI. In
30:    the interactive case CLAUDE.md's conditional still surfaces the
31:    errors — it just doesn't block the subprocess path on a fake-user
32:    response.
33:    """
34:    env = os.environ.copy()
35:    env.pop("CLAUDECODE", None)
36:    env["PF_HEADLESS_AGENT"] = "1"
37:    return env
38:DATA_DIR = BASE_DIR / "data"
39:AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
40:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
41:PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
42:BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
43:CONFIG_FILE = BASE_DIR / "config.json"
44:ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log.jsonl"
45:WATCH_LOG_FILE = DATA_DIR / "watch_log.jsonl"
46:
47:from portfolio.tickers import (
48:    ALL_TICKERS as KNOWN_TICKERS,
49:)
50:from portfolio.tickers import (
51:    CRYPTO_SYMBOLS as CRYPTO_TICKERS,
52:)
53:
54:# Position watch exit thresholds
55:STOP_PCT = -2.0       # hard stop
56:TARGET_PCT = 2.0      # take profit
57:WARN_PCT = -1.5       # early warning
58:TIME_WARN_MINS = 180  # 3h — consider exit
59:TIME_MAX_MINS = 300   # 5h — hard time exit
60:
61:
62:def _load_journal_for_ticker(ticker, max_entries=5):
63:    """Load last N journal entries that mention this ticker with a non-neutral outlook."""
64:    all_entries = load_jsonl(JOURNAL_FILE)
65:    entries = []
66:    for entry in all_entries:
67:        tickers = entry.get("tickers", {})
68:        info = tickers.get(ticker, {})
69:        if info.get("outlook", "neutral") != "neutral":
70:            entries.append(entry)
71:    return entries[-max_entries:]
72:
73:
74:def _get_holdings(ticker):
75:    """Get holdings for this ticker from both portfolio strategies."""
76:    holdings = {}
77:    for label, filepath in [("patient", PORTFOLIO_FILE), ("bold", BOLD_FILE)]:
78:        if not filepath.exists():
79:            continue
80:        try:
81:            pf = load_json(filepath)
82:            h = pf.get("holdings", {}).get(ticker, {})
83:            if h.get("shares", 0) > 0:
84:                holdings[label] = h
85:        except Exception as e:
86:            logger.debug("Failed to load portfolio %s: %s", label, e)
87:            continue
88:    return holdings
89:
90:
91:def _build_analysis_prompt(ticker, summary):
92:    """Build the rich analysis prompt for Claude."""
93:    sig = summary.get("signals", {}).get(ticker)
94:    if not sig:
95:        return None
96:
97:    extra = sig.get("extra", {})
98:    votes = extra.get("_votes", {})
99:    price = sig["price_usd"]
100:    regime = sig.get("regime", "unknown")
101:    rsi = sig["rsi"]
102:    macd = sig["macd_hist"]
103:    bb = sig["bb_position"]
104:    atr_pct = sig.get("atr_pct", 0)
105:    vol_raw = extra.get("volume_ratio")
106:    vol = f"{vol_raw:.2f}" if isinstance(vol_raw, (int, float)) else "N/A"
107:    buy_c = extra.get("_buy_count", 0)
108:    sell_c = extra.get("_sell_count", 0)
109:    total = extra.get("_total_applicable", 21)
110:    hold_c = total - buy_c - sell_c
111:    w_conf = sig.get("weighted_confidence", 0)
112:    confluence = sig.get("confluence_score", 0)
113:
114:    # Signal accuracy
115:    acc_data = summary.get("signal_accuracy_1d", {}).get("signals", {})
116:
117:    def _vote_str(name):
118:        vote = votes.get(name, "HOLD")
119:        acc = acc_data.get(name, {})
120:        acc_pct = f"{acc['accuracy']*100:.0f}%" if acc else "N/A"
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
250:
251:    # Normalize short names
252:    short_map = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
253:    ticker = short_map.get(ticker, ticker)
254:
255:    if ticker not in KNOWN_TICKERS:
256:        print(f"Unknown ticker: {ticker}")
257:        print(f"Valid tickers: {', '.join(sorted(KNOWN_TICKERS))}")
258:        return
259:
260:    if not AGENT_SUMMARY_FILE.exists():
261:        print("No agent_summary.json found. Run --report first to generate signal data.")
262:        return
263:
264:    summary = load_json(AGENT_SUMMARY_FILE)
265:    if summary is None:
266:        print("Failed to parse agent_summary.json.")
267:        return
268:
269:    if ticker not in summary.get("signals", {}):
270:        print(f"No signal data for {ticker} in agent_summary.json. Run --report first.")
271:        return
272:
273:    prompt = _build_analysis_prompt(ticker, summary)
274:    if not prompt:
275:        print(f"Failed to build analysis prompt for {ticker}.")
276:        return
277:
278:    print(f"Analyzing {ticker}...\n")
279:
280:    t0 = time.monotonic()
281:    try:
282:        result = subprocess.run(
283:            ["claude", "-p", prompt, "--max-turns", "1"],
284:            capture_output=True,
285:            text=True,
286:            timeout=120,
287:            env=_clean_env(),
288:            stdin=subprocess.DEVNULL,
289:        )
290:        elapsed = time.monotonic() - t0
291:        output = result.stdout.strip()
292:
293:        # BUG-200/201 pattern (2026-04-16): auth-failure check before trusting
294:        # output. Manual-CLI path, but recording to critical_errors still
295:        # helps future Claude sessions notice when auth expires.
296:        from portfolio.claude_gate import detect_auth_failure
297:        scan = f"{output}\n{result.stderr or ''}"
298:        if detect_auth_failure(scan, caller="analyze_cli", context={"ticker": ticker}):
299:            print(f"Claude auth failure while analyzing {ticker}. Run `claude` interactively to re-login.")
300:            return
301:
302:        if result.returncode != 0 or not output:
303:            print(f"Claude returned code {result.returncode}")
304:            if result.stderr:
305:                print(f"stderr: {result.stderr[:500]}")
306:            return
307:
308:        print(output)
309:        print(f"\n({elapsed:.1f}s)")
310:
311:        _log_analysis(ticker, output, elapsed)
312:
313:        # Send to Telegram
314:        try:
315:            config = load_json(CONFIG_FILE)
316:            price = summary["signals"][ticker]["price_usd"]
317:            tg_msg = f"*ANALYSIS: {ticker}* (${price:,.2f})\n\n{output}"
318:            _send_telegram(tg_msg, config)
319:            print("Sent to Telegram.")
320:        except Exception as e:
321:            print(f"Telegram failed: {e}")
322:
323:    except subprocess.TimeoutExpired:
324:        elapsed = time.monotonic() - t0
325:        print(f"Claude timed out after {elapsed:.0f}s")
326:    except FileNotFoundError:
327:        print("claude not found in PATH")
328:    except Exception as e:
329:        print(f"Error: {e}")
330:
331:
332:# ---------------------------------------------------------------------------
333:# Position watchdog (--watch) — Claude-powered scalp exit monitor
334:# ---------------------------------------------------------------------------
335:
336:SHORT_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "XAU": "XAU-USD", "XAG": "XAG-USD"}
337:
338:# Claude analysis triggers
339:CLAUDE_INTERVAL_MINS = 15       # max time between Claude calls
340:CLAUDE_PRICE_THRESHOLD = 0.5    # % price move triggers re-analysis
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
720:                                "entry": entry_price, "pct": round(pct, 2)})
721:                    resolved.add(ticker)
722:                    continue
723:
724:            # Check if all done
725:            active = [t for t in positions if t not in resolved]
726:            if not active:
727:                print("\nAll positions resolved. Stopping watch.")
728:                break
729:
730:            # --- Phase 2: Claude-powered analysis ---
731:            should_call, reason = _should_call_claude(
732:                last_claude_time, last_claude_prices, current_prices,
733:                last_claude_actions, current_actions,
734:            )
735:
736:            if should_call:
737:                # Only analyze active (non-resolved) positions
738:                active_positions = {t: positions[t] for t in active}
739:                prompt = _build_watch_prompt(active_positions, summary, elapsed_mins)
740:                claude_call_count += 1
741:
742:                print(f"\n  [{ts_str}] Calling Claude (#{claude_call_count}, reason: {reason})...")
743:                t0 = time.time()
744:
745:                try:
746:                    result = subprocess.run(
747:                        ["claude", "-p", prompt, "--max-turns", "1"],
748:                        capture_output=True,
749:                        text=True,
750:                        timeout=60,
751:                        env=_clean_env(),
752:                        stdin=subprocess.DEVNULL,
753:                    )
754:                    c_elapsed = time.time() - t0
755:                    output = result.stdout.strip()
756:
757:                    if result.returncode == 0 and output:
758:                        decisions, overall = _parse_watch_response(output, active)
759:                        print(f"  Claude ({c_elapsed:.1f}s): {overall}")
760:
761:                        for ticker in active:
762:                            dec = decisions.get(ticker, {})
763:                            dec_action = dec.get("action", "HOLD")
764:                            dec_reason = dec.get("reason", "")
765:                            entry_price = positions[ticker]
766:                            price = current_prices.get(ticker, 0)
767:                            pct = ((price - entry_price) / entry_price) * 100 if price else 0
768:                            lev_pct = pct * 5
769:
770:                            state = _get_signal_state(ticker, summary)
771:                            sell_tfs = state["sell_tfs"] if state else "?"
772:                            buy_tfs = state["buy_tfs"] if state else "?"
773:                            action = state["action"] if state else "?"
774:
775:                            if dec_action == "SELL":
776:                                print(f"  ** Claude: SELL {ticker} — {dec_reason} **")
777:                                if config:
778:                                    tg_msg = (
779:                                        f"*WATCH SELL: {ticker}*\n\n"
780:                                        f"Claude recommends EXIT\n"
781:                                        f"P&L: {pct:+.2f}% ({lev_pct:+.1f}% on 5x)\n"
782:                                        f"Price: ${price:,.2f} (entry ${entry_price:,.2f})\n"
783:                                        f"Signals: {action} | TFs: {sell_tfs}S/{buy_tfs}B\n"
784:                                        f"_Reason: {dec_reason}_\n"
785:                                        f"_Hold time: {elapsed_mins:.0f}min_"
786:                                    )
787:                                    _send_telegram(tg_msg, config)
788:                                _log_watch({
789:                                    "ticker": ticker, "type": "claude_sell",
790:                                    "price": price, "entry": entry_price,
791:                                    "pct": round(pct, 2), "reason": dec_reason,
792:                                })
793:                            else:
794:                                print(f"  Claude: HOLD {ticker} — {dec_reason}")
795:
796:                        _log_watch({
797:                            "type": "claude_analysis", "call_num": claude_call_count,
798:                            "reason": reason, "elapsed_s": round(c_elapsed, 2),
799:                            "decisions": {t: decisions.get(t, {}) for t in active},
800:                            "overall": overall,
801:                        })
802:                    else:
803:                        print(f"  Claude returned code {result.returncode} ({c_elapsed:.1f}s)")
804:                        if result.stderr:
805:                            print(f"  stderr: {result.stderr[:200]}")
806:
807:                except subprocess.TimeoutExpired:
808:                    print("  Claude timed out after 60s")
809:                except FileNotFoundError:
810:                    print("  claude not found in PATH — falling back to mechanical only")
811:                except Exception as e:
812:                    print(f"  Claude error: {e}")
813:
814:                last_claude_time = time.time()
815:                last_claude_prices = dict(current_prices)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
360:
361:    return "\n".join(lines)
362:
363:
364:MAX_ACTIVE_BET_SECONDS = 6 * 3600  # 6 hours — auto-expire stale bets
365:
366:
367:def _format_window_closed(ticker, direction, price_at_trigger, current_price, elapsed_minutes):
368:    """Format a 'window closed' notification for an expired active bet."""
369:    if price_at_trigger and price_at_trigger > 0:
370:        pct = ((current_price - price_at_trigger) / price_at_trigger) * 100
371:        price_line = (
372:            f"Entry price: ${price_at_trigger:,.2f} \u2192 Current: "
373:            f"${current_price:,.2f} ({pct:+.1f}%)"
374:        )
375:    else:
376:        price_line = f"Current: ${current_price:,.2f}"
377:
378:    return (
379:        f"\u26aa *BIG BET CLOSED: {direction} {ticker}*\n\n"
380:        f"Setup expired after {elapsed_minutes:.0f}m. Conditions no longer met.\n"
381:        f"{price_line}"
382:    )
383:
384:
385:def _resolve_cooldown_minutes(bigbet_cfg):
386:    """Resolve cooldown in minutes from config, with backwards compatibility.
387:
388:    Checks ``cooldown_minutes`` first (new key, default 10).
389:    Falls back to ``cooldown_hours`` (legacy key) converted to minutes.
390:    """
391:    if "cooldown_minutes" in bigbet_cfg:
392:        return bigbet_cfg["cooldown_minutes"]
393:    if "cooldown_hours" in bigbet_cfg:
394:        return bigbet_cfg["cooldown_hours"] * 60
395:    return 10  # default: 10 minutes
396:
397:
398:def _update_streak(condition_streaks, key, met, now):
399:    """Increment or reset a condition streak. Returns current count.
400:
401:    Streaks auto-expire if the last update was > MAX_STREAK_AGE_S ago.
402:    """
403:    if met:
404:        entry = condition_streaks.get(key)
405:        if entry and (now - entry[1]) <= MAX_STREAK_AGE_S:
406:            count = entry[0] + 1
407:        else:
408:            count = 1  # fresh or stale — start at 1
409:        condition_streaks[key] = [count, now]
410:        return count
411:    # Not met — reset
412:    condition_streaks.pop(key, None)
413:    return 0
414:
415:
416:def check_bigbet(signals, prices_usd, fx_rate, tf_data, config):
417:    bigbet_cfg = config.get("bigbet", {})
418:    min_conditions = bigbet_cfg.get("min_conditions", 3)
419:    min_persistence = bigbet_cfg.get("min_persistence", 2)
420:    min_probability = bigbet_cfg.get("min_probability", 6)
421:    cooldown_minutes = _resolve_cooldown_minutes(bigbet_cfg)
422:    cooldown_seconds = cooldown_minutes * 60
423:
424:    state = _load_state()
425:    cooldowns = state.get("cooldowns", {})
426:    price_history = state.get("price_history", {})
427:    active_bets = state.get("active_bets", {})
428:    condition_streaks = state.get("condition_streaks", {})
429:    now = time.time()
430:    changed = False
431:
432:    # --- Phase 1: Check existing active bets for expiry ---
433:    expired_keys = []
434:    for bet_key, bet_info in list(active_bets.items()):
435:        triggered_at = bet_info.get("triggered_at", 0)
436:        elapsed = now - triggered_at
437:
438:        # Auto-expire after MAX_ACTIVE_BET_SECONDS (6h)
439:        if elapsed > MAX_ACTIVE_BET_SECONDS:
440:            expired_keys.append(bet_key)
441:            # Parse ticker and direction from key
442:            parts = bet_key.rsplit("_", 1)
443:            if len(parts) == 2:
444:                ticker_k, direction_k = parts
445:            else:
446:                continue
447:            current_price = prices_usd.get(ticker_k, 0)
448:            elapsed_min = elapsed / 60
449:            msg = _format_window_closed(
450:                ticker_k, direction_k,
451:                bet_info.get("price_at_trigger", 0),
452:                current_price, elapsed_min,
453:            )
454:            logger.info("BIG BET EXPIRED (6h): %s", bet_key)
455:            try:
456:                _send_telegram(msg, config)
457:            except Exception as e:
458:                logger.warning("Big Bet telegram failed: %s", e)
459:            changed = True
460:            continue
461:
462:        # Re-evaluate conditions to see if setup is still active
463:        parts = bet_key.rsplit("_", 1)
464:        if len(parts) != 2:
465:            expired_keys.append(bet_key)
466:            continue
467:        ticker_k, direction_k = parts
468:
469:        if ticker_k not in signals:
470:            # Ticker no longer in signals — expire
471:            expired_keys.append(bet_key)
472:            current_price = prices_usd.get(ticker_k, 0)
473:            elapsed_min = elapsed / 60
474:            msg = _format_window_closed(
475:                ticker_k, direction_k,
476:                bet_info.get("price_at_trigger", 0),
477:                current_price, elapsed_min,
478:            )
479:            logger.info("BIG BET CLOSED (ticker gone): %s", bet_key)
480:            try:
481:                _send_telegram(msg, config)
482:            except Exception as e:
483:                logger.warning("Big Bet telegram failed: %s", e)
484:            changed = True
485:            continue
486:
487:        bull_conds, bear_conds, _ = _evaluate_conditions(
488:            ticker_k, signals, prices_usd, tf_data
489:        )
490:        relevant_conds = bull_conds if direction_k == "BULL" else bear_conds
491:
492:        if len(relevant_conds) < min_conditions:
493:            # Conditions no longer met — send window closed
494:            expired_keys.append(bet_key)
495:            current_price = prices_usd.get(ticker_k, 0)
496:            elapsed_min = elapsed / 60
497:            msg = _format_window_closed(
498:                ticker_k, direction_k,
499:                bet_info.get("price_at_trigger", 0),
500:                current_price, elapsed_min,
501:            )
502:            logger.info("BIG BET CLOSED (conditions faded): %s after %.0fm", bet_key, elapsed_min)
503:            try:
504:                _send_telegram(msg, config)
505:            except Exception as e:
506:                logger.warning("Big Bet telegram failed: %s", e)
507:            changed = True
508:
509:    for key in expired_keys:
510:        active_bets.pop(key, None)
511:
512:    # --- Phase 2: Evaluate new alerts ---
513:    for ticker in signals:
514:        price = prices_usd.get(ticker, 0)
515:        if price <= 0:
516:            continue
517:
518:        bull_conds, bear_conds, extra_info = _evaluate_conditions(
519:            ticker, signals, prices_usd, tf_data
520:        )
521:
522:        # 4. 24h price change — from stored history
523:        hist = price_history.get(ticker, [])
524:        if hist:
525:            # Find price ~24h ago (closest entry to 86400s ago)
526:            target_time = now - 86400
527:            closest = min(hist, key=lambda h: abs(h["t"] - target_time))
528:            if abs(closest["t"] - target_time) < 7200:  # within 2h tolerance
529:                old_price = closest["p"]
530:                pct_change = ((price - old_price) / old_price) * 100
531:                if pct_change <= -5:
532:                    bull_conds.append(
533:                        f"Price {pct_change:+.1f}% in 24h (${old_price:,.0f}\u2192${price:,.0f})"
534:                    )
535:                if pct_change >= 5:
536:                    bear_conds.append(
537:                        f"Price {pct_change:+.1f}% in 24h (${old_price:,.0f}\u2192${price:,.0f})"
538:                    )
539:
540:        # Update price history — keep last 48h of entries, sample every ~10min
541:        if not hist or (now - hist[-1]["t"]) >= 600:
542:            hist.append({"t": now, "p": price})
543:            # Prune entries older than 48h
544:            cutoff = now - 172800
545:            hist = [h for h in hist if h["t"] >= cutoff]
546:            price_history[ticker] = hist
547:            changed = True
548:
549:        # Check BULL and BEAR alerts with persistence + probability gating
550:        for direction, conds in [("BULL", bull_conds), ("BEAR", bear_conds)]:
551:            cd_key = f"{ticker}_{direction}"
552:            met = len(conds) >= min_conditions
553:            streak = _update_streak(condition_streaks, cd_key, met, now)
554:
555:            if not met:
556:                continue
557:
558:            if streak < min_persistence:
559:                logger.info(
560:                    "Big Bet: %s %s (%d/%d) — persistence %d/%d",
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
560:                    "Big Bet: %s %s (%d/%d) — persistence %d/%d",
561:                    direction, ticker, len(conds), TOTAL_CONDITIONS,
562:                    streak, min_persistence,
563:                )
564:                changed = True
565:                continue
566:
567:            last_alert = cooldowns.get(cd_key, 0)
568:            if now - last_alert <= cooldown_seconds:
569:                remaining = cooldown_seconds - (now - last_alert)
570:                logger.info(
571:                    "Big Bet: %s %s (%d/%d) — cooldown (%.0fm left)",
572:                    direction, ticker, len(conds), TOTAL_CONDITIONS,
573:                    remaining / 60,
574:                )
575:                continue
576:
577:            probability, l2_reasoning = invoke_layer2_eval(
578:                ticker, direction, conds, signals, tf_data, prices_usd, config
579:            )
580:
581:            # Probability gate — require eval to succeed and meet threshold
582:            if probability is None or probability < min_probability:
583:                logger.info(
584:                    "Big Bet: %s %s (%d/%d) — blocked by probability (%s < %d)",
585:                    direction, ticker, len(conds), TOTAL_CONDITIONS,
586:                    probability, min_probability,
587:                )
588:                continue
589:
590:            msg = _format_alert(
591:                ticker, direction, conds, prices_usd, fx_rate, extra_info,
592:                probability=probability, l2_reasoning=l2_reasoning,
593:            )
594:            logger.info(
595:                "BIG BET ALERT: %s %s (%d/%d conditions, %d/10 prob)",
596:                direction, ticker, len(conds), TOTAL_CONDITIONS, probability,
597:            )
598:            try:
599:                _send_telegram(msg, config)
600:            except Exception as e:
601:                logger.warning("Big Bet telegram failed: %s", e)
602:            cooldowns[cd_key] = now
603:            active_bets[cd_key] = {
604:                "triggered_at": now,
605:                "conditions": list(conds),
606:                "price_at_trigger": price,
607:            }
608:            changed = True
609:
610:    if changed:
611:        state["cooldowns"] = cooldowns
612:        state["price_history"] = price_history
613:        state["active_bets"] = active_bets
614:        state["condition_streaks"] = condition_streaks
615:        _save_state(state)
616:
617:
618:def _send_telegram(msg, config):
619:    send_or_store(msg, config, category="bigbet")
620:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
1:"""Telegram alert budgeting — prevents alert fatigue.
2:
3:Priority levels:
4:    1 = Normal (subject to budget)
5:    2 = Important (subject to budget but gets priority in queue)
6:    3 = Emergency (bypasses budget — stop-loss, circuit breaker, crash)
7:"""
8:from __future__ import annotations
9:
10:import logging
11:import time
12:from collections import deque
13:
14:logger = logging.getLogger("portfolio.alert_budget")
15:
16:PRIORITY_EMERGENCY = 3
17:PRIORITY_IMPORTANT = 2
18:PRIORITY_NORMAL = 1
19:
20:
21:class AlertBudget:
22:    """Token-bucket style alert rate limiter with priority bypass."""
23:
24:    def __init__(self, max_per_hour: int = 3, window_seconds: int = 3600):
25:        self.max_per_hour = max_per_hour
26:        self.window_seconds = window_seconds
27:        self._sent_timestamps: deque[float] = deque()
28:        self._buffer: list[str] = []
29:
30:    def _prune_old(self) -> None:
31:        """Remove timestamps outside the current window."""
32:        cutoff = time.time() - self.window_seconds
33:        while self._sent_timestamps and self._sent_timestamps[0] < cutoff:
34:            self._sent_timestamps.popleft()
35:
36:    def should_send(self, message: str, priority: int = PRIORITY_NORMAL) -> bool:
37:        """Check if an alert should be sent or buffered."""
38:        if priority >= PRIORITY_EMERGENCY:
39:            self._sent_timestamps.append(time.time())
40:            return True
41:        self._prune_old()
42:        if len(self._sent_timestamps) < self.max_per_hour:
43:            self._sent_timestamps.append(time.time())
44:            return True
45:        self._buffer.append(message)
46:        return False
47:
48:    def flush_buffer(self) -> list[str]:
49:        """Return and clear buffered messages."""
50:        buffered = self._buffer.copy()
51:        self._buffer.clear()
52:        return buffered
53:
54:    @property
55:    def remaining_budget(self) -> int:
56:        self._prune_old()
57:        return max(0, self.max_per_hour - len(self._sent_timestamps))
58:
59:    @property
60:    def buffer_size(self) -> int:
61:        return len(self._buffer)
62:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Circuit breaker for data source API calls.
2:
3:Prevents repeated calls to failing APIs. States:
4:  CLOSED  — normal operation, requests pass through
5:  OPEN    — API is failing, requests blocked until recovery timeout
6:  HALF_OPEN — testing recovery, one request allowed through
7:"""
8:
9:import enum
10:import logging
11:import threading
12:import time
13:
14:logger = logging.getLogger("portfolio.circuit_breaker")
15:
16:
17:class State(enum.Enum):
18:    CLOSED = "CLOSED"
19:    OPEN = "OPEN"
20:    HALF_OPEN = "HALF_OPEN"
21:
22:
23:class CircuitBreaker:
24:    """Thread-safe circuit breaker for a single data source."""
25:
26:    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
27:                 max_recovery_timeout: int = 300):
28:        self.name = name
29:        self.failure_threshold = failure_threshold
30:        self.recovery_timeout = recovery_timeout
31:        self._base_recovery_timeout = recovery_timeout
32:        self._max_recovery_timeout = max_recovery_timeout
33:        self._state = State.CLOSED
34:        self._failure_count = 0
35:        self._last_failure_time: float | None = None
36:        self._lock = threading.Lock()
37:        self._half_open_probe_sent = False  # BUG-93: Only one request in HALF_OPEN
38:
39:    @property
40:    def state(self) -> State:
41:        return self._state
42:
43:    def record_success(self) -> None:
44:        """Record a successful request. Resets failure count; HALF_OPEN -> CLOSED."""
45:        with self._lock:
46:            if self._state == State.HALF_OPEN:
47:                logger.info("Circuit breaker '%s': HALF_OPEN -> CLOSED (recovery confirmed)", self.name)
48:                self._state = State.CLOSED
49:                self._half_open_probe_sent = False  # BUG-93: Reset probe flag
50:                # BUG-245: Reset backoff on successful recovery
51:                self.recovery_timeout = self._base_recovery_timeout
52:            self._failure_count = 0
53:
54:    def record_failure(self) -> None:
55:        """Record a failed request. Increments count; CLOSED -> OPEN at threshold, HALF_OPEN -> OPEN."""
56:        with self._lock:
57:            self._failure_count += 1
58:            self._last_failure_time = time.monotonic()
59:
60:            if self._state == State.HALF_OPEN:
61:                # BUG-245: Exponential backoff — double timeout on each failed
62:                # recovery probe, capped at max. Reduces retry pressure during
63:                # extended outages (e.g., Binance maintenance windows).
64:                prev_timeout = self.recovery_timeout
65:                self.recovery_timeout = min(
66:                    self.recovery_timeout * 2, self._max_recovery_timeout
67:                )
68:                logger.warning(
69:                    "Circuit breaker '%s': HALF_OPEN -> OPEN (recovery failed, %d failures, "
70:                    "next probe in %ds, was %ds)",
71:                    self.name, self._failure_count, self.recovery_timeout, prev_timeout,
72:                )
73:                self._state = State.OPEN
74:                self._half_open_probe_sent = False  # BUG-93: Reset probe flag
75:            elif self._state == State.CLOSED and self._failure_count >= self.failure_threshold:
76:                logger.warning(
77:                    "Circuit breaker '%s': CLOSED -> OPEN (threshold %d reached)",
78:                    self.name, self.failure_threshold,
79:                )
80:                self._state = State.OPEN
81:
82:    def allow_request(self) -> bool:
83:        """Return True if a request should proceed."""
84:        with self._lock:
85:            if self._state == State.CLOSED:
86:                return True
87:
88:            if self._state == State.OPEN:
89:                if self._last_failure_time is None:
90:                    return False
91:                elapsed = time.monotonic() - self._last_failure_time
92:                if elapsed >= self.recovery_timeout:
93:                    logger.info(
94:                        "Circuit breaker '%s': OPEN -> HALF_OPEN (%.1fs elapsed, testing recovery)",
95:                        self.name, elapsed,
96:                    )
97:                    self._state = State.HALF_OPEN
98:                    self._half_open_probe_sent = True  # BUG-93: This IS the probe
99:                    return True
100:                return False
101:
102:            # BUG-93/BUG-187: HALF_OPEN — the probe request is always sent via
103:            # the OPEN→HALF_OPEN transition above (which sets probe_sent=True and
104:            # returns True). This branch handles the case where a second request
105:            # arrives while still in HALF_OPEN (waiting for success/failure).
106:            return False
107:
108:    def get_status(self) -> dict:
109:        """Return current circuit breaker status."""
110:        with self._lock:
111:            return {
112:                "name": self.name,
113:                "state": self._state.value,
114:                "failure_count": self._failure_count,
115:                "last_failure_time": self._last_failure_time,
116:            }
117:
118:    def reset(self) -> None:
119:        """Force the breaker back to CLOSED with zero failures.
120:
121:        Intended use: operational override (manual recovery) and test
122:        isolation. Production code should NOT call this in normal flow
123:        — let record_success/record_failure drive the state machine.
124:
125:        2026-05-02: added when test_consensus xdist flakes traced back
126:        to module-level breakers tripping during one test and leaking
127:        into the next on the same xdist worker.
128:        """
129:        with self._lock:
130:            self._state = State.CLOSED
131:            self._failure_count = 0
132:            self._last_failure_time = None
133:            self._half_open_probe_sent = False
134:            self.recovery_timeout = self._base_recovery_timeout
135:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
1:"""Cumulative price change tracker — rolling 1d/3d/7d price changes.
2:
3:Logs hourly price snapshots to data/price_snapshots_hourly.jsonl and computes
4:rolling changes so messages can show "XAG +12.4% 7d".
5:"""
6:
7:import json
8:import logging
9:import time
10:from datetime import UTC, datetime, timedelta
11:from pathlib import Path
12:
13:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
14:from portfolio.shared_state import _cached
15:
16:logger = logging.getLogger("portfolio.cumulative_tracker")
17:
18:BASE_DIR = Path(__file__).resolve().parent.parent
19:DATA_DIR = BASE_DIR / "data"
20:SNAPSHOTS_FILE = DATA_DIR / "price_snapshots_hourly.jsonl"
21:
22:# Minimum interval between snapshots (55 minutes)
23:_MIN_SNAPSHOT_INTERVAL_SEC = 55 * 60
24:
25:# Cache TTL for cumulative summary
26:_CUMULATIVE_CACHE_TTL = 300  # 5 min
27:
28:
29:def maybe_log_hourly_snapshot(prices_usd):
30:    """Append a price snapshot if >55 min since last entry.
31:
32:    Args:
33:        prices_usd: dict {ticker: price_usd} for all tracked instruments.
34:
35:    Returns:
36:        True if a snapshot was logged, False if skipped (too recent).
37:    """
38:    if not prices_usd:
39:        return False
40:
41:    # Check last snapshot timestamp
42:    last_ts = _get_last_snapshot_ts()
43:    now = time.time()
44:    if last_ts and (now - last_ts) < _MIN_SNAPSHOT_INTERVAL_SEC:
45:        return False
46:
47:    entry = {
48:        "ts": datetime.now(UTC).isoformat(),
49:        "prices": {k: round(v, 4) for k, v in prices_usd.items() if v},
50:    }
51:
52:    atomic_append_jsonl(SNAPSHOTS_FILE, entry)
53:    logger.debug("Logged hourly price snapshot (%d tickers)", len(entry["prices"]))
54:    return True
55:
56:
57:def _get_last_snapshot_ts():
58:    """Get the timestamp of the most recent snapshot as epoch seconds.
59:
60:    Reads just the last line of the file for efficiency.
61:    """
62:    if not SNAPSHOTS_FILE.exists():
63:        return None
64:
65:    last_line = None
66:    try:
67:        with open(SNAPSHOTS_FILE, "rb") as f:
68:            # Seek to end and read backwards to find last line
69:            f.seek(0, 2)
70:            size = f.tell()
71:            if size == 0:
72:                return None
73:            # Read last 2KB (more than enough for one JSON line)
74:            read_size = min(size, 2048)
75:            f.seek(size - read_size)
76:            chunk = f.read().decode("utf-8", errors="replace")
77:            lines = chunk.strip().split("\n")
78:            last_line = lines[-1].strip()
79:    except (OSError, IndexError):
80:        return None
81:
82:    if not last_line:
83:        return None
84:
85:    try:
86:        entry = json.loads(last_line)
87:        ts = datetime.fromisoformat(entry["ts"])
88:        return ts.timestamp()
89:    except (json.JSONDecodeError, KeyError, ValueError):
90:        return None
91:
92:
93:def compute_rolling_changes(tickers=None, snapshots=None):
94:    """Compute rolling price changes over 1d, 3d, 7d windows.
95:
96:    Args:
97:        tickers: Optional list of tickers to compute for. None = all.
98:        snapshots: Pre-loaded snapshot list (for testing). None = load from file.
99:
100:    Returns:
101:        dict: {ticker: {"change_1d": +2.3, "change_3d": +5.1, "change_7d": +12.4}}
102:        Changes are in percent. None values if insufficient data.
103:    """
104:    if snapshots is None:
105:        snapshots = load_jsonl(SNAPSHOTS_FILE)
106:
107:    if not snapshots:
108:        return {}
109:
110:    now = datetime.now(UTC)
111:    latest = snapshots[-1]
112:    latest_prices = latest.get("prices", {})
113:
114:    windows = {
115:        "change_1d": timedelta(days=1),
116:        "change_3d": timedelta(days=3),
117:        "change_7d": timedelta(days=7),
118:    }
119:
120:    result = {}
121:
122:    for ticker, current_price in latest_prices.items():
123:        if tickers and ticker not in tickers:
124:            continue
125:        if not current_price or current_price <= 0:
126:            continue
127:
128:        changes = {}
129:        for label, delta in windows.items():
130:            target_ts = now - delta
131:            old_price = _find_closest_price(snapshots, ticker, target_ts)
132:            if old_price and old_price > 0:
133:                changes[label] = round(((current_price - old_price) / old_price) * 100, 2)
134:            else:
135:                changes[label] = None
136:
137:        result[ticker] = changes
138:
139:    return result
140:
141:
142:def _find_closest_price(snapshots, ticker, target_ts, max_hours=6):
143:    """Find the price closest to target_ts within max_hours tolerance.
144:
145:    Args:
146:        snapshots: List of snapshot dicts (sorted by time).
147:        ticker: Ticker to look up.
148:        target_ts: Target datetime (UTC).
149:        max_hours: Maximum acceptable distance in hours.
150:
151:    Returns:
152:        float or None: The closest price found, or None if none within range.
153:    """
154:    best_price = None
155:    best_delta = None
156:
157:    for snap in snapshots:
158:        try:
159:            snap_ts = datetime.fromisoformat(snap["ts"])
160:        except (KeyError, ValueError):
161:            continue
162:
163:        price = snap.get("prices", {}).get(ticker)
164:        if price is None:
165:            continue
166:
167:        delta = abs((snap_ts - target_ts).total_seconds()) / 3600
168:        if delta > max_hours:
169:            continue
170:
171:        if best_delta is None or delta < best_delta:
172:            best_price = price
173:            best_delta = delta
174:
175:    return best_price
176:
177:
178:def get_cumulative_summary(tickers=None):
179:    """Main entry point. Returns rolling changes + top movers.
180:
181:    Cached for 5 minutes via shared_state._cached().
182:
183:    Args:
184:        tickers: Optional list of tickers. None = all.
185:
186:    Returns:
187:        dict: {
188:            "ticker_changes": {ticker: {"change_1d": ..., "change_3d": ..., "change_7d": ...}},
189:            "movers": [{"ticker": ..., "change_3d": ..., "change_7d": ...}]
190:        }
191:    """
192:    def _compute():
193:        changes = compute_rolling_changes(tickers=tickers)
194:
195:        # Identify movers: abs(3d) > 5% or abs(7d) > 10%
196:        movers = []
197:        for ticker, c in changes.items():
198:            c3d = c.get("change_3d")
199:            c7d = c.get("change_7d")
200:            if (c3d is not None and abs(c3d) > 5.0) or \
201:               (c7d is not None and abs(c7d) > 10.0):
202:                movers.append({
203:                    "ticker": ticker,
204:                    "change_3d": c3d,
205:                    "change_7d": c7d,
206:                })
207:
208:        # Sort movers by absolute 7d change (or 3d if 7d unavailable)
209:        movers.sort(key=lambda m: abs(m.get("change_7d") or m.get("change_3d") or 0), reverse=True)
210:
211:        return {
212:            "ticker_changes": changes,
213:            "movers": movers,
214:        }
215:
216:    cache_key = f"cumulative_summary_{','.join(tickers) if tickers else 'all'}"
217:    return _cached(cache_key, _CUMULATIVE_CACHE_TTL, _compute)
218:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:import json
2:import logging
3:import re
4:from collections import defaultdict
5:from datetime import UTC, datetime, timedelta
6:from pathlib import Path
7:
8:logger = logging.getLogger("portfolio.journal")
9:
10:from portfolio.file_utils import atomic_write_text, load_json
11:from portfolio.tickers import ALL_TICKERS
12:
13:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
14:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
15:CONTEXT_FILE = DATA_DIR / "layer2_context.md"
16:PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
17:BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
18:
19:TIER_FULL = 2
20:TIER_COMPACT = 4
21:
22:
23:def load_recent(max_entries=10, max_age_hours=8):
24:    if not JOURNAL_FILE.exists():
25:        return []
26:    cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
27:    entries = []
28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
29:        for line in f:
30:            line = line.strip()
31:            if not line:
32:                continue
33:            try:
34:                entry = json.loads(line)
35:                ts = datetime.fromisoformat(entry["ts"])
36:                if ts >= cutoff:
37:                    entries.append(entry)
38:            except (json.JSONDecodeError, KeyError, ValueError):
39:                continue
40:    return entries[-max_entries:]
41:
42:
43:def _is_all_hold(entry):
44:    decisions = entry.get("decisions", {})
45:    for strat in ("patient", "bold"):
46:        d = decisions.get(strat, {})
47:        if d.get("action", "HOLD") != "HOLD":
48:            return False
49:    return True
50:
51:
52:def _non_neutral_tickers(entry):
53:    tickers = entry.get("tickers", {})
54:    return {
55:        k: v for k, v in tickers.items() if v.get("outlook", "neutral") != "neutral"
56:    }
57:
58:
59:def _fmt_time(ts_str):
60:    ts = datetime.fromisoformat(ts_str)
61:    return ts.strftime("%H:%M UTC")
62:
63:
64:def _fmt_time_range(ts_start, ts_end):
65:    t0 = datetime.fromisoformat(ts_start).strftime("%H:%M")
66:    t1 = datetime.fromisoformat(ts_end).strftime("%H:%M UTC")
67:    return f"{t0}–{t1}"
68:
69:
70:def _entry_age_hours(entry, now=None):
71:    if now is None:
72:        now = datetime.now(UTC)
73:    ts = datetime.fromisoformat(entry["ts"])
74:    return (now - ts).total_seconds() / 3600
75:
76:
77:def _append_entry(lines, entry):
78:    ts = _fmt_time(entry["ts"])
79:    trigger = entry.get("trigger", "unknown")
80:    regime = entry.get("regime", "unknown")
81:
82:    lines.append(f"**{ts}** | trigger: {trigger}")
83:
84:    reflection = entry.get("reflection")
85:    if reflection:
86:        lines.append(f"_Reflection: {reflection}_")
87:
88:    lines.append(f"regime: {regime}")
89:
90:    decisions = entry.get("decisions", {})
91:    for strat in ("patient", "bold"):
92:        d = decisions.get(strat, {})
93:        action = d.get("action", "HOLD")
94:        reasoning = d.get("reasoning", "")
95:        lines.append(f"{strat}: {action} — {reasoning}")
96:
97:    for ticker, info in _non_neutral_tickers(entry).items():
98:        outlook = info.get("outlook", "neutral")
99:        thesis = info.get("thesis", "")
100:        levels = info.get("levels", [])
101:        level_str = f" (S:{levels[0]} R:{levels[1]})" if len(levels) == 2 else ""
102:        conviction = info.get("conviction")
103:        conv_str = f" [{int(conviction * 100)}%]" if conviction else ""
104:        lines.append(f"{ticker}: {outlook}{conv_str} — {thesis}{level_str}")
105:
106:        debate = info.get("debate")
107:        if debate and isinstance(debate, dict):
108:            bull = debate.get("bull", "")
109:            bear = debate.get("bear", "")
110:            synthesis = debate.get("synthesis", "")
111:            if bull:
112:                lines.append(f"  Bull: {bull}")
113:            if bear:
114:                lines.append(f"  Bear: {bear}")
115:            if synthesis:
116:                lines.append(f"  Synthesis: {synthesis}")
117:
118:    lines.append("")
119:
120:
121:def _append_entry_compact(lines, entry):
122:    ts = _fmt_time(entry["ts"])
123:    decisions = entry.get("decisions", {})
124:    p_action = decisions.get("patient", {}).get("action", "HOLD")
125:    b_action = decisions.get("bold", {}).get("action", "HOLD")
126:
127:    ticker_parts = []
128:    for ticker, info in _non_neutral_tickers(entry).items():
129:        outlook = info.get("outlook", "neutral")
130:        conviction = info.get("conviction")
131:        conv_str = f"({int(conviction * 100)}%)" if conviction else ""
132:        ticker_parts.append(f"{ticker}={outlook}{conv_str}")
133:
134:    ticker_str = " | " + ", ".join(ticker_parts) if ticker_parts else ""
135:    lines.append(f"**{ts}** | patient: {p_action} / bold: {b_action}{ticker_str}")
136:    lines.append("")
137:
138:
139:def _append_entry_oneline(lines, entry):
140:    ts = _fmt_time(entry["ts"])
141:    regime = entry.get("regime", "unknown")
142:    decisions = entry.get("decisions", {})
143:    p_action = decisions.get("patient", {}).get("action", "HOLD")
144:    b_action = decisions.get("bold", {}).get("action", "HOLD")
145:    lines.append(f"{ts} {regime} P:{p_action}/B:{b_action}")
146:
147:
148:def _build_continuation_chains(entries):
149:    ts_map = {}
150:    for e in entries:
151:        ts_map[e["ts"]] = e
152:
153:    children = defaultdict(list)
154:    for e in entries:
155:        parent_ts = e.get("continues")
156:        if parent_ts and parent_ts in ts_map:
157:            children[parent_ts].append(e["ts"])
158:
159:    roots = set()
160:    for e in entries:
161:        parent_ts = e.get("continues")
162:        if parent_ts and parent_ts in ts_map:
163:            continue
164:        if e["ts"] in children:
165:            roots.add(e["ts"])
166:
167:    chains = []
168:    for root_ts in sorted(roots):
169:        chain = [root_ts]
170:        current = root_ts
171:        while current in children:
172:            next_ts = children[current][0]
173:            chain.append(next_ts)
174:            current = next_ts
175:        if len(chain) >= 2:
176:            chains.append(chain)
177:
178:    return chains, ts_map
179:
180:
181:def _load_portfolio_pnl():
182:    data = {}
183:    for label, filepath in [("patient", PORTFOLIO_FILE), ("bold", BOLD_FILE)]:
184:        pf = load_json(filepath)
185:        if pf is None:
186:            continue
187:        try:
188:            holdings = pf.get("holdings", {})
189:            holding_tickers = [t for t, h in holdings.items() if h.get("shares", 0) > 0]
190:            data[label] = {
191:                "cash_sek": pf.get("cash_sek", 0),
192:                "initial_value_sek": pf.get("initial_value_sek", 500000),
193:                "total_fees_sek": pf.get("total_fees_sek", 0),
194:                "trades": len(pf.get("transactions", [])),
195:                "holdings": holding_tickers,
196:            }
197:        except (ValueError, AttributeError):
198:            continue
199:    return data
200:
201:
202:def _detect_warnings(entries):
203:    if not entries:
204:        return []
205:    warnings = []
206:
207:    ticker_runs = defaultdict(list)
208:    for e in entries:
209:        tickers = e.get("tickers", {})
210:        prices = e.get("prices", {})
211:        for ticker, info in tickers.items():
212:            outlook = info.get("outlook", "neutral")
213:            if outlook != "neutral":
214:                price = prices.get(ticker)
215:                ticker_runs[ticker].append((outlook, price))
216:
217:    for ticker, runs in ticker_runs.items():
218:        if len(runs) >= 3:
219:            outlooks = [r[0] for r in runs]
220:            prices_list = [r[1] for r in runs if r[1] is not None]
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
220:            prices_list = [r[1] for r in runs if r[1] is not None]
221:            if len(set(outlooks)) == 1 and len(prices_list) >= 2:
222:                outlook = outlooks[0]
223:                first_price = prices_list[0]
224:                last_price = prices_list[-1]
225:                if first_price == 0:
226:                    continue
227:                pct_change = (last_price - first_price) / first_price
228:                if outlook == "bullish" and pct_change < -0.005:
229:                    warnings.append(
230:                        f"{ticker}: thesis (bullish) contradicted — price dropped {abs(pct_change):.1%}"
231:                    )
232:                elif outlook == "bearish" and pct_change > 0.005:
233:                    warnings.append(
234:                        f"{ticker}: thesis (bearish) contradicted — price rose {pct_change:.1%}"
235:                    )
236:
237:    for strat in ("patient", "bold"):
238:        actions = []
239:        for e in entries:
240:            d = e.get("decisions", {}).get(strat, {})
241:            action_str = d.get("action", "HOLD")
242:            match = re.match(r"(BUY|SELL)\s+(\S+)", action_str)
243:            if match:
244:                actions.append((match.group(1), match.group(2)))
245:            else:
246:                actions.append((action_str, None))
247:
248:        for i in range(len(actions) - 2):
249:            a1, t1 = actions[i]
250:            a3, t3 = actions[i + 2]
251:            if t1 and t3 and t1 == t3 and ((a1 == "BUY" and a3 == "SELL") or (a1 == "SELL" and a3 == "BUY")):
252:                    warnings.append(
253:                        f"{strat}: whipsaw on {t1} ({a1}→{a3} within 3 entries)"
254:                    )
255:
256:        ticker_trade_count = defaultdict(int)
257:        for action, ticker in actions:
258:            if ticker and action in ("BUY", "SELL"):
259:                ticker_trade_count[ticker] += 1
260:        for ticker, count in ticker_trade_count.items():
261:            if count >= 3:
262:                warnings.append(
263:                    f"{strat}: churning {ticker} ({count} trades in window)"
264:                )
265:
266:    if len(entries) >= 2:
267:        regimes = [e.get("regime", "unknown") for e in entries]
268:        if len(set(regimes)) == 1:
269:            t0 = datetime.fromisoformat(entries[0]["ts"])
270:            t1 = datetime.fromisoformat(entries[-1]["ts"])
271:            span_hours = (t1 - t0).total_seconds() / 3600
272:            if span_hours >= 8:
273:                warnings.append(
274:                    f"Regime stuck: {regimes[0]} for {span_hours:.0f}h — reassess"
275:                )
276:
277:    return warnings
278:
279:
280:def build_context(entries, portfolio_data=None, now=None):
281:    if not entries:
282:        return "## Your Memory\n\nNo previous invocations. Fresh start.\n"
283:
284:    if now is None:
285:        now = datetime.now(UTC)
286:
287:    lines = []
288:
289:    regimes = [e.get("regime", "unknown") for e in entries]
290:    last_regime = regimes[-1]
291:    streak = 0
292:    for r in reversed(regimes):
293:        if r == last_regime:
294:            streak += 1
295:        else:
296:            break
297:    hours_span = 0
298:    if len(entries) >= 2:
299:        t0 = datetime.fromisoformat(entries[0]["ts"])
300:        t1 = datetime.fromisoformat(entries[-1]["ts"])
301:        hours_span = (t1 - t0).total_seconds() / 3600
302:
303:    lines.append(f"## Your Memory (last {hours_span:.0f}h, {len(entries)} invocations)")
304:    lines.append("")
305:    lines.append(
306:        f"**Regime:** {last_regime} ({streak} invocation{'s' if streak != 1 else ''})"
307:    )
308:    lines.append("")
309:    lines.append("### Recent Decisions")
310:    lines.append("")
311:
312:    i = 0
313:    while i < len(entries):
314:        entry = entries[i]
315:        age = _entry_age_hours(entry, now)
316:
317:        if _is_all_hold(entry):
318:            hold_start = i
319:            while i < len(entries) and _is_all_hold(entries[i]):
320:                i += 1
321:            hold_count = i - hold_start
322:
323:            if hold_count == 1 and age < TIER_FULL:
324:                _append_entry(lines, entry)
325:            elif hold_count == 1:
326:                _append_entry_oneline(lines, entry)
327:            else:
328:                ts_range = _fmt_time_range(
329:                    entries[hold_start]["ts"], entries[i - 1]["ts"]
330:                )
331:                lines.append(f"**{ts_range}** | {hold_count}x HOLD (no setups)")
332:                lines.append("")
333:        else:
334:            if age < TIER_FULL:
335:                _append_entry(lines, entry)
336:            elif age < TIER_COMPACT:
337:                _append_entry_compact(lines, entry)
338:            else:
339:                _append_entry_oneline(lines, entry)
340:            i += 1
341:
342:    watchlist = []
343:    for e in reversed(entries):
344:        wl = e.get("watchlist", [])
345:        if wl:
346:            watchlist = wl
347:            break
348:    if watchlist:
349:        lines.append("### Watchlist")
350:        lines.append("")
351:        for item in watchlist:
352:            lines.append(f"- {item}")
353:        lines.append("")
354:
355:    chains, ts_map = _build_continuation_chains(entries)
356:    if chains:
357:        lines.append("### Thesis Chains")
358:        lines.append("")
359:        for chain in chains:
360:            time_parts = [_fmt_time(ts).replace(" UTC", "") for ts in chain]
361:            tickers_in_chain = set()
362:            for ts in chain:
363:                e = ts_map[ts]
364:                for t in _non_neutral_tickers(e):
365:                    tickers_in_chain.add(t)
366:            ticker_str = (
367:                ", ".join(sorted(tickers_in_chain)) if tickers_in_chain else "general"
368:            )
369:            lines.append(f"{'  →  '.join(time_parts)} UTC: {ticker_str}")
370:        lines.append("")
371:
372:    last = entries[-1]
373:    prices = last.get("prices", {})
374:    if prices:
375:        lines.append("### Prices at Last Entry")
376:        lines.append("")
377:        parts = []
378:        for t in sorted(ALL_TICKERS):
379:            p = prices.get(t)
380:            if p is not None:
381:                parts.append(f"{t}: ${p:,.2f}" if p >= 100 else f"{t}: ${p:,.4f}")
382:        lines.append(" | ".join(parts))
383:        lines.append("")
384:
385:    if portfolio_data:
386:        lines.append("### Portfolio Snapshot")
387:        lines.append("")
388:        for label in ("patient", "bold"):
389:            d = portfolio_data.get(label)
390:            if not d:
391:                continue
392:            cash = d.get("cash_sek", 0)
393:            fees = d.get("total_fees_sek", 0) or 0
394:            trades = d.get("trades", 0)
395:            holdings = d.get("holdings", [])
396:            holding_str = ", ".join(holdings) if holdings else "none"
397:            lines.append(
398:                f"**{label.title()}:** {cash:,.0f} SEK cash | "
399:                f"{trades} trades | {fees:,.0f} fees | holding: {holding_str}"
400:            )
401:        lines.append("")
402:
403:    # Warrant positions
404:    try:
405:        from portfolio.warrant_portfolio import load_warrant_state
406:        w_state = load_warrant_state()
407:        w_holdings = w_state.get("holdings", {})
408:        if w_holdings:
409:            lines.append("### Warrant Positions")
410:            lines.append("")
411:            for key, h in w_holdings.items():
412:                name = h.get("name", key)
413:                underlying = h.get("underlying", "?")
414:                leverage = h.get("leverage", 1)
415:                units = h.get("units", 0)
416:                lines.append(f"**{name}** ({underlying} {leverage}x): {units} units")
417:            lines.append("")
418:    except Exception as e:
419:        import logging as _logging
420:        _logging.getLogger("portfolio.journal").debug("Warrant state load failed: %s", e)
421:
422:    warns = _detect_warnings(entries)
423:    if warns:
424:        lines.append("### Warnings")
425:        lines.append("")
426:        for w in warns:
427:            lines.append(f"- {w}")
428:        lines.append("")
429:
430:    return "\n".join(lines)
431:
432:
433:def _load_config():
434:    """Load config.json for smart retrieval setting."""
435:    config_file = DATA_DIR.parent / "config.json"
436:    return load_json(config_file, default={}) or {}
437:
438:
439:def _get_current_market_state():
440:    """Load current signals, held tickers, regime, and prices for smart retrieval."""
441:    try:
442:        summary_file = DATA_DIR / "agent_summary_compact.json"
443:        if not summary_file.exists():
444:            summary_file = DATA_DIR / "agent_summary.json"
445:        if not summary_file.exists():
446:            return None
447:        summary = load_json(summary_file)
448:        if summary is None:
449:            return None
450:        signals = summary.get("signals", {})
451:
452:        # Detect held tickers
453:        held = set()
454:        for fname in ("portfolio_state.json", "portfolio_state_bold.json"):
455:            pf = load_json(DATA_DIR / fname)
456:            if pf is None:
457:                continue
458:            for t, pos in pf.get("holdings", {}).items():
459:                if pos.get("shares", 0) > 0:
460:                    held.add(t)
461:
462:        # Detect dominant regime
463:        regimes = []
464:        for sig in signals.values():
465:            r = sig.get("regime")
466:            if r:
467:                regimes.append(r)
468:        regime = max(set(regimes), key=regimes.count) if regimes else ""
469:
470:        # Prices
471:        prices = {}
472:        for ticker, sig in signals.items():
473:            p = sig.get("price_usd")
474:            if p:
475:                prices[ticker] = p
476:
477:        return {
478:            "signals": signals,
479:            "held_tickers": list(held),
480:            "regime": regime,
481:            "prices": prices,
482:        }
483:    except Exception as e:
484:        logger.warning("Journal load failed: %s", e, exc_info=True)
485:        return None
486:
487:
488:def _append_vector_memory_section(md, config, market_state, bm25_entries):
489:    """Append semantic memory results to context markdown if enabled."""
490:    vm_cfg = config.get("vector_memory", {})
491:    if not vm_cfg.get("enabled", False):
492:        return md
493:    try:
494:        from portfolio.vector_memory import get_semantic_context
495:        bm25_ts = {e.get("ts", "") for e in bm25_entries} if bm25_entries else set()
496:        top_k = vm_cfg.get("top_k", 5)
497:        collection = vm_cfg.get("collection", "trade_journal")
498:        results = get_semantic_context(
499:            market_state, bm25_timestamps=bm25_ts,
500:            top_k=top_k, collection_name=collection,
501:        )
502:        if not results:
503:            return md
504:        lines = [md.rstrip(), "", "### Semantic Memory", ""]
505:        for r in results:
506:            ts = r.get("ts", "unknown")
507:            regime = r.get("regime", "")
508:            dist = r.get("distance", 0)
509:            # Show first 200 chars of the matched text
510:            text_preview = r.get("text", "")[:200]
511:            if len(r.get("text", "")) > 200:
512:                text_preview += "..."
513:            lines.append(f"**{ts}** (regime: {regime}, dist: {dist:.3f})")
514:            lines.append(text_preview)
515:            lines.append("")
516:        return "\n".join(lines)
517:    except Exception:
518:        return md
519:
520:
521:def _append_reflection_section(md, config):
522:    """Append recent reflection to context markdown if available."""
523:    if not config.get("reflection", {}).get("enabled", False):
524:        return md
525:    try:
526:        from portfolio.reflection import load_latest_reflection
527:        ref = load_latest_reflection()
528:        if not ref:
529:            return md
530:        lines = [md.rstrip(), "", "### Recent Reflection", ""]
531:        for label in ("patient", "bold"):
532:            m = ref.get(label, {})
533:            trades = m.get("trades", 0)
534:            win_rate = m.get("win_rate")
535:            total_pnl = m.get("total_pnl_pct", 0)
536:            wr_str = f"{win_rate:.0%}" if win_rate is not None else "n/a"
537:            lines.append(f"**{label.title()}:** {trades} trades, win rate {wr_str}, PnL {total_pnl:+.1f}%")
538:        insights = ref.get("insights", [])
539:        if insights:
540:            lines.append("")
541:            for insight in insights:
542:                lines.append(f"- {insight}")
543:        lines.append("")
544:        return "\n".join(lines)
545:    except Exception:
546:        return md
547:
548:
549:def write_context():
550:    config = _load_config()
551:    smart = config.get("journal", {}).get("smart_retrieval", True)
552:
553:    if smart:
554:        try:
555:            from portfolio.journal_index import retrieve_relevant_entries
556:            market_state = _get_current_market_state()
557:            if market_state:
558:                entries = retrieve_relevant_entries(
559:                    signals=market_state["signals"],
560:                    held_tickers=market_state["held_tickers"],
561:                    regime=market_state["regime"],
562:                    prices=market_state["prices"],
563:                    k=8,
564:                )
565:                if entries:
566:                    portfolio_data = _load_portfolio_pnl()
567:                    md = build_context(entries, portfolio_data=portfolio_data)
568:                    md = _append_reflection_section(md, config)
569:                    md = _append_vector_memory_section(md, config, market_state, entries)
570:                    atomic_write_text(CONTEXT_FILE, md)
571:                    return len(entries)
572:        except Exception as e:
573:            import logging as _logging
574:            _logging.getLogger("portfolio.journal").debug("Smart retrieval failed, falling back to chronological: %s", e)
575:
576:    # Fallback: chronological (original behavior)
577:    entries = load_recent()
578:    portfolio_data = _load_portfolio_pnl()
579:    md = build_context(entries, portfolio_data=portfolio_data)
580:    md = _append_reflection_section(md, config)
581:    md = _append_vector_memory_section(md, config, None, entries)
582:    atomic_write_text(CONTEXT_FILE, md)
583:    return len(entries)
584:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Smart journal retrieval using BM25 relevance ranking.
2:
3:Replaces chronological "last N entries" with keyword-relevance-ranked retrieval
4:so Layer 2 sees the most contextually relevant prior analyses, not just the
5:most recent.
6:"""
7:
8:import json
9:import logging
10:import math
11:import re
12:from collections import Counter
13:from datetime import UTC, datetime
14:from pathlib import Path
15:
16:logger = logging.getLogger("portfolio.journal_index")
17:
18:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
19:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
20:
21:
22:# ---------------------------------------------------------------------------
23:# Minimal BM25 implementation (no external dependencies)
24:# ---------------------------------------------------------------------------
25:
26:class BM25:
27:    """Okapi BM25 ranking function for document retrieval.
28:
29:    BM25 scores documents by term frequency with diminishing returns
30:    (saturation) and inverse document frequency. No external deps needed.
31:    """
32:
33:    def __init__(self, k1=1.5, b=0.75):
34:        self.k1 = k1
35:        self.b = b
36:        self.doc_count = 0
37:        self.avg_doc_len = 0
38:        self.doc_lens = []
39:        self.term_doc_freq = Counter()  # term -> number of docs containing it
40:        self.doc_term_freqs = []  # list of Counter per document
41:
42:    def fit(self, documents):
43:        """Index a list of token lists.
44:
45:        Args:
46:            documents: list of list[str] (tokenized documents).
47:        """
48:        self.doc_count = len(documents)
49:        self.doc_lens = [len(d) for d in documents]
50:        self.avg_doc_len = sum(self.doc_lens) / self.doc_count if self.doc_count else 1
51:        self.term_doc_freq = Counter()
52:        self.doc_term_freqs = []
53:
54:        for doc in documents:
55:            tf = Counter(doc)
56:            self.doc_term_freqs.append(tf)
57:            for term in set(doc):
58:                self.term_doc_freq[term] += 1
59:
60:    def _idf(self, term):
61:        """Compute inverse document frequency for a term."""
62:        df = self.term_doc_freq.get(term, 0)
63:        if df == 0:
64:            return 0
65:        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
66:
67:    def score(self, query_tokens):
68:        """Score all documents against a query.
69:
70:        Args:
71:            query_tokens: list[str] of query terms.
72:
73:        Returns:
74:            list[float] of scores (one per document, same order as fit()).
75:        """
76:        scores = []
77:        for i in range(self.doc_count):
78:            s = 0
79:            tf_doc = self.doc_term_freqs[i]
80:            doc_len = self.doc_lens[i]
81:            for term in query_tokens:
82:                idf = self._idf(term)
83:                tf = tf_doc.get(term, 0)
84:                numerator = tf * (self.k1 + 1)
85:                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
86:                s += idf * numerator / denominator if denominator > 0 else 0
87:            scores.append(s)
88:        return scores
89:
90:    def top_k(self, query_tokens, k=8):
91:        """Return top-k document indices by BM25 score.
92:
93:        Args:
94:            query_tokens: list[str] of query terms.
95:            k: number of results to return.
96:
97:        Returns:
98:            list of (index, score) tuples, sorted by score descending.
99:        """
100:        scores = self.score(query_tokens)
101:        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
102:        return [(i, s) for i, s in indexed[:k] if s > 0]
103:
104:
105:# ---------------------------------------------------------------------------
106:# Journal Index
107:# ---------------------------------------------------------------------------
108:
109:# Price level buckets for matching "similar price environment"
110:_PRICE_BUCKETS = {
111:    "BTC-USD": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000],
112:    "ETH-USD": [1000, 1500, 2000, 2500, 3000, 4000, 5000],
113:    "XAU-USD": [1800, 1900, 2000, 2100, 2200],
114:    "XAG-USD": [25, 30, 35, 50, 75, 100, 120],
115:}
116:
117:
118:def _price_bucket(ticker, price):
119:    """Convert a price to a searchable bucket token."""
120:    buckets = _PRICE_BUCKETS.get(ticker)
121:    if not buckets or price is None:
122:        return None
123:    for b in buckets:
124:        if price < b:
125:            return f"{ticker}_below_{b}"
126:    return f"{ticker}_above_{buckets[-1]}"
127:
128:
129:def _tokenize_entry(entry):
130:    """Extract searchable tokens from a journal entry.
131:
132:    Tokens include: tickers mentioned, regime, outlook keywords, thesis words,
133:    watchlist items, price level buckets, decision actions.
134:    """
135:    tokens = []
136:
137:    # Regime
138:    regime = entry.get("regime", "")
139:    if regime:
140:        tokens.append(f"regime_{regime}")
141:
142:    # Trigger
143:    trigger = entry.get("trigger", "")
144:    if trigger:
145:        tokens.append(f"trigger_{trigger}")
146:
147:    # Decisions
148:    decisions = entry.get("decisions", {})
149:    for strat in ("patient", "bold"):
150:        d = decisions.get(strat, {})
151:        action = d.get("action", "HOLD")
152:        if action != "HOLD":
153:            tokens.append(f"{strat}_{action.lower()}")
154:        reasoning = d.get("reasoning", "")
155:        if reasoning:
156:            tokens.extend(_clean_words(reasoning))
157:
158:    # Tickers and their outlooks
159:    tickers = entry.get("tickers", {})
160:    for ticker, info in tickers.items():
161:        tokens.append(ticker.lower())
162:        outlook = info.get("outlook", "neutral")
163:        if outlook != "neutral":
164:            tokens.append(f"{ticker.lower()}_{outlook}")
165:        thesis = info.get("thesis", "")
166:        if thesis:
167:            tokens.extend(_clean_words(thesis))
168:        conviction = info.get("conviction", 0)
169:        if conviction >= 0.7:
170:            tokens.append(f"{ticker.lower()}_high_conviction")
171:
172:        # Debate fields (bull/bear/synthesis)
173:        debate = info.get("debate")
174:        if debate and isinstance(debate, dict):
175:            for field in ("bull", "bear", "synthesis"):
176:                text = debate.get(field, "")
177:                if text:
178:                    tokens.extend(_clean_words(text))
179:
180:    # Price buckets
181:    prices = entry.get("prices", {})
182:    for ticker, price in prices.items():
183:        bucket = _price_bucket(ticker, price)
184:        if bucket:
185:            tokens.append(bucket.lower())
186:
187:    # Watchlist
188:    for item in entry.get("watchlist", []):
189:        tokens.extend(_clean_words(item))
190:
191:    # Reflection
192:    reflection = entry.get("reflection", "")
193:    if reflection:
194:        tokens.extend(_clean_words(reflection))
195:
196:    return tokens
197:
198:
199:def _clean_words(text):
200:    """Split text into lowercase word tokens, filtering noise."""
201:    if not text:
202:        return []
203:    words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
204:    # Filter very short words and common stop words
205:    stop = {"the", "a", "an", "is", "was", "are", "be", "to", "of", "and",
206:            "in", "for", "on", "at", "by", "or", "no", "not", "but", "with"}
207:    return [w for w in words if len(w) > 1 and w not in stop]
208:
209:
210:def _compute_importance(entry, now=None):
211:    """Compute importance score for a journal entry.
212:
213:    Factors:
214:    - Time decay: more recent entries score higher
215:    - Trade action: entries with actual trades are more important
216:    - Conviction: high-conviction entries matter more
217:    - Reflection: entries with reflections carry lessons
218:
219:    Returns:
220:        float: importance score (0.0 to 1.0)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
220:        float: importance score (0.0 to 1.0)
221:    """
222:    if now is None:
223:        now = datetime.now(UTC)
224:
225:    score = 0.5  # base
226:
227:    # Time decay: entries from last 2h get full score, exponential decay after
228:    try:
229:        ts = datetime.fromisoformat(entry.get("ts", ""))
230:        if ts.tzinfo is None:
231:            ts = ts.replace(tzinfo=UTC)
232:        age_hours = (now - ts).total_seconds() / 3600
233:        # Half-life of 4 hours
234:        decay = 0.5 ** (age_hours / 4)
235:        score += 0.3 * decay
236:    except (ValueError, TypeError):
237:        pass
238:
239:    # Trade action boost
240:    decisions = entry.get("decisions", {})
241:    for strat in ("patient", "bold"):
242:        action = decisions.get(strat, {}).get("action", "HOLD")
243:        if action != "HOLD":
244:            score += 0.1
245:
246:    # High conviction boost
247:    tickers = entry.get("tickers", {})
248:    max_conviction = max(
249:        (info.get("conviction", 0) for info in tickers.values()),
250:        default=0,
251:    )
252:    if max_conviction >= 0.7:
253:        score += 0.1
254:
255:    # Reflection boost (contains lessons)
256:    if entry.get("reflection"):
257:        score += 0.05
258:
259:    return min(score, 1.0)
260:
261:
262:class JournalIndex:
263:    """BM25-indexed journal for relevance-ranked retrieval."""
264:
265:    def __init__(self):
266:        self.entries = []
267:        self.bm25 = BM25()
268:        self.importances = []
269:
270:    def build(self, entries):
271:        """Index a list of journal entries.
272:
273:        Args:
274:            entries: list of journal entry dicts.
275:        """
276:        self.entries = entries
277:        documents = [_tokenize_entry(e) for e in entries]
278:        self.bm25.fit(documents)
279:        now = datetime.now(UTC)
280:        self.importances = [_compute_importance(e, now) for e in entries]
281:
282:    def query(self, market_state, k=8):
283:        """Retrieve the most relevant journal entries for current market state.
284:
285:        Args:
286:            market_state: dict with keys like:
287:                - held_tickers: list[str]
288:                - regime: str
289:                - prices: dict[str, float]
290:                - signals: dict (ticker -> signal data)
291:            k: number of entries to return.
292:
293:        Returns:
294:            list of journal entry dicts, ranked by relevance.
295:        """
296:        if not self.entries:
297:            return []
298:
299:        query_tokens = _build_query_tokens(market_state)
300:        if not query_tokens:
301:            # Fallback: return most recent
302:            return self.entries[-k:]
303:
304:        results = self.bm25.top_k(query_tokens, k=k * 2)  # Get more, then filter
305:
306:        # Re-rank by BM25 score * importance
307:        ranked = []
308:        for idx, bm25_score in results:
309:            importance = self.importances[idx] if idx < len(self.importances) else 0.5
310:            combined = bm25_score * importance
311:            ranked.append((idx, combined))
312:
313:        ranked.sort(key=lambda x: x[1], reverse=True)
314:
315:        # Return top-k entries
316:        return [self.entries[idx] for idx, _ in ranked[:k]]
317:
318:
319:def _build_query_tokens(market_state):
320:    """Convert current market state into query tokens for BM25."""
321:    tokens = []
322:
323:    regime = market_state.get("regime", "")
324:    if regime:
325:        tokens.append(f"regime_{regime}")
326:
327:    for ticker in market_state.get("held_tickers", []):
328:        tokens.append(ticker.lower())
329:
330:    prices = market_state.get("prices", {})
331:    for ticker, price in prices.items():
332:        bucket = _price_bucket(ticker, price)
333:        if bucket:
334:            tokens.append(bucket.lower())
335:
336:    # Add tickers with non-HOLD signals
337:    signals = market_state.get("signals", {})
338:    for ticker, sig in signals.items():
339:        action = sig.get("action", "HOLD") if isinstance(sig, dict) else "HOLD"
340:        if action != "HOLD":
341:            tokens.append(ticker.lower())
342:            tokens.append(f"{ticker.lower()}_{action.lower()}")
343:
344:    return tokens
345:
346:
347:# ---------------------------------------------------------------------------
348:# Top-level retrieval function
349:# ---------------------------------------------------------------------------
350:
351:def retrieve_relevant_entries(signals, held_tickers, regime, prices, k=8):
352:    """Retrieve the most relevant journal entries for the current market state.
353:
354:    This is the main entry point called by journal.py.
355:
356:    Args:
357:        signals: dict of ticker -> signal data.
358:        held_tickers: list of currently held ticker symbols.
359:        regime: str (current market regime).
360:        prices: dict of ticker -> current USD price.
361:        k: number of entries to return.
362:
363:    Returns:
364:        list of journal entry dicts, ranked by relevance.
365:        Falls back to chronological (most recent) on any error.
366:    """
367:    if not JOURNAL_FILE.exists():
368:        return []
369:
370:    # Load all entries
371:    entries = []
372:    try:
373:        with open(JOURNAL_FILE, encoding="utf-8") as f:
374:            for line in f:
375:                line = line.strip()
376:                if not line:
377:                    continue
378:                try:
379:                    entries.append(json.loads(line))
380:                except json.JSONDecodeError:
381:                    continue
382:    except OSError:
383:        return []
384:
385:    if not entries:
386:        return []
387:
388:    # Build index and query
389:    index = JournalIndex()
390:    index.build(entries)
391:
392:    market_state = {
393:        "held_tickers": held_tickers or [],
394:        "regime": regime or "",
395:        "prices": prices or {},
396:        "signals": signals or {},
397:    }
398:
399:    return index.query(market_state, k=k)
400:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Backfill outcomes for Layer 2 autonomous decisions.
2:
3:Reads data/layer2_decisions.jsonl, checks if enough time has elapsed for
4:each prediction horizon, fetches historical prices, and writes outcome
5:records to data/layer2_decision_outcomes.jsonl.
6:"""
7:
8:from __future__ import annotations
9:
10:import datetime as dt
11:import logging
12:from pathlib import Path
13:
14:from portfolio.file_utils import atomic_append_jsonl, load_jsonl
15:
16:logger = logging.getLogger("portfolio.decision_outcome_tracker")
17:
18:BASE_DIR = Path(__file__).resolve().parent.parent
19:DECISIONS_FILE = BASE_DIR / "data" / "layer2_decisions.jsonl"
20:OUTCOMES_FILE = BASE_DIR / "data" / "layer2_decision_outcomes.jsonl"
21:
22:HORIZONS = {"1d": 86400, "3d": 259200}
23:
24:
25:def backfill_decision_outcomes(max_entries: int = 500) -> int:
26:    """Backfill outcomes for layer2 decisions. Returns count of new outcomes."""
27:    from portfolio.outcome_tracker import _fetch_historical_price
28:
29:    decisions = load_jsonl(DECISIONS_FILE)
30:    if not decisions:
31:        return 0
32:
33:    # Load existing outcomes to avoid duplicates
34:    existing: set[tuple[str | None, str | None, str | None]] = set()
35:    existing_outcomes = load_jsonl(OUTCOMES_FILE)
36:    for o in existing_outcomes or []:
37:        existing.add((o.get("decision_ts"), o.get("ticker"), o.get("horizon")))
38:
39:    now = dt.datetime.now(dt.UTC)
40:    new_count = 0
41:
42:    # Process most recent entries first (limited to max_entries)
43:    for decision in decisions[-max_entries:]:
44:        ts_str = decision.get("ts")
45:        if not ts_str:
46:            continue
47:        try:
48:            decision_ts = dt.datetime.fromisoformat(ts_str)
49:        except (ValueError, TypeError):
50:            continue
51:
52:        predictions = decision.get("predictions") or {}
53:        prices = decision.get("prices") or {}
54:
55:        for ticker, pred in predictions.items():
56:            outlook = pred.get("outlook")
57:            if outlook == "neutral":
58:                continue  # nothing to score
59:
60:            base_price = prices.get(ticker)
61:            if not base_price:
62:                continue
63:
64:            for horizon_name, horizon_secs in HORIZONS.items():
65:                if (ts_str, ticker, horizon_name) in existing:
66:                    continue
67:
68:                target_dt = decision_ts + dt.timedelta(seconds=horizon_secs)
69:                if now < target_dt:
70:                    continue  # not enough time elapsed
71:
72:                # _fetch_historical_price expects a Unix timestamp (float)
73:                target_ts = target_dt.timestamp()
74:                try:
75:                    hist_price = _fetch_historical_price(ticker, target_ts)
76:                except Exception:
77:                    continue
78:                if not hist_price:
79:                    continue
80:
81:                change_pct = ((hist_price - base_price) / base_price) * 100
82:                correct = (outlook == "bullish" and change_pct > 0) or \
83:                          (outlook == "bearish" and change_pct < 0)
84:
85:                outcome = {
86:                    "decision_ts": ts_str,
87:                    "ticker": ticker,
88:                    "horizon": horizon_name,
89:                    "outlook": outlook,
90:                    "conviction": pred.get("conviction", 0),
91:                    "recommendation": pred.get("recommendation"),
92:                    "base_price_usd": base_price,
93:                    "outcome_price_usd": round(hist_price, 4),
94:                    "change_pct": round(change_pct, 4),
95:                    "correct": correct,
96:                    "resolved_at": now.isoformat(),
97:                    "regime": decision.get("regime"),
98:                }
99:                atomic_append_jsonl(OUTCOMES_FILE, outcome)
100:                existing.add((ts_str, ticker, horizon_name))
101:                new_count += 1
102:
103:    if new_count:
104:        logger.info("Backfilled %d Layer 2 decision outcomes", new_count)
105:    return new_count
106:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1:"""Perception gate — filters low-value Layer 2 invocations.
2:
3:A rule-based pre-invocation filter (NOT an LLM call). Checks signal
4:consensus strength and trigger importance. If the gate decides to skip,
5:the agent is not invoked, saving tokens and latency.
6:
7:Config:
8:    "perception_gate": {
9:        "enabled": true,
10:        "min_signal_strength": 0.3,
11:        "skip_tiers": [1]
12:    }
13:"""
14:
15:import logging
16:from pathlib import Path
17:
18:from portfolio.api_utils import load_config
19:from portfolio.file_utils import load_json
20:
21:logger = logging.getLogger("portfolio.perception_gate")
22:
23:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
24:
25:# Trigger keywords that always bypass the gate
26:_BYPASS_KEYWORDS = ("consensus", "F&G crossed", "post-trade")
27:
28:
29:def should_invoke(reasons, tier, config=None):
30:    """Decide whether to invoke Layer 2.
31:
32:    Args:
33:        reasons: list[str] of trigger reasons.
34:        tier: int (1, 2, or 3).
35:        config: optional config dict. Loaded from disk if None.
36:
37:    Returns:
38:        (should_invoke: bool, reason: str) explaining the decision.
39:    """
40:    if config is None:
41:        config = load_config()
42:
43:    gate_cfg = config.get("perception_gate", {})
44:    if not gate_cfg.get("enabled", False):
45:        return True, "gate disabled"
46:
47:    skip_tiers = gate_cfg.get("skip_tiers", [1])
48:    if tier not in skip_tiers:
49:        return True, f"T{tier} not in skip_tiers"
50:
51:    # Force-bypass for important triggers
52:    for reason in reasons:
53:        for keyword in _BYPASS_KEYWORDS:
54:            if keyword in reason:
55:                return True, f"bypass: {keyword!r} in trigger"
56:
57:    # Check signals from compact summary
58:    min_strength = gate_cfg.get("min_signal_strength", 0.3)
59:    summary = _load_compact_summary()
60:    if summary is None:
61:        return True, "no summary available, pass through"
62:
63:    signals = summary.get("signals", {})
64:    if not signals:
65:        return False, "no signals in summary"
66:
67:    max_confidence = 0.0
68:    non_hold_count = 0
69:    for _ticker, sig in signals.items():
70:        if not isinstance(sig, dict):
71:            continue
72:        action = sig.get("action", "HOLD")
73:        conf = sig.get("confidence", 0.0)
74:        if action != "HOLD":
75:            non_hold_count += 1
76:            if conf > max_confidence:
77:                max_confidence = conf
78:
79:    if non_hold_count == 0:
80:        return False, "no non-HOLD signals"
81:
82:    if max_confidence < min_strength:
83:        return False, f"max confidence {max_confidence:.2f} < {min_strength}"
84:
85:    return True, f"{non_hold_count} active signals, max conf {max_confidence:.2f}"
86:
87:
88:def _load_compact_summary():
89:    """Load the compact summary JSON."""
90:    path = DATA_DIR / "agent_summary_compact.json"
91:    result = load_json(path)
92:    if result is not None:
93:        return result
94:    path = DATA_DIR / "agent_summary.json"
95:    return load_json(path)
96:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
120:        "patient": {"action": "HOLD", "reasoning": patient_reasoning},
121:        "bold": {"action": "HOLD", "reasoning": bold_reasoning},
122:    }
123:
124:    # Build ticker entries for journal
125:    ticker_entries = {}
126:    for ticker, pred in predictions.items():
127:        if pred["outlook"] != "neutral":
128:            ticker_entries[ticker] = {
129:                "outlook": pred["outlook"],
130:                "thesis": pred["thesis"],
131:                "conviction": pred["conviction"],
132:                "levels": pred.get("levels", []),
133:            }
134:
135:    # Watchlist
136:    watchlist = _build_watchlist(predictions, reasons)
137:
138:    # Write journal
139:    journal_entry = {
140:        "ts": now.isoformat(),
141:        "source": "autonomous",
142:        "trigger": "; ".join(reasons) if reasons else "unknown",
143:        "regime": regime,
144:        "reflection": reflection,
145:        "continues": prev_entry["ts"] if prev_entry else None,
146:        "decisions": decisions,
147:        "tickers": ticker_entries,
148:        "watchlist": watchlist,
149:        "prices": {t: prices_usd.get(t) for t in signals if prices_usd.get(t) is not None},
150:    }
151:    atomic_append_jsonl(JOURNAL_FILE, journal_entry)
152:
153:    # Write decision log
154:    decision_log = {
155:        "ts": now.isoformat(),
156:        "source": "autonomous",
157:        "tier": tier,
158:        "trigger": "; ".join(reasons) if reasons else "unknown",
159:        "regime": regime,
160:        "predictions": predictions,
161:        "prices": {t: prices_usd.get(t) for t in predictions if prices_usd.get(t) is not None},
162:        "hold_count": hold_count,
163:        "sell_count": sell_count,
164:        "fx_rate": fx_rate,
165:    }
166:    atomic_append_jsonl(DECISIONS_FILE, decision_log)
167:
168:    # Telegram
169:    if _should_send(predictions, reasons, tier):
170:        msg = _build_telegram(
171:            actionable, hold_count, sell_count, state, bold_state,
172:            prices_usd, fx_rate, signals, tf_data, predictions,
173:            config, tier, regime, reflection, reasons,
174:        )
175:        try:
176:            send_or_store(msg, config, category="analysis")
177:            _update_throttle()
178:            logger.info("Autonomous message sent (%d chars)", len(msg))
179:        except Exception:
180:            logger.exception("Autonomous telegram send failed")
181:    else:
182:        logger.info("Autonomous: throttled (routine HOLD)")
183:
184:
185:# ---------------------------------------------------------------------------
186:# Ticker classification
187:# ---------------------------------------------------------------------------
188:
189:def _classify_tickers(signals, patient_state, bold_state, tier, triggered_tickers):
190:    """Classify tickers into actionable set based on tier.
191:
192:    Returns: (actionable_dict, top_hold_list, hold_count, sell_count)
193:    """
194:    if not signals:
195:        return {}, [], 0, 0
196:
197:    # Held tickers across both portfolios
198:    held = set()
199:    for pf in (patient_state, bold_state):
200:        for t, h in pf.get("holdings", {}).items():
201:            if h.get("shares", 0) > 0:
202:                held.add(t)
203:
204:    actionable = {}
205:    hold_count = 0
206:    sell_count = 0
207:
208:    if tier == 1:
209:        # T1: only held positions
210:        for ticker in held:
211:            if ticker in signals:
212:                actionable[ticker] = signals[ticker]
213:        # Count remaining
214:        for ticker, sig in signals.items():
215:            if ticker not in actionable:
216:                if sig["action"] == "SELL":
217:                    sell_count += 1
218:                else:
219:                    hold_count += 1
220:    elif tier == 2:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/autonomous.py:28:from portfolio.telegram_notifications import escape_markdown_v1
portfolio/autonomous.py:36:THROTTLE_FILE = DATA_DIR / "autonomous_throttle.json"
portfolio/autonomous.py:169:    if _should_send(predictions, reasons, tier):
portfolio/autonomous.py:170:        msg = _build_telegram(
portfolio/autonomous.py:180:            logger.exception("Autonomous telegram send failed")
portfolio/autonomous.py:484:def _build_telegram(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:492:        return _build_telegram_mode_b(
portfolio/autonomous.py:498:    return _build_telegram_mode_a(
portfolio/autonomous.py:505:def _build_telegram_mode_a(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:658:def _build_telegram_mode_b(actionable, hold_count, sell_count, patient_state, bold_state,
portfolio/autonomous.py:786:def _should_send(predictions, reasons, tier):
portfolio/autonomous.py:808:    data = load_json(THROTTLE_FILE, default={})
portfolio/autonomous.py:826:        atomic_write_json(THROTTLE_FILE, data)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
780:
781:
782:# ---------------------------------------------------------------------------
783:# Throttle logic
784:# ---------------------------------------------------------------------------
785:
786:def _should_send(predictions, reasons, tier):
787:    """Determine whether to send Telegram for this invocation."""
788:    # Always send for trades, F&G extremes, T3, post-trade
789:    has_action = any(p["recommendation"] in ("BUY", "SELL") for p in predictions.values())
790:    if has_action:
791:        return True
792:    # Skip noise if all reasons are explicit consensus HOLD only.
793:    normalized = [str(r).strip().lower() for r in (reasons or []) if str(r).strip()]
794:    if normalized and all(("consensus" in r and "hold" in r and "buy" not in r and "sell" not in r) for r in normalized):
795:        return False
796:    if tier >= 3:
797:        return True
798:    for r in reasons:
799:        r_lower = r.lower()
800:        if "f&g" in r_lower or "fear" in r_lower:
801:            return True
802:        if "post-trade" in r_lower:
803:            return True
804:        if "consensus" in r_lower and ("buy" in r_lower or "sell" in r_lower):
805:            return True
806:
807:    # Routine HOLD: throttle
808:    data = load_json(THROTTLE_FILE, default={})
809:    last_send = data.get("last_send")
810:    if last_send:
811:        try:
812:            last_dt = datetime.fromisoformat(last_send)
813:            age = (datetime.now(UTC) - last_dt).total_seconds()
814:            if age < _HOLD_COOLDOWN_SECONDS:
815:                return False
816:        except (ValueError, TypeError):
817:            pass
818:    return True
819:
820:
821:def _update_throttle():
822:    """Update throttle timestamp."""
823:    data = {"last_send": datetime.now(UTC).isoformat()}
824:    try:
825:        from portfolio.file_utils import atomic_write_json
826:        atomic_write_json(THROTTLE_FILE, data)
827:    except Exception:
828:        logger.warning("Failed to update throttle file")
829:
830:
831:# ---------------------------------------------------------------------------
832:# Helpers
833:# ---------------------------------------------------------------------------
834:
835:def _load_bold_state_safe():
836:    """Load bold state without crashing."""
837:    try:
838:        return load_bold_state()
839:    except Exception:
840:        return {
841:            "cash_sek": 500000,
842:            "holdings": {},
843:            "transactions": [],
844:            "initial_value_sek": 500000,
845:            "total_fees_sek": 0,
846:        }
847:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 2ms:
portfolio/main.py:74:    fh = open(_SINGLETON_LOCK_FILE, "a+", encoding="utf-8")
portfolio/agent_invocation.py:427:        with open(agent_log_path, "rb") as f:
portfolio/agent_invocation.py:489:        result = subprocess.run(
portfolio/agent_invocation.py:840:        log_fh = open(agent_log_path, "a", encoding="utf-8")
portfolio/agent_invocation.py:858:        _agent_proc = subprocess.Popen(
portfolio/claude_gate.py:8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
portfolio/claude_gate.py:88:        with open(CONFIG_FILE, encoding="utf-8") as f:
portfolio/claude_gate.py:89:            cfg = json.load(f)
portfolio/claude_gate.py:282:# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
portfolio/claude_gate.py:313:            res = subprocess.run(
portfolio/claude_gate.py:364:    proc = subprocess.Popen(
portfolio/gpu_gate.py:47:        proc = subprocess.run(
portfolio/gpu_gate.py:214:                fd = os.open(str(_GPU_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
portfolio/llm_prewarmer.py:124:        with open(STATE_FILE, "rb") as f:
portfolio/llm_prewarmer.py:148:                with open(STATE_FILE, encoding="utf-8") as f:
portfolio/llama_server.py:101:            result = subprocess.run(
portfolio/llama_server.py:115:                    subprocess.run(
portfolio/llama_server.py:120:            result = subprocess.run(
portfolio/llama_server.py:141:            result = subprocess.run(
portfolio/llama_server.py:149:            with open(f"/proc/{pid}/cmdline") as f:
portfolio/llama_server.py:164:            with open(_PID_FILE, encoding="utf-8") as f:
portfolio/llama_server.py:172:                    subprocess.run(
portfolio/llama_server.py:187:        with open(_PID_FILE, "w", encoding="utf-8") as f:
portfolio/llama_server.py:199:            with open(_PID_FILE, encoding="utf-8") as f:
portfolio/llama_server.py:255:        result = subprocess.run(
portfolio/llama_server.py:294:        result = subprocess.run(
portfolio/llama_server.py:456:        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
portfolio/llama_server.py:499:            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
portfolio/llama_server.py:507:                with open(_LOCK_FILE, encoding="utf-8") as f:
portfolio/llama_server.py:511:                    result = subprocess.run(
portfolio/multi_agent_layer2.py:167:            log_fh = open(log_path, "w", encoding="utf-8")
portfolio/multi_agent_layer2.py:168:            proc = subprocess.Popen(
portfolio/journal.py:28:    with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio/journal_index.py:373:        with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio/telegram_poller.py:325:        # raw open()/json.load(). Two reasons:
portfolio/telegram_poller.py:327:        #   2. config.json is a symlink to an external file; raw open() can
portfolio/analyze.py:282:        result = subprocess.run(
portfolio/analyze.py:746:                    result = subprocess.run(
portfolio/bigbet.py:175:        result = subprocess.run(
portfolio/cumulative_tracker.py:67:        with open(SNAPSHOTS_FILE, "rb") as f:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/agent_invocation.py:809:    claude_cmd = shutil.which("claude")
portfolio/agent_invocation.py:844:        agent_env.pop("CLAUDECODE", None)
portfolio/agent_invocation.py:845:        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
portfolio/agent_invocation.py:855:        agent_env["PF_HEADLESS_AGENT"] = "1"
portfolio/multi_agent_layer2.py:136:    claude_cmd = shutil.which("claude")
portfolio/multi_agent_layer2.py:143:    agent_env.pop("CLAUDECODE", None)
portfolio/multi_agent_layer2.py:144:    agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
portfolio/multi_agent_layer2.py:151:    agent_env["PF_HEADLESS_AGENT"] = "1"
portfolio/bigbet.py:169:        # P2 (2026-04-17): PF_HEADLESS_AGENT=1 so the Claude subprocess skips
portfolio/bigbet.py:174:        bigbet_env["PF_HEADLESS_AGENT"] = "1"
portfolio/bigbet.py:176:            ["claude", "-p", prompt, "--max-turns", "1"],
portfolio/analyze.py:24:    """Return env dict without CLAUDECODE to avoid nested-session errors.
portfolio/analyze.py:26:    P2 (2026-04-17): sets PF_HEADLESS_AGENT=1 so the Claude subprocess
portfolio/analyze.py:35:    env.pop("CLAUDECODE", None)
portfolio/analyze.py:36:    env["PF_HEADLESS_AGENT"] = "1"
portfolio/analyze.py:283:            ["claude", "-p", prompt, "--max-turns", "1"],
portfolio/analyze.py:288:            stdin=subprocess.DEVNULL,
portfolio/analyze.py:747:                        ["claude", "-p", prompt, "--max-turns", "1"],
portfolio/analyze.py:752:                        stdin=subprocess.DEVNULL,
portfolio/claude_gate.py:103:    env.pop("CLAUDECODE", None)
portfolio/claude_gate.py:104:    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
portfolio/claude_gate.py:260:    return shutil.which("claude")
portfolio/claude_gate.py:368:        stdin=subprocess.DEVNULL,
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
portfolio/main.py:969:_MAX_CRASH_ALERTS = 5  # stop sending alerts after this many consecutive crashes
portfolio/main.py:970:_MAX_CRASH_BACKOFF = 300  # max sleep between crashes (5 min)
portfolio/main.py:971:_CRASH_SUMMARY_INTERVAL = 100  # send a summary every N crashes after suppression
portfolio/main.py:991:_consecutive_crashes = _load_crash_counter()
portfolio/main.py:997:    After _MAX_CRASH_ALERTS consecutive crashes, stops sending alerts
portfolio/main.py:999:    _CRASH_SUMMARY_INTERVAL crashes so operators retain visibility.
portfolio/main.py:1002:    global _consecutive_crashes
portfolio/main.py:1003:    _consecutive_crashes += 1
portfolio/main.py:1004:    _save_crash_counter(_consecutive_crashes)
portfolio/main.py:1006:    if _consecutive_crashes > _MAX_CRASH_ALERTS:
portfolio/main.py:1009:            _consecutive_crashes, _MAX_CRASH_ALERTS, error_msg[:200],
portfolio/main.py:1012:        if _consecutive_crashes % _CRASH_SUMMARY_INTERVAL == 0:
portfolio/main.py:1017:                    f"CRASH LOOP SUMMARY: {_consecutive_crashes} consecutive crashes\n\n"
portfolio/main.py:1029:        text = f"LOOP CRASH #{_consecutive_crashes}\n\n{error_msg[:3000]}"
portfolio/main.py:1030:        if _consecutive_crashes == _MAX_CRASH_ALERTS:
portfolio/main.py:1044:    base_delay = min(10 * (2 ** (_consecutive_crashes - 1)), _MAX_CRASH_BACKOFF)
portfolio/main.py:1046:    logger.info("Crash backoff: sleeping %.0fs (crash #%d)", delay, _consecutive_crashes)
portfolio/main.py:1058:# _consecutive_crashes was incremented by a non-failing _crash_alert call.
portfolio/main.py:1097:        n = max(_consecutive_crashes, 1)
portfolio/main.py:1108:    global _consecutive_crashes
portfolio/main.py:1109:    if _consecutive_crashes > 0:
portfolio/main.py:1110:        logger.info("Recovered after %d consecutive crashes", _consecutive_crashes)
portfolio/main.py:1111:        _consecutive_crashes = 0
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/main.py:849:                _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
portfolio/agent_invocation.py:769:                # TODO: run specialists in background thread, collect results async.
portfolio/bigbet.py:155:    Never blocks — returns (None, "") on any failure.
portfolio/bigbet.py:186:        # bypasses claude_gate's invoke_claude wrapper, so a "Not logged in"
portfolio/health.py:35:            state["last_invocation_ts"] = state["last_trigger_time"]
portfolio/health.py:183:    last_ts = state.get("last_invocation_ts")
portfolio/health.py:195:            wb_state["last_invocation_ts"] = last_ts
portfolio/health.py:204:        logger.warning("Corrupt last_invocation_ts in health state: %r", last_ts)
portfolio/trigger.py:446:    last_full = state.get("last_full_review_time", 0)
portfolio/trigger.py:493:        state["last_full_review_time"] = time.time()
portfolio/claude_gate.py:69:# lock TODO below.
portfolio/claude_gate.py:656:        ``last_invocation_ts``, ``last_caller``, ``enabled``.
portfolio/claude_gate.py:681:        "last_invocation_ts": last_ts,
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
807:    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)
808:
809:    if triggered or force_report:
810:        reasons_list = reasons if reasons else ["startup"]
811:        summary = write_agent_summary(signals, prices_usd, fx_rate, state, tf_data, reasons_list)
812:        report.summary_written = True
813:        logger.info("Trigger: %s", ', '.join(reasons_list))
814:
815:        # Classify tier and write tier-specific context
816:        from portfolio.reporting import write_tiered_summary
817:        from portfolio.trigger import classify_tier, update_tier_state
818:        tier = classify_tier(reasons_list)
819:        triggered_tickers = _extract_triggered_tickers(reasons_list)
820:        write_tiered_summary(summary, tier, triggered_tickers)
821:        update_tier_state(tier)
822:        logger.info("Tier: T%d (%s)", tier, TIER_CONFIG.get(tier, {}).get('label', 'UNKNOWN'))
823:
824:        try:
825:            from portfolio.outcome_tracker import log_signal_snapshot
826:            log_signal_snapshot(signals, prices_usd, fx_rate, reasons_list)
827:        except Exception as e:
828:            logger.warning("signal logging failed: %s", e)
829:
830:        # 2026-05-04: Wrap long-blocking work (Layer 2 T2/T3 = 600-900s
831:        # subprocess; autonomous fallback = bounded but not instant) in a
832:        # heartbeat keepalive. update_health() (the normal heartbeat write)
833:        # only runs at end-of-cycle, so without periodic ticks the
834:        # dashboard /api/health flips stale 300s into any triggering cycle
835:        # even though the loop is alive and waiting on Claude CLI.
836:        # The context manager's __exit__ runs on exceptions too, so the
837:        # daemon thread is always cleaned up. Skip-paths (NO_TELEGRAM,
838:        # outside agent window) are NOT wrapped — they don't block.
839:        from portfolio.health import heartbeat_keepalive
840:
841:        layer2_cfg = config.get("layer2", {})
842:        if os.environ.get("NO_TELEGRAM"):
843:            logger.info("[NO_TELEGRAM] Skipping agent invocation")
844:            _log_trigger(reasons_list, "skipped_test", tier=tier)
845:        elif layer2_cfg.get("enabled", True):
846:            if _is_agent_window():
847:                with heartbeat_keepalive():
848:                    result = invoke_agent(reasons_list, tier=tier)
849:                _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
850:            else:
851:                logger.info("Layer 2: outside market window, skipping")
852:                _log_trigger(reasons_list, "skipped_offhours", tier=tier)
853:        else:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
867:    # Big Bet detection — can invoke a 30s Claude subprocess per qualifying
868:    # candidate (portfolio/bigbet.py:invoke_layer2_eval), with no per-cycle
869:    # cap. Wrapped in keepalive so heartbeat stays fresh across multi-minute
870:    # bigbet sweeps that would otherwise re-trip the dashboard stale gate.
871:    bigbet_cfg = config.get("bigbet", {})
872:    if bigbet_cfg.get("enabled", False):
873:        try:
874:            from portfolio.bigbet import check_bigbet
875:            from portfolio.health import heartbeat_keepalive
876:            with heartbeat_keepalive():
877:                check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
878:        except Exception as e:
879:            logger.warning("Big Bet check failed: %s", e)
880:
881:    # ISKBETS monitoring — same shape: each qualifying ticker can fire a 30s
882:    # Claude gate subprocess (portfolio/iskbets.py:invoke_layer2_gate). With
883:    # 5 Tier-1 tickers configured the worst case is ~150s of subprocess work,
884:    # well past the 300s heartbeat threshold when stacked with bigbet+L2.
885:    try:
886:        from portfolio.health import heartbeat_keepalive
887:        from portfolio.iskbets import check_iskbets
888:        with heartbeat_keepalive():
889:            check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
890:    except Exception as e:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
296:    # Advance rotation counter — next flush will target the next LLM in rotation.
297:    # Only bumped when at least one phase had work (we already returned early
298:    # for an empty flush above). Wrapping behaviour in is_llm_on_cycle handles
299:    # arbitrary large counters.
300:    _ss._full_llm_cycle_count += 1
301:
302:    # 2026-05-11 (feat/llm-prewarmer Stage 3 Phase 1): pre-warm the NEXT
303:    # LLM in rotation right now, while we still hold no Chronos/gpu_gate.
304:    # Goal: by the time the *next* loop cycle hits (~60 s away), the
305:    # required model is already resident — Chronos's gpu_gate("chronos",
306:    # timeout=30) no longer races a mid-flight cold swap. The prewarmer
307:    # contract is exception-safe (NEVER raises), but we wrap it in an
308:    # outer try/except as a second backstop because a broken prewarmer
309:    # must not cascade into the flush path.
310:    try:
311:        from portfolio.llm_prewarmer import prewarm_next_model
312:        prewarm_next_model(_ss._full_llm_cycle_count)
313:    except Exception as e:
314:        logger.warning("llm prewarmer dispatch failed (non-fatal): %s", e)
---
290:        # Fire the dummy query. n_predict=1 keeps the prompt-completion
291:        # work minimal — the load + KV-cache-prime cost dominates and
292:        # that's what we actually want to pay before the next loop cycle.
293:        from portfolio.llama_server import query_llama_server
294:        t0 = time.monotonic()
295:        logger.info(
296:            "llm_prewarmer start: counter=%d slot=%s server=%s",
297:            counter, next_slot, server_slot,
298:        )
299:        text = query_llama_server(
300:            server_slot, "test", n_predict=1, temperature=0.0,
301:        )
302:        duration = time.monotonic() - t0
303:
304:        if text is None:
305:            # query_llama_server returns None on failure but does not
306:            # raise. Treat that as a soft failure: state is recorded but
307:            # outcome is logged so we can see prewarm failures in the
308:            # JSONL.
309:            logger.warning(
310:                "llm_prewarmer query returned None: counter=%d slot=%s in %.1fs",
311:                counter, next_slot, duration,
312:            )
313:            _write_state(counter, next_slot, server_slot,
314:                         outcome="query_none", duration_s=duration)
315:            return False
---
538:def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
539:                       top_p=0.2, stop=None):
540:    """Query the shared llama-server. Swaps model if needed.
541:
542:    Thread-safe and cross-process-safe via file lock.
543:    Returns completion text or None (caller should fall back to subprocess).
544:    """
545:    cfg = _MODEL_CONFIGS.get(name)
546:    if cfg is None:
547:        return None
548:
549:    # BUG-165: Hold both locks for the entire model-swap + query operation.
550:    # Releasing locks between swap and query allowed another thread/process to
551:    # swap the model mid-query, killing the server and causing silent failures.
552:    # Serialization is correct here — only one 8B model fits in VRAM at a time.
553:    with _thread_lock:
554:        fh = _acquire_file_lock(timeout=300)
555:        if fh is None:
556:            return None
557:        try:
558:            if not _ensure_model(name):
559:                return None
560:            text = _query_http(prompt, n_predict, temperature, top_p, stop)
561:            if text is None:
562:                logger.warning("llama-server %s returned empty response", name)
563:            return text
564:        except Exception as e:
565:            logger.warning("llama-server %s query failed: %s", name, e)
566:            return None
567:        finally:
568:            _release_file_lock(fh)
569:
570:
571:def _query_http(prompt, n_predict=1024, temperature=0.0, top_p=0.2, stop=None):
572:    """Send an HTTP completion request. No locking — caller must hold locks.
573:
574:    2026-04-10 (perf/llama-swap-reduction): added `cache_prompt: true`. This
575:    is a llama.cpp server feature (stable since b2000, our binary is the
576:    cuda13 build from 2025 which supports it) that tells the server to
577:    reuse the KV cache across successive requests whenever the new prompt
578:    shares a token prefix with the previous one. Our Ministral prompts in
579:    _build_prompt share ~300 tokens of fixed boilerplate (the [INST] header,
580:    the analysis questions 1-5, the JSON schema at the bottom) and differ
581:    only in the per-ticker Market Data / Sentiment / Headlines block. On
582:    the 4 sequential per-ticker Ministral queries per cycle, this saves
583:    ~300 prefill tokens × 3 queries = ~900 tokens of prefill work, which
584:    on an 8B Q5_K_M is roughly ~5-15 s per batch. If the server build
585:    doesn't recognize the field, it is silently ignored — no breakage.
586:    """
587:    body = {
588:        "prompt": prompt,
589:        "n_predict": n_predict,
590:        "temperature": temperature,
591:        "top_p": top_p,
592:        "cache_prompt": True,
593:    }
594:    if stop:
595:        body["stop"] = stop
596:    r = _requests.post(
597:        f"http://127.0.0.1:{_PORT}/completion",
598:        json=body,
599:        timeout=240,
600:    )
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
836:        # check_agent_completion() can read only this invocation's output
837:        # (for auth-error detection) and not the entire log history.
838:        global _agent_log_start_offset
839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
840:        log_fh = open(agent_log_path, "a", encoding="utf-8")
841:        # Strip Claude Code session markers to avoid "nested session" error
842:        # when the parent process tree has Claude Code running
843:        agent_env = os.environ.copy()
844:        agent_env.pop("CLAUDECODE", None)
845:        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
846:        # Increase Node.js stack size to prevent stack overflow in Claude CLI
847:        agent_env["NODE_OPTIONS"] = "--stack-size=16384"
848:        # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
849:        # STARTUP CHECK protocol doesn't ask "How would you like to proceed?"
850:        # when it finds unresolved critical_errors.jsonl entries. The agent
851:        # has no stdin (pipe only), so any prompt that blocks on user input
852:        # makes it hit the tier timeout with zero work done. The CLAUDE.md
853:        # conditional turns that into "log the unresolved entries in your
854:        # journal entry and proceed with the trigger task".
855:        agent_env["PF_HEADLESS_AGENT"] = "1"
856:        _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
857:        _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
858:        _agent_proc = subprocess.Popen(
859:            cmd,
860:            cwd=str(BASE_DIR),
861:            stdout=log_fh,
862:            stderr=subprocess.STDOUT,
863:            env=agent_env,
864:        )
---
165:        try:
166:            log_path = DATA_DIR / f"_specialist_{name}.log"
167:            log_fh = open(log_path, "w", encoding="utf-8")
168:            proc = subprocess.Popen(
169:                cmd,
170:                cwd=str(BASE_DIR),
171:                stdout=log_fh,
172:                stderr=subprocess.STDOUT,
173:                env=agent_env,
174:            )
175:            proc._log_fh = log_fh  # attach for cleanup
176:            proc._name = name
177:            procs.append(proc)
178:            logger.info("Specialist %s launched pid=%s", name, proc.pid)
179:        except Exception as e:
180:            logger.error("Failed to launch specialist %s: %s", name, e)
181:
182:    return procs
183:
184:
185:def wait_for_specialists(
186:    procs: list[subprocess.Popen],
187:    timeout: int = 150,
188:) -> dict[str, bool]:
189:    """Wait for all specialist agents to complete.
190:
191:    Returns dict of specialist_name -> success (True/False).
192:    """
193:    results = {}
194:    deadline = time.time() + timeout
195:
196:    for proc in procs:
197:        remaining = max(1, deadline - time.time())
198:        name = getattr(proc, "_name", "unknown")
199:        try:
200:            proc.wait(timeout=remaining)
201:            success = proc.returncode == 0
202:            results[name] = success
203:            if not success:
204:                logger.warning("Specialist %s exited with code %d", name, proc.returncode)
205:        except subprocess.TimeoutExpired:
206:            logger.warning("Specialist %s timed out, killing", name)
207:            proc.kill()
208:            proc.wait(timeout=5)
209:            results[name] = False
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
485:    kill_ok = True
486:    if platform.system() == "Windows":
487:        # BUG-92: Check taskkill return code to detect kill failure
488:        # BUG-189: rc=128 means process already exited — treat as success
489:        result = subprocess.run(
490:            ["taskkill", "/F", "/T", "/PID", str(pid)],
491:            capture_output=True,
492:        )
493:        if result.returncode not in (0, 128):
494:            logger.error(
495:                "taskkill failed (rc=%d): %s",
496:                result.returncode,
497:                result.stderr.decode(errors="replace").strip(),
498:            )
499:            kill_ok = False
500:        elif result.returncode == 128:
501:            logger.info("Agent pid=%s already exited (rc=128)", pid)
502:    else:
503:        _agent_proc.kill()
504:    try:
505:        _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
506:    except subprocess.TimeoutExpired:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
180:    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
181:    last_ts = None
182:    state = load_health()
183:    last_ts = state.get("last_invocation_ts")
184:
185:    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
186:    if not last_ts:
187:        invocations_file = DATA_DIR / "invocations.jsonl"
188:        last_ts = last_jsonl_entry(invocations_file, field="ts")
189:        if last_ts is None:
190:            return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
191:        # Write back to health state so subsequent calls hit the cache
192:        # instead of re-parsing the JSONL file every time.
193:        with _health_lock:
194:            wb_state = load_health()
195:            wb_state["last_invocation_ts"] = last_ts
196:            atomic_write_json(HEALTH_FILE, wb_state)
197:
198:    if not last_ts:
199:        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
200:
201:    try:
202:        last = datetime.fromisoformat(last_ts)
203:    except (ValueError, TypeError):
204:        logger.warning("Corrupt last_invocation_ts in health state: %r", last_ts)
205:        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}
206:    now = datetime.now(UTC)
207:    age = (now - last).total_seconds()
208:
209:    # DST-aware market hours check
210:    from portfolio.market_timing import get_market_state
211:    market_state, _, _ = get_market_state()
212:    market_open = (market_state == "open")
213:    threshold = max_market_seconds if market_open else max_offhours_seconds
214:
215:    return {
216:        "silent": age > threshold,
217:        "age_seconds": round(age, 1),
218:        "threshold": threshold,
---
219:    triggered_consensus = state.get("triggered_consensus", {})
220:    for ticker, sig in signals.items():
221:        action = sig["action"]
222:        last_tc = triggered_consensus.get(ticker, "HOLD")
223:        if action in ("BUY", "SELL") and last_tc == "HOLD":
224:            conf = sig.get("confidence", 0)
225:            # Ranging regime dampening: skip low-confidence consensus triggers
226:            ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
227:            if (
228:                ticker_regime == "ranging"
229:                and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
230:                and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
231:            ):
232:                logger.info(
233:                    "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
234:                    "(min %.0f%%)",
235:                    ticker, action, conf * 100,
236:                    RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
237:                )
238:                # Still update baseline so we don't re-trigger next cycle
239:                triggered_consensus[ticker] = action
240:                continue
---
490:    if state is None:
491:        state = _load_state()
492:    if tier == 3:
493:        state["last_full_review_time"] = time.time()
494:    _save_state(state)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
20:def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
21:                  last_trigger_reason: str = None, error: str = None):
22:    """Called at end of each Layer 1 cycle to update health state."""
23:    with _health_lock:
24:        state = load_health()
25:        state["last_heartbeat"] = datetime.now(UTC).isoformat()
26:        state["cycle_count"] = cycle_count
27:        state["signals_ok"] = signals_ok
28:        state["signals_failed"] = signals_failed
29:        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
30:        if last_trigger_reason:
31:            state["last_trigger_reason"] = last_trigger_reason
32:            state["last_trigger_time"] = datetime.now(UTC).isoformat()
33:            # Cache the invocation timestamp so check_agent_silence() can avoid
34:            # re-parsing invocations.jsonl on every call.
35:            state["last_invocation_ts"] = state["last_trigger_time"]
36:        if error:
37:            state["errors"] = state.get("errors", [])[-19:] + [
38:                {"ts": datetime.now(UTC).isoformat(), "error": error}
39:            ]
40:            state["error_count"] = state.get("error_count", 0) + 1
41:        atomic_write_json(HEALTH_FILE, state)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
80:def _load_config_layer2_enabled() -> bool:
81:    """Check ``config.json -> layer2.enabled``.
82:
83:    Returns True if the key is missing or the file cannot be read (fail-open
84:    for the config check — the module-level CLAUDE_ENABLED flag is the hard
85:    gate).
86:    """
87:    try:
88:        with open(CONFIG_FILE, encoding="utf-8") as f:
89:            cfg = json.load(f)
90:        return cfg.get("layer2", {}).get("enabled", True)
91:    except Exception:
92:        # Config unreadable — don't block on that alone.
93:        return True
94:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
portfolio/market_timing.py:16:# 2026-04-09: all bumped to 600s (10 min). The reduced 5-ticker universe +
portfolio/market_timing.py:17:# warm fingpt daemon means we no longer need a 60s fast tick — giving the
portfolio/market_timing.py:19:# meaningful. Weekend was already 600s. See docs/PLAN_FINGPT_DAEMON.md.
portfolio/market_timing.py:20:INTERVAL_MARKET_OPEN = 600    # 10 min — previously 60s (pre-daemon era)
portfolio/market_timing.py:21:INTERVAL_MARKET_CLOSED = 600  # 10 min — previously 120s
portfolio/market_timing.py:22:INTERVAL_WEEKEND = 600        # 10 min — unchanged
portfolio/market_timing.py:46:    oct31 = date(year, 10, 31)
portfolio/market_timing.py:48:    eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
portfolio/market_timing.py:112:    b, c = divmod(year, 100)
portfolio/market_timing.py:167:    Covers all 10 NYSE holidays including observed-date shifts.
portfolio/market_timing.py:297:    now_min = now.hour * 60 + now.minute
portfolio/market_timing.py:298:    open_min = open_hour * 60 + 30 - pre_market_buffer_min   # NYSE opens at :30
portfolio/market_timing.py:299:    close_min = close_hour * 60 + post_market_buffer_min      # NYSE closes at :00
portfolio/market_timing.py:334:        return "weekend", always_on, INTERVAL_WEEKEND
portfolio/market_timing.py:337:        return "holiday", always_on, INTERVAL_MARKET_CLOSED
portfolio/market_timing.py:341:        return "open", all_symbols, INTERVAL_MARKET_OPEN
portfolio/market_timing.py:342:    return "closed", always_on, INTERVAL_MARKET_CLOSED
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/agent_invocation.py:26:TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
portfolio/agent_invocation.py:928:            send_or_store(notify_msg, config, category="invocation")
portfolio/agent_invocation.py:1274:            send_or_store(
portfolio/agent_invocation.py:1284:            send_or_store(
portfolio/agent_invocation.py:1309:                send_or_store(
portfolio/main.py:360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl", "claude_invocations.jsonl"):
portfolio/main.py:772:            send_or_store(_fail_msg, config, category="error")
portfolio/main.py:936:                    send_or_store(msg, config, category="error")
portfolio/main.py:948:                    send_or_store(msg, config, category="error")
portfolio/main.py:1021:                send_or_store(text, config, category="error")
portfolio/main.py:1033:        send_or_store(text, config, category="error")
portfolio/main.py:1156:                    send_or_store(msg, config, category="error")
portfolio/digest.py:267:        send_or_store(msg, config, category="digest")
portfolio/daily_digest.py:274:            send_or_store(msg, config, category="daily_digest")
portfolio/weekly_digest.py:280:    log_file = DATA_DIR / "telegram_messages.jsonl"
portfolio/weekly_digest.py:291:        result = send_telegram(msg, config)
portfolio/autonomous.py:176:            send_or_store(msg, config, category="analysis")
portfolio/bigbet.py:456:                _send_telegram(msg, config)
portfolio/bigbet.py:481:                _send_telegram(msg, config)
portfolio/bigbet.py:504:                _send_telegram(msg, config)
portfolio/bigbet.py:599:                _send_telegram(msg, config)
portfolio/bigbet.py:618:def _send_telegram(msg, config):
portfolio/bigbet.py:619:    send_or_store(msg, config, category="bigbet")
portfolio/regime_alerts.py:152:    Also logs the message to telegram_messages.jsonl.
portfolio/regime_alerts.py:181:        send_or_store(msg, config, category="regime")
portfolio/telegram_notifications.py:35:def send_telegram(msg, config):
portfolio/telegram_notifications.py:139:        send_or_store(msg, config, category="analysis")
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
266:        msg: The message string to send.
267:
268:    Returns:
269:        requests.Response or None on error.
270:    """
271:    config = load_json(CONFIG_FILE, default={})
272:    token = config.get("telegram", {}).get("token")
273:    chat_id = config.get("telegram", {}).get("chat_id")
274:
275:    if not token or not chat_id:
276:        print("ERROR: Telegram config missing token or chat_id")
277:        return None
278:
279:    # Save locally first
280:    log_file = DATA_DIR / "telegram_messages.jsonl"
281:    entry = {
282:        "ts": datetime.now(UTC).isoformat(),
283:        "text": msg,
284:        "type": "weekly_digest",
285:    }
286:    atomic_append_jsonl(log_file, entry)
287:
288:    # Send via shared module
289:    try:
290:        from portfolio.telegram_notifications import send_telegram
291:        result = send_telegram(msg, config)
292:        print(f"Telegram sent: {result}")
293:        return result
294:    except Exception as e:
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
portfolio/main.py:14:- digest.py — 4-hour digest builder
portfolio/main.py:15:- daily_digest.py — morning daily digest (focus instruments + movers)
portfolio/main.py:145:from portfolio.digest import _maybe_send_digest  # noqa: E402, F401
portfolio/main.py:264:    """Post-cycle housekeeping: digest, daily digest, message throttle flush, AV refresh.
portfolio/main.py:282:    _maybe_send_digest(config)
portfolio/main.py:290:        from portfolio.daily_digest import maybe_send_daily_digest
portfolio/main.py:291:        _track("daily_digest", maybe_send_daily_digest, config)
portfolio/main.py:293:        logger.warning("daily digest import failed: %s", e_dd)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
356:    # Prune unbounded JSONL files to prevent disk exhaustion (BUG-59).
357:    # Per-file isolation: a locked file doesn't block the others.
358:    from portfolio.file_utils import prune_jsonl
359:    _prune_failures = []
360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl", "claude_invocations.jsonl"):
361:        try:
362:            prune_jsonl(DATA_DIR / name, max_entries=5000)
363:        except Exception as e_prune:
364:            _prune_failures.append(name)
365:            logger.warning("JSONL prune failed for %s: %s", name, e_prune)
366:    if report is not None:
367:        report.post_cycle_results["jsonl_prune"] = len(_prune_failures) == 0
368:    # Fin command self-improvement: backfill outcomes + evolve lessons (daily)
369:    try:
370:        from portfolio.fin_evolve import maybe_evolve
371:        _track("fin_evolve", maybe_evolve, config)
372:    except Exception as e_evolve:
373:        logger.warning("Fin evolve import failed: %s", e_evolve)
374:    # Scheduled crypto analysis report (08:00, 13:00, 18:00 CET)
375:    try:
376:        from portfolio.crypto_scheduler import maybe_run_crypto_report
377:        _track("crypto_scheduler", maybe_run_crypto_report, config)
378:    except Exception as e_crypto:
379:        logger.warning("Crypto scheduler import failed: %s", e_crypto)
380:    # Signal postmortem (daily — uses accuracy cache, generates once per day)
381:    try:
382:        from portfolio.file_utils import load_json as _lj
383:        from portfolio.signal_postmortem import POSTMORTEM_FILE, generate_postmortem
384:        pm = _lj(POSTMORTEM_FILE)
385:        # Regenerate if missing or stale (>20 hours old)
386:        if not pm or (time.time() - pm.get("_epoch", 0)) > 72000:
387:            result = generate_postmortem()
388:            if result:
389:                result["_epoch"] = time.time()
390:                from portfolio.file_utils import atomic_write_json as _awj
391:                _awj(POSTMORTEM_FILE, result)
392:        if report is not None:
393:            report.post_cycle_results["signal_postmortem"] = True
394:    except Exception as e_pm:
395:        logger.warning("Signal postmortem failed: %s", e_pm)
396:        if report is not None:
397:            report.post_cycle_results["signal_postmortem"] = False
398:    # H25/L3: Rotate unbounded JSONL files approximately once per hour.
399:    # Was cycle-count-based (every 60 cycles assumed 60s cadence = 1h). After
400:    # the 2026-04-09 cadence bump to 600s that would become once per 10h, so
401:    # this is now driven by wall-clock via a monotonic timestamp tracked on
402:    # shared_state so it survives cross-module access within one process.
403:    _now_rot_ts = time.monotonic()
404:    _last_rot_ts = getattr(_ss, "_last_log_rotation_ts", None)
405:    should_rotate_logs = _last_rot_ts is None or (_now_rot_ts - _last_rot_ts) >= 3600
406:    if should_rotate_logs:
407:        try:
408:            from portfolio.log_rotation import rotate_all
409:            rotation_results = rotate_all()
410:            rotated = [r for r in rotation_results if r.get("status") == "rotated"]
411:            if rotated:
412:                logger.info("Log rotation: %d file(s) rotated: %s",
413:                            len(rotated), [r["file"] for r in rotated])
414:            if report is not None:
415:                report.post_cycle_results["log_rotation"] = True
416:            _ss._last_log_rotation_ts = _now_rot_ts
417:        except Exception as e_rot:
418:            logger.warning("Log rotation failed: %s", e_rot)
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 0ms:
1221:        cycle_started = time.monotonic()
1222:        try:
1223:            report = run(force_report=False, active_symbols=active_symbols)
1224:            _run_post_cycle(config, report=report)
1225:            _reset_crash_counter()
codex
I’ve isolated the remaining findings to failures that are both code-real and operationally expensive: leaked worker threads, synchronous prewarm blocking, false-green health/tier bookkeeping, and raw Claude paths that bypass the hardened launcher. I’m writing the final review now with exact line references and missing tests.
  ✓ Read all in-scope orchestration files with line numbers
  ✓ Trace subprocess, concurrency, trigger, alerting, and journaling failure modes
  → Draft adversarial findings with exact file:line references
  ✓ Read all in-scope orchestration files with line numbers
  ✓ Trace subprocess, concurrency, trigger, alerting, and journaling failure modes
  ✓ Draft adversarial findings with exact file:line references
codex
# Codex adversarial review: orchestration
## Summary
The Layer 2 auth-detection work is materially better than the March-April failure mode, but the orchestrator still has several paths that can look healthy while doing the wrong thing. The code comments also say the loop cadence was bumped from 60s to 600s on April 9, 2026; the blockers below still matter because they either leak work across cycles or add multi-minute blocking to the main path.

## P0 — Blockers
- portfolio/main.py:657 — Ticker-pool timeouts leak live worker threads past the 8-worker contract. Why it bites: the timeout path only calls `future.cancel()` plus `pool.shutdown(wait=False, cancel_futures=True)`, which does not stop already-running ticker jobs; every slow/stuck cycle can leave orphaned workers running while the next cycle creates a fresh pool, so concurrency grows beyond 8 and stale ticker work keeps mutating shared caches/rate limits in the background. Fix: make ticker work killable (process-based isolation) or enforce hard timeouts inside `_process_ticker` and do not roll a new executor until the old workers are actually gone.
- portfolio/llm_batch.py:310 — The new prewarmer is synchronous and inherits the full llama-server blocking budget. Why it bites: `prewarm_next_model()` immediately calls `query_llama_server()` (`portfolio/llm_prewarmer.py:299`), which can wait 300s on the file lock and 240s on HTTP (`portfolio/llama_server.py:553-600`); a 1-token warm-up can therefore pin the post-cycle path for minutes. Fix: prewarm asynchronously, or give prewarm its own short budget and skip immediately on lock contention/server slowness.

## P1 — High
- portfolio/health.py:35 — `check_agent_silence()` is false-green because it tracks triggers, not real Layer 2 executions. Why it bites: `update_health()` writes `last_invocation_ts = last_trigger_time`, and the fallback path just reads the last row in `invocations.jsonl` (`portfolio/health.py:183-196`); `_log_trigger()` records `skipped_busy`, `skipped_offhours`, `blocked_*`, etc. (`portfolio/agent_invocation.py:243-251`, `portfolio/main.py:844-852`), so a dead or permanently gated Layer 2 can look healthy indefinitely. Fix: persist a dedicated timestamp only on successful spawn/completion, and have silence checks ignore non-execution statuses.
- portfolio/main.py:821 — Tier state is advanced before any Layer 2 work actually runs. Why it bites: `update_tier_state(tier)` runs before `invoke_agent()`, so a busy/off-hours/gated/disabled path still stamps `last_full_review_time` (`portfolio/trigger.py:492-494`); the next real T3 can be suppressed for 4 hours, or the first-of-day full review can be consumed without any review happening. Fix: update tier state only after a real Layer 2/autonomous review starts or completes.
- portfolio/main.py:356 — `_run_post_cycle()` prunes and rotates live JSONL journals while asynchronous Layer 2 may still be writing them. Why it bites: `run()` launches Layer 2 asynchronously (`portfolio/main.py:847-849`) and the loop immediately calls `_run_post_cycle()` (`portfolio/main.py:1223-1224`), which prunes `invocations.jsonl`, `layer2_journal.jsonl`, and `telegram_messages.jsonl`, then rotates logs; that races the active writer on the exact files used for completion detection and journaling. Fix: skip prune/rotation for live Layer 2 files while `_agent_proc` exists, or coordinate with a writer-side lock.
- portfolio/bigbet.py:175 — Big Bet bypasses the hardened Claude launcher and the `CLAUDECODE` rule, and the loop can call it repeatedly in one cycle. Why it bites: the main loop explicitly allows one 30s Claude eval per qualifying candidate with no per-cycle cap (`portfolio/main.py:867-877`), while `invoke_layer2_eval()` uses raw `subprocess.run(["claude", "-p", ...])`, inherits `CLAUDECODE`, and does not use the tree-kill helper; one noisy cycle can spend minutes on optional evals and nested-session environments silently disable the feature. Fix: route Big Bet through `claude_gate` semantics (`_clean_env()` + tree kill) and enforce a hard per-cycle eval budget.
- portfolio/agent_invocation.py:851 — The main Layer 2 subprocess is documented as headless, but it still inherits stdin. Why it bites: the comment says “has no stdin,” yet `Popen(...)` omits `stdin=subprocess.DEVNULL`; if CLAUDE.md or the CLI emits any unexpected prompt, the process can block on real stdin until the full 120/600/900s tier timeout. The same bug exists in specialist launches (`portfolio/multi_agent_layer2.py:168-174`). Fix: set `stdin=subprocess.DEVNULL` on every headless `claude -p` spawn.

## P2 — Medium
- portfolio/trigger.py:238 — Ranging-regime dampening permanently consumes the consensus edge it suppresses. Why it bites: when a low-confidence ranging BUY/SELL is suppressed, the code still writes `triggered_consensus[ticker] = action`; if confidence strengthens on later cycles without first reverting to HOLD, Layer 2 never sees the now-actionable consensus. Fix: keep suppressed signals in a pending state and only advance the consensus baseline after a real trigger or a reset to HOLD.
- portfolio/claude_gate.py:87 — The central Claude gate still uses raw `open()/json.load()` and fails open on read errors. Why it bites: concurrent atomic config writes or a transient symlink/read failure return `True`, so Claude invocations keep running even when ops set `layer2.enabled=false`; this also violates the project’s “atomic I/O via file_utils only” rule. Fix: use `load_json()` and decide the disable gate fail-safe, not fail-open.

## P3 — Low
- portfolio/journal.py:28 — Journal context building still scans raw JSONL files with plain `open()`; `portfolio/journal_index.py:373` does the same for BM25 retrieval. Why it bites: concurrent append/rotation is treated as “skip malformed line,” so the newest memory can disappear silently from `layer2_context.md` exactly when Layer 2 is calibrating off recent history. Fix: move these reads onto coordinated `file_utils` helpers or tail/index snapshots, not raw full-file scans.

## Tests missing
- A timeout test that proves no ticker worker threads survive `BUG-178` pool timeout handling and that concurrency never exceeds 8 after repeated slow cycles.
- A prewarmer test that holds the llama-server lock/server open and asserts prewarm returns within a short bounded budget instead of inheriting 300s/240s waits.
- A health test that shows `check_agent_silence()` does not advance on `skipped_busy`, `skipped_offhours`, `blocked_*`, or trigger-only cycles.
- A tier-state test that shows `last_full_review_time` changes only after a real T3 execution, not after a skipped/gated classification.
- A trigger test for the ranging dampener: weak BUY suppressed first, stronger BUY later without a HOLD reset should still fire once.
- A live-writer test that runs Layer 2 append activity concurrently with prune/rotation and proves `layer2_journal.jsonl` / `telegram_messages.jsonl` entries are not lost or misclassified.
tokens used
234,064
# Codex adversarial review: orchestration
## Summary
The Layer 2 auth-detection work is materially better than the March-April failure mode, but the orchestrator still has several paths that can look healthy while doing the wrong thing. The code comments also say the loop cadence was bumped from 60s to 600s on April 9, 2026; the blockers below still matter because they either leak work across cycles or add multi-minute blocking to the main path.

## P0 — Blockers
- portfolio/main.py:657 — Ticker-pool timeouts leak live worker threads past the 8-worker contract. Why it bites: the timeout path only calls `future.cancel()` plus `pool.shutdown(wait=False, cancel_futures=True)`, which does not stop already-running ticker jobs; every slow/stuck cycle can leave orphaned workers running while the next cycle creates a fresh pool, so concurrency grows beyond 8 and stale ticker work keeps mutating shared caches/rate limits in the background. Fix: make ticker work killable (process-based isolation) or enforce hard timeouts inside `_process_ticker` and do not roll a new executor until the old workers are actually gone.
- portfolio/llm_batch.py:310 — The new prewarmer is synchronous and inherits the full llama-server blocking budget. Why it bites: `prewarm_next_model()` immediately calls `query_llama_server()` (`portfolio/llm_prewarmer.py:299`), which can wait 300s on the file lock and 240s on HTTP (`portfolio/llama_server.py:553-600`); a 1-token warm-up can therefore pin the post-cycle path for minutes. Fix: prewarm asynchronously, or give prewarm its own short budget and skip immediately on lock contention/server slowness.

## P1 — High
- portfolio/health.py:35 — `check_agent_silence()` is false-green because it tracks triggers, not real Layer 2 executions. Why it bites: `update_health()` writes `last_invocation_ts = last_trigger_time`, and the fallback path just reads the last row in `invocations.jsonl` (`portfolio/health.py:183-196`); `_log_trigger()` records `skipped_busy`, `skipped_offhours`, `blocked_*`, etc. (`portfolio/agent_invocation.py:243-251`, `portfolio/main.py:844-852`), so a dead or permanently gated Layer 2 can look healthy indefinitely. Fix: persist a dedicated timestamp only on successful spawn/completion, and have silence checks ignore non-execution statuses.
- portfolio/main.py:821 — Tier state is advanced before any Layer 2 work actually runs. Why it bites: `update_tier_state(tier)` runs before `invoke_agent()`, so a busy/off-hours/gated/disabled path still stamps `last_full_review_time` (`portfolio/trigger.py:492-494`); the next real T3 can be suppressed for 4 hours, or the first-of-day full review can be consumed without any review happening. Fix: update tier state only after a real Layer 2/autonomous review starts or completes.
- portfolio/main.py:356 — `_run_post_cycle()` prunes and rotates live JSONL journals while asynchronous Layer 2 may still be writing them. Why it bites: `run()` launches Layer 2 asynchronously (`portfolio/main.py:847-849`) and the loop immediately calls `_run_post_cycle()` (`portfolio/main.py:1223-1224`), which prunes `invocations.jsonl`, `layer2_journal.jsonl`, and `telegram_messages.jsonl`, then rotates logs; that races the active writer on the exact files used for completion detection and journaling. Fix: skip prune/rotation for live Layer 2 files while `_agent_proc` exists, or coordinate with a writer-side lock.
- portfolio/bigbet.py:175 — Big Bet bypasses the hardened Claude launcher and the `CLAUDECODE` rule, and the loop can call it repeatedly in one cycle. Why it bites: the main loop explicitly allows one 30s Claude eval per qualifying candidate with no per-cycle cap (`portfolio/main.py:867-877`), while `invoke_layer2_eval()` uses raw `subprocess.run(["claude", "-p", ...])`, inherits `CLAUDECODE`, and does not use the tree-kill helper; one noisy cycle can spend minutes on optional evals and nested-session environments silently disable the feature. Fix: route Big Bet through `claude_gate` semantics (`_clean_env()` + tree kill) and enforce a hard per-cycle eval budget.
- portfolio/agent_invocation.py:851 — The main Layer 2 subprocess is documented as headless, but it still inherits stdin. Why it bites: the comment says “has no stdin,” yet `Popen(...)` omits `stdin=subprocess.DEVNULL`; if CLAUDE.md or the CLI emits any unexpected prompt, the process can block on real stdin until the full 120/600/900s tier timeout. The same bug exists in specialist launches (`portfolio/multi_agent_layer2.py:168-174`). Fix: set `stdin=subprocess.DEVNULL` on every headless `claude -p` spawn.

## P2 — Medium
- portfolio/trigger.py:238 — Ranging-regime dampening permanently consumes the consensus edge it suppresses. Why it bites: when a low-confidence ranging BUY/SELL is suppressed, the code still writes `triggered_consensus[ticker] = action`; if confidence strengthens on later cycles without first reverting to HOLD, Layer 2 never sees the now-actionable consensus. Fix: keep suppressed signals in a pending state and only advance the consensus baseline after a real trigger or a reset to HOLD.
- portfolio/claude_gate.py:87 — The central Claude gate still uses raw `open()/json.load()` and fails open on read errors. Why it bites: concurrent atomic config writes or a transient symlink/read failure return `True`, so Claude invocations keep running even when ops set `layer2.enabled=false`; this also violates the project’s “atomic I/O via file_utils only” rule. Fix: use `load_json()` and decide the disable gate fail-safe, not fail-open.

## P3 — Low
- portfolio/journal.py:28 — Journal context building still scans raw JSONL files with plain `open()`; `portfolio/journal_index.py:373` does the same for BM25 retrieval. Why it bites: concurrent append/rotation is treated as “skip malformed line,” so the newest memory can disappear silently from `layer2_context.md` exactly when Layer 2 is calibrating off recent history. Fix: move these reads onto coordinated `file_utils` helpers or tail/index snapshots, not raw full-file scans.

## Tests missing
- A timeout test that proves no ticker worker threads survive `BUG-178` pool timeout handling and that concurrency never exceeds 8 after repeated slow cycles.
- A prewarmer test that holds the llama-server lock/server open and asserts prewarm returns within a short bounded budget instead of inheriting 300s/240s waits.
- A health test that shows `check_agent_silence()` does not advance on `skipped_busy`, `skipped_offhours`, `blocked_*`, or trigger-only cycles.
- A tier-state test that shows `last_full_review_time` changes only after a real T3 execution, not after a skipped/gated classification.
- A trigger test for the ranging dampener: weak BUY suppressed first, stronger BUY later without a HOLD reset should still fire once.
- A live-writer test that runs Layer 2 append activity concurrently with prune/rotation and proves `layer2_journal.jsonl` / `telegram_messages.jsonl` entries are not lost or misclassified.
