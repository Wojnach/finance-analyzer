# Scheduled Tasks Inventory

Generated 2026-06-11 from `scripts/win/install-*-task.ps1` (audit batch 12).
This is a documentary inventory, not an installed-state report — it lists what
each installer *would* create. To verify the live state run:

```powershell
Get-ScheduledTask -TaskName 'PF-*' | Select-Object TaskName, State
```

> Schedule/command columns are distilled from each installer's
> `New-ScheduledTaskTrigger` / action lines; re-grep the `.ps1` if an entry
> looks stale. Some installers carry extra repetition or boundary triggers not
> shown here.

| Task | Primary schedule | Runs | Installer |
|------|------------------|------|-----------|
| PF-DataLoop | At logon (+ auto-restart) | `pf-loop.bat` (main Layer 1 loop) | install-data-loop-task.ps1 |
| PF-LoopResume | At logon, retry every 5 min | `pf-loop.bat` (resume if down) | install-loop-resume-task.ps1 |
| PF-MetalsLoop | At logon | metals loop (`metals-loop.bat`) | install-metals-loop-task.ps1 |
| PF-CryptoLoop | At logon | crypto swing loop | install-crypto-loop-task.ps1 |
| PF-OilLoop | At logon | oil swing loop | install-oil-loop-task.ps1 |
| PF-MstrLoop | At logon | MSTR shadow loop | install-mstr-loop-task.ps1 |
| PF-Dashboard | At logon | `dashboard/app.py` (port 5055) | install-dashboard-task.ps1 |
| PF-GoldDigger | (see installer) | GoldDigger bot | install-golddigger-task.ps1 |
| PF-LogRotate | Hourly (repetition) | `portfolio/log_rotation.py` | install-log-rotate-task.ps1 |
| PF-FixAgentDispatcher | Every 10 min (repetition) | `scripts/fix_agent_dispatcher.py` | install-fix-agent-task.ps1 |
| PF-PendingPickups | Daily | `scripts/process_pending_pickups.py` | install-pending-pickups-task.ps1 |
| PF-LoopHealthDaily | Daily 08:00 | loop health summary | install-loop-health-daily-task.ps1 |
| PF-LoopHealthReport-20260515 | Daily 18:00 | `scripts/loop_health_report.py` | install-loop-health-report-task.ps1 |
| PF-LoopHealthWatchdog | At logon, every 30 min | `scripts/loop_health_watchdog.py` | install-loop-health-watchdog-task.ps1 |
| PF-ShadowReview | Daily 03:30 | `scripts/win/shadow-review.bat` | install-shadow-review-task.ps1 |
| PF-MetaLearnerRetrain | Daily ~18:00 | meta-learner retrain | install-meta-learner-task.ps1 |
| PF-LLMBackfill | Daily (repetition) | `scripts/win/pf-llm-backfill.bat` | install-llm-backfill-task.ps1 |
| PF-LocalLlmReport | Daily | `pf-local-llm-report.bat` | install-local-llm-report-task.ps1 |
| PF-Prophecy | Daily 10:00 | `scripts/prophecy-daily.bat` | install-prophecy-task.ps1 |
| PF-ClaudeUpdate | Weekly | Claude CLI self-update | install-claude-update-task.ps1 |

## Disabled on purpose — do NOT re-enable

These installers are tracked and runnable but the tasks are **deliberately
disabled** (see `scripts/win/RC_DISABLED_DO_NOT_REENABLE.md`, and the tracked
`DISABLED` headers in `after-hours-research.bat` / `signal-research.bat`):

| Task | Why off | Installer |
|------|---------|-----------|
| PF-RCKeepalive | RC session churn — clutters claude.ai sidebar | install-rc-keepalive-task.ps1 |
| PF-RC-Watchdog | RC backstop, off with the group | install-rc-watchdog-task.ps1 |
| (PF-RemoteControl*) | RC servers | install-rc-server-task.ps1 |
| PF-AfterHoursResearch | Claude-spend task, frozen 2026-06-05 | install-research-task.ps1 |
| PF-SignalResearch | Claude-spend task, frozen 2026-06-05 | install-signal-research-task.ps1 |
| PF-AdversarialReview | Claude-spend review task | install-adversarial-review-task.ps1 |

## Notes

- Task names are case-inconsistent across docs (`PF-MstrLoop` vs `PF-MSTRLoop`,
  `PF-MetalsLoop` vs metals fast-tick): the installer var is authoritative.
- `PF-FixAgentDispatcher` is documented in `CLAUDE.md` but the 2026-06-10 audit
  found it is NOT installed in production.
- There is no machine-readable manifest yet (audit finding B12); this file is
  the closest thing. A `verify` driver comparing `Get-ScheduledTask` against a
  manifest is a proposed follow-up.
