# ⛔ RC remote-control servers are DISABLED ON PURPOSE — do not re-enable

**Status:** Disabled 2026-06-01. Previously disabled before that and got
re-enabled by accident. This file is the guard that was missing.

## What this is

Three Claude Code "remote-control" servers — named **Trading**, **Development**,
**Research** — used to run continuously so they stayed visible in the
claude.ai/code session picker. They were driven by Windows Scheduled Tasks:

| Task | Role |
|------|------|
| `PF-RemoteControl` | Launch all 3 RC servers on logon (`rc-server-ensure.ps1`) |
| `PF-RemoteControl-Wake` | Relaunch on wake-from-sleep |
| `PF-RCKeepalive` | Every 5 min: kill idle servers so the `.bat` loop restarts and refreshes Anthropic's ~20-min registration TTL (`rc-keepalive.ps1`) |
| `PF-RC-Watchdog` | Backstop liveness check |

## Why it's off

The keepalive **recycles the servers every few minutes** (staggered 13/15/17 min
idle thresholds). Each recycle relaunches `claude`, which creates a **brand-new
session entry** in the claude.ai/code sidebar. Result: dozens of
Trading/Development/Research sessions pile up under "Today" every day, plus
constant background `claude.exe` churn. Not wanted.

## How it gets accidentally re-enabled

These installer scripts are still present and will RE-CREATE + ENABLE the tasks
if run:

- `scripts/win/install-rc-keepalive-task.ps1`
- `scripts/win/install-research-task.ps1`
- `scripts/win/install-signal-research-task.ps1`
- any aggregate setup that invokes the above
- `scripts/win/rc-server-ensure.ps1` will relaunch the 3 servers if run directly

**Do not run them.** If a setup script calls them, comment those lines out.

## To confirm it's still off

```powershell
Get-ScheduledTask -TaskName 'PF-RemoteControl','PF-RemoteControl-Wake','PF-RCKeepalive','PF-RC-Watchdog |
  Select-Object TaskName, State    # all should read Disabled
```

## To intentionally turn it back on (only if you really mean it)

```powershell
schtasks /change /tn PF-RemoteControl      /enable
schtasks /change /tn PF-RemoteControl-Wake /enable
schtasks /change /tn PF-RCKeepalive        /enable
schtasks /change /tn PF-RC-Watchdog        /enable
```
