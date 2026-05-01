<#
.SYNOPSIS
  Patch the cloudflared Windows service's ImagePath to include tunnel-run args.

.DESCRIPTION
  `cloudflared service install` registers the binary with NO arguments by default.
  Windows starts the bare exe, which exits immediately because cloudflared.exe
  with no args defaults to printing help. Result: SCM reports "process terminated
  unexpectedly" on every start, the tunnel never connects, the dashboard is
  unreachable from the public hostname.

  This script writes the correct ImagePath:
      "C:\Program Files (x86)\cloudflared\cloudflared.exe" --no-autoupdate tunnel run <TUNNEL_NAME>

  Idempotent — safe to re-apply on every reinstall. Must run elevated (admin).
  Discovered 2026-04-30 during initial deployment; not documented in
  Cloudflare's official Windows service guide.

.PARAMETER TunnelName
  The cloudflared tunnel name (e.g. "finance-dashboard"). Required.

.PARAMETER CloudflaredPath
  Full path to cloudflared.exe. Defaults to the standard winget install path.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelName,

    [string]$CloudflaredPath = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
)

$registryPath = 'HKLM:\System\CurrentControlSet\Services\cloudflared'

if (-not (Test-Path $registryPath)) {
    Write-Error "Service registry key not found at $registryPath. Run 'cloudflared service install' first."
    exit 1
}

# Build the value with single-quoted PS string so embedded quotes/parens are
# literal — avoids the `(x86)` sub-expression trap that bites bash/double-
# quoted PS commands.
$value = '"' + $CloudflaredPath + '" --no-autoupdate tunnel run ' + $TunnelName

Set-ItemProperty -Path $registryPath -Name 'ImagePath' -Value $value

$applied = (Get-ItemProperty -Path $registryPath -Name 'ImagePath').ImagePath
Write-Output "ImagePath set to: $applied"
