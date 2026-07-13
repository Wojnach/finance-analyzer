# Install PF-LlamaRemote: static llama-server serving phi4_mini on :8788
# for the Deck main loop over Tailscale (Option A, 2026-07-13).
# Run as admin (firewall rule). Idempotent.
#
# Port 8788 deliberately != 8787 so llm backtests / local rotation on this
# machine keep working alongside. VRAM: phi4 Q4 ~2.4 GB + one swapped 8B
# fits the 3080's 10 GB.
$model = "Q:\models\phi4-mini-reasoning-gguf\microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf"
$bin = "Q:\models\llama-cpp-bin\llama-server.exe"
$deckIp = "100.75.67.98"

New-NetFirewallRule -DisplayName "llama-remote-8788-deck-only" `
    -Direction Inbound -Protocol TCP -LocalPort 8788 `
    -RemoteAddress $deckIp -Action Allow -ErrorAction SilentlyContinue | Out-Null

$action = New-ScheduledTaskAction -Execute $bin -Argument `
    "-m `"$model`" --host 0.0.0.0 --port 8788 -ngl 99 -c 4096 --no-webui"
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "PF-LlamaRemote" -Action $action `
    -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "PF-LlamaRemote"
Write-Host "PF-LlamaRemote installed + started (phi4_mini on :8788, firewall scoped to $deckIp)"
