# Kill existing loop processes and restart via Task Scheduler
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*main.py*' -and $_.CommandLine -notlike '*--report*' -and $_.Name -eq 'python.exe' }
foreach ($p in $procs) {
    Write-Host "Killing loop pid=$($p.ProcessId): $($p.CommandLine)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
# Restart via scheduled task
schtasks /Run /TN "PF-DataLoop"
Write-Host "PF-DataLoop restarted"
