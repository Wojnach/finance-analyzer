$cfPath = "C:\Program Files (x86)\cloudflared"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -notlike "*cloudflared*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$cfPath", "Machine")
    Write-Output "Added $cfPath to system PATH"
} else {
    Write-Output "cloudflared already in system PATH"
}
