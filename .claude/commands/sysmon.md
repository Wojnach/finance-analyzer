System monitor — quick CPU, RAM, GPU check.

Run: `powershell.exe -NoProfile -Command "$cpu = (Get-CimInstance Win32_Processor).LoadPercentage; $os = Get-CimInstance Win32_OperatingSystem; $ramUsed = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory)/1MB, 1); $ramTotal = [math]::Round($os.TotalVisibleMemorySize/1MB, 1); $ramPct = [math]::Round($ramUsed/$ramTotal*100); $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits; Write-Host \"CPU: ${cpu}% | RAM: ${ramUsed}/${ramTotal}GB (${ramPct}%) | GPU: $gpu\""`

Format the GPU output as: `GPU: {util}% | VRAM: {used}/{total}MB | Temp: {temp}C`

Show as a single compact table. Flag anything abnormal:
- CPU > 80% sustained
- RAM > 80%
- GPU > 95% for more than a few seconds (normal during LLM inference bursts)
- GPU temp > 80C
