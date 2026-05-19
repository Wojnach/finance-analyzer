"""Lightweight system monitor — writes rolling snapshots to a log file.
Run in background: python scripts/sysmon.py
Read latest: tail data/sysmon.log
"""
import subprocess
import time
import json
import os
import sys
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sysmon.log")
INTERVAL = 5  # seconds
MAX_LINES = 500  # rolling window

def get_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_pct": int(parts[0]),
            "temp_c": int(parts[1]),
            "power_w": float(parts[2]),
            "vram_used_mb": int(parts[3]),
            "vram_total_mb": int(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}

def get_gpu_procs():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            timeout=5, text=True
        )
        procs = []
        for line in out.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                sm = parts[3]
                if sm != "-" and int(sm) > 0:
                    procs.append({"pid": int(parts[1]), "sm_pct": int(sm), "name": parts[-1]})
        return procs
    except Exception:
        return []

def get_cpu_top(duration=2):
    """Measure actual CPU usage over `duration` seconds via PowerShell."""
    ps_script = f"""
$s1 = @{{}}
Get-Process | ForEach-Object {{ $s1[$_.Id] = $_.CPU }}
Start-Sleep -Seconds {duration}
Get-Process | ForEach-Object {{
    $prev = $s1[$_.Id]
    if ($prev -ne $null) {{
        $delta = $_.CPU - $prev
        if ($delta -gt 0.05) {{
            $ram = [math]::Round($_.WorkingSet64/1MB, 0)
            Write-Output "$($_.Id)|$($_.ProcessName)|$([math]::Round($delta,3))|$ram"
        }}
    }}
}}
"""
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            timeout=duration + 10, text=True
        ).strip()
        procs = []
        for line in out.split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                cpu_s = float(parts[2])
                pct = round(cpu_s / duration * 100, 1)
                procs.append({
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "cpu_pct_core": pct,
                    "ram_mb": int(parts[3]),
                })
        procs.sort(key=lambda x: x["cpu_pct_core"], reverse=True)
        return procs[:15]
    except Exception as e:
        return [{"error": str(e)}]

def write_snapshot(snapshot):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # Append
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")
    # Trim to rolling window
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_LINES:])
    except Exception:
        pass

def format_snapshot(snap):
    """Human-readable format for the log."""
    ts = snap["timestamp"]
    gpu = snap.get("gpu", {})
    lines = [f"\n=== {ts} ==="]
    if "error" not in gpu:
        lines.append(f"GPU: {gpu['gpu_pct']}% SM | {gpu['temp_c']}C | {gpu['power_w']}W | {gpu['vram_used_mb']}/{gpu['vram_total_mb']}MB VRAM")
    gpu_procs = snap.get("gpu_procs", [])
    if gpu_procs:
        lines.append("GPU active: " + ", ".join(f"{p['name']}(PID {p['pid']}) {p['sm_pct']}%SM" for p in gpu_procs))

    cpu_top = snap.get("cpu_top", [])
    if cpu_top and "error" not in cpu_top[0]:
        lines.append("CPU top:")
        for p in cpu_top[:10]:
            lines.append(f"  {p['pid']:>6}  {p['name']:<25} {p['cpu_pct_core']:>6.1f}% core  {p['ram_mb']:>5}MB")
    return "\n".join(lines)

def main():
    print(f"sysmon: logging to {os.path.abspath(LOG_PATH)} every {INTERVAL}s", flush=True)
    # Write readable log instead of JSON for easy tailing
    readable_log = LOG_PATH.replace(".log", "_readable.log")

    while True:
        try:
            snap = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": get_gpu(),
                "gpu_procs": get_gpu_procs(),
                "cpu_top": get_cpu_top(duration=3),
            }
            write_snapshot(snap)
            # Also write readable
            readable = format_snapshot(snap)
            with open(readable_log, "a", encoding="utf-8") as f:
                f.write(readable + "\n")
            # Trim readable too
            try:
                with open(readable_log, "r", encoding="utf-8") as f:
                    rlines = f.readlines()
                if len(rlines) > 2000:
                    with open(readable_log, "w", encoding="utf-8") as f:
                        f.writelines(rlines[-1500:])
            except Exception:
                pass

            print(readable, flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"sysmon error: {e}", flush=True)

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
