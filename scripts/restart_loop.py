"""Kill crash-looping main loop and restart fresh."""
import os
import subprocess
import time

print(f"[{time.strftime('%H:%M:%S')}] Restarting data loop...")

env = os.environ.copy()
env.pop("CLAUDECODE", None)
env.pop("CLAUDE_CODE_ENTRYPOINT", None)
# Force PYTHONPATH so portfolio package is importable regardless of CWD
env["PYTHONPATH"] = r"Q:\finance-analyzer"

log = open(r"Q:\finance-analyzer\data\loop_out.txt", "a")
log.write(f"\n=== RESTART {time.strftime('%Y-%m-%d %H:%M:%S')} (PYTHONPATH set) ===\n")
log.flush()

proc = subprocess.Popen(
    [r"Q:\finance-analyzer\.venv\Scripts\python.exe", "-u", r"Q:\finance-analyzer\portfolio\main.py", "--loop"],
    cwd=r"Q:\finance-analyzer",
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    creationflags=0x00000008 | 0x00000200,
)
print(f"  PID {proc.pid}")
