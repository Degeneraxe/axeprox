import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ps_script = root / "scripts" / "watchdog_run_live.ps1"

# 0x00000008 = DETACHED_PROCESS, 0x00000200 = CREATE_NEW_PROCESS_GROUP
DETACHED_FLAGS = 0x00000008 | 0x00000200

cmd = [
    "powershell.exe",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", str(ps_script),
]

proc = subprocess.Popen(
    cmd,
    cwd=str(root),
    creationflags=DETACHED_FLAGS,
    close_fds=True,
)

pid_file = root / "logs" / "watchdog.pid"
pid_file.parent.mkdir(parents=True, exist_ok=True)
pid_file.write_text(str(proc.pid), encoding="ascii")

print(f"WATCHDOG_PID={proc.pid}")
