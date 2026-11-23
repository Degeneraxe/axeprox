# ArbSense Watchdog for unattended long-run data collection
# Automatically restarts scripts/run_live.py if it exits

$ProjectRoot = "C:\Users\SIDDHU\.gemini\antigravity\scratch\arbsense"
Set-Location $ProjectRoot

$PythonExe = "C:\Users\SIDDHU\AppData\Local\Programs\Python\Python313\python.exe"
$LogsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

$LogFile = Join-Path $LogsDir "run_live.log"
$WatchdogLog = Join-Path $LogsDir "watchdog.log"
$PidFile = Join-Path $LogsDir "watchdog.pid"

$PID | Out-File -FilePath $PidFile -Encoding ascii

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] [Watchdog] Watchdog service started (PID $PID)" | Out-File -Append -FilePath $WatchdogLog -Encoding ascii

$env:PYTHONUNBUFFERED = "1"

while ($true) {
    $ts_start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts_start] [Watchdog] Starting python scripts/run_live.py..." | Out-File -Append -FilePath $WatchdogLog -Encoding ascii
    "[$ts_start] [Watchdog] === RUN_LIVE SESSION STARTED ===" | Out-File -Append -FilePath $LogFile -Encoding ascii

    # Run unbuffered via cmd.exe redirection directly into log file
    cmd.exe /c "`"$PythonExe`" scripts\run_live.py >> `"$LogFile`" 2>&1"
    $exitCode = $LASTEXITCODE

    $ts_exit = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts_exit] [Watchdog] Process exited with code $exitCode. Restarting in 10s..." | Out-File -Append -FilePath $WatchdogLog -Encoding ascii
    "[$ts_exit] [Watchdog] === PROCESS EXITED (Code $exitCode) - RESTARTING IN 10s ===" | Out-File -Append -FilePath $LogFile -Encoding ascii

    Start-Sleep -Seconds 10
}
