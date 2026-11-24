"""STEP 0: Data & Diagnostics Snapshotting Service.

Performs:
1. Morning report execution and capture to review_evidence/00_morning_diagnostics_<YYYYMMDD_HHMM>.txt
2. Run metadata and last 5 [STATUS 60s] lines extraction to review_evidence/00_run_metadata_<YYYYMMDD_HHMM>.txt
3. Creation of review_evidence/data_snapshot_<YYYYMMDD_HHMM>/ with backup .db, run_live.log, and MANIFEST.txt
4. External copy to Documents/ArbSense_backups/data_snapshot_<YYYYMMDD_HHMM>/ and SHA-256 hash verification
"""
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIMESTAMP_TAG = "20261123_1003"
BACKUP_DB_SRC = PROJECT_ROOT / "backups" / "arbsense_20261123_043050.db"
LIVE_LOG_SRC = PROJECT_ROOT / "logs" / "run_live.log"

REVIEW_DIR = PROJECT_ROOT / "review_evidence"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTICS_FILE = REVIEW_DIR / f"00_morning_diagnostics_{TIMESTAMP_TAG}.txt"
RUN_METADATA_FILE = REVIEW_DIR / f"00_run_metadata_{TIMESTAMP_TAG}.txt"
SNAPSHOT_DIR = REVIEW_DIR / f"data_snapshot_{TIMESTAMP_TAG}"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_ROOT = Path("C:/Users/SIDDHU/Documents/ArbSense_backups")
EXTERNAL_SNAPSHOT_DIR = EXTERNAL_ROOT / f"data_snapshot_{TIMESTAMP_TAG}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("EXECUTING STEP 0: DATA AND DIAGNOSTICS SNAPSHOT")
    print(f"Timestamp Tag:        {TIMESTAMP_TAG}")
    print(f"Backup Source DB:     {BACKUP_DB_SRC}")
    print(f"Live Log Source:      {LIVE_LOG_SRC}")
    print("=" * 80)

    # 1. Run morning_report.py and save full output
    print("\n[*] Running scripts/morning_report.py...")
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "morning_report.py")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    morning_report_text = res.stdout
    with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
        f.write(morning_report_text)
    print(f"[+] Saved morning diagnostics to: {DIAGNOSTICS_FILE}")

    # 2. Extract last 5 [STATUS 60s] lines from logs/run_live.log
    print("\n[*] Extracting last 5 [STATUS 60s] lines from logs/run_live.log...")
    status_lines = []
    if LIVE_LOG_SRC.exists():
        with open(LIVE_LOG_SRC, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "[STATUS 60s]" in line:
                    status_lines.append(line.strip())
        status_lines = status_lines[-5:]

    # Extract run_metadata rows from backup DB
    print("[*] Extracting run_metadata rows from backup DB...")
    conn = sqlite3.connect(f"file:{BACKUP_DB_SRC.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT run_id, dataset_version, config_hash, git_commit, latency_threshold_ms, created_at, config_json FROM run_metadata ORDER BY run_id ASC;")
    metadata_rows = cur.fetchall()
    conn.close()

    with open(RUN_METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ARBSENSE RUN METADATA & LIVE STATUS LINES\n")
        f.write(f"Snapshot Time: {TIMESTAMP_TAG}\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- LAST 5 [STATUS 60s] LINES FROM logs/run_live.log ---\n")
        for line in status_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("--- RUN METADATA ROWS FROM DATABASE ---\n")
        for r in metadata_rows:
            f.write(f"Run ID:                {r['run_id']}\n")
            f.write(f"Dataset Version:       {r['dataset_version']}\n")
            f.write(f"Config Hash:           {r['config_hash']}\n")
            f.write(f"Git Commit:            {r['git_commit']}\n")
            f.write(f"Latency Threshold:     {r['latency_threshold_ms']} ms\n")
            f.write(f"Created At:            {r['created_at']}\n")
            f.write(f"Canonical Config JSON: {r['config_json']}\n")
            f.write("-" * 80 + "\n")
    print(f"[+] Saved run metadata and status lines to: {RUN_METADATA_FILE}")

    # 3. Create snapshot folder and copy files
    print(f"\n[*] Populating snapshot directory: {SNAPSHOT_DIR}...")
    dest_db = SNAPSHOT_DIR / BACKUP_DB_SRC.name
    dest_log = SNAPSHOT_DIR / "run_live.log"

    print(f"    - Copying backup DB -> {dest_db}...")
    shutil.copy2(BACKUP_DB_SRC, dest_db)

    print(f"    - Copying live log -> {dest_log}...")
    shutil.copy2(LIVE_LOG_SRC, dest_log)

    # Compute hashes and create MANIFEST.txt
    print("[*] Computing SHA-256 hashes for manifest...")
    db_size = dest_db.stat().st_size
    db_hash = sha256_file(dest_db)

    log_size = dest_log.stat().st_size
    log_hash = sha256_file(dest_log)

    git_commit = "d4f164e71311069f4a9ce02c370886578b66e1b8"
    dataset_version = "v2.0"

    manifest_file = SNAPSHOT_DIR / "MANIFEST.txt"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ARBSENSE DATA SNAPSHOT MANIFEST\n")
        f.write(f"Snapshot Timestamp:  {TIMESTAMP_TAG}\n")
        f.write(f"Git Commit:          {git_commit}\n")
        f.write(f"Dataset Version:     {dataset_version}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"File:    {dest_db.name}\n")
        f.write(f"Size:    {db_size} bytes ({db_size / (1024*1024):.2f} MB)\n")
        f.write(f"SHA-256: {db_hash}\n\n")
        f.write(f"File:    {dest_log.name}\n")
        f.write(f"Size:    {log_size} bytes ({log_size / (1024*1024):.2f} MB)\n")
        f.write(f"SHA-256: {log_hash}\n")
    print(f"[+] Created MANIFEST.txt at: {manifest_file}")

    manifest_hash = sha256_file(manifest_file)

    # 4. Copy whole snapshot folder to external backup location
    print(f"\n[*] Copying snapshot folder to external backup destination: {EXTERNAL_SNAPSHOT_DIR}...")
    EXTERNAL_SNAPSHOT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if EXTERNAL_SNAPSHOT_DIR.exists():
        shutil.rmtree(EXTERNAL_SNAPSHOT_DIR)
    shutil.copytree(SNAPSHOT_DIR, EXTERNAL_SNAPSHOT_DIR)

    # Verify hashes match after copying
    print("[*] Verifying file integrity and hashes in external destination...")
    ext_db = EXTERNAL_SNAPSHOT_DIR / dest_db.name
    ext_log = EXTERNAL_SNAPSHOT_DIR / "run_live.log"
    ext_manifest = EXTERNAL_SNAPSHOT_DIR / "MANIFEST.txt"

    ext_db_hash = sha256_file(ext_db)
    ext_log_hash = sha256_file(ext_log)
    ext_manifest_hash = sha256_file(ext_manifest)

    assert ext_db_hash == db_hash, f"Hash mismatch for {dest_db.name}!"
    assert ext_log_hash == log_hash, f"Hash mismatch for {dest_log.name}!"
    assert ext_manifest_hash == manifest_hash, f"Hash mismatch for {manifest_file.name}!"
    print("[+] All hashes match perfectly between source and external backup destination!")

    # 5. Print Final Summary
    print("\n" + "=" * 80)
    print("STEP 0 COMPLETION SUMMARY")
    print("=" * 80)
    print(f"1. Backup DB Source:              {BACKUP_DB_SRC}")
    print(f"2. Morning Diagnostics File:      {DIAGNOSTICS_FILE}")
    print(f"3. Run Metadata & Status File:    {RUN_METADATA_FILE}")
    print(f"4. Project Snapshot Directory:    {SNAPSHOT_DIR}")
    print(f"   - Database Copy:               {dest_db.name} ({db_size} bytes)")
    print(f"   - Database SHA-256:            {db_hash}")
    print(f"   - Log File Copy:               {dest_log.name} ({log_size} bytes)")
    print(f"   - Log SHA-256:                 {log_hash}")
    print(f"   - Manifest:                    {manifest_file.name}")
    print(f"5. External Snapshot Directory:   {EXTERNAL_SNAPSHOT_DIR}")
    print(f"   - External Verification:       VERIFIED OK (All SHA-256 hashes match)")
    print("=" * 80)


if __name__ == "__main__":
    main()
