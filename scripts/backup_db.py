"""Database Backup Utility with Retention Management.

Supports:
1. SQLite WAL Engine: Performs atomic online backup using VACUUM INTO / WAL checkpointing,
   preserving database integrity along with -wal and -shm files.
2. TimescaleDB / PostgreSQL: Executes pg_dump with custom compression.
3. Retention Policy: Prunes old backup archives exceeding configured retention period (default 7 days).
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BackupService]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BackupDB")


def backup_sqlite(db_path: Path, backup_dir: Path) -> Path:
    """Safely backs up SQLite database using Python sqlite3's online backup API."""
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database file not found at: {db_path}")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_backup_file = backup_dir / f"arbsense_{timestamp}.db"

    logger.info("Initiating SQLite online backup via sqlite3.backup API from %s to %s...", db_path, target_backup_file)

    src_conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(str(target_backup_file))
    with dst_conn:
        src_conn.backup(dst_conn, pages=100)
    src_conn.close()

    # Query and print event count in the copy
    cur = dst_conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM spread_events")
        event_cnt = cur.fetchone()[0]
    except Exception:
        event_cnt = 0
    try:
        cur.execute("SELECT COUNT(*) FROM orderbook_ticks")
        tick_cnt = cur.fetchone()[0]
    except Exception:
        tick_cnt = 0
    dst_conn.close()

    size_bytes = target_backup_file.stat().st_size
    print(f"[BackupSuccess] Created: {target_backup_file.name} ({size_bytes} bytes) | Events in backup: {event_cnt} | Ticks in backup: {tick_cnt}")
    logger.info("Backup complete: %s | Events: %d | Ticks: %d", target_backup_file.name, event_cnt, tick_cnt)
    return target_backup_file


def backup_timescaledb(db_url: str, backup_dir: Path) -> Path:
    """Executes pg_dump for TimescaleDB / PostgreSQL database."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dump_file = backup_dir / f"arbsense_timescale_{timestamp}.dump"

    logger.info("Initiating TimescaleDB / Postgres backup using pg_dump to %s...", target_dump_file)
    cmd = [
        "pg_dump",
        "--format=c",
        "--file", str(target_dump_file),
        db_url,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pg_dump failed with return code {res.returncode}: {res.stderr}")

    logger.info("pg_dump completed: %s (%d bytes)", target_dump_file.name, target_dump_file.stat().st_size)
    return target_dump_file


def prune_old_backups(backup_dir: Path, retention_days: int) -> int:
    """Deletes backup files or directories older than retention_days."""
    if retention_days <= 0:
        return 0

    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=retention_days)
    pruned_count = 0

    logger.info("Checking backup retention (threshold: older than %d days / before %s)...", retention_days, cutoff_time.isoformat())

    for item in backup_dir.iterdir():
        if item.name.startswith("arbsense_"):
            mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime, tz=datetime.timezone.utc)
            if mtime < cutoff_time:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    logger.info("Pruned expired backup: %s (modified %s)", item.name, mtime.strftime("%Y-%m-%d"))
                    pruned_count += 1
                except Exception as e:
                    logger.warning("Could not delete expired backup %s: %s", item.name, e)

    return pruned_count


def main():
    parser = argparse.ArgumentParser(description="ArbSense Automated Database Backup & Retention Tool")
    parser.add_argument("--sqlite-path", default="data/arbsense.db", help="Path to SQLite database")
    parser.add_argument("--backup-dir", default="backups", help="Target backup directory")
    parser.add_argument("--retention-days", type=int, default=7, help="Retention period in days (default: 7)")
    parser.add_argument("--force-postgres", action="store_true", help="Force pg_dump using DATABASE_URL")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    if not backup_dir.is_absolute():
        backup_dir = PROJECT_ROOT / backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)

    db_url = os.getenv("DATABASE_URL")
    performed_backup = False

    # 1. TimescaleDB / Postgres backup if forced or configured
    if args.force_postgres and db_url:
        try:
            backup_timescaledb(db_url, backup_dir)
            performed_backup = True
        except Exception as e:
            logger.error("TimescaleDB backup failed: %s", e)
            sys.exit(1)

    # 2. SQLite backup if sqlite file exists or default
    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = PROJECT_ROOT / sqlite_path

    if sqlite_path.exists():
        try:
            backup_sqlite(sqlite_path, backup_dir)
            performed_backup = True
        except Exception as e:
            logger.error("SQLite backup failed: %s", e)
            sys.exit(1)
    elif not performed_backup:
        logger.error("No database found to backup at %s and no active Postgres connection.", sqlite_path)
        sys.exit(1)

    # 3. Enforce retention policy
    pruned = prune_old_backups(backup_dir, args.retention_days)
    logger.info("Backup workflow finished successfully (Backups pruned: %d).", pruned)


if __name__ == "__main__":
    main()
