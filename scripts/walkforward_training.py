"""Walk-Forward Model Validation Entry Script (Legacy alias delegating to models.train).

See models.train and models.walk_forward for full implementation.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.train import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy alias to models/train.py")
    parser.add_argument("--db-path", "--db", type=str, default="data/arbsense.db", dest="db")
    parser.add_argument("--dataset-version", type=str, default=None)
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--db", args.db]
    if args.dataset_version:
        sys.argv.extend(["--dataset-version", args.dataset_version])
    train_main()
