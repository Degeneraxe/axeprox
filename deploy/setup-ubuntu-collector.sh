#!/usr/bin/env bash
# Run as the Ubuntu login user inside a clean checkout on Ubuntu 24.04.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: numpy==2.3.2 PyYAML==6.0.3 websockets==16.0
.venv/bin/python scripts/run_live.py --help
.venv/bin/python -c 'from config.settings import AppConfig; print("Configured pairs:", AppConfig.load().markets.supported_pairs)'
df -h .
