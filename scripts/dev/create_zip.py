import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'backups', 'runs', '.venv', 'venv'}
exclude_exts = {'.pyc', '.pyo'}

target_zip = PROJECT_ROOT / "arbsense_cloud.zip"

with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        for f in files:
            if any(f.endswith(ext) for ext in exclude_exts):
                continue
            if f.endswith('.zip'):
                continue
            if f.endswith('.db') and f != 'arbsense.db':
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            zf.write(full_path, rel_path)

print(f"Archive created at: {target_zip} ({round(target_zip.stat().st_size / (1024*1024), 2)} MB)")
