"""
verify_raw.py

Integrity check for the raw-video store. Confirms that:
    - every video in data/catalog.db exists on disk and its bytes still
      hash to the recorded SHA-256 (nothing was altered or corrupted)
    - every file under data/raw/ is registered in the catalog (no orphans)

Run it after `dvc pull`, before an analysis run, or on a schedule.
Exits with status 1 if any problem is found.

Usage:
    python src/verify_raw.py
"""

import sqlite3
import sys

from ingest_video import DB_PATH, RAW_DIR, REPO_ROOT, sha256_file


def main():
    if not DB_PATH.exists():
        sys.exit(f"Catalog not found: {DB_PATH} (run `dvc pull` first?)")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT sha256, storage_path FROM videos").fetchall()
    conn.close()

    problems = 0
    registered = set()
    for digest, rel_path in rows:
        path = REPO_ROOT / rel_path
        registered.add(path.resolve())
        if not path.exists():
            print(f"  [MISSING]  {rel_path}")
            problems += 1
        elif sha256_file(path) != digest:
            print(f"  [ALTERED]  {rel_path}")
            problems += 1

    on_disk = {p.resolve() for p in RAW_DIR.rglob("*") if p.is_file()} if RAW_DIR.exists() else set()
    for orphan in sorted(on_disk - registered):
        print(f"  [ORPHAN]   {orphan.relative_to(REPO_ROOT).as_posix()}")
        problems += 1

    print(f"Checked {len(rows)} catalogued video(s): {problems} problem(s).")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
