"""
ingest_video.py

Add new source videos to the project's raw-data store and catalog.

For each video it:
    1. computes the SHA-256 of the file bytes (the video's permanent ID)
    2. skips it if that hash is already in the catalog (duplicate)
    3. reads technical metadata with ffprobe (codec, resolution, fps, ...)
    4. checks it against the recording protocol and records any warnings
    5. copies it to data/raw/sha256/<first 2 hex>/<hash>.<ext>
    6. re-hashes the copy to confirm it is byte-identical
    7. inserts a row into data/catalog.db
    8. removes the original from the inbox (unless --keep)

Afterwards, version and upload with DVC (see README, "Data management").

Usage:
    python src/ingest_video.py data/inbox/ --player P01 --session-date 2026-09-14 \
        --stroke forehand_drive --angle side --consent
    python src/ingest_video.py data/inbox/clip.mov --player P02 ... --keep
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "sha256"
DB_PATH = REPO_ROOT / "data" / "catalog.db"
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
CAMERA_ANGLES = ("side", "front", "diagonal")

# Recording-protocol thresholds (see README, "Recording guidance")
MIN_FPS = 60
MIN_DURATION_S = 1.0


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file, streamed in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> dict:
    """Read technical metadata of the first video stream with ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration:format=format_name,duration:format_tags",
        "-of", "json", str(path),
    ]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = out["streams"][0] if out.get("streams") else {}
    fmt = out.get("format", {})
    tags = fmt.get("tags", {})

    num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else None
    duration = stream.get("duration") or fmt.get("duration")

    return {
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": round(fps, 3) if fps else None,
        "duration_s": round(float(duration), 3) if duration else None,
        "recorded_at": tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"),
    }


def qa_checks(meta: dict) -> list[str]:
    """Return a list of recording-protocol violations (empty = all good)."""
    warnings = []
    if meta["fps"] is None or meta["fps"] < MIN_FPS:
        warnings.append(f"fps {meta['fps']} < {MIN_FPS}")
    if meta["duration_s"] is None or meta["duration_s"] < MIN_DURATION_S:
        warnings.append(f"duration {meta['duration_s']}s < {MIN_DURATION_S}s")
    if meta["recorded_at"] is None:
        warnings.append("no recording timestamp (file may have been re-encoded)")
    return warnings


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def ensure_player_and_session(conn, player_id, session_id, session_date, consent):
    conn.execute(
        "INSERT OR IGNORE INTO players (player_id, consent_given, consent_date) VALUES (?, ?, ?)",
        (player_id, int(consent), session_date if consent else None),
    )
    if consent:
        conn.execute(
            "UPDATE players SET consent_given = 1, consent_date = COALESCE(consent_date, ?) WHERE player_id = ?",
            (session_date, player_id),
        )
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, player_id, session_date) VALUES (?, ?, ?)",
        (session_id, player_id, session_date),
    )


def ingest_one(conn, src: Path, args, session_id: str) -> str:
    """Ingest a single file. Returns 'added', 'duplicate' or 'failed'."""
    digest = sha256_file(src)
    if conn.execute("SELECT 1 FROM videos WHERE sha256 = ?", (digest,)).fetchone():
        print(f"  [skip] {src.name}: duplicate of {digest[:12]}")
        return "duplicate"

    meta = probe_video(src)
    warnings = qa_checks(meta)

    dest = RAW_DIR / digest[:2] / f"{digest}{src.suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if sha256_file(dest) != digest:
        dest.unlink()
        print(f"  [FAIL] {src.name}: copy verification failed, nothing stored")
        return "failed"

    try:
        with conn:
            conn.execute(
                """INSERT INTO videos (sha256, storage_path, original_filename, size_bytes,
                       container, codec, width, height, fps, duration_s, recorded_at,
                       session_id, stroke_type, camera_angle, take_id, qa_warnings, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    digest, dest.relative_to(REPO_ROOT).as_posix(), src.name, src.stat().st_size,
                    meta["container"], meta["codec"], meta["width"], meta["height"],
                    meta["fps"], meta["duration_s"], meta["recorded_at"],
                    session_id, args.stroke, args.angle, args.take,
                    "; ".join(warnings) or None, args.notes,
                ),
            )
    except sqlite3.Error:
        dest.unlink()
        raise

    if not args.keep:
        src.unlink()

    status = f"{len(warnings)} QA warning(s): " + "; ".join(warnings) if warnings else "QA ok"
    print(f"  [add]  {src.name} -> {digest[:12]}  ({meta['width']}x{meta['height']} "
          f"@ {meta['fps']} fps, {meta['duration_s']}s)  {status}")
    return "added"


def collect_videos(paths) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(p)
        else:
            print(f"  [warn] ignoring non-video path: {p}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Ingest source videos into the raw store and catalog.")
    parser.add_argument("paths", nargs="+", help="Video files and/or folders (e.g. data/inbox/)")
    parser.add_argument("--player", required=True, help="Pseudonymous player ID, e.g. P01")
    parser.add_argument("--session-date", required=True, help="Recording date, YYYY-MM-DD")
    parser.add_argument("--stroke", required=True, help="Stroke type, e.g. forehand_drive")
    parser.add_argument("--angle", required=True, choices=CAMERA_ANGLES, help="Camera angle")
    parser.add_argument("--take", default=None,
                        help="Take ID shared by clips filmed simultaneously from different angles, e.g. T01")
    parser.add_argument("--consent", action="store_true", help="Player has given signed consent")
    parser.add_argument("--notes", default=None, help="Free-text note stored on every ingested video")
    parser.add_argument("--keep", action="store_true", help="Do not delete originals after ingest")
    args = parser.parse_args()

    files = collect_videos(args.paths)
    if not files:
        sys.exit("No video files found.")

    session_id = f"{args.session_date}_{args.player}"
    conn = connect_db()
    with conn:
        ensure_player_and_session(conn, args.player, session_id, args.session_date, args.consent)

    print(f"Ingesting {len(files)} file(s) into session {session_id}")
    results = [ingest_one(conn, f, args, session_id) for f in files]
    conn.close()

    print(f"\nDone: {results.count('added')} added, {results.count('duplicate')} duplicate(s), "
          f"{results.count('failed')} failed.")
    if results.count("added"):
        print("Next: dvc add data/raw data/catalog.db  ->  git commit  ->  dvc push  ->  git push")


if __name__ == "__main__":
    main()
