-- Video catalog schema (SQLite).
-- Raw video bytes live in data/raw/ (tracked by DVC); this database only
-- stores metadata. A video's identity is the SHA-256 of its file bytes.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,              -- e.g. 'P01' (pseudonymous)
    display_name  TEXT,
    handedness    TEXT CHECK (handedness IN ('right', 'left') OR handedness IS NULL),
    level         TEXT,                          -- beginner / intermediate / advanced
    consent_given INTEGER NOT NULL DEFAULT 0,    -- 1 = signed consent on file
    consent_date  TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,              -- '<YYYY-MM-DD>_<player_id>'
    player_id     TEXT NOT NULL REFERENCES players(player_id),
    session_date  TEXT NOT NULL,
    location      TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS videos (
    sha256            TEXT PRIMARY KEY CHECK (length(sha256) = 64),
    storage_path      TEXT NOT NULL UNIQUE,      -- repo-relative, e.g. data/raw/sha256/ab/<hash>.mov
    original_filename TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    container         TEXT,
    codec             TEXT,
    width             INTEGER,
    height            INTEGER,
    fps               REAL,
    duration_s        REAL,
    recorded_at       TEXT,                      -- from file metadata when available
    session_id        TEXT NOT NULL REFERENCES sessions(session_id),
    stroke_type       TEXT NOT NULL,             -- e.g. forehand_drive
    camera_angle      TEXT NOT NULL CHECK (camera_angle IN ('side', 'front', 'diagonal')),
    take_id           TEXT,                      -- groups clips filmed simultaneously from different cameras
    qa_warnings       TEXT,                      -- '; '-separated protocol violations
    notes             TEXT,
    ingested_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_sha256   TEXT NOT NULL REFERENCES videos(sha256),
    annotator      TEXT NOT NULL,                -- e.g. 'coach_01'
    rating_1to5    INTEGER CHECK (rating_1to5 BETWEEN 1 AND 5),
    flagged_issue  TEXT,
    notes          TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_videos_session ON videos(session_id);
CREATE INDEX IF NOT EXISTS idx_videos_stroke  ON videos(stroke_type);
CREATE INDEX IF NOT EXISTS idx_videos_take    ON videos(take_id);
CREATE INDEX IF NOT EXISTS idx_annotations_video ON annotations(video_sha256);
