# Table Tennis Vision Trainer — Research Starter (Phase 1-2)

Starter pipeline for the video-based stroke analysis research project:
record a stroke, extract body pose over time, and visualize joint metrics
to compare against a coach's assessment.

This covers **Phase 1 (data collection) and Phase 2 (pose extraction +
visualization)** from the roadmap. Phase 3 (rule-based / reference
comparison against coach ratings) builds directly on top of this.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the MediaPipe pose model (~30 MB, git-ignored) into `models/`:

```bash
curl -L -o models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Expected SHA-256: `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`

## Folder structure

```
Table-Tennis-Master/
├── data/
│   ├── inbox/            # drop new videos here (temporary, git-ignored)
│   ├── raw/              # immutable source videos, named by SHA-256 (DVC-tracked)
│   ├── catalog.db        # SQLite metadata catalog (DVC-tracked)
│   ├── pose_data/        # generated: pose CSVs + metric plots
│   ├── segments/         # generated: per-rep segments
│   ├── reference/        # generated: reference technique profiles
│   └── coach_notes/      # coach ratings per clip (see template.csv)
├── db/
│   └── schema.sql        # catalog schema: players, sessions, videos, annotations
├── src/
│   ├── ingest_video.py      # inbox -> hash, probe, QA, store, catalog
│   ├── verify_raw.py        # integrity check of data/raw vs. catalog
│   ├── pose_extraction.py   # video -> pose landmark CSV
│   ├── angles.py             # joint angle / metric calculations
│   └── visualize.py          # pose CSV -> metric plots + metric CSV
└── requirements.txt
```

## Data management

Source videos are never committed to git. Each part lives where it belongs:

| What | Where |
|---|---|
| Code, schema, pipeline config | Git / GitHub |
| Raw video bytes | `data/raw/`, versioned with [DVC](https://dvc.org), stored in the DVC remote |
| Video metadata (player, session, stroke, fps, QA, ...) | `data/catalog.db` (SQLite, schema in `db/schema.sql`) |

A video's identity is the **SHA-256 of its file bytes**. Files are stored as
`data/raw/sha256/<first 2 hex>/<hash>.<ext>` and are never edited — all
derived data (pose CSVs, segments, reports) is regenerated from them.

**Getting the data after cloning:**

```bash
dvc pull                    # downloads data/raw and data/catalog.db
python src/verify_raw.py    # confirms every file matches its hash
```

**Adding new videos:**

1. Copy the *original* files off the phone into `data/inbox/` (USB or a
   sync that keeps originals — messengers re-encode and destroy quality).
2. Ingest them:
   ```bash
   python src/ingest_video.py data/inbox/ --player P01 --session-date 2026-09-14 \
       --stroke forehand_drive --angle side --consent
   ```
   This hashes each file, skips duplicates, reads fps/resolution/duration
   with `ffprobe`, records protocol violations (e.g. fps < 60) in
   `qa_warnings`, copies the file into `data/raw/`, verifies the copy, adds
   it to the catalog, and removes it from the inbox.
3. Version and upload:
   ```bash
   dvc add data/raw data/catalog.db
   git commit -m "Add session 2026-09-14 (P01): 6 forehand clips"
   dvc push
   git push
   ```
4. Run `dvc status -c` — only once it reports everything in sync, delete
   the clips from the phone.

Requires `ffmpeg` (for `ffprobe`) on `PATH`. Players are identified only by
pseudonymous IDs; consent is recorded per player in the catalog.

## Recording guidance (Phase 1)

- **Camera angle:** side-on to the player, consistent across all clips.
  This is the angle that makes elbow/shoulder/wrist trajectories most
  interpretable — front-on view loses a lot of the arm's depth motion.
- **Frame rate:** 60fps minimum, 120fps if your phone/camera supports it.
  Table tennis strokes are fast; at 30fps you will lose the contact
  moment to motion blur or miss it between frames entirely.
- **Consistency:** same stroke, same rough distance from camera, same
  lighting where possible — keeps early comparisons cleaner while you're
  validating whether the signal is even meaningful.
- **Labeling:** no need to rename files — ingest records player, session,
  stroke and camera angle in the catalog (see "Data management").

## Running the pipeline

1. Ingest the video (see "Data management") and look up its path in
   `data/raw/` via the catalog.
2. Extract pose landmarks:
   ```bash
   python src/pose_extraction.py data/raw/sha256/<xx>/<hash>.mov --out data/pose_data/forehand_01.csv
   ```
   This writes `data/pose_data/forehand_01.csv` — one row per frame with
   x/y/z/visibility for all 33 MediaPipe body landmarks.
3. Compute and plot metrics:
   ```bash
   python src/visualize.py data/pose_data/forehand_01.csv
   ```
   This writes:
   - `data/pose_data/forehand_01_metrics.png` — elbow angle, shoulder
     rotation, and wrist height over time
   - `data/pose_data/forehand_01_metrics.csv` — the same metrics as data,
     for later comparison against coach ratings
4. Have your coach fill in `data/coach_notes/template.csv` for the same
   clips (rating 1-5, flagged issue, free-text notes).
5. **Eyeball check (do this before building anything fancier):** open a
   few `_metrics.png` plots next to the coach's notes for the same clips.
   Does a low-rated clip show a visibly different elbow-angle pattern
   than a high-rated one? If yes, you have a real signal to build Phase 3
   on. If not, that's an important finding too — it tells you which
   metrics aren't discriminative and pushes you toward better ones
   (e.g. tracking the paddle, adding hip rotation, contact-point timing).

## Consistency score and comparison to your own best reps

`src/rep_analysis.py` needs no coach labels and no expert data: it compares
a player's reps with each other.

```bash
python src/pose_extraction.py data/raw/sha256/00/<hash>.mov --out data/pose_data/<hash12>.csv
python src/rep_analysis.py data/pose_data/<hash12>.csv --name 2026-09-27_P01_side_near
```

- Reps are found as peaks of wrist speed **while the wrist is rising** (the
  forward swing of a drive goes low-back -> high-front; the return swing,
  often just as fast in shadow strokes, goes down). Implausibly fast/slow
  peaks and — in side view — moments where the player turns to face the
  camera (walking to the phone) are rejected.
- Metrics use MediaPipe **world coordinates** (metres), so angles are not
  distorted by the 16:9 frame. Low-confidence landmarks are dropped.
- **Consistency score** = 100 − mean robust spread across reps, as % of each
  metric's range of motion.
- **Reference reps** = your 3 most *typical* reps by default, or the reps you
  pass with `--reference-reps` (e.g. the ones your coach marks as good).
  Typical is not the same as correct.
- Outputs in `data/processed/rep_analysis/<name>/`: `detection_*.png` (check
  first: one red dot per forward swing), `reps_overlay.png`,
  `rep_scores.png`, `reps.csv`, `summary.json`.
- **Film the side view from the hitting-arm side** (right side for a
  right-hander). From the other side the body hides the hitting arm and
  tracking fails — the analysis then (correctly) finds almost no usable reps.
- The hitting arm is set with `--side` (default `RIGHT`), not auto-detected:
  the occluded far arm jitters and would look like the faster one.

## Known limitations to expect (and document)

- MediaPipe is trained on general human motion, not fast racket sports —
  expect visibility drops or noisy landmarks around the contact moment.
- The paddle itself isn't tracked by body pose models — if contact
  timing/paddle angle matters (it will), you'll need a separate
  detector or manual annotation for that.
- `shoulder_rotation` here is a 2D proxy (HIP-SHOULDER-ELBOW angle), not
  true 3D torso rotation — good enough to start, but worth flagging as a
  simplification if this becomes a written report.

## Phase 1b: segmenting multi-rep shadow videos (now included)

If a video has you performing the same shadow stroke repeatedly with a
brief pause at ready position between reps, `segment_strokes.py`
automatically finds those pauses and splits the video into individual
repetitions — so each rep gets its own angle data instead of one long
blended signal.

```bash
python src/segment_strokes.py data/raw/sha256/<xx>/<hash>.mov
```

This will:
- Extract pose (or reuse an existing pose CSV if one already exists at
  `data/pose_data/<stem>.csv`)
- Compute wrist speed over time and detect the low-speed pauses between
  reps
- Save per-rep metrics to `data/segments/side_shadow_01/rep_01_metrics.csv`,
  `rep_02_metrics.csv`, etc. (elbow angle, shoulder rotation, wrist
  height, and wrist speed itself)
- Save `overview.png` — **check this first**: it plots wrist speed with
  the detected boundaries marked as vertical lines, so you can confirm
  they actually line up with real reps before trusting the segments
- Save `segments_summary.json` with start/end times per rep

Add `--export-clips` to also save a trimmed `.mp4` per detected rep —
useful later for coach review or as candidate reference clips.

**This is heuristic, not ground truth.** If `overview.png` shows missed
or spurious boundaries, tune:
- `--min-distance-sec` (default 0.5) — minimum time between reps; raise
  it if your reps are slower/more deliberate
- `--prominence` (default 0.005) — how deep a speed dip must be to
  count as a real pause; lower it if pauses aren't being detected,
  raise it if noise is creating false boundaries

**Note on hand-posture detail:** this tracks the wrist as a single
point (via MediaPipe Pose) — good for swing trajectory and arm angles,
but it doesn't see fingers or grip. If you want literal grip/finger-level
posture rather than wrist/arm trajectory, that needs MediaPipe Hands as
an additional layer, not yet in this starter.

## Phase 3: comparing against a reference (now included)

This is comparison against a reference profile, **not** model training —
with a handful of clips there isn't enough data to train something that
would generalize. See `src/reference_profile.py` and
`src/compare_to_reference.py`.

**Important assumption:** reference and test clips should each be
trimmed to roughly one stroke (backswing through follow-through). The
scripts don't detect stroke boundaries for you yet.

1. Pick 1-3 coach-rated "clean technique" clips and run pose extraction
   on each (as in Phase 2).
2. Build a reference profile:
   ```bash
   python src/reference_profile.py \
       data/pose_data/forehand_ref_01.csv data/pose_data/forehand_ref_02.csv \
       --out data/reference/forehand_drive_profile.csv
   ```
   With 2+ clips, each is DTW-aligned to the first clip's timing, then
   averaged — giving a mean curve plus a std band (your tolerance
   range) for each metric. With only 1 clip, a small fixed tolerance is
   used instead since there's no cross-clip variance to measure.

3. Compare a new/test clip against that profile:
   ```bash
   python src/compare_to_reference.py \
       data/pose_data/forehand_test_01.csv \
       data/reference/forehand_drive_profile.csv
   ```
   This DTW-aligns the test clip's metrics onto the reference's time
   axis (so timing differences between clips don't get mistaken for
   technique differences), then reports:
   - An overall deviation score per metric (mean |z-score| vs. the
     reference band)
   - Flagged time segments where the deviation exceeds a threshold
     (default: |z| > 1.5), with real timestamps
   - A comparison plot (`*_comparison.png`) and a JSON report
     (`*_comparison.json`)

4. **Validate against the coach.** For each test clip, compare the
   flagged segments/deviation score against what your coach
   independently noted in `data/coach_notes/`. Do high-deviation clips
   line up with what the coach flagged as technically off? This
   agreement (or disagreement) is the actual research finding — and is
   what tells you whether `elbow_angle` / `shoulder_rotation` /
   `wrist_height` are the right metrics, or whether you need to add
   others (e.g. paddle angle, hip rotation, contact timing).

## Future work (not yet in this starter)

- If you eventually collect a much larger labeled dataset (50-100+
  rated clips per stroke, many players), a lightweight learned model
  (pose-trajectory features -> predicted rating) becomes feasible and
  could replace/complement the DTW+threshold approach.
- Automatic stroke-boundary detection, so clips don't need manual
  trimming.
- Paddle tracking (separate from body pose) if contact angle/timing
  turns out to matter as much as coaches typically say it does.
