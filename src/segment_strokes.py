"""
segment_strokes.py

Phase 1: given a video of repeated shadow strokes (same stroke performed
back-to-back with a brief pause at ready position between reps), this:

  1. Extracts pose landmarks for the whole video (or reuses an existing
     pose CSV from pose_extraction.py).
  2. Computes wrist speed over time as a motion signal.
  3. Finds the low-speed "pause" points between reps to automatically
     segment the video into individual stroke repetitions.
  4. Computes angle metrics (elbow angle, shoulder rotation, wrist
     height, wrist speed) for each segment separately.
  5. Saves per-segment pose/metrics CSVs, a summary of detected
     segments, an overview QA plot, and (optionally) trimmed video
     clips per repetition.

This is heuristic segmentation based on motion, not ground truth — the
overview plot exists specifically so you can eyeball whether the
detected boundaries actually line up with real reps before trusting the
segments downstream. Expect to tune --min-distance-sec and --prominence
for your own footage/pace.

Usage:
    python src/segment_strokes.py data/raw_clips/side_shadow_01.mp4

    # if you already ran pose_extraction.py separately:
    python src/segment_strokes.py data/raw_clips/side_shadow_01.mp4 \
        --pose-csv data/pose_data/side_shadow_01.csv

    # export a trimmed .mp4 per detected repetition:
    python src/segment_strokes.py data/raw_clips/side_shadow_01.mp4 --export-clips
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from angles import elbow_angle, shoulder_rotation, wrist_height_relative_to_shoulder
from pose_extraction import extract_pose_from_video

METRICS = {
    "elbow_angle": elbow_angle,
    "shoulder_rotation": shoulder_rotation,
    "wrist_height": wrist_height_relative_to_shoulder,
}


def get_or_extract_pose(video_path: Path, pose_csv_arg) -> pd.DataFrame:
    if pose_csv_arg:
        pose_csv = Path(pose_csv_arg)
        if not pose_csv.exists():
            raise FileNotFoundError(f"--pose-csv given but not found: {pose_csv}")
        return pd.read_csv(pose_csv)

    default_csv = Path("data/pose_data") / f"{video_path.stem}.csv"
    if default_csv.exists():
        print(f"Reusing existing pose CSV: {default_csv}")
        return pd.read_csv(default_csv)

    print("No existing pose CSV found, running pose extraction...")
    return extract_pose_from_video(video_path, default_csv)


def compute_wrist_speed(df: pd.DataFrame, side: str = "RIGHT", smooth_window: int = 5) -> np.ndarray:
    """
    Frame-to-frame displacement of the wrist landmark (normalized image
    coordinates), as a proxy for swing speed. Smoothed with a rolling
    mean so single noisy frames don't create false pauses/peaks.
    """
    x = df[f"{side}_WRIST_x"].to_numpy()
    y = df[f"{side}_WRIST_y"].to_numpy()

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx**2 + dy**2)

    speed_series = pd.Series(speed).rolling(window=smooth_window, center=True, min_periods=1).mean()
    return speed_series.to_numpy()


def find_rep_boundaries(speed: np.ndarray, fps: float, min_distance_sec: float, prominence: float):
    """
    Find low-speed "pause" points (valleys) in the wrist-speed signal.
    Consecutive valleys become segment boundaries.
    """
    min_distance_frames = max(1, int(min_distance_sec * fps))
    inverted = -speed
    valley_idx, _ = find_peaks(inverted, distance=min_distance_frames, prominence=prominence)

    # Always include the very start and end of the clip as boundaries
    boundaries = sorted(set([0] + valley_idx.tolist() + [len(speed) - 1]))
    return boundaries


def build_segments(boundaries, min_segment_frames: int = 10):
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= min_segment_frames:
            segments.append((start, end))
    return segments


def compute_segment_metrics(df: pd.DataFrame, start: int, end: int, speed: np.ndarray, side: str = "RIGHT") -> pd.DataFrame:
    seg_df = df.iloc[start:end + 1].reset_index(drop=True)
    metrics = {"timestamp": seg_df["timestamp"].to_numpy()}
    for name, fn in METRICS.items():
        metrics[name] = seg_df.apply(lambda r: fn(r, side), axis=1).to_numpy()
    metrics["wrist_speed"] = speed[start:end + 1]
    return pd.DataFrame(metrics)


def plot_overview(speed: np.ndarray, timestamps: np.ndarray, boundaries, title: str, output_png: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, speed, color="tab:blue", label="Wrist speed (smoothed)")
    for b in boundaries:
        plt.axvline(timestamps[b], color="tab:red", linestyle="--", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Wrist speed (normalized units/frame)")
    plt.title(f"{title} \u2014 detected rep boundaries")
    plt.legend(loc="upper right")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"Saved QA overview plot to {output_png}")


def export_clip(video_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment a multi-rep shadow-stroke video and compute angle metrics per rep."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--pose-csv", default=None, help="Reuse an existing pose CSV instead of re-extracting")
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT", "LEFT"])
    parser.add_argument("--min-distance-sec", type=float, default=0.5,
                         help="Minimum time between reps, in seconds (tune to your pace)")
    parser.add_argument("--prominence", type=float, default=0.005,
                         help="Minimum dip prominence to count as a pause between reps (tune per footage)")
    parser.add_argument("--min-segment-frames", type=int, default=10,
                         help="Discard segments shorter than this many frames (noise filter)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: data/segments/<video_stem>)")
    parser.add_argument("--export-clips", action="store_true", help="Also export a trimmed .mp4 per detected rep")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/segments") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = get_or_extract_pose(video_path, args.pose_csv)
    fps_est = 1.0 / np.median(np.diff(df["timestamp"].to_numpy())) if len(df) > 1 else 30.0

    speed = compute_wrist_speed(df, side=args.side)
    boundaries = find_rep_boundaries(speed, fps_est, args.min_distance_sec, args.prominence)
    segments = build_segments(boundaries, args.min_segment_frames)

    print(f"\nDetected {len(segments)} repetition(s) from {len(boundaries)} boundary points.")

    plot_overview(speed, df["timestamp"].to_numpy(), boundaries, video_path.stem, out_dir / "overview.png")

    summary = []
    for i, (start, end) in enumerate(segments, start=1):
        seg_metrics = compute_segment_metrics(df, start, end, speed, side=args.side)
        seg_csv = out_dir / f"rep_{i:02d}_metrics.csv"
        seg_metrics.to_csv(seg_csv, index=False)

        start_time = float(df["timestamp"].iloc[start])
        end_time = float(df["timestamp"].iloc[end])
        summary.append({
            "rep": i,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        })
        print(f"  Rep {i:02d}: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s) -> {seg_csv}")

        if args.export_clips:
            clip_path = out_dir / f"rep_{i:02d}.mp4"
            export_clip(video_path, start, end, fps_est, clip_path)
            print(f"           exported clip -> {clip_path}")

    with open(out_dir / "segments_summary.json", "w") as f:
        json.dump({
            "video": str(video_path),
            "side": args.side,
            "fps_estimate": fps_est,
            "num_reps": len(segments),
            "segments": summary,
        }, f, indent=2)

    print(f"\nSaved summary to {out_dir / 'segments_summary.json'}")
    print("Check overview.png to confirm the detected boundaries actually match real reps —")
    print("if not, rerun with different --min-distance-sec / --prominence values.")
