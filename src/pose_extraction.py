"""
pose_extraction.py

Phase 1-2 tool: run MediaPipe Pose Landmarker over a video clip and export
per-frame landmark coordinates to a CSV file.

Usage:
    python src/pose_extraction.py data/raw/sha256/00/<hash>.mov --out data/pose_data/side_01.csv

Output columns per landmark (33 MediaPipe body landmarks):
    <NAME>_x, _y      normalized to [0, 1] by image width / height
                      (use for drawing on the frame, NOT for angles:
                      x and y have different scales on non-square video)
    <NAME>_z          rough depth relative to the hips (same scale as x)
    <NAME>_visibility MediaPipe's confidence the landmark is visible (0-1)
    <NAME>_wx/_wy/_wz 3D "world" coordinates in metres, origin between the
                      hips, y pointing down. Undistorted -> use for angles.

Notes:
    - `timestamp` is each frame's real presentation time from the container,
      so variable-frame-rate clips (e.g. iPhone low-light) are timed correctly.
    - Frames are downscaled to --max-width before inference; the model works
      on a 256x256 crop internally, so 4K input only costs decode time.
    - Fast table-tennis strokes can cause motion blur / occlusion by the
      paddle. Expect gaps (NaNs) or low-visibility frames around contact —
      this is a known limitation worth documenting, not a bug to hide.
    - Model file: models/pose_landmarker_heavy.task (see README, "Setup").
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "pose_landmarker_heavy.task"
LANDMARK_NAMES = [lm.name for lm in vision.PoseLandmark]


def extract_pose_from_video(
    video_path,
    output_csv,
    model_path=DEFAULT_MODEL,
    max_width: int = 1280,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    rows = []
    frame_idx = 0
    last_ts_ms = -1
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= 0 and frame_idx > 0:
                timestamp = frame_idx / fps  # container gave no timestamp
            ts_ms = max(int(round(timestamp * 1000)), last_ts_ms + 1)  # must strictly increase
            last_ts_ms = ts_ms

            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, ts_ms)

            row = {"frame": frame_idx, "timestamp": timestamp}
            if result.pose_landmarks:
                for name, lm, wl in zip(LANDMARK_NAMES, result.pose_landmarks[0], result.pose_world_landmarks[0]):
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_visibility"] = lm.visibility
                    row[f"{name}_wx"] = wl.x
                    row[f"{name}_wy"] = wl.y
                    row[f"{name}_wz"] = wl.z
            rows.append(row)
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  {frame_idx}/{total} frames")

    cap.release()

    columns = ["frame", "timestamp"] + [
        f"{name}_{suffix}" for name in LANDMARK_NAMES
        for suffix in ("x", "y", "z", "visibility", "wx", "wy", "wz")
    ]
    df = pd.DataFrame(rows, columns=columns)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    duration = df["timestamp"].iloc[-1] if len(df) else 0.0
    missing_frac = df["NOSE_x"].isna().mean() if len(df) else 0.0
    print(f"Processed {frame_idx} frames ({duration:.1f}s, nominal {fps:.1f} fps)")
    print(f"Frames with no pose detected: {missing_frac:.1%}")
    print(f"Saved landmarks to {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks from a table tennis stroke video."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/pose_data/<video_stem>.csv)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pose Landmarker .task model file")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale frames wider than this")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else Path("data/pose_data") / f"{video_path.stem}.csv"
    extract_pose_from_video(video_path, out_path, model_path=args.model, max_width=args.max_width)
