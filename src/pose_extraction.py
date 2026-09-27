"""
pose_extraction.py

Phase 1-2 tool: run MediaPipe Pose over a video clip and export per-frame
landmark coordinates (x, y, z, visibility) to a CSV file.

Usage:
    python src/pose_extraction.py data/raw_clips/forehand_01.mp4
    python src/pose_extraction.py data/raw_clips/forehand_01.mp4 --out data/pose_data/forehand_01.csv

Notes:
    - x, y are normalized to [0, 1] relative to image width/height.
    - z is a rough depth estimate relative to the hips (same scale as x).
    - visibility is MediaPipe's confidence that the landmark is visible (0-1).
    - Fast table-tennis strokes can cause motion blur / occlusion by the
      paddle. Expect gaps (NaNs) or low-visibility frames around contact —
      this is a known limitation worth documenting, not a bug to hide.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
LANDMARK_NAMES = [lm.name for lm in mp_pose.PoseLandmark]


def extract_pose_from_video(
    video_path,
    output_csv,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            row = {"frame": frame_idx, "timestamp": timestamp}
            if results.pose_landmarks:
                for name, lm in zip(LANDMARK_NAMES, results.pose_landmarks.landmark):
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_visibility"] = lm.visibility
            else:
                for name in LANDMARK_NAMES:
                    row[f"{name}_x"] = None
                    row[f"{name}_y"] = None
                    row[f"{name}_z"] = None
                    row[f"{name}_visibility"] = None

            rows.append(row)
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    missing_frac = df["NOSE_x"].isna().mean() if "NOSE_x" in df else 0.0
    print(f"Processed {frame_idx} frames ({frame_idx / fps:.1f}s @ {fps:.1f} fps)")
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
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity (0=fastest/least accurate, 2=slowest/most accurate)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else Path("data/pose_data") / f"{video_path.stem}.csv"
    extract_pose_from_video(video_path, out_path, model_complexity=args.model_complexity)
