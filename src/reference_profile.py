"""
reference_profile.py

Build a reference technique profile from one or more coach-approved
"clean technique" clips (their pose CSVs from pose_extraction.py).

If multiple reference clips are given, each is DTW-aligned onto the
first clip's time axis, then averaged — giving a smoother reference
curve plus a std band across clips, which becomes your tolerance range
in compare_to_reference.py. With a single reference clip, a small fixed
tolerance is used instead (there's no cross-clip variance to measure).

Assumption: each reference clip should already be trimmed to roughly one
stroke (backswing through follow-through) — this script does not detect
stroke boundaries for you.

Usage:
    python src/reference_profile.py \
        data/pose_data/forehand_ref_01.csv data/pose_data/forehand_ref_02.csv \
        --out data/reference/forehand_drive_profile.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from angles import elbow_angle, shoulder_rotation, wrist_height_relative_to_shoulder
from dtw import align_series_to_reference

METRICS = {
    "elbow_angle": elbow_angle,
    "shoulder_rotation": shoulder_rotation,
    "wrist_height": wrist_height_relative_to_shoulder,
}


def compute_metric_series(pose_csv: Path, side: str = "RIGHT") -> pd.DataFrame:
    df = pd.read_csv(pose_csv)
    out = {"timestamp": df["timestamp"].to_numpy()}
    for name, fn in METRICS.items():
        out[name] = df.apply(lambda r: fn(r, side), axis=1).to_numpy()
    return pd.DataFrame(out)


def build_reference_profile(pose_csvs, side: str = "RIGHT") -> pd.DataFrame:
    series_list = [compute_metric_series(Path(p), side) for p in pose_csvs]
    base = series_list[0]
    n = len(base)

    profile = {"timestamp": base["timestamp"].to_numpy()}

    for metric in METRICS:
        aligned_all = [base[metric].to_numpy()]
        for other in series_list[1:]:
            aligned = align_series_to_reference(other[metric].to_numpy(), base[metric].to_numpy())
            aligned_all.append(aligned)

        stacked = np.vstack(aligned_all)  # shape: (n_clips, n_frames)
        profile[f"{metric}_mean"] = stacked.mean(axis=0)
        profile[f"{metric}_std"] = (
            stacked.std(axis=0) if len(aligned_all) > 1 else np.full(n, 5.0)
        )

    return pd.DataFrame(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a reference technique profile from one or more clean-technique pose CSVs."
    )
    parser.add_argument("pose_csvs", nargs="+", help="Pose CSV(s) of clean/reference strokes")
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT", "LEFT"])
    parser.add_argument("--out", required=True, help="Output path for the reference profile CSV")
    args = parser.parse_args()

    profile_df = build_reference_profile(args.pose_csvs, side=args.side)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(out_path, index=False)
    print(f"Built reference profile from {len(args.pose_csvs)} clip(s) -> {out_path}")
