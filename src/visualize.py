"""
visualize.py

Phase 2 tool: load a pose CSV (from pose_extraction.py), compute a small
set of interpretable metrics over time, plot them, and save both the
plot (PNG) and the metrics themselves (CSV) for later comparison against
coach notes.

Usage:
    python src/visualize.py data/pose_data/forehand_01.csv
    python src/visualize.py data/pose_data/forehand_01.csv --side LEFT
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from angles import elbow_angle, shoulder_rotation, wrist_height_relative_to_shoulder


def compute_metrics(df: pd.DataFrame, side: str = "RIGHT") -> pd.DataFrame:
    metrics = {
        "timestamp": df["timestamp"],
        "elbow_angle": df.apply(lambda r: elbow_angle(r, side), axis=1),
        "shoulder_rotation": df.apply(lambda r: shoulder_rotation(r, side), axis=1),
        "wrist_height": df.apply(
            lambda r: wrist_height_relative_to_shoulder(r, side), axis=1
        ),
    }
    return pd.DataFrame(metrics)


def plot_metrics(metrics_df: pd.DataFrame, title: str, output_png: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(metrics_df["timestamp"], metrics_df["elbow_angle"])
    axes[0].set_ylabel("Elbow angle (deg)")
    axes[0].set_title(f"{title} — joint metrics over time")

    axes[1].plot(metrics_df["timestamp"], metrics_df["shoulder_rotation"])
    axes[1].set_ylabel("Shoulder rotation (deg)")

    axes[2].plot(metrics_df["timestamp"], metrics_df["wrist_height"])
    axes[2].set_ylabel("Wrist height\n(rel. to shoulder)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot joint-angle trajectories from a pose CSV."
    )
    parser.add_argument("pose_csv", help="Path to pose CSV produced by pose_extraction.py")
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT", "LEFT"])
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.pose_csv)
    df = pd.read_csv(csv_path)
    metrics_df = compute_metrics(df, side=args.side)

    out_path = Path(args.out) if args.out else Path("data/pose_data") / f"{csv_path.stem}_metrics.png"
    plot_metrics(metrics_df, title=csv_path.stem, output_png=out_path)

    metrics_csv = out_path.with_suffix(".csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")
