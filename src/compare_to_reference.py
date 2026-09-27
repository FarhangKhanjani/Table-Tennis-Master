"""
compare_to_reference.py

Compare a new stroke clip against a reference technique profile (built
by reference_profile.py) using DTW alignment, and flag which parts of
the stroke deviate most.

Usage:
    python src/compare_to_reference.py \
        data/pose_data/forehand_test_01.csv \
        data/reference/forehand_drive_profile.csv \
        --out data/pose_data/forehand_test_01_comparison.png

Outputs:
    - A PNG plotting the reference curve (mean ± 1 std) against the
      test clip's DTW-aligned curve, per metric.
    - A JSON report with an overall deviation score per metric and a
      list of flagged time segments (where the test clip's z-score
      exceeds DEVIATION_THRESHOLD).

This is a starting point, not a validated diagnostic tool: the
DEVIATION_THRESHOLD and the three metrics tracked are things to tune
once you compare these flags against your coach's actual ratings.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from angles import elbow_angle, shoulder_rotation, wrist_height_relative_to_shoulder
from dtw import align_series_to_reference

METRICS = {
    "elbow_angle": elbow_angle,
    "shoulder_rotation": shoulder_rotation,
    "wrist_height": wrist_height_relative_to_shoulder,
}

DEVIATION_THRESHOLD = 1.5  # flag segments where |z-score| exceeds this


def compute_metric_series(pose_csv: Path, side: str = "RIGHT") -> pd.DataFrame:
    df = pd.read_csv(pose_csv)
    out = {"timestamp": df["timestamp"].to_numpy()}
    for name, fn in METRICS.items():
        out[name] = df.apply(lambda r: fn(r, side), axis=1).to_numpy()
    return pd.DataFrame(out)


def compare_to_reference(test_csv: Path, profile_csv: Path, side: str = "RIGHT"):
    test_series = compute_metric_series(test_csv, side)
    profile = pd.read_csv(profile_csv)
    ref_timestamps = profile["timestamp"].to_numpy()

    results = {}
    flagged_segments = []

    for metric in METRICS:
        ref_mean = profile[f"{metric}_mean"].to_numpy()
        ref_std = profile[f"{metric}_std"].to_numpy()
        ref_std_safe = np.where(ref_std < 1e-3, 1e-3, ref_std)

        aligned_test = align_series_to_reference(test_series[metric].to_numpy(), ref_mean)
        z_scores = (aligned_test - ref_mean) / ref_std_safe

        results[metric] = {
            "aligned_test": aligned_test,
            "z_scores": z_scores,
            "overall_deviation": float(np.mean(np.abs(z_scores))),
        }

        flagged = np.abs(z_scores) > DEVIATION_THRESHOLD
        in_segment = False
        start_idx = 0
        for idx, is_flagged in enumerate(flagged):
            if is_flagged and not in_segment:
                start_idx = idx
                in_segment = True
            elif not is_flagged and in_segment:
                flagged_segments.append({
                    "metric": metric,
                    "start_time": float(ref_timestamps[start_idx]),
                    "end_time": float(ref_timestamps[idx - 1]),
                    "avg_z_score": float(np.mean(z_scores[start_idx:idx])),
                })
                in_segment = False
        if in_segment:
            flagged_segments.append({
                "metric": metric,
                "start_time": float(ref_timestamps[start_idx]),
                "end_time": float(ref_timestamps[-1]),
                "avg_z_score": float(np.mean(z_scores[start_idx:])),
            })

    return results, flagged_segments, ref_timestamps, profile


def plot_comparison(results, ref_timestamps, profile, title: str, output_png: Path):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 3 * len(METRICS)), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        ref_mean = profile[f"{metric}_mean"].to_numpy()
        ref_std = profile[f"{metric}_std"].to_numpy()

        ax.plot(ref_timestamps, ref_mean, label="Reference (mean)", color="tab:green")
        ax.fill_between(
            ref_timestamps, ref_mean - ref_std, ref_mean + ref_std,
            color="tab:green", alpha=0.2, label="Reference \u00b11 std"
        )
        ax.plot(
            ref_timestamps, results[metric]["aligned_test"],
            label="Test clip (DTW-aligned)", color="tab:red"
        )
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(f"{title} \u2014 test vs. reference")
    axes[-1].set_xlabel("Reference time (s)")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"Saved comparison plot to {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a test stroke clip against a reference technique profile."
    )
    parser.add_argument("test_pose_csv", help="Pose CSV of the clip to evaluate")
    parser.add_argument("profile_csv", help="Reference profile CSV from reference_profile.py")
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT", "LEFT"])
    parser.add_argument("--out", default=None, help="Output PNG path for the comparison plot")
    args = parser.parse_args()

    test_path = Path(args.test_pose_csv)
    profile_path = Path(args.profile_csv)

    results, flagged_segments, ref_timestamps, profile = compare_to_reference(
        test_path, profile_path, side=args.side
    )

    print("\nOverall deviation score per metric (mean |z-score|, lower = closer to reference):")
    for metric, r in results.items():
        print(f"  {metric}: {r['overall_deviation']:.2f}")

    if flagged_segments:
        print(f"\nFlagged segments (|z-score| > {DEVIATION_THRESHOLD}):")
        for seg in flagged_segments:
            print(
                f"  [{seg['metric']}] {seg['start_time']:.2f}s - {seg['end_time']:.2f}s "
                f"(avg z={seg['avg_z_score']:.2f})"
            )
    else:
        print("\nNo segments exceeded the deviation threshold.")

    out_path = Path(args.out) if args.out else test_path.parent / f"{test_path.stem}_comparison.png"
    plot_comparison(results, ref_timestamps, profile, title=test_path.stem, output_png=out_path)

    report = {
        "test_clip": str(test_path),
        "reference_profile": str(profile_path),
        "overall_deviation": {m: r["overall_deviation"] for m, r in results.items()},
        "flagged_segments": flagged_segments,
    }
    report_path = out_path.with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")
