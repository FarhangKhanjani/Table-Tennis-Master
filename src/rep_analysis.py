"""
rep_analysis.py

Consistency score and "compare to your own best reps" for a set of stroke
repetitions from one player (same stroke, same camera angle). Needs no
coach labels and no expert data.

Pipeline:
    1. Load pose CSV(s) from pose_extraction.py (world coordinates, metres).
    2. Fill short tracking gaps and smooth (Savitzky-Golay).
    3. Compute per-frame metrics: elbow angle, shoulder angle
       (HIP-SHOULDER-ELBOW), wrist height vs. shoulder, wrist speed.
    4. Detect reps as peaks of wrist speed (the forward swing is the fastest
       part of a drive) and cut a fixed window around each peak, so every
       rep is aligned on the same event.
    5. Pick reference reps: the ones you pass with --reference-reps (e.g.
       coach-approved), otherwise your k most *typical* reps (smallest
       average distance to all other reps). Note: "typical" is not the same
       as "technically correct" -- only a coach label can say that.
    6. Score:
       - consistency: how much reps vary around the mean curve, as a % of
         each metric's range of motion; score = 100 - mean variability
       - per-rep deviation from the reference reps (same units), plus the
         single biggest difference (metric, timing, size) for each rep

Usage:
    python src/rep_analysis.py data/pose_data/00597fe3c667.csv data/pose_data/684e6e48cf24.csv \
        --name 2026-09-27_P01_side
    python src/rep_analysis.py ... --reference-reps 3,7,12
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from angles import elbow_angle, shoulder_rotation, wrist_height_relative_to_shoulder

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "processed" / "rep_analysis"

JOINTS = ("SHOULDER", "ELBOW", "WRIST", "HIP")
# metric name -> (display label, unit, per-row function, scale factor to unit)
METRICS = {
    "elbow_angle": ("Elbow angle", "deg", elbow_angle, 1.0),
    "shoulder_angle": ("Shoulder angle (hip-shoulder-elbow)", "deg", shoulder_rotation, 1.0),
    "wrist_height": ("Wrist height vs. shoulder", "cm", wrist_height_relative_to_shoulder, 100.0),
    "wrist_speed": ("Wrist speed", "m/s", None, 1.0),
}
GRID_POINTS = 121


# ---------------------------------------------------------------- signals

def clean_and_smooth(df: pd.DataFrame, side: str, fps: float, max_gap_s: float = 0.05,
                     min_visibility: float = 0.5, smooth_s: float = 0.1) -> pd.DataFrame:
    """Drop low-confidence landmarks, interpolate short gaps, smooth."""
    out = df[["frame", "timestamp"]].copy()
    confident = np.ones(len(df), dtype=bool)
    for joint in JOINTS:
        confident &= (df[f"{side}_{joint}_visibility"] >= min_visibility).to_numpy()
    out["tracked"] = confident
    window = max(5, int(round(smooth_s * fps)) | 1)  # odd length
    limit = max(1, int(round(max_gap_s * fps)))
    for joint in JOINTS:
        for axis in ("wx", "wy", "wz"):
            col = f"{side}_{joint}_{axis}"
            s = df[col].where(confident).interpolate(limit=limit, limit_area="inside")
            valid = s.notna().to_numpy()
            values = s.to_numpy().copy()
            if valid.sum() > window:
                values[valid] = savgol_filter(values[valid], window, polyorder=2)
            out[col] = values
    return out


def facing_ratio(raw: pd.DataFrame, aspect: float) -> np.ndarray:
    """
    Apparent shoulder width / torso length in the image. Small when the player
    is side-on to the camera, large when facing it. In a side-view clip a high
    value means the player turned away from the stroke (e.g. walking to the
    phone to stop recording).
    """
    def dist(ax, ay, bx, by):
        return np.hypot((raw[ax] - raw[bx]) * aspect, raw[ay] - raw[by])
    shoulder_w = dist("LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y")
    mid = lambda a, b: (raw[a] + raw[b]) / 2
    torso = np.hypot(
        (mid("LEFT_SHOULDER_x", "RIGHT_SHOULDER_x") - mid("LEFT_HIP_x", "RIGHT_HIP_x")) * aspect,
        mid("LEFT_SHOULDER_y", "RIGHT_SHOULDER_y") - mid("LEFT_HIP_y", "RIGHT_HIP_y"),
    )
    return (shoulder_w / torso).to_numpy()


def compute_metrics(df: pd.DataFrame, side: str) -> pd.DataFrame:
    m = pd.DataFrame({"timestamp": df["timestamp"], "tracked": df["tracked"], "facing": df["facing"]})
    for name, (_, _, fn, scale) in METRICS.items():
        if fn is not None:
            m[name] = df.apply(lambda r: fn(r, side), axis=1) * scale
    pos = df[[f"{side}_WRIST_wx", f"{side}_WRIST_wy", f"{side}_WRIST_wz"]].to_numpy()
    t = df["timestamp"].to_numpy()
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    m["wrist_speed"] = np.concatenate([[np.nan], speed])
    return m


def detect_reps(metrics: pd.DataFrame, fps: float, min_interval_s: float, min_height_frac: float,
                min_speed_ratio: float = 0.5, max_speed_ratio: float = 2.5):
    """
    Forward-swing speed peaks. Only frames where the wrist is rising count:
    a drive swings low-back -> high-front, while the return swing goes down,
    and in shadow strokes both can be similarly fast. Peaks far above the
    typical rep (walking to the camera, tracking glitches) or far below it
    (small twitches between strokes) are rejected.
    """
    speed = metrics["wrist_speed"].fillna(0).to_numpy()
    rising = np.gradient(metrics["wrist_height"].interpolate().bfill().ffill().to_numpy()) > 0
    signal = np.where(rising, speed, 0.0)
    ref = np.nanpercentile(signal[signal > 0], 90)
    peaks, _ = find_peaks(
        signal,
        distance=max(1, int(min_interval_s * fps)),
        height=min_height_frac * ref,
        prominence=0.3 * ref,
    )
    if len(peaks):
        typical = np.median(signal[peaks])
        keep = (signal[peaks] >= min_speed_ratio * typical) & (signal[peaks] <= max_speed_ratio * typical)
        peaks = peaks[keep]
    return peaks


def cut_reps(metrics: pd.DataFrame, peaks, grid: np.ndarray, clip: str, max_missing: float = 0.2,
             max_facing: float | None = None):
    """Resample every metric on a common time grid centred on each speed peak."""
    t = metrics["timestamp"].to_numpy()
    reps = []
    for k, p in enumerate(peaks, start=1):
        t0 = t[p]
        if t0 + grid[0] < t[0] or t0 + grid[-1] > t[-1]:
            continue
        in_win = (t >= t0 + grid[0]) & (t <= t0 + grid[-1])
        if 1 - metrics["tracked"].to_numpy()[in_win].mean() > max_missing:
            continue
        if max_facing is not None and np.nanmax(metrics["facing"].to_numpy()[in_win]) > max_facing:
            continue
        curves = {}
        for name in METRICS:
            y = metrics[name].to_numpy()
            ok = ~np.isnan(y)
            curves[name] = np.interp(t0 + grid, t[ok], y[ok])
        reps.append({"clip": clip, "rep_in_clip": k, "peak_time": float(t0), "curves": curves})
    return reps


# ---------------------------------------------------------------- scoring

def stack(reps, name):
    return np.stack([r["curves"][name] for r in reps])


def metric_scales(reps):
    """Range of motion of the mean curve: the yardstick for 'how big is a difference'."""
    scales = {}
    for name in METRICS:
        mean = stack(reps, name).mean(axis=0)
        scales[name] = max(float(mean.max() - mean.min()), 1e-6)
    return scales


def pairwise_distance(reps, scales):
    n = len(reps)
    d = np.zeros((n, n))
    for name in METRICS:
        X = stack(reps, name) / scales[name]
        diff = X[:, None, :] - X[None, :, :]
        d += np.sqrt((diff ** 2).mean(axis=2))
    return d / len(METRICS)


def phase_label(t_rel: float) -> str:
    if t_rel < -0.15:
        return "backswing"
    if t_rel < 0.0:
        return "forward swing"
    if t_rel < 0.05:
        return "around peak speed / contact"
    return "follow-through"


def score_reps(reps, grid, reference_ids, k_typical):
    scales = metric_scales(reps)
    dist = pairwise_distance(reps, scales)
    typicality = dist.sum(axis=1) / max(len(reps) - 1, 1)

    if reference_ids:
        ref_idx = [i - 1 for i in reference_ids]
        ref_source = "user-selected"
    else:
        ref_idx = list(np.argsort(typicality)[:k_typical])
        ref_source = f"{len(ref_idx)} most typical reps (automatic)"

    ref_mean = {name: stack([reps[i] for i in ref_idx], name).mean(axis=0) for name in METRICS}

    # Consistency: across-rep spread at each time point, as % of range of motion.
    # Robust SD (1.4826 * MAD) so a single badly tracked rep can't dominate.
    def robust_sd(X):
        return 1.4826 * np.median(np.abs(X - np.median(X, axis=0)), axis=0)

    variability = {
        name: float(robust_sd(stack(reps, name)).mean() / scales[name] * 100) for name in METRICS
    }
    consistency = max(0.0, 100.0 - float(np.mean(list(variability.values()))))

    rows = []
    for i, r in enumerate(reps):
        row = {
            "rep": i + 1, "clip": r["clip"], "rep_in_clip": r["rep_in_clip"],
            "peak_time_s": round(r["peak_time"], 3),
            "is_reference": i in ref_idx,
            "typicality": round(float(typicality[i]) * 100, 1),
        }
        worst = (0.0, None, None, None)
        devs = []
        for name, (label, unit, _, _) in METRICS.items():
            diff = r["curves"][name] - ref_mean[name]
            rms_pct = float(np.sqrt((diff ** 2).mean()) / scales[name] * 100)
            devs.append(rms_pct)
            row[f"{name}_dev_pct"] = round(rms_pct, 1)
            j = int(np.argmax(np.abs(diff)))
            if abs(diff[j]) / scales[name] > worst[0]:
                worst = (abs(diff[j]) / scales[name], name, j, diff[j])
        row["deviation_pct"] = round(float(np.mean(devs)), 1)

        c = r["curves"]
        peak = int(np.argmin(np.abs(grid)))
        row["peak_speed_mps"] = round(float(c["wrist_speed"].max()), 2)
        row["elbow_at_peak_deg"] = round(float(c["elbow_angle"][peak]), 1)
        row["min_elbow_backswing_deg"] = round(float(c["elbow_angle"][:peak].min()), 1)
        row["wrist_height_at_peak_cm"] = round(float(c["wrist_height"][peak]), 1)

        _, name, j, d = worst
        label, unit, _, _ = METRICS[name]
        row["biggest_difference"] = (
            f"{label}: {d:+.1f} {unit} vs. reference, {grid[j] * 1000:+.0f} ms from peak speed "
            f"({phase_label(grid[j])})"
        )
        rows.append(row)

    return pd.DataFrame(rows), {
        "consistency_score": round(consistency, 1),
        "variability_pct_of_range": {k: round(v, 1) for k, v in variability.items()},
        "range_of_motion": {k: round(v, 2) for k, v in scales.items()},
        "reference": {"source": ref_source, "reps": [int(i) + 1 for i in ref_idx]},
    }, ref_mean


# ---------------------------------------------------------------- plots

def plot_detection(metrics, peaks, kept_times, title, out_png):
    t = metrics["timestamp"].to_numpy()
    s = metrics["wrist_speed"].to_numpy()
    plt.figure(figsize=(14, 3.5))
    plt.plot(t, s, lw=0.8, color="tab:blue", label="Wrist speed")
    plt.plot(t[peaks], s[peaks], "x", color="tab:gray", label="Detected peak (dropped)")
    kept = np.isin(t[peaks], kept_times)
    plt.plot(t[peaks][kept], s[peaks][kept], "o", color="tab:red", label="Rep (kept)")
    plt.xlabel("Time (s)")
    plt.ylabel("m/s")
    plt.title(f"{title} - rep detection (QA: every forward swing should have a red dot)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


def plot_overlay(reps, grid, ref_mean, table, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ms = grid * 1000
    ref_set = set(table.loc[table["is_reference"], "rep"])
    for ax, (name, (label, unit, _, _)) in zip(axes.flat, METRICS.items()):
        X = stack(reps, name)
        for i, y in enumerate(X, start=1):
            if i not in ref_set:
                ax.plot(ms, y, color="tab:gray", lw=0.6, alpha=0.5)
        mean, sd = X.mean(axis=0), X.std(axis=0)
        ax.fill_between(ms, mean - sd, mean + sd, color="tab:blue", alpha=0.15, label="all reps: mean +/- 1 SD")
        ax.plot(ms, ref_mean[name], color="tab:green", lw=2.2, label="reference reps (mean)")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_title(label)
        ax.set_ylabel(unit)
    for ax in axes[1]:
        ax.set_xlabel("ms relative to peak wrist speed")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("All reps aligned on peak wrist speed (grey = individual reps)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_scores(table, out_png):
    colors = ["tab:green" if r else "tab:blue" for r in table["is_reference"]]
    plt.figure(figsize=(max(8, len(table) * 0.35), 3.8))
    plt.bar(table["rep"], table["deviation_pct"], color=colors)
    plt.axhline(table["deviation_pct"].median(), color="k", ls="--", lw=0.8, label="median")
    plt.xticks(table["rep"])
    plt.xlabel("Rep (green = reference)")
    plt.ylabel("Deviation from reference\n(% of range of motion)")
    plt.title("How far each rep is from your reference reps (lower = closer)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Consistency score and comparison to your own best reps.")
    parser.add_argument("pose_csvs", nargs="+", help="Pose CSVs (same player, stroke and camera angle)")
    parser.add_argument("--name", required=True, help="Output folder name, e.g. 2026-09-27_P01_side")
    # Not auto-detected: in side views the far arm is occluded, and its jittery
    # landmarks look faster than the real hitting arm.
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT", "LEFT"], help="Hitting arm (handedness)")
    parser.add_argument("--pre", type=float, default=0.45, help="Seconds before peak speed in each rep window")
    parser.add_argument("--post", type=float, default=0.35, help="Seconds after peak speed in each rep window")
    parser.add_argument("--min-interval", type=float, default=0.6, help="Minimum seconds between reps")
    parser.add_argument("--min-height", type=float, default=0.5,
                        help="Peak must reach this fraction of the clip's 95th-percentile wrist speed")
    parser.add_argument("--reference-reps", default=None, help="Comma-separated rep numbers to use as reference")
    parser.add_argument("--k-typical", type=int, default=3, help="Number of automatic reference reps")
    parser.add_argument("--max-facing", type=float, default=0.5,
                        help="Side view only: drop reps where the player turns to face the camera "
                             "(shoulder width / torso length above this). 0 disables, e.g. for front view.")
    parser.add_argument("--aspect", type=float, default=16 / 9, help="Video width / height")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-args.pre, args.post, GRID_POINTS)

    reps, sides = [], {}
    for csv in map(Path, args.pose_csvs):
        raw = pd.read_csv(csv)
        fps = 1.0 / np.median(np.diff(raw["timestamp"].to_numpy()))
        side = args.side
        sides[csv.stem] = side
        cleaned = clean_and_smooth(raw, side, fps)
        cleaned["facing"] = facing_ratio(raw, args.aspect)
        metrics = compute_metrics(cleaned, side)
        peaks = detect_reps(metrics, fps, args.min_interval, args.min_height)
        clip_reps = cut_reps(metrics, peaks, grid, csv.stem,
                             max_facing=args.max_facing if args.max_facing > 0 else None)
        plot_detection(metrics, peaks, [r["peak_time"] for r in clip_reps], csv.stem,
                       out_dir / f"detection_{csv.stem}.png")
        print(f"{csv.stem}: hitting arm {side}, {fps:.1f} fps, {len(peaks)} peaks -> {len(clip_reps)} usable reps")
        reps += clip_reps

    if len(reps) < 3:
        raise SystemExit("Fewer than 3 usable reps - check the detection plots and thresholds.")

    reference_ids = [int(x) for x in args.reference_reps.split(",")] if args.reference_reps else None
    table, summary, ref_mean = score_reps(reps, grid, reference_ids, args.k_typical)

    table.to_csv(out_dir / "reps.csv", index=False)
    summary.update({
        "inputs": [str(p) for p in args.pose_csvs], "hitting_arm": sides, "num_reps": len(reps),
        "window_s": [-args.pre, args.post],
        "note": "Reference = most typical reps unless user-selected; typical != technically correct.",
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_overlay(reps, grid, ref_mean, table, out_dir / "reps_overlay.png")
    plot_scores(table, out_dir / "rep_scores.png")

    print(f"\nReps analysed: {len(reps)}   Reference: {summary['reference']['source']} "
          f"-> reps {summary['reference']['reps']}")
    print(f"CONSISTENCY SCORE: {summary['consistency_score']} / 100")
    for name, v in summary["variability_pct_of_range"].items():
        print(f"  {METRICS[name][0]:<38} varies {v:5.1f}% of its range of motion")
    print("\nFurthest from your reference reps:")
    for _, r in table.sort_values("deviation_pct", ascending=False).head(3).iterrows():
        print(f"  Rep {r['rep']:>2} ({r['clip']} @ {r['peak_time_s']:.1f}s): "
              f"{r['deviation_pct']}% -> {r['biggest_difference']}")
    print(f"\nOutputs in {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
