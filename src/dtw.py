"""
dtw.py

Minimal, dependency-free Dynamic Time Warping (DTW) implementation.

DTW finds the best alignment between two time series that may differ in
speed or timing — e.g. one player's stroke takes 0.6s and another's
takes 0.75s, or the backswing starts a few frames earlier/later.
Comparing two strokes frame-by-frame without this would penalize timing
differences that have nothing to do with technique quality.

This is intentionally a simple O(n*m) implementation rather than a
production-grade library — fine for stroke-length sequences (a few
hundred frames at most), and it keeps the algorithm inspectable if you
want to explain/justify it in a report.
"""

import numpy as np


def dtw_align(seq_a: np.ndarray, seq_b: np.ndarray):
    """
    Compute DTW distance and alignment path between two 1D sequences.

    Returns:
        distance: float, total cumulative alignment cost
        path: list of (i, j) index pairs mapping seq_a[i] <-> seq_b[j]
    """
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        moves = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
        move = int(np.argmin(moves))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i = i - 1
        else:
            j = j - 1
    path.reverse()

    return cost[n, m], path


def align_series_to_reference(test_seq: np.ndarray, ref_seq: np.ndarray) -> np.ndarray:
    """
    Align test_seq onto ref_seq's time axis using DTW.

    Returns an array the same length as ref_seq, where each entry is the
    average of all test_seq values mapped to that reference index.
    Reference indices with no mapped test point are filled by linear
    interpolation.
    """
    _, path = dtw_align(test_seq, ref_seq)

    buckets = {j: [] for j in range(len(ref_seq))}
    for i, j in path:
        buckets[j].append(test_seq[i])

    aligned = np.full(len(ref_seq), np.nan)
    for j, values in buckets.items():
        if values:
            aligned[j] = float(np.mean(values))

    nan_mask = np.isnan(aligned)
    if nan_mask.any():
        idx = np.arange(len(ref_seq))
        aligned[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], aligned[~nan_mask])

    return aligned
