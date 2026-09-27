"""
angles.py

Helper functions to turn raw MediaPipe landmark coordinates (from a row of
the pose CSV) into interpretable biomechanical metrics: joint angles,
relative heights, etc.

These are the building blocks for Phase 3 (rule-based / reference
comparison) — start simple here, validate against coach ratings, then
add more metrics as you and your coach identify what actually matters.
"""

import numpy as np


def _has_world(row, name) -> bool:
    return f"{name}_wx" in row.index


def _get_point(row, name):
    """
    3D point for a landmark. Prefers MediaPipe world coordinates (metres,
    same scale on every axis); falls back to normalized image coordinates
    for older pose CSVs, which distort angles on non-square video.
    """
    if _has_world(row, name):
        return np.array([row[f"{name}_wx"], row[f"{name}_wy"], row[f"{name}_wz"]], dtype=float)
    return np.array(
        [row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]], dtype=float
    )


def joint_angle(row, a_name, b_name, c_name):
    """
    Angle at joint B (in degrees), formed by the three points A-B-C.
    Example: elbow_angle uses SHOULDER-ELBOW-WRIST.
    """
    a = _get_point(row, a_name)
    b = _get_point(row, b_name)
    c = _get_point(row, c_name)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def elbow_angle(row, side: str = "RIGHT") -> float:
    """Elbow flexion angle: SHOULDER-ELBOW-WRIST."""
    return joint_angle(row, f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST")


def shoulder_rotation(row, side: str = "RIGHT") -> float:
    """
    Rough proxy for shoulder/torso rotation: HIP-SHOULDER-ELBOW angle.
    Not a substitute for true 3D rotation, but a cheap first signal.
    """
    return joint_angle(row, f"{side}_HIP", f"{side}_SHOULDER", f"{side}_ELBOW")


def wrist_height_relative_to_shoulder(row, side: str = "RIGHT") -> float:
    """
    Positive = wrist above shoulder, negative = below.
    (y-axis increases downward, so we flip the sign.) In metres when world
    coordinates are available, otherwise in normalized image units.
    """
    axis = "wy" if _has_world(row, side + "_WRIST") else "y"
    return row[f"{side}_SHOULDER_{axis}"] - row[f"{side}_WRIST_{axis}"]


def knee_bend(row, side: str = "RIGHT") -> float:
    """Knee flexion angle: HIP-KNEE-ANKLE. Useful for stance/weight-transfer checks."""
    return joint_angle(row, f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE")
