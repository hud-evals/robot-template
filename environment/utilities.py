"""Shared helpers for the benchmark demos."""

from __future__ import annotations

import math

import numpy as np


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3-vector axis-angle (LIBERO convention)."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] / den * (2.0 * math.acos(w))).astype(np.float32)


__all__ = ["quat2axisangle"]
