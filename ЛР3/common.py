from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("Нулевой вектор нельзя нормализовать")
    return v / n


def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(n)
    if abs(n[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, t)
    u = normalize(u)
    v = np.cross(n, u)
    return u, v


def frame_to_world(
    d_n: np.ndarray,
    d_u: np.ndarray,
    d_v: np.ndarray,
    axis_n: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    axis_n = normalize(axis_n)
    axis_u = normalize(axis_u)
    axis_v = normalize(axis_v)
    return (
        np.asarray(d_n).reshape(-1, 1) * axis_n
        + np.asarray(d_u).reshape(-1, 1) * axis_u
        + np.asarray(d_v).reshape(-1, 1) * axis_v
    )
