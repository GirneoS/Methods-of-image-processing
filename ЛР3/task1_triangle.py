from __future__ import annotations

import numpy as np

from common import normalize

N_SAMPLES = 100_000


def sample_uniform_triangle(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    v1 = np.asarray(v1, dtype=float).reshape(3)
    v2 = np.asarray(v2, dtype=float).reshape(3)
    v3 = np.asarray(v3, dtype=float).reshape(3)

    r1 = rng.random(n)
    r2 = rng.random(n)
    sr = np.sqrt(r1)
    w1 = 1.0 - sr
    w2 = sr * (1.0 - r2)
    w3 = sr * r2

    p = w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
    return p


def barycentric(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    v1 = np.asarray(v1, dtype=float).reshape(3)
    v2 = np.asarray(v2, dtype=float).reshape(3)
    v3 = np.asarray(v3, dtype=float).reshape(3)
    m = np.stack([v2 - v1, v3 - v1], axis=1)
    rhs = (p - v1).T
    w2w3, _, _, _ = np.linalg.lstsq(m, rhs, rcond=None)
    w2 = w2w3[0]
    w3 = w2w3[1]
    w1 = 1.0 - w2 - w3
    return np.stack([w1, w2, w3], axis=1)


def verify_inside(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> tuple[int, float]:
    w = barycentric(p, v1, v2, v3)
    min_w = w.min(axis=1)
    inside = np.sum(min_w >= -1e-9)
    worst = float(min_w.min())
    return int(inside), worst


def run(rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng(42)

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 1.0])
    v3 = np.array([1.0, 2.0, 0.5])

    p = sample_uniform_triangle(v1, v2, v3, N_SAMPLES, rng)
    n_in, worst = verify_inside(p, v1, v2, v3)

    e1 = normalize(v2 - v1)
    e2 = normalize(v3 - v1)
    n = normalize(np.cross(e1, e2))
    p_plane = np.sum((p - v1) * n, axis=1)

    print("Задание 1: равномерно по площади в треугольнике")
    print(f"  N = {N_SAMPLES}")
    print(f"  точек с w1,w2,w3 >= 0 (с допуском): {n_in} / {N_SAMPLES}")
    print(f"  минимальный барицентрический вес: {worst:.3e}")
    print(f"  отклонение от плоскости (должно ~0): max|dot| = {np.max(np.abs(p_plane)):.3e}")
