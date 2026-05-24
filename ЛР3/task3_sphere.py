from __future__ import annotations

import numpy as np

from common import frame_to_world, orthonormal_basis_from_normal

N_SAMPLES = 100_000


def sample_uniform_unit_sphere(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = rng.random(n)
    v = rng.random(n)
    cos_theta = 2.0 * u - 1.0
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * v
    d_n = cos_theta
    d_u = sin_theta * np.cos(phi)
    d_v = sin_theta * np.sin(phi)
    return d_n, d_u, d_v


def run(rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng(42)

    axis_n = np.array([0.0, 0.0, 1.0])
    axis_u, axis_v = orthonormal_basis_from_normal(axis_n)

    d_n, d_u, d_v = sample_uniform_unit_sphere(N_SAMPLES, rng)
    dirs = frame_to_world(d_n, d_u, d_v, axis_n, axis_u, axis_v)
    norms = np.linalg.norm(dirs, axis=1)

    print("Задание 3: равномерно по поверхности единичной сферы (направления)")
    print(f"  N = {N_SAMPLES}")
    print(f"  | |d| - 1 | max = {np.max(np.abs(norms - 1.0)):.3e}")
