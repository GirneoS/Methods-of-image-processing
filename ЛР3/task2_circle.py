from __future__ import annotations

import numpy as np

from common import orthonormal_basis_from_normal

N_SAMPLES = 100_000


def sample_uniform_disk_3d(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(3)
    u_axis, v_axis = orthonormal_basis_from_normal(normal)

    u = rng.random(n)
    t = rng.random(n)
    r = float(radius) * np.sqrt(u)
    phi = 2.0 * np.pi * t

    offset = r[:, None] * (np.cos(phi)[:, None] * u_axis + np.sin(phi)[:, None] * v_axis)
    return center + offset


def verify_disk(
    p: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> tuple[int, float, float]:
    center = np.asarray(center, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    d = p - center
    in_plane = np.abs(np.sum(d * n, axis=1))
    d_tan = d - np.sum(d * n, axis=1)[:, None] * n
    dist = np.linalg.norm(d_tan, axis=1)
    ok = (dist <= float(radius) + 1e-7) & (in_plane < 1e-7)
    return int(np.sum(ok)), float(dist.max()), float(in_plane.max())


def run(rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng(42)

    center = np.array([1.0, -0.5, 2.0])
    normal = np.array([1.0, 1.0, 1.0])
    radius = 1.5

    p = sample_uniform_disk_3d(center, normal, radius, N_SAMPLES, rng)
    n_ok, max_r, max_plane = verify_disk(p, center, normal, radius)

    print("Задание 2: равномерно по площади в круге (диск в 3D)")
    print(f"  N = {N_SAMPLES}, R = {radius}")
    print(f"  точек в круге (с допуском): {n_ok} / {N_SAMPLES}")
    print(f"  max расстояние по плоскости до центра: {max_r:.6f} (ожидается <= {radius})")
    print(f"  max отклонение от плоскости: {max_plane:.3e}")
