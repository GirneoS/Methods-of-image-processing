"""
Задание 2. Равномерное распределение в круге.

Метод:
    r = R * sqrt(u1),  phi = 2*pi*u2,  u1,u2 ~ U[0,1]
    P = C + r*cos(phi)*e1 + r*sin(phi)*e2     (e1,e2 ортонорм. в плоскости N)

Доказательства равномерности:
    A) Все точки в круге: ||P - C|| <= R, и лежат в плоскости N (dot(P-C, N)=0).
    B) KS-тест: при равномерности по площади r^2 ~ U[0, R^2], phi ~ U[0, 2pi].
    C) chi2-тест: круг разбит на K=20 концентрических колец РАВНОЙ ПЛОЩАДИ
       (граница k-го кольца r_k = R*sqrt(k/K)); каждое должно содержать N/K точек.
"""
from __future__ import annotations

import numpy as np

from common import orthonormal_basis_from_normal
from stats  import ks_test_uniform, chi2_uniform, fmt_ks, fmt_chi2

N_SAMPLES = 100_000


def sample_uniform_disk_3d(center, normal, radius, n, rng):
    center = np.asarray(center, dtype=float).reshape(3)
    e1, e2 = orthonormal_basis_from_normal(normal)
    u1 = rng.random(n); u2 = rng.random(n)
    r = float(radius) * np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    offset = r[:, None] * (np.cos(phi)[:, None]*e1 + np.sin(phi)[:, None]*e2)
    return center + offset, r, phi


def run(rng=None):
    rng = rng or np.random.default_rng(42)
    center = np.array([1.0, -0.5, 2.0])
    normal = np.array([1.0, 1.0, 1.0])
    R = 1.5

    p, r, phi = sample_uniform_disk_3d(center, normal, R, N_SAMPLES, rng)

    # A) Принадлежность кругу
    n_hat = normal / np.linalg.norm(normal)
    d = p - center
    plane_dev = float(np.max(np.abs(d @ n_hat)))
    r_actual = np.linalg.norm(d - (d @ n_hat)[:, None]*n_hat, axis=1)
    inside = int(np.sum(r_actual <= R + 1e-7))

    # B) KS-тесты
    # r^2 / R^2 должно быть U[0,1]
    ks_r2  = ks_test_uniform(r**2, lo=0.0, hi=R**2)
    # phi / (2pi) должно быть U[0,1]
    ks_phi = ks_test_uniform(phi, lo=0.0, hi=2*np.pi)

    # C) chi2 по концентрическим кольцам равной площади.
    #    r^2/R^2  in  [0,1] разбиваем на K=20 равных интервалов.
    K = 20
    bins = np.clip(((r**2) / R**2 * K).astype(int), 0, K-1)
    counts = np.bincount(bins, minlength=K)
    chi2_ring = chi2_uniform(counts, N_SAMPLES / K)

    print("Задание 2: равномерное распределение в круге")
    print(f"  N = {N_SAMPLES}, R = {R}")
    print(f"  A) Точек в круге:                        {inside} / {N_SAMPLES}")
    print(f"     Отклонение от плоскости:              {plane_dev:.3e}")
    print(f"  B) KS-тесты (выборка должна быть U[0,1]):")
    print(fmt_ks("r^2 / R^2 ~ U[0,1]",  *ks_r2))
    print(fmt_ks("phi / (2pi) ~ U[0,1]", *ks_phi))
    print(f"  C) chi2 на {K} концентрических кольцах равной площади:")
    print(fmt_chi2("число точек в каждом кольце ~= N/K", *chi2_ring))

    return {
        "inside": inside, "plane_dev": plane_dev,
        "ks_r2": ks_r2, "ks_phi": ks_phi,
        "chi2_ring": chi2_ring, "K": K,
        "r": r, "phi": phi, "points": p,
    }
