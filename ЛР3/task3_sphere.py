"""
Задание 3. Равномерное распределение на сфере (направления).

Метод:
    cos(t) = 1 - 2*u1   in  [-1, 1]
    phi    = 2*pi*u2    in  [0, 2pi]
    d = (sint cosphi, sint sinphi, cos theta)

Доказательства равномерности:
    A) Все векторы единичные: |||d| - 1|| ~= 0.
    B) KS-тесты: cos theta ~ U[-1,1], phi ~ U[0, 2pi].
    C) chi2-тест: сферу разбиваем на K=200 ячеек равного телесного угла
       (10 поясов по cos theta x 20 секторов по phi - все ячейки равноплощадные).
"""
from __future__ import annotations

import numpy as np

from common import frame_to_world, orthonormal_basis_from_normal
from stats  import ks_test_uniform, chi2_uniform, fmt_ks, fmt_chi2

N_SAMPLES = 100_000


def sample_uniform_unit_sphere(n, rng):
    u1 = rng.random(n); u2 = rng.random(n)
    cos_t = 2.0 * u1 - 1.0
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    phi   = 2.0 * np.pi * u2
    return cos_t, sin_t, phi


def run(rng=None):
    rng = rng or np.random.default_rng(42)
    axis_n = np.array([0.0, 0.0, 1.0])
    axis_u, axis_v = orthonormal_basis_from_normal(axis_n)

    cos_t, sin_t, phi = sample_uniform_unit_sphere(N_SAMPLES, rng)
    dirs = frame_to_world(cos_t, sin_t*np.cos(phi), sin_t*np.sin(phi),
                          axis_n, axis_u, axis_v)
    norms = np.linalg.norm(dirs, axis=1)

    # A) единичность
    unit_dev = float(np.max(np.abs(norms - 1.0)))

    # B) KS-тесты
    ks_cos = ks_test_uniform(cos_t, lo=-1.0, hi=1.0)
    ks_phi = ks_test_uniform(phi,   lo=0.0,  hi=2*np.pi)

    # C) chi2 по равным телесным углам: 10 поясов x 20 секторов = 200 ячеек
    N_THETA, N_PHI = 10, 20
    K = N_THETA * N_PHI
    # cos theta  in  [-1, 1] → бин 0..N_THETA-1
    bin_c = np.clip(((cos_t + 1.0) / 2.0 * N_THETA).astype(int), 0, N_THETA-1)
    bin_p = np.clip((phi / (2*np.pi) * N_PHI).astype(int), 0, N_PHI-1)
    flat = bin_c * N_PHI + bin_p
    counts = np.bincount(flat, minlength=K)
    chi2_res = chi2_uniform(counts, N_SAMPLES / K)

    print("Задание 3: равномерное распределение на единичной сфере")
    print(f"  N = {N_SAMPLES}")
    print(f"  A) Единичность векторов:                  | |d|-1 | max = {unit_dev:.3e}")
    print(f"  B) KS-тесты:")
    print(fmt_ks("cos theta ~ U[-1, 1]", *ks_cos))
    print(fmt_ks("phi ~ U[0, 2pi]",     *ks_phi))
    print(f"  C) chi2 на {K} ячейках равного телесного угла ({N_THETA}x{N_PHI}):")
    print(fmt_chi2("равномерность по сфере", *chi2_res))

    return {
        "unit_dev": unit_dev, "ks_cos": ks_cos, "ks_phi": ks_phi,
        "chi2": chi2_res, "K": K,
        "cos_t": cos_t, "phi": phi, "dirs": dirs,
    }
