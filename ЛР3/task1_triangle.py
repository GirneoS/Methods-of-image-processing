"""
Задание 1. Равномерное распределение в треугольнике.

Метод:
    w1 = 1 - sqrt(u1)
    w2 = sqrt(u1) * (1 - u2)
    w3 = sqrt(u1) * u2
    P  = w1*V1 + w2*V2 + w3*V3
    где u1, u2 ~ U[0,1].

Доказательства равномерности:
    A) Все точки внутри: w1, w2, w3 >= 0 и w1 + w2 + w3 = 1.
    B) KS-тест: барицентрические координаты w_i имеют известную
       маргинальную CDF F(w) = 1 - (1-w)^2 = 2w - w^2 для равномерного
       по площади распределения. Преобразование U = F(w_i) должно
       быть равномерно на [0,1].
    C) chi2-тест: разбиваем единичный (w2,w3)-треугольник на K=100
       равных по площади ячеек, считаем N_i, сравниваем с E = N/K.
"""
from __future__ import annotations

import numpy as np

from common import normalize
from stats  import (ks_test_uniform, chi2_uniform,
                    fmt_ks, fmt_chi2)

N_SAMPLES = 100_000


def sample_uniform_triangle(v1, v2, v3, n, rng):
    v1 = np.asarray(v1, dtype=float).reshape(3)
    v2 = np.asarray(v2, dtype=float).reshape(3)
    v3 = np.asarray(v3, dtype=float).reshape(3)
    u1 = rng.random(n); u2 = rng.random(n)
    sr = np.sqrt(u1)
    w1 = 1.0 - sr
    w2 = sr * (1.0 - u2)
    w3 = sr * u2
    p = w1[:, None]*v1 + w2[:, None]*v2 + w3[:, None]*v3
    return p, np.stack([w1, w2, w3], axis=1)


def barycentric(p, v1, v2, v3):
    m   = np.stack([v2 - v1, v3 - v1], axis=1)
    rhs = (p - v1).T
    sol, *_ = np.linalg.lstsq(m, rhs, rcond=None)
    w2 = sol[0]; w3 = sol[1]; w1 = 1.0 - w2 - w3
    return np.stack([w1, w2, w3], axis=1)


def run(rng=None):
    rng = rng or np.random.default_rng(42)
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 1.0])
    v3 = np.array([1.0, 2.0, 0.5])

    p, w_gen = sample_uniform_triangle(v1, v2, v3, N_SAMPLES, rng)

    # A) Все ли внутри (по точным барицентрическим координатам точек)?
    w = barycentric(p, v1, v2, v3)
    inside = int(np.sum(w.min(axis=1) >= -1e-9))

    # Точки в плоскости треугольника
    n  = normalize(np.cross(v2 - v1, v3 - v1))
    plane_dev = float(np.max(np.abs(np.sum((p - v1) * n, axis=1))))

    # B) KS-тест на маргинальной CDF для w_i:
    #    pdf(w) = 2(1-w), CDF F(w) = 2w - w^2 на [0,1]
    #    F(w_i) должно быть U[0,1].
    cdf = lambda w: 2*w - w**2
    ks_w1 = ks_test_uniform(cdf(w[:, 0]))
    ks_w2 = ks_test_uniform(cdf(w[:, 1]))
    ks_w3 = ks_test_uniform(cdf(w[:, 2]))

    # C) chi2 на барицентрической сетке.
    #    Делим единичный треугольник {w2>=0, w3>=0, w2+w3<=1}
    #    на K=100 ячеек одинаковой площади (10 x 10 квадратиков,
    #    каждый разрезан на 2 треугольника, итого 200 ячеек).
    K_SIDE = 10
    bin2 = np.clip((w[:, 1] * K_SIDE).astype(int), 0, K_SIDE - 1)
    bin3 = np.clip((w[:, 2] * K_SIDE).astype(int), 0, K_SIDE - 1)
    # тип ячейки: 0 = нижний треугольник квадрата (w2+w3 - i - j <= 1 на дробной части),
    #             1 = верхний
    frac2 = w[:, 1] * K_SIDE - bin2
    frac3 = w[:, 2] * K_SIDE - bin3
    cell_type = (frac2 + frac3 > 1.0).astype(int)
    flat_id = bin2 * (K_SIDE * 2) + bin3 * 2 + cell_type

    # ячейки, лежащие целиком вне исходного треугольника (i+j >= K_SIDE
    # для нижнего и i+j > K_SIDE-1 для верхнего) - не используем
    valid = []
    for i in range(K_SIDE):
        for j in range(K_SIDE):
            if i + j < K_SIDE - 1:
                valid += [i*(K_SIDE*2) + j*2 + 0, i*(K_SIDE*2) + j*2 + 1]
            elif i + j == K_SIDE - 1:
                valid += [i*(K_SIDE*2) + j*2 + 0]
    valid = np.array(valid)
    counts = np.bincount(flat_id, minlength=K_SIDE*K_SIDE*2)[valid]
    K = len(valid)
    E = N_SAMPLES / K   # все ячейки одинаковой площади
    chi2_res = chi2_uniform(counts, E)

    print("Задание 1: равномерное распределение в треугольнике")
    print(f"  N = {N_SAMPLES}")
    print(f"  A) Точек внутри треугольника:           {inside} / {N_SAMPLES}")
    print(f"     Отклонение от плоскости:             {plane_dev:.3e}")
    print(f"  B) KS-тест на маргинальной CDF (барицентрических координат):")
    print(fmt_ks("F(w1) ~ U[0,1]", *ks_w1))
    print(fmt_ks("F(w2) ~ U[0,1]", *ks_w2))
    print(fmt_ks("F(w3) ~ U[0,1]", *ks_w3))
    print(f"  C) chi2-тест на {K} равных по площади ячейках:")
    print(fmt_chi2(f"равномерность по треугольнику", *chi2_res))

    return {
        "inside": inside, "plane_dev": plane_dev,
        "ks_w1": ks_w1, "ks_w2": ks_w2, "ks_w3": ks_w3,
        "chi2": chi2_res, "K": K,
        "points": p, "w": w,
    }
