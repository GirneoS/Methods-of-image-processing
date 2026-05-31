"""
Задание 4. Косинусное распределение на полусфере вокруг N.

Метод (Мальмотт):
    cos(t) = sqrt(u1)
    phi    = 2*pi*u2
    d_local = (sint cosphi, sint sinphi, cos theta),  затем перевод в мир через ось N.

Плотность вероятности pdf(d) = cos(t)/pi - пропорциональна cos(угла) с N.

Доказательства корректности (КОСИНУСНОЕ, не равномерное):
    A) Все векторы единичные.
    B) Все векторы в верхнем полушарии: cos(угла с N) >= 0.
    C) KS-тест на маргиналах:
       pdf(t) = 2 cos theta sin t → переменная X = cos2_theta должна быть U[0,1];
       phi должно быть U[0, 2pi].
    D) Гистограмма cos theta с теоретической кривой pdf(cos theta)=2cos theta (для отчёта).
    E) chi2-тест: бины по cos2_theta и по phi; ожидание E_ij = N/K (по построению).
"""
from __future__ import annotations

import numpy as np

from common import frame_to_world, orthonormal_basis_from_normal
from stats  import ks_test_uniform, chi2_uniform, fmt_ks, fmt_chi2

N_SAMPLES = 100_000


def sample_cosine_hemisphere(n, rng):
    u1 = rng.random(n); u2 = rng.random(n)
    cos_t = np.sqrt(u1)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    phi   = 2.0 * np.pi * u2
    return cos_t, sin_t, phi


def run(rng=None):
    rng = rng or np.random.default_rng(42)
    axis_n = np.array([0.0, 1.0, 0.0])
    axis_u, axis_v = orthonormal_basis_from_normal(axis_n)

    cos_t, sin_t, phi = sample_cosine_hemisphere(N_SAMPLES, rng)
    dirs = frame_to_world(cos_t, sin_t*np.cos(phi), sin_t*np.sin(phi),
                          axis_n, axis_u, axis_v)
    norms = np.linalg.norm(dirs, axis=1)
    cos_with_n = (dirs @ axis_n) / norms

    # A) единичность
    unit_dev = float(np.max(np.abs(norms - 1.0)))
    # B) верхняя полусфера
    min_cos  = float(cos_with_n.min())

    # C) KS-тесты:
    #    pdf(t) = 2 cos theta sin t на [0, pi/2]
    #    Сделаем замену X = cos2_theta.  dX = -2 cos theta sin t dt
    #    pdf_X(x) = pdf_t(t) / |dX/dt| = (2 cos theta sint)/(2 cos theta sint) = 1
    #    То есть X = cos^2theta ~ U[0, 1].
    ks_cos2 = ks_test_uniform(cos_t**2, lo=0.0, hi=1.0)
    ks_phi  = ks_test_uniform(phi,      lo=0.0, hi=2*np.pi)

    # E) chi2 на сетке по cos2_theta x phi. По построению все ячейки равной "меры":
    #    каждой ячейке соответствует одинаковый интеграл от pdf.
    N_C, N_P = 10, 20
    K = N_C * N_P
    bin_c = np.clip((cos_t**2 * N_C).astype(int), 0, N_C-1)
    bin_p = np.clip((phi / (2*np.pi) * N_P).astype(int), 0, N_P-1)
    flat = bin_c * N_P + bin_p
    counts = np.bincount(flat, minlength=K)
    chi2_res = chi2_uniform(counts, N_SAMPLES / K)

    # D) гистограмма cos theta для визуальной проверки кривой pdf = 2 cos theta
    bins = np.linspace(0.0, 1.0, 21)
    hist, edges = np.histogram(cos_t, bins=bins)
    # нормировка к плотности: hist / (N * width)
    width = edges[1] - edges[0]
    pdf_emp   = hist / (N_SAMPLES * width)
    centers   = 0.5 * (edges[1:] + edges[:-1])
    pdf_theor = 2.0 * centers           # ожидаемая pdf(cos theta) = 2 cos theta

    print("Задание 4: косинусное распределение на полусфере вокруг N")
    print(f"  N = {N_SAMPLES}")
    print(f"  A) Единичность:            | |d|-1 | max = {unit_dev:.3e}")
    print(f"  B) Верхнее полушарие:      min cos(d, N) = {min_cos:.6f}  (>= 0)")
    print(f"  C) KS-тесты:")
    print(fmt_ks("cos^2theta ~ U[0, 1] (марг. cos theta для cos-pdf)", *ks_cos2))
    print(fmt_ks("phi ~ U[0, 2pi]",                              *ks_phi))
    print(f"  D) Сравнение эмпирической pdf(cos theta) с теоретической 2*cos theta:")
    print(f"     bin_center | pdf_emp | pdf_theor | err")
    for i in range(0, len(centers), 4):
        err = abs(pdf_emp[i] - pdf_theor[i])
        print(f"     {centers[i]:8.3f}  {pdf_emp[i]:7.4f}  {pdf_theor[i]:7.4f}  {err:7.4f}")
    print(f"  E) chi2 на {K} ячейках сетки cos2_theta x phi:")
    print(fmt_chi2("равные \"объёмы вероятности\"", *chi2_res))

    return {
        "unit_dev": unit_dev, "min_cos": min_cos,
        "ks_cos2": ks_cos2, "ks_phi": ks_phi,
        "chi2": chi2_res, "K": K,
        "cos_t": cos_t, "phi": phi, "dirs": dirs,
        "pdf_emp": pdf_emp, "pdf_theor": pdf_theor, "centers": centers,
    }
