"""
Статистические тесты для доказательства распределений (ЛР3).

Реализованы в чистом NumPy, без scipy:

1) KS-тест (Колмогорова-Смирнова): сравнивает эмпирическую функцию
   распределения F_n(x) с теоретической F(x). Если D = max|F_n - F|
   меньше критического значения D_crit ~= 1.358/sqrt(N) - гипотеза
   о соответствии распределению НЕ отвергается на уровне α=0.05.

2) χ^2 (хи-квадрат): разбиваем область определения на K непересекающихся
   ячеек одинаковой меры (по теоретическому распределению), считаем число
   точек O_i, попавших в каждую. При равномерном распределении ожидание
   E_i = N/K. Статистика: chi2 = sum( (O_i - E_i)^2 / E_i ).
   Сравнивается с критическим значением χ^2(K-1, α=0.05).
"""

from __future__ import annotations

import numpy as np


# ── KS-тест ───────────────────────────────────────────────────────────────────

def ks_test_uniform(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> tuple[float, float, bool]:
    """
    Проверить гипотезу, что выборка x имеет равномерное распределение U[lo, hi].
    Возвращает (D, D_crit_alpha005, passed).
    """
    x = np.asarray(x).ravel()
    n = len(x)
    # переводим x → u  in  [0,1]: u = (x - lo) / (hi - lo)
    u = (x - lo) / (hi - lo)
    u_sorted = np.sort(u)
    # эмпирическая CDF
    F_emp_lo = np.arange(n)        / n   # F_n(x_i^-)
    F_emp_hi = (np.arange(n) + 1)  / n   # F_n(x_i^+)
    # теоретическая CDF U[0,1]: F(u) = u
    D_lo = np.max(np.abs(F_emp_lo - u_sorted))
    D_hi = np.max(np.abs(F_emp_hi - u_sorted))
    D = float(max(D_lo, D_hi))
    D_crit = 1.358 / np.sqrt(n)    # α = 0.05
    return D, float(D_crit), bool(D <= D_crit)


def ks_test_cdf(x: np.ndarray, cdf_func) -> tuple[float, float, bool]:
    """
    KS-тест для произвольной CDF F(x). cdf_func(x) - функция, возвращающая F(x).
    """
    x = np.asarray(x).ravel()
    n = len(x)
    x_sorted = np.sort(x)
    F_theor = cdf_func(x_sorted)
    F_emp_lo = np.arange(n)       / n
    F_emp_hi = (np.arange(n) + 1) / n
    D = float(max(
        np.max(np.abs(F_emp_lo - F_theor)),
        np.max(np.abs(F_emp_hi - F_theor))))
    D_crit = 1.358 / np.sqrt(n)
    return D, float(D_crit), bool(D <= D_crit)


# ── χ^2 критическое значение (без scipy) ────────────────────────────────────────

_CHI2_CRIT_005 = {
    9:  16.92, 19: 30.14, 24: 36.42, 29: 42.56, 49: 66.34,
    63: 82.53, 99: 123.23, 124: 151.45, 199: 233.99, 399: 446.07,
}

def chi2_critical(df: int, alpha: float = 0.05) -> float:
    """Приближение χ^2_crit при α=0.05 (таблично + Wilson-Hilferty для df>=30)."""
    if df in _CHI2_CRIT_005:
        return _CHI2_CRIT_005[df]
    # Wilson-Hilferty: χ^2(df, α) ~= df * (1 - 2/(9 df) + z_α * sqrt(2/(9 df)))^3
    z_005 = 1.6449  # для α=0.05 (одностороннее)
    factor = 1.0 - 2.0 / (9 * df) + z_005 * np.sqrt(2.0 / (9 * df))
    return float(df * factor ** 3)


# ── χ^2 тест ───────────────────────────────────────────────────────────────────

def chi2_uniform(counts: np.ndarray, expected: float | np.ndarray) -> tuple[float, float, bool]:
    """
    Хи-квадрат тест: counts - массив наблюдаемых N_i, expected - ожидание
    (скаляр или массив). Возвращает (chi2, crit, passed).
    """
    counts = np.asarray(counts, dtype=float).ravel()
    if np.isscalar(expected):
        E = np.full_like(counts, float(expected))
    else:
        E = np.asarray(expected, dtype=float).ravel()
    mask = E > 0
    chi2 = float(np.sum((counts[mask] - E[mask]) ** 2 / E[mask]))
    df = int(mask.sum()) - 1
    crit = chi2_critical(df)
    return chi2, crit, bool(chi2 <= crit)


# ── Удобный вывод ─────────────────────────────────────────────────────────────

def fmt_ks(name: str, D: float, D_crit: float, passed: bool) -> str:
    verdict = "PASS" if passed else "FAIL"
    return f"  {name:<40s} D = {D:.5f}  D_crit = {D_crit:.5f}  {verdict}"


def fmt_chi2(name: str, chi2: float, crit: float, passed: bool) -> str:
    verdict = "PASS" if passed else "FAIL"
    return f"  {name:<40s} chi2 = {chi2:7.2f}  crit = {crit:7.2f}  {verdict}"
