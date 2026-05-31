"""
Простые наглядные доказательства равномерности по образцу проверяющего.

Идея: внутри большой фигуры размещаем НЕСКОЛЬКО ОДИНАКОВЫХ маленьких фигурок
(квадратиков / колпаков / конусов). Считаем, сколько случайных точек попало
в каждую. Если распределение равномерное — числа должны быть БЛИЗКИМИ.

  Треугольник  -> 3 одинаковых квадрата внутри
  Круг         -> 4 одинаковых квадрата внутри
  Сфера        -> 3 одинаковых сферических колпака
  Косинус      -> 6 конусов от вектора N (числа должны падать по cos(угла))
"""
from __future__ import annotations

import numpy as np


# ── Треугольник: 3 одинаковых квадрата внутри ────────────────────────────────

def count_in_squares_triangle(points_2d: np.ndarray, squares: list) -> list[int]:
    """Подсчитать число точек в каждом квадрате. squares = [(x0,y0,side), ...]."""
    out = []
    for (x0, y0, side) in squares:
        m = ((points_2d[:, 0] >= x0) & (points_2d[:, 0] <= x0 + side)
             & (points_2d[:, 1] >= y0) & (points_2d[:, 1] <= y0 + side))
        out.append(int(m.sum()))
    return out


def proof_triangle(points_2d: np.ndarray) -> tuple[list, list]:
    """3 одинаковых квадрата размером 0.18 внутри равностороннего треугольника
    с вершинами (0,0), (1,0), (0.5, sqrt(3)/2)."""
    side = 0.15
    squares = [
        (0.10, 0.05, side),                 # нижний-левый
        (0.75, 0.05, side),                 # нижний-правый
        (0.425, 0.50, side),                # верх
    ]
    counts = count_in_squares_triangle(points_2d, squares)
    return squares, counts


# ── Круг: 4 одинаковых квадрата внутри ────────────────────────────────────────

def proof_circle(points_2d: np.ndarray) -> tuple[list, list]:
    """4 одинаковых квадрата размером 0.4 внутри круга радиуса 1."""
    side = 0.4
    squares = [
        (-0.55, -0.55, side),
        ( 0.15, -0.55, side),
        (-0.55,  0.15, side),
        ( 0.15,  0.15, side),
    ]
    counts = []
    for (x0, y0, s) in squares:
        m = ((points_2d[:, 0] >= x0) & (points_2d[:, 0] <= x0 + s)
             & (points_2d[:, 1] >= y0) & (points_2d[:, 1] <= y0 + s))
        counts.append(int(m.sum()))
    return squares, counts


# ── Сфера: 3 одинаковых сферических колпака ──────────────────────────────────

def proof_sphere(dirs: np.ndarray) -> tuple[list, list]:
    """3 одинаковых колпака (cap) с угловым радиусом 20° вокруг трёх осей."""
    cap_angle = np.deg2rad(20.0)
    cos_min = np.cos(cap_angle)
    centers = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    counts = []
    for c in centers:
        cosines = dirs @ c
        counts.append(int(np.sum(cosines >= cos_min)))
    return centers, counts, cap_angle


# ── Косинус: конусы с растущим углом к N ─────────────────────────────────────

def proof_cosine(dirs: np.ndarray, axis_n: np.ndarray) -> tuple[list, list, list]:
    """
    Делим полусферу на 6 поясов по углу с N: [0-15], [15-30], [30-45],
    [45-60], [60-75], [75-90] градусов.

    Для КОСИНУСНОГО распределения число точек в i-м поясе:
        N_i = N * (cos²θ_lo - cos²θ_hi)
    То есть оно ПАДАЕТ с ростом угла — это и есть отличие от равномерного.

    Возвращает (границы углов в радианах, наблюдаемые N_i, теоретические N_i).
    """
    cos_n = dirs @ axis_n
    edges_deg = [0, 15, 30, 45, 60, 75, 90]
    edges = np.deg2rad(edges_deg)
    N = len(dirs)
    observed, theoretical = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (cos_n >= np.cos(hi)) & (cos_n < np.cos(lo) + 1e-12)
        observed.append(int(m.sum()))
        # CDF для cos-pdf: F(θ) = sin²θ. Доля в интервале = sin²hi - sin²lo
        theoretical.append(int(round(N * (np.sin(hi)**2 - np.sin(lo)**2))))
    return edges_deg, observed, theoretical
