"""Monte-Carlo sampling utilities for path tracing.

Includes cosine-weighted hemisphere sampling (for Lambertian BRDF),
uniform triangle sampling, and a local coordinate frame builder.
"""

import math
import numpy as np

from vec3 import vec3, normalize, dot, cross


def build_onb(n: np.ndarray):
    """Построить ортонормированный базис (касательная, бикасательная, нормаль) с осью Z = *n*.

    Вспомогательный вектор выбирается так, чтобы избежать почти коллинеарности с *n*:
    если нормаль почти совпадает с X, берётся (0,1,0), иначе (1,0,0). Результат —
    праворучная тройка (tangent, bitangent, normal) для перевода локальных направлений
    (из выборки по косинусу) в мировые координаты.
    """
    if abs(n[0]) > 0.9:
        helper = vec3(0.0, 1.0, 0.0)
    else:
        helper = vec3(1.0, 0.0, 0.0)
    tangent = normalize(cross(n, helper))
    bitangent = cross(n, tangent)
    return tangent, bitangent, n


def cosine_weighted_hemisphere(n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайное направление на полусфере над нормалью *n* с плотностью pdf ∝ cos(θ)/π.

    Это правильная важностная выборка для диффузной (ламбертовой) BRDF: направления
    ближе к нормали встречаются чаще. В локальной системе (z вверх) используется
    классическое преобразование равномерных (r1, r2) в (θ, φ), затем поворот в мир.
    """
    r1 = rng.random()
    r2 = rng.random()
    phi = 2.0 * math.pi * r1
    sqrt_r2 = math.sqrt(r2)

    # Local coordinates
    x = math.cos(phi) * sqrt_r2
    y = math.sin(phi) * sqrt_r2
    z = math.sqrt(1.0 - r2)

    t, b, _ = build_onb(n)
    return normalize(t * x + b * y + n * z)


def sample_triangle(tri, rng: np.random.Generator) -> np.ndarray:
    """Равномерно по площади выбрать точку на треугольнике *tri*.

    Два независимых числа u, v ∈ [0,1]; если u+v>1, отражаем относительно диагонали
    единичного квадрата барицентрических координат — стандартный трюк для равномерности
    на треугольнике. Точка: v0 + u·(v1−v0) + v·(v2−v0).
    """
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return tri.v0 + u * (tri.v1 - tri.v0) + v * (tri.v2 - tri.v0)
