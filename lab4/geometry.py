"""Triangle mesh primitives and vectorised Möller–Trumbore ray–triangle intersection."""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from vec3 import vec3, dot, cross, normalize


EPSILON = 1e-7


@dataclass
class Material:
    """Surface optical properties.

    ``diffuse`` and ``specular`` are RGB reflectance coefficients.
    Their per-channel sum must not exceed 1 (energy conservation).
    ``emission`` is the emitted radiance (non-zero only for light sources).
    """
    diffuse:  np.ndarray = None
    specular: np.ndarray = None
    emission: np.ndarray = None

    def __post_init__(self):
        """Подставить нулевые RGB-векторы там, где коэффициенты не заданы явно.

        Так материал всегда имеет три компонента: диффузное отражение, зеркальное и собственное излучение.
        """
        if self.diffuse is None:
            self.diffuse = vec3(0.0, 0.0, 0.0)
        if self.specular is None:
            self.specular = vec3(0.0, 0.0, 0.0)
        if self.emission is None:
            self.emission = vec3(0.0, 0.0, 0.0)


@dataclass
class Triangle:
    """Треугольник сцены: три вершины в пространстве и материал поверхности."""
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material: Material

    @property
    def normal(self) -> np.ndarray:
        """Единичная нормаль к плоскости треугольника (направление векторного произведения рёбер).

        Ориентация зависит от порядка вершин v0→v1→v2 (правило правой руки).
        """
        return normalize(cross(self.v1 - self.v0, self.v2 - self.v0))

    @property
    def area(self) -> float:
        """Площадь треугольника — половина модуля векторного произведения рёбер.

        Нужна для равномерной выборки точки на треугольнике-источнике и для весов мощности света.
        """
        return float(0.5 * np.linalg.norm(cross(self.v1 - self.v0, self.v2 - self.v0)))


@dataclass
class HitRecord:
    """Результат пересечения луча с поверхностью: расстояние, точка, нормаль, материал."""
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material


class SceneAccelerator:
    """Pre-computes triangle data for vectorised intersection tests."""

    def __init__(self, triangles: List[Triangle]):
        """Запаковать список треугольников в массивы NumPy для пакетного пересечения одним лучом.

        Для каждого треугольника хранятся v0 и два ребра (edge1, edge2); нормали дублируются
        для быстрого доступа после выбора ближайшего попадания.
        """
        self.triangles = triangles
        n = len(triangles)
        self.v0 = np.array([t.v0 for t in triangles])           # (N, 3)
        self.edge1 = np.array([t.v1 - t.v0 for t in triangles]) # (N, 3)
        self.edge2 = np.array([t.v2 - t.v0 for t in triangles]) # (N, 3)
        self.normals = np.array([t.normal for t in triangles])   # (N, 3)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[HitRecord]:
        """Найти ближайшее пересечение луча origin + t·direction с любым треугольником сцены.

        Алгоритм Мёллера–Трумбора для каждого треугольника; все треугольники проверяются
        векторизованно за один проход. Из всех допустимых попаданий (t > 0, точка внутри треугольника)
        выбирается минимальное t. Нормаль разворачивается против направления луча,
        чтобы она всегда была «внешней» относительно пришедшего луча.
        """
        h = np.cross(direction, self.edge2)             # (N, 3)
        a = np.sum(self.edge1 * h, axis=1)              # (N,)

        valid = np.abs(a) > EPSILON
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]

        s = origin - self.v0                             # (N, 3)
        u = f * np.sum(s * h, axis=1)
        valid &= (u >= 0.0) & (u <= 1.0)

        q = np.cross(s, self.edge1)
        v = f * np.sum(direction * q, axis=1)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t = f * np.sum(self.edge2 * q, axis=1)
        valid &= t > EPSILON

        if not np.any(valid):
            return None

        t[~valid] = np.inf
        idx = int(np.argmin(t))
        t_hit = t[idx]

        point = origin + t_hit * direction
        normal = self.normals[idx].copy()
        if np.dot(normal, direction) > 0:
            normal = -normal

        return HitRecord(
            t=t_hit,
            point=point,
            normal=normal,
            material=self.triangles[idx].material,
        )
