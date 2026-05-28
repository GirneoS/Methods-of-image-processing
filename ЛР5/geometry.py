"""Triangle mesh primitives + ray-triangle intersection (Mller-Trumbore).

В отличие от ЛР4 здесь у Triangle и HitRecord есть поле ``object_id`` -
логический номер объекта сцены. Нужен для билатерального фильтра: пиксели
разных объектов не смешиваются (жёсткая маска границы).
"""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from vec3 import vec3, dot, cross, normalize


EPSILON = 1e-7


@dataclass
class Material:
    diffuse:  np.ndarray = None
    specular: np.ndarray = None
    emission: np.ndarray = None

    def __post_init__(self):
        if self.diffuse  is None: self.diffuse  = vec3(0, 0, 0)
        if self.specular is None: self.specular = vec3(0, 0, 0)
        if self.emission is None: self.emission = vec3(0, 0, 0)


@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material: Material
    object_id: int = 0       # NEW: логический id объекта сцены

    @property
    def normal(self) -> np.ndarray:
        return normalize(cross(self.v1 - self.v0, self.v2 - self.v0))

    @property
    def area(self) -> float:
        return float(0.5 * np.linalg.norm(cross(self.v1 - self.v0, self.v2 - self.v0)))


@dataclass
class HitRecord:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material
    object_id: int = 0       # NEW: id объекта, в который попал луч


class SceneAccelerator:
    """Vectorised intersection over a triangle list."""

    def __init__(self, triangles: List[Triangle]):
        self.triangles = triangles
        self.v0      = np.array([t.v0 for t in triangles])
        self.edge1   = np.array([t.v1 - t.v0 for t in triangles])
        self.edge2   = np.array([t.v2 - t.v0 for t in triangles])
        self.normals = np.array([t.normal for t in triangles])
        self.obj_ids = np.array([t.object_id for t in triangles], dtype=np.int32)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[HitRecord]:
        h = np.cross(direction, self.edge2)
        a = np.sum(self.edge1 * h, axis=1)

        valid = np.abs(a) > EPSILON
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]

        s = origin - self.v0
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

        point  = origin + t_hit * direction
        normal = self.normals[idx].copy()
        if np.dot(normal, direction) > 0:
            normal = -normal

        return HitRecord(
            t=t_hit,
            point=point,
            normal=normal,
            material=self.triangles[idx].material,
            object_id=int(self.obj_ids[idx]),
        )
