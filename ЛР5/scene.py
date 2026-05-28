"""Simple scene definition: spheres and planes for G-buffer rendering."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Material:
    color: np.ndarray      # RGB diffuse colour
    emission: np.ndarray   # RGB emission
    roughness: float = 1.0


@dataclass
class HitInfo:
    t: float
    point: np.ndarray
    normal: np.ndarray
    obj_id: int
    material: Material


class Sphere:
    def __init__(self, center, radius, material, obj_id):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.material = material
        self.obj_id = obj_id

    def intersect(self, origin, direction) -> Optional[HitInfo]:
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t = (-b - sq) / (2 * a)
        if t < 1e-4:
            t = (-b + sq) / (2 * a)
        if t < 1e-4:
            return None
        point = origin + t * direction
        normal = (point - self.center) / self.radius
        return HitInfo(t, point, normal, self.obj_id, self.material)


class Plane:
    """Axis-aligned plane (infinite)."""
    def __init__(self, axis: int, value: float, side: int, material, obj_id):
        self.axis = axis     # 0=x, 1=y, 2=z
        self.value = value
        self.side = side     # +1 or -1 normal direction
        self.material = material
        self.obj_id = obj_id

    def intersect(self, origin, direction) -> Optional[HitInfo]:
        d = direction[self.axis]
        if abs(d) < 1e-9:
            return None
        t = (self.value - origin[self.axis]) / d
        if t < 1e-4:
            return None
        point = origin + t * direction
        normal = np.zeros(3)
        normal[self.axis] = float(self.side)
        return HitInfo(t, point, normal, self.obj_id, self.material)


def build_cornell_box():
    """Cornell Box: walls + 2 spheres + light (sphere on top)."""
    white  = Material(np.array([0.73, 0.73, 0.73]), np.zeros(3))
    red    = Material(np.array([0.65, 0.05, 0.05]), np.zeros(3))
    green  = Material(np.array([0.12, 0.45, 0.15]), np.zeros(3))
    blue_m = Material(np.array([0.10, 0.20, 0.70]), np.zeros(3))
    yellow = Material(np.array([0.80, 0.70, 0.10]), np.zeros(3))
    light  = Material(np.array([0.0,  0.0,  0.0 ]), np.array([15.0, 15.0, 15.0]))

    objects = [
        # Walls
        Plane(0,  0.0,  1, red,   obj_id=1),   # left
        Plane(0,  5.5, -1, green, obj_id=2),   # right
        Plane(1,  0.0,  1, white, obj_id=3),   # floor
        Plane(1,  5.5, -1, white, obj_id=4),   # ceiling
        Plane(2, 10.0, -1, white, obj_id=5),   # back
        # Spheres
        Sphere([2.0, 1.0, 7.5], 1.0, blue_m,  obj_id=6),
        Sphere([3.8, 1.2, 6.0], 1.2, yellow,  obj_id=7),
        # Light sphere near ceiling
        Sphere([2.75, 5.0, 7.0], 0.6, light,  obj_id=0),
    ]
    return objects
