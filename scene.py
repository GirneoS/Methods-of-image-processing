"""Pre-built Cornell Box scene and OBJ loader utility."""

from typing import List, Tuple

import numpy as np

from vec3 import vec3
from geometry import Triangle, Material
from camera import Camera


def _quad(v0, v1, v2, v3, mat: Material) -> List[Triangle]:
    """Разбить четырёхугольник на два треугольника с общим материалом *mat*.

    Вершины v0, v1, v2, v3 задаются в порядке обхода против часовой стрелки при взгляде
    на внешнюю сторону грани — так согласованы нормали с остальной сценой Cornell Box.
    Диагональ: (v0, v2).
    """
    return [Triangle(v0, v1, v2, mat), Triangle(v0, v2, v3, mat)]


def build_cornell_box() -> Tuple[List[Triangle], List[Triangle], Camera]:
    """Собрать классическую сцену Cornell Box: комната, два блока, площадной свет, камера.

    Возвращает три значения:
    - полный список треугольников для пересечений (включая люк в потолке);
    - только треугольники источника света (те же объекты — для выборки в NEE);
    - эталонную камеру (в main может быть пересоздана с другим разрешением/FOV).

    Единицы и координаты соответствуют распространённым данным Cornell Box в миллиметрах.
    """
    white = Material(diffuse=vec3(0.73, 0.73, 0.73), specular=vec3(0.73, 0.73, 0.73))
    red   = Material(diffuse=vec3(0.65, 0.05, 0.05))
    green = Material(diffuse=vec3(0.12, 0.45, 0.15))
    mirror = Material(specular=vec3(0.95, 0.95, 0.95))
    light_mat = Material(emission=vec3(17.0, 12.0, 4.0))

    tris: List[Triangle] = []

    # Floor (y = 0)
    tris += _quad(vec3(552.8, 0, 0), vec3(0, 0, 0),
                  vec3(0, 0, 559.2), vec3(549.6, 0, 559.2), white)

    # Ceiling (y = 548.8)
    tris += _quad(vec3(556.0, 548.8, 0), vec3(556.0, 548.8, 559.2),
                  vec3(0, 548.8, 559.2), vec3(0, 548.8, 0), white)

    # Back wall (z = 559.2)
    tris += _quad(vec3(549.6, 0, 559.2), vec3(0, 0, 559.2),
                  vec3(0, 548.8, 559.2), vec3(556.0, 548.8, 559.2), white)

    # Left wall – red (x = 552.8)
    tris += _quad(vec3(552.8, 0, 0), vec3(549.6, 0, 559.2),
                  vec3(556.0, 548.8, 559.2), vec3(556.0, 548.8, 0), red)

    # Right wall – green (x = 0)
    tris += _quad(vec3(0, 0, 559.2), vec3(0, 0, 0),
                  vec3(0, 548.8, 0), vec3(0, 548.8, 559.2), green)

    # ---------- Short block (diffuse white) ----------
    bh = 165.0
    tris += _quad(vec3(130, bh, 65), vec3(82, bh, 225),
                  vec3(240, bh, 272), vec3(290, bh, 114), white)
    tris += _quad(vec3(290, 0, 114), vec3(290, bh, 114),
                  vec3(240, bh, 272), vec3(240, 0, 272), white)
    tris += _quad(vec3(130, 0, 65), vec3(130, bh, 65),
                  vec3(290, bh, 114), vec3(290, 0, 114), white)
    tris += _quad(vec3(82, 0, 225), vec3(82, bh, 225),
                  vec3(130, bh, 65), vec3(130, 0, 65), white)
    tris += _quad(vec3(240, 0, 272), vec3(240, bh, 272),
                  vec3(82, bh, 225), vec3(82, 0, 225), white)

    # ---------- Tall block (mirror) ----------
    th = 330.0
    tris += _quad(vec3(423, th, 247), vec3(265, th, 296),
                  vec3(314, th, 456), vec3(472, th, 406), mirror)
    tris += _quad(vec3(423, 0, 247), vec3(423, th, 247),
                  vec3(472, th, 406), vec3(472, 0, 406), mirror)
    tris += _quad(vec3(472, 0, 406), vec3(472, th, 406),
                  vec3(314, th, 456), vec3(314, 0, 456), mirror)
    tris += _quad(vec3(314, 0, 456), vec3(314, th, 456),
                  vec3(265, th, 296), vec3(265, 0, 296), mirror)
    tris += _quad(vec3(265, 0, 296), vec3(265, th, 296),
                  vec3(423, th, 247), vec3(423, 0, 247), mirror)

    # ---------- Ceiling light ----------
    light_tris = _quad(vec3(343, 548.7, 227), vec3(343, 548.7, 332),
                       vec3(213, 548.7, 332), vec3(213, 548.7, 227), light_mat)
    tris += light_tris

    cam = Camera(
        eye=vec3(278, 273, -800),
        target=vec3(278, 273, 0),
        fov_deg=39.3,
        width=512,
        height=512,
    )

    return tris, light_tris, cam


def load_obj(filepath: str, mat: Material, scale: float = 1.0) -> List[Triangle]:
    """Загрузить простой OBJ: только строки вершин «v» и граней «f» (треугольники или полигоны).

    Вершины умножаются на *scale*. Многоугольные грани веером разбиваются на треугольники
    (0, i, i+1). Индексы в OBJ с 1, в коде — с 0. Всем граням назначается один материал *mat*.
    Нормали/текстурные координаты из OBJ не читаются.
    """
    vertices = []
    triangles = []

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                v = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * scale
                vertices.append(v)
            elif parts[0] == "f":
                idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idxs) - 1):
                    triangles.append(Triangle(
                        vertices[idxs[0]],
                        vertices[idxs[i]],
                        vertices[idxs[i + 1]],
                        mat,
                    ))

    return triangles
