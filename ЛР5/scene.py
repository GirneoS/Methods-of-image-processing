"""Cornell Box scene (взят из ЛР4) с присвоением object_id каждому объекту.

object_id:
    1 - пол
    2 - потолок
    3 - задняя стена
    4 - левая (красная) стена
    5 - правая (зелёная) стена
    6 - короткий белый блок
    7 - высокий зеркальный блок
    8 - источник света (на потолке)
"""

from typing import List, Tuple

import numpy as np

from vec3 import vec3
from geometry import Triangle, Material
from camera import Camera


def _quad(v0, v1, v2, v3, mat: Material, obj_id: int) -> List[Triangle]:
    return [
        Triangle(v0, v1, v2, mat, object_id=obj_id),
        Triangle(v0, v2, v3, mat, object_id=obj_id),
    ]


def build_cornell_box() -> Tuple[List[Triangle], List[Triangle], Camera]:
    white  = Material(diffuse=vec3(0.73, 0.73, 0.73))
    red    = Material(diffuse=vec3(0.65, 0.05, 0.05))
    green  = Material(diffuse=vec3(0.12, 0.45, 0.15))
    mirror = Material(specular=vec3(0.95, 0.95, 0.95))
    light_mat = Material(emission=vec3(17.0, 12.0, 4.0))

    tris: List[Triangle] = []

    # Floor (id=1)
    tris += _quad(vec3(552.8, 0, 0), vec3(0, 0, 0),
                  vec3(0, 0, 559.2), vec3(549.6, 0, 559.2), white, 1)

    # Ceiling (id=2)
    tris += _quad(vec3(556.0, 548.8, 0), vec3(556.0, 548.8, 559.2),
                  vec3(0, 548.8, 559.2), vec3(0, 548.8, 0), white, 2)

    # Back wall (id=3)
    tris += _quad(vec3(549.6, 0, 559.2), vec3(0, 0, 559.2),
                  vec3(0, 548.8, 559.2), vec3(556.0, 548.8, 559.2), white, 3)

    # Left wall - red (id=4)
    tris += _quad(vec3(552.8, 0, 0), vec3(549.6, 0, 559.2),
                  vec3(556.0, 548.8, 559.2), vec3(556.0, 548.8, 0), red, 4)

    # Right wall - green (id=5)
    tris += _quad(vec3(0, 0, 559.2), vec3(0, 0, 0),
                  vec3(0, 548.8, 0), vec3(0, 548.8, 559.2), green, 5)

    # Short white block (id=6)
    bh = 165.0
    tris += _quad(vec3(130, bh, 65), vec3(82, bh, 225),
                  vec3(240, bh, 272), vec3(290, bh, 114), white, 6)
    tris += _quad(vec3(290, 0, 114), vec3(290, bh, 114),
                  vec3(240, bh, 272), vec3(240, 0, 272), white, 6)
    tris += _quad(vec3(130, 0, 65), vec3(130, bh, 65),
                  vec3(290, bh, 114), vec3(290, 0, 114), white, 6)
    tris += _quad(vec3(82, 0, 225), vec3(82, bh, 225),
                  vec3(130, bh, 65), vec3(130, 0, 65), white, 6)
    tris += _quad(vec3(240, 0, 272), vec3(240, bh, 272),
                  vec3(82, bh, 225), vec3(82, 0, 225), white, 6)

    # Tall mirror block (id=7)
    th = 330.0
    tris += _quad(vec3(423, th, 247), vec3(265, th, 296),
                  vec3(314, th, 456), vec3(472, th, 406), mirror, 7)
    tris += _quad(vec3(423, 0, 247), vec3(423, th, 247),
                  vec3(472, th, 406), vec3(472, 0, 406), mirror, 7)
    tris += _quad(vec3(472, 0, 406), vec3(472, th, 406),
                  vec3(314, th, 456), vec3(314, 0, 456), mirror, 7)
    tris += _quad(vec3(314, 0, 456), vec3(314, th, 456),
                  vec3(265, th, 296), vec3(265, 0, 296), mirror, 7)
    tris += _quad(vec3(265, 0, 296), vec3(265, th, 296),
                  vec3(423, th, 247), vec3(423, 0, 247), mirror, 7)

    # Ceiling light (id=8)
    light_tris = _quad(vec3(343, 548.7, 227), vec3(343, 548.7, 332),
                       vec3(213, 548.7, 332), vec3(213, 548.7, 227), light_mat, 8)
    tris += light_tris

    cam = Camera(
        eye=vec3(278, 273, -800),
        target=vec3(278, 273, 0),
        fov_deg=39.3,
        width=512,
        height=512,
    )
    return tris, light_tris, cam
