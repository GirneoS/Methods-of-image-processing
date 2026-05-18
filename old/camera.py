"""Pinhole camera model with configurable position, orientation, and field of view."""

import math
import numpy as np

from vec3 import vec3, normalize, cross


class Camera:
    """Точечная (pinhole) камера: лучи выходят из одной точки *eye* через прямоугольное окно изображения."""

    def __init__(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = vec3(0, 1, 0),
        fov_deg: float = 60.0,
        width: int = 512,
        height: int = 512,
    ):
        """Настроить камеру: позиция, точка взгляда, «верх» мира, вертикальный FOV и размер кадра.

        Строится ортонормированный базис: forward к цели, right = forward × up, cam_up = right × forward.
        Плоскость изображения задаётся на единичном расстоянии вдоль forward; половины углов
        half_w, half_h определяют размер «окна» по тангенсу половины FOV и соотношению сторон.
        Углы пиксельной сетки интерполируются от нижнего левого угла окна (_lower_left)
        вдоль векторов _horizontal и _vertical.
        """
        self.eye = eye
        self.width = width
        self.height = height

        forward = normalize(target - eye)
        right = normalize(cross(forward, up))
        cam_up = cross(right, forward)

        aspect = width / height
        half_h = math.tan(math.radians(fov_deg) / 2.0)
        half_w = half_h * aspect

        self._lower_left = forward - half_w * right - half_h * cam_up
        self._horizontal = 2.0 * half_w * right
        self._vertical = 2.0 * half_h * cam_up

    def get_ray(self, px: float, py: float) -> np.ndarray:
        """Направление (единичный вектор) луча из *eye* через дробные координаты пикселя *(px, py)*.

        u = px/width, v = py/height пробегают [0,1) по кадру; при подстановке случайного смещения
        внутри пикселя (в render) получается антиалиасинг. Возвращаемое значение — только
        направление; начало луча всегда camera.eye.
        """
        u = px / self.width
        v = py / self.height
        direction = self._lower_left + u * self._horizontal + v * self._vertical
        return normalize(direction)
