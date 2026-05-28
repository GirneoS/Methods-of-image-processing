"""Lightweight 3-component vector helpers built on NumPy arrays.

Every public function accepts and returns plain ``np.ndarray`` of shape (3,)``
so the rest of the code stays free of wrapper-class overhead while remaining
readable.
"""

import numpy as np


def vec3(x: float, y: float, z: float) -> np.ndarray:
    """Собрать трёхмерный вектор (или точку) как массив NumPy из трёх вещественных чисел.

    Используется по всему рендереру для координат, цветов RGB и направлений лучей.
    Тип float64 даёт достаточную точность для трассировки.
    """
    return np.array([x, y, z], dtype=np.float64)


def length(v: np.ndarray) -> float:
    """Длина (евклидова норма) вектора *v* — расстояние от начала координат до конца вектора."""
    return np.linalg.norm(v)


def normalize(v: np.ndarray) -> np.ndarray:
    """Вернуть вектор единичной длины с тем же направлением, что и *v*.

    Если *v* нулевой, возвращается *v* без изменений (избегаем деления на ноль).
    Нужна для направлений лучей и нормалей к поверхностям.
    """
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Скалярное произведение векторов *a* и *b*.

    Применяется в освещении: косинус угла между нормалью и направлением света,
    проверка видимости стороны треугольника и т.д.
    """
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Векторное произведение *a* × *b* — перпендикуляр к обоим векторам.

    Используется для построения орта камеры (right, up) и для нормали треугольника
    как произведение рёбер v1−v0 и v2−v0.
    """
    return np.cross(a, b)


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Отразить вектор *v* относительно единичной нормали поверхности *n*.

    Формула: v − 2(v·n)n — закон зеркального отражения для направления луча.
    *v* и *n* предполагаются в одной системе координат (например, направление падения и наружная нормаль).
    """
    return v - 2.0 * dot(v, n) * n
