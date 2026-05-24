"""
ЛР_1. Вычисление яркости в точках на поверхности треугольника.
Курс: Методы обработки изображений.

Формулы:
  I(theta) = I0 * D(theta)             -- цветная сила излучения источника
  E = I(theta) * cos(alpha) / R^2      -- цветная освещённость точки
  P = P0 + u*e1 + v*e2                 -- локальные -> глобальные координаты
  N = normalize((P1-P0) x (P2-P0))     -- нормаль плоскости
  L_vec = light_pos - P                -- вектор до источника
  BRDF = kd/pi + ks*(m+2)/(2*pi)*(H·N)^m
  H = normalize(L + V)                 -- средний вектор
  brightness = sum_i( E_i * BRDF * surface_color )
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

class Light:
    """Точечный источник света.

    position  : (3,) координаты в мировом пространстве
    intensity : (3,) «цветная» сила излучения I0 (R,G,B)
    axis      : (3,) ось источника (None -> изотропный)
    m_spot    : показатель диаграммы направленности D(θ) = cos^m(θ)
    """

    def __init__(self, position, intensity, axis=None, m_spot=1.0):
        self.position = np.array(position, dtype=float)
        self.intensity = np.array(intensity, dtype=float)
        self.axis = np.array(axis, dtype=float) if axis is not None else None
        self.m_spot = float(m_spot)

    def directional_intensity(self, direction_to_point):
        """I(θ) = I0 * D(θ).  direction_to_point — единичный вектор."""
        if self.axis is None:
            return self.intensity.copy()
        cos_theta = np.dot(self.axis, direction_to_point)
        cos_theta = max(0.0, cos_theta)
        return self.intensity * (cos_theta ** self.m_spot)


# ---------------------------------------------------------------------------
# Геометрические функции
# ---------------------------------------------------------------------------

def compute_normal(p0, p1, p2):
    """Единичная нормаль плоскости треугольника (p0, p1, p2)."""
    e1 = np.array(p1, dtype=float) - np.array(p0, dtype=float)
    e2 = np.array(p2, dtype=float) - np.array(p0, dtype=float)
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Точки коллинеарны — нормаль не определена.")
    return n / norm


def local_to_global(p0, e1, e2, u, v):
    """P = P0 + u*e1 + v*e2."""
    return np.array(p0, dtype=float) + u * np.array(e1, dtype=float) + v * np.array(e2, dtype=float)


# ---------------------------------------------------------------------------
# Освещение
# ---------------------------------------------------------------------------

def compute_illuminance(light: Light, point, normal):
    """Освещённость E = I(θ) * cos(α) / R².

    Возвращает (E_rgb, L_dir_unit).
    """
    vec = light.position - np.array(point, dtype=float)
    R = np.linalg.norm(vec)
    if R < 1e-12:
        return np.zeros(3), np.zeros(3)


    L_dir = vec / R

    cos_alpha = max(0.0, np.dot(L_dir, np.array(normal, dtype=float)))
    I_dir = light.directional_intensity(-L_dir)   # ось смотрит от источника
    E = I_dir * cos_alpha / (R * R)
    return E, L_dir


def halfway_vector(L, V):
    """H = normalize(L + V)."""
    h = L + V
    n = np.linalg.norm(h)
    return h / n if n > 1e-12 else np.zeros(3)


def brdf(kd, ks, m, H, N):
    """Модель Блинна–Фонга.

    kd — коэффициент диффузного отражения
    ks — коэффициент зеркального отражения
    m  — показатель блика
    """
    HdotN = max(0.0, np.dot(H, N))
    diffuse = kd / math.pi
    specular = ks * (m + 2) / (2.0 * math.pi) * (HdotN ** m)
    return diffuse + specular


def compute_brightness(point, normal, lights, view_dir, surface_color, kd, ks, m):
    """Яркость L = Σ E_i * BRDF * surface_color.

    Parameters
    ----------
    point         : мировые координаты точки
    normal        : единичная нормаль поверхности
    lights        : list[Light]
    view_dir      : направление наблюдения (от камеры к точке)
    surface_color : (3,) «цвет» поверхности
    kd, ks, m     : параметры BRDF
    """
    point = np.array(point, dtype=float)
    normal = np.array(normal, dtype=float)

    # V — направление от точки к наблюдателю (обратно view_dir)
    V = -np.array(view_dir, dtype=float)
    V /= np.linalg.norm(V)


    if np.dot(V, normal) < 0:
        return np.zeros(3)

    total = np.zeros(3)
    for light in lights:
        E, L_dir = compute_illuminance(light, point, normal)
        H = halfway_vector(L_dir, V)
        b = brdf(kd, ks, m, H, normal)
        total += E * b * np.array(surface_color, dtype=float)

    return total


# ---------------------------------------------------------------------------
# Демонстрация (входные данные + расчёт)
# ---------------------------------------------------------------------------

def demo():
    # --- Треугольник ---
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = np.array([4.0, 0.0, 0.0])
    P2 = np.array([0.0, 4.0, 0.0])

    e1 = P1 - P0   # ребро 1
    e2 = P2 - P0   # ребро 2
    normal = compute_normal(P0, P1, P2)

    print("=" * 60)
    print("Треугольник: P0={}, P1={}, P2={}".format(P0, P1, P2))
    print("Нормаль:    ", normal)

    # --- Источники света ---
    lights = [
        Light(position=[2.0, 2.0, 5.0], intensity=[1.0, 1.0, 1.0],
              axis=None),                           # изотропный белый
        Light(position=[-3.0, 3.0, 4.0], intensity=[0.8, 0.2, 0.2],
              axis=[0.5, -0.5, -1.0], m_spot=2.0),  # красный направленный
    ]

    # --- Материал и наблюдение ---
    surface_color = np.array([0.6, 0.8, 0.6])  # зеленоватая поверхность
    kd, ks, m_shine = 0.7, 0.3, 20.0
    view_dir = np.array([0.0, 0.0, 1.0])       # камера сверху

    # --- Локальные координаты тестовых точек ---
    local_points = [
        (0.0, 0.0),   # вершина P0
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.25, 0.25),
    ]

    print("\n{:=<60}".format(""))
    print("1. Освещённость по локальным координатам")
    print("{:-<60}".format(""))
    print("{:<8} {:<8} | {:<12} {:<12} {:<12}".format(
        "u", "v", "E.R", "E.G", "E.B"))

    global_points = []
    for u, v in local_points:
        P_global = local_to_global(P0, e1, e2, u, v)
        global_points.append(P_global)
        E_total = np.zeros(3)
        for light in lights:
            E, _ = compute_illuminance(light, P_global, normal)
            E_total += E
        print("{:<8.3f} {:<8.3f} | {:<12.5f} {:<12.5f} {:<12.5f}".format(
            u, v, E_total[0], E_total[1], E_total[2]))

    print("\n{:=<60}".format(""))
    print("2. Освещённость по глобальным координатам")
    print("{:-<60}".format(""))
    print("{:<30} | {:<12} {:<12} {:<12}".format("Точка", "E.R", "E.G", "E.B"))

    for P_global in global_points:
        E_total = np.zeros(3)
        for light in lights:
            E, _ = compute_illuminance(light, P_global, normal)
            E_total += E
        print("{:<30} | {:<12.5f} {:<12.5f} {:<12.5f}".format(
            str(np.round(P_global, 3)), E_total[0], E_total[1], E_total[2]))

    print("\n{:=<60}".format(""))
    print("3. Яркость точек (BRDF Блинна–Фонга)")
    print("{:-<60}".format(""))
    print("kd={}, ks={}, m={}, color={}, view={}".format(
        kd, ks, m_shine, surface_color, view_dir))
    print("{:<30} | {:<12} {:<12} {:<12}".format("Точка", "L.R", "L.G", "L.B"))

    for P_global in global_points:
        L = compute_brightness(P_global, normal, lights,
                               view_dir, surface_color, kd, ks, m_shine)
        print("{:<30} | {:<12.5f} {:<12.5f} {:<12.5f}".format(
            str(np.round(P_global, 3)), L[0], L[1], L[2]))


if __name__ == "__main__":
    demo()
