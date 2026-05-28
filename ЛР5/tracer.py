"""
Path tracer (на базе ЛР4) с раздельным выходом прямой/вторичной яркости.

Главное отличие от ЛР4: для пикселя возвращаются:
    direct   - прямая яркость в первой точке пересечения (NEE);
                для видимого источника = его emission.
    indirect - всё, что приходит после первого диффузного отскока
                (включая emission на дальнейших отскоках).
Это нужно для билатеральной фильтрации (ЛР5):
прямую и вторичную компоненты фильтруют ОТДЕЛЬНО.

Алгоритм:
    1. Пускаем луч из камеры. Если ничего не попало -> (0,0).
    2. Если первая поверхность - источник: direct = emission, indirect = 0.
    3. Иначе:
        direct   = kd * NEE(hit)                       (один отскок)
        indirect = kd * trace_path(secondary_ray, ...) (всё остальное)
    4. Для зеркал: direct = 0, indirect = trace по отражённому лучу.
"""

import math
from typing import List, Tuple

import numpy as np

from vec3 import vec3, dot, normalize, reflect
from geometry import Triangle, HitRecord, SceneAccelerator
from sampling import cosine_weighted_hemisphere, sample_triangle


MAX_DEPTH = 8


# ---------- русская рулетка ---------------------------------------------------
def _pick_event(mat, rng):
    d_avg = float(np.mean(mat.diffuse))
    s_avg = float(np.mean(mat.specular))
    total = d_avg + s_avg
    if total < 1e-9:
        return "absorb", vec3(0, 0, 0)
    survive = min(total, 1.0)
    if rng.random() > survive:
        return "absorb", vec3(0, 0, 0)
    p_d = d_avg / total
    if rng.random() < p_d:
        return "diffuse", mat.diffuse / (survive * p_d)
    return "specular", mat.specular / (survive * (1.0 - p_d))


# ---------- NEE (одиночная выборка источника) ---------------------------------
def _direct_light(hit, lights, accel, rng) -> np.ndarray:
    if not lights:
        return vec3(0, 0, 0)
    powers = np.array([t.area * float(np.max(t.material.emission)) for t in lights])
    tot = powers.sum()
    if tot < 1e-12:
        return vec3(0, 0, 0)
    probs = powers / tot
    idx = rng.choice(len(lights), p=probs)
    light = lights[idx]

    lp = sample_triangle(light, rng)
    to_l = lp - hit.point
    d2 = float(np.dot(to_l, to_l))
    d  = math.sqrt(d2)
    ld = to_l / d

    cos_hit = dot(hit.normal, ld)
    if cos_hit <= 0:
        return vec3(0, 0, 0)
    cos_l = -dot(light.normal, ld)
    if cos_l <= 0:
        return vec3(0, 0, 0)
    shadow = accel.intersect(hit.point + hit.normal * 1e-4, ld)
    if shadow is not None and shadow.t < d - 1e-3:
        return vec3(0, 0, 0)
    pdf = probs[idx] / light.area
    return light.material.emission * cos_hit * cos_l / (d2 * math.pi * pdf)


# ---------- полный path tracer (тот же, что в ЛР4, с фильтром emission) -------
def _trace_path(origin, direction, accel, lights, rng) -> np.ndarray:
    colour = vec3(0, 0, 0)
    throughput = vec3(1, 1, 1)
    last_specular = True

    for _ in range(MAX_DEPTH):
        hit = accel.intersect(origin, direction)
        if hit is None:
            break
        if last_specular:
            colour = colour + throughput * hit.material.emission
        if float(np.mean(hit.material.diffuse)) > 1e-9:
            colour = colour + throughput * hit.material.diffuse * _direct_light(hit, lights, accel, rng)

        event, weight = _pick_event(hit.material, rng)
        if event == "absorb":
            break
        if event == "diffuse":
            new_dir = cosine_weighted_hemisphere(hit.normal, rng)
            last_specular = False
        else:
            new_dir = reflect(direction, hit.normal)
            last_specular = True

        throughput = throughput * weight
        origin = hit.point + hit.normal * 1e-4
        direction = normalize(new_dir)
    return colour


# ---------- ОСНОВНАЯ ФУНКЦИЯ: трассировка с G-буфером -------------------------
def trace_with_gbuffer(
    origin: np.ndarray,
    direction: np.ndarray,
    accel: SceneAccelerator,
    lights: List[Triangle],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    """
    Возвращает: (direct, indirect, depth, normal, object_id)
        direct, indirect - RGB-яркости в первой точке пересечения;
        depth            - расстояние до этой точки (inf если миссы);
        normal           - нормаль в точке (нули если миссы);
        object_id        - id объекта (-1 если миссы).
    """
    hit = accel.intersect(origin, direction)
    if hit is None:
        return vec3(0, 0, 0), vec3(0, 0, 0), float("inf"), vec3(0, 0, 0), -1

    depth  = hit.t
    normal = hit.normal
    obj_id = hit.object_id

    # 1) видим сам источник - direct = его emission
    if float(np.mean(hit.material.emission)) > 1e-9:
        return hit.material.emission.copy(), vec3(0, 0, 0), depth, normal, obj_id

    # 2) зеркало - direct = 0; indirect = trace_path по отражённому лучу
    if float(np.mean(hit.material.diffuse)) < 1e-9 and float(np.mean(hit.material.specular)) > 1e-9:
        refl = reflect(direction, hit.normal)
        ind = hit.material.specular * _trace_path(
            hit.point + hit.normal * 1e-4, refl, accel, lights, rng)
        return vec3(0, 0, 0), ind, depth, normal, obj_id

    # 3) диффуз: direct = kd * NEE, indirect = kd * trace_path по cos-выборке
    direct = hit.material.diffuse * _direct_light(hit, lights, accel, rng)
    new_dir = cosine_weighted_hemisphere(hit.normal, rng)
    indirect_color = _trace_path(
        hit.point + hit.normal * 1e-4, new_dir, accel, lights, rng)
    indirect = hit.material.diffuse * indirect_color
    return direct, indirect, depth, normal, obj_id
