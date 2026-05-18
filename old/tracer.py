"""Core path-tracing integrator with next-event estimation (direct light sampling)."""

import math
from typing import List

import numpy as np

from vec3 import vec3, dot, normalize, reflect
from geometry import Triangle, PointLight, HitRecord, SceneAccelerator
from sampling import cosine_weighted_hemisphere, sample_triangle


MAX_DEPTH = 10


def _pick_event(mat, rng: np.random.Generator):
    """Выбрать исход отскока луча: диффузное отражение, зеркальное или поглощение (русская рулетка).

    Средние по каналам диффузный и зеркальный коэффициенты задают суммарную «альбедо»-подобную
    величину; с вероятностью 1 − min(сумма, 1) луч обрывается (поглощение). Иначе с вероятностью,
    пропорциональной доле диффуза/зеркала, выбирается тип отражения. Вес *weight* подбирается
    так, чтобы математическое ожидание вклада совпадало с физической BRDF (несмещённая оценка).

    Returns
    -------
    event : str
        'diffuse' | 'specular' | 'absorb'.
    weight : np.ndarray
        RGB-множитель пропускания (throughput) для выбранного события.
    """
    d_avg = float(np.mean(mat.diffuse))
    s_avg = float(np.mean(mat.specular))
    total = d_avg + s_avg

    if total < 1e-9:
        return "absorb", vec3(0, 0, 0)

    survive = min(total, 1.0)
    if rng.random() > survive:
        return "absorb", vec3(0, 0, 0)

    p_diffuse = d_avg / total
    if rng.random() < p_diffuse:
        weight = mat.diffuse / (survive * p_diffuse)
        return "diffuse", weight
    else:
        weight = mat.specular / (survive * (1.0 - p_diffuse))
        return "specular", weight


def _direct_light(
    hit: HitRecord,
    area_lights: List[Triangle],
    point_lights: List[PointLight],
    accel: SceneAccelerator,
    rng: np.random.Generator,
) -> np.ndarray:
    """Оценить прямое освещение в точке попадания (next-event estimation, один луч на источник).

    Объединяет площадные и точечные источники в одну схему важностной выборки.

    Площадные источники (Triangle + emission)
    ------------------------------------------
    Выбирается случайная точка на поверхности треугольника. Вклад:
        Le · (diffuse/π) · cos(θ_hit) · cos(θ_light) / (dist² · pdf)
    где pdf = prob_area / area. Член cos(θ_light) отражает то, что излучение
    ламбертовой поверхности убывает с углом (нормаль источника учитывается).

    Точечные источники (PointLight)
    --------------------------------
    Позиция фиксирована — выборка по площади не нужна. Излучение изотропно:
    нормаль источника отсутствует, поэтому cos(θ_light) = 1 (нет убывания
    по углу). Вклад:
        I · (diffuse/π) · cos(θ_hit) / (dist² · prob_point)
    где I — интенсивность (Вт/ср), prob_point — вероятность выбора этого
    источника из общего пула.

    Выбор источника
    ---------------
    Все источники конкурируют с вероятностями ∝ мощности:
    - площадной: area × mean(emission)
    - точечный:  4π × mean(intensity)  — полная изотропная мощность (Вт)

    Returns
    -------
    np.ndarray
        Оценка RGB-яркости от прямого света (одна выборка Монте-Карло).
    """
    if not area_lights and not point_lights:
        return vec3(0, 0, 0)

    area_powers = (
        np.array([tri.area * float(np.mean(tri.material.emission)) for tri in area_lights])
        if area_lights else np.empty(0)
    )
    point_powers = (
        np.array([4.0 * math.pi * float(np.mean(pl.intensity)) for pl in point_lights])
        if point_lights else np.empty(0)
    )

    all_powers = np.concatenate([area_powers, point_powers])
    total_power = all_powers.sum()
    if total_power < 1e-12:
        return vec3(0, 0, 0)

    probs = all_powers / total_power
    idx = int(rng.choice(len(all_powers), p=probs))
    n_area = len(area_lights)

    if idx < n_area:
        # --- площадной источник ---
        light = area_lights[idx]
        light_point = sample_triangle(light, rng)
        to_light = light_point - hit.point
        dist2 = float(np.dot(to_light, to_light))
        dist = math.sqrt(dist2)
        light_dir = to_light / dist

        cos_hit = dot(hit.normal, light_dir)
        if cos_hit <= 0:
            return vec3(0, 0, 0)

        cos_light = -dot(light.normal, light_dir)
        if cos_light <= 0:
            return vec3(0, 0, 0)

        shadow_hit = accel.intersect(hit.point + hit.normal * 1e-4, light_dir)
        if shadow_hit is not None and shadow_hit.t < dist - 1e-4:
            return vec3(0, 0, 0)

        pdf = probs[idx] / light.area  # p(выбрать треугольник) / площадь
        brdf = hit.material.diffuse / math.pi
        return light.material.emission * brdf * cos_hit * cos_light / (dist2 * pdf)

    else:
        # --- точечный источник ---
        pl = point_lights[idx - n_area]
        to_light = pl.position - hit.point
        dist2 = float(np.dot(to_light, to_light))
        dist = math.sqrt(dist2)
        light_dir = to_light / dist

        cos_hit = dot(hit.normal, light_dir)
        if cos_hit <= 0:
            return vec3(0, 0, 0)

        shadow_hit = accel.intersect(hit.point + hit.normal * 1e-4, light_dir)
        if shadow_hit is not None and shadow_hit.t < dist - 1e-4:
            return vec3(0, 0, 0)

        # Нет cos_light: точечный источник изотропен.
        # Нет деления на площадь: нет площадного pdf — только prob выбора источника.
        brdf = hit.material.diffuse / math.pi
        return pl.intensity * brdf * cos_hit / (dist2 * probs[idx])


def trace_path(
    origin: np.ndarray,
    direction: np.ndarray,
    accel: SceneAccelerator,
    lights: List[Triangle],
    rng: np.random.Generator,
    point_lights: List[PointLight] = None,
) -> np.ndarray:
    """Проследить один путь камеры: накопить RGB-яркость вдоль цепочки отскоков.

    На каждом пересечении: если это первая точка и материал светится — добавляется emission
    (видимые источники с камеры). Для диффузной поверхности добавляется оценка прямого света.
    Затем русская рулетка выбирает новое направление (косинусная полусфера или зеркало),
    throughput умножается на вес события, луч смещается чуть вдоль нормали против самопересечения.
    Цикл ограничен MAX_DEPTH отскоков.

    Parameters
    ----------
    origin, direction
        Начало и нормированное направление луча (обычно камера и get_ray).
    accel
        Ускоритель сцены для пересечений.
    lights
        Список треугольников площадных источников (подмножество сцены) для NEE.
    rng
        Генератор случайных чисел для воспроизводимости при заданном seed.
    point_lights
        Список точечных источников PointLight для NEE. Может быть None или [].

    Returns
    -------
    np.ndarray
        Суммарная наблюдаемая яркость вдоль пути (HDR, без тонмаппинга).
    """
    if point_lights is None:
        point_lights = []

    colour = vec3(0, 0, 0)
    throughput = vec3(1, 1, 1)

    for depth in range(MAX_DEPTH):
        hit = accel.intersect(origin, direction)
        if hit is None:
            break

        if depth == 0:
            colour = colour + throughput * hit.material.emission

        if float(np.mean(hit.material.diffuse)) > 1e-9:
            colour = colour + throughput * _direct_light(hit, lights, point_lights, accel, rng)

        event, weight = _pick_event(hit.material, rng)
        if event == "absorb":
            break

        if event == "diffuse":
            new_dir = cosine_weighted_hemisphere(hit.normal, rng)
        else:
            new_dir = reflect(direction, hit.normal)

        throughput = throughput * weight
        origin = hit.point + hit.normal * 1e-4
        direction = normalize(new_dir)

    return colour
