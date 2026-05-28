"""Core path-tracing integrator with next-event estimation (direct light sampling)."""

import math
from typing import List

import numpy as np

from vec3 import vec3, dot, normalize, reflect
from geometry import Triangle, HitRecord, SceneAccelerator
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
    lights: List[Triangle],
    accel: SceneAccelerator,
    rng: np.random.Generator,
) -> np.ndarray:
    """Оценить геометрический вклад прямого освещения (без BRDF поверхности).

    Источник выбирается с вероятностью ∝ площадь × максимальная компонента яркости (более яркий
    источник получает больший вес). На нём равномерно выбирается точка. Возвращается:
    Le · cos(θ_hit) · cos(θ_light) / (r² · π · pdf), где pdf — плотность выбора точки
    с учётом вероятности выбора треугольника. Множитель diffuse поверхности прибавляется
    снаружи (в trace_path) как произведение цвета луча на цвет поверхности.

    Returns
    -------
    np.ndarray
        RGB-оценка геометрического члена от прямого света (без kd поверхности).
    """
    if not lights:
        return vec3(0, 0, 0)

    # Fix 1: взвешиваем по максимальной компоненте emission (более яркий → больший вес)
    powers = np.array([tri.area * float(np.max(tri.material.emission)) for tri in lights])
    total_power = powers.sum()
    if total_power < 1e-12:
        return vec3(0, 0, 0)

    probs = powers / total_power
    idx = rng.choice(len(lights), p=probs)
    light = lights[idx]

    light_point = sample_triangle(light, rng)
    to_light = light_point - hit.point
    dist2 = float(np.dot(to_light, to_light))
    dist = math.sqrt(dist2)
    light_dir = to_light / dist

    cos_hit = dot(hit.normal, light_dir)
    if cos_hit <= 0:
        return vec3(0, 0, 0)

    light_normal = light.normal
    cos_light = -dot(light_normal, light_dir)
    if cos_light <= 0:
        return vec3(0, 0, 0)

    shadow_hit = accel.intersect(hit.point + hit.normal * 1e-4, light_dir)
    if shadow_hit is not None and shadow_hit.t < dist - 1e-4:
        return vec3(0, 0, 0)

    area_light = light.area
    # Fix 2: делим на плотность вероятности, чтобы учитывать все источники света
    pdf = probs[idx] / area_light
    # Fix 3: kd поверхности здесь НЕ включаем — его добавляет вызывающая сторона явно
    radiance = light.material.emission * cos_hit * cos_light / (dist2 * math.pi * pdf)
    return radiance


def trace_path(
    origin: np.ndarray,
    direction: np.ndarray,
    accel: SceneAccelerator,
    lights: List[Triangle],
    rng: np.random.Generator,
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
        Список треугольников-источников (подмножество сцены) для NEE.
    rng
        Генератор случайных чисел для воспроизводимости при заданном seed.

    Returns
    -------
    np.ndarray
        Суммарная наблюдаемая яркость вдоль пути (HDR, без тонмаппинга).
    """
    colour = vec3(0, 0, 0)
    # Fix 4: явно ведём «цвет луча» — произведение цветов всех поверхностей вдоль пути
    ray_color = vec3(1, 1, 1)

    for depth in range(MAX_DEPTH):
        hit = accel.intersect(origin, direction)
        if hit is None:
            break

        if depth == 0:
            colour = colour + ray_color * hit.material.emission

        if float(np.mean(hit.material.diffuse)) > 1e-9:
            # Fix 3: явно умножаем цвет луча на цвет поверхности (kd),
            # а _direct_light возвращает только геометрический член без BRDF
            colour = colour + ray_color * hit.material.diffuse * _direct_light(hit, lights, accel, rng)

        event, weight = _pick_event(hit.material, rng)
        if event == "absorb":
            break

        if event == "diffuse":
            new_dir = cosine_weighted_hemisphere(hit.normal, rng)
        else:
            new_dir = reflect(direction, hit.normal)

        # Fix 4: обновляем цвет луча — накапливаем произведение цветов поверхностей
        ray_color = ray_color * weight
        origin = hit.point + hit.normal * 1e-4
        direction = normalize(new_dir)

    return colour
