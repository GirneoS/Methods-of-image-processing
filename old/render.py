"""Rendering loop: iterates over pixels, accumulates samples, and writes the image."""

import sys
import time

import numpy as np

from vec3 import vec3
from tracer import trace_path
from camera import Camera
from geometry import Triangle, PointLight, SceneAccelerator
from typing import List


def tone_map(hdr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Преобразовать HDR-картинку (float RGB) в 8-битную LDR для сохранения в PPM/PNG.

    Шаги:
    1. Считается воспринимаемая яркость по стандарту (взвешенное Y).
    2. Весь HDR масштабируется так, чтобы средняя яркость кадра стала 0.5 (простая нормировка экспозиции).
    3. Значения обрезаются в [0, 1] (clipping пересветов).
    4. Гамма-коррекция: out = in^(1/γ), чтобы учесть нелинейность дисплея (обычно γ≈2.2).
    5. Умножение на 255 и приведение к uint8.

    Parameters
    ----------
    hdr
        Массив формы (H, W, 3), значения — накопленная яркость трассировщика.
    gamma
        Показатель гаммы дисплея.

    Returns
    -------
    np.ndarray
        Изображение uint8 формы (H, W, 3).
    """
    luminance = 0.2126 * hdr[:, :, 0] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 2]
    mean_lum = np.mean(luminance) + 1e-10
    mapped = hdr * (0.5 / mean_lum)
    mapped = np.clip(mapped, 0.0, 1.0)
    corrected = np.power(mapped, 1.0 / gamma)
    return (corrected * 255).astype(np.uint8)


def save_ppm(path: str, img: np.ndarray):
    """Записать RGB-изображение в бинарный формат P6 PPM (заголовок ASCII, пиксели сырые RGB).

    Ожидается *img* типа uint8, форма (высота, ширина, 3). Удобен как минимальный
    переносимый формат без внешних библиотек.
    """
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())


def render(
    triangles: List[Triangle],
    lights: List[Triangle],
    camera: Camera,
    spp: int = 16,
    output: str = "output.ppm",
    seed: int = 42,
    point_lights: List[PointLight] = None,
):
    """Отрендерить сцену: для каждого пикселя *spp* независимых путей, усреднить, тонмаппинг, файл.

    Строится SceneAccelerator по всем треугольникам. Для антиалиасинга координаты луча
    (px, py) = (i, j) + случайное смещение в [0,1)². Накопленный цвет делится на spp.
    В конце вызываются tone_map и save_ppm; при установленном Pillow дополнительно пишется PNG
    с тем же базовым именем.

    Parameters
    ----------
    triangles
        Полная геометрия сцены (включая треугольники источников).
    lights
        Подсписок треугольников с emission > 0 для прямого освещения в трассировщике.
    camera
        Камера (разрешение берётся из camera.width/height).
    spp
        Число сэмплов (лучей) на пиксель; больше — меньше шум, дольше счёт.
    output
        Путь к выходному PPM.
    seed
        Зерно ГПСЧ для воспроизводимости.
    point_lights
        Список точечных источников PointLight. По умолчанию пустой список.
    """
    if point_lights is None:
        point_lights = []

    accel = SceneAccelerator(triangles)
    w, h = camera.width, camera.height
    hdr = np.zeros((h, w, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)

    total_pixels = w * h
    start = time.time()

    for j in range(h):
        pct = (j * w) / total_pixels * 100
        elapsed = time.time() - start
        eta = (elapsed / max(j * w, 1)) * (total_pixels - j * w)
        sys.stdout.write(f"\rRendering: {pct:5.1f}%  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
        sys.stdout.flush()

        for i in range(w):
            colour = vec3(0, 0, 0)
            for _ in range(spp):
                px = i + rng.random()
                py = j + rng.random()
                direction = camera.get_ray(px, py)
                colour = colour + trace_path(
                    camera.eye, direction, accel, lights, rng, point_lights
                )
            hdr[j, i] = colour / spp

    elapsed = time.time() - start
    sys.stdout.write(f"\rRendering: 100.0%  elapsed {elapsed:.0f}s           \n")

    ldr = tone_map(hdr)
    save_ppm(output, ldr)
    print(f"Saved {output}  ({w}×{h}, {spp} spp)")

    try:
        from PIL import Image
        png_path = output.rsplit(".", 1)[0] + ".png"
        Image.fromarray(ldr).save(png_path)
        print(f"Saved {png_path}")
    except ImportError:
        pass
