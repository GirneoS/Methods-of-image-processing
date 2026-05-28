"""
Рендер Cornell Box (как в ЛР4) с полным G-буфером для ЛР5.

Возвращает для каждого пикселя:
    color    - direct + indirect (зашумлённое изображение, HDR)
    direct   - прямая яркость   (NEE в первой точке пересечения)
    indirect - вторичная яркость (всё, что после первого отскока)
    depth    - расстояние до первой поверхности
    normal   - нормаль в первой точке
    obj_id   - индекс объекта сцены (или -1)
"""

import sys
import time
import numpy as np

from vec3 import vec3
from camera import Camera
from scene import build_cornell_box
from geometry import SceneAccelerator
from tracer import trace_with_gbuffer


def render_gbuffer(width: int = 256, height: int = 256, spp: int = 4, seed: int = 42):
    triangles, lights, _ = build_cornell_box()
    # Камера ЛР4, но с нужным разрешением
    cam = Camera(
        eye=vec3(278, 273, -800),
        target=vec3(278, 273, 0),
        fov_deg=39.3,
        width=width,
        height=height,
    )
    accel = SceneAccelerator(triangles)
    rng = np.random.default_rng(seed)

    direct   = np.zeros((height, width, 3), dtype=np.float64)
    indirect = np.zeros((height, width, 3), dtype=np.float64)
    depth    = np.full((height, width),  np.inf, dtype=np.float64)
    normal   = np.zeros((height, width, 3), dtype=np.float64)
    obj_id   = np.full((height, width), -1, dtype=np.int32)

    start = time.time()
    for j in range(height):
        elapsed = time.time() - start
        eta = elapsed / max(j * width, 1) * (width * height - j * width)
        sys.stdout.write(f"\rGBuffer: {j*100/height:5.1f}%  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
        sys.stdout.flush()

        for i in range(width):
            # G-buffer от центра пикселя - детерминированно (без шума)
            d0 = cam.get_ray(i + 0.5, (height - 1 - j) + 0.5)
            _, _, dpt, nrm, oid = trace_with_gbuffer(cam.eye, d0, accel, lights, rng)
            depth [j, i] = dpt
            normal[j, i] = nrm
            obj_id[j, i] = oid

            # яркость - среднее spp выборок (шумно при малом spp)
            ds = np.zeros(3)
            ins = np.zeros(3)
            for _ in range(spp):
                px = i + rng.random()
                py = (height - 1 - j) + rng.random()
                d = cam.get_ray(px, py)
                dr, ind, _, _, _ = trace_with_gbuffer(cam.eye, d, accel, lights, rng)
                ds  += dr
                ins += ind
            direct  [j, i] = ds  / spp
            indirect[j, i] = ins / spp

    elapsed = time.time() - start
    sys.stdout.write(f"\rGBuffer: 100.0%  elapsed {elapsed:.0f}s             \n")

    finite = depth[np.isfinite(depth)]
    if len(finite):
        depth[~np.isfinite(depth)] = finite.max()

    color = direct + indirect
    return (color.astype(np.float32),
            direct.astype(np.float32),
            indirect.astype(np.float32),
            depth.astype(np.float32),
            normal.astype(np.float32),
            obj_id)
