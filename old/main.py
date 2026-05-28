#!/usr/bin/env python3
"""Path-tracer entry point with a command-line interface.

Usage examples
--------------
  python main.py                              # defaults: 512×512, 16 spp
  python main.py --width 256 --height 256 --spp 4 --output test.ppm
  python main.py --spp 64 --width 512 --height 512
"""

import argparse

from vec3 import vec3
from scene import build_cornell_box
from camera import Camera
from geometry import PointLight
from render import render


def parse_args():
    """Разобрать аргументы командной строки: разрешение, spp, файл, FOV, seed, режим источника.

    Все параметры имеют значения по умолчанию, совместимые с типичным рендером Cornell Box.
    Флаг --point-light добавляет один демо-точечный источник в центр комнаты вместо
    (или вместе с) площадного потолочного источника.
    """
    p = argparse.ArgumentParser(description="Monte-Carlo path tracer (Cornell Box)")
    p.add_argument("--width",       type=int,   default=512,        help="Image width  (default 512)")
    p.add_argument("--height",      type=int,   default=512,        help="Image height (default 512)")
    p.add_argument("--spp",         type=int,   default=16,         help="Samples per pixel (default 16)")
    p.add_argument("--output",      type=str,   default="output.ppm", help="Output file (PPM)")
    p.add_argument("--fov",         type=float, default=39.3,       help="Vertical FOV in degrees")
    p.add_argument("--seed",        type=int,   default=42,         help="RNG seed for reproducibility")
    p.add_argument("--point-light", action="store_true",
                   help="Добавить точечный источник в центр комнаты (демо)")
    return p.parse_args()


def main():
    """Точка входа: загрузить Cornell Box, пересоздать камеру под CLI, запустить render().

    Камера из сцены берёт только положение *eye*; цель взгляда и FOV задаются заново,
    чтобы можно было менять угол обзора и размер изображения без правки scene.py.
    """
    args = parse_args()

    triangles, lights, cam = build_cornell_box()

    cam = Camera(
        eye=cam.eye,
        target=vec3(278, 273, 0),
        fov_deg=args.fov,
        width=args.width,
        height=args.height,
    )

    # Демо-точечный источник: висит в центре комнаты под потолком.
    # Интенсивность ~15 000 Вт/ср даёт сопоставимую освещённость с площадным потолочным источником.
    point_lights = []
    if args.point_light:
        point_lights = [PointLight(
            position=vec3(278, 500, 280),
            intensity=vec3(15000, 10500, 3500),
        )]

    n_pt = len(point_lights)
    print(f"Scene: {len(triangles)} triangles, {len(lights)} area light triangles, {n_pt} point lights")
    print(f"Image: {args.width}×{args.height}, {args.spp} spp, seed={args.seed}")
    print()

    render(triangles, lights, cam, spp=args.spp, output=args.output,
           seed=args.seed, point_lights=point_lights)


if __name__ == "__main__":
    main()
