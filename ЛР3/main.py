#!/usr/bin/env python3

# ТОЧКА ВХОДА, ЗАПУСКАТЬ ЭТОТ ФАЙЛ

from __future__ import annotations

import argparse
import sys

import numpy as np

import task1_triangle
import task2_circle
import task3_sphere
import task4_cosine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.task == 1:
        task1_triangle.run(rng)
    elif args.task == 2:
        task2_circle.run(rng)
    elif args.task == 3:
        task3_sphere.run(rng)
    elif args.task == 4:
        task4_cosine.run(rng)
    else:
        print("Invalid task")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
