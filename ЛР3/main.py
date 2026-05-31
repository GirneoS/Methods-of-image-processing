#!/usr/bin/env python3
"""
ЛР3, точка входа.

Запуск:
    python main.py 1            # только задание 1
    python main.py all          # все 4 задания + графики
    python main.py all --plots  # сохранить графики в plots/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import task1_triangle
import task2_circle
import task3_sphere
import task4_cosine


def save_plots(r1, r2, r3, r4, outdir="plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[matplotlib не установлен, графики не сохранены]")
        return
    os.makedirs(outdir, exist_ok=True)

    # ── ЗАДАНИЕ 1: треугольник + гистограммы барицентрических координат ──
    p = r1["points"]; w = r1["w"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(p[::50, 0], p[::50, 1], s=1, alpha=0.4)
    axes[0].set_title("Точки в треугольнике (проекция XY)")
    axes[0].set_aspect("equal")
    axes[1].hist(w[:, 0], bins=40, density=True, alpha=0.6, label="w1")
    axes[1].hist(w[:, 1], bins=40, density=True, alpha=0.6, label="w2")
    axes[1].hist(w[:, 2], bins=40, density=True, alpha=0.6, label="w3")
    xs = np.linspace(0, 1, 200)
    axes[1].plot(xs, 2*(1-xs), "k--", label="pdf теор = 2(1−w)")
    axes[1].set_title("Маргинальные распределения w_i")
    axes[1].legend(); axes[1].set_xlabel("w"); axes[1].set_ylabel("pdf")
    plt.tight_layout(); plt.savefig(f"{outdir}/task1.png", dpi=120); plt.close()

    # ── ЗАДАНИЕ 2: круг + гистограммы r и phi ──
    r = r2["r"]; phi = r2["phi"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    R = 1.5
    th = np.linspace(0, 2*np.pi, 200)
    axes[0].plot(R*np.cos(th), R*np.sin(th), "k--")
    axes[0].scatter(r[::50]*np.cos(phi[::50]), r[::50]*np.sin(phi[::50]),
                    s=1, alpha=0.4)
    axes[0].set_title("Точки в круге (локальная плоскость)")
    axes[0].set_aspect("equal")
    axes[1].hist(r**2, bins=40, density=True)
    axes[1].axhline(1/R**2, color="k", linestyle="--",
                    label=f"теор. 1/R² = {1/R**2:.3f}")
    axes[1].set_title("pdf(r²) — должна быть const = 1/R²")
    axes[1].set_xlabel("r²"); axes[1].legend()
    axes[2].hist(phi, bins=40, density=True)
    axes[2].axhline(1/(2*np.pi), color="k", linestyle="--",
                    label="теор. 1/(2π)")
    axes[2].set_title("pdf(φ) — должна быть const = 1/(2π)")
    axes[2].set_xlabel("φ"); axes[2].legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task2.png", dpi=120); plt.close()

    # ── ЗАДАНИЕ 3: сфера + гистограммы cos θ и phi ──
    cos_t = r3["cos_t"]; phi3 = r3["phi"]; d = r3["dirs"]
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(d[::100, 0], d[::100, 1], d[::100, 2], s=1, alpha=0.4)
    ax1.set_title("Направления на сфере")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(cos_t, bins=40, density=True)
    ax2.axhline(0.5, color="k", linestyle="--", label="теор. 1/2")
    ax2.set_title("pdf(cos θ) — должна быть const = 1/2")
    ax2.set_xlabel("cos θ"); ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(phi3, bins=40, density=True)
    ax3.axhline(1/(2*np.pi), color="k", linestyle="--", label="теор. 1/(2π)")
    ax3.set_title("pdf(φ) — должна быть const = 1/(2π)")
    ax3.set_xlabel("φ"); ax3.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task3.png", dpi=120); plt.close()

    # ── ЗАДАНИЕ 4: косинусное + гистограмма cos θ vs 2 cos θ ──
    cos_t4 = r4["cos_t"]; phi4 = r4["phi"]; d4 = r4["dirs"]
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(d4[::100, 0], d4[::100, 1], d4[::100, 2], s=1, alpha=0.4)
    ax1.set_title("Косинусное распределение направлений")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(cos_t4, bins=40, density=True, alpha=0.6, label="эмпир.")
    xs = np.linspace(0, 1, 200)
    ax2.plot(xs, 2*xs, "k--", label="теор. pdf = 2 cos θ")
    ax2.set_title("pdf(cos θ): эмпир. vs теор. (косинусное)")
    ax2.set_xlabel("cos θ"); ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(phi4, bins=40, density=True)
    ax3.axhline(1/(2*np.pi), color="k", linestyle="--", label="теор. 1/(2π)")
    ax3.set_title("pdf(φ) — должна быть const = 1/(2π)")
    ax3.set_xlabel("φ"); ax3.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task4.png", dpi=120); plt.close()

    print(f"\n[графики сохранены в {outdir}/]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["1","2","3","4","all"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    # независимые rng для каждой задачи, чтобы результаты не зависели
    # от порядка вызова и порядкового состояния общего генератора
    def rng_for(k): return np.random.default_rng(args.seed + k)

    if args.task == "all":
        r1 = task1_triangle.run(rng_for(1)); print()
        r2 = task2_circle.run  (rng_for(2)); print()
        r3 = task3_sphere.run  (rng_for(3)); print()
        r4 = task4_cosine.run  (rng_for(4))
        if args.plots:
            save_plots(r1, r2, r3, r4)
        return

    {"1": task1_triangle.run, "2": task2_circle.run,
     "3": task3_sphere.run,  "4": task4_cosine.run}[args.task](rng_for(int(args.task)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
