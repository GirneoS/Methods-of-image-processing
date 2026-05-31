#!/usr/bin/env python3
"""
ЛР3, точка входа.

Запуск:
    python main.py 1            # только задание 1
    python main.py all          # все 4 задания
    python main.py all --plots  # сохранить графики в plots/

Доказательство равномерности - НАГЛЯДНЫЙ метод:
    Вписываем в большую фигуру несколько ОДИНАКОВЫХ маленьких фигурок,
    считаем точки в каждой. Числа должны быть близкими -> распределение
    равномерное.
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
import proof_shape


# ──────────────────────────────────────────────────────────────────────────────
# Запуск выборки + наглядное доказательство (квадраты/колпаки/конусы)
# ──────────────────────────────────────────────────────────────────────────────

def shape_proof_triangle(rng):
    """Равносторонний треугольник в локальных 2D координатах."""
    V1 = np.array([0.0, 0.0])
    V2 = np.array([1.0, 0.0])
    V3 = np.array([0.5, np.sqrt(3)/2])
    # выборка по барицентрическому методу
    u1 = rng.random(100_000); u2 = rng.random(100_000)
    sr = np.sqrt(u1)
    w1 = 1.0 - sr; w2 = sr*(1.0 - u2); w3 = sr*u2
    p = w1[:, None]*V1 + w2[:, None]*V2 + w3[:, None]*V3
    squares, counts = proof_shape.proof_triangle(p)

    print("\n=== ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ (треугольник) ===")
    print(f"В треугольник вписано 3 ОДИНАКОВЫХ квадрата (сторона = {squares[0][2]}):")
    for i, n in enumerate(counts, 1):
        print(f"  Квадрат {i}: {n} точек")
    spread = (max(counts) - min(counts)) / np.mean(counts) * 100
    print(f"  Разброс: {spread:.1f}% от среднего  ->  числа близки -> равномерно")
    return V1, V2, V3, p, squares, counts


def shape_proof_circle(rng):
    R = 1.0
    u1 = rng.random(100_000); u2 = rng.random(100_000)
    r = R * np.sqrt(u1); phi = 2*np.pi*u2
    p = np.stack([r*np.cos(phi), r*np.sin(phi)], axis=1)
    squares, counts = proof_shape.proof_circle(p)

    print("\n=== ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ (круг) ===")
    print(f"В круг вписано 4 ОДИНАКОВЫХ квадрата (сторона = {squares[0][2]}):")
    for i, n in enumerate(counts, 1):
        print(f"  Квадрат {i}: {n} точек")
    spread = (max(counts) - min(counts)) / np.mean(counts) * 100
    print(f"  Разброс: {spread:.1f}% от среднего  ->  числа близки -> равномерно")
    return p, squares, counts


def shape_proof_sphere(rng):
    u1 = rng.random(100_000); u2 = rng.random(100_000)
    cos_t = 2*u1 - 1; sin_t = np.sqrt(1 - cos_t**2); phi = 2*np.pi*u2
    dirs = np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], axis=1)
    centers, counts, cap_angle = proof_shape.proof_sphere(dirs)

    print("\n=== ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ (сфера) ===")
    print(f"На сфере выделено 3 ОДИНАКОВЫХ колпака (угловой радиус = {np.rad2deg(cap_angle):.0f} град.):")
    for i, n in enumerate(counts, 1):
        print(f"  Колпак {i}: {n} точек")
    spread = (max(counts) - min(counts)) / np.mean(counts) * 100
    print(f"  Разброс: {spread:.1f}% от среднего  ->  числа близки -> равномерно")
    return dirs, centers, counts, cap_angle


def shape_proof_cosine(rng):
    axis_n = np.array([0.0, 0.0, 1.0])
    u1 = rng.random(100_000); u2 = rng.random(100_000)
    cos_t = np.sqrt(u1); sin_t = np.sqrt(1 - cos_t**2); phi = 2*np.pi*u2
    dirs = np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], axis=1)
    edges_deg, obs, theor = proof_shape.proof_cosine(dirs, axis_n)

    print("\n=== ДОКАЗАТЕЛЬСТВО КОСИНУСНОГО распределения ===")
    print("Полусфера разбита на пояса по углу с N. Для КОСИНУСНОГО распределения")
    print("число точек должно ПАДАТЬ с ростом угла (~ sin² hi - sin² lo).")
    print()
    print(f"  {'Пояс (град.)':<15} {'Точек (факт)':>14} {'Теор. (cos-pdf)':>17} {'ошибка':>8}")
    for i in range(len(edges_deg) - 1):
        err = abs(obs[i] - theor[i]) / max(theor[i], 1) * 100
        print(f"  {edges_deg[i]:>4}-{edges_deg[i+1]:<3}        {obs[i]:>10}      {theor[i]:>13}    {err:>5.1f}%")
    return dirs, axis_n, edges_deg, obs, theor


# ──────────────────────────────────────────────────────────────────────────────
# Графики (в стиле образца проверяющего)
# ──────────────────────────────────────────────────────────────────────────────

def make_shape_plots(tri_data, cir_data, sph_data, cos_data, outdir="plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except ImportError:
        print("[matplotlib не установлен]")
        return
    os.makedirs(outdir, exist_ok=True)

    # ── треугольник ──
    V1, V2, V3, p, squares, counts = tri_data
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.4)
    triangle = plt.Polygon([V1, V2, V3], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    colors = ["red", "green", "blue"]
    for (x0, y0, side), col, n in zip(squares, colors, counts):
        ax.add_patch(Rectangle((x0, y0), side, side, fill=False,
                               edgecolor=col, linewidth=2.5))
        ax.text(x0 + side/2, y0 - 0.05, f"{n}", color=col,
                ha="center", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title("Треугольник: равномерное распределение\n3 одинаковых квадрата -> числа близки")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.15, 1.0)
    plt.tight_layout(); plt.savefig(f"{outdir}/proof_triangle.png", dpi=120); plt.close()

    # ── круг ──
    p, squares, counts = cir_data
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.4)
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k-", linewidth=2)
    colors = ["red", "green", "blue", "orange"]
    for (x0, y0, side), col, n in zip(squares, colors, counts):
        ax.add_patch(Rectangle((x0, y0), side, side, fill=False,
                               edgecolor=col, linewidth=2.5))
        ax.text(x0 + side/2, y0 + side + 0.05, f"{n}", color=col,
                ha="center", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title("Круг: равномерное распределение\n4 одинаковых квадрата -> числа близки")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    plt.tight_layout(); plt.savefig(f"{outdir}/proof_circle.png", dpi=120); plt.close()

    # ── сфера ──
    dirs, centers, counts, cap_angle = sph_data
    cos_min = np.cos(cap_angle)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(dirs[::5, 0], dirs[::5, 1], dirs[::5, 2], s=1, alpha=0.15, c="gray", label="Все точки")
    colors = ["red", "green", "blue"]
    names = ["Колпак 1", "Колпак 2", "Колпак 3"]
    for c, col, name, n in zip(centers, colors, names, counts):
        cosines = dirs @ c
        in_cap = dirs[cosines >= cos_min]
        ax.scatter(in_cap[:, 0], in_cap[:, 1], in_cap[:, 2], s=3, c=col, label=f"{name}: {n}")
        ax.plot([0, c[0]], [0, c[1]], [0, c[2]], color=col, linewidth=2)
    ax.set_title(f"Сфера: равномерное распределение\n3 одинаковых колпака (радиус {np.rad2deg(cap_angle):.0f}°)")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/proof_sphere.png", dpi=120); plt.close()

    # ── косинус ──
    dirs, axis_n, edges_deg, obs, theor = cos_data
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(dirs[::5, 0], dirs[::5, 1], dirs[::5, 2], s=2, alpha=0.4)
    ax.set_title("Косинусное распределение (вдоль Z)")
    fig.delaxes(axes[0])

    ax2 = axes[1]
    x = np.arange(len(obs))
    width = 0.4
    ax2.bar(x - width/2, obs,   width, color="steelblue", label="Факт (наша выборка)")
    ax2.bar(x + width/2, theor, width, color="orange",   label="Теор. cos-pdf")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{edges_deg[i]}-{edges_deg[i+1]}°" for i in range(len(obs))])
    ax2.set_xlabel("угол с N")
    ax2.set_ylabel("число точек")
    ax2.set_title("Число точек по поясам угла с N\nдля косинусного распределения должно ПАДАТЬ ~ cos θ")
    ax2.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/proof_cosine.png", dpi=120); plt.close()

    print(f"\n[наглядные доказательства сохранены: {outdir}/proof_*.png]")


# ──────────────────────────────────────────────────────────────────────────────
# Старые графики и точные стат-тесты (KS, χ²) - оставлены для отчёта
# ──────────────────────────────────────────────────────────────────────────────

def save_plots(r1, r2, r3, r4, outdir="plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(outdir, exist_ok=True)

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
    axes[1].axhline(1/R**2, color="k", linestyle="--", label=f"теор. 1/R² = {1/R**2:.3f}")
    axes[1].set_title("pdf(r²) — const = 1/R²")
    axes[1].set_xlabel("r²"); axes[1].legend()
    axes[2].hist(phi, bins=40, density=True)
    axes[2].axhline(1/(2*np.pi), color="k", linestyle="--", label="теор. 1/(2π)")
    axes[2].set_title("pdf(φ) — const = 1/(2π)")
    axes[2].set_xlabel("φ"); axes[2].legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task2.png", dpi=120); plt.close()

    cos_t = r3["cos_t"]; phi3 = r3["phi"]; d = r3["dirs"]
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(d[::100, 0], d[::100, 1], d[::100, 2], s=1, alpha=0.4)
    ax1.set_title("Направления на сфере")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(cos_t, bins=40, density=True)
    ax2.axhline(0.5, color="k", linestyle="--", label="теор. 1/2")
    ax2.set_title("pdf(cos θ) — const = 1/2")
    ax2.set_xlabel("cos θ"); ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(phi3, bins=40, density=True)
    ax3.axhline(1/(2*np.pi), color="k", linestyle="--", label="теор. 1/(2π)")
    ax3.set_title("pdf(φ) — const = 1/(2π)")
    ax3.set_xlabel("φ"); ax3.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task3.png", dpi=120); plt.close()

    cos_t4 = r4["cos_t"]; phi4 = r4["phi"]; d4 = r4["dirs"]
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(d4[::100, 0], d4[::100, 1], d4[::100, 2], s=1, alpha=0.4)
    ax1.set_title("Косинусное распределение направлений")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(cos_t4, bins=40, density=True, alpha=0.6, label="эмпир.")
    xs = np.linspace(0, 1, 200)
    ax2.plot(xs, 2*xs, "k--", label="теор. pdf = 2 cos θ")
    ax2.set_title("pdf(cos θ): эмпир. vs теор.")
    ax2.set_xlabel("cos θ"); ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(phi4, bins=40, density=True)
    ax3.axhline(1/(2*np.pi), color="k", linestyle="--", label="теор. 1/(2π)")
    ax3.set_title("pdf(φ) — const = 1/(2π)")
    ax3.set_xlabel("φ"); ax3.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/task4.png", dpi=120); plt.close()


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["1","2","3","4","all"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    def rng_for(k): return np.random.default_rng(args.seed + k)

    if args.task == "all":
        # точные стат-тесты (KS + χ²) для отчёта
        r1 = task1_triangle.run(rng_for(1)); print()
        r2 = task2_circle.run  (rng_for(2)); print()
        r3 = task3_sphere.run  (rng_for(3)); print()
        r4 = task4_cosine.run  (rng_for(4))

        # наглядные доказательства (одинаковые фигурки внутри)
        tri = shape_proof_triangle(rng_for(11))
        cir = shape_proof_circle  (rng_for(12))
        sph = shape_proof_sphere  (rng_for(13))
        cos = shape_proof_cosine  (rng_for(14))

        if args.plots:
            save_plots(r1, r2, r3, r4)
            make_shape_plots(tri, cir, sph, cos)
        return

    {"1": task1_triangle.run, "2": task2_circle.run,
     "3": task3_sphere.run,  "4": task4_cosine.run}[args.task](rng_for(int(args.task)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
