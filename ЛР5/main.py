"""
Лабораторная работа №5 — Фильтрация синтезированных изображений
Метод: Билатеральная фильтрация с использованием G-буфера

Запуск:
    python main.py [--spp N] [--radius R] [--sigma-s S] [--sigma-d D] [--sigma-n N]

Выходные файлы (в текущей папке):
    noisy.png       — зашумлённое изображение (малое spp)
    filtered.png    — отфильтрованное изображение
    depth.png       — карта глубины
    normal.png      — карта нормалей
    objid.png       — карта объектов
    comparison.png  — сравнение: noisy | filtered
"""

import argparse
import numpy as np
import os

# ── try to import matplotlib ──────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from gbuffer  import render_gbuffer
from bilateral import bilateral_filter_fast
from verify   import energy_conservation, noise_reduction, print_report


def tonemap(img: np.ndarray) -> np.ndarray:
    """Reinhard tonemap + gamma 2.2 → uint8."""
    img = np.clip(img, 0, None)
    img = img / (1 + img)                       # Reinhard
    img = np.clip(img ** (1 / 2.2), 0, 1)       # gamma
    return (img * 255).astype(np.uint8)


def save_png(path: str, img_uint8: np.ndarray) -> None:
    if HAS_PLT:
        plt.imsave(path, img_uint8)
        print(f"  saved {path}")
    else:
        print(f"  [matplotlib not found] skipping {path}")


def save_comparison(path: str, left: np.ndarray, right: np.ndarray,
                    label_l="Noisy", label_r="Filtered") -> None:
    if not HAS_PLT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(left);  axes[0].set_title(label_l); axes[0].axis("off")
    axes[1].imshow(right); axes[1].set_title(label_r); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def visualise_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return (d * 255).astype(np.uint8)


def visualise_normal(normal: np.ndarray) -> np.ndarray:
    n = (normal + 1) / 2
    return (np.clip(n, 0, 1) * 255).astype(np.uint8)


def visualise_objid(obj_id: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    ids = np.unique(obj_id)
    palette = {int(oid): rng.integers(50, 230, 3).tolist() for oid in ids}
    out = np.zeros((*obj_id.shape, 3), dtype=np.uint8)
    for oid, col in palette.items():
        mask = (obj_id == oid)
        out[mask] = col
    return out


def main():
    parser = argparse.ArgumentParser(description="ЛР5 — Bilateral Filter")
    parser.add_argument("--spp",     type=int,   default=4,   help="samples per pixel (noisy render)")
    parser.add_argument("--res",     type=int,   default=256, help="image resolution (square)")
    parser.add_argument("--radius",  type=int,   default=7,   help="filter radius in pixels")
    parser.add_argument("--sigma-s", type=float, default=4.0, help="spatial sigma")
    parser.add_argument("--sigma-d", type=float, default=0.8, help="depth sigma")
    parser.add_argument("--sigma-n", type=float, default=0.4, help="normal sigma (rad)")
    parser.add_argument("--outdir",  type=str,   default=".",  help="output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── 1. Render G-buffer ────────────────────────────────────────────────────
    print(f"Rendering {args.res}x{args.res} at {args.spp} spp ...")
    color, depth, normal, obj_id = render_gbuffer(
        H=args.res, W=args.res, spp=args.spp, seed=42)
    print("  render done.")

    # ── 2. Apply bilateral filter ─────────────────────────────────────────────
    print(f"Applying bilateral filter (radius={args.radius}, "
          f"ss={args.sigma_s}, sd={args.sigma_d}, sn={args.sigma_n}) ...")
    filtered = bilateral_filter_fast(
        color, depth, normal, obj_id,
        radius=args.radius,
        sigma_s=args.sigma_s,
        sigma_d=args.sigma_d,
        sigma_n=args.sigma_n,
    )
    print("  filter done.")

    # ── 3. Verification ───────────────────────────────────────────────────────
    energy = energy_conservation(color, filtered, obj_id)
    noise  = noise_reduction(color, filtered)
    print_report(energy, noise)

    # ── 4. Save images ────────────────────────────────────────────────────────
    noisy_u8    = tonemap(color)
    filtered_u8 = tonemap(filtered)
    depth_u8    = visualise_depth(depth)
    normal_u8   = visualise_normal(normal)
    objid_u8    = visualise_objid(obj_id)

    save_png(os.path.join(args.outdir, "noisy.png"),    noisy_u8)
    save_png(os.path.join(args.outdir, "filtered.png"), filtered_u8)
    if len(depth_u8.shape) == 2:
        if HAS_PLT:
            plt.imsave(os.path.join(args.outdir, "depth.png"),  depth_u8,  cmap="plasma")
            print(f"  saved {os.path.join(args.outdir, 'depth.png')}")
    save_png(os.path.join(args.outdir, "normal.png"),   normal_u8)
    save_png(os.path.join(args.outdir, "objid.png"),    objid_u8)
    save_comparison(
        os.path.join(args.outdir, "comparison.png"),
        noisy_u8, filtered_u8,
        label_l=f"Noisy ({args.spp} spp)",
        label_r=f"Filtered (r={args.radius})",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
