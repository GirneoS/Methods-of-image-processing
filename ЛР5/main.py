"""
ЛР5 - точка входа.

Запуск:
    python main.py --res 256 --spp 4 --radius 7

Что делает:
    1. Рендерит сцену из ЛР4 (Cornell Box) с G-буфером:
       direct, indirect, depth, normal, object_id.
    2. Применяет к (direct + indirect) простой box-фильтр - размывает края.
    3. Применяет к direct и indirect ОТДЕЛЬНО билатеральный фильтр с G-буфером;
       результат складывает: filtered = filtered_direct + filtered_indirect.
    4. Сохраняет картинки: noisy / box / bilateral / G-buffer карты / comparison.png.
    5. Печатает отчёт: энергия по объектам, std-шум до/после.
"""

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gbuffer  import render_gbuffer
from bilateral import bilateral_filter_split, box_filter
from verify   import energy_per_object, noise_stats, print_report


def tonemap(hdr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Авто-экспозиция: средняя яркость = 0.5, затем gamma -> uint8."""
    Y = 0.2126*hdr[...,0] + 0.7152*hdr[...,1] + 0.0722*hdr[...,2]
    mean_lum = float(np.mean(Y)) + 1e-10
    img = hdr * (0.5 / mean_lum)
    img = np.clip(img, 0, 1) ** (1.0 / gamma)
    return (img * 255).astype(np.uint8)


def vis_depth(d):
    x = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return (x * 255).astype(np.uint8)

def vis_normal(n):
    return (np.clip((n + 1) / 2, 0, 1) * 255).astype(np.uint8)

def vis_objid(obj_id):
    rng = np.random.default_rng(0)
    out = np.zeros((*obj_id.shape, 3), dtype=np.uint8)
    for oid in np.unique(obj_id):
        col = rng.integers(60, 230, 3)
        out[obj_id == oid] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res",     type=int,   default=256)
    ap.add_argument("--spp",     type=int,   default=4)
    ap.add_argument("--radius",  type=int,   default=7)
    ap.add_argument("--sigma-s", type=float, default=4.0)
    ap.add_argument("--sigma-d", type=float, default=0.05)
    ap.add_argument("--sigma-n", type=float, default=0.4)
    ap.add_argument("--outdir",  type=str,   default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Рендер
    print(f"Render Cornell Box {args.res}x{args.res}, spp={args.spp} ...")
    color, direct, indirect, depth, normal, obj_id = render_gbuffer(
        width=args.res, height=args.res, spp=args.spp, seed=42)

    # 2. Простой box-фильтр (для сравнения)
    print("Box filter (simple averaging) ...")
    box_out = box_filter(color, radius=args.radius)

    # 3. Билатеральный фильтр (раздельно direct/indirect)
    print(f"Bilateral filter r={args.radius}, sigma_s={args.sigma_s}, "
          f"sigma_d={args.sigma_d}, sigma_n={args.sigma_n} ...")
    fd, fi, bi_out = bilateral_filter_split(
        direct, indirect, depth, normal, obj_id,
        radius=args.radius, sigma_s=args.sigma_s,
        sigma_d=args.sigma_d, sigma_n=args.sigma_n)

    # 4. Верификация
    print_report(energy_per_object(color, box_out, obj_id),
                 noise_stats(color, box_out),  title="BOX filter")
    print_report(energy_per_object(color, bi_out, obj_id),
                 noise_stats(color, bi_out),   title="BILATERAL filter (G-buffer)")

    # 5. Картинки
    noisy_u8 = tonemap(color)
    box_u8   = tonemap(box_out)
    bi_u8    = tonemap(bi_out)
    dpt_u8   = vis_depth(depth)
    nrm_u8   = vis_normal(normal)
    oid_u8   = vis_objid(obj_id)

    def save(name, im, cmap=None):
        path = os.path.join(args.outdir, name)
        if cmap:
            plt.imsave(path, im, cmap=cmap)
        else:
            plt.imsave(path, im)
        print(f"  saved {path}")

    save("noisy.png",      noisy_u8)
    save("box.png",        box_u8)
    save("bilateral.png",  bi_u8)
    save("depth.png",      dpt_u8, cmap="plasma")
    save("normal.png",     nrm_u8)
    save("objid.png",      oid_u8)

    # comparison 3-way
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f"Noisy ({args.spp} spp)",
              f"Box filter r={args.radius}",
              f"Bilateral G-buffer r={args.radius}"]
    for ax, im, t in zip(axes, [noisy_u8, box_u8, bi_u8], titles):
        ax.imshow(im); ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.join(args.outdir, 'comparison.png')}")

    # G-buffer карты вместе
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dpt_u8, cmap="plasma"); axes[0].set_title("Depth");   axes[0].axis("off")
    axes[1].imshow(nrm_u8);                axes[1].set_title("Normal");  axes[1].axis("off")
    axes[2].imshow(oid_u8);                axes[2].set_title("Object ID"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "gbuffer.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.join(args.outdir, 'gbuffer.png')}")

    print("Done.")


if __name__ == "__main__":
    main()
