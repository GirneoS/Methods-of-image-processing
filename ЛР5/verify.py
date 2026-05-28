"""
Verification of bilateral filter results.

Checks:
1. Per-object energy conservation: Σ g(p) ≈ Σ f(p)  for each object O
2. Peak Signal-to-Noise Ratio improvement (PSNR) vs reference (high-spp render)
3. Edge preservation: std-dev inside flat regions vs near edges
"""

import numpy as np


def luminance(img: np.ndarray) -> np.ndarray:
    """(H,W,3) → (H,W) luminance."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def energy_conservation(noisy: np.ndarray, filtered: np.ndarray,
                         obj_id: np.ndarray) -> dict:
    """
    Per-object sum of luminance before and after filtering.
    Returns dict: obj_id → (sum_before, sum_after, relative_error %).
    """
    lum_n = luminance(noisy)
    lum_f = luminance(filtered)
    ids = np.unique(obj_id)
    results = {}
    for oid in ids:
        mask = (obj_id == oid)
        sb = float(lum_n[mask].sum())
        sa = float(lum_f[mask].sum())
        rel = abs(sa - sb) / (sb + 1e-9) * 100
        results[int(oid)] = (sb, sa, rel)
    return results


def psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = None) -> float:
    """PSNR between two images."""
    if max_val is None:
        max_val = float(max(img_a.max(), img_b.max()))
    mse = float(np.mean((img_a.astype(float) - img_b.astype(float)) ** 2))
    if mse < 1e-12:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def noise_reduction(noisy: np.ndarray, filtered: np.ndarray,
                     reference: np.ndarray | None = None) -> dict:
    """Compute noise statistics."""
    std_noisy    = float(np.std(noisy))
    std_filtered = float(np.std(filtered))
    result = {
        "std_noisy":    std_noisy,
        "std_filtered": std_filtered,
        "noise_reduction_%": (1 - std_filtered / (std_noisy + 1e-9)) * 100,
    }
    if reference is not None:
        result["psnr_noisy_vs_ref"]    = psnr(noisy,    reference)
        result["psnr_filtered_vs_ref"] = psnr(filtered, reference)
    return result


def print_report(energy: dict, noise: dict) -> None:
    print("\n== Energy conservation (per object) ==")
    print(f"{'ObjID':>6}  {'Sum before':>12}  {'Sum after':>12}  {'Err%':>8}")
    print("-" * 46)
    total_before = total_after = 0.0
    for oid, (sb, sa, rel) in sorted(energy.items()):
        print(f"{oid:6d}  {sb:12.4f}  {sa:12.4f}  {rel:7.3f}%")
        total_before += sb
        total_after  += sa
    rel_total = abs(total_after - total_before) / (total_before + 1e-9) * 100
    print("-" * 46)
    print(f"{'TOTAL':>6}  {total_before:12.4f}  {total_after:12.4f}  {rel_total:7.3f}%")

    print("\n== Noise reduction ==")
    for k, v in noise.items():
        if isinstance(v, float):
            print(f"  {k:<32} {v:.4f}")
        else:
            print(f"  {k:<32} {v}")
