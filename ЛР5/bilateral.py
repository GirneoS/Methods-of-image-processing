"""
Bilateral filter with G-buffer guidance.

Formula (slide 32/45):
    g(p) = (1/Wp) * Σ_{q∈S} f(q) * Gs(|p−q|) * Gr(||p−q||)

where:
    Gs — spatial Gaussian (pixel distance)
    Gr — range kernel composed from:
            • depth difference   (σ_d)
            • normal similarity  (σ_n)
            • hard object-id edge (weight = 0 if obj_id differs)

Energy conservation (slide 45):
    Normalization per pixel: divide by Wp = Σ Gs * Gr
    Per-object brightness conservation guaranteed by the normalization.
"""

import numpy as np
# ── Pure NumPy implementation ─────────────────────────────────────────────────

def bilateral_filter(
    color:   np.ndarray,   # (H,W,3)
    depth:   np.ndarray,   # (H,W)
    normal:  np.ndarray,   # (H,W,3)
    obj_id:  np.ndarray,   # (H,W) int
    radius:  int   = 5,
    sigma_s: float = 3.0,  # spatial std (pixels)
    sigma_d: float = 0.5,  # depth std
    sigma_n: float = 0.3,  # normal angle std (radians)
) -> np.ndarray:
    """
    Apply G-buffer-guided bilateral filter.

    Returns filtered image (H,W,3) with per-object energy conservation.
    """
    H, W, _ = color.shape
    out = np.zeros_like(color)

    # Precompute Gaussian coefficients
    inv2ss = 1.0 / (2 * sigma_s ** 2)
    inv2sd = 1.0 / (2 * sigma_d ** 2)
    inv2sn = 1.0 / (2 * sigma_n ** 2)

    for i in range(H):
        for j in range(W):
            acc   = np.zeros(3, dtype=np.float64)
            Wp    = 0.0

            p_depth  = float(depth[i, j])
            p_normal = normal[i, j].astype(np.float64)
            p_id     = int(obj_id[i, j])

            i0, i1 = max(0, i - radius), min(H, i + radius + 1)
            j0, j1 = max(0, j - radius), min(W, j + radius + 1)

            for qi in range(i0, i1):
                for qj in range(j0, j1):
                    # Hard edge: skip different objects
                    if int(obj_id[qi, qj]) != p_id:
                        continue

                    # Spatial Gaussian Gs
                    dist2_px = (qi - i) ** 2 + (qj - j) ** 2
                    Gs = np.exp(-dist2_px * inv2ss)

                    # Depth Gaussian
                    dd = float(depth[qi, qj]) - p_depth
                    Gd = np.exp(-(dd * dd) * inv2sd)

                    # Normal Gaussian (cosine → angle)
                    q_normal = normal[qi, qj].astype(np.float64)
                    cos_a = np.clip(np.dot(p_normal, q_normal), -1.0, 1.0)
                    angle2 = np.arccos(cos_a) ** 2
                    Gn = np.exp(-angle2 * inv2sn)

                    w = Gs * Gd * Gn
                    acc += w * color[qi, qj]
                    Wp  += w

            if Wp > 1e-12:
                out[i, j] = acc / Wp
            else:
                out[i, j] = color[i, j]

    return out.astype(np.float32)


def bilateral_filter_fast(
    color:   np.ndarray,
    depth:   np.ndarray,
    normal:  np.ndarray,
    obj_id:  np.ndarray,
    radius:  int   = 5,
    sigma_s: float = 3.0,
    sigma_d: float = 0.5,
    sigma_n: float = 0.3,
) -> np.ndarray:
    """
    Vectorised (NumPy) version — much faster than pure Python loops.
    """
    H, W, _ = color.shape
    out = np.zeros_like(color, dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    inv2ss = 1.0 / (2 * sigma_s ** 2)
    inv2sd = 1.0 / (2 * sigma_d ** 2)
    inv2sn = 1.0 / (2 * sigma_n ** 2)

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            # Shift: q = (i+di, j+dj)
            qi = np.clip(np.arange(H) + di, 0, H - 1)
            qj = np.clip(np.arange(W) + dj, 0, W - 1)
            # Build shifted arrays
            c_shift = color [np.ix_(qi, qj)]   # (H,W,3)
            d_shift = depth [np.ix_(qi, qj)]   # (H,W)
            n_shift = normal[np.ix_(qi, qj)]   # (H,W,3)
            id_shift= obj_id[np.ix_(qi, qj)]   # (H,W)

            # Spatial kernel
            Gs = np.exp(-(di * di + dj * dj) * inv2ss)   # scalar

            # Depth kernel
            dd = d_shift - depth
            Gd = np.exp(-(dd * dd) * inv2sd)              # (H,W)

            # Normal kernel
            dot = np.sum(normal * n_shift, axis=-1)       # (H,W)
            dot = np.clip(dot, -1.0, 1.0)
            angle2 = np.arccos(dot) ** 2
            Gn = np.exp(-angle2 * inv2sn)                 # (H,W)

            # Edge mask: same object id
            same_obj = (id_shift == obj_id).astype(np.float32)  # (H,W)

            w = Gs * Gd * Gn * same_obj                   # (H,W)

            out         += w[:, :, np.newaxis] * c_shift
            weight_sum  += w

    mask = weight_sum > 1e-12
    out[mask] /= weight_sum[mask, np.newaxis]
    out[~mask] = color[~mask]

    return out.astype(np.float32)
