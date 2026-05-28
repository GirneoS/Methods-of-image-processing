"""
Билатеральный фильтр с использованием G-буфера (ЛР5).

Формула:
    g(p) = (1/Wp) * SUM_{q in window} f(q) * w(p,q)

Веса:
    spatial_w  = exp( -(dx^2 + dy^2) / (2 * sigma_s^2) )
    depth_w    = exp( -(d_q - d_p)^2 / (2 * sigma_d^2) )
    normal_w   = exp( -angle(n_p, n_q)^2 / (2 * sigma_n^2) )
    object_w   = 1 если obj_id(p)==obj_id(q), иначе 0    (жёсткое ребро)

    w(p,q) = spatial * depth * normal * object

Прямую и вторичную яркость фильтруют ОТДЕЛЬНО (по тем же весам),
затем складывают: filtered_total = filtered_direct + filtered_indirect.
"""

import numpy as np
from typing import Tuple


def bilateral_filter(
    img:     np.ndarray,    # (H,W,3) яркость
    depth:   np.ndarray,    # (H,W)
    normal:  np.ndarray,    # (H,W,3)
    obj_id:  np.ndarray,    # (H,W) int
    radius:  int   = 7,
    sigma_s: float = 4.0,
    sigma_d: float = 0.05,  # доля от диапазона глубин
    sigma_n: float = 0.4,   # радианы
) -> np.ndarray:
    """
    Векторизованная (NumPy) билатеральная фильтрация одного канала яркости.
    Размер окна = (2r+1) x (2r+1). Сложность ~ O(H*W*r^2).
    """
    H, W, _ = img.shape
    out         = np.zeros_like(img, dtype=np.float64)
    weight_sum  = np.zeros((H, W), dtype=np.float64)

    # depth_sigma: задаётся как доля от диапазона глубин
    d_range = float(depth.max() - depth.min() + 1e-9)
    sigma_d_abs = sigma_d * d_range

    inv2ss = 1.0 / (2 * sigma_s   ** 2)
    inv2sd = 1.0 / (2 * sigma_d_abs ** 2)
    inv2sn = 1.0 / (2 * sigma_n   ** 2)

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            qi = np.clip(np.arange(H) + di, 0, H - 1)
            qj = np.clip(np.arange(W) + dj, 0, W - 1)

            c_shift  = img   [np.ix_(qi, qj)]
            d_shift  = depth [np.ix_(qi, qj)]
            n_shift  = normal[np.ix_(qi, qj)]
            id_shift = obj_id[np.ix_(qi, qj)]

            # spatial Gauss (scalar)
            Gs = np.exp(-(di*di + dj*dj) * inv2ss)

            # depth Gauss
            dd = d_shift - depth
            Gd = np.exp(-(dd * dd) * inv2sd)

            # normal Gauss
            dot = np.sum(normal * n_shift, axis=-1)
            dot = np.clip(dot, -1.0, 1.0)
            angle2 = np.arccos(dot) ** 2
            Gn = np.exp(-angle2 * inv2sn)

            # object mask (жёсткое ребро)
            same_obj = (id_shift == obj_id).astype(np.float32)

            w = Gs * Gd * Gn * same_obj
            out        += w[:, :, np.newaxis] * c_shift
            weight_sum += w

    mask = weight_sum > 1e-12
    out[mask]  /= weight_sum[mask, np.newaxis]
    out[~mask]  = img[~mask]
    return out.astype(np.float32)


def bilateral_filter_split(
    direct:   np.ndarray,
    indirect: np.ndarray,
    depth:    np.ndarray,
    normal:   np.ndarray,
    obj_id:   np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прямую и вторичную яркость фильтруют ОТДЕЛЬНО (так требует ТЗ).
    Возвращает (filtered_direct, filtered_indirect, filtered_total).
    """
    fd = bilateral_filter(direct,   depth, normal, obj_id, **kwargs)
    fi = bilateral_filter(indirect, depth, normal, obj_id, **kwargs)
    return fd, fi, fd + fi


# ── Простой усредняющий фильтр (для сравнения с билатеральным) ────────────────
def box_filter(img: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Простое арифметическое среднее в окне (2r+1)x(2r+1) — типичный low-pass
    фильтр со слайдов 4-10 презентации. Размывает границы.
    """
    H, W, _ = img.shape
    out         = np.zeros_like(img, dtype=np.float64)
    weight_sum  = np.zeros((H, W), dtype=np.float64)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            qi = np.clip(np.arange(H) + di, 0, H - 1)
            qj = np.clip(np.arange(W) + dj, 0, W - 1)
            c_shift = img[np.ix_(qi, qj)]
            out        += c_shift
            weight_sum += 1.0
    return (out / weight_sum[:, :, np.newaxis]).astype(np.float32)
