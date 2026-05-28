"""
G-buffer renderer: for each pixel stores
  - color   (H x W x 3) — noisy path-traced radiance
  - depth   (H x W)     — distance to first hit
  - normal  (H x W x 3) — surface normal at first hit
  - obj_id  (H x W)     — integer object index at first hit
"""

import numpy as np
from typing import List
from scene import HitInfo, build_cornell_box


# ── Camera ───────────────────────────────────────────────────────────────────

def get_ray(i, j, H, W, fov_deg=50.0):
    """Return (origin, direction) for pixel (i=row, j=col)."""
    aspect = W / H
    fov_rad = np.radians(fov_deg)
    scale = np.tan(fov_rad / 2)

    px = (2 * (j + 0.5) / W - 1) * aspect * scale
    py = (1 - 2 * (i + 0.5) / H) * scale

    origin = np.array([2.75, 2.75, -0.5])
    direction = np.array([px, py, 1.0])
    direction /= np.linalg.norm(direction)
    return origin, direction


# ── Scene intersection ────────────────────────────────────────────────────────

def scene_intersect(origin, direction, objects):
    closest = None
    for obj in objects:
        hit = obj.intersect(origin, direction)
        if hit is not None and (closest is None or hit.t < closest.t):
            closest = hit
    return closest


# ── Simple path tracer (1 bounce diffuse + NEE) ───────────────────────────────

def _lights(objects):
    return [o for o in objects if np.any(o.material.emission > 0)]


def _rand_cosine(normal, rng):
    """Cosine-weighted hemisphere sample."""
    u1, u2 = rng.random(), rng.random()
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    lx, ly, lz = r * np.cos(phi), r * np.sin(phi), np.sqrt(max(0.0, 1 - u1))
    # Build ONB
    up = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = np.cross(normal, up); t /= np.linalg.norm(t)
    b = np.cross(normal, t)
    return lx * t + ly * b + lz * normal


def trace_direct(hit: HitInfo, objects, lights, rng) -> np.ndarray:
    """Direct lighting estimate (NEE)."""
    if not lights:
        return np.zeros(3)
    light = lights[rng.integers(len(lights))]
    # Sample point on light sphere
    u, v = rng.random(), rng.random()
    theta = np.arccos(1 - 2 * u)
    phi = 2 * np.pi * v
    lp = light.center + light.radius * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)])
    to_l = lp - hit.point
    dist = np.linalg.norm(to_l)
    ld = to_l / dist
    cos_hit = np.dot(hit.normal, ld)
    if cos_hit <= 0:
        return np.zeros(3)
    shadow = scene_intersect(hit.point + hit.normal * 1e-4, ld, objects)
    if shadow is not None and shadow.t < dist - 1e-3:
        return np.zeros(3)
    area = 4 * np.pi * light.radius ** 2
    pdf = 1.0 / (area * len(lights))
    cos_light = max(0, -np.dot(light.material.emission / (np.linalg.norm(light.material.emission) + 1e-9), ld))
    return hit.material.color * light.material.emission * cos_hit / (dist ** 2 * np.pi * pdf + 1e-9)


def path_radiance(origin, direction, objects, lights, rng, depth=0) -> np.ndarray:
    """2-bounce path tracer."""
    hit = scene_intersect(origin, direction, objects)
    if hit is None:
        return np.zeros(3)
    colour = hit.material.emission.copy()
    if depth < 2 and np.mean(hit.material.color) > 1e-6:
        direct = trace_direct(hit, objects, lights, rng)
        colour = colour + direct
        if depth < 1:
            new_dir = _rand_cosine(hit.normal, rng)
            indirect = path_radiance(
                hit.point + hit.normal * 1e-4, new_dir, objects, lights, rng, depth + 1)
            colour = colour + hit.material.color * indirect
    return colour


# ── G-buffer render ───────────────────────────────────────────────────────────

def render_gbuffer(H: int = 256, W: int = 256, spp: int = 4, seed: int = 42):
    """
    Render scene and return:
      color  (H,W,3) float32   — averaged radiance (noisy)
      depth  (H,W)   float32
      normal (H,W,3) float32
      obj_id (H,W)   int32
    """
    rng = np.random.default_rng(seed)
    objects = build_cornell_box()
    lights = _lights(objects)

    color  = np.zeros((H, W, 3), dtype=np.float64)
    depth  = np.full((H, W), np.inf, dtype=np.float64)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    obj_id = np.full((H, W), -1, dtype=np.int32)

    for i in range(H):
        if i % 32 == 0:
            print(f"  row {i}/{H} …")
        for j in range(W):
            # G-buffer from first hit (deterministic, no jitter)
            o, d = get_ray(i, j, H, W)
            hit = scene_intersect(o, d, objects)
            if hit is not None:
                depth[i, j]    = hit.t
                normal[i, j]   = hit.normal
                obj_id[i, j]   = hit.obj_id

            # Noisy colour: average over spp samples
            acc = np.zeros(3)
            for _ in range(spp):
                # jitter ray slightly for anti-aliasing / noise
                jitter = rng.standard_normal(2) * 0.4
                o2, d2 = get_ray(i + jitter[0]/H, j + jitter[1]/W, H, W)
                acc += path_radiance(o2, d2, objects, lights, rng)
            color[i, j] = acc / spp

    # Replace inf depth with max finite depth
    finite = depth[np.isfinite(depth)]
    if len(finite):
        depth[~np.isfinite(depth)] = finite.max()

    return color.astype(np.float32), depth.astype(np.float32), \
           normal.astype(np.float32), obj_id
