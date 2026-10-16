"""2D geometry helpers: polygons, edge extraction, ray intersection, mirroring.

All lengths are metres and all angles radians.  Functions are vectorised where
it matters (ray tracing calls them once per frame with hundreds of scatterers
and hundreds of wall segments).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12

__all__ = [
    "unit",
    "rotate",
    "circle_polygon",
    "rect_polygon",
    "wall_polygon",
    "polygon_edges",
    "edge_normals_outward",
    "polygon_signed_area",
    "ray_segment_intersections",
    "mirror_point",
    "mirror_points",
    "reflect_direction",
    "point_in_polygon",
    "point_segment_distance",
    "project_u",
    "subdivide_segment",
]


def unit(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis, guarding against zero vectors."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


def rotate(points: np.ndarray, angle_rad: float, origin=(0.0, 0.0)) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    origin = np.asarray(origin, dtype=np.float64)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts - origin) @ rot.T + origin


def circle_polygon(x: float, y: float, r: float, n: int = 64) -> np.ndarray:
    """CCW regular polygon approximating a circle."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.stack([x + r * np.cos(t), y + r * np.sin(t)], axis=1)


def rect_polygon(x: float, y: float, w: float, h: float, rot_rad: float = 0.0) -> np.ndarray:
    """CCW rectangle centred at ``(x, y)``, rotated by ``rot_rad``."""
    local = np.array(
        [[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]],
        dtype=np.float64,
    )
    return rotate(local, rot_rad) + np.array([x, y], dtype=np.float64)


def wall_polygon(points: np.ndarray, thickness: float) -> np.ndarray:
    """Give an open polyline a body by offsetting it ``+-thickness/2``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < 2:
        raise ValueError("wall requires at least 2 points")
    half = max(float(thickness), _EPS) / 2.0
    if pts.shape[0] == 2:
        d = unit(pts[1] - pts[0])
        tang = np.stack([d, d])
    else:
        seg = unit(np.diff(pts, axis=0))
        tang = np.empty_like(pts)
        tang[0] = seg[0]
        tang[-1] = seg[-1]
        tang[1:-1] = unit(seg[:-1] + seg[1:])
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return np.vstack([pts + nrm * half, (pts - nrm * half)[::-1]])


def polygon_edges(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the closed edge list ``(a, b)`` of a polygon (last edge wraps)."""
    pts = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    return pts, np.roll(pts, -1, axis=0)


def polygon_signed_area(poly: np.ndarray) -> float:
    pts = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def edge_normals_outward(poly: np.ndarray) -> np.ndarray:
    """Outward unit normals of a *closed* polygon's edges.

    Relies on the right-hand normal ``(dy, -dx)`` which points outward for a
    CCW polygon; the sign is flipped automatically for CW input.
    """
    a, b = polygon_edges(poly)
    d = b - a
    n = unit(np.stack([d[:, 1], -d[:, 0]], axis=1))
    if polygon_signed_area(poly) < 0.0:
        n = -n
    return n


def ray_segment_intersections(
    origin: np.ndarray,
    dirs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ray/segment intersection test.

    Parameters
    ----------
    origin : (2,) or (N, 2)  ray origin(s)
    dirs : (N, 2)  ray directions (need not be normalised)
    a, b : (M, 2)  segment endpoints

    Returns
    -------
    t : (N, M) distance parameter along the ray
    u : (N, M) normalised position along the segment
    hit : (N, M) bool mask where ``t >= 0`` and ``0 <= u <= 1``
    """
    origin = np.asarray(origin, dtype=np.float64)
    if origin.ndim == 1:
        origin = origin[None, :]
    origin = origin[:, None, :]                          # (N, 1, 2)
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    e = b - a

    d_x = dirs[:, None, 0]
    d_y = dirs[:, None, 1]
    e_x = e[None, :, 0]
    e_y = e[None, :, 1]

    denom = d_x * e_y - d_y * e_x                       # (N, M)
    safe = np.where(np.abs(denom) < _EPS, np.nan, denom)
    ao = a[None, :, :] - origin                         # (N, M, 2)
    ao_x = ao[:, :, 0]
    ao_y = ao[:, :, 1]

    t = (ao_x * e_y - ao_y * e_x) / safe
    u = (ao_x * d_y - ao_y * d_x) / safe
    hit = np.isfinite(t) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return t, u, hit


def mirror_point(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirror point ``p`` across the infinite line through ``a`` and ``b``."""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), _EPS)
    return p - 2.0 * (float(n @ p) - float(n @ a)) * n


def mirror_points(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), _EPS)
    s = pts @ n - float(n @ a)
    return pts - 2.0 * s[:, None] * n[None, :]


def reflect_direction(d: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Specular reflection of direction ``d`` about ``normal``."""
    d = np.asarray(d, dtype=np.float64)
    n = unit(np.asarray(normal, dtype=np.float64))
    return d - 2.0 * np.sum(d * n, axis=-1, keepdims=True) * n


def point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = float(point[0]), float(point[1])
    poly = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    xs, ys = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = ((ys > y) != (y2 > y)) & (x < (x2 - xs) * (y - ys) / (y2 - ys + _EPS) + xs)
    return bool(np.count_nonzero(cond) % 2 == 1)


def project_u(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Normalised projection of ``point`` onto segment ``a-b`` (clamped to [0,1])."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < _EPS:
        return 0.0
    return float(np.clip(float((np.asarray(point) - a) @ ab) / denom, 0.0, 1.0))


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    u = project_u(point, a, b)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - (a + u * (b - a))))


def subdivide_segment(a: np.ndarray, b: np.ndarray, seg_len: float) -> tuple[np.ndarray, float]:
    """Split ``a-b`` into pieces of at most ``seg_len``.

    Returns the piece centres (K, 2) and the actual piece length.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    n = max(1, int(np.ceil(length / max(seg_len, _EPS))))
    ts = (np.arange(n, dtype=np.float64) + 0.5) / n
    return a[None, :] + ts[:, None] * (b - a)[None, :], length / n
