"""Ray tracing: line-of-sight occlusion and image-method multipath.

Model (``docs/DESIGN.md`` sections 6.3 / 6.4):

* **Occlusion** -- a scatterer is invisible when the straight segment from the
  radar crosses an *opaque* edge (``tau < 0.5``).  Transparent edges attenuate
  by ``sqrt(tau)`` instead of blocking.
* **Multipath** -- specular reflections are solved exactly with the image
  method.  Order 1 mirrors the radar across one wall; order 2 mirrors twice.
  Every reflection point must land inside its wall segment and every leg must
  be unobstructed.

Path semantics -- one entry of :class:`EchoPaths` describes a *one-way*
geometric path whose length equals the equivalent point-target range::

    tau = 2 * equiv_range / c          (two-way delay)
    f_b = 2 * S * equiv_range / c      (beat frequency)

``theta_rad`` is the **departure** azimuth from the radar (the array sees the
first hop), ``v_eff`` is the effective radial velocity ``d(equiv_range)/dt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .materials import Material
from .radar_config import RadarConfig
from .scene import EdgeSet, Scene
from .scatterers import ScattererSet

__all__ = ["EchoPaths", "trace_paths"]

# intersection tolerances: reflection points sit exactly on their wall, so
# hits closer than _TOL_NEAR (own origin) or at the far end of a leg
# (_TOL_FAR, the reflecting wall itself) must not count as blockers.
_TOL_NEAR = 1e-6
_TOL_FAR = 1e-6

_OCCLUSION_CHUNK = 1024


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #
@dataclass
class EchoPaths:
    """Flattened list of return paths."""

    scatterer: np.ndarray       # (P,) index into ScattererSet
    equiv_range: np.ndarray     # (P,) metres, one-way geometric path length
    theta_rad: np.ndarray       # (P,) departure azimuth (world frame, radians)
    v_eff: np.ndarray           # (P,) effective radial velocity (+ == away)
    cos_psi: np.ndarray         # (P,) incidence cosine at the scatterer
    gain: np.ndarray            # (P,) complex amplitude product of reflections
    atten: np.ndarray           # (P,) float transmission factor along the path
    bounce: np.ndarray          # (P,) int, 0 = direct

    def __len__(self) -> int:
        return int(self.scatterer.shape[0])

    @property
    def count(self) -> int:
        return len(self)

    @classmethod
    def empty(cls) -> "EchoPaths":
        return cls(
            scatterer=np.zeros(0, dtype=int),
            equiv_range=np.zeros(0),
            theta_rad=np.zeros(0),
            v_eff=np.zeros(0),
            cos_psi=np.zeros(0),
            gain=np.zeros(0, dtype=complex),
            atten=np.zeros(0),
            bounce=np.zeros(0, dtype=int),
        )

    def counts_by_bounce(self) -> dict[int, int]:
        if not len(self):
            return {}
        values, counts = np.unique(self.bounce, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def select(self, mask: np.ndarray) -> "EchoPaths":
        return EchoPaths(
            self.scatterer[mask], self.equiv_range[mask], self.theta_rad[mask],
            self.v_eff[mask], self.cos_psi[mask], self.gain[mask],
            self.atten[mask], self.bounce[mask],
        )

    def concat(self, other: "EchoPaths") -> "EchoPaths":
        if not len(self):
            return other
        if not len(other):
            return self
        return EchoPaths(
            np.concatenate([self.scatterer, other.scatterer]),
            np.concatenate([self.equiv_range, other.equiv_range]),
            np.concatenate([self.theta_rad, other.theta_rad]),
            np.concatenate([self.v_eff, other.v_eff]),
            np.concatenate([self.cos_psi, other.cos_psi]),
            np.concatenate([self.gain, other.gain]),
            np.concatenate([self.atten, other.atten]),
            np.concatenate([self.bounce, other.bounce]),
        )


# --------------------------------------------------------------------------- #
# occlusion
# --------------------------------------------------------------------------- #
def _blocked_and_atten(
    origin: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    edges: EdgeSet,
    opaque: np.ndarray,
    log_tau_half: np.ndarray,
    *,
    chunk: int = _OCCLUSION_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """Test ``len(dirs)`` rays against every edge.

    Returns ``(blocked, atten)`` as float/bool arrays of shape ``(len(dirs),)``.
    ``log_tau_half`` is ``0.5*log(tau)`` per edge (0 for opaque edges, which
    never reach the attenuation sum).
    """
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    t_max = np.asarray(t_max, dtype=np.float64).reshape(-1)
    n_rays = dirs.shape[0]
    blocked = np.zeros(n_rays, dtype=bool)
    log_atten = np.zeros(n_rays, dtype=np.float64)

    if len(edges) == 0 or n_rays == 0:
        return blocked, np.exp(log_atten)

    for start in range(0, n_rays, chunk):
        stop = min(start + chunk, n_rays)
        t, _u, hit = geo.ray_segment_intersections(
            origin, dirs[start:stop], edges.a, edges.b
        )
        hit &= (t > _TOL_NEAR) & (t < t_max[start:stop, None] - _TOL_FAR)
        if not hit.any():
            continue
        blocked[start:stop] = (hit & opaque[None, :]).any(axis=1)
        log_atten[start:stop] = (hit * log_tau_half[None, :]).sum(axis=1)

    return blocked, np.exp(log_atten)


def _material_table(scene: Scene, edges: EdgeSet) -> list[Material]:
    ids = scene.material_ids()
    return [scene.materials[ids[int(i)]] for i in edges.material_index]


def _make_occluder(scene: Scene, edges: EdgeSet):
    materials = _material_table(scene, edges)
    opaque = np.array([m.opaque for m in materials], dtype=bool)
    tau = np.array([1.0 if m.opaque else max(m.tau, 1e-6) for m in materials])
    log_tau_half = np.where(opaque, 0.0, 0.5 * np.log(tau))
    return opaque, log_tau_half


# --------------------------------------------------------------------------- #
# image method helpers
# --------------------------------------------------------------------------- #
def _mirror_radar(radar_pos: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirror a point across the infinite line through ``a`` and ``b``."""
    radar_pos = np.asarray(radar_pos, dtype=np.float64).reshape(2)
    d = b - a
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    s = float(n @ radar_pos) - float(n @ a)
    return radar_pos - 2.0 * s * n


def _reflect_point_via_image(
    image: np.ndarray, target: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find where the line ``image -> target`` crosses the wall ``a-b``.

    Returns ``(point, u, valid)`` where ``u`` is the normalised position along
    the wall segment.
    """
    origin = np.atleast_2d(np.asarray(image, dtype=np.float64))
    target = np.atleast_2d(np.asarray(target, dtype=np.float64))
    w = target - origin
    t, u, hit = geo.ray_segment_intersections(
        image, w, np.atleast_2d(a), np.atleast_2d(b)
    )
    point = origin + t[:, 0][:, None] * w
    valid = hit[:, 0] & (t[:, 0] > 1e-9) & (t[:, 0] < 1.0 - 1e-12)
    return point, u[:, 0], valid


# --------------------------------------------------------------------------- #
# path tracing
# --------------------------------------------------------------------------- #
def trace_paths(
    scene: Scene,
    scatterers: ScattererSet,
    radar_pos: np.ndarray,
    radar_vel: np.ndarray | None = None,
    scatterer_vel: np.ndarray | None = None,
    *,
    cfg: RadarConfig | None = None,
    max_order: int | None = None,
    wall_candidates: int = 16,
    wall_candidates_order2: int = 8,
    max_multipath_paths: int = 400,
    max_range_m: float | None = None,
    include_direct: bool = True,
) -> EchoPaths:
    """Trace direct + multipath returns for every scatterer.

    Parameters
    ----------
    radar_pos : (2,) world position of the radar phase centre
    radar_vel : (2,) world velocity of the radar (used for ``v_eff``)
    scatterer_vel : (N, 2) optional per-scatterer velocity (default: static)
    max_order : reflection order cap (defaults to ``cfg.max_reflection_order``)
    wall_candidates : how many walls (nearest to the radar) may act as the
        first reflector
    wall_candidates_order2 : how many for the *second* reflector
    max_multipath_paths : per-order cap on multiplexed paths, strongest kept
    max_range_m : drop paths longer than this (default: ``cfg.max_range_m``).
        Without the gate a long path folds back into the range window and shows
        up as a phantom target, because the beat frequency wraps at ``fs``.
    """
    radar_pos = np.asarray(radar_pos, dtype=np.float64).reshape(2)
    radar_vel = (
        np.zeros(2, dtype=np.float64)
        if radar_vel is None
        else np.asarray(radar_vel, dtype=np.float64).reshape(2)
    )
    if max_order is None:
        max_order = int(cfg.max_reflection_order) if cfg is not None else 1
    max_order = int(max_order)
    if max_range_m is None and cfg is not None:
        max_range_m = float(cfg.max_range_m)
    if scatterer_vel is None:
        scatterer_vel = np.zeros_like(scatterers.position)
    scatterer_vel = np.asarray(scatterer_vel, dtype=np.float64)

    if not len(scatterers):
        return EchoPaths.empty()

    edges = scene.build_edges()
    opaque, log_tau_half = _make_occluder(scene, edges)

    direct = (
        _direct_paths(radar_pos, radar_vel, scatterers, scatterer_vel,
                      edges, opaque, log_tau_half)
        if include_direct
        else EchoPaths.empty()
    )

    if max_order < 1 or len(edges) == 0:
        return _range_gate(direct, max_range_m)

    multipath = _multipath_paths(
        scene, edges, opaque, log_tau_half, scatterers, scatterer_vel,
        radar_pos, radar_vel,
        max_order=max_order,
        wall_candidates=wall_candidates,
        wall_candidates_order2=wall_candidates_order2,
        max_paths=max_multipath_paths,
    )
    return _range_gate(direct.concat(multipath), max_range_m)


def _range_gate(paths: EchoPaths, max_range_m: float | None) -> EchoPaths:
    """Drop paths beyond the unambiguous range window."""
    if paths.count == 0 or max_range_m is None:
        return paths
    inside = paths.equiv_range <= float(max_range_m)
    if inside.all():
        return paths
    return paths.select(inside)


def _direct_paths(
    radar_pos, radar_vel, scatterers, scatterer_vel,
    edges, opaque, log_tau_half,
) -> EchoPaths:
    delta = scatterers.position - radar_pos[None, :]
    rng = np.linalg.norm(delta, axis=1)
    keep = rng > 1e-6
    if not keep.any():
        return EchoPaths.empty()

    idx = np.nonzero(keep)[0]
    rng = rng[idx]
    unit = delta[idx] / rng[:, None]

    blocked, atten = _blocked_and_atten(radar_pos, unit, rng, edges, opaque, log_tau_half)
    visible = ~blocked
    if not visible.any():
        return EchoPaths.empty()

    idx = idx[visible]
    rng = rng[visible]
    unit = unit[visible]
    atten = atten[visible]

    # incidence cosine at the scatterer (radar direction == -unit)
    scatter_dir = -unit
    cos_psi = np.sum(scatter_dir * scatterers.normal[idx], axis=1)
    isotropic = np.linalg.norm(scatterers.normal[idx], axis=1) < 1e-9
    cos_psi = np.where(isotropic, 1.0, np.clip(cos_psi, 0.0, 1.0))

    v_eff = np.sum(unit * (scatterer_vel[idx] - radar_vel[None, :]), axis=1)
    theta = np.arctan2(unit[:, 1], unit[:, 0])

    return EchoPaths(
        scatterer=idx,
        equiv_range=rng,
        theta_rad=theta,
        v_eff=v_eff,
        cos_psi=cos_psi,
        gain=np.ones(idx.size, dtype=complex),
        atten=atten,
        bounce=np.zeros(idx.size, dtype=int),
    )


def _wall_candidates(
    radar_pos: np.ndarray,
    edges: EdgeSet,
    opaque: np.ndarray,
    log_tau_half: np.ndarray,
    k: int,
    *,
    max_edges_per_owner: int = 4,
) -> np.ndarray:
    """Indices of up to ``k`` *useful* reflectors seen from ``radar_pos``.

    Ranking by distance alone is wrong: a circle is approximated by 64 tiny
    edges, so a nearby bush would fill the whole quota and the long walls -- the
    only things that actually produce specular paths -- would never be
    considered.  Rank by ``edge_length / distance`` instead and cap how many
    edges a single primitive may contribute.
    """
    if len(edges) == 0 or k <= 0:
        return np.zeros(0, dtype=int)

    centres = 0.5 * (edges.a + edges.b)
    delta = centres - radar_pos[None, :]
    rng = np.linalg.norm(delta, axis=1)
    usable = rng > 1e-6
    if not usable.any():
        return np.zeros(0, dtype=int)

    unit = delta[usable] / rng[usable][:, None]
    blocked, _ = _blocked_and_atten(
        radar_pos, unit, rng[usable], edges, opaque, log_tau_half
    )

    merit = np.full(len(edges), -1.0)
    index = np.nonzero(usable)[0][~blocked]
    merit[index] = edges.length()[index] / np.maximum(rng[index], 1e-3)

    chosen: list[int] = []
    per_owner: dict[int, int] = {}
    for candidate in np.argsort(-merit):
        if merit[candidate] <= 0.0:
            break
        owner = int(edges.owner[candidate])
        if per_owner.get(owner, 0) >= int(max_edges_per_owner):
            continue
        per_owner[owner] = per_owner.get(owner, 0) + 1
        chosen.append(int(candidate))
        if len(chosen) >= int(k):
            break
    return np.asarray(chosen, dtype=int)


def _multipath_paths(
    scene: Scene, edges: EdgeSet, opaque, log_tau_half, scatterers, scatterer_vel,
    radar_pos, radar_vel, *, max_order, wall_candidates, wall_candidates_order2, max_paths,
) -> EchoPaths:
    materials = _material_table(scene, edges)
    normals = scatterers.normal
    isotropic = np.linalg.norm(normals, axis=1) < 1e-9
    positions = scatterers.position

    walls1 = _wall_candidates(radar_pos, edges, opaque, log_tau_half, wall_candidates)
    if walls1.size == 0:
        return EchoPaths.empty()

    records: dict[str, list[np.ndarray]] = {
        "scatterer": [], "equiv_range": [], "theta_rad": [], "v_eff": [],
        "cos_psi": [], "gain": [], "atten": [], "bounce": [],
    }
    leg1_origins: list[np.ndarray] = []
    leg1_dirs: list[np.ndarray] = []
    leg1_len: list[np.ndarray] = []
    leg2_origins: list[np.ndarray] = []
    leg2_dirs: list[np.ndarray] = []
    leg2_len: list[np.ndarray] = []
    leg3: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    record_owner: list[np.ndarray] = []       # index into the per-record lists

    # ---- order 1 ---------------------------------------------------------
    flat_offset = 0
    for w in walls1:
        a, b = edges.a[w], edges.b[w]
        image = _mirror_radar(radar_pos, a, b)
        p1, _u, valid = _reflect_point_via_image(image, positions, a, b)
        if not valid.any():
            continue
        path_len = np.linalg.norm(positions - image[None, :], axis=1)

        rad = radar_pos[None, :]
        leg1_vec = p1 - rad
        l1 = np.linalg.norm(leg1_vec, axis=1)
        leg2_vec = positions - p1
        l2 = np.linalg.norm(leg2_vec, axis=1)

        good = valid & (path_len > 1e-3) & (l1 > 1e-6) & (l2 > 1e-6)
        if not good.any():
            continue
        idx = np.nonzero(good)[0]

        u1 = leg1_vec[idx] / l1[idx][:, None]
        u2 = leg2_vec[idx] / l2[idx][:, None]

        cos_wall = np.abs(np.sum(u1 * edges.normal[w][None, :], axis=1))
        gain = materials[w].reflection_amplitude(np.arccos(np.clip(cos_wall, 0.0, 1.0)))

        cos_psi = np.where(
            isotropic[idx], 1.0, np.clip(np.sum(-u2 * normals[idx], axis=1), 0.0, 1.0)
        )
        theta = np.arctan2(u1[:, 1], u1[:, 0])
        v_eff = np.sum(u2 * scatterer_vel[idx], axis=1) - np.sum(u1 * radar_vel[None, :], axis=1)

        start = flat_offset
        _push(records, idx, path_len[idx], theta, v_eff, cos_psi, gain, 1)
        leg1_origins.append(np.repeat(radar_pos[None, :], idx.size, axis=0))
        leg1_dirs.append(u1); leg1_len.append(l1[idx])
        leg2_origins.append(p1[idx]); leg2_dirs.append(u2); leg2_len.append(l2[idx])
        leg3.append(None)
        record_owner.append(np.arange(start, start + idx.size))
        flat_offset += idx.size

    # ---- order 2 ---------------------------------------------------------
    if max_order >= 2:
        for w in walls1:
            a1, b1 = edges.a[w], edges.b[w]
            image1 = _mirror_radar(radar_pos, a1, b1)
            # second reflector candidates: nearest walls to the first image
            walls2 = _wall_candidates_nearest(image1, edges, wall_candidates_order2, exclude=w)
            for v in walls2:
                a2, b2 = edges.a[v], edges.b[v]
                image2 = _mirror_radar(image1, a2, b2)

                p2, _u2, valid2 = _reflect_point_via_image(image2, positions, a2, b2)
                if not valid2.any():
                    continue
                # fold back to the first reflection point using the image of image1
                _p1, _u1w, valid1 = _reflect_point_via_image(image1, p2, a1, b1)
                good = valid2 & valid1
                if not good.any():
                    continue
                idx = np.nonzero(good)[0]
                p1_pts = _p1[idx]
                p2_pts = p2[idx]

                path_len = np.linalg.norm(positions[idx] - image2[None, :], axis=1)
                leg1_vec = p1_pts - radar_pos[None, :]
                leg2_vec = p2_pts - p1_pts
                leg3_vec = positions[idx] - p2_pts
                l1 = np.linalg.norm(leg1_vec, axis=1)
                l2 = np.linalg.norm(leg2_vec, axis=1)
                l3 = np.linalg.norm(leg3_vec, axis=1)
                ok = (l1 > 1e-6) & (l2 > 1e-6) & (l3 > 1e-6) & (path_len > 1e-3)
                if not ok.any():
                    continue
                idx = idx[ok]
                p1_pts, p2_pts = p1_pts[ok], p2_pts[ok]
                l1, l2, l3 = l1[ok], l2[ok], l3[ok]
                path_len = path_len[ok]
                leg1_vec, leg2_vec, leg3_vec = leg1_vec[ok], leg2_vec[ok], leg3_vec[ok]

                u1 = leg1_vec / l1[:, None]
                u2d = leg2_vec / l2[:, None]
                u3 = leg3_vec / l3[:, None]

                cos_w1 = np.abs(np.sum(u1 * edges.normal[w][None, :], axis=1))
                cos_w2 = np.abs(np.sum(u2d * edges.normal[v][None, :], axis=1))
                gain = materials[w].reflection_amplitude(np.arccos(np.clip(cos_w1, 0, 1))) * \
                    materials[v].reflection_amplitude(np.arccos(np.clip(cos_w2, 0, 1)))

                cos_psi = np.where(
                    isotropic[idx], 1.0, np.clip(np.sum(-u3 * normals[idx], axis=1), 0.0, 1.0)
                )
                theta = np.arctan2(u1[:, 1], u1[:, 0])
                v_eff = np.sum(u3 * scatterer_vel[idx], axis=1) - np.sum(
                    u1 * radar_vel[None, :], axis=1
                )

                start = flat_offset
                _push(records, idx, path_len, theta, v_eff, cos_psi, gain, 2)
                leg1_origins.append(np.repeat(radar_pos[None, :], idx.size, axis=0))
                leg1_dirs.append(u1); leg1_len.append(l1)
                leg2_origins.append(p1_pts); leg2_dirs.append(u2d); leg2_len.append(l2)
                leg3.append((p2_pts, u3, l3))
                record_owner.append(np.arange(start, start + idx.size))
                flat_offset += idx.size

    if not records["scatterer"]:
        return EchoPaths.empty()

    total = int(sum(arr.size for arr in records["scatterer"]))
    candidates = _assemble(records, total)

    # flat record index -> group index, so occlusion runs in per-wall batches
    flat_to_group = np.empty(total, dtype=int)
    for group_index, owner in enumerate(record_owner):
        flat_to_group[owner] = group_index

    # ---- prune by estimated energy before the (expensive) occlusion test --
    # The budget is allocated PER reflection order.  A global cap would starve
    # higher orders: two-bounce paths carry two reflection losses, so their
    # energy estimate is far below the single-bounce population and they would
    # always be pruned away before the occlusion test ever sees them.
    energy = (np.abs(candidates.gain) ** 2) / np.maximum(candidates.equiv_range, 1e-3) ** 4
    quotas = {1: max_paths, 2: max(32, max_paths // 4)}
    parts: list[np.ndarray] = []
    for order in np.unique(candidates.bounce):
        group = np.nonzero(candidates.bounce == order)[0]
        quota = quotas.get(int(order), max_paths)
        if max_paths > 0 and group.size > quota:
            group = group[np.argpartition(-energy[group], quota - 1)[:quota]]
        parts.append(group)
    keep = (np.sort(np.concatenate(parts)) if parts
            else np.arange(candidates.count))

    blocked_flags = np.zeros(keep.size, dtype=bool)
    atten_factors = np.ones(keep.size, dtype=np.float64)

    for group_index in np.unique(flat_to_group[keep]):
        owner = record_owner[int(group_index)]
        pos = np.clip(np.searchsorted(keep, owner), 0, keep.size - 1)
        in_group = keep[pos] == owner
        if not in_group.any():
            continue
        sel = np.nonzero(in_group)[0]           # positions inside this group's arrays
        out_pos = pos[sel]                      # matching positions inside `keep`

        b1, a1_ = _blocked_and_atten(
            leg1_origins[group_index][sel], leg1_dirs[group_index][sel],
            leg1_len[group_index][sel], edges, opaque, log_tau_half,
        )
        b2, a2_ = _blocked_and_atten(
            leg2_origins[group_index][sel], leg2_dirs[group_index][sel],
            leg2_len[group_index][sel], edges, opaque, log_tau_half,
        )
        blocked = b1 | b2
        atten = a1_ * a2_

        third = leg3[group_index]
        if third is not None:
            o3, d3, t3 = third
            b3, a3_ = _blocked_and_atten(o3[sel], d3[sel], t3[sel], edges, opaque, log_tau_half)
            blocked = blocked | b3
            atten = atten * a3_

        blocked_flags[out_pos] = blocked
        atten_factors[out_pos] = atten

    valid = ~blocked_flags
    if not valid.any():
        return EchoPaths.empty()

    out = candidates.select(keep).select(valid)
    out.atten = atten_factors[valid]
    order = np.argsort(out.bounce, kind="stable")
    return out.select(order)


def _wall_candidates_nearest(
    origin: np.ndarray, edges: EdgeSet, k: int, exclude: int
) -> np.ndarray:
    """Same merit ranking as :func:`_wall_candidates` but without an
    occlusion test (the second hop of a two-bounce path is checked anyway)."""
    centres = 0.5 * (edges.a + edges.b)
    rng = np.linalg.norm(centres - origin[None, :], axis=1)
    merit = edges.length() / np.maximum(rng, 1e-3)
    merit[exclude] = -1.0

    chosen: list[int] = []
    per_owner: dict[int, int] = {}
    for candidate in np.argsort(-merit):
        if merit[candidate] <= 0.0:
            break
        owner = int(edges.owner[candidate])
        if per_owner.get(owner, 0) >= 4:
            continue
        per_owner[owner] = per_owner.get(owner, 0) + 1
        chosen.append(int(candidate))
        if len(chosen) >= int(k):
            break
    return np.asarray(chosen, dtype=int)


def _push(records, idx, path_len, theta, v_eff, cos_psi, gain, bounce) -> None:
    records["scatterer"].append(idx)
    records["equiv_range"].append(np.asarray(path_len, dtype=np.float64))
    records["theta_rad"].append(np.asarray(theta, dtype=np.float64))
    records["v_eff"].append(np.asarray(v_eff, dtype=np.float64))
    records["cos_psi"].append(np.asarray(cos_psi, dtype=np.float64))
    records["gain"].append(np.asarray(gain, dtype=complex))
    records["bounce"].append(np.full(np.size(path_len), int(bounce), dtype=int))


def _assemble(records, total: int) -> EchoPaths:
    return EchoPaths(
        scatterer=np.concatenate(records["scatterer"]),
        equiv_range=np.concatenate(records["equiv_range"]),
        theta_rad=np.concatenate(records["theta_rad"]),
        v_eff=np.concatenate(records["v_eff"]),
        cos_psi=np.concatenate(records["cos_psi"]),
        gain=np.concatenate(records["gain"]),
        atten=np.ones(total, dtype=np.float64),
        bounce=np.concatenate(records["bounce"]),
    )
