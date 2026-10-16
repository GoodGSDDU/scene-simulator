"""Convert authored primitives into the point-scatterer set used by the echo model.

A polygon outline is split into pieces of at most ``segment_len`` metres; each
piece centre becomes one scatterer carrying the outward normal of its edge.
The piece length also serves as the effective 2D scattering aperture
(``A_eff = seg_len * height_m``) in the radar-equation amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .scene import Scene

__all__ = ["ScattererSet", "build_scatterers"]

KIND_BOUNDARY = 0
KIND_POINT = 2


@dataclass
class ScattererSet:
    position: np.ndarray        # (N, 2)
    normal: np.ndarray          # (N, 2) unit outward; (0,0) == isotropic point
    seg_len: np.ndarray         # (N,) effective 2D aperture
    material_index: np.ndarray  # (N,) index into Scene.material_ids()
    owner: np.ndarray           # (N,) index into Scene.ordered()
    kind: np.ndarray            # (N,) KIND_* code

    def __len__(self) -> int:
        return int(self.position.shape[0])

    @property
    def count(self) -> int:
        return len(self)

    def is_isotropic(self) -> np.ndarray:
        return np.linalg.norm(self.normal, axis=1) < 1e-9

    @classmethod
    def empty(cls) -> "ScattererSet":
        z = np.zeros((0, 2), dtype=np.float64)
        return cls(z, z.copy(), np.zeros(0), np.zeros(0, dtype=int),
                   np.zeros(0, dtype=int), np.zeros(0, dtype=int))

    def subset(self, mask: np.ndarray) -> "ScattererSet":
        return ScattererSet(
            self.position[mask], self.normal[mask], self.seg_len[mask],
            self.material_index[mask], self.owner[mask], self.kind[mask],
        )


def build_scatterers(
    scene: Scene,
    *,
    segment_len: float = 0.5,
    max_scatterers: int = 5000,
    point_height_m: float = 1.0,
    include_interior: bool = False,
) -> ScattererSet:
    """Discretise ``scene`` into scatterers.

    ``segment_len`` is automatically relaxed when the estimate exceeds
    ``max_scatterers``.
    """
    if segment_len <= 0.0:
        raise ValueError("segment_len must be positive")
    segment_len = float(segment_len)
    edge_sets: list[tuple] = []
    total_edge_len = 0.0
    n_edges = 0
    n_points = 0

    for owner, prim in enumerate(scene.ordered()):
        if not prim.visible:
            continue
        if prim.kind == "point":
            n_points += 1
            continue
        poly = prim.outline()
        if poly is None or poly.shape[0] < 3:
            continue
        a, b = geo.polygon_edges(poly)
        normal = geo.edge_normals_outward(poly)
        lengths = np.linalg.norm(b - a, axis=1)
        edge_sets.append((owner, a, b, normal, lengths))
        total_edge_len += float(lengths.sum())
        n_edges += int(lengths.size)

    # --- relax segment_len so the scatterer budget can be met -------------
    if n_edges:
        estimate = total_edge_len / segment_len + n_edges
        budget = max(int(max_scatterers) - n_points - max(2 * n_edges, 0), 1)
        if estimate > budget:
            segment_len = max(total_edge_len / max(budget - n_edges, 1), segment_len)

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    seg_lens: list[np.ndarray] = []
    mat_idx: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    material_ids = scene.material_ids()

    for owner, a, b, normal, lengths in edge_sets:
        prim = scene.ordered()[owner]
        mi = material_ids.index(prim.material_id) if prim.material_id in material_ids else 0
        n_pieces = np.maximum(1, np.ceil(lengths / segment_len).astype(int))
        for e in range(a.shape[0]):
            centres, piece = geo.subdivide_segment(a[e], b[e], segment_len)
            k = centres.shape[0]
            positions.append(centres)
            normals.append(np.repeat(normal[e][None, :], k, axis=0))
            seg_lens.append(np.full(k, piece))
            mat_idx.append(np.full(k, mi, dtype=int))
            owners.append(np.full(k, owner, dtype=int))
            kinds.append(np.full(k, KIND_BOUNDARY, dtype=int))
        del n_pieces

    if n_points:
        for owner, prim in enumerate(scene.ordered()):
            if not prim.visible or prim.kind != "point":
                continue
            mi = material_ids.index(prim.material_id) if prim.material_id in material_ids else 0
            positions.append(np.array([[prim.params["x"], prim.params["y"]]], dtype=np.float64))
            normals.append(np.zeros((1, 2), dtype=np.float64))
            seg_lens.append(np.array([float(point_height_m)]))
            mat_idx.append(np.array([mi], dtype=int))
            owners.append(np.array([owner], dtype=int))
            kinds.append(np.array([KIND_POINT], dtype=int))

    if not positions:
        return ScattererSet.empty()

    result = ScattererSet(
        position=np.vstack(positions),
        normal=np.vstack(normals),
        seg_len=np.concatenate(seg_lens),
        material_index=np.concatenate(mat_idx),
        owner=np.concatenate(owners),
        kind=np.concatenate(kinds),
    )
    if len(result) > max_scatterers:
        keep = np.argsort(-result.seg_len)[: int(max_scatterers)]
        result = result.subset(np.sort(keep))
    return result
