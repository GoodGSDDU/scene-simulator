"""Scene graph: primitives, materials, edge extraction and JSON I/O.

The scene is the *authored* description (what the user draws).  Ray tracing
does not use it directly -- :mod:`radar_sim.core.scatterers` converts it into a
point-scatterer set first.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import geometry as geo
from .materials import Material, preset_materials

KINDS = ("circle", "rect", "polygon", "wall", "point")

__all__ = ["KINDS", "Primitive", "EdgeSet", "Scene"]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
@dataclass
class Primitive:
    """A drawable / scatterable scene element."""

    kind: str
    params: dict[str, Any]
    id: str = ""
    name: str = ""
    material_id: str = "concrete"
    layer: int = 0
    visible: bool = True
    locked: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown primitive kind {self.kind!r}, expected one of {KINDS}")
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if not self.name:
            self.name = f"{self.kind}-{self.id}"

    # --- geometry ---------------------------------------------------------
    def outline(self) -> np.ndarray | None:
        """Closed polygon outline in world coordinates, or ``None`` for points."""
        p = self.params
        if self.kind == "circle":
            return geo.circle_polygon(p["x"], p["y"], p["r"], int(p.get("segments", 64)))
        if self.kind == "rect":
            return geo.rect_polygon(
                p["x"], p["y"], p["w"], p["h"], np.deg2rad(p.get("rot_deg", 0.0))
            )
        if self.kind == "polygon":
            return np.asarray(p["points"], dtype=np.float64)
        if self.kind == "wall":
            return geo.wall_polygon(
                np.asarray(p["points"], dtype=np.float64), float(p.get("thickness", 0.3))
            )
        return None                                             # point target

    def centroid(self) -> tuple[float, float]:
        if self.kind == "point":
            return float(self.params["x"]), float(self.params["y"])
        poly = self.outline()
        assert poly is not None
        c = poly.mean(axis=0)
        return float(c[0]), float(c[1])

    # --- editing ----------------------------------------------------------
    def translate(self, dx: float, dy: float) -> None:
        p = self.params
        if self.kind in ("circle", "rect", "point"):
            p["x"] += float(dx)
            p["y"] += float(dy)
        else:
            pts = np.asarray(p["points"], dtype=np.float64) + np.array([dx, dy])
            p["points"] = pts.tolist()

    def rotate_by(self, dtheta_rad: float, origin: tuple[float, float] | None = None) -> None:
        p = self.params
        if self.kind in ("circle", "point"):
            return
        if self.kind == "rect":
            p["rot_deg"] = float((p.get("rot_deg", 0.0) + np.rad2deg(dtheta_rad)) % 360.0)
            if origin is not None:
                cx, cy = self.centroid()
                rot_origin = geo.rotate(np.array([[cx, cy]]), dtheta_rad, origin)[0]
                self.translate(float(rot_origin[0] - cx), float(rot_origin[1] - cy))
            return
        origin = origin or self.centroid()
        pts = geo.rotate(np.asarray(p["points"], dtype=np.float64), dtheta_rad, origin)
        p["points"] = pts.tolist()

    def scale_by(self, factor: float, origin: tuple[float, float] | None = None) -> None:
        p = self.params
        factor = max(float(factor), 1e-3)
        if self.kind == "circle":
            p["r"] = max(p["r"] * factor, 0.05)
            return
        if self.kind == "rect":
            p["w"] = max(p["w"] * factor, 0.05)
            p["h"] = max(p["h"] * factor, 0.05)
            return
        if self.kind == "point":
            return
        origin = origin or self.centroid()
        o = np.asarray(origin, dtype=np.float64)
        pts = o + (np.asarray(p["points"], dtype=np.float64) - o) * factor
        p["points"] = pts.tolist()

    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of this primitive."""
        if self.kind == "point":
            x, y = self.params["x"], self.params["y"]
            return (x, y, x, y)
        poly = self.outline()
        assert poly is not None
        return (
            float(poly[:, 0].min()),
            float(poly[:, 1].min()),
            float(poly[:, 0].max()),
            float(poly[:, 1].max()),
        )

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "material_id": self.material_id,
            "layer": self.layer,
            "visible": self.visible,
            "locked": self.locked,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Primitive":
        payload = dict(data)
        payload["params"] = dict(payload.get("params", {}))
        valid = {f for f in cls.__dataclass_fields__}           # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in valid})


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
@dataclass
class EdgeSet:
    """Flat list of reflective segments with outward normals."""

    a: np.ndarray               # (M, 2)
    b: np.ndarray               # (M, 2)
    normal: np.ndarray          # (M, 2) unit, outward
    material_index: np.ndarray  # (M,) index into Scene.material_ids()
    owner: np.ndarray           # (M,) index into Scene.primitives

    def __len__(self) -> int:
        return int(self.a.shape[0])

    @property
    def count(self) -> int:
        return len(self)

    def length(self) -> np.ndarray:
        return np.linalg.norm(self.b - self.a, axis=1)

    @classmethod
    def empty(cls) -> "EdgeSet":
        z = np.zeros((0, 2), dtype=np.float64)
        return cls(z, z, z.copy(), np.zeros(0, dtype=int), np.zeros(0, dtype=int))


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    bounds: tuple[float, float] = (100.0, 60.0)
    materials: dict[str, Material] = field(default_factory=preset_materials)
    primitives: list[Primitive] = field(default_factory=list)

    # --- materials --------------------------------------------------------
    def material_ids(self) -> list[str]:
        """Ordered material ids; the position is the index used everywhere."""
        return list(self.materials.keys())

    def material_index(self, material_id: str) -> int:
        try:
            return self.material_ids().index(material_id)
        except ValueError as exc:                                # pragma: no cover
            raise KeyError(f"unknown material {material_id!r}") from exc

    def material(self, material_id: str) -> Material:
        if material_id not in self.materials:
            raise KeyError(f"unknown material {material_id!r}")
        return self.materials[material_id]

    def material_of(self, prim: Primitive) -> Material:
        return self.material(prim.material_id)

    def ensure_material(self, material: Material) -> str:
        """Register ``material`` (overwriting) and return its id."""
        self.materials[material.id] = material
        return material.id

    # --- primitives -------------------------------------------------------
    def add(self, prim: Primitive) -> Primitive:
        if any(p.id == prim.id for p in self.primitives):
            prim.id = uuid.uuid4().hex[:8]
        self.primitives.append(prim)
        return prim

    def remove(self, prim_id: str) -> bool:
        before = len(self.primitives)
        self.primitives = [p for p in self.primitives if p.id != prim_id]
        return len(self.primitives) != before

    def by_id(self, prim_id: str) -> Primitive | None:
        for p in self.primitives:
            if p.id == prim_id:
                return p
        return None

    def visible_primitives(self) -> list[Primitive]:
        return [p for p in self.primitives if p.visible]

    def ordered(self) -> list[Primitive]:
        """Draw / trace order (by layer, then insertion order)."""
        return sorted(self.primitives, key=lambda p: (p.layer,))

    def bounds_all(self) -> tuple[float, float, float, float]:
        if not self.primitives:
            return (0.0, 0.0, float(self.bounds[0]), float(self.bounds[1]))
        boxes = np.array([p.bounds() for p in self.primitives], dtype=np.float64)
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].min()),
            float(boxes[:, 2].max()),
            float(boxes[:, 3].max()),
        )

    # --- edges ------------------------------------------------------------
    def build_edges(self) -> EdgeSet:
        """Flatten every visible non-point primitive into segments."""
        material_ids = self.material_ids()
        index_of = {mid: i for i, mid in enumerate(material_ids)}

        a_list: list[np.ndarray] = []
        b_list: list[np.ndarray] = []
        n_list: list[np.ndarray] = []
        m_list: list[int] = []
        o_list: list[int] = []

        for owner, prim in enumerate(self.ordered()):
            if not prim.visible or prim.kind == "point":
                continue
            poly = prim.outline()
            if poly is None or poly.shape[0] < 3:
                continue
            a, b = geo.polygon_edges(poly)
            n = geo.edge_normals_outward(poly)
            a_list.append(a)
            b_list.append(b)
            n_list.append(n)
            m_list.extend([index_of.get(prim.material_id, 0)] * a.shape[0])
            o_list.extend([owner] * a.shape[0])

        if not a_list:
            return EdgeSet.empty()

        a = np.vstack(a_list)
        b = np.vstack(b_list)
        n = np.vstack(n_list)
        keep = np.linalg.norm(b - a, axis=1) > 1e-9
        return EdgeSet(
            a[keep],
            b[keep],
            n[keep],
            np.asarray(m_list, dtype=int)[keep],
            np.asarray(o_list, dtype=int)[keep],
        )

    def opaque_mask(self, edges: EdgeSet) -> np.ndarray:
        """Boolean mask of edges that block the line of sight."""
        return np.array(
            [self.materials[self.material_ids()[int(i)]].opaque for i in edges.material_index],
            dtype=bool,
        )

    def transmission_loss(self, edges: EdgeSet, index: int, distance_m: float) -> float:
        mat = self.materials[self.material_ids()[int(edges.material_index[index])]]
        return mat.transmission_loss_amplitude(distance_m)

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "units": {"length": "m", "angle": "deg"},
            "bounds": {"width": float(self.bounds[0]), "height": float(self.bounds[1])},
            "materials": [m.to_dict() for m in self.materials.values()],
            "primitives": [p.to_dict() for p in self.primitives],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        bounds = data.get("bounds", {})
        materials = {
            m["id"]: Material.from_dict(m) for m in data.get("materials", [])
        } or preset_materials()
        prims = [Primitive.from_dict(p) for p in data.get("primitives", [])]
        return cls(
            bounds=(float(bounds.get("width", 100.0)), float(bounds.get("height", 60.0))),
            materials=materials,
            primitives=prims,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # --- factories --------------------------------------------------------
    @classmethod
    def empty(cls, width: float = 100.0, height: float = 60.0) -> "Scene":
        return cls(bounds=(width, height))

    @classmethod
    def demo(cls) -> "Scene":
        """Small demo: two buildings, an L-shaped wall, a tree, a point target.

        Everything sits in the lower-right half so a vehicle starting near
        ``(8, 42)`` looking along ``+X`` has a clear view of the whole scene.
        """
        scene = cls.empty(100.0, 60.0)
        scene.add(Primitive("rect", {"x": 40.0, "y": 30.0, "w": 12.0, "h": 8.0, "rot_deg": 15.0},
                            name="建筑A", material_id="metal", layer=1))
        scene.add(Primitive("rect", {"x": 72.0, "y": 18.0, "w": 16.0, "h": 6.0, "rot_deg": -10.0},
                            name="建筑B", material_id="concrete", layer=1))
        scene.add(Primitive("wall", {"points": [[25.0, 4.0], [25.0, 24.0], [56.0, 24.0]],
                                     "thickness": 0.4},
                            name="围墙", material_id="concrete", layer=0))
        scene.add(Primitive("circle", {"x": 60.0, "y": 46.0, "r": 1.5}, name="树",
                            material_id="vegetation", layer=2))
        scene.add(Primitive("circle", {"x": 34.0, "y": 12.0, "r": 2.0}, name="灌木",
                            material_id="wood", layer=2))
        scene.add(Primitive("point", {"x": 20.0, "y": 30.0}, name="标定点",
                            material_id="metal", layer=2))
        return scene
