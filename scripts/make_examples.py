"""Regenerate the example scene files under ``examples/scenes``.

Usage::

    python scripts/make_examples.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_sim.core.radar_config import RadarConfig  # noqa: E402
from radar_sim.core.scene import Primitive, Scene  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "examples" / "scenes"

VEHICLE_DRIVE_THROUGH = {
    "start": [8.0, 42.0, 0.0],
    "length": 4.5, "width": 1.9, "wheelbase": 2.7,
    "max_speed": 15.0, "max_accel": 3.0, "max_steer_deg": 35.0,
    "radar_offset": [1.5, 0.0],
    "mode": "path",
    "path": [[8.0, 42.0], [46.0, 42.0], [46.0, 14.0], [78.0, 14.0]],
    "path_speed": 6.0,
}

SIM_4S = {"seed": 42, "frame_period_s": 0.08, "duration_s": 4.0,
          "segment_len_m": 0.5, "max_paths": 256}


def _finish(scene: Scene, vehicle: dict, simulation: dict | None = None) -> dict:
    data = scene.to_dict()
    data["vehicle"] = vehicle
    data["radar"] = dict(RadarConfig().to_dict())
    data["simulation"] = dict(simulation or SIM_4S)
    return data


def build_demo() -> dict:
    """Open-view demo scene shipped with the simulator."""
    return _finish(Scene.demo(), VEHICLE_DRIVE_THROUGH)


def build_corridor() -> dict:
    """A compact corridor: the vehicle drives *between* two reflective walls.

    Kept deliberately short so single- **and** double-bounce ghosts stay inside
    the 50 m unambiguous window.  Use ``--order 1`` (default) for single-bounce
    and ``--order 2`` for double-bounce; ghost ranges come out longer than the
    direct range, which is the signature to look for.
    """
    scene = Scene.empty(40.0, 60.0)
    scene.add(Primitive("wall", {"points": [[8.0, 20.0], [32.0, 20.0]], "thickness": 0.4},
                        name="下侧墙", material_id="concrete"))
    scene.add(Primitive("wall", {"points": [[8.0, 40.0], [32.0, 40.0]], "thickness": 0.4},
                        name="上侧墙", material_id="metal"))
    scene.add(Primitive("wall", {"points": [[32.0, 20.0], [32.0, 40.0]], "thickness": 0.4},
                        name="尽头墙", material_id="metal"))
    scene.add(Primitive("point", {"x": 24.0, "y": 32.0}, name="金属目标",
                        material_id="metal"))
    scene.add(Primitive("circle", {"x": 18.0, "y": 26.0, "r": 1.2}, name="植被",
                        material_id="vegetation"))
    vehicle = {
        "start": [6.0, 30.0, 0.0],
        "mode": "path",
        "path": [[6.0, 30.0], [28.0, 30.0]],
        "path_speed": 4.0,
    }
    return _finish(scene, vehicle)


def build_occlusion() -> dict:
    """A metal blocker right in front of the vehicle, hiding a target behind it.

    Great for verifying the occlusion logic: the target must not appear.
    """
    scene = Scene.empty(60.0, 40.0)
    scene.add(Primitive("rect", {"x": 20.0, "y": 20.0, "w": 1.0, "h": 10.0},
                        name="遮挡板", material_id="metal"))
    scene.add(Primitive("point", {"x": 34.0, "y": 20.0}, name="被遮挡目标",
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 34.0, "y": 32.0}, name="侧方可见目标",
                        material_id="metal"))
    vehicle = {"start": [6.0, 20.0, 0.0], "mode": "path",
               "path": [[6.0, 20.0], [50.0, 20.0]], "path_speed": 4.0}
    return _finish(scene, vehicle)


def build_material_showcase() -> dict:
    """Same geometry, seven materials: a quick reflectivity comparison chart."""
    scene = Scene.empty(90.0, 60.0)
    materials = ("metal", "concrete", "glass", "wood", "human", "vegetation", "absorber")
    for index, material in enumerate(materials):
        scene.add(Primitive(
            "rect",
            {"x": 30.0, "y": 8.0 + 7.0 * index, "w": 1.0, "h": 5.0},
            name=material, material_id=material,
        ))
    vehicle = {"start": [6.0, 30.0, 0.0], "mode": "manual",
               "path": None, "path_speed": 0.0}
    return _finish(scene, vehicle, {"seed": 7, "frame_period_s": 0.08,
                                    "duration_s": 0.8})


BUILDERS = {
    "demo.scene.json": build_demo,
    "corridor.scene.json": build_corridor,
    "occlusion.scene.json": build_occlusion,
    "material_showcase.scene.json": build_material_showcase,
}


def main(argv: list[str] | None = None) -> int:
    out_dir = Path(argv[0]) if argv else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        data = builder()
        target = out_dir / name
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {target}  ({len(data['primitives'])} primitives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
