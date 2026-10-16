"""GUI entry point.

Usage::

    python -m radar_sim.main                # built-in demo scene
    python -m radar_sim.main --scene my.scene.json
    python -m radar_sim.main --empty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.radar_config import RadarConfig
from .core.scene import Scene
from .core.simulator import SimulationSettings
from .core.vehicle import Vehicle, VehicleConfig

__all__ = ["main", "build_window"]


def _demo_data() -> dict:
    scene = Scene.demo()
    data = scene.to_dict()
    data["vehicle"] = {
        "start": [8.0, 42.0, 0.0],
        "length": 4.5, "width": 1.9, "wheelbase": 2.7,
        "max_speed": 15.0, "max_accel": 3.0, "max_steer_deg": 35.0,
        "radar_offset": [1.5, 0.0],
        "mode": "path",
        "path": [[8.0, 42.0], [46.0, 42.0], [46.0, 14.0], [78.0, 14.0]],
        "path_speed": 6.0,
    }
    data["radar"] = dict(RadarConfig().to_dict())
    data["simulation"] = {"seed": 42, "frame_period_s": 0.08, "duration_s": 4.0}
    return data


def build_window(scene_data: dict | None = None, scene_path: Path | None = None):
    """Create the :class:`MainWindow` from a scene dictionary."""
    from .ui.main_window import MainWindow

    data = scene_data if scene_data is not None else _demo_data()
    scene = Scene.from_dict(data)
    radar_cfg = RadarConfig.from_dict(data.get("radar") or {})
    vehicle_block = data.get("vehicle") or {}
    vehicle = Vehicle(config=VehicleConfig.from_dict(vehicle_block),
                      path=vehicle_block.get("path"))
    settings = SimulationSettings.from_dict(data.get("simulation"))
    return MainWindow(scene=scene, radar_cfg=radar_cfg, vehicle=vehicle,
                      settings=settings, scene_path=scene_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="radar-sim",
        description="2D 场景雷达仿真器（图形界面）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", type=Path, help="打开一个 .scene.json")
    group.add_argument("--empty", action="store_true", help="从空场景开始")
    parser.add_argument(
        "--selftest", nargs="?", const="", default=None, metavar="报告路径",
        help="不打开界面，跑一遍完整自检并写出报告（打包后验证用）",
    )
    parser.add_argument("--version", action="store_true", help="显示版本后退出")
    args, qt_args = parser.parse_known_args(argv)

    if args.version:
        from . import __version__
        print(f"radar-sim {__version__}")
        return 0

    if args.selftest is not None:
        from .selftest import run as run_selftest
        target = Path(args.selftest) if args.selftest else None
        return run_selftest(target)

    from PySide6.QtWidgets import QApplication

    application = QApplication([sys.argv[0], *qt_args])

    if args.scene is not None:
        data = json.loads(args.scene.read_text(encoding="utf-8"))
        window = build_window(data, args.scene)
    elif args.empty:
        empty = Scene.empty(100.0, 60.0).to_dict()
        empty["vehicle"] = {"start": [10.0, 30.0, 0.0],
                            "path": [[10.0, 30.0], [60.0, 30.0]]}
        window = build_window(empty)
    else:
        window = build_window()

    window.show()
    window.canvas.fit_to_scene()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
