"""Headless command line: batch dataset generation and scene inspection.

Examples::

    python -m radar_sim.cli --demo --out dataset/demo --frames 20
    python -m radar_sim.cli --scene examples/scenes/demo.scene.json --out run01 \\
        --duration 4 --order 2 --layout ti_dca1000
    python -m radar_sim.cli --inspect --scene examples/scenes/demo.scene.json
    python -m radar_sim.cli --list-materials
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from radar_sim import __version__
from radar_sim.core.materials import PRESETS
from radar_sim.core.radar_config import MIMO_MODES, RadarConfig
from radar_sim.core.scene import Scene
from radar_sim.core.simulator import SimulationSettings
from radar_sim.core.vehicle import Vehicle, VehicleConfig
from radar_sim.core.simulator import Simulator

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar-sim-cli",
        description="2D scene radar simulator -- headless dataset generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--scene", type=Path, help="path to a .scene.json file")
    src.add_argument("--demo", action="store_true", help="use the built-in demo scene")

    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: dataset/run; "
                             "figures/ when using --process)")
    parser.add_argument("--frames", type=int, default=None,
                        help="number of radar frames (overrides --duration)")
    parser.add_argument("--duration", type=float, default=None, help="seconds to simulate")
    parser.add_argument("--frame-period", type=float, default=None,
                        help="seconds between radar frames (default 0.08 = 12.5 Hz)")

    parser.add_argument("--layout", choices=("ti_dca1000", "interleaved_iq"),
                        default="ti_dca1000", help="signal.bin byte layout")
    parser.add_argument("--scale-mode", choices=("frame0", "per_frame", "fixed"),
                        default="frame0", help="IQ scaling strategy")
    parser.add_argument("--iq-scale", type=float, default=None,
                        help="required when --scale-mode=fixed")

    parser.add_argument("--order", type=int, choices=(0, 1, 2), default=None,
                        help="max specular reflection order (multipath)")
    parser.add_argument("--segment-len", type=float, default=None,
                        help="scene discretisation step in metres")
    parser.add_argument("--max-paths", type=int, default=None,
                        help="cap on synthesised echo paths per frame")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    parser.add_argument("--chirps", type=int, default=None, help="chirps per frame")
    parser.add_argument("--samples", type=int, default=None, help="samples per chirp")
    parser.add_argument("--tx", type=int, default=None, help="tx elements")
    parser.add_argument("--rx", type=int, default=None, help="rx elements")
    parser.add_argument("--mimo", choices=MIMO_MODES, default=None, help="MIMO mode")
    parser.add_argument("--no-noise", action="store_true", help="disable thermal noise")
    parser.add_argument("--no-truth", action="store_true",
                        help="skip pointcloud_gt.csv")

    parser.add_argument("--speed", type=float, default=None,
                        help="path-following speed in m/s")
    parser.add_argument("--path", type=str, default=None,
                        help="semicolon separated waypoints, e.g. '5,10;40,10;40,40'")

    parser.add_argument("--inspect", action="store_true",
                        help="print the scene/radar summary and exit")
    parser.add_argument("--list-materials", action="store_true",
                        help="print the material preset table and exit")
    parser.add_argument("--selftest", nargs="?", const="", default=None,
                        metavar="REPORT",
                        help="run the bundled self check and exit "
                             "(used to verify a frozen build)")

    # Processing modes.  These run the shipped templates *inside* this
    # executable, which is what makes the .exe self-sufficient: the templates
    # need numpy and matplotlib, and both are already in the bundle.  Without
    # this, an exe-only user could generate data but never look at it.
    parser.add_argument("--process", type=Path, default=None, metavar="DATASET",
                        help="process an exported dataset into figures and exit "
                             "(runs templates/python/process_demo.py)")
    parser.add_argument("--compare", type=Path, default=None, metavar="DATASET",
                        help="compare detections with the ground truth and exit "
                             "(runs templates/python/compare_gt.py)")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame index for --process")

    parser.add_argument("--quiet", action="store_true", help="no per-frame progress")
    parser.add_argument("--version", action="version", version=f"radar-sim {__version__}")
    return parser


# --------------------------------------------------------------------------- #
# running the shipped templates from inside a frozen executable
# --------------------------------------------------------------------------- #
def _run_template(name: str, dataset: Path, out_dir: Path,
                  extra: list[str] | None = None) -> int:
    """Execute ``templates/python/<name>.py`` as ``__main__``.

    Reusing the template files instead of re-implementing the processing keeps
    a single source of truth: whatever the user gets when they copy the
    templates into their own project is exactly what this runs.
    """
    from .selftest import find_templates

    folder = find_templates()
    if folder is None:
        print("error: 找不到 templates/python 目录。\n"
              "      文件夹版请确认 templates/ 与 exe 在同一目录下；\n"
              "      单文件版请用 --selftest 检查包是否完整。", file=sys.stderr)
        return 2

    script = folder / f"{name}.py"
    if not script.exists():
        print(f"error: 缺少模板 {script}", file=sys.stderr)
        return 2

    # cfar.py / process_demo.py use absolute imports (``from radar_io import``)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

    # The templates write straight into --out and do not create it.
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: 无法创建输出目录 {out_dir}: {exc}", file=sys.stderr)
        return 2

    saved_argv = sys.argv
    sys.argv = [str(script), "--data", str(dataset), "--out", str(out_dir)]
    sys.argv += list(extra or [])
    try:
        import runpy
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:                    # templates call sys.exit(main())
        return int(exc.code) if exc.code else 0
    finally:
        sys.argv = saved_argv
    return 0


def _parse_path(spec: str) -> list[list[float]]:
    points: list[list[float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"bad waypoint {chunk!r}, expected 'x,y'")
        points.append([float(parts[0]), float(parts[1])])
    if len(points) < 2:
        raise ValueError("--path needs at least two waypoints")
    return points


def _print_materials() -> None:
    print(f"{'id':<12} {'name':<10} {'rho':>5} {'eta':>5} {'tau':>5} "
          f"{'abs_dB':>7} {'sigma':>6}")
    print("-" * 56)
    for m in PRESETS:
        print(f"{m.id:<12} {m.name:<10} {m.rho:>5.2f} {m.eta:>5.2f} {m.tau:>5.2f} "
              f"{m.absorption_db:>7.1f} {m.sigma_psi_deg:>6.1f}")


def _demo_data() -> dict:
    """Built-in demo: open view for the ego vehicle, L-wall on its right."""
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
    data["simulation"] = {
        "seed": 42, "frame_period_s": 0.08, "duration_s": 4.0,
        "enable_noise": True, "enable_multipath": True,
    }
    return data


def _load_data(args: argparse.Namespace) -> tuple[Scene, dict]:
    if args.scene is not None:
        data = json.loads(args.scene.read_text(encoding="utf-8"))
        return Scene.from_dict(data), data
    if args.demo:
        data = _demo_data()
        return Scene.from_dict(data), data
    raise SystemExit("specify --scene FILE or --demo (see --help)")


def _apply_overrides(cfg: RadarConfig, args: argparse.Namespace) -> RadarConfig:
    updates: dict[str, object] = {}
    if args.chirps is not None:
        updates["chirps_per_frame"] = args.chirps
    if args.samples is not None:
        updates["samples_per_chirp"] = args.samples
    if args.tx is not None:
        updates["tx_elements"] = args.tx
    if args.rx is not None:
        updates["rx_elements"] = args.rx
    if args.mimo is not None:
        updates["mimo_mode"] = args.mimo
    if args.order is not None:
        updates["max_reflection_order"] = args.order
    if args.no_noise:
        updates["enable_noise"] = False
    if not updates:
        return cfg
    cfg = cfg.with_updates(**updates)
    if cfg.mimo_mode == "tdm" and cfg.chirps_per_frame % cfg.tx_elements != 0:
        cfg = cfg.fixed_for_tdm()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest is not None:
        from .selftest import run as run_selftest
        target = Path(args.selftest) if args.selftest else None
        return run_selftest(target)

    if args.process is not None:
        extra = ["--frame", str(args.frame)] if args.frame is not None else []
        return _run_template("process_demo", args.process,
                             args.out or Path("figures"), extra)

    if args.compare is not None:
        return _run_template("compare_gt", args.compare, args.out or Path("."))

    if args.list_materials:
        _print_materials()
        return 0

    if args.out is None:
        args.out = Path("dataset/run")

    scene, data = _load_data(args)

    radar_cfg = RadarConfig.from_dict(data.get("radar") or {})
    radar_cfg = _apply_overrides(radar_cfg, args)
    for warning in radar_cfg.validate():
        print(f"warning: {warning}", file=sys.stderr)

    settings = SimulationSettings.from_dict(data.get("simulation"))
    if args.frame_period is not None:
        settings.frame_period_s = args.frame_period
    if args.frames is not None:
        settings.duration_s = args.frames * settings.frame_period_s
    elif args.duration is not None:
        settings.duration_s = args.duration
    if args.seed is not None:
        settings.seed = args.seed
    if args.segment_len is not None:
        settings.segment_len_m = args.segment_len
    if args.max_paths is not None:
        settings.max_paths = args.max_paths
    settings.__post_init__()

    vehicle_block = data.get("vehicle") or {}
    vehicle = Vehicle(config=VehicleConfig.from_dict(vehicle_block))
    if args.speed is not None:
        vehicle.config.path_speed_mps = args.speed

    path: list[list[float]] | None = None
    if args.path:
        path = _parse_path(args.path)
    elif vehicle_block.get("path"):
        path = [list(p) for p in vehicle_block["path"]]
    if path is None:
        x0, y0, _yaw = vehicle.config.start
        path = [[x0, y0], [x0 + 40.0, y0]]
    vehicle.set_path(path)

    simulator = Simulator(scene, radar_cfg, vehicle, settings)
    summary = simulator.summary()

    print(f"=== 2D Radar Scene Simulator {__version__} ===")
    print(f"scene      : {'demo' if args.demo else args.scene} "
          f"({summary['scene_primitives']} primitives, {summary['scatterers']} scatterers)")
    print(f"radar      : Nc={radar_cfg.chirps_per_frame} Nv={radar_cfg.virtual_elements} "
          f"Ns={radar_cfg.samples_per_chirp}  mimo={radar_cfg.mimo_mode} "
          f"order={radar_cfg.max_reflection_order}")
    print(f"derived    : dR={summary['range_resolution_m']} m  "
          f"Rmax={summary['max_range_m']} m  "
          f"vmax=+-{summary['max_velocity_mps']} m/s  "
          f"dv={summary['velocity_resolution_mps']} m/s  "
          f"dAz={summary['angular_resolution_deg']} deg")
    print(f"vehicle    : {vehicle.mode} mode, {vehicle.config.path_speed_mps} m/s, "
          f"{len(path)} waypoints")
    print(f"frames     : {summary['frames']} x {settings.frame_period_s * 1e3:.0f} ms "
          f"= {summary['duration_s']} s")
    print(f"data       : {summary['frame_bytes'] / 1024:.0f} KiB/frame, "
          f"{summary['dataset_bytes'] / 1024 / 1024:.1f} MiB total, "
          f"layout={args.layout}")

    if args.inspect:
        print("\n--inspect: nothing written.")
        return 0

    frames_done = [0]
    started = time.perf_counter()

    def on_frame(frame) -> None:                                # noqa: ANN001
        frames_done[0] = frame.index + 1
        if not args.quiet and (frame.index + 1) % max(1, summary["frames"] // 10) == 0:
            print(f"  frame {frame.index + 1}/{summary['frames']}  "
                  f"paths={frame.result.paths.count}  "
                  f"raytrace={frame.timings_ms['raytrace']:.1f} ms  "
                  f"synth={frame.timings_ms['synthesis']:.1f} ms")

    print(f"\nwriting to {args.out} ...")
    meta = simulator.export(
        args.out,
        layout=args.layout,
        iq_scale_mode=args.scale_mode,
        iq_scale=args.iq_scale,
        include_truth=not args.no_truth,
        on_frame=on_frame,
    )
    wall = time.perf_counter() - started

    signal = meta.get("signal", {})
    perf = meta.get("performance", {})
    print(f"done in {wall:.2f} s for {frames_done[0]} frames "
          f"({wall / max(frames_done[0], 1) * 1e3:.1f} ms/frame)")
    print(f"mean raytrace {perf.get('raytrace_ms_mean', 0):.1f} ms | "
          f"mean synthesis {perf.get('synthesis_ms_mean', 0):.1f} ms")
    if signal.get("clip_ratio", 0.0) > 1e-3:
        print(f"warning: {signal['clip_ratio'] * 100:.3f}% of IQ samples clipped; "
              f"consider iq_scale_mode=fixed with a larger value")

    print("outputs    :")
    for name in ("signal.bin", "meta.json", "pointcloud_gt.csv", "README.txt"):
        path_out = Path(args.out) / name
        if path_out.exists():
            print(f"  {name:<20} {path_out.stat().st_size / 1024:>10.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
