"""Quick performance probe for one radar frame (default IWR1843 config).

Usage::

    python scripts/bench.py [--frames 5] [--segment-len 0.5] [--order 1]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from radar_sim.core import (
    Primitive,
    RadarConfig,
    Scene,
    build_scatterers,
    synthesize_frame,
    trace_paths,
)


def build_scene(width: float = 100.0, height: float = 60.0) -> Scene:
    """A typical demo scene: two buildings, an L-shaped wall, a tree, a bush."""
    scene = Scene.empty(width, height)
    scene.add(Primitive("rect", {"x": 40.0, "y": 34.0, "w": 12.0, "h": 8.0, "rot_deg": 15.0},
                        material_id="metal"))
    scene.add(Primitive("rect", {"x": 70.0, "y": 20.0, "w": 16.0, "h": 6.0, "rot_deg": -10.0},
                        material_id="concrete"))
    scene.add(Primitive("wall", {"points": [[25.0, 5.0], [25.0, 25.0], [55.0, 25.0]],
                                 "thickness": 0.4}, material_id="concrete"))
    scene.add(Primitive("circle", {"x": 58.0, "y": 45.0, "r": 1.5}, material_id="vegetation"))
    scene.add(Primitive("circle", {"x": 30.0, "y": 15.0, "r": 2.0}, material_id="wood"))
    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--segment-len", type=float, default=0.5)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    cfg = RadarConfig.iwr1843(enable_noise=args.noise,
                              max_reflection_order=args.order)
    scene = build_scene()
    scatterers = build_scatterers(scene, segment_len=args.segment_len)
    rng = np.random.default_rng(0)

    print(f"radar  : Nc={cfg.chirps_per_frame} Nv={cfg.virtual_elements} Ns={cfg.samples_per_chirp}"
          f"  frame={cfg.frame_bytes / 1024:.0f} KiB")
    print(f"scene  : {len(scene.primitives)} primitives, {scatterers.count} scatterers"
          f" (segment_len={args.segment_len} m), multipath order {args.order}")

    timings: list[tuple[float, float, int]] = []
    for frame in range(args.frames):
        radar_pos = np.array([12.0 + 0.5 * frame, 42.0])
        t0 = time.perf_counter()
        paths = trace_paths(scene, scatterers, radar_pos, cfg=cfg)
        t1 = time.perf_counter()
        result = synthesize_frame(paths, scatterers, scene, cfg, rng=rng,
                                  max_paths=args.max_paths)
        t2 = time.perf_counter()
        timings.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, paths.count))

    trace_ms = np.mean([t[0] for t in timings])
    synth_ms = np.mean([t[1] for t in timings])
    paths_n = np.mean([t[2] for t in timings])
    print(f"paths  : {paths_n:.0f} traced, {result.paths.count} synthesised"
          f" {result.stats['paths_by_bounce']}")
    print(f"timing : raytrace {trace_ms:.1f} ms | synth {synth_ms:.1f} ms"
          f" | total {trace_ms + synth_ms:.1f} ms   (budget 80 ms @ 12.5 Hz)")
    print(f"verdict: {'OK' if trace_ms + synth_ms < 80.0 else 'TOO SLOW'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
