"""Simulation loop: drive the vehicle, trace, synthesise and stream to disk.

The loop advances the vehicle physics in small substeps, then produces exactly
one radar frame per ``frame_period_s``.  Frames can be consumed live (GUI) or
streamed straight into a :class:`~radar_sim.core.io_export.DatasetWriter`
(headless batch runs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .io_export import DatasetWriter
from .radar import FrameResult, synthesize_frame
from .radar_config import RadarConfig
from .raytrace import trace_paths
from .scene import Scene
from .scatterers import ScattererSet, build_scatterers
from .vehicle import Vehicle, VehicleConfig

__all__ = ["SimulationSettings", "FrameOutput", "Simulator"]

_MAX_PHYSICS_DT_S = 0.01


@dataclass
class SimulationSettings:
    frame_period_s: float = 0.08
    duration_s: float = 4.0
    seed: int = 42
    max_paths: int = 256
    dynamic_range_db: float = 60.0
    height_m: float = 1.0
    segment_len_m: float = 0.5

    def __post_init__(self) -> None:
        if self.frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be positive")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

    @property
    def num_frames(self) -> int:
        return max(1, int(round(self.duration_s / self.frame_period_s)))

    def substeps(self) -> int:
        return max(1, int(np.ceil(self.frame_period_s / _MAX_PHYSICS_DT_S)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_period_s": self.frame_period_s,
            "duration_s": self.duration_s,
            "num_frames": self.num_frames,
            "seed": self.seed,
            "max_paths": self.max_paths,
            "dynamic_range_db": self.dynamic_range_db,
            "height_m": self.height_m,
            "segment_len_m": self.segment_len_m,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SimulationSettings":
        if not data:
            return cls()
        payload = dict(data)
        if "frame_period_s" not in payload and "fps_radar" in payload:
            payload["frame_period_s"] = 1.0 / float(payload["fps_radar"])
        payload.pop("num_frames", None)
        valid = {f for f in cls.__dataclass_fields__}            # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in valid})


@dataclass
class FrameOutput:
    index: int
    t_s: float
    pose: dict[str, Any]
    result: FrameResult
    radar_pos: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(2))
    timings_ms: dict[str, float] = field(default_factory=dict)


class Simulator:
    """Owns the scene, the radar config, the vehicle and the frame clock."""

    def __init__(
        self,
        scene: Scene,
        radar_cfg: RadarConfig,
        vehicle: Vehicle | None = None,
        settings: SimulationSettings | None = None,
        *,
        scatterers: ScattererSet | None = None,
        segment_len_m: float | None = None,
    ) -> None:
        radar_cfg.validate()
        self.scene = scene
        self.cfg = radar_cfg
        self.vehicle = vehicle or Vehicle()
        self.settings = settings or SimulationSettings()
        self.scatterers = scatterers or build_scatterers(
            scene, segment_len=segment_len_m or self.settings.segment_len_m
        )
        self.rng = np.random.default_rng(self.settings.seed)
        self.frame_index = 0
        self.elapsed_s = 0.0
        self.poses: list[dict[str, Any]] = []
        self.last: FrameOutput | None = None
        self._writer: DatasetWriter | None = None
        self._include_truth = True
        self._trace_ms: list[float] = []
        self._synth_ms: list[float] = []
        self._physics_accum_s = 0.0

    # ------------------------------------------------------------------ #
    @property
    def num_frames(self) -> int:
        return self.settings.num_frames

    @property
    def finished(self) -> bool:
        return self.frame_index >= self.num_frames

    def reset(self, *, keep_rng: bool = False) -> None:
        self.vehicle.reset()
        self.frame_index = 0
        self.elapsed_s = 0.0
        self.poses.clear()
        self.last = None
        self._trace_ms.clear()
        self._synth_ms.clear()
        self._physics_accum_s = 0.0
        if not keep_rng:
            self.rng = np.random.default_rng(self.settings.seed)

    def rebuild_scatterers(self, segment_len_m: float | None = None) -> int:
        """Recall after the scene has been edited."""
        self.scatterers = build_scatterers(
            self.scene, segment_len=segment_len_m or self.settings.segment_len_m
        )
        return self.scatterers.count

    # ------------------------------------------------------------------ #
    # writer plumbing
    # ------------------------------------------------------------------ #
    def attach_writer(self, writer: DatasetWriter, *, include_truth: bool = True) -> None:
        self._writer = writer
        self._include_truth = include_truth

    def detach_writer(self, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self._writer is None:
            return None
        writer = self._writer
        writer.vehicle["config"] = self.vehicle.config.to_dict()
        writer.vehicle["driving_mode"] = self.vehicle.mode
        writer.vehicle["poses"] = list(self.poses)
        self._writer = None
        return writer.close(extra)

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    @property
    def pending_frames(self) -> int:
        """How many radar frames the accumulated physics time can support."""
        return int(self._physics_accum_s / self.settings.frame_period_s)

    def advance_physics(self, dt: float) -> None:
        """Advance the vehicle only (used by the GUI for smooth 60 Hz motion).

        ``dt`` is split into steps of at most ``_MAX_PHYSICS_DT_S`` so that a
        large GUI tick does not change the bike model's behaviour.
        """
        if dt <= 0.0 or self.finished:
            return
        substeps = max(1, int(np.ceil(dt / _MAX_PHYSICS_DT_S)))
        for _ in range(substeps):
            self.vehicle.step(dt / substeps)
        self._physics_accum_s += dt

    def step(self) -> FrameOutput | None:
        """Advance exactly one radar frame -- the unit of determinism."""
        self.advance_physics(float(self.settings.frame_period_s))
        return self.produce_frame()

    def produce_frame(self) -> FrameOutput | None:
        """Trace + synthesise one frame without touching the physics clock."""
        if self.finished:
            return None
        self._physics_accum_s = max(
            0.0, self._physics_accum_s - self.settings.frame_period_s
        )
        radar_pos, radar_yaw, radar_vel = self.vehicle.radar_pose()

        t0 = time.perf_counter()
        paths = trace_paths(self.scene, self.scatterers, radar_pos, radar_vel, cfg=self.cfg)
        t1 = time.perf_counter()
        result = synthesize_frame(
            paths,
            self.scatterers,
            self.scene,
            self.cfg,
            height_m=self.settings.height_m,
            max_paths=self.settings.max_paths,
            dynamic_range_db=self.settings.dynamic_range_db,
            rng=self.rng,
            include_truth=self._include_truth,
        )
        t2 = time.perf_counter()
        trace_ms = (t1 - t0) * 1e3
        synth_ms = (t2 - t1) * 1e3
        self._trace_ms.append(trace_ms)
        self._synth_ms.append(synth_ms)

        pose = self.vehicle.state.as_dict(frame=self.frame_index, t_s=self.elapsed_s)
        pose["radar_x"] = round(float(radar_pos[0]), 6)
        pose["radar_y"] = round(float(radar_pos[1]), 6)
        pose["radar_yaw_rad"] = round(float(radar_yaw), 9)
        pose["paths_used"] = int(result.paths.count)

        output = FrameOutput(
            index=self.frame_index,
            t_s=self.elapsed_s,
            pose=pose,
            result=result,
            radar_pos=radar_pos,
            timings_ms={"raytrace": trace_ms, "synthesis": synth_ms},
        )
        self.poses.append(pose)
        self.last = output
        self.frame_index += 1
        self.elapsed_s += self.settings.frame_period_s

        if self._writer is not None:
            self._writer.add_frame(
                result.cube,
                frame_meta=pose,
                truth=result.truth if self._include_truth else None,
            )
        return output

    def run(
        self,
        on_frame: Callable[[FrameOutput], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[FrameOutput]:
        frames: list[FrameOutput] = []
        while not self.finished:
            if should_stop is not None and should_stop():
                break
            frame = self.step()
            if frame is None:
                break
            frames.append(frame)
            if on_frame is not None:
                on_frame(frame)
        return frames

    # ------------------------------------------------------------------ #
    def export(
        self,
        out_dir: str,
        *,
        layout: str = "ti_dca1000",
        iq_scale_mode: str = "frame0",
        iq_scale: float | None = None,
        include_truth: bool = True,
        on_frame: Callable[[FrameOutput], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Reset, run the whole simulation and write the dataset."""
        self.reset()
        writer = DatasetWriter(
            out_dir,
            self.cfg,
            layout=layout,
            iq_scale_mode=iq_scale_mode,
            iq_scale=iq_scale,
            scene=self.scene,
            simulation=self.settings.to_dict(),
        )
        self.attach_writer(writer, include_truth=include_truth)
        self.run(on_frame=on_frame, should_stop=should_stop)
        extra = {
            "performance": {
                "raytrace_ms_mean": float(np.mean(self._trace_ms)) if self._trace_ms else 0.0,
                "synthesis_ms_mean": float(np.mean(self._synth_ms)) if self._synth_ms else 0.0,
                "frames_completed": self.frame_index,
            }
        }
        meta = self.detach_writer(extra) or {}
        return meta

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "scene_primitives": len(self.scene.primitives),
            "scatterers": self.scatterers.count,
            "vehicles": 1,
            "driving_mode": self.vehicle.mode,
            "frames": self.num_frames,
            "frame_period_s": self.settings.frame_period_s,
            "duration_s": round(self.settings.duration_s, 6),
            "cube_shape": (
                int(cfg.chirps_per_frame), int(cfg.virtual_elements),
                int(cfg.samples_per_chirp),
            ),
            "frame_bytes": int(cfg.frame_bytes),
            "dataset_bytes": int(cfg.frame_bytes) * self.num_frames,
            "range_resolution_m": round(cfg.range_resolution_m, 4),
            "max_range_m": round(cfg.max_range_m, 3),
            "max_velocity_mps": round(cfg.max_velocity_mps, 4),
            "velocity_resolution_mps": round(cfg.velocity_resolution_mps, 5),
            "angular_resolution_deg": round(float(np.degrees(cfg.angular_resolution_rad)), 3),
        }

    @classmethod
    def from_scene_dict(
        cls,
        scene: Scene,
        data: dict[str, Any] | None = None,
        *,
        radar_cfg: RadarConfig | None = None,
        settings: SimulationSettings | None = None,
    ) -> "Simulator":
        """Build a simulator from a ``.scene.json`` dictionary."""
        data = data or {}
        vehicle_block = data.get("vehicle", {}) or {}
        config = VehicleConfig.from_dict(vehicle_block)
        path = vehicle_block.get("path")
        vehicle = Vehicle(config=config, path=path)
        vehicle.config.path_speed_mps = float(
            vehicle_block.get("path_speed", vehicle_block.get("path_speed_mps",
                                                             config.path_speed_mps))
        )
        radar = radar_cfg or RadarConfig.from_dict(data.get("radar") or {})
        sim_settings = settings or SimulationSettings.from_dict(data.get("simulation"))
        return cls(scene, radar, vehicle, sim_settings)

    def paths_of_primitive(self, owner: int) -> Sequence[np.ndarray]:
        """Debug helper: world-space outline of a traced primitive owner."""
        ordered = self.scene.ordered()
        if not 0 <= owner < len(ordered):
            return []
        outline = ordered[owner].outline()
        return [outline] if outline is not None else []
