"""Simulation loop tests: frame clock, poses, export and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core.radar_config import RadarConfig
from radar_sim.core.scene import Primitive, Scene
from radar_sim.core.simulator import SimulationSettings, Simulator
from radar_sim.core.vehicle import Vehicle, VehicleConfig

SMALL_RADAR = dict(chirps_per_frame=16, samples_per_chirp=32, tx_elements=2, rx_elements=2)


def _cfg(**overrides) -> RadarConfig:
    params = dict(SMALL_RADAR)
    params.update(overrides)
    return RadarConfig.iwr1843(**params)


def _scene() -> Scene:
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("rect", {"x": 20.0, "y": 6.0, "w": 6.0, "h": 2.0}, material_id="metal"))
    scene.add(Primitive("point", {"x": 8.0, "y": 4.0}, material_id="wood"))
    return scene


def _simulator(**overrides) -> Simulator:
    settings = SimulationSettings(
        frame_period_s=0.08, duration_s=0.32, seed=1, segment_len_m=1.0
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    vehicle = Vehicle(VehicleConfig(start=(2.0, 4.0, 0.0), path_speed_mps=4.0))
    vehicle.set_path([[2.0, 4.0], [40.0, 4.0]])
    return Simulator(_scene(), _cfg(), vehicle, settings)


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def test_settings_frame_count():
    assert SimulationSettings(frame_period_s=0.08, duration_s=4.0).num_frames == 50
    assert SimulationSettings(frame_period_s=0.1, duration_s=1.0).num_frames == 10
    assert SimulationSettings(frame_period_s=0.08, duration_s=0.04).num_frames == 1


def test_settings_rejects_bad_values():
    with pytest.raises(ValueError):
        SimulationSettings(frame_period_s=0.0)
    with pytest.raises(ValueError):
        SimulationSettings(duration_s=-1.0)


def test_settings_substeps_bound_the_physics_step():
    settings = SimulationSettings(frame_period_s=0.08)
    assert settings.substeps() == 8
    assert settings.frame_period_s / settings.substeps() <= 0.0100001


def test_settings_roundtrip_and_legacy_fps_key():
    data = SimulationSettings.from_dict({"fps_radar": 10.0, "duration_s": 2.0, "seed": 7})
    assert data.frame_period_s == pytest.approx(0.1)
    assert data.duration_s == pytest.approx(2.0)
    assert data.seed == 7
    assert SimulationSettings.from_dict(None) == SimulationSettings()


# --------------------------------------------------------------------------- #
# stepping
# --------------------------------------------------------------------------- #
def test_step_advances_the_clock():
    sim = _simulator(duration_s=0.24)
    assert sim.num_frames == 3
    frame0 = sim.step()
    assert frame0 is not None
    assert frame0.index == 0
    assert frame0.t_s == pytest.approx(0.0)
    assert sim.frame_index == 1
    assert sim.elapsed_s == pytest.approx(0.08)

    frame1 = sim.step()
    assert frame1 is not None
    assert frame1.t_s == pytest.approx(0.08)


def test_step_returns_none_when_finished():
    sim = _simulator(duration_s=0.16)
    assert sim.step() is not None
    assert sim.step() is not None
    assert sim.finished
    assert sim.step() is None


def test_cube_shape_and_pose_record():
    sim = _simulator()
    frame = sim.step()
    assert frame is not None
    assert frame.result.cube.shape == (
        sim.cfg.chirps_per_frame, sim.cfg.virtual_elements, sim.cfg.samples_per_chirp
    )
    pose = frame.pose
    for key in ("frame", "t_s", "x", "y", "phi_rad", "speed_mps",
                "radar_x", "radar_y", "radar_yaw_rad", "paths_used"):
        assert key in pose
    assert pose["radar_x"] > pose["x"], "radar sits ahead of the body origin"
    assert sim.poses == [pose]


def test_vehicle_actually_moves_across_frames():
    sim = _simulator(duration_s=1.6)
    sim.run()
    xs = [p["x"] for p in sim.poses]
    assert xs[-1] > xs[0] + 1.0, "vehicle did not move"
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_run_collects_every_frame():
    sim = _simulator(duration_s=0.4)
    frames = sim.run()
    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_run_honours_the_stop_flag():
    sim = _simulator(duration_s=1.0)
    stop_after = {"n": 0}

    def should_stop() -> bool:
        stop_after["n"] += 1
        return stop_after["n"] > 2

    frames = sim.run(should_stop=should_stop)
    assert len(frames) == 2
    assert not sim.finished


def test_reset_restores_everything():
    sim = _simulator(duration_s=0.32)
    first = sim.run()
    sim.reset()
    assert sim.frame_index == 0
    assert sim.elapsed_s == 0.0
    assert sim.poses == []
    assert sim.last is None
    second = sim.run()
    assert np.array_equal(first[0].result.cube, second[0].result.cube)


def test_rebuild_scatterers_picks_up_scene_edits():
    sim = _simulator(segment_len_m=1.0)
    before = sim.scatterers.count
    sim.scene.add(Primitive("circle", {"x": 30.0, "y": 30.0, "r": 3.0}))
    after = sim.rebuild_scatterers()
    assert after > before
    assert sim.scatterers.count == after


def test_summary_reports_derived_quantities():
    sim = _simulator()
    summary = sim.summary()
    assert summary["frames"] == sim.num_frames
    assert summary["dataset_bytes"] == sim.cfg.frame_bytes * sim.num_frames
    assert summary["driving_mode"] == "path"
    assert summary["scatterers"] > 0
    assert summary["max_range_m"] == pytest.approx(sim.cfg.max_range_m, abs=1e-3)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_writes_a_complete_dataset(tmp_path):
    sim = _simulator(duration_s=0.24)
    meta = sim.export(tmp_path / "run", include_truth=True)

    assert (tmp_path / "run" / "signal.bin").stat().st_size == 3 * sim.cfg.frame_bytes
    assert meta["simulation"]["num_frames"] == 3
    assert meta["vehicle"]["driving_mode"] == "path"
    assert len(meta["vehicle"]["poses"]) == 3
    assert meta["vehicle"]["config"]["wheelbase_m"] == pytest.approx(2.7)
    assert meta["performance"]["frames_completed"] == 3
    assert (tmp_path / "run" / "pointcloud_gt.csv").exists()


def test_export_without_truth_skips_the_csv(tmp_path):
    sim = _simulator(duration_s=0.16)
    sim.export(tmp_path / "run", include_truth=False)
    assert not (tmp_path / "run" / "pointcloud_gt.csv").exists()


def test_export_is_reproducible(tmp_path):
    a = _simulator(duration_s=0.24)
    a.export(tmp_path / "a", include_truth=False)
    b = _simulator(duration_s=0.24)
    b.export(tmp_path / "b", include_truth=False)
    assert (tmp_path / "a" / "signal.bin").read_bytes() == \
        (tmp_path / "b" / "signal.bin").read_bytes()


def test_export_layout_choice_is_recorded(tmp_path):
    sim = _simulator(duration_s=0.08)
    meta = sim.export(tmp_path / "run", layout="interleaved_iq")
    assert meta["signal"]["layout"] == "interleaved_iq"


def test_export_resets_previous_progress(tmp_path):
    sim = _simulator(duration_s=0.16)
    sim.run()
    assert sim.frame_index == 2
    meta = sim.export(tmp_path / "run")
    assert meta["signal"]["num_frames"] == 2
    assert len(meta["vehicle"]["poses"]) == 2


def test_attach_and_detach_writer(tmp_path):
    from radar_sim.core.io_export import DatasetWriter

    sim = _simulator(duration_s=0.16)
    writer = DatasetWriter(tmp_path / "run", sim.cfg, scene=sim.scene)
    sim.attach_writer(writer)
    sim.run()
    meta = sim.detach_writer()
    assert meta is not None
    assert meta["signal"]["num_frames"] == 2
    assert sim.detach_writer() is None


# --------------------------------------------------------------------------- #
# construction from a scene dict
# --------------------------------------------------------------------------- #
def test_from_scene_dict_builds_vehicle_and_radar():
    data = _scene().to_dict()
    data["vehicle"] = {
        "start": [3.0, 5.0, 45.0],
        "path": [[3.0, 5.0], [30.0, 20.0]],
        "path_speed": 7.5,
        "wheelbase": 3.0,
    }
    data["radar"] = _cfg().to_dict()
    data["simulation"] = {"frame_period_s": 0.1, "duration_s": 0.5, "seed": 9}

    sim = Simulator.from_scene_dict(Scene.from_dict(data), data)
    assert sim.vehicle.mode == "path"
    assert sim.vehicle.config.path_speed_mps == pytest.approx(7.5)
    assert sim.vehicle.config.wheelbase_m == pytest.approx(3.0)
    assert sim.vehicle.state.x == pytest.approx(3.0)
    assert sim.cfg.chirps_per_frame == 16
    assert sim.settings.seed == 9
    assert sim.num_frames == 5


def test_manual_mode_standstill_produces_identical_frames():
    settings = SimulationSettings(frame_period_s=0.08, duration_s=0.16, seed=2,
                                  segment_len_m=1.0)
    vehicle = Vehicle(VehicleConfig(start=(2.0, 4.0, 0.0)))      # no path, no input
    sim = Simulator(_scene(), _cfg(enable_noise=False), vehicle, settings)
    frames = sim.run()
    assert len(frames) == 2
    assert np.allclose(frames[0].result.cube, frames[1].result.cube)
    assert vehicle.state.speed == pytest.approx(0.0)
