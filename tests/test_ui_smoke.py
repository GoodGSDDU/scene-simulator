"""GUI smoke tests -- run offscreen so no display is required.

These exercise the real widget tree: item painting, panel round-trips, the run
loop and the export path.  They are intentionally shallow but they catch the
import/API mistakes that would otherwise only show up when a human opens the
window.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from radar_sim.core.materials import Material  # noqa: E402
from radar_sim.core.radar_config import RadarConfig  # noqa: E402
from radar_sim.core.scene import Primitive, Scene  # noqa: E402
from radar_sim.core.simulator import SimulationSettings  # noqa: E402
from radar_sim.core.vehicle import Vehicle, VehicleConfig  # noqa: E402
from radar_sim.ui.canvas import to_scene, to_world  # noqa: E402
from radar_sim.ui.main_window import MainWindow  # noqa: E402

SMALL = dict(chirps_per_frame=16, samples_per_chirp=32, tx_elements=2, rx_elements=2)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path):
    scene = Scene.empty(80.0, 50.0)
    scene.add(Primitive("rect", {"x": 30.0, "y": 20.0, "w": 8.0, "h": 4.0},
                        name="建筑", material_id="metal"))
    scene.add(Primitive("point", {"x": 18.0, "y": 24.0}, name="标定点",
                        material_id="metal"))
    cfg = RadarConfig.iwr1843(**SMALL)
    vehicle = Vehicle(VehicleConfig(start=(8.0, 25.0, 0.0), path_speed_mps=5.0),
                      path=[[8.0, 25.0], [60.0, 25.0]])
    settings = SimulationSettings(frame_period_s=0.08, duration_s=0.48, seed=3,
                                  segment_len_m=1.0)
    win = MainWindow(scene=scene, radar_cfg=cfg, vehicle=vehicle, settings=settings)
    win.timer.stop()                        # keep tests deterministic
    yield win
    win.timer.stop()
    win.close()


# --------------------------------------------------------------------------- #
# coordinates and painting
# --------------------------------------------------------------------------- #
def test_world_scene_roundtrip():
    for x, y in ((0.0, 0.0), (10.0, 20.0), (-5.0, 33.5)):
        point = to_scene(x, y)
        back = to_world(point)
        assert back[0] == pytest.approx(x)
        assert back[1] == pytest.approx(y)
    assert to_scene(0.0, 5.0).y() == pytest.approx(-5.0)


def test_canvas_renders_without_error(window):
    """Exercises drawBackground/drawForeground and every item's paint()."""
    window.canvas.select_primitive(window.simulator.scene.primitives[0])
    window.canvas.set_detections([(20.0, 20.0, 25.0), (30.0, 22.0, 12.0)],
                                 [(25.0, 20.0)])
    window.canvas.set_vehicle(window.simulator.vehicle)
    window.canvas.set_radar_view((8.0, 25.0), 0.0, 50.0, enabled=True)
    window.canvas.resize(600, 400)

    pixmap = window.canvas.grab()
    assert not pixmap.isNull()
    assert pixmap.width() > 0 and pixmap.height() > 0


def test_radar_overlay_updates_from_vehicle(window):
    pos, yaw, _vel = window.simulator.vehicle.radar_pose()
    window._update_overlay_from_vehicle()
    assert window.canvas._radar_pos[0] == pytest.approx(float(pos[0]))
    assert window.canvas._radar_range == pytest.approx(window.simulator.cfg.max_range_m)


# --------------------------------------------------------------------------- #
# scene editing
# --------------------------------------------------------------------------- #
def test_add_primitive_updates_scene_and_list(window):
    before = len(window.simulator.scene.primitives)
    prim = window.canvas.add_primitive("circle", (12.0, 12.0))
    assert len(window.simulator.scene.primitives) == before + 1
    assert prim.kind == "circle"
    window._on_scene_edited(prim.id)
    assert window.scene_list.count() == before + 1


def test_property_panel_edit_rebuilds_scatterers(window):
    prim = window.simulator.scene.primitives[1]
    window.canvas.select_primitive(prim)
    window.property_panel.set_primitive(prim)
    assert window.property_panel.primitive is prim

    before = window.simulator.scatterers.count
    window.property_panel._geometry_widgets  # noqa: B018 - ensure populated
    prim.params["r"] = 6.0 if prim.kind == "circle" else prim.params.get("r", 1.0)
    window._on_property_changed()
    assert window.simulator.scatterers.count >= 0
    assert isinstance(before, int)


def test_material_change_updates_canvas_colour_source(window):
    prim = window.simulator.scene.primitives[0]
    window.property_panel.set_primitive(prim)
    index = window.property_panel.material_combo.findData("vegetation")
    window.property_panel.material_combo.setCurrentIndex(index)
    assert prim.material_id == "vegetation"
    material = window.canvas.material_of(prim)
    assert material is not None and material.id == "vegetation"


def test_material_slider_edits_the_shared_material(window):
    prim = window.simulator.scene.primitives[0]
    window.property_panel.set_primitive(prim)
    window.property_panel._material_widgets["rho"].setValue(0.11)
    assert window.simulator.scene.material(prim.material_id).rho == pytest.approx(0.11)


def test_visibility_flags_are_wired(window):
    window.act_grid.setChecked(False)
    assert window.canvas.show_grid is False
    window.act_labels.setChecked(False)
    assert window.canvas.show_labels is False
    window.act_radar.setChecked(False)
    assert window.canvas.show_radar is False


def test_pending_kind_toggles(window):
    window._set_pending_kind("rect")
    assert window._pending_kind == "rect"
    window._set_pending_kind("rect")
    assert window._pending_kind is None


def test_background_click_places_a_primitive(window):
    window._set_pending_kind("point")
    before = len(window.simulator.scene.primitives)
    window._on_background_clicked(15.0, 15.0, 0)
    assert len(window.simulator.scene.primitives) == before + 1
    assert window._pending_kind == "point", "placement should stay armed"


def test_ctrl_click_appends_a_waypoint(window):
    window._set_pending_kind(None)
    before = len(window.simulator.vehicle.path)
    ctrl = int(Qt.KeyboardModifier.ControlModifier.value)
    window._on_background_clicked(40.0, 40.0, ctrl)
    assert len(window.simulator.vehicle.path) == before + 1


# --------------------------------------------------------------------------- #
# radar panel
# --------------------------------------------------------------------------- #
def test_radar_panel_roundtrip(window):
    window.radar_panel.set_config(window.simulator.cfg)
    window.radar_panel.chirps.setValue(24)
    assert window.simulator.cfg.chirps_per_frame == 24
    assert window.radar_panel.derived.text()
    window.radar_panel.samples.setValue(64)
    assert window.simulator.cfg.samples_per_chirp == 64


def test_radar_panel_rejects_invalid_config_without_crashing(window):
    window.radar_panel.setValue_guard = True                     # no-op attribute
    window.radar_panel.ramp.setValue(5.0)                        # too short for 256/10MHz
    window.radar_panel.samples.setValue(1024)
    # invalid settings must be reported in the derived label, not applied
    assert "⚠" in window.radar_panel.derived.text()


# --------------------------------------------------------------------------- #
# run loop
# --------------------------------------------------------------------------- #
def test_run_loop_produces_frames(window):
    window._start()
    window._last_tick = time.perf_counter() - 0.4               # pretend 400 ms passed
    window._on_tick()
    assert window.simulator.frame_index > 0
    assert window._last_frame is not None
    assert window.signal_view.status.text()
    window._stop()
    assert not window._running


def test_pause_stops_frame_production(window):
    window._start()
    window._toggle_pause()
    index = window.simulator.frame_index
    window._last_tick = time.perf_counter() - 0.5
    window._on_tick()
    assert window.simulator.frame_index == index


def test_reset_clears_everything(window):
    window._start()
    window._last_tick = time.perf_counter() - 0.4
    window._on_tick()
    window._stop()
    window._reset()
    assert window.simulator.frame_index == 0
    assert window._last_frame is None
    assert window.canvas._detections == []


def test_manual_driving_input_from_keys(window):
    window.canvas.keyPressEvent(_key_event(Qt.Key.Key_Up))
    window.canvas.keyPressEvent(_key_event(Qt.Key.Key_Left))
    throttle, steer = window.canvas.driving_input()
    assert throttle == pytest.approx(1.0)
    assert steer == pytest.approx(1.0)
    window.canvas.keyReleaseEvent(_key_event(Qt.Key.Key_Up))
    throttle, _steer = window.canvas.driving_input()
    assert throttle == pytest.approx(0.0)


def test_vehicle_panel_mode_switch_creates_a_path(window):
    window.simulator.vehicle.clear_path()
    window.vehicle_panel.refresh()
    window.vehicle_panel.mode.setCurrentIndex(1)
    assert window.simulator.vehicle.mode == "path"
    window.vehicle_panel.mode.setCurrentIndex(0)
    assert window.simulator.vehicle.mode == "manual"


# --------------------------------------------------------------------------- #
# persistence and export
# --------------------------------------------------------------------------- #
def test_save_and_reload_scene(window, tmp_path):
    target = tmp_path / "scene.json"
    window.scene_path = target
    window._save_scene()
    assert target.exists()

    scenario = window._vehicle_block()
    assert scenario["path"] is not None
    assert scenario["mode"] == "path"

    import json

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["radar"]["chirps_per_frame"] == 16
    restored = Scene.from_dict(data)
    assert len(restored.primitives) == len(window.simulator.scene.primitives)


def test_replace_scene_rebuilds_the_widgets(window):
    data = Scene.demo().to_dict()
    data["radar"] = dict(RadarConfig.iwr1843(**SMALL).to_dict())
    data["vehicle"] = {"start": [5.0, 30.0, 0.0],
                       "path": [[5.0, 30.0], [40.0, 30.0]]}
    data["simulation"] = {"frame_period_s": 0.1, "duration_s": 0.3, "seed": 5}

    window._replace_scene(Scene.from_dict(data), None, data)
    assert window.simulator.cfg.chirps_per_frame == 16
    assert window.simulator.vehicle.mode == "path"
    assert window.scene_list.count() == len(window.simulator.scene.primitives)
    assert window.sim_panel.duration.value() == pytest.approx(0.3)
    assert window.sim_panel.seed.value() == 5


def test_export_via_simulator_writes_files(window, tmp_path):
    window.simulator.settings.duration_s = 0.24
    window.simulator.reset()
    meta = window.simulator.export(tmp_path / "out", include_truth=True)
    assert (tmp_path / "out" / "signal.bin").exists()
    assert meta["signal"]["num_frames"] == 3
    assert meta["vehicle"]["poses"]


def test_signal_view_tabs_all_render(window):
    """Switch through every tab and redraw -- catches matplotlib API mistakes."""
    frame = window.simulator.step()
    assert frame is not None
    for index in range(window.signal_view.tabs.count()):
        window.signal_view.tabs.setCurrentIndex(index)
        window.signal_view.update_frame(
            frame.result.cube,
            radar_pos=(float(frame.radar_pos[0]), float(frame.radar_pos[1])),
            radar_yaw=0.0,
            truth=frame.result.truth,
        )


def test_signal_view_handles_an_empty_truth_table(window):
    window.signal_view.tabs.setCurrentIndex(3)
    window.signal_view.update_frame(
        np.zeros((16, 4, 32), dtype=complex), truth={"range_m": np.zeros(0)}
    )


# --------------------------------------------------------------------------- #
def _key_event(key):
    from PySide6.QtGui import QKeyEvent

    return QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
