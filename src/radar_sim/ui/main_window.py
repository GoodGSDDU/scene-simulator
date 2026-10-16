"""Main window: assembles the scene editor, panels, signal view and run loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..core.radar_config import RadarConfig
from ..core.scene import Primitive, Scene
from ..core.simulator import SimulationSettings, Simulator
from ..core.vehicle import Vehicle, VehicleConfig
from .canvas import SceneCanvas
from .panels import PropertyPanel, RadarPanel, SimPanel, StatsPanel, VehiclePanel
from .signal_view import SignalView

RENDER_INTERVAL_MS = 16
MAX_MULTIPLIER = 20.0

_KIND_LABELS = (
    ("circle", "⬤ 圆"),
    ("rect", "▬ 矩形"),
    ("polygon", "▲ 多边形"),
    ("wall", "▤ 墙"),
    ("point", "◆ 点目标"),
)


class MainWindow(QMainWindow):
    def __init__(
        self,
        scene: Scene | None = None,
        radar_cfg: RadarConfig | None = None,
        vehicle: Vehicle | None = None,
        settings: SimulationSettings | None = None,
        *,
        scene_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("2D 场景雷达仿真器")
        self.resize(1500, 950)

        self.scene_path = scene_path
        self._pending_kind: str | None = None
        self._running = False
        self._paused = False
        self._last_tick = time.perf_counter()
        self._last_frame = None

        self._make_simulator(scene, radar_cfg, vehicle, settings)
        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._wire()
        self._refresh_scene_list()
        self._sync_sim_panel()
        self._update_overlay_from_vehicle()
        self._update_title()

        self.timer = QTimer(self)
        self.timer.setInterval(RENDER_INTERVAL_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _make_simulator(self, scene, radar_cfg, vehicle, settings) -> None:
        self.simulator = Simulator(
            scene if scene is not None else Scene.demo(),
            radar_cfg if radar_cfg is not None else RadarConfig.iwr1843(),
            vehicle if vehicle is not None else Vehicle(),
            settings if settings is not None else SimulationSettings(
                frame_period_s=0.08, duration_s=4.0, seed=42
            ),
        )

    def _build_actions(self) -> None:
        self.act_new = QAction("新建场景(&N)", self)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)
        self.act_open = QAction("打开场景(&O)...", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_save = QAction("保存场景(&S)", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save_as = QAction("另存为...", self)
        self.act_export = QAction("导出仿真数据(&E)...", self)
        self.act_export.setShortcut("Ctrl+Shift+E")
        self.act_quit = QAction("退出(&Q)", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)

        self.act_grid = QAction("显示网格", self, checkable=True, checked=True)
        self.act_labels = QAction("显示名称", self, checkable=True, checked=True)
        self.act_radar = QAction("显示雷达视场", self, checkable=True, checked=True)
        self.act_detections = QAction("显示检测点云", self, checkable=True, checked=True)
        self.act_truth = QAction("显示真值点云", self, checkable=True, checked=True)
        self.act_fit = QAction("适应窗口(&F)", self)
        self.act_fit.setShortcut("Ctrl+0")

        self.act_about = QAction("关于", self)

    def _build_ui(self) -> None:
        self.canvas = SceneCanvas(self.simulator.scene)
        self.signal_view = SignalView(self.simulator.cfg)

        self.property_panel = PropertyPanel(self.simulator.scene)
        self.radar_panel = RadarPanel(self.simulator.cfg)
        self.vehicle_panel = VehiclePanel(self.simulator.vehicle)
        self.stats_panel = StatsPanel()
        self.sim_panel = SimPanel()

        tabs = QTabWidget()
        tabs.addTab(self.radar_panel, "雷达")
        tabs.addTab(self.vehicle_panel, "车辆")
        tabs.addTab(self.property_panel, "图元属性")
        tabs.addTab(self.stats_panel, "统计")

        right = QDockWidget("参数", self)
        right.setWidget(tabs)
        right.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right)

        left = QDockWidget("场景元素", self)
        left.setWidget(self._build_left_panel())
        left.setMinimumWidth(190)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.signal_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        bottom = QToolBar("运行", self)
        bottom.setMovable(False)
        bottom.addWidget(self.sim_panel)
        dock = QDockWidget("运行控制", self)
        dock.setWidget(bottom)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.statusBar().showMessage("就绪")
        self.sim_panel.set_running(False)

    def _build_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        palette = QToolBar()
        palette.setOrientation(Qt.Orientation.Vertical)
        self.palette_buttons: dict[str, QPushButton] = {}
        for kind, label in _KIND_LABELS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(f"点击后在画布上放置{label}")
            button.clicked.connect(lambda _c=False, k=kind: self._set_pending_kind(k))
            self.palette_buttons[kind] = button
            palette.addWidget(button)
        layout.addWidget(palette)

        layout.addWidget(QLabel("图元列表"))
        self.scene_list = QListWidget()
        layout.addWidget(self.scene_list, 1)
        return widget

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件(&F)")
        for action in (self.act_new, self.act_open, self.act_save, self.act_save_as):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        view_menu = self.menuBar().addMenu("视图(&V)")
        for action in (self.act_grid, self.act_labels, self.act_radar,
                       self.act_detections, self.act_truth, self.act_fit):
            view_menu.addAction(action)

        help_menu = self.menuBar().addMenu("帮助(&H)")
        help_menu.addAction(self.act_about)

    def _wire(self) -> None:
        self.act_new.triggered.connect(self._new_scene)
        self.act_open.triggered.connect(self._open_scene)
        self.act_save.triggered.connect(self._save_scene)
        self.act_save_as.triggered.connect(lambda: self._save_scene(force_dialog=True))
        self.act_export.triggered.connect(self._export)
        self.act_quit.triggered.connect(self.close)
        self.act_about.triggered.connect(self._about)

        self.act_grid.toggled.connect(lambda v: self._set_canvas_option("show_grid", v))
        self.act_labels.toggled.connect(lambda v: self._set_canvas_option("show_labels", v))
        self.act_radar.toggled.connect(lambda v: self._set_canvas_option("show_radar", v))
        self.act_detections.toggled.connect(
            lambda v: self._set_canvas_option("show_detections", v))
        self.act_truth.toggled.connect(lambda v: self._set_canvas_option("show_truth", v))
        self.act_fit.triggered.connect(self.canvas.fit_to_scene)

        self.canvas.scene_edited.connect(self._on_scene_edited)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        self.canvas.background_clicked.connect(self._on_background_clicked)

        self.property_panel.changed.connect(self._on_property_changed)
        self.radar_panel.changed.connect(self._on_radar_changed)
        self.vehicle_panel.changed.connect(self._on_vehicle_changed)
        self.vehicle_panel.mode_changed.connect(self._on_mode_changed)
        self.scene_list.currentItemChanged.connect(self._on_list_selection)

        self.sim_panel.start_requested.connect(self._start)
        self.sim_panel.pause_requested.connect(self._toggle_pause)
        self.sim_panel.stop_requested.connect(self._stop)
        self.sim_panel.reset_requested.connect(self._reset)
        self.sim_panel.export_requested.connect(self._export)
        self.sim_panel.settings_changed.connect(self._apply_sim_settings)

    # ------------------------------------------------------------------ #
    # run loop
    # ------------------------------------------------------------------ #
    def _sync_sim_panel(self) -> None:
        """Mirror the simulator settings into the run-control widgets."""
        settings = self.simulator.settings
        for widget, value in ((self.sim_panel.duration, settings.duration_s),
                              (self.sim_panel.seed, settings.seed)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _apply_sim_settings(self) -> None:
        settings = self.simulator.settings
        settings.duration_s = float(self.sim_panel.duration.value())
        settings.seed = int(self.sim_panel.seed.value())
        if not self._running:
            self.simulator.reset()

    def _on_tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now

        # always feed the manual driving keys so the car reacts while paused
        if self.canvas.radar_enabled and self.simulator.vehicle.mode == "manual":
            throttle, steer = self.canvas.driving_input()
            self.simulator.vehicle.set_manual_input(throttle, steer)

        if not self._running or self._paused:
            return

        multiplier = self.sim_panel.time_multiplier
        scale = MAX_MULTIPLIER if multiplier <= 0.0 else multiplier
        self.simulator.advance_physics(dt * scale)

        produced = 0
        while self.simulator.pending_frames > 0 and not self.simulator.finished:
            frame = self.simulator.produce_frame()
            if frame is None:
                break
            self._show_frame(frame)
            produced += 1
            if produced >= 8:                    # never block the UI for long
                break

        self.canvas.set_vehicle(self.simulator.vehicle)
        self._update_overlay_from_vehicle()
        self.canvas.viewport().update()

        if self.simulator.finished:
            self._stop()
            self.statusBar().showMessage("仿真结束，可以导出数据")

        if produced or self.simulator.last is not None:
            self._update_stats()

    def _show_frame(self, frame) -> None:
        self._last_frame = frame
        cfg = self.simulator.cfg
        self.signal_view.set_status(
            f"frame {frame.index + 1}/{self.simulator.num_frames}   "
            f"t={frame.t_s:.2f}s   paths={frame.result.paths.count}   "
            f"raytrace={frame.timings_ms['raytrace']:.1f}ms   "
            f"synth={frame.timings_ms['synthesis']:.1f}ms"
        )
        self.signal_view.update_frame(
            frame.result.cube,
            radar_pos=(float(frame.radar_pos[0]), float(frame.radar_pos[1])),
            radar_yaw=float(self.simulator.vehicle.state.yaw),
            truth=frame.result.truth if frame.result.truth else None,
        )
        cloud = self.signal_view.detections
        points = []
        if cloud.get("x") is not None and cloud["x"].size:
            points = list(zip(cloud["x"], cloud["y"], cloud["snr_db"]))
        truth_points = self._truth_points(frame)
        self.canvas.set_detections(points, truth_points)
        self.sim_panel.set_progress(frame.index + 1, self.simulator.num_frames)

    def _truth_points(self, frame) -> list[tuple[float, float]]:
        truth = frame.result.truth
        if not truth:
            return []
        ranges = np.asarray(truth.get("range_m", ()))
        if ranges.size == 0:
            return []
        angles = (np.deg2rad(np.asarray(truth["azimuth_deg"]))
                  + self.simulator.vehicle.state.yaw)
        return list(zip(float(frame.radar_pos[0]) + ranges * np.cos(angles),
                        float(frame.radar_pos[1]) + ranges * np.sin(angles)))

    def _update_overlay_from_vehicle(self) -> None:
        vehicle = self.simulator.vehicle
        pos, yaw, _vel = vehicle.radar_pose()
        self.canvas.set_radar_view((float(pos[0]), float(pos[1])), float(yaw),
                                   self.simulator.cfg.max_range_m, enabled=True)
        self.canvas.set_vehicle(vehicle)

    def _update_stats(self) -> None:
        frame = self._last_frame
        summary = self.simulator.summary()
        values = {
            "帧": f"{self.simulator.frame_index}/{self.simulator.num_frames}",
            "散射点": summary["scatterers"],
            "驱动模式": "路径" if self.simulator.vehicle.mode == "path" else "手动",
            "车速": f"{self.simulator.vehicle.state.speed:.2f} m/s",
            "位置": f"({self.simulator.vehicle.state.x:.1f}, "
                    f"{self.simulator.vehicle.state.y:.1f})",
        }
        if frame is not None:
            values.update({
                "回波路径": frame.result.paths.count,
                "多径分解": str(frame.result.stats.get("paths_by_bounce", {})),
                "射线追踪": f"{frame.timings_ms['raytrace']:.1f} ms",
                "IQ 合成": f"{frame.timings_ms['synthesis']:.1f} ms",
            })
        self.stats_panel.update_stats(values)

    # ------------------------------------------------------------------ #
    # run control
    # ------------------------------------------------------------------ #
    def _start(self) -> None:
        if self.simulator.finished:
            self.simulator.reset()
        self._running = True
        self._paused = False
        self._last_tick = time.perf_counter()
        self.sim_panel.set_running(True)
        self.canvas.setFocus()
        self.statusBar().showMessage("仿真运行中 —— 方向键 / WASD 驾驶")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.statusBar().showMessage("已暂停" if self._paused else "继续运行")

    def _stop(self) -> None:
        self._running = False
        self._paused = False
        self.sim_panel.set_running(False)
        self.canvas.clear_keys()

    def _reset(self) -> None:
        self._stop()
        self.simulator.reset()
        self.vehicle_panel.apply_start_pose()
        self.canvas.clear_overlays()
        self.signal_view.clear()
        self.signal_view.set_status("已重置")
        self.sim_panel.set_progress(0, self.simulator.num_frames)
        self._last_frame = None
        self._update_overlay_from_vehicle()
        self.canvas.viewport().update()
        self._update_stats()
        self._update_title()

    # ------------------------------------------------------------------ #
    # scene editing
    # ------------------------------------------------------------------ #
    def _set_pending_kind(self, kind: str | None) -> None:
        if kind is not None and self._pending_kind == kind:
            kind = None                                          # click again to cancel
        self._pending_kind = kind
        for key, button in self.palette_buttons.items():
            button.setChecked(key == kind)
        if kind:
            self.statusBar().showMessage(
                f"在画布上点击放置「{kind}」（可连续放置，再点按钮取消）"
            )
        else:
            self.statusBar().showMessage("已取消放置")

    def _on_background_clicked(self, x: float, y: float, modifiers: int) -> None:
        if self._pending_kind:
            prim = self.canvas.add_primitive(self._pending_kind, (x, y))
            self._refresh_scene_list()
            self.property_panel.set_primitive(prim)
            self._on_scene_edited(prim.id)
            return

        if modifiers & int(Qt.KeyboardModifier.ControlModifier.value):
            vehicle = self.simulator.vehicle
            path = vehicle.path
            points = [] if path is None else [list(p) for p in path]
            points.append([float(x), float(y)])
            if len(points) >= 2:
                vehicle.set_path(points)
                self.vehicle_panel.refresh()
                self._on_vehicle_changed()
                self.statusBar().showMessage(f"已添加路径点，共 {len(points)} 个")

    def _on_scene_edited(self, prim_id: str) -> None:
        del prim_id
        self.simulator.scene = self.canvas.scene_model
        count = self.simulator.rebuild_scatterers()
        self._refresh_scene_list()
        self.canvas.viewport().update()
        self.statusBar().showMessage(f"场景已更新：{len(self.simulator.scene.primitives)} 个图元 / "
                                     f"{count} 个散射点")

    def _on_property_changed(self) -> None:
        self.canvas.viewport().update()
        self._on_scene_edited("")

    def _on_selection_changed(self, prim) -> None:
        self.property_panel.set_primitive(prim)
        self._sync_list_selection(prim)

    def _on_list_selection(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        prim = self.simulator.scene.by_id(current.data(Qt.ItemDataRole.UserRole))
        if prim is not None and prim is not self.property_panel.primitive:
            self.canvas.select_primitive(prim)
            self.property_panel.set_primitive(prim)

    def _refresh_scene_list(self) -> None:
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        for prim in self.simulator.scene.primitives:
            material = self.canvas.material_of(prim)
            suffix = f" · {material.name}" if material else ""
            item = QListWidgetItem(f"{prim.name}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, prim.id)
            self.scene_list.addItem(item)
        self.scene_list.blockSignals(False)
        self._sync_list_selection(self.property_panel.primitive)

    def _sync_list_selection(self, prim) -> None:
        self.scene_list.blockSignals(True)
        for row in range(self.scene_list.count()):
            item = self.scene_list.item(row)
            item.setSelected(prim is not None
                             and item.data(Qt.ItemDataRole.UserRole) == prim.id)
        self.scene_list.blockSignals(False)

    # ------------------------------------------------------------------ #
    def _on_radar_changed(self, cfg: RadarConfig) -> None:
        try:
            cfg.validate()
        except ValueError:
            self.statusBar().showMessage("雷达参数非法，已忽略（详见雷达页提示）")
            return
        self.simulator.cfg = cfg
        self.signal_view.set_config(cfg)
        self._update_overlay_from_vehicle()
        self.canvas.viewport().update()
        self._update_title()

    def _on_vehicle_changed(self) -> None:
        self.vehicle_panel.refresh()
        self._update_overlay_from_vehicle()
        self.canvas.viewport().update()

    def _on_mode_changed(self, mode: str) -> None:
        self.statusBar().showMessage(
            "手动驾驶：方向键 / WASD，空格刹车" if mode == "manual"
            else "路径模式：Ctrl+左键 在画布上添加路径点"
        )
        self.canvas.clear_keys()
        self._on_vehicle_changed()

    def _set_canvas_option(self, name: str, value: bool) -> None:
        setattr(self.canvas, name, bool(value))
        self.canvas.viewport().update()

    # ------------------------------------------------------------------ #
    # files
    # ------------------------------------------------------------------ #
    def _update_title(self) -> None:
        name = self.scene_path.name if self.scene_path else "未命名场景"
        self.setWindowTitle(f"2D 场景雷达仿真器 — {name}")

    def _new_scene(self) -> None:
        self._replace_scene(Scene.empty(100.0, 60.0), None)
        self.statusBar().showMessage("已新建空场景")

    def _open_scene(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "打开场景", "",
                                                    "场景文件 (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            scene = Scene.from_dict(data)
            self._replace_scene(scene, Path(path), data)
            self.statusBar().showMessage(f"已打开 {path}")
        except Exception as exc:                                # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(exc))

    def _replace_scene(self, scene: Scene, path: Path | None,
                       data: dict | None = None) -> None:
        self._stop()
        vehicle_block = (data or {}).get("vehicle") or {}
        radar_block = (data or {}).get("radar") or {}
        sim_block = (data or {}).get("simulation") or {}

        radar_cfg = RadarConfig.from_dict(radar_block) if radar_block else self.simulator.cfg
        vehicle = Vehicle(config=VehicleConfig.from_dict(vehicle_block),
                          path=vehicle_block.get("path"))
        settings = SimulationSettings.from_dict(sim_block) if sim_block else \
            self.simulator.settings

        self.scene_path = path
        self._make_simulator(scene, radar_cfg, vehicle, settings)

        self.canvas.scene_model = scene
        self.canvas.rebuild()
        self.canvas.set_vehicle(vehicle)
        self.signal_view.set_config(radar_cfg)
        self.property_panel.set_scene(scene)
        self.property_panel.set_primitive(None)
        self.radar_panel.set_config(radar_cfg)
        self.vehicle_panel.vehicle = vehicle
        self.vehicle_panel.refresh()
        self.sim_panel.duration.blockSignals(True)
        self.sim_panel.seed.blockSignals(True)
        self.sim_panel.duration.setValue(settings.duration_s)
        self.sim_panel.seed.setValue(settings.seed)
        self.sim_panel.duration.blockSignals(False)
        self.sim_panel.seed.blockSignals(False)
        self._refresh_scene_list()
        self._update_overlay_from_vehicle()
        self._update_title()
        self._update_stats()

    def _save_scene(self, force_dialog: bool = False) -> None:
        path = self.scene_path
        if path is None or force_dialog:
            chosen, _filter = QFileDialog.getSaveFileName(
                self, "保存场景", str(path or "scene.json"), "场景文件 (*.json)")
            if not chosen:
                return
            path = Path(chosen)
        data = self.simulator.scene.to_dict()
        data["radar"] = self.simulator.cfg.to_dict()
        data["vehicle"] = self._vehicle_block()
        data["simulation"] = self.simulator.settings.to_dict()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.scene_path = path
        self._update_title()
        self.statusBar().showMessage(f"已保存到 {path}")

    def _vehicle_block(self) -> dict:
        vehicle = self.simulator.vehicle
        return {
            "start": [round(vehicle.state.x, 6), round(vehicle.state.y, 6),
                      round(math.degrees(vehicle.state.yaw), 4)],
            "length": vehicle.config.length_m,
            "width": vehicle.config.width_m,
            "wheelbase": vehicle.config.wheelbase_m,
            "max_speed": vehicle.config.max_speed_mps,
            "max_accel": vehicle.config.max_accel_mps2,
            "max_steer_deg": vehicle.config.max_steer_deg,
            "radar_offset": list(vehicle.config.radar_offset_m),
            "mode": "path" if vehicle.mode == "path" else "manual",
            "path": None if vehicle.path is None else [[round(x, 6), round(y, 6)]
                                                       for x, y in vehicle.path],
            "path_speed": vehicle.config.path_speed_mps,
        }

    def _export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择导出目录", "dataset")
        if not folder:
            return
        layout = self.sim_panel.layout_name
        self._apply_sim_settings()
        self._stop()
        self.statusBar().showMessage("正在导出，请稍候 ...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            meta = self.simulator.export(
                folder, layout=layout, include_truth=self.sim_panel.truth.isChecked()
            )
        except Exception as exc:                                # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QApplication.restoreOverrideCursor()

        signal = meta.get("signal", {})
        perf = meta.get("performance", {})
        size_mb = signal.get("data_bytes", 0) / 1024 / 1024
        QMessageBox.information(
            self, "导出完成",
            f"目录：{folder}\n"
            f"帧数：{signal.get('num_frames', 0)}\n"
            f"布局：{signal.get('layout', '')}\n"
            f"数据量：{size_mb:.1f} MiB\n"
            f"平均耗时：射线 {perf.get('raytrace_ms_mean', 0):.1f} ms / "
            f"合成 {perf.get('synthesis_ms_mean', 0):.1f} ms\n\n"
            f"文件：signal.bin、meta.json、pointcloud_gt.csv、README.txt",
        )
        self.statusBar().showMessage(f"已导出到 {folder}")
        self._reset()

    def _about(self) -> None:
        QMessageBox.information(
            self, "关于",
            "2D 场景雷达仿真器 0.1.0\n"
            "FMCW 波形级仿真 + 理想点云真值\n"
            "镜像法射线追踪（默认 1 次反射）\n\n"
            "数据处理模板见 templates/ 目录",
        )

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._stop()
        super().closeEvent(event)
