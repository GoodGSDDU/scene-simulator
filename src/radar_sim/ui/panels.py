"""Dockable panels: primitive/material properties, radar config, vehicle, run control."""

from __future__ import annotations

import math
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.materials import PRESETS, Material
from ..core.radar_config import MIMO_MODES, RadarConfig
from ..core.scene import Scene
from ..core.vehicle import Vehicle, VehicleConfig

_GEOMETRY_LABELS = {
    "x": ("X (m)", -500.0, 500.0, 2),
    "y": ("Y (m)", -500.0, 500.0, 2),
    "r": ("半径 (m)", 0.1, 200.0, 2),
    "w": ("宽 (m)", 0.1, 200.0, 2),
    "h": ("高 (m)", 0.1, 200.0, 2),
    "rot_deg": ("旋转 (°)", -180.0, 180.0, 1),
    "thickness": ("厚度 (m)", 0.02, 5.0, 2),
}

_MATERIAL_FIELDS = (
    ("rho", "反射率 ρ", 0.0, 1.0, 0.01),
    ("eta", "粗糙度 η", 0.0, 1.0, 0.01),
    ("tau", "透射率 τ", 0.0, 1.0, 0.01),
    ("absorption_db", "吸收 (dB)", 0.0, 40.0, 0.5),
    ("phase_deg", "反射相位 (°)", -180.0, 180.0, 1.0),
    ("sigma_psi_deg", "镜面窗口 (°)", 1.0, 45.0, 0.5),
)


def _dspin(value: float, low: float, high: float, decimals: int = 2,
           step: float = 0.1, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(float(value))
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _ispin(value: int, low: int, high: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(int(value))
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


# --------------------------------------------------------------------------- #
# primitive / material
# --------------------------------------------------------------------------- #
class PropertyPanel(QWidget):
    """Geometry and material editor for the selected primitive."""

    changed = Signal()
    material_changed = Signal(str)

    def __init__(self, scene: Scene, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.primitive = None
        self._updating = False
        self._geometry_widgets: dict[str, QDoubleSpinBox] = {}
        self._material_widgets: dict[str, QDoubleSpinBox] = {}

        self.title = QLabel("未选中图元")
        self.title.setStyleSheet("font-weight:600; color:#ffd166;")

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("图元名称")
        self.name_edit.editingFinished.connect(self._on_name)

        self.material_combo = QComboBox()
        self.material_combo.currentIndexChanged.connect(self._on_material)

        self.geometry_box = QGroupBox("几何")
        self.geometry_form = QFormLayout(self.geometry_box)

        self.material_box = QGroupBox("材质参数")
        self.material_form = QFormLayout(self.material_box)
        for key, label, low, high, step in _MATERIAL_FIELDS:
            box = _dspin(0.5, low, high, 2, step)
            box.valueChanged.connect(lambda _v, k=key: self._on_material_field(k))
            self._material_widgets[key] = box
            self.material_form.addRow(label, box)

        self.reset_material_btn = QPushButton("恢复预设材质")
        self.reset_material_btn.clicked.connect(self._on_reset_material)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        form = QFormLayout()
        form.addRow("名称", self.name_edit)
        form.addRow("材质", self.material_combo)
        layout.addLayout(form)
        layout.addWidget(self.geometry_box)
        layout.addWidget(self.material_box)
        layout.addWidget(self.reset_material_btn)
        layout.addStretch(1)

        self.set_scene(scene)
        self.set_primitive(None)

    # ------------------------------------------------------------------ #
    def set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self._updating = True
        self.material_combo.clear()
        for material in scene.materials.values():
            self.material_combo.addItem(f"{material.name} ({material.id})", material.id)
        self._updating = False
        self.set_primitive(self.primitive)

    def set_primitive(self, prim) -> None:
        self.primitive = prim
        enabled = prim is not None
        for widget in (self.name_edit, self.material_combo, self.geometry_box,
                       self.material_box, self.reset_material_btn):
            widget.setEnabled(enabled)
        if not enabled:
            self.title.setText("未选中图元")
            return

        self._updating = True
        self.title.setText(f"{prim.kind} · {prim.id}")
        self.name_edit.setText(prim.name)
        index = self.material_combo.findData(prim.material_id)
        if index >= 0:
            self.material_combo.setCurrentIndex(index)

        while self.geometry_form.rowCount():
            self.geometry_form.removeRow(0)
        self._geometry_widgets.clear()
        for key, value in prim.params.items():
            if key not in _GEOMETRY_LABELS or not isinstance(value, (int, float)):
                continue
            label, low, high, decimals = _GEOMETRY_LABELS[key]
            box = _dspin(float(value), low, high, decimals)
            box.valueChanged.connect(lambda _v, k=key: self._on_geometry(k))
            self._geometry_widgets[key] = box
            self.geometry_form.addRow(label, box)
        if prim.kind == "polygon":
            self.geometry_form.addRow(QLabel("顶点在画布上不可编辑（本期）"))
        self._load_material_values()
        self._updating = False

    def refresh_material_values(self) -> None:
        self._updating = True
        self._load_material_values()
        self._updating = False

    def _load_material_values(self) -> None:
        if self.primitive is None:
            return
        try:
            material = self.scene.material(self.primitive.material_id)
        except KeyError:
            return
        for key, box in self._material_widgets.items():
            box.setValue(float(getattr(material, key)))

    # ------------------------------------------------------------------ #
    def _on_name(self) -> None:
        if self.primitive is None or self._updating:
            return
        self.primitive.name = self.name_edit.text() or self.primitive.kind
        self.changed.emit()

    def _on_geometry(self, key: str) -> None:
        if self.primitive is None or self._updating:
            return
        self.primitive.params[key] = float(self._geometry_widgets[key].value())
        self.changed.emit()

    def _on_material(self) -> None:
        if self.primitive is None or self._updating:
            return
        material_id = self.material_combo.currentData()
        self.primitive.material_id = str(material_id)
        self.refresh_material_values()
        self.material_changed.emit(str(material_id))
        self.changed.emit()

    def _on_material_field(self, key: str) -> None:
        if self.primitive is None or self._updating:
            return
        try:
            material = self.scene.material(self.primitive.material_id)
        except KeyError:
            return
        setattr(material, key, float(self._material_widgets[key].value()))
        self.changed.emit()

    def _on_reset_material(self) -> None:
        if self.primitive is None:
            return
        preset = next((m for m in PRESETS if m.id == self.primitive.material_id), None)
        if preset is None:
            return
        self.scene.ensure_material(Material.from_dict(preset.to_dict()))
        self.refresh_material_values()
        self.changed.emit()


# --------------------------------------------------------------------------- #
# radar
# --------------------------------------------------------------------------- #
class RadarPanel(QWidget):
    """Every parameter from DESIGN.md section 5.1 plus the derived read-out."""

    changed = Signal(object)

    def __init__(self, cfg: RadarConfig, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self._updating = False

        self.start_freq = _dspin(cfg.start_freq_hz / 1e9, 1.0, 100.0, 3, 0.5, " GHz")
        self.slope = _dspin(cfg.slope_hz_per_s / 1e12, 1.0, 100.0, 3, 0.5, " THz/s")
        self.ramp = _dspin(cfg.ramp_time_s * 1e6, 5.0, 500.0, 1, 5.0, " µs")
        self.idle = _dspin(cfg.idle_time_s * 1e6, 0.0, 500.0, 1, 5.0, " µs")
        self.samples = _ispin(cfg.samples_per_chirp, 32, 1024)
        self.rate = _dspin(cfg.sample_rate_hz / 1e6, 0.5, 50.0, 2, 0.5, " MHz")
        self.chirps = _ispin(cfg.chirps_per_frame, 4, 1024)
        self.tx = _ispin(cfg.tx_elements, 1, 8)
        self.rx = _ispin(cfg.rx_elements, 1, 8)
        self.mimo = QComboBox()
        self.mimo.addItems(list(MIMO_MODES))
        self.mimo.setCurrentText(cfg.mimo_mode)
        self.power = _dspin(cfg.tx_power_dbm, -10.0, 30.0, 1, 1.0, " dBm")
        self.gain = _dspin(cfg.antenna_gain_dbi, 0.0, 30.0, 1, 1.0, " dBi")
        self.nf = _dspin(cfg.noise_figure_db, 0.0, 30.0, 1, 1.0, " dB")
        self.loss = _dspin(cfg.system_loss_db, 0.0, 20.0, 1, 1.0, " dB")
        self.bits = _ispin(cfg.adc_bits, 8, 16, " bit")
        self.frame_period = _dspin(cfg.frame_period_s * 1e3, 10.0, 500.0, 1, 5.0, " ms")
        self.order = _ispin(cfg.max_reflection_order, 0, 2)
        self.noise = QCheckBox("启用热噪声")
        self.noise.setChecked(cfg.enable_noise)

        self.derived = QLabel()
        self.derived.setWordWrap(True)
        self.derived.setStyleSheet("color:#8fd3ff; font-family:Consolas,monospace;")

        waveform = QGroupBox("波形")
        form = QFormLayout(waveform)
        for label, widget in (("起始频率", self.start_freq), ("调频斜率", self.slope),
                              ("扫频时间 Tr", self.ramp), ("空闲时间", self.idle),
                              ("采样点 Ns", self.samples), ("采样率 Fs", self.rate),
                              ("chirp 数 Nc", self.chirps)):
            form.addRow(label, widget)

        array = QGroupBox("阵列 / MIMO")
        form2 = QFormLayout(array)
        form2.addRow("TX 数", self.tx)
        form2.addRow("RX 数", self.rx)
        form2.addRow("MIMO 模式", self.mimo)

        frontend = QGroupBox("收发前端 / 帧")
        form3 = QFormLayout(frontend)
        for label, widget in (("发射功率", self.power), ("天线增益", self.gain),
                              ("噪声系数", self.nf), ("系统损耗", self.loss),
                              ("ADC 位宽", self.bits), ("帧周期", self.frame_period),
                              ("最大反射次数", self.order)):
            form3.addRow(label, widget)
        form3.addRow("", self.noise)

        layout = QVBoxLayout(self)
        layout.addWidget(waveform)
        layout.addWidget(array)
        layout.addWidget(frontend)
        layout.addWidget(QLabel("派生指标"))
        layout.addWidget(self.derived)
        layout.addStretch(1)

        for widget in (self.start_freq, self.slope, self.ramp, self.idle, self.rate,
                       self.power, self.gain, self.nf, self.loss, self.frame_period):
            widget.valueChanged.connect(self._emit)
        for widget in (self.samples, self.chirps, self.tx, self.rx, self.bits, self.order):
            widget.valueChanged.connect(self._emit)
        self.mimo.currentIndexChanged.connect(self._emit)
        self.noise.toggled.connect(self._emit)
        self._emit()

    # ------------------------------------------------------------------ #
    def set_config(self, cfg: RadarConfig) -> None:
        self.cfg = cfg
        self._updating = True
        self.start_freq.setValue(cfg.start_freq_hz / 1e9)
        self.slope.setValue(cfg.slope_hz_per_s / 1e12)
        self.ramp.setValue(cfg.ramp_time_s * 1e6)
        self.idle.setValue(cfg.idle_time_s * 1e6)
        self.samples.setValue(int(cfg.samples_per_chirp))
        self.rate.setValue(cfg.sample_rate_hz / 1e6)
        self.chirps.setValue(int(cfg.chirps_per_frame))
        self.tx.setValue(int(cfg.tx_elements))
        self.rx.setValue(int(cfg.rx_elements))
        self.mimo.setCurrentText(cfg.mimo_mode)
        self.power.setValue(cfg.tx_power_dbm)
        self.gain.setValue(cfg.antenna_gain_dbi)
        self.nf.setValue(cfg.noise_figure_db)
        self.loss.setValue(cfg.system_loss_db)
        self.bits.setValue(int(cfg.adc_bits))
        self.frame_period.setValue(cfg.frame_period_s * 1e3)
        self.order.setValue(int(cfg.max_reflection_order))
        self.noise.setChecked(cfg.enable_noise)
        self._updating = False
        self._emit()

    def _emit(self) -> None:
        if self._updating:
            return
        cfg = RadarConfig(
            start_freq_hz=self.start_freq.value() * 1e9,
            slope_hz_per_s=self.slope.value() * 1e12,
            ramp_time_s=self.ramp.value() * 1e-6,
            idle_time_s=self.idle.value() * 1e-6,
            samples_per_chirp=int(self.samples.value()),
            sample_rate_hz=self.rate.value() * 1e6,
            chirps_per_frame=int(self.chirps.value()),
            tx_elements=int(self.tx.value()),
            rx_elements=int(self.rx.value()),
            element_spacing_ratio=self.cfg.element_spacing_ratio,
            mimo_mode=self.mimo.currentText(),
            tx_power_dbm=self.power.value(),
            antenna_gain_dbi=self.gain.value(),
            noise_figure_db=self.nf.value(),
            system_loss_db=self.loss.value(),
            adc_bits=int(self.bits.value()),
            frame_period_s=self.frame_period.value() * 1e-3,
            enable_noise=self.noise.isChecked(),
            max_reflection_order=int(self.order.value()),
        )
        self.cfg = cfg
        self._refresh_derived()
        self.changed.emit(cfg)

    def _refresh_derived(self) -> None:
        cfg = self.cfg
        warnings: list[str] = []
        try:
            warnings = cfg.validate()
        except ValueError as exc:
            self.derived.setText(f"⚠ {exc}")
            self.derived.setStyleSheet("color:#ff6b6b; font-family:Consolas,monospace;")
            return
        self.derived.setStyleSheet("color:#8fd3ff; font-family:Consolas,monospace;")
        lines = [
            f"λ       {cfg.wavelength_m * 1e3:.3f} mm",
            f"dR      {cfg.range_resolution_m:.3f} m",
            f"Rmax    {cfg.max_range_m:.2f} m",
            f"vmax    ±{cfg.max_velocity_mps:.3f} m/s",
            f"dv      {cfg.velocity_resolution_mps:.4f} m/s",
            f"dAz     {math.degrees(cfg.angular_resolution_rad):.2f}°",
            f"帧长    {cfg.frame_duration_s * 1e3:.2f} ms",
            f"单帧    {cfg.frame_bytes / 1024:.0f} KiB",
        ]
        if warnings:
            lines += ["", *[f"⚠ {w}" for w in warnings]]
        self.derived.setText("\n".join(lines))


# --------------------------------------------------------------------------- #
# vehicle
# --------------------------------------------------------------------------- #
class VehiclePanel(QWidget):
    changed = Signal()
    mode_changed = Signal(str)

    def __init__(self, vehicle: Vehicle, parent=None) -> None:
        super().__init__(parent)
        self.vehicle = vehicle
        self._updating = False

        cfg = vehicle.config
        self.start_x = _dspin(vehicle.state.x, -500.0, 500.0, 2)
        self.start_y = _dspin(vehicle.state.y, -500.0, 500.0, 2)
        self.start_yaw = _dspin(math.degrees(vehicle.state.yaw), -180.0, 180.0, 1)
        self.speed = _dspin(cfg.path_speed_mps, 0.5, 40.0, 1, 0.5, " m/s")
        self.max_speed = _dspin(cfg.max_speed_mps, 1.0, 60.0, 1, 1.0, " m/s")
        self.wheelbase = _dspin(cfg.wheelbase_m, 1.0, 8.0, 2, 0.1, " m")
        self.mode = QComboBox()
        self.mode.addItems(["手动驾驶 (键盘)", "沿路径自动"])
        self.mode.setCurrentIndex(0 if vehicle.mode == "manual" else 1)
        self.path_label = QLabel()
        self.path_hint = QLabel("提示：路径模式下在画布 Ctrl+左键 添加路径点")
        self.path_hint.setWordWrap(True)
        self.path_hint.setStyleSheet("color:#7f8fa4; font-size:11px;")
        self.clear_path = QPushButton("清空路径")

        box = QGroupBox("车辆")
        form = QFormLayout(box)
        for label, widget in (("起点 X", self.start_x), ("起点 Y", self.start_y),
                              ("朝向", self.start_yaw), ("巡航速度", self.speed),
                              ("最大速度", self.max_speed), ("轴距", self.wheelbase)):
            form.addRow(label, widget)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(QLabel("驾驶模式"))
        layout.addWidget(self.mode)
        layout.addWidget(self.path_label)
        layout.addWidget(self.path_hint)
        layout.addWidget(self.clear_path)
        layout.addStretch(1)

        for widget in (self.start_x, self.start_y, self.start_yaw, self.speed,
                       self.max_speed, self.wheelbase):
            widget.valueChanged.connect(self._apply)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.clear_path.clicked.connect(self._on_clear_path)
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        self._updating = True
        path = self.vehicle.path
        count = 0 if path is None else len(path)
        self.path_label.setText(f"路径点：{count}")
        self.path_label.setStyleSheet("color:#c9d4e2;")
        self.mode.setCurrentIndex(0 if self.vehicle.mode == "manual" else 1)
        self._updating = False

    def _apply(self) -> None:
        if self._updating:
            return
        cfg = self.vehicle.config
        cfg.path_speed_mps = self.speed.value()
        cfg.max_speed_mps = self.max_speed.value()
        cfg.wheelbase_m = self.wheelbase.value()
        self.changed.emit()

    def _on_mode(self, index: int) -> None:
        if self._updating:
            return
        if index == 1 and self.vehicle.path is None:
            start = [self.vehicle.state.x, self.vehicle.state.y]
            self.vehicle.set_path([start, [start[0] + 40.0, start[1]]])
        elif index == 0:
            self.vehicle.clear_path()
        self.refresh()
        self.mode_changed.emit("path" if index == 1 else "manual")
        self.changed.emit()

    def _on_clear_path(self) -> None:
        self.vehicle.clear_path()
        self.refresh()
        self.changed.emit()

    def apply_start_pose(self) -> None:
        """Move the vehicle to the panel's start pose and reset it there."""
        self.vehicle.config.start = (self.start_x.value(), self.start_y.value(),
                                     self.start_yaw.value())
        self.vehicle.reset()
        self.changed.emit()


# --------------------------------------------------------------------------- #
# run control
# --------------------------------------------------------------------------- #
class SimPanel(QWidget):
    start_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()
    export_requested = Signal()
    multiplier_changed = Signal(float)
    settings_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.start_btn = QPushButton("▶ 开始仿真")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("⏹ 停止")
        self.reset_btn = QPushButton("⟲ 重置")
        self.export_btn = QPushButton("💾 导出数据")

        self.multiplier = QComboBox()
        for text, value in (("1×", 1.0), ("2×", 2.0), ("5×", 5.0), ("最大", 0.0)):
            self.multiplier.addItem(text, value)
        self.duration = _dspin(4.0, 0.08, 120.0, 2, 0.5, " s")
        self.seed = _ispin(42, 0, 2 ** 31 - 1)
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("TI DCA1000 原生", "ti_dca1000")
        self.layout_combo.addItem("简化交错 IQ", "interleaved_iq")
        self.truth = QCheckBox("导出真值点云 CSV")
        self.truth.setChecked(True)

        self.progress = QLabel("frame 0 / 0")
        self.progress.setStyleSheet("color:#ffd166; font-family:Consolas,monospace;")

        run_box = QGroupBox("仿真控制")
        run_form = QFormLayout(run_box)
        buttons = QHBoxLayout()
        for btn in (self.start_btn, self.pause_btn):
            buttons.addWidget(btn)
        run_form.addRow(buttons)
        buttons2 = QHBoxLayout()
        for btn in (self.stop_btn, self.reset_btn):
            buttons2.addWidget(btn)
        run_form.addRow(buttons2)
        run_form.addRow("倍速", self.multiplier)
        run_form.addRow("时长", self.duration)
        run_form.addRow("随机种子", self.seed)

        out_box = QGroupBox("导出")
        out_form = QFormLayout(out_box)
        out_form.addRow("布局", self.layout_combo)
        out_form.addRow("", self.truth)
        out_form.addRow(self.export_btn)
        out_form.addRow(self.progress)

        layout = QVBoxLayout(self)
        layout.addWidget(run_box)
        layout.addWidget(out_box)
        layout.addStretch(1)

        self.start_btn.clicked.connect(self.start_requested)
        self.pause_btn.clicked.connect(self.pause_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.reset_btn.clicked.connect(self.reset_requested)
        self.export_btn.clicked.connect(self.export_requested)
        self.multiplier.currentIndexChanged.connect(
            lambda: self.multiplier_changed.emit(float(self.multiplier.currentData()))
        )
        for widget in (self.duration, self.seed):
            widget.valueChanged.connect(self.settings_changed)

    # ------------------------------------------------------------------ #
    @property
    def time_multiplier(self) -> float:
        return float(self.multiplier.currentData())

    @property
    def layout_name(self) -> str:
        return str(self.layout_combo.currentData())

    def set_running(self, running: bool) -> None:
        for widget in (self.start_btn, self.reset_btn, self.export_btn):
            widget.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)

    def set_progress(self, frame: int, total: int, extra: str = "") -> None:
        text = f"frame {frame} / {total}"
        if extra:
            text += f"   {extra}"
        self.progress.setText(text)


# --------------------------------------------------------------------------- #
# scene stats
# --------------------------------------------------------------------------- #
class StatsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.label = QLabel("—")
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color:#93a4b8; font-family:Consolas,monospace;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addStretch(1)

    def update_stats(self, values: dict[str, Any] | Callable[[], dict[str, Any]]) -> None:
        data = values() if callable(values) else values
        self.label.setText("\n".join(f"{k:<16}{v}" for k, v in data.items()))
