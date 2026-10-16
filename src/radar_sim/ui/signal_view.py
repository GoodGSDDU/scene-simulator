"""Signal view tabs: raw IQ, range profile, range-Doppler map, point cloud.

Only the visible tab is redrawn, so the 12.5 Hz frame loop stays cheap.
The processing here calls the very same ``radar_sim.core`` helpers that the
shipped templates use, so what you see is what the templates will reproduce.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..core.detection import extract_pointcloud
from ..core.framing import (
    assert_cube_shape,
    periodic_window,
    range_doppler_map,
    range_profile,
    velocity_axis,
)
from ..core.radar_config import RadarConfig
from ..style import use_project_style

use_project_style()

_FG = "#c9d4e2"
_BG = "#0d1117"
_ACCENT = "#7bd7ff"
_DETECT = "#ff5d5d"
_TRUTH = "#4ade80"


def _style(axes) -> None:
    axes.set_facecolor(_BG)
    axes.tick_params(colors=_FG, labelsize=7)
    for spine in axes.spines.values():
        spine.set_color("#2a3441")
    axes.grid(True, color="#1d2530", linewidth=0.6)


class SignalView(QWidget):
    """Container with one matplotlib canvas per tab."""

    def __init__(self, cfg: RadarConfig, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.detections: dict[str, np.ndarray] = {}

        self.tabs = QTabWidget()
        self.status = QLabel("等待仿真 ...")
        self.status.setStyleSheet(f"color:{_FG}; font-family:Consolas,monospace;")

        self.figure = Figure(figsize=(6.0, 2.6), dpi=100, facecolor=_BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = {
            name: self.figure.add_subplot(1, 4, i + 1)
            for i, name in enumerate(("iq", "range", "rd", "cloud"))
        }
        self.figure.subplots_adjust(left=0.06, right=0.985, bottom=0.20, top=0.90,
                                    wspace=0.30)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs)
        layout.addWidget(self.canvas)
        layout.addWidget(self.status)

        for name, title in (("iq", "原始 IQ (chirp0/RX0)"),
                            ("range", "距离像 (dB)"),
                            ("rd", "距离-多普勒 (dB)"),
                            ("cloud", "点云 vs 真值")):
            self.axes[name].set_title(title, color=_FG, fontsize=8)
            self.tabs.addTab(QWidget(), title)
        self.tabs.setCurrentIndex(0)
        self.clear()

    # ------------------------------------------------------------------ #
    def set_config(self, cfg: RadarConfig) -> None:
        self.cfg = cfg
        self.clear()

    def clear(self) -> None:
        for axes in self.axes.values():
            axes.clear()
            _style(axes)
        self.axes["iq"].set_xlabel("采样点", color=_FG, fontsize=7)
        self.axes["range"].set_xlabel("距离 (m)", color=_FG, fontsize=7)
        self.axes["rd"].set_xlabel("距离 (m)", color=_FG, fontsize=7)
        self.axes["rd"].set_ylabel("速度 (m/s)", color=_FG, fontsize=7)
        self.axes["cloud"].set_xlabel("X (m)", color=_FG, fontsize=7)
        self.axes["cloud"].set_ylabel("Y (m)", color=_FG, fontsize=7)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def update_frame(
        self,
        cube: np.ndarray,
        *,
        radar_pos: tuple[float, float] = (0.0, 0.0),
        radar_yaw: float = 0.0,
        truth: dict[str, np.ndarray] | None = None,
        detect: bool = True,
        pfa: float = 1e-3,
    ) -> None:
        cfg = self.cfg
        cube = assert_cube_shape(cube, cfg)
        visible = self.tabs.currentIndex()

        if visible == 0:
            self._plot_iq(cube, cfg)
        elif visible == 1:
            self._plot_range(cube, cfg)
        elif visible == 2:
            self._plot_rd(cube, cfg, pfa if detect else None)
        else:
            self._plot_cloud(cube, cfg, radar_pos, radar_yaw, truth, pfa)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _plot_iq(self, cube: np.ndarray, cfg: RadarConfig) -> None:
        axes = self.axes["iq"]
        axes.clear()
        _style(axes)
        chirp = cube[0, 0, :]
        n_show = min(64, chirp.size)
        index = np.arange(n_show)
        axes.plot(index, chirp[:n_show].real, color=_ACCENT, linewidth=1.0, label="I")
        axes.plot(index, chirp[:n_show].imag, color="#ff9f43", linewidth=1.0, label="Q")
        axes.legend(fontsize=6, labelcolor=_FG, facecolor=_BG, edgecolor="#2a3441")
        axes.set_xlabel("采样点", color=_FG, fontsize=7)
        axes.set_title("原始 IQ (chirp0/RX0)", color=_FG, fontsize=8)

    def _plot_range(self, cube: np.ndarray, cfg: RadarConfig) -> None:
        axes = self.axes["range"]
        axes.clear()
        _style(axes)
        profile = 20.0 * np.log10(np.maximum(range_profile(cube, cfg), 1e-30))
        axis = np.arange(profile.size) * cfg.range_bin_m
        axes.plot(axis, profile, color=_ACCENT, linewidth=1.0)
        axes.set_xlim(0.0, cfg.max_range_m)
        axes.set_xlabel("距离 (m)", color=_FG, fontsize=7)
        axes.set_ylabel("dB", color=_FG, fontsize=7)
        axes.set_title("距离像 (dB)", color=_FG, fontsize=8)

    def _plot_rd(self, cube: np.ndarray, cfg: RadarConfig, pfa: float | None) -> None:
        axes = self.axes["rd"]
        axes.clear()
        _style(axes)
        rd = range_doppler_map(cube, cfg)
        rd_db = 10.0 * np.log10(np.maximum(rd, 1e-30))
        peak = float(rd_db.max())
        axes.imshow(rd_db, aspect="auto", cmap="turbo", origin="lower",
                    extent=(0.0, cfg.max_range_m, -cfg.max_velocity_mps,
                            cfg.max_velocity_mps),
                    vmin=peak - 45.0, vmax=peak, interpolation="nearest")
        axes.set_xlim(0.0, cfg.max_range_m)
        if pfa is not None:
            cloud = self.detections = extract_pointcloud(cube, cfg, pfa=pfa)
            if cloud["range_m"].size:
                axes.plot(cloud["range_m"], cloud["velocity_mps"], "o",
                          markersize=4, markerfacecolor="none",
                          markeredgecolor=_DETECT, markeredgewidth=0.9)
        axes.set_xlabel("距离 (m)", color=_FG, fontsize=7)
        axes.set_ylabel("速度 (m/s)", color=_FG, fontsize=7)
        axes.set_title("距离-多普勒 (dB)", color=_FG, fontsize=8)

    def _plot_cloud(
        self,
        cube: np.ndarray,
        cfg: RadarConfig,
        radar_pos: tuple[float, float],
        radar_yaw: float,
        truth: dict[str, np.ndarray] | None,
        pfa: float,
    ) -> None:
        axes = self.axes["cloud"]
        axes.clear()
        _style(axes)
        cloud = self.detections = extract_pointcloud(
            cube, cfg, pfa=pfa, radar_pos=radar_pos, radar_yaw_rad=radar_yaw
        )
        if cloud["x"].size:
            axes.scatter(cloud["x"], cloud["y"], s=14, c=cloud["snr_db"],
                         cmap="autumn", edgecolors="none", label="检测")
        if truth is not None and np.size(truth.get("range_m", ())):
            ranges = np.asarray(truth["range_m"])
            azimuths = np.deg2rad(np.asarray(truth["azimuth_deg"])) + radar_yaw
            axes.scatter(float(radar_pos[0]) + ranges * np.cos(azimuths),
                         float(radar_pos[1]) + ranges * np.sin(azimuths),
                         s=10, marker="+", c=_TRUTH, linewidths=0.8, label="真值")
        axes.plot([radar_pos[0]], [radar_pos[1]], marker="^", color=_ACCENT,
                  markersize=6)
        axes.set_xlabel("X (m)", color=_FG, fontsize=7)
        axes.set_ylabel("Y (m)", color=_FG, fontsize=7)
        axes.set_title("点云 vs 真值", color=_FG, fontsize=8)
        if axes.get_legend_handles_labels()[0]:
            axes.legend(fontsize=6, labelcolor=_FG, facecolor=_BG,
                        edgecolor="#2a3441")

    # ------------------------------------------------------------------ #
    def set_status(self, text: str) -> None:
        self.status.setText(text)
