"""PySide6 user interface (imports Qt -- keep it out of :mod:`radar_sim.core`)."""

from .canvas import PrimitiveItem, SceneCanvas, to_scene, to_world
from .main_window import MainWindow
from .panels import PropertyPanel, RadarPanel, SimPanel, StatsPanel, VehiclePanel
from .signal_view import SignalView

__all__ = [
    "MainWindow",
    "SceneCanvas",
    "PrimitiveItem",
    "to_scene",
    "to_world",
    "PropertyPanel",
    "RadarPanel",
    "VehiclePanel",
    "SimPanel",
    "StatsPanel",
    "SignalView",
]
