"""Scene editor canvas: ``QGraphicsView`` based 2D editor and radar overlay.

World coordinates have ``+Y`` up (DESIGN.md section 2); Qt scene coordinates
have ``+Y`` down.  Everything crossing that boundary goes through
:func:`to_scene` / :func:`to_world` -- there is no global flip transform, so
text and handles stay upright.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Iterable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)

from ..core.materials import Material
from ..core.scene import Primitive, Scene
from ..core.vehicle import Vehicle

HANDLE_SIZE = 9.0                 # scene units
GRID_MINOR_M = 5.0
GRID_MAJOR_M = 25.0

BG = QColor("#11161d")
GRID_MINOR = QColor("#1d2530")
GRID_MAJOR = QColor("#28323f")
AXIS = QColor("#3d4b5c")
BOUNDS = QColor("#4a5a6e")
VEHICLE = QColor("#ffd166")
RADAR = QColor("#7bd7ff")
DETECTION = QColor("#ff5d5d")
TRUTH = QColor("#4ade80")
TEXT = QColor("#93a4b8")


def to_scene(x: float, y: float) -> QPointF:
    """World metres -> Qt scene units (Y is flipped)."""
    return QPointF(float(x), -float(y))


def to_world(point: QPointF) -> tuple[float, float]:
    """Qt scene units -> world metres."""
    return float(point.x()), -float(point.y())


def material_colour(material: Material | None) -> QColor:
    return QColor(material.color if material else "#8a939e")


# --------------------------------------------------------------------------- #
# primitive item
# --------------------------------------------------------------------------- #
class PrimitiveItem(QGraphicsItem):
    """Draggable / scalable rendering of one :class:`Primitive`."""

    def __init__(self, prim: Primitive, canvas: "SceneCanvas") -> None:
        super().__init__()
        self.prim = prim
        self.canvas = canvas
        self._drag: str | None = None
        self._press_scene = QPointF()
        self._snapshot: dict | None = None
        self._last_factor = 1.0
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10.0 + prim.layer)

    # --- geometry ---------------------------------------------------------
    def polygon(self) -> QPolygonF:
        outline = self.prim.outline()
        if outline is None:
            x, y = self.prim.params["x"], self.prim.params["y"]
            return QPolygonF([to_scene(x - 0.6, y), to_scene(x + 0.6, y),
                              to_scene(x, y + 0.6)])
        return QPolygonF([to_scene(px, py) for px, py in outline])

    def boundingRect(self) -> QRectF:
        return self.polygon().boundingRect().adjusted(
            -HANDLE_SIZE, -HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE
        )

    def handle_rect(self) -> QRectF:
        r = self.polygon().boundingRect()
        return QRectF(r.right() - HANDLE_SIZE / 2.0, r.top() - HANDLE_SIZE / 2.0,
                      HANDLE_SIZE, HANDLE_SIZE)

    # --- painting ---------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option
        material = self.canvas.material_of(self.prim)
        base = material_colour(material)
        reflectivity = material.rho if material else 0.5

        fill = QColor(base)
        fill.setAlphaF(0.18 + 0.62 * reflectivity)          # brighter == more reflective
        outline = QColor(base).lighter(140) if material else QColor("#c9d4e2")
        if self.isSelected():
            outline = QColor("#ffd166")

        pen = QPen(outline)
        pen.setCosmetic(True)
        pen.setWidthF(2.0 if self.isSelected() else 1.2)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(self.polygon())

        if self.isSelected():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#ffd166")))
            painter.drawRect(self.handle_rect())

        if self.canvas.show_labels:
            r = self.polygon().boundingRect()
            flag = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(TEXT))
            painter.drawText(QRectF(r.left(), r.bottom() + 1.0, max(r.width(), 30.0), 14.0),
                             int(flag), self.prim.name)

    # --- interaction ------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        inside = self.handle_rect().contains(event.pos())
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if inside
                       else Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.prim.locked:
            super().mousePressEvent(event)
            return
        self._press_scene = event.scenePos()
        self._snapshot = copy.deepcopy(self.prim.params)
        self._last_factor = 1.0
        self._drag = "scale" if self.handle_rect().contains(event.pos()) else "move"
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None or self._snapshot is None:
            super().mouseMoveEvent(event)
            return
        dx, dy = to_world(event.scenePos())
        px, py = to_world(self._press_scene)
        self.prim.params = copy.deepcopy(self._snapshot)

        if self._drag == "move":
            self.prim.translate(dx - px, dy - py)
        else:
            cx, cy = self.prim.centroid()
            d0 = max(math.hypot(px - cx, py - cy), 1e-3)
            d1 = math.hypot(dx - cx, dy - cy)
            factor = d1 / d0
            if factor > 1e-3:
                self.prim.scale_by(factor / self._last_factor)
                self._last_factor = factor
        self.prepareGeometryChange()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self._snapshot = None
            self.canvas.notify_scene_edited()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu()
        actions = {
            menu.addAction("锁定 / 解锁"): "lock",
            menu.addAction("上移一层"): "raise",
            menu.addAction("下移一层"): "lower",
            menu.addAction("复制"): "duplicate",
        }
        menu.addSeparator()
        actions[menu.addAction("删除")] = "delete"
        chosen = menu.exec(event.screenPos())
        if chosen is None:
            return
        self.canvas.handle_context_action(self.prim, actions[chosen])


# --------------------------------------------------------------------------- #
# canvas
# --------------------------------------------------------------------------- #
class SceneCanvas(QGraphicsView):
    """Interactive 2D editor showing the scene, the radar and the ego vehicle."""

    scene_edited = Signal(str)          # primitive id, "" for structural changes
    selection_changed = Signal(object)  # Primitive | None
    view_changed = Signal()
    key_state_changed = Signal()
    background_clicked = Signal(float, float, int)   # world x, world y, modifiers

    def __init__(self, scene: Scene, parent=None) -> None:
        super().__init__(parent)
        self.scene_model = scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(BG))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._items: dict[str, PrimitiveItem] = {}
        self.radar_enabled = False
        self.show_labels = True
        self.show_grid = True
        self.show_radar = True
        self.show_detections = True
        self.show_truth = True

        self._radar_pos = (0.0, 0.0)
        self._radar_yaw = 0.0
        self._radar_range = 0.0
        self._detections: list[tuple[float, float, float]] = []
        self._truth: list[tuple[float, float]] = []
        self._vehicle = None
        self._pressed_keys: set[int] = set()

        self.rebuild()

    # ------------------------------------------------------------------ #
    # model <-> items
    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        for item in self._items.values():
            self._scene.removeItem(item)
        self._items.clear()
        for prim in self.scene_model.primitives:
            if not prim.visible:
                continue
            item = PrimitiveItem(prim, self)
            self._scene.addItem(item)
            self._items[prim.id] = item
        self._refresh_extent()
        self.viewport().update()

    def _refresh_extent(self) -> None:
        width, height = self.scene_model.bounds
        self._scene.setSceneRect(QRectF(-5.0, -float(height) - 5.0,
                                        float(width) + 10.0, float(height) + 10.0))

    def item_for(self, prim: Primitive) -> PrimitiveItem | None:
        return self._items.get(prim.id)

    def material_of(self, prim: Primitive) -> Material | None:
        try:
            return self.scene_model.material(prim.material_id)
        except KeyError:
            return None

    def notify_scene_edited(self) -> None:
        prim = self.selected_primitive()
        self.viewport().update()
        self.scene_edited.emit(prim.id if prim else "")

    def selected_primitive(self) -> Primitive | None:
        for item in self._scene.selectedItems():
            if isinstance(item, PrimitiveItem):
                return item.prim
        return None

    def select_primitive(self, prim: Primitive | None) -> None:
        self._scene.clearSelection()
        if prim is None:
            self.selection_changed.emit(None)
            return
        item = self._items.get(prim.id)
        if item is not None:
            item.setSelected(True)
            self.ensureVisible(item, 40.0, 40.0)

    def add_primitive(self, kind: str, world_xy: tuple[float, float]) -> Primitive:
        x, y = world_xy
        params: dict = {}
        if kind == "circle":
            params = {"x": x, "y": y, "r": 2.0}
        elif kind == "rect":
            params = {"x": x, "y": y, "w": 6.0, "h": 3.0, "rot_deg": 0.0}
        elif kind == "polygon":
            params = {"points": [[x - 3.0, y - 2.0], [x + 3.0, y - 2.0], [x, y + 3.0]]}
        elif kind == "wall":
            params = {"points": [[x - 6.0, y], [x + 6.0, y]], "thickness": 0.4}
        elif kind == "point":
            params = {"x": x, "y": y}
        else:
            raise ValueError(f"unknown primitive kind {kind!r}")

        prim = Primitive(kind, params, material_id="concrete")
        self.scene_model.add(prim)
        self.rebuild()
        self.select_primitive(prim)
        self.notify_scene_edited()
        return prim

    def handle_context_action(self, prim: Primitive, action: str) -> None:
        if action == "delete":
            self.scene_model.remove(prim.id)
        elif action == "duplicate":
            clone = Primitive.from_dict(prim.to_dict())
            clone.id = ""
            clone.translate(3.0, -3.0)
            self.scene_model.add(clone)
        elif action == "lock":
            prim.locked = not prim.locked
        elif action in ("raise", "lower"):
            delta = 1 if action == "raise" else -1
            prim.layer = int(prim.layer) + delta
        self.rebuild()
        if action != "delete":
            self.select_primitive(prim)
        self.notify_scene_edited()

    # ------------------------------------------------------------------ #
    # overlays
    # ------------------------------------------------------------------ #
    def set_radar_view(self, position: tuple[float, float], yaw: float,
                       max_range_m: float, enabled: bool = True) -> None:
        self._radar_pos = (float(position[0]), float(position[1]))
        self._radar_yaw = float(yaw)
        self._radar_range = float(max_range_m)
        self.radar_enabled = bool(enabled)
        self.viewport().update()

    def set_detections(self, points: Iterable[tuple[float, float, float]],
                       truth: Iterable[tuple[float, float]] | None = None) -> None:
        self._detections = [(float(x), float(y), float(s)) for x, y, s in points]
        self._truth = [(float(x), float(y)) for x, y in (truth or [])]
        self.viewport().update()

    def set_vehicle(self, vehicle: Vehicle | None) -> None:
        self._vehicle = vehicle
        self.viewport().update()

    def clear_overlays(self) -> None:
        self._detections = []
        self._truth = []
        self.viewport().update()

    # ------------------------------------------------------------------ #
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, BG)
        if not self.show_grid:
            return

        left = math.floor(rect.left() / GRID_MINOR_M) * GRID_MINOR_M
        right = math.ceil(rect.right() / GRID_MINOR_M) * GRID_MINOR_M
        top = math.floor(rect.top() / GRID_MINOR_M) * GRID_MINOR_M
        bottom = math.ceil(rect.bottom() / GRID_MINOR_M) * GRID_MINOR_M

        pen_minor = QPen(GRID_MINOR)
        pen_minor.setCosmetic(True)
        pen_major = QPen(GRID_MAJOR)
        pen_major.setCosmetic(True)

        x = left
        while x <= right:
            painter.setPen(pen_major if abs(x % GRID_MAJOR_M) < 1e-6 else pen_minor)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += GRID_MINOR_M
        y = top
        while y <= bottom:
            painter.setPen(pen_major if abs(y % GRID_MAJOR_M) < 1e-6 else pen_minor)
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += GRID_MINOR_M

        width, height = self.scene_model.bounds
        pen = QPen(BOUNDS)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(to_scene(0.0, float(height)), to_scene(float(width), 0.0)))

        pen = QPen(AXIS)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(to_scene(-5.0, 0.0), to_scene(float(width) + 5.0, 0.0))
        painter.drawLine(to_scene(0.0, -5.0), to_scene(0.0, float(height) + 5.0))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        del rect

        if self.radar_enabled and self.show_radar and self._radar_range > 0.0:
            self._draw_radar(painter)

        if self.show_truth and self._truth:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(TRUTH))
            for x, y in self._truth:
                painter.drawEllipse(to_scene(x, y), 2.5, 2.5)

        if self.show_detections and self._detections:
            for x, y, snr in self._detections:
                colour = QColor(DETECTION)
                colour.setAlphaF(min(0.35 + 0.06 * max(snr, 0.0), 1.0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(colour))
                painter.drawEllipse(to_scene(x, y), 3.5, 3.5)

        if self._vehicle is not None:
            self._draw_vehicle(painter)

    def _draw_radar(self, painter: QPainter) -> None:
        x, y = self._radar_pos
        rng = self._radar_range
        half = math.pi / 2.0
        steps = 40
        pts = [to_scene(x, y)]
        for i in range(steps + 1):
            angle = self._radar_yaw - half + 2.0 * half * i / steps
            pts.append(to_scene(x + rng * math.cos(angle), y + rng * math.sin(angle)))

        fill = QColor(RADAR)
        fill.setAlphaF(0.055)
        pen = QPen(RADAR)
        pen.setCosmetic(True)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(QPolygonF(pts))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(RADAR))
        painter.drawEllipse(to_scene(x, y), 3.5, 3.5)

        boresight = to_scene(x + rng * math.cos(self._radar_yaw),
                             y + rng * math.sin(self._radar_yaw))
        pen = QPen(RADAR)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(to_scene(x, y), boresight)

    def _draw_vehicle(self, painter: QPainter) -> None:
        assert self._vehicle is not None
        corners = self._vehicle.corners()
        poly = QPolygonF([to_scene(px, py) for px, py in corners])
        pen = QPen(VEHICLE)
        pen.setCosmetic(True)
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 209, 102, 60)))
        painter.drawPolygon(poly)

        pos, yaw, _vel = self._vehicle.radar_pose()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(RADAR))
        painter.drawEllipse(to_scene(float(pos[0]), float(pos[1])), 3.0, 3.0)

        nose = to_scene(self._vehicle.state.x + 2.2 * math.cos(yaw),
                        self._vehicle.state.y + 2.2 * math.sin(yaw))
        pen = QPen(VEHICLE)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(to_scene(self._vehicle.state.x, self._vehicle.state.y), nose)

    # ------------------------------------------------------------------ #
    # navigation and keyboard
    # ------------------------------------------------------------------ #
    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)
        self.view_changed.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.itemAt(event.position().toPoint()) is None:
                wx, wy = to_world(event.scenePos())
                self.background_clicked.emit(wx, wy, int(event.modifiers().value))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)
        self.selection_changed.emit(self.selected_primitive())

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, PrimitiveItem):
            self.select_primitive(item.prim)
            self.selection_changed.emit(item.prim)
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        self._pressed_keys.add(event.key())
        self.key_state_changed.emit()
        if event.key() in _DRIVE_KEYS:
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
        self._pressed_keys.discard(event.key())
        self.key_state_changed.emit()
        if event.key() in _DRIVE_KEYS:
            event.accept()
            return
        super().keyReleaseEvent(event)

    def driving_input(self) -> tuple[float, float]:
        """``(throttle, steer)`` in ``[-1, 1]`` from the currently held keys."""
        keys = self._pressed_keys
        throttle = 0.0
        steer = 0.0
        if keys & {Qt.Key.Key_Up, Qt.Key.Key_W}:
            throttle += 1.0
        if keys & {Qt.Key.Key_Down, Qt.Key.Key_S}:
            throttle -= 1.0
        if keys & {Qt.Key.Key_Left, Qt.Key.Key_A}:
            steer += 1.0
        if keys & {Qt.Key.Key_Right, Qt.Key.Key_D}:
            steer -= 1.0
        if Qt.Key.Key_Space in keys:
            throttle = -1.0
        return throttle, steer

    def clear_keys(self) -> None:
        self._pressed_keys.clear()

    def fit_to_scene(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view_changed.emit()


_DRIVE_KEYS = {
    Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
    Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Space,
}
