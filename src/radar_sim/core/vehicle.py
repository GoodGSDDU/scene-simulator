"""Vehicle model: kinematic bicycle with manual drive and pure-pursuit path following.

The radar is bolted to the vehicle at ``radar_offset_m`` in the body frame, so
the radar pose (position, yaw, velocity) comes straight from the vehicle state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["VehicleConfig", "VehicleState", "Vehicle"]

# short JSON names used by .scene.json -> dataclass field names
_CONFIG_ALIASES = {
    "length": "length_m",
    "width": "width_m",
    "wheelbase": "wheelbase_m",
    "max_speed": "max_speed_mps",
    "max_accel": "max_accel_mps2",
    "radar_offset": "radar_offset_m",
    "path_speed": "path_speed_mps",
}

ARRIVAL_RADIUS_M = 1.0


@dataclass
class VehicleConfig:
    length_m: float = 4.5
    width_m: float = 1.9
    wheelbase_m: float = 2.7
    max_speed_mps: float = 15.0
    max_accel_mps2: float = 3.0
    max_steer_deg: float = 35.0
    steer_rate_deg_s: float = 60.0
    radar_offset_m: tuple[float, float] = (1.5, 0.0)
    path_speed_mps: float = 6.0
    start: tuple[float, float, float] = (5.0, 10.0, 0.0)      # x, y, yaw_deg

    @property
    def max_steer_rad(self) -> float:
        return math.radians(self.max_steer_deg)

    @property
    def steer_rate_rad_s(self) -> float:
        return math.radians(self.steer_rate_deg_s)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["start"] = list(self.start)
        data["radar_offset_m"] = list(self.radar_offset_m)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VehicleConfig":
        """Accept both the dataclass names and the short JSON names.

        ``.scene.json`` (DESIGN.md section 4.4) uses ``wheelbase``,
        ``max_speed``, ``radar_offset`` ... while the dataclass carries explicit
        ``_m`` / ``_mps`` suffixes.
        """
        if not data:
            return cls()
        payload = {
            _CONFIG_ALIASES.get(key, key): value for key, value in data.items()
        }
        if "start" in payload:
            payload["start"] = tuple(payload["start"])
        if "radar_offset_m" in payload:
            payload["radar_offset_m"] = tuple(payload["radar_offset_m"])
        valid = {f for f in cls.__dataclass_fields__}           # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in valid})


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0                # radians, CCW from +X
    speed: float = 0.0              # m/s along the body +X
    steer: float = 0.0              # front wheel angle, radians

    def as_dict(self, frame: int | None = None, t_s: float = 0.0) -> dict[str, Any]:
        out: dict[str, Any] = {
            "t_s": round(float(t_s), 6),
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "phi_rad": round(self.yaw, 9),
            "speed_mps": round(self.speed, 6),
        }
        if frame is not None:
            out["frame"] = int(frame)
        return out


class Vehicle:
    """Kinematic bicycle model with two interchangeable control modes."""

    def __init__(
        self,
        config: VehicleConfig | None = None,
        state: VehicleState | None = None,
        path: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.config = config or VehicleConfig()
        if state is None:
            x, y, yaw_deg = self.config.start
            state = VehicleState(x=float(x), y=float(y),
                                 yaw=math.radians(float(yaw_deg)))
        self.state = state

        self.manual_throttle = 0.0        # -1 .. 1
        self.manual_steer = 0.0           # -1 .. 1
        self.target_speed = 0.0
        self.target_steer = 0.0
        self.arrived = False
        self._path: np.ndarray | None = None
        self._path_progress = 0.0         # arc length travelled along the path
        if path is not None:
            self.set_path(path)

    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        return "path" if self._path is not None else "manual"

    @property
    def path(self) -> np.ndarray | None:
        return None if self._path is None else self._path.copy()

    def set_path(self, points: Sequence[Sequence[float]] | None) -> None:
        """Set the follow path.  ``None`` / empty clears it.

        A single point is ambiguous and therefore rejected.
        """
        if points is None or len(points) == 0:
            self._path = None
            self._path_progress = 0.0
            self.arrived = False
            return
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("path must be a sequence of (x, y) points")
        if pts.shape[0] < 2:
            raise ValueError("a path needs at least two waypoints")
        self._path = pts
        self._path_progress = 0.0
        self.arrived = False

    def clear_path(self) -> None:
        self.set_path(None)

    def set_manual_input(self, throttle: float, steer: float) -> None:
        """``throttle``/``steer`` in ``[-1, 1]``."""
        self.manual_throttle = float(np.clip(throttle, -1.0, 1.0))
        self.manual_steer = float(np.clip(steer, -1.0, 1.0))

    def reset(self, state: VehicleState | None = None) -> None:
        if state is None:
            x, y, yaw_deg = self.config.start
            state = VehicleState(x=float(x), y=float(y),
                                 yaw=math.radians(float(yaw_deg)))
        self.state = state
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self.target_speed = 0.0
        self.target_steer = 0.0
        self.arrived = False
        self._path_progress = 0.0

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> VehicleState:
        """Advance the model by ``dt`` seconds."""
        if dt <= 0.0:
            return self.state
        cfg = self.config

        if self._path is not None:
            self._update_path_targets()
        else:
            self.target_speed = self.manual_throttle * cfg.max_speed_mps
            self.target_steer = self.manual_steer * cfg.max_steer_rad

        state = self.state
        max_dv = cfg.max_accel_mps2 * dt
        state.speed += float(np.clip(self.target_speed - state.speed, -max_dv, max_dv))
        state.speed = float(np.clip(state.speed, -cfg.max_speed_mps, cfg.max_speed_mps))

        max_ds = cfg.steer_rate_rad_s * dt
        state.steer += float(np.clip(self.target_steer - state.steer, -max_ds, max_ds))
        state.steer = float(np.clip(state.steer, -cfg.max_steer_rad, cfg.max_steer_rad))

        # yaw first, then position with the updated heading (midpoint-ish)
        state.yaw = _wrap_pi(state.yaw + state.speed / cfg.wheelbase_m
                             * math.tan(state.steer) * dt)
        state.x += state.speed * math.cos(state.yaw) * dt
        state.y += state.speed * math.sin(state.yaw) * dt
        return state

    # ------------------------------------------------------------------ #
    # pure pursuit
    # ------------------------------------------------------------------ #
    @property
    def lookahead_m(self) -> float:
        return max(3.0, 0.8 * abs(self.state.speed))

    def _update_path_targets(self) -> None:
        """Pure pursuit with a deceleration profile and a latched arrival.

        ``arrived`` is *sticky*: once the vehicle reaches the end of the path it
        stops and never resumes, otherwise overshooting past the last waypoint
        would make the goal check fail and the car would accelerate away again.
        """
        pts = self._path
        assert pts is not None
        pos = np.array([self.state.x, self.state.y])

        if self.arrived:
            self.target_speed = 0.0
            self.target_steer = 0.0
            return

        goal, remaining = self._lookahead_point(pos, self.lookahead_m)
        if remaining <= ARRIVAL_RADIUS_M or float(np.linalg.norm(pts[-1] - pos)) <= ARRIVAL_RADIUS_M:
            self.arrived = True
            self.target_speed = 0.0
            self.target_steer = 0.0
            return

        alpha = _wrap_pi(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - self.state.yaw)
        lookahead = max(float(np.linalg.norm(goal - pos)), 1e-3)
        delta = math.atan2(2.0 * self.config.wheelbase_m * math.sin(alpha), lookahead)
        self.target_steer = float(
            np.clip(delta, -self.config.max_steer_rad, self.config.max_steer_rad)
        )
        # brake profile: v = sqrt(2*a*s) so the car eases to a stop on the goal
        brake_speed = math.sqrt(2.0 * self.config.max_accel_mps2 * remaining)
        self.target_speed = float(min(self.config.path_speed_mps, brake_speed))

    def _lookahead_point(
        self, pos: np.ndarray, lookahead: float
    ) -> tuple[np.ndarray, float]:
        """Return ``(goal_point, remaining_arc_length_to_path_end)``."""
        pts = self._path
        assert pts is not None
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        best_d = np.inf
        best_s = 0.0
        for i in range(seg.shape[0]):
            length = seg_len[i]
            if length < 1e-12:
                continue
            t = float(np.clip((pos - pts[i]) @ seg[i] / (length * length), 0.0, 1.0))
            proj = pts[i] + t * seg[i]
            d = float(np.linalg.norm(proj - pos))
            if d < best_d:
                best_d = d
                best_s = cum[i] + t * length

        remaining = max(cum[-1] - best_s, 0.0)
        target_s = min(best_s + lookahead, cum[-1])
        index = int(np.searchsorted(cum, target_s, side="right") - 1)
        index = int(np.clip(index, 0, seg.shape[0] - 1))
        length = seg_len[index]
        if length < 1e-12:
            return pts[index], remaining
        t = (target_s - cum[index]) / length
        return pts[index] + t * seg[index], remaining

    # ------------------------------------------------------------------ #
    # derived poses
    # ------------------------------------------------------------------ #
    def radar_pose(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Return ``(position, yaw, velocity)`` of the radar phase centre."""
        off_x, off_y = self.config.radar_offset_m
        cos_y, sin_y = math.cos(self.state.yaw), math.sin(self.state.yaw)
        position = np.array([
            self.state.x + off_x * cos_y - off_y * sin_y,
            self.state.y + off_x * sin_y + off_y * cos_y,
        ])
        velocity = np.array([
            self.state.speed * cos_y,
            self.state.speed * sin_y,
        ])
        return position, self.state.yaw, velocity

    def corners(self) -> np.ndarray:
        """Four body corners in world coordinates (CCW)."""
        half_l = self.config.length_m / 2.0
        half_w = self.config.width_m / 2.0
        local = np.array([
            [half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w],
        ])
        cos_y, sin_y = math.cos(self.state.yaw), math.sin(self.state.yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        return local @ rot.T + np.array([self.state.x, self.state.y])

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "state": self.state.as_dict(),
            "mode": self.mode,
            "path": None if self._path is None else self._path.tolist(),
        }


def _wrap_pi(angle: float) -> float:
    """Wrap to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
