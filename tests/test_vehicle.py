"""Vehicle kinematics and pure-pursuit tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_sim.core.vehicle import Vehicle, VehicleConfig, VehicleState, _wrap_pi


def _vehicle(**overrides) -> Vehicle:
    return Vehicle(config=VehicleConfig(start=(0.0, 0.0, 0.0), **overrides))


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_default_config_matches_the_design():
    cfg = VehicleConfig()
    assert cfg.length_m == pytest.approx(4.5)
    assert cfg.width_m == pytest.approx(1.9)
    assert cfg.wheelbase_m == pytest.approx(2.7)
    assert cfg.max_speed_mps == pytest.approx(15.0)
    assert cfg.max_accel_mps2 == pytest.approx(3.0)
    assert cfg.max_steer_deg == pytest.approx(35.0)
    assert cfg.radar_offset_m == (1.5, 0.0)
    assert cfg.max_steer_rad == pytest.approx(math.radians(35.0))


def test_config_roundtrip():
    cfg = VehicleConfig(start=(3.0, 4.0, 90.0), radar_offset_m=(2.0, 0.5))
    restored = VehicleConfig.from_dict(cfg.to_dict())
    assert restored.start == (3.0, 4.0, 90.0)
    assert restored.radar_offset_m == (2.0, 0.5)
    assert VehicleConfig.from_dict(None) == VehicleConfig()


def test_start_pose_is_applied():
    vehicle = Vehicle(VehicleConfig(start=(3.0, 4.0, 30.0)))
    assert vehicle.state.x == pytest.approx(3.0)
    assert vehicle.state.y == pytest.approx(4.0)
    assert vehicle.state.yaw == pytest.approx(math.radians(30.0))
    assert vehicle.mode == "manual"


# --------------------------------------------------------------------------- #
# manual drive
# --------------------------------------------------------------------------- #
def test_acceleration_limit_is_respected():
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 0.0)
    vehicle.step(1.0)
    assert vehicle.state.speed == pytest.approx(3.0)            # max_accel * 1 s


def test_reaches_max_speed_and_does_not_exceed_it():
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 0.0)
    for _ in range(700):                                        # 7 s > 15/3 s
        vehicle.step(0.01)
    assert vehicle.state.speed == pytest.approx(15.0)


def test_braking_is_limited_too():
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 0.0)
    for _ in range(700):
        vehicle.step(0.01)
    assert vehicle.state.speed == pytest.approx(15.0)
    vehicle.set_manual_input(-1.0, 0.0)
    vehicle.step(0.5)
    assert vehicle.state.speed == pytest.approx(15.0 - 1.5)


def test_straight_line_distance():
    """Zero to 3 m/s in 1 s covers 1.5 m along +X."""
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 0.0)
    for _ in range(100):
        vehicle.step(0.01)
    assert vehicle.state.x == pytest.approx(1.5, abs=0.05)
    assert vehicle.state.y == pytest.approx(0.0, abs=1e-9)


def test_steering_curve_sign():
    vehicle = _vehicle()
    vehicle.set_manual_input(0.5, 1.0)          # steer left (positive)
    for _ in range(100):
        vehicle.step(0.01)
    assert vehicle.state.steer > 0.0
    assert vehicle.state.yaw > 0.0              # yaw increases CCW
    assert vehicle.state.y > 0.0

    other = _vehicle()
    other.set_manual_input(0.5, -1.0)
    for _ in range(100):
        other.step(0.01)
    assert other.state.yaw < 0.0
    assert other.state.y < 0.0


def test_steer_is_clamped_to_the_config():
    vehicle = _vehicle()
    vehicle.set_manual_input(0.2, 1.0)
    for _ in range(400):
        vehicle.step(0.01)
    assert vehicle.state.steer == pytest.approx(vehicle.config.max_steer_rad, rel=1e-6)


def test_steer_rate_limits_the_response():
    vehicle = _vehicle()
    vehicle.set_manual_input(0.1, 1.0)
    vehicle.step(0.1)                           # 60 deg/s -> 6 deg in 100 ms
    assert vehicle.state.steer == pytest.approx(math.radians(6.0), rel=1e-6)


def test_zero_step_is_a_noop():
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 1.0)
    before = vehicle.state.as_dict()
    vehicle.step(0.0)
    assert vehicle.state.as_dict() == before


def test_manual_input_is_clipped():
    vehicle = _vehicle()
    vehicle.set_manual_input(5.0, -9.0)
    assert vehicle.manual_throttle == pytest.approx(1.0)
    assert vehicle.manual_steer == pytest.approx(-1.0)


def test_reset_restores_the_start_pose():
    vehicle = _vehicle()
    vehicle.set_manual_input(1.0, 1.0)
    for _ in range(50):
        vehicle.step(0.02)
    vehicle.reset()
    assert vehicle.state.x == pytest.approx(0.0)
    assert vehicle.state.y == pytest.approx(0.0)
    assert vehicle.state.speed == pytest.approx(0.0)
    assert vehicle.state.steer == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# path following
# --------------------------------------------------------------------------- #
def test_set_path_requires_two_points():
    vehicle = _vehicle()
    with pytest.raises(ValueError):
        vehicle.set_path([[0.0, 0.0]])
    vehicle.set_path([[0.0, 0.0], [1.0, 0.0]])
    assert vehicle.mode == "path"
    # None / empty clears the path instead of raising
    vehicle.set_path(None)
    assert vehicle.mode == "manual"
    vehicle.set_path([[0.0, 0.0], [1.0, 0.0]])
    vehicle.set_path([])
    assert vehicle.mode == "manual"


def test_path_mode_drives_forward_and_reaches_the_lookahead():
    vehicle = _vehicle(path_speed_mps=6.0)
    vehicle.set_path([[0.0, 0.0], [50.0, 0.0]])
    for _ in range(200):
        vehicle.step(0.01)
    assert vehicle.state.x > 5.0
    assert vehicle.state.speed > 0.0
    assert abs(vehicle.state.y) < 0.2


def test_path_mode_stops_at_the_end():
    vehicle = _vehicle(path_speed_mps=6.0)
    vehicle.set_path([[0.0, 0.0], [12.0, 0.0]])
    for _ in range(800):
        vehicle.step(0.01)
    assert vehicle.arrived
    assert vehicle.state.x == pytest.approx(12.0, abs=1.6)
    assert vehicle.state.speed == pytest.approx(0.0, abs=1e-9)


def test_arrival_is_latched_and_does_not_resume():
    """Overshooting the goal must not re-arm the speed controller."""
    vehicle = _vehicle(path_speed_mps=8.0)
    vehicle.set_path([[0.0, 0.0], [6.0, 0.0]])
    for _ in range(1200):
        vehicle.step(0.01)
    assert vehicle.arrived
    assert vehicle.state.speed == pytest.approx(0.0, abs=1e-9)
    assert vehicle.state.x < 12.0, "vehicle drove past the end of the path"


def test_pure_pursuit_converges_onto_an_offset_path():
    """Start 4 m to the side of a straight path; the controller must close in."""
    vehicle = Vehicle(config=VehicleConfig(start=(0.0, 4.0, 0.0), path_speed_mps=6.0))
    vehicle.set_path([[0.0, 0.0], [60.0, 0.0]])
    distances = []
    for _ in range(1500):
        vehicle.step(0.01)
        distances.append(abs(vehicle.state.y))
        if vehicle.arrived:
            break
    assert min(distances[:300]) < abs(4.0) * 0.5, "controller is not converging"
    assert distances[-1] < 1.0, "final lateral error too large"


def test_pure_pursuit_follows_a_turn():
    vehicle = Vehicle(config=VehicleConfig(start=(2.0, 0.0, 0.0), path_speed_mps=5.0))
    vehicle.set_path([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]])
    for _ in range(3000):
        vehicle.step(0.01)
        if vehicle.arrived:
            break
    assert vehicle.arrived
    assert vehicle.state.x == pytest.approx(20.0, abs=2.5)
    assert vehicle.state.y == pytest.approx(20.0, abs=2.5)


def test_lookahead_grows_with_speed():
    vehicle = _vehicle()
    assert vehicle.lookahead_m == pytest.approx(3.0)            # standstill floor
    vehicle.state.speed = 10.0
    assert vehicle.lookahead_m == pytest.approx(8.0)


def test_config_accepts_short_json_names():
    cfg = VehicleConfig.from_dict({
        "start": [1.0, 2.0, 30.0],
        "length": 5.0,
        "width": 2.0,
        "wheelbase": 3.0,
        "max_speed": 20.0,
        "max_accel": 4.0,
        "radar_offset": [2.0, 0.0],
        "path_speed": 8.0,
    })
    assert cfg.length_m == pytest.approx(5.0)
    assert cfg.width_m == pytest.approx(2.0)
    assert cfg.wheelbase_m == pytest.approx(3.0)
    assert cfg.max_speed_mps == pytest.approx(20.0)
    assert cfg.max_accel_mps2 == pytest.approx(4.0)
    assert cfg.radar_offset_m == (2.0, 0.0)
    assert cfg.path_speed_mps == pytest.approx(8.0)


# --------------------------------------------------------------------------- #
# radar pose / body geometry
# --------------------------------------------------------------------------- #
def test_radar_pose_offset_is_rotated_with_the_body():
    vehicle = _vehicle()
    pos, yaw, vel = vehicle.radar_pose()
    assert pos == pytest.approx([1.5, 0.0])
    assert yaw == pytest.approx(0.0)
    assert vel == pytest.approx([0.0, 0.0])

    vehicle.state.yaw = math.pi / 2
    vehicle.state.speed = 2.0
    pos, yaw, vel = vehicle.radar_pose()
    assert pos == pytest.approx([0.0, 1.5], abs=1e-12)          # offset now along +Y
    assert yaw == pytest.approx(math.pi / 2)
    assert vel == pytest.approx([0.0, 2.0], abs=1e-12)


def test_corners_form_the_body_rectangle():
    vehicle = _vehicle()
    corners = vehicle.corners()
    assert corners.shape == (4, 2)
    span = corners.max(axis=0) - corners.min(axis=0)
    assert span[0] == pytest.approx(vehicle.config.length_m)
    assert span[1] == pytest.approx(vehicle.config.width_m)
    assert corners.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_to_dict_shape():
    vehicle = _vehicle()
    data = vehicle.to_dict()
    assert data["mode"] == "manual"
    assert data["path"] is None
    assert "config" in data and "state" in data


def test_wrap_pi():
    assert _wrap_pi(0.0) == pytest.approx(0.0)
    assert _wrap_pi(math.pi) == pytest.approx(-math.pi)        # [-pi, pi) convention
    assert _wrap_pi(3.0 * math.pi) == pytest.approx(-math.pi)
    assert _wrap_pi(-3.0 * math.pi) == pytest.approx(-math.pi)
    assert abs(_wrap_pi(math.pi + 0.1)) < math.pi
    assert _wrap_pi(0.1) == pytest.approx(0.1)


def test_state_serialisation_rounds():
    state = VehicleState(x=1.23456789, y=2.0, yaw=0.5, speed=3.0, steer=0.1)
    data = state.as_dict(frame=7, t_s=0.56)
    assert data["frame"] == 7
    assert data["t_s"] == pytest.approx(0.56)
    assert data["phi_rad"] == pytest.approx(0.5)
