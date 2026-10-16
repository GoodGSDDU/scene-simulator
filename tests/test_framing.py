"""Framing tests -- covers acceptance cases T14, T15, T16 from DESIGN.md.

The synthetic cube is built here from the closed-form FMCW model *without*
calling any production synthesis code, so a bin-mapping bug in
:mod:`radar_sim.core.framing` cannot hide behind a matching bug in the
simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core.constants import C_LIGHT
from radar_sim.core.framing import (
    angle_axis,
    angle_spectrum,
    assert_cube_shape,
    empty_cube,
    expected_cube_shape,
    index_to_angle,
    index_to_range,
    index_to_velocity,
    periodic_window,
    range_axis,
    range_doppler_map,
    range_profile,
    to_pointcloud_xy,
    velocity_axis,
)
from radar_sim.core.radar_config import RadarConfig


@pytest.fixture
def cfg() -> RadarConfig:
    return RadarConfig.iwr1843()


def synth_single(cfg: RadarConfig, R: float, theta_rad: float, v: float, amp: float = 1.0) -> np.ndarray:
    """Analytic single-scatterer cube, shape ``(Nc, Nv, Ns)``.

    Follows ``x = exp(j2pi*(fb/fs)*n) * exp(j2pi*fd*Tc*m) * exp(j2pi*d*k*sin(th)/lambda)``
    (+v means moving away, which raises the chirp-to-chirp phase).
    """
    n = np.arange(cfg.samples_per_chirp)
    m = np.arange(cfg.chirps_per_frame)
    k = np.arange(cfg.virtual_elements)
    fb = 2.0 * cfg.slope_hz_per_s * R / C_LIGHT
    fd = 2.0 * v / cfg.wavelength_m
    phase = (
        2.0 * np.pi * (fb / cfg.sample_rate_hz) * n[None, None, :]
        + 2.0 * np.pi * fd * cfg.tx_chirp_period_s * m[:, None, None]
        + 2.0 * np.pi * cfg.element_spacing_m * k[None, :, None] * np.sin(theta_rad) / cfg.wavelength_m
    )
    return amp * np.exp(1j * phase)


# --------------------------------------------------------------------------- #
# T14 -- axes
# --------------------------------------------------------------------------- #
def test_periodic_window_is_periodic_not_symmetric():
    w = periodic_window(8, "hann")
    # the DFT-even form is the first N samples of the symmetric (N+1)-point window
    assert np.allclose(w, np.hanning(9)[:-1])
    assert w[0] == pytest.approx(0.0)
    assert not np.allclose(w, np.hanning(8)), "symmetric window leaked in"


def test_periodic_window_variants():
    assert np.allclose(periodic_window(5, "rect"), 1.0)
    assert periodic_window(16, "hamming").max() == pytest.approx(1.0)
    assert periodic_window(16, "blackman").min() >= -1e-12
    with pytest.raises(ValueError):
        periodic_window(8, "triangle")


def test_range_axis_t14(cfg: RadarConfig):
    ax = range_axis(cfg)
    assert ax.shape == (cfg.range_bins,)
    assert ax.shape == (cfg.samples_per_chirp,)
    assert ax[0] == 0.0
    assert np.allclose(np.diff(ax), cfg.range_bin_m)
    # the last bin sits one bin below Rmax -- no truncation to Ns/2
    assert ax[-1] == pytest.approx(cfg.max_range_m - cfg.range_bin_m, rel=1e-9)
    assert ax[-1] == pytest.approx(49.80, abs=0.02)


def test_velocity_axis_t14(cfg: RadarConfig):
    ax = velocity_axis(cfg)
    assert ax.shape == (cfg.chirps_per_frame,)
    assert ax[cfg.chirps_per_frame // 2] == 0.0
    steps = np.diff(ax)
    assert np.allclose(steps, steps[0])
    assert steps[0] == pytest.approx(cfg.velocity_resolution_mps, rel=1e-12)
    assert ax[0] == pytest.approx(-cfg.max_velocity_mps, rel=1e-3)
    assert ax[-1] < cfg.max_velocity_mps


def test_velocity_axis_rejects_linspace():
    """Regression guard for the reference project's linspace velocity axis."""
    for nc in (8, 16, 64, 128):
        cfg = RadarConfig.iwr1843(chirps_per_frame=nc)
        ax = velocity_axis(cfg)
        assert 0.0 in ax, "even-length axes must contain the zero-velocity bin"
        bad = np.linspace(-cfg.max_velocity_mps, cfg.max_velocity_mps, nc)
        assert not np.allclose(ax, bad), "velocity axis degenerated into linspace"


def test_angle_axis_covers_full_field_of_view(cfg: RadarConfig):
    ax = angle_axis(180, cfg.element_spacing_m, cfg.wavelength_m)
    assert ax.shape == (180,)
    assert ax[90] == pytest.approx(0.0)
    assert np.rad2deg(ax[0]) <= -60.0
    assert np.rad2deg(ax[-1]) >= 60.0


def test_index_helpers_are_inverse_of_axes(cfg: RadarConfig):
    assert index_to_range(10, cfg) == pytest.approx(range_axis(cfg)[10])
    assert index_to_velocity(96, cfg) == pytest.approx(velocity_axis(cfg)[96])
    ang = angle_axis(180, cfg.element_spacing_m, cfg.wavelength_m)
    assert index_to_angle(121, 180, cfg) == pytest.approx(ang[121])


# --------------------------------------------------------------------------- #
# cube shape guard
# --------------------------------------------------------------------------- #
def test_cube_shape_guard(cfg: RadarConfig):
    cube = empty_cube(cfg)
    assert cube.shape == expected_cube_shape(cfg) == (128, 12, 256)
    assert assert_cube_shape(cube, cfg).shape == (128, 12, 256)
    # the reference project's swapped convention must be rejected loudly
    with pytest.raises(ValueError, match=r"\(Nc, Nv, Ns\)"):
        assert_cube_shape(np.zeros((128, 256, 12)), cfg)


# --------------------------------------------------------------------------- #
# T15 -- bin alignment
# --------------------------------------------------------------------------- #
def test_range_bin_alignment_t15(cfg: RadarConfig):
    for R in (4.0, 12.0, 25.0, 41.3):
        k = int(np.argmax(range_profile(synth_single(cfg, R, 0.0, 0.0), cfg)))
        assert abs(index_to_range(k, cfg) - R) <= cfg.range_bin_m


def test_velocity_bin_alignment_t15(cfg: RadarConfig):
    for v in (-5.0, -1.5, 0.0, 2.0, 4.5):
        cube = synth_single(cfg, 20.0, 0.0, v)
        rd = range_doppler_map(cube, cfg)
        i, _ = np.unravel_index(int(np.argmax(rd)), rd.shape)
        assert abs(index_to_velocity(i, cfg) - v) <= cfg.velocity_resolution_mps


def test_range_doppler_peak_is_at_expected_cell_t15(cfg: RadarConfig):
    R, v = 12.0, 3.0
    rd = range_doppler_map(synth_single(cfg, R, 0.0, v), cfg)
    assert rd.shape == (cfg.chirps_per_frame, cfg.range_bins)
    i, k = np.unravel_index(int(np.argmax(rd)), rd.shape)
    assert abs(index_to_range(k, cfg) - R) <= cfg.range_bin_m
    assert abs(index_to_velocity(i, cfg) - v) <= cfg.velocity_resolution_mps


@pytest.mark.parametrize("theta_deg", [-45.0, -20.0, 0.0, 20.0, 45.0])
def test_angle_bin_alignment_t15(cfg: RadarConfig, theta_deg: float):
    theta = np.deg2rad(theta_deg)
    R = 20.0
    cube = synth_single(cfg, R, theta, 0.0)
    k_r = int(np.argmax(range_profile(cube, cfg)))
    snapshot = cube[:, :, k_r].mean(axis=0)
    power, axis = angle_spectrum(snapshot, cfg, n_fft=180)
    measured = float(axis[int(np.argmax(power))])
    assert abs(np.rad2deg(measured) - theta_deg) < 1.0
    assert abs(np.rad2deg(measured) - theta_deg) < np.degrees(cfg.angular_resolution_rad) / 4
    assert np.sign(measured) == np.sign(theta) or theta_deg == 0.0


def test_angle_spectrum_axis_matches_angle_axis(cfg: RadarConfig):
    snap = np.ones(cfg.virtual_elements, dtype=complex)         # broadside target
    power, axis = angle_spectrum(snap, cfg, n_fft=128)
    assert power.shape == axis.shape == (128,)
    assert abs(float(axis[int(np.argmax(power))])) < np.deg2rad(2.0)


# --------------------------------------------------------------------------- #
# T16 -- sign conventions
# --------------------------------------------------------------------------- #
def test_positive_velocity_means_moving_away_t16(cfg: RadarConfig):
    v = 5.0
    rd_pos = range_doppler_map(synth_single(cfg, 15.0, 0.0, +v), cfg)
    rd_neg = range_doppler_map(synth_single(cfg, 15.0, 0.0, -v), cfg)
    i_pos, _ = np.unravel_index(int(np.argmax(rd_pos)), rd_pos.shape)
    i_neg, _ = np.unravel_index(int(np.argmax(rd_neg)), rd_neg.shape)
    half = cfg.chirps_per_frame // 2
    assert i_pos > half, "+v must peak in the upper half of the fftshifted RD map"
    assert i_neg < half, "-v must peak in the lower half"
    assert index_to_velocity(i_pos, cfg) == pytest.approx(+v, abs=cfg.velocity_resolution_mps)
    assert index_to_velocity(i_neg, cfg) == pytest.approx(-v, abs=cfg.velocity_resolution_mps)


def test_phase_increases_with_range():
    """phi = 4*pi*R/lambda -- a larger range must mean a larger phase."""
    lam = 3.85e-3
    r1, r2 = 10.0, 10.5
    phi1 = 4.0 * np.pi * r1 / lam
    phi2 = 4.0 * np.pi * r2 / lam
    assert phi2 > phi1


# --------------------------------------------------------------------------- #
# point cloud conversion
# --------------------------------------------------------------------------- #
def test_to_pointcloud_xy_broadside(cfg: RadarConfig):
    pts = to_pointcloud_xy(np.array([10.0]), np.array([0.0]), (5.0, 5.0))
    assert pts[0] == pytest.approx([15.0, 5.0])


def test_to_pointcloud_xy_respects_yaw_and_sign():
    # azimuth is relative to boresight: az=0 with yaw=+90deg -> straight ahead along +Y
    pts = to_pointcloud_xy(np.array([10.0]), np.array([0.0]), (0.0, 0.0), np.pi / 2)
    assert pts[0] == pytest.approx([0.0, 10.0], abs=1e-9)
    # +30 deg is to the LEFT (CCW) of boresight
    pts = to_pointcloud_xy(np.array([10.0]), np.array([np.radians(30.0)]), (0.0, 0.0))
    assert pts[0][0] > 0.0 and pts[0][1] > 0.0
    # -30 deg is to the RIGHT (-Y)
    pts = to_pointcloud_xy(np.array([10.0]), np.array([-np.radians(30.0)]), (0.0, 0.0))
    assert pts[0][1] < 0.0
    # radar position offset is applied
    pts = to_pointcloud_xy(np.array([10.0]), np.array([0.0]), (100.0, -5.0))
    assert pts[0] == pytest.approx([110.0, -5.0])
