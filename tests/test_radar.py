"""Synthesis tests -- acceptance cases T1, T2, T3 and the Doppler symbol guard."""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core.detection import extract_pointcloud
from radar_sim.core.radar import path_amplitudes, path_snr_db, synthesize_frame
from radar_sim.core.radar_config import RadarConfig
from radar_sim.core.raytrace import trace_paths
from radar_sim.core.scene import Primitive, Scene
from radar_sim.core.scatterers import build_scatterers

RADAR = np.array([0.0, 0.0])


def _run(target_xy, *, radar_vel=(0.0, 0.0), cfg=None, seed=7, noise=True):
    cfg = cfg or RadarConfig.iwr1843(enable_noise=noise)
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": target_xy[0], "y": target_xy[1]}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, radar_vel=np.array(radar_vel), max_order=0)
    return synthesize_frame(paths, sc, scene, cfg, rng=np.random.default_rng(seed)), cfg, scene


# --------------------------------------------------------------------------- #
# T1 -- static single target
# --------------------------------------------------------------------------- #
def test_static_target_range_velocity_angle_t1():
    result, cfg, _scene = _run((10.0, 0.0))
    assert result.paths.count == 1

    cloud = extract_pointcloud(result.cube, cfg)
    assert cloud["range_m"].size >= 1, "CFAR found nothing"

    # the strongest detection must be the target
    best = int(np.argmax(cloud["power_db"]))
    assert abs(float(cloud["range_m"][best]) - 10.0) <= 2 * cfg.range_bin_m
    assert abs(float(cloud["velocity_mps"][best])) <= cfg.velocity_resolution_mps
    assert abs(float(cloud["azimuth_deg"][best])) <= 7.0


def test_static_target_has_no_doppler_shift():
    result, cfg, _scene = _run((10.0, 0.0), noise=False)
    from radar_sim.core.framing import range_doppler_map

    power = range_doppler_map(result.cube, cfg)
    i, _k = np.unravel_index(int(np.argmax(power)), power.shape)
    assert i == cfg.chirps_per_frame // 2, "zero velocity must sit on the centre chirp bin"


def test_noiseless_cube_peak_lands_on_the_expected_range_bin():
    result, cfg, _scene = _run((10.0, 0.0), noise=False)
    # a noiseless single tone has constant magnitude, so take an FFT, not |x|
    profile = np.abs(np.fft.fft(result.cube[0, 0, :]))
    k = int(np.argmax(profile))
    assert abs(k * cfg.range_bin_m - 10.0) <= cfg.range_bin_m


def test_noiseless_cube_is_a_pure_tone():
    result, _cfg, _scene = _run((10.0, 0.0), noise=False)
    magnitude = np.abs(result.cube)
    assert float(magnitude.std()) < 1e-12 * float(magnitude.mean())
    assert float(magnitude.mean()) == pytest.approx(float(result.amplitude[0]), rel=1e-9)


# --------------------------------------------------------------------------- #
# T2 -- moving target (radar motion makes the target move relatively)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", [-5.0, -2.0, 2.0, 5.0])
def test_moving_target_velocity_sign_and_magnitude_t2(speed: float):
    """Radar closing at +speed onto a target ahead reads -speed (approaching)."""
    result, cfg, _scene = _run((20.0, 0.0), radar_vel=(speed, 0.0))
    cloud = extract_pointcloud(result.cube, cfg)
    assert cloud["range_m"].size >= 1
    best = int(np.argmax(cloud["power_db"]))
    measured = float(cloud["velocity_mps"][best])
    assert measured == pytest.approx(-speed, abs=1.5 * cfg.velocity_resolution_mps)


def test_apart_from_the_range_lag_the_velocity_is_exact():
    result, cfg, _scene = _run((20.0, 0.0), radar_vel=(3.0, 0.0), noise=False)
    from radar_sim.core.framing import index_to_velocity, range_doppler_map

    power = range_doppler_map(result.cube, cfg)
    i, _k = np.unravel_index(int(np.argmax(power)), power.shape)
    assert index_to_velocity(i, cfg) == pytest.approx(-3.0, abs=cfg.velocity_resolution_mps)


# --------------------------------------------------------------------------- #
# T3 -- angle sweep
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle_deg", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_angle_sweep_t3(angle_deg: float):
    theta = np.deg2rad(angle_deg)
    result, cfg, _scene = _run((15.0 * np.cos(theta), 15.0 * np.sin(theta)))
    cloud = extract_pointcloud(result.cube, cfg, n_angle_fft=360)
    assert cloud["range_m"].size >= 1
    best = int(np.argmax(cloud["power_db"]))
    measured = float(cloud["azimuth_deg"][best])
    assert abs(measured - angle_deg) < np.degrees(cfg.angular_resolution_rad)
    assert np.sign(measured) == np.sign(angle_deg) or angle_deg == 0.0


def test_no_left_right_ambiguity():
    """Boresight is +X and the field of view is +-90 deg, so a target above the
    axis must read a positive azimuth and one below must read negative."""
    upper, cfg, _ = _run((8.0, 8.0), noise=False)          # +45 deg
    lower, _cfg2, _ = _run((8.0, -8.0), noise=False)       # -45 deg
    from radar_sim.core.detection import estimate_azimuth
    from radar_sim.core.framing import range_doppler_complex, range_doppler_map

    def measured_azimuth(result, config):
        power = range_doppler_map(result.cube, config)
        i, k = np.unravel_index(int(np.argmax(power)), power.shape)
        rd_cube = range_doppler_complex(result.cube, config)
        azimuth, _peak = estimate_azimuth(rd_cube, config, [i], [k], n_fft=360)
        return float(np.rad2deg(azimuth[0]))

    az_up = measured_azimuth(upper, cfg)
    az_down = measured_azimuth(lower, cfg)
    assert az_up == pytest.approx(45.0, abs=2.0)
    assert az_down == pytest.approx(-45.0, abs=2.0)


# --------------------------------------------------------------------------- #
# amplitudes / SNR
# --------------------------------------------------------------------------- #
def test_path_amplitude_follows_radar_equation():
    near, cfg, _ = _run((5.0, 0.0), noise=False)
    far, _cfg2, _ = _run((20.0, 0.0), noise=False)
    a_near = float(near.amplitude[0])
    a_far = float(far.amplitude[0])
    # 1/R^2 amplitude law: quadrupling the range gives 16x less amplitude
    assert a_near / a_far == pytest.approx(16.0, rel=1e-6)


def test_metal_returns_more_than_vegetation():
    def amp_of(material: str) -> float:
        cfg = RadarConfig.iwr1843(enable_noise=False)
        scene = Scene.empty(60.0, 60.0)
        scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id=material))
        sc = build_scatterers(scene)
        paths = trace_paths(scene, sc, RADAR, max_order=0)
        return float(path_amplitudes(paths, sc, scene, cfg)[0][0])

    metal = amp_of("metal")
    vegetation = amp_of("vegetation")
    assert 20.0 * np.log10(metal / vegetation) > 15.0


def test_transparent_material_gives_a_weaker_echo():
    def amp_of(material: str) -> float:
        cfg = RadarConfig.iwr1843(enable_noise=False)
        scene = Scene.empty(60.0, 60.0)
        scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id=material))
        sc = build_scatterers(scene)
        paths = trace_paths(scene, sc, RADAR, max_order=0)
        return float(path_amplitudes(paths, sc, scene, cfg)[0][0])

    metal = amp_of("metal")
    glass = amp_of("glass")           # tau = 0.85 -> only 15% reflects
    assert glass < 0.3 * metal


def test_noise_floor_matches_thermal_model():
    cfg = RadarConfig.iwr1843(enable_noise=True)
    scene = Scene.empty(10.0, 10.0)
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    result = synthesize_frame(paths, sc, scene, cfg, rng=np.random.default_rng(1))
    rms = float(np.sqrt(np.mean(np.abs(result.cube) ** 2)))
    assert rms == pytest.approx(np.sqrt(cfg.noise_power_w()), rel=0.15)


def test_noise_can_be_disabled():
    result, _cfg, _scene = _run((10.0, 0.0), noise=False)
    magnitude = np.abs(result.cube)
    assert float(magnitude.min()) > 0.0
    # no noise -> perfectly flat magnitude, i.e. a single complex exponential
    assert float(magnitude.std()) < 1e-12 * float(magnitude.mean())


def test_deterministic_with_the_same_seed_t9():
    a, cfg, _ = _run((10.0, 0.0), seed=123)
    b, _cfg2, _ = _run((10.0, 0.0), seed=123)
    assert np.array_equal(a.cube, b.cube)
    c, _cfg3, _ = _run((10.0, 0.0), seed=124)
    assert not np.array_equal(a.cube, c.cube)


def test_path_snr_increases_after_range_fft_matched_filter():
    result, cfg, _scene = _run((10.0, 0.0))
    per_sample = float(result.snr_db[0])
    # the range FFT buys roughly the processing gain of the window
    gain_db = 20.0 * np.log10(max(cfg.samples_per_chirp / 2.0, 1.0))
    assert per_sample + gain_db > 20.0


# --------------------------------------------------------------------------- #
# pruning / bookkeeping
# --------------------------------------------------------------------------- #
def test_pruning_keeps_the_strongest_paths():
    cfg = RadarConfig.iwr1843(enable_noise=False)
    scene = Scene.empty(200.0, 200.0)
    for i in range(30):
        scene.add(Primitive("point", {"x": 5.0 + 2.0 * i, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    result = synthesize_frame(paths, sc, scene, cfg, max_paths=5)
    assert result.paths.count == 5
    assert result.stats["paths_pruned"] == 25
    # the five kept paths must be the five nearest targets
    assert np.all(result.paths.equiv_range <= 5.0 + 2.0 * 5)


def test_stats_report_bounce_breakdown():
    result, _cfg, _ = _run((10.0, 0.0))
    assert result.stats["paths_used"] == 1
    assert result.stats["paths_by_bounce"] == {0: 1}
    assert "timings_ms" in result.stats


def test_truth_table_columns():
    result, _cfg, _ = _run((10.0, 0.0))
    truth = result.truth
    assert set(truth) >= {
        "scatterer", "owner", "path_type", "range_m", "azimuth_deg",
        "radial_vel_mps", "amplitude_db", "rcs_eff", "snr_db",
    }
    assert truth["path_type"][0] == "direct"
    assert truth["range_m"][0] == pytest.approx(10.0)
    assert truth["amplitude_db"][0] == pytest.approx(0.0)      # only path -> 0 dB


def test_truth_table_is_empty_for_empty_scene():
    cfg = RadarConfig.iwr1843(enable_noise=False)
    scene = Scene.empty(10.0, 10.0)
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR)
    result = synthesize_frame(paths, sc, scene, cfg)
    assert result.cube.shape == (128, 12, 256)
    assert result.truth["range_m"].size == 0


def test_cube_shape_is_guaranteed():
    result, cfg, _ = _run((10.0, 0.0))
    from radar_sim.core.framing import expected_cube_shape

    assert result.cube.shape == expected_cube_shape(cfg)


# --------------------------------------------------------------------------- #
# TDM MIMO mode
# --------------------------------------------------------------------------- #
def test_tdm_mode_places_returns_only_on_own_chirps():
    cfg = RadarConfig.iwr1843(enable_noise=False, chirps_per_frame=126, mimo_mode="tdm")
    assert cfg.chirps_per_frame % cfg.tx_elements == 0
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    result = synthesize_frame(paths, sc, scene, cfg)

    active = np.nonzero(np.abs(result.cube).sum(axis=(1, 2)) > 1e-30)[0]
    assert active.size > 0
    # virtual element k is illuminated only on chirps m with m % Nt == k // Nr
    for k in range(cfg.virtual_elements):
        rows = np.nonzero(np.abs(result.cube[:, k, :]).sum(axis=1) > 1e-30)[0]
        assert rows.size == cfg.chirps_per_frame // cfg.tx_elements
        assert np.all(rows % cfg.tx_elements == k // cfg.rx_elements)


def test_ideal_mode_fills_every_chirp_of_every_element():
    cfg = RadarConfig.iwr1843(enable_noise=False)
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    result = synthesize_frame(paths, sc, scene, cfg)
    assert np.all(np.abs(result.cube).sum(axis=2) > 1e-30)


def test_tdm_velocity_uses_the_effective_chirp_period():
    """A TDM frame spans the same CPI, so the velocity reading must not scale.

    1.5 m/s stays inside the TDM unambiguous window (lambda/(4*Nt*Tc) = 2.0).
    """
    for mode in ("ideal", "tdm"):
        cfg = RadarConfig.iwr1843(
            enable_noise=False, chirps_per_frame=126, mimo_mode=mode
        )
        scene = Scene.empty(60.0, 60.0)
        scene.add(Primitive("point", {"x": 20.0, "y": 0.0}, material_id="metal"))
        sc = build_scatterers(scene)
        paths = trace_paths(scene, sc, RADAR, radar_vel=np.array([1.5, 0.0]), max_order=0)
        result = synthesize_frame(paths, sc, scene, cfg)
        if mode == "tdm":
            # de-interleave onto one TX before the doppler FFT
            cube = result.cube[0:: cfg.tx_elements, : cfg.rx_elements, :]
            from radar_sim.core.framing import periodic_window

            cube = cube * periodic_window(cube.shape[0], "hann")[:, None, None]
            spec = np.fft.fftshift(np.fft.fft(cube, axis=0), axes=0)
            power = (np.abs(spec) ** 2).sum(axis=(1, 2))
            i = int(np.argmax(power))
            v_res = cfg.wavelength_m / (2.0 * cube.shape[0] * cfg.tx_chirp_period_s)
            measured = (i - cube.shape[0] / 2.0) * v_res
        else:
            from radar_sim.core.framing import range_doppler_map, velocity_axis

            power = range_doppler_map(result.cube, cfg)
            i, _k = np.unravel_index(int(np.argmax(power)), power.shape)
            measured = float(velocity_axis(cfg)[i])
        assert measured == pytest.approx(-1.5, abs=0.1), f"{mode} mode mis-scaled"
