"""RadarConfig derived-quantity and validation tests.

The expected numbers are computed from the closed-form formulas in
``docs/DESIGN.md`` section 5.1 with the exact speed of light, so a formula
mistake (e.g. using ``S*Tr`` instead of ``S*Ns/fs``) fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core.constants import C_LIGHT
from radar_sim.core.radar_config import MIMO_MODES, RadarConfig

# closed-form expectations for the IWR1843 preset
EXP_CENTER_FREQ_HZ = 77.0e9 + 0.5 * 29.982e12 * 60e-6          # 77.89946 GHz
EXP_LAMBDA_M = C_LIGHT / EXP_CENTER_FREQ_HZ                    # 3.8485 mm
EXP_VALID_BW_HZ = 29.982e12 * (256 / 10e6)                     # 767.54 MHz
EXP_RANGE_BIN_M = C_LIGHT / (2.0 * EXP_VALID_BW_HZ)            # 0.19529 m
EXP_MAX_RANGE_M = C_LIGHT * 10e6 / (2.0 * 29.982e12)           # 49.995 m
EXP_VMAX = EXP_LAMBDA_M / (4.0 * 160e-6)                       # 6.0132 m/s
EXP_DV = EXP_LAMBDA_M / (2.0 * 128 * 160e-6)                   # 0.093956 m/s


@pytest.fixture
def cfg() -> RadarConfig:
    return RadarConfig.iwr1843()


def test_defaults_match_iwr1843_preset(cfg: RadarConfig):
    assert cfg.start_freq_hz == pytest.approx(77.0e9)
    assert cfg.slope_hz_per_s == pytest.approx(29.982e12)
    assert cfg.chirp_period_s == pytest.approx(160e-6)
    assert cfg.samples_per_chirp == 256
    assert cfg.chirps_per_frame == 128
    assert cfg.virtual_elements == 12
    assert cfg.mimo_mode == "ideal"
    assert cfg.max_reflection_order == 1


def test_center_frequency_and_wavelength(cfg: RadarConfig):
    assert cfg.center_freq_hz == pytest.approx(EXP_CENTER_FREQ_HZ, rel=1e-12)
    assert cfg.wavelength_m == pytest.approx(EXP_LAMBDA_M, rel=1e-12)
    assert 3.84e-3 < cfg.wavelength_m < 3.86e-3


def test_valid_bandwidth_uses_sampling_window_not_ramp(cfg: RadarConfig):
    """Regression guard: resolution must use S*Ns/fs, never S*Tr."""
    assert cfg.valid_bandwidth_hz == pytest.approx(EXP_VALID_BW_HZ, rel=1e-12)
    ramp_bandwidth = cfg.slope_hz_per_s * cfg.ramp_time_s          # 1.799 GHz
    assert cfg.valid_bandwidth_hz < ramp_bandwidth
    assert cfg.valid_bandwidth_hz / ramp_bandwidth == pytest.approx(256 / 10e6 / 60e-6)


def test_range_quantities(cfg: RadarConfig):
    assert cfg.range_bin_m == pytest.approx(EXP_RANGE_BIN_M, rel=1e-12)
    # range resolution and bin spacing are the same thing by construction
    assert cfg.range_resolution_m == pytest.approx(cfg.range_bin_m, rel=1e-12)
    assert cfg.max_range_m == pytest.approx(EXP_MAX_RANGE_M, rel=1e-12)
    assert cfg.range_bins == 256
    # all Ns bins are usable (complex sampling), so the last bin sits next to Rmax
    assert cfg.range_bins * cfg.range_bin_m == pytest.approx(cfg.max_range_m, rel=1e-9)


def test_velocity_quantities(cfg: RadarConfig):
    assert cfg.max_velocity_mps == pytest.approx(EXP_VMAX, rel=1e-12)
    assert cfg.velocity_resolution_mps == pytest.approx(EXP_DV, rel=1e-12)
    assert cfg.chirps_per_tx == cfg.chirps_per_frame
    assert cfg.tx_chirp_period_s == cfg.chirp_period_s


def test_tdm_scales_velocity_by_tx_count(cfg: RadarConfig):
    ideal = cfg.with_updates(chirps_per_frame=126)
    tdm = ideal.with_updates(mimo_mode="tdm")
    assert tdm.chirps_per_frame % tdm.tx_elements == 0
    assert tdm.chirps_per_tx == 42
    assert tdm.tx_chirp_period_s == pytest.approx(ideal.chirp_period_s * ideal.tx_elements)
    # tdm is hardware faithful, but buys 3x less unambiguous velocity
    assert tdm.max_velocity_mps == pytest.approx(ideal.max_velocity_mps / ideal.tx_elements)
    # velocity RESOLUTION is unchanged: the coherent processing interval is identical
    assert tdm.velocity_resolution_mps == pytest.approx(ideal.velocity_resolution_mps)


def test_array_quantities(cfg: RadarConfig):
    assert cfg.element_spacing_m == pytest.approx(cfg.wavelength_m / 2.0, rel=1e-12)
    assert cfg.angular_resolution_rad == pytest.approx(2.0 / 12.0, rel=1e-12)
    assert np.degrees(cfg.angular_resolution_rad) == pytest.approx(9.5493, rel=1e-4)
    assert cfg.max_unambiguous_sin == pytest.approx(1.0)


def test_data_size(cfg: RadarConfig):
    assert cfg.samples_per_frame == 128 * 12 * 256
    assert cfg.frame_bytes == 4 * 128 * 12 * 256
    assert cfg.frame_bytes == 1_572_864
    assert cfg.frame_duration_s == pytest.approx(128 * 160e-6)
    assert cfg.frame_duration_s <= cfg.frame_period_s


def test_noise_power_is_positive_and_scales_with_bandwidth(cfg: RadarConfig):
    p_full = cfg.noise_power_w()
    p_half = cfg.noise_power_w(cfg.sample_rate_hz / 2)
    assert p_full > 0
    assert p_full == pytest.approx(2.0 * p_half, rel=1e-12)


def test_validate_accepts_defaults(cfg: RadarConfig):
    assert cfg.validate() == []


def test_validate_rejects_sampling_window_longer_than_ramp():
    with pytest.raises(ValueError, match="sampling window"):
        RadarConfig.iwr1843(samples_per_chirp=1024)


def test_validate_rejects_frame_period_too_short():
    with pytest.raises(ValueError, match="frame_period_s"):
        RadarConfig.iwr1843(chirps_per_frame=600)


def test_validate_rejects_bad_mimo_mode():
    with pytest.raises(ValueError, match="mimo_mode"):
        RadarConfig.iwr1843(mimo_mode="mimo").validate()


def test_validate_rejects_indivisible_tdm():
    with pytest.raises(ValueError, match="multiple of tx_elements"):
        RadarConfig.iwr1843(mimo_mode="tdm", strict=False).validate()


def test_fixed_for_tdm_produces_valid_config():
    fixed = RadarConfig.iwr1843(mimo_mode="tdm", strict=False).fixed_for_tdm()
    assert fixed.chirps_per_frame == 126
    fixed.validate()


def test_validate_warns_about_grating_lobes():
    warnings = RadarConfig.iwr1843(element_spacing_ratio=0.8).validate()
    assert any("grating lobes" in w for w in warnings)


def test_validate_warns_about_tdm_velocity_limit():
    warnings = RadarConfig.iwr1843(chirps_per_frame=120, mimo_mode="tdm", strict=False).validate()
    assert any("alias" in w for w in warnings)


def test_mimo_modes_constant():
    assert MIMO_MODES == ("ideal", "tdm")


def test_serialisation_roundtrip(cfg: RadarConfig):
    data = cfg.to_dict()
    restored = RadarConfig.from_dict(data)
    assert restored == cfg
    assert restored.center_freq_hz == pytest.approx(cfg.center_freq_hz)


def test_derived_block_is_json_serialisable(cfg: RadarConfig):
    import json

    derived = cfg.derived()
    for key in ("wavelength_m", "range_bin_m", "max_range_m", "max_velocity_mps",
                "velocity_resolution_mps", "angular_resolution_deg", "frame_bytes"):
        assert key in derived
    json.dumps(derived)                                        # must not raise
    assert derived["frame_bytes"] == cfg.frame_bytes


def test_frozen_config_is_immutable(cfg: RadarConfig):
    with pytest.raises(Exception):
        cfg.samples_per_chirp = 64                          # type: ignore[misc]
    assert cfg.with_updates(samples_per_chirp=64).samples_per_chirp == 64
