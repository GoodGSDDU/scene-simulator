"""Physical constants and the project-wide sign/axis conventions.

**This module is the single source of truth for conventions.**
Any change here must be mirrored in ``docs/DESIGN.md`` section 2.1 and must
keep the T14-T17 tests green.

Conventions
-----------
* Geometry is a 2D top-down plane: ``+X`` right (east), ``+Y`` up (north).
* Azimuth ``theta`` is measured CCW from ``+X`` in ``(-pi, pi]``.
* Echo phase is ``phi = 4*pi*R/lambda`` -- phase *increases* with range.
* Radial velocity is positive when the scatterer moves **away** from the
  radar (range increasing).
* Array steering uses ``exp(+j * 2*pi * d * k * sin(theta) / lambda)``.
"""

from __future__ import annotations

# --- physical constants ---------------------------------------------------
C_LIGHT = 299_792_458.0          # m/s (exact)
BOLTZMANN = 1.380_649e-23        # J/K
T0_KELVIN = 290.0                # reference noise temperature

# --- convention guards (do not change) ------------------------------------
PHASE_SIGN = +1.0                # phi = PHASE_SIGN * 4*pi*R/lambda
VELOCITY_SIGN = +1.0             # +1 == "moving away"
STEERING_SIGN = +1.0             # +1 == exp(+j 2 pi d k sin(theta)/lambda)

# --- default radar preset (TI IWR1843, see DESIGN.md section 5.1) ---------
IWR1843_PRESET = {
    "start_freq_hz": 77.0e9,
    "slope_hz_per_s": 29.982e12,
    "ramp_time_s": 60.0e-6,
    "idle_time_s": 100.0e-6,
    "samples_per_chirp": 256,
    "sample_rate_hz": 10.0e6,
    "chirps_per_frame": 128,
    "tx_elements": 3,
    "rx_elements": 4,
    "element_spacing_ratio": 0.5,
    "mimo_mode": "ideal",
    "tx_power_dbm": 12.0,
    "antenna_gain_dbi": 12.0,
    "noise_figure_db": 12.0,
    "system_loss_db": 3.0,
    "adc_bits": 16,
    "frame_period_s": 0.08,
    "enable_noise": True,
    "max_reflection_order": 1,
}
