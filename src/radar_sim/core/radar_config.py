"""FMCW radar configuration plus every derived quantity.

All derived properties follow ``docs/DESIGN.md`` section 5.1.  The important
ones to get right (they were wrong in the reference project):

* ``range_resolution_m`` uses the **valid** bandwidth ``S*Ns/fs``, not ``S*Tr``.
* ``max_range_m`` uses ``c*fs/(2S)`` -- complex (IQ) sampling makes the full
  ``fs`` unambiguous for a non-negative beat frequency, so all ``Ns`` range
  bins are usable (not just ``Ns/2``).
* ``velocity`` uses ``chirps_per_tx`` / ``tx_chirp_period_s`` so TDM MIMO does
  not scale the velocity axis by ``tx_elements``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .constants import BOLTZMANN, C_LIGHT, IWR1843_PRESET, T0_KELVIN

MIMO_MODES = ("ideal", "tdm")

__all__ = ["RadarConfig", "MIMO_MODES"]


@dataclass(frozen=True)
class RadarConfig:
    # --- waveform ---------------------------------------------------------
    start_freq_hz: float = 77.0e9
    slope_hz_per_s: float = 29.982e12
    ramp_time_s: float = 60.0e-6
    idle_time_s: float = 100.0e-6
    samples_per_chirp: int = 256
    sample_rate_hz: float = 10.0e6
    chirps_per_frame: int = 128
    # --- antenna array ----------------------------------------------------
    tx_elements: int = 3
    rx_elements: int = 4
    element_spacing_ratio: float = 0.5      # d = ratio * lambda
    mimo_mode: str = "ideal"
    # --- RF front end -----------------------------------------------------
    tx_power_dbm: float = 12.0
    antenna_gain_dbi: float = 12.0
    noise_figure_db: float = 12.0
    system_loss_db: float = 3.0
    adc_bits: int = 16
    # --- frame / scenario -------------------------------------------------
    frame_period_s: float = 0.08
    enable_noise: bool = True
    max_reflection_order: int = 1

    # ------------------------------------------------------------------ #
    # derived: waveform
    # ------------------------------------------------------------------ #
    @property
    def chirp_period_s(self) -> float:
        return self.ramp_time_s + self.idle_time_s

    @property
    def center_freq_hz(self) -> float:
        return self.start_freq_hz + 0.5 * self.slope_hz_per_s * self.ramp_time_s

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.center_freq_hz

    @property
    def sampling_window_s(self) -> float:
        return self.samples_per_chirp / self.sample_rate_hz

    @property
    def valid_bandwidth_hz(self) -> float:
        """Bandwidth actually swept while the beat signal is being sampled."""
        return self.slope_hz_per_s * self.sampling_window_s

    @property
    def range_bin_m(self) -> float:
        """Range spacing between adjacent FFT bins: c*fs/(2*S*Ns)."""
        return C_LIGHT * self.sample_rate_hz / (
            2.0 * self.slope_hz_per_s * self.samples_per_chirp
        )

    @property
    def range_resolution_m(self) -> float:
        """c/(2*B_valid).  Equals :attr:`range_bin_m` by construction."""
        return C_LIGHT / (2.0 * self.valid_bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """c*fs/(2*S) -- unambiguous thanks to complex IQ sampling."""
        return C_LIGHT * self.sample_rate_hz / (2.0 * self.slope_hz_per_s)

    @property
    def range_bins(self) -> int:
        """All ``Ns`` bins are valid (complex sampling)."""
        return int(self.samples_per_chirp)

    # ------------------------------------------------------------------ #
    # derived: array
    # ------------------------------------------------------------------ #
    @property
    def element_spacing_m(self) -> float:
        return self.element_spacing_ratio * self.wavelength_m

    @property
    def virtual_elements(self) -> int:
        return int(self.tx_elements) * int(self.rx_elements)

    @property
    def angular_resolution_rad(self) -> float:
        return 2.0 / max(self.virtual_elements, 1)

    @property
    def max_unambiguous_sin(self) -> float:
        """|sin(theta)| limit before grating lobes / array ambiguity."""
        return min(1.0, 0.5 * self.wavelength_m / self.element_spacing_m)

    # ------------------------------------------------------------------ #
    # derived: doppler (MIMO aware)
    # ------------------------------------------------------------------ #
    @property
    def chirps_per_tx(self) -> int:
        if self.mimo_mode == "tdm":
            return int(self.chirps_per_frame) // int(self.tx_elements)
        return int(self.chirps_per_frame)

    @property
    def tx_chirp_period_s(self) -> float:
        """Chirp repetition interval as seen by a single virtual element."""
        if self.mimo_mode == "tdm":
            return self.chirp_period_s * int(self.tx_elements)
        return self.chirp_period_s

    @property
    def max_velocity_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.tx_chirp_period_s)

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.chirps_per_frame * self.chirp_period_s)

    # ------------------------------------------------------------------ #
    # derived: data size / noise
    # ------------------------------------------------------------------ #
    @property
    def samples_per_frame(self) -> int:
        return int(self.chirps_per_frame) * self.virtual_elements * int(self.samples_per_chirp)

    @property
    def frame_int16_count(self) -> int:
        return 2 * self.samples_per_frame

    @property
    def frame_bytes(self) -> int:
        return 4 * self.samples_per_frame

    def noise_power_w(self, receiver_bandwidth_hz: float | None = None) -> float:
        """Thermal noise power at the receiver input (W)."""
        bw = receiver_bandwidth_hz or self.sample_rate_hz
        return (
            BOLTZMANN
            * T0_KELVIN
            * bw
            * (10.0 ** (self.noise_figure_db / 10.0))
            * (10.0 ** (self.system_loss_db / 10.0))
        )

    @property
    def frame_duration_s(self) -> float:
        return self.chirps_per_frame * self.chirp_period_s

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Raise :class:`ValueError` on hard errors, return soft warnings."""
        if self.mimo_mode not in MIMO_MODES:
            raise ValueError(f"mimo_mode must be one of {MIMO_MODES}, got {self.mimo_mode!r}")
        if not (8 <= self.adc_bits <= 16):
            raise ValueError("adc_bits must be within [8, 16]")
        if self.samples_per_chirp < 8:
            raise ValueError("samples_per_chirp must be >= 8")
        if self.chirps_per_frame < 4:
            raise ValueError("chirps_per_frame must be >= 4")
        if self.tx_elements < 1 or self.rx_elements < 1:
            raise ValueError("tx/rx element counts must be >= 1")
        if self.sampling_window_s > self.ramp_time_s + 1e-12:
            raise ValueError(
                f"sampling window {self.sampling_window_s * 1e6:.1f} us exceeds ramp time "
                f"{self.ramp_time_s * 1e6:.1f} us -- raise sample_rate_hz or lower samples_per_chirp"
            )
        if self.frame_period_s + 1e-12 < self.frame_duration_s:
            raise ValueError(
                f"frame_period_s {self.frame_period_s * 1e3:.1f} ms < frame duration "
                f"{self.frame_duration_s * 1e3:.1f} ms -- lower chirps_per_frame or raise frame_period_s"
            )
        if self.mimo_mode == "tdm" and self.chirps_per_frame % self.tx_elements != 0:
            raise ValueError(
                f"chirps_per_frame ({self.chirps_per_frame}) must be a multiple of tx_elements "
                f"({self.tx_elements}) in tdm mode; call fixed_for_tdm() to auto-adjust"
            )
        if self.max_reflection_order not in (0, 1, 2):
            raise ValueError("max_reflection_order must be 0, 1 or 2")

        warnings: list[str] = []
        if self.element_spacing_ratio > 0.5 + 1e-9:
            warnings.append(
                f"element_spacing_ratio {self.element_spacing_ratio:.2f} > 0.5 creates grating lobes"
            )
        if self.mimo_mode == "tdm" and self.max_velocity_mps < 3.0:
            warnings.append(
                f"tdm mode limits |v| to {self.max_velocity_mps:.2f} m/s -- fast targets will alias"
            )
        return warnings

    def fixed_for_tdm(self) -> "RadarConfig":
        """Return a copy whose chirp count is divisible by ``tx_elements``."""
        n = int(self.chirps_per_frame)
        t = int(self.tx_elements)
        return replace(self, chirps_per_frame=max(t, (n // t) * t))

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def derived(self) -> dict[str, Any]:
        """Derived block written into ``meta.json``."""
        return {
            "wavelength_m": self.wavelength_m,
            "center_freq_hz": self.center_freq_hz,
            "valid_bandwidth_hz": self.valid_bandwidth_hz,
            "range_bin_m": self.range_bin_m,
            "range_resolution_m": self.range_resolution_m,
            "max_range_m": self.max_range_m,
            "max_velocity_mps": self.max_velocity_mps,
            "velocity_resolution_mps": self.velocity_resolution_mps,
            "angular_resolution_deg": math.degrees(self.angular_resolution_rad),
            "chirps_per_tx": self.chirps_per_tx,
            "tx_chirp_period_s": self.tx_chirp_period_s,
            "frame_duration_s": self.frame_duration_s,
            "frame_bytes": self.frame_bytes,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RadarConfig":
        valid = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})

    def with_updates(self, **kwargs: Any) -> "RadarConfig":
        return replace(self, **kwargs)

    @classmethod
    def iwr1843(cls, *, strict: bool = True, **overrides: Any) -> "RadarConfig":
        """Default preset (TI IWR1843), with optional overrides.

        ``strict=False`` builds the config without validating it, which is
        needed to construct intermediate states (e.g. switching to ``tdm``
        before calling :meth:`fixed_for_tdm`).
        """
        params = dict(IWR1843_PRESET)
        params.update(overrides)
        cfg = cls(**params)
        if strict:
            cfg.validate()
        return cfg
