"""FMCW IQ synthesis: turn traced echo paths into a raw ADC cube.

Per path the contribution is::

    x[m, k, n] = A * exp(j*2*pi*(fb/fs)*n)                # range (fast time)
                   * exp(j*2*pi*fd*Tc_tx*m_loc)           # doppler (slow time)
                   * exp(j*2*pi*d*k*sin(theta)/lambda)    # angle (array)
                   * exp(j*4*pi*R/lambda)                 # two-way carrier phase

with ``fb = 2*S*R/c`` and ``fd = 2*v_eff/lambda``.  ``Tc_tx`` is the chirp
repetition interval *as seen by one virtual element* -- ``Tc`` in ``ideal``
MIMO mode, ``Nt*Tc`` in ``tdm`` mode where a virtual element only fires every
``Nt`` chirps.

Amplitudes come from the radar equation with a 2D equivalent RCS
``sigma = |a_scatter * gain|^2 * seg_len * height``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import C_LIGHT
from .radar_config import RadarConfig
from .raytrace import EchoPaths
from .scene import Scene
from .scatterers import ScattererSet

__all__ = [
    "FrameResult",
    "synthesize_frame",
    "path_amplitudes",
    "path_snr_db",
    "add_thermal_noise",
]

_SYNTH_CHUNK = 64


@dataclass
class FrameResult:
    """Everything produced for one radar frame."""

    cube: np.ndarray                 # (Nc, Nv, Ns) complex128
    paths: EchoPaths                 # the paths that were actually synthesised
    amplitude: np.ndarray            # (P,) per-path IQ amplitude
    sigma_eff: np.ndarray            # (P,) 2D equivalent RCS (m^2)
    snr_db: np.ndarray               # (P,) per-path signal-to-noise ratio
    truth: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# amplitudes
# --------------------------------------------------------------------------- #
def path_amplitudes(
    paths: EchoPaths,
    scatterers: ScattererSet,
    scene: Scene,
    cfg: RadarConfig,
    height_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(amplitude, sigma_eff)`` for every path.

    ``amplitude`` is ``sqrt(Pr / P_ref)`` with ``P_ref = 1 W``; the absolute
    scale only matters for ``iq_scale_mode="fixed"``.
    """
    count = paths.count
    if count == 0:
        return np.zeros(0), np.zeros(0)

    a_scatter = np.ones(count, dtype=complex)
    mat_index = scatterers.material_index[paths.scatterer]
    material_ids = scene.material_ids()
    for mi in np.unique(mat_index):
        sel = mat_index == mi
        material = scene.materials[material_ids[int(mi)]]
        a_scatter[sel] = material.reflection_amplitude(
            np.arccos(np.clip(paths.cos_psi[sel], 0.0, 1.0))
        )

    a_total = a_scatter * paths.gain * paths.atten
    aperture = scatterers.seg_len[paths.scatterer] * float(height_m)
    sigma_eff = np.abs(a_total) ** 2 * aperture

    pt_w = (10.0 ** (cfg.tx_power_dbm / 10.0)) / 1000.0
    gain = 10.0 ** (cfg.antenna_gain_dbi / 10.0)
    loss = 10.0 ** (cfg.system_loss_db / 10.0)
    lam = cfg.wavelength_m
    rng = np.maximum(paths.equiv_range, 1e-3)

    pr = pt_w * gain * gain * lam * lam * sigma_eff / ((4.0 * np.pi) ** 3 * rng ** 4 * loss)
    return np.sqrt(pr), sigma_eff


def path_snr_db(amplitude: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Per-path SNR referenced to the thermal noise floor (per sample)."""
    if np.size(amplitude) == 0:
        return np.zeros(0)
    noise_amplitude = np.sqrt(cfg.noise_power_w())
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(amplitude, 1e-300) / noise_amplitude)


def _prune(
    amplitude: np.ndarray, max_paths: int, dynamic_range_db: float
) -> np.ndarray:
    """Keep the strongest paths down to ``dynamic_range_db`` below the peak."""
    count = amplitude.size
    if count == 0:
        return np.zeros(0, dtype=int)
    peak = float(amplitude.max())
    if peak <= 0.0:
        return np.zeros(0, dtype=int)
    keep = amplitude >= peak * (10.0 ** (-abs(dynamic_range_db) / 20.0))
    idx = np.nonzero(keep)[0]
    if max_paths > 0 and idx.size > max_paths:
        order = np.argpartition(-amplitude[idx], max_paths - 1)[:max_paths]
        idx = np.sort(idx[order])
    return idx


# --------------------------------------------------------------------------- #
# synthesis
# --------------------------------------------------------------------------- #
def _chirp_groups(cfg: RadarConfig) -> list[tuple[slice, slice, np.ndarray]]:
    """Return ``[(row_slice, element_slice, local_chirp_indices), ...]``.

    Both slices must stay ``slice`` objects so that ``cube[rows, elems]`` is a
    writable *view* -- fancy indexing with integer arrays would silently return
    a copy and the accumulation would be thrown away.

    In ``tdm`` mode TX ``t`` fires on chirps ``t, t+Nt, ...`` and only drives
    its own ``Nr`` virtual elements, so each group touches a contiguous block of
    the element axis.
    """
    total = int(cfg.chirps_per_frame)
    nt = int(cfg.tx_elements)
    nr = int(cfg.rx_elements)
    if cfg.mimo_mode == "tdm":
        groups = []
        for tx in range(nt):
            count = len(range(tx, total, nt))
            groups.append(
                (slice(tx, total, nt), slice(tx * nr, (tx + 1) * nr), np.arange(count))
            )
        return groups
    return [(slice(None), slice(0, nt * nr), np.arange(total))]


def synthesize_frame(
    paths: EchoPaths,
    scatterers: ScattererSet,
    scene: Scene,
    cfg: RadarConfig,
    *,
    height_m: float = 1.0,
    max_paths: int = 256,
    dynamic_range_db: float = 60.0,
    enable_noise: bool | None = None,
    rng: np.random.Generator | None = None,
    include_truth: bool = True,
) -> FrameResult:
    """Synthesise one raw ADC cube from traced paths."""
    t0 = time.perf_counter()
    amplitude, sigma_eff = path_amplitudes(paths, scatterers, scene, cfg, height_m=height_m)
    snr = path_snr_db(amplitude, cfg)

    keep = _prune(amplitude, max_paths, dynamic_range_db)
    n_pruned = int(amplitude.size - keep.size)

    cube = np.zeros(
        (int(cfg.chirps_per_frame), int(cfg.virtual_elements), int(cfg.samples_per_chirp)),
        dtype=np.complex128,
    )
    if keep.size:
        paths = paths.select(keep)
        amplitude = amplitude[keep]
        sigma_eff = sigma_eff[keep]
        snr = snr[keep]
        _accumulate(cube, paths, amplitude, cfg)
    t_synth = time.perf_counter()

    noise = enable_noise if enable_noise is not None else bool(cfg.enable_noise)
    if noise:
        _add_noise_inplace(cube, cfg, rng)
    t_noise = time.perf_counter()

    truth: dict[str, np.ndarray] = {}
    if include_truth:
        truth = _truth_table(paths, scatterers, sigma_eff, snr)

    stats = {
        "paths_total": int(paths.count) + n_pruned,
        "paths_used": int(paths.count),
        "paths_pruned": n_pruned,
        "paths_by_bounce": paths.counts_by_bounce(),
        "scatterers": len(scatterers),
        "timings_ms": {
            "amplitudes": (t_synth - t0) * 1e3,
            "accumulate": (t_noise - t_synth) * 1e3,
            "noise": (time.perf_counter() - t_noise) * 1e3,
        },
    }
    return FrameResult(cube=cube, paths=paths, amplitude=amplitude,
                       sigma_eff=sigma_eff, snr_db=snr, truth=truth, stats=stats)


def _accumulate(
    cube: np.ndarray, paths: EchoPaths, amplitude: np.ndarray, cfg: RadarConfig
) -> None:
    lam = cfg.wavelength_m
    n = np.arange(int(cfg.samples_per_chirp), dtype=np.float64)
    k = np.arange(int(cfg.virtual_elements), dtype=np.float64)

    fb = 2.0 * cfg.slope_hz_per_s * paths.equiv_range / C_LIGHT
    fd = 2.0 * paths.v_eff / lam
    carrier = amplitude * np.exp(1j * 4.0 * np.pi * paths.equiv_range / lam)

    e_range = np.exp(1j * (2.0 * np.pi * fb / cfg.sample_rate_hz)[:, None] * n[None, :])
    e_angle = np.exp(
        1j
        * (2.0 * np.pi * cfg.element_spacing_m * np.sin(paths.theta_rad) / lam)[:, None]
        * k[None, :]
    )
    doppler_rate = 2.0 * np.pi * fd * cfg.tx_chirp_period_s

    for rows, elements, local in _chirp_groups(cfg):
        e_doppler = np.exp(1j * doppler_rate[:, None] * local[None, :]) * carrier[:, None]
        view = cube[rows, elements, :]          # slices -> writable view
        angle_block = e_angle[:, elements]
        k_offset = int(elements.start or 0)
        del k_offset
        for start in range(0, paths.count, _SYNTH_CHUNK):
            stop = min(start + _SYNTH_CHUNK, paths.count)
            view += np.einsum(
                "pm,pn,pk->mkn",
                e_doppler[start:stop],
                e_range[start:stop],
                angle_block[start:stop],
                optimize=True,
            )


def _add_noise_inplace(
    cube: np.ndarray, cfg: RadarConfig, rng: np.random.Generator | None
) -> None:
    rng = rng or np.random.default_rng(0)
    amplitude = np.sqrt(cfg.noise_power_w() / 2.0)
    noise = rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
    cube += (amplitude * noise).astype(cube.dtype, copy=False)


def add_thermal_noise(
    cube: np.ndarray, cfg: RadarConfig, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return a noisy copy of ``cube`` (non-mutating convenience wrapper)."""
    out = np.array(cube, copy=True)
    _add_noise_inplace(out, cfg, rng)
    return out


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def _truth_table(
    paths: EchoPaths,
    scatterers: ScattererSet,
    sigma_eff: np.ndarray,
    snr_db: np.ndarray,
) -> dict[str, np.ndarray]:
    """Ideal point cloud (the second output channel, DESIGN.md section 7.5)."""
    if not paths.count:
        return {
            "scatterer": np.zeros(0, dtype=int),
            "owner": np.zeros(0, dtype=int),
            "path_type": np.zeros(0, dtype="<U8"),
            "range_m": np.zeros(0),
            "azimuth_deg": np.zeros(0),
            "radial_vel_mps": np.zeros(0),
            "amplitude_db": np.zeros(0),
            "rcs_eff": np.zeros(0),
            "snr_db": np.zeros(0),
        }
    amplitude = np.sqrt(np.maximum(sigma_eff, 0.0))
    with np.errstate(divide="ignore"):
        amp_db = 20.0 * np.log10(np.maximum(amplitude, 1e-300))
    peak = float(np.max(amp_db)) if amp_db.size else 0.0
    labels = np.array(["direct", "bounce1", "bounce2"], dtype="<U8")
    return {
        "scatterer": paths.scatterer.astype(int),
        "owner": scatterers.owner[paths.scatterer].astype(int),
        "path_type": labels[np.clip(paths.bounce, 0, 2)],
        "range_m": paths.equiv_range,
        "azimuth_deg": np.rad2deg(paths.theta_rad),
        "radial_vel_mps": paths.v_eff,
        "amplitude_db": amp_db - peak,
        "rcs_eff": sigma_eff,
        "snr_db": snr_db,
    }
