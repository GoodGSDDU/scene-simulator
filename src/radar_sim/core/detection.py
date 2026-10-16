"""Detection chain shared by the viewer and by every template.

Standard implementations only -- CA-CFAR with the analytic scaling factor
``alpha = N_ref * (Pfa^(-1/N_ref) - 1)``, FFT beamforming for azimuth, and a
greedy peak picker.  The reference project used ``mean + 1.5*sigma`` and a
declared-but-unused ``Pfa``; that is deliberately not repeated here.
"""

from __future__ import annotations

import numpy as np

from .framing import (
    angle_axis,
    assert_cube_shape,
    periodic_window,
    range_doppler_complex,
    velocity_axis,
)
from .radar_config import RadarConfig

__all__ = [
    "box_sum",
    "ca_cfar_alpha",
    "ca_cfar_2d",
    "pick_peaks",
    "estimate_azimuth",
    "extract_pointcloud",
]

_MIN_PFA = 1e-12
_MAX_PFA = 0.5


def box_sum(a: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Sum over a ``(2*kh+1, 2*kw+1)`` window, zero-padded, same shape out."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    padded = np.zeros((h + 2 * kh, w + 2 * kw), dtype=np.float64)
    padded[kh : kh + h, kw : kw + w] = a
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)))

    y0 = np.arange(h)
    y1 = y0 + 2 * kh + 1
    x0 = np.arange(w)
    x1 = x0 + 2 * kw + 1
    return (
        cum[np.ix_(y1, x1)]
        - cum[np.ix_(y0, x1)]
        - cum[np.ix_(y1, x0)]
        + cum[np.ix_(y0, x0)]
    )


def ca_cfar_alpha(n_ref: int, pfa: float) -> float:
    """CA-CFAR scaling factor for ``n_ref`` reference cells."""
    if n_ref <= 0:
        raise ValueError("n_ref must be positive")
    if not _MIN_PFA <= pfa <= _MAX_PFA:
        raise ValueError(f"pfa must be within [{_MIN_PFA}, {_MAX_PFA}]")
    return float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))


def ca_cfar_2d(
    power: np.ndarray,
    *,
    guard: tuple[int, int] = (2, 2),
    train: tuple[int, int] = (4, 4),
    pfa: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell-averaging CFAR on a 2D power map.

    Returns ``(mask, threshold, alpha)``.  The CUT is excluded (it sits inside
    the guard window), and the border where training cells would fall outside
    the map is forced to ``False``.
    """
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError("power map must be 2D")
    gh, gw = int(guard[0]), int(guard[1])
    th, tw = int(train[0]), int(train[1])
    if min(gh, gw, th, tw) < 0:
        raise ValueError("guard/train sizes must be non-negative")

    outer = box_sum(power, gh + th, gw + tw)
    inner = box_sum(power, gh, gw)
    n_outer = (2 * (gh + th) + 1) * (2 * (gw + tw) + 1)
    n_inner = (2 * gh + 1) * (2 * gw + 1)
    n_ref = n_outer - n_inner

    noise_mean = np.maximum(outer - inner, 0.0) / n_ref
    alpha = ca_cfar_alpha(n_ref, pfa)
    threshold = alpha * noise_mean

    mask = (power > threshold) & (noise_mean > 0.0)
    bh, bw = gh + th, gw + tw
    if bh:
        mask[:bh, :] = False
        mask[-bh:, :] = False
    if bw:
        mask[:, :bw] = False
        mask[:, -bw:] = False
    return mask, threshold, alpha


def pick_peaks(
    power: np.ndarray,
    mask: np.ndarray,
    *,
    guard: tuple[int, int] = (2, 2),
    max_peaks: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy non-maximum suppression over CFAR hits.

    Returns ``(i, k)`` index arrays sorted by descending power.
    """
    power = np.asarray(power, dtype=np.float64)
    hits = np.argwhere(np.asarray(mask, dtype=bool))
    if hits.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    values = power[hits[:, 0], hits[:, 1]]
    order = np.argsort(-values)
    hits = hits[order]
    values = values[order]

    gh, gw = int(guard[0]), int(guard[1])
    taken_i: list[int] = []
    taken_k: list[int] = []
    for i, k in hits:
        ok = True
        for ti, tk in zip(taken_i, taken_k):
            if abs(int(i) - ti) <= gh and abs(int(k) - tk) <= gw:
                ok = False
                break
        if ok:
            taken_i.append(int(i))
            taken_k.append(int(k))
            if len(taken_i) >= max_peaks:
                break
    return np.asarray(taken_i, dtype=int), np.asarray(taken_k, dtype=int)


def estimate_azimuth(
    rd_cube: np.ndarray,
    cfg: RadarConfig,
    doppler_index: int | np.ndarray,
    range_index: int | np.ndarray,
    *,
    n_fft: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT beamforming for one or more range-Doppler cells.

    ``rd_cube`` must be the complex cube from
    :func:`radar_sim.core.framing.range_doppler_complex`, so the snapshot
    already includes the coherent integration gain of both FFTs.  (Averaging a
    few raw chirps instead costs ~20 dB of gain and destroys the DOA at low
    per-sample SNR.)

    Returns ``(azimuth_rad, peak_power)``.
    """
    rd_cube = np.asarray(rd_cube)
    if rd_cube.ndim != 3:
        raise ValueError("rd_cube must be (Nc, Nv, Ns)")
    di = np.atleast_1d(np.asarray(doppler_index, dtype=int))
    ki = np.atleast_1d(np.asarray(range_index, dtype=int))
    if di.size != ki.size:
        raise ValueError("doppler_index and range_index must have the same length")

    n_fft = int(max(n_fft, 8 * cfg.virtual_elements))
    window = periodic_window(cfg.virtual_elements, "hann")
    axis = angle_axis(n_fft, cfg.element_spacing_m, cfg.wavelength_m)

    azimuth = np.zeros(di.size, dtype=np.float64)
    peak = np.zeros(di.size, dtype=np.float64)
    for idx in range(di.size):
        snap = rd_cube[di[idx], :, ki[idx]] * window
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(snap, n_fft))) ** 2
        best = int(np.argmax(spectrum))
        azimuth[idx] = float(axis[best])
        peak[idx] = float(spectrum[best])
    return azimuth, peak


def extract_pointcloud(
    cube: np.ndarray,
    cfg: RadarConfig,
    *,
    guard: tuple[int, int] = (2, 2),
    train: tuple[int, int] = (4, 4),
    pfa: float = 1e-3,
    n_angle_fft: int = 180,
    max_peaks: int = 512,
    radar_pos: tuple[float, float] = (0.0, 0.0),
    radar_yaw_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Full processing chain: range-Doppler -> CFAR -> DOA -> point cloud."""
    cube = assert_cube_shape(cube, cfg)
    rd_cube = range_doppler_complex(cube, cfg)
    rd = (np.abs(rd_cube) ** 2).sum(axis=1)
    mask, threshold, alpha = ca_cfar_2d(rd, guard=guard, train=train, pfa=pfa)
    di, ki = pick_peaks(rd, mask, guard=guard, max_peaks=max_peaks)

    if di.size == 0:
        empty = np.zeros(0)
        return {
            "range_m": empty, "velocity_mps": empty, "azimuth_deg": empty,
            "azimuth_rad": empty, "x": empty, "y": empty, "power_db": empty,
            "snr_db": empty, "doppler_index": np.zeros(0, dtype=int),
            "range_index": np.zeros(0, dtype=int),
            "rd_map": rd, "cfar_mask": mask, "cfar_threshold": threshold,
            "cfar_alpha": np.asarray(alpha),
        }

    azimuth, _peak_power = estimate_azimuth(rd_cube, cfg, di, ki, n_fft=n_angle_fft)

    ranges = ki.astype(np.float64) * cfg.range_bin_m
    velocities = velocity_axis(cfg)[di]
    power = rd[di, ki]
    noise = np.maximum(threshold[di, ki] / alpha, 1e-300)
    snr_db = 10.0 * np.log10(np.maximum(power, 1e-300) / noise)
    angles = azimuth + float(radar_yaw_rad)

    return {
        "range_m": ranges,
        "velocity_mps": velocities,
        "azimuth_deg": np.rad2deg(azimuth),
        "azimuth_rad": azimuth,
        "x": float(radar_pos[0]) + ranges * np.cos(angles),
        "y": float(radar_pos[1]) + ranges * np.sin(angles),
        "power_db": 10.0 * np.log10(np.maximum(power, 1e-300)),
        "snr_db": snr_db,
        "doppler_index": di,
        "range_index": ki,
        "rd_map": rd,
        "cfar_mask": mask,
        "cfar_threshold": threshold,
        "cfar_alpha": np.asarray(alpha),
    }
