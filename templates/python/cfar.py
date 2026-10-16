"""Detection chain: CA-CFAR, FFT beamforming, point-cloud extraction.

Self-contained (numpy only).  Standard formulas only -- see ``docs/DESIGN.md``
section 10.6.  In particular CFAR uses the analytic scaling factor

.. math::

    \\alpha = N_{ref} \\left( P_{fa}^{-1/N_{ref}} - 1 \\right)

rather than an ad-hoc ``mean + k*sigma`` threshold, so the requested false
alarm rate is actually meaningful.
"""

from __future__ import annotations

import numpy as np

from radar_io import RadarFile, periodic_window


# --------------------------------------------------------------------------- #
# CFAR
# --------------------------------------------------------------------------- #
def _box_sum(a: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Sum over a ``(2*kh+1, 2*kw+1)`` window (zero padded, same shape out)."""
    h, w = a.shape
    padded = np.zeros((h + 2 * kh, w + 2 * kw))
    padded[kh : kh + h, kw : kw + w] = a
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    y0 = np.arange(h)
    y1 = y0 + 2 * kh + 1
    x0 = np.arange(w)
    x1 = x0 + 2 * kw + 1
    return (cum[np.ix_(y1, x1)] - cum[np.ix_(y0, x1)]
            - cum[np.ix_(y1, x0)] + cum[np.ix_(y0, x0)])


def ca_cfar_alpha(n_ref: int, pfa: float) -> float:
    return float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))


def ca_cfar_2d(power, *, guard=(2, 2), train=(4, 4), pfa=1e-3):
    """Cell-averaging CFAR.

    Returns ``(mask, threshold, noise_mean, alpha)``.
    """
    power = np.asarray(power, dtype=np.float64)
    gh, gw = int(guard[0]), int(guard[1])
    th, tw = int(train[0]), int(train[1])

    outer = _box_sum(power, gh + th, gw + tw)
    inner = _box_sum(power, gh, gw)
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
    return mask, threshold, noise_mean, alpha


def pick_peaks(power, mask, *, guard=(2, 2), max_peaks=512):
    """Greedy non-maximum suppression over the CFAR hits."""
    hits = np.argwhere(np.asarray(mask, dtype=bool))
    if hits.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    power = np.asarray(power, dtype=np.float64)
    order = np.argsort(-power[hits[:, 0], hits[:, 1]])
    hits = hits[order]
    gh, gw = int(guard[0]), int(guard[1])

    kept_i: list[int] = []
    kept_k: list[int] = []
    for i, k in hits:
        if all(abs(int(i) - a) > gh or abs(int(k) - b) > gw
               for a, b in zip(kept_i, kept_k)):
            kept_i.append(int(i))
            kept_k.append(int(k))
            if len(kept_i) >= max_peaks:
                break
    return np.asarray(kept_i), np.asarray(kept_k)


# --------------------------------------------------------------------------- #
# DOA
# --------------------------------------------------------------------------- #
def estimate_azimuth(rd_cube, rf: RadarFile, doppler_index, range_index, *, n_fft=180):
    """FFT beamforming on the complex range-Doppler cube.

    ``rd_cube`` must come from :meth:`RadarFile.range_doppler_complex` so the
    snapshot carries the coherent gain of both FFTs.
    """
    di = np.atleast_1d(np.asarray(doppler_index, dtype=int))
    ki = np.atleast_1d(np.asarray(range_index, dtype=int))
    n_fft = int(max(n_fft, 8 * rf.n_elements))
    window = periodic_window(rf.n_elements, "hann")
    axis = rf.angle_axis(n_fft)

    azimuth = np.zeros(di.size)
    peak = np.zeros(di.size)
    for n in range(di.size):
        snap = rd_cube[di[n], :, ki[n]] * window
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(snap, n_fft))) ** 2
        best = int(np.argmax(spectrum))
        azimuth[n] = axis[best]
        peak[n] = spectrum[best]
    return azimuth, peak


# --------------------------------------------------------------------------- #
# full chain
# --------------------------------------------------------------------------- #
def extract_pointcloud(
    rf: RadarFile,
    frame: int = 0,
    *,
    guard=(2, 2),
    train=(4, 4),
    pfa=1e-3,
    n_angle_fft=180,
    max_peaks=512,
    radar_pos=(0.0, 0.0),
    radar_yaw_rad=0.0,
):
    """range-Doppler -> CFAR -> DOA -> point cloud.

    Returns a dict with ``range_m``, ``velocity_mps``, ``azimuth_deg``, ``x``,
    ``y``, ``power_db``, ``snr_db`` plus the intermediate ``rd_map`` and
    ``cfar_mask``.
    """
    rd_cube = rf.range_doppler_complex(frame)
    rd = (np.abs(rd_cube) ** 2).sum(axis=1)
    mask, threshold, noise_mean, alpha = ca_cfar_2d(
        rd, guard=guard, train=train, pfa=pfa
    )
    di, ki = pick_peaks(rd, mask, guard=guard, max_peaks=max_peaks)

    empty = np.zeros(0)
    if di.size == 0:
        return {
            "range_m": empty, "velocity_mps": empty, "azimuth_deg": empty,
            "x": empty, "y": empty, "power_db": empty, "snr_db": empty,
            "rd_map": 10 * np.log10(np.maximum(rd, 1e-30)), "cfar_mask": mask,
            "cfar_alpha": np.asarray(alpha), "doppler_index": di, "range_index": ki,
        }

    azimuth, _peak = estimate_azimuth(rd_cube, rf, di, ki, n_fft=n_angle_fft)

    r_axis = rf.range_axis()
    v_axis = rf.velocity_axis()
    ranges = r_axis[ki]
    velocities = v_axis[di]
    power = rd[di, ki]
    noise = np.maximum(noise_mean[di, ki], 1e-300)
    angles = azimuth + float(radar_yaw_rad)

    return {
        "range_m": ranges,
        "velocity_mps": velocities,
        "azimuth_deg": np.rad2deg(azimuth),
        "azimuth_rad": azimuth,
        "x": float(radar_pos[0]) + ranges * np.cos(angles),
        "y": float(radar_pos[1]) + ranges * np.sin(angles),
        "power_db": 10.0 * np.log10(np.maximum(power, 1e-30)),
        "snr_db": 10.0 * np.log10(np.maximum(power / noise, 1e-300)),
        "doppler_index": di,
        "range_index": ki,
        "rd_map": 10.0 * np.log10(np.maximum(rd, 1e-30)),
        "cfar_mask": mask,
        "cfar_alpha": np.asarray(alpha),
    }
