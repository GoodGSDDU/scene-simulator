"""Standard FFT axes and the canonical range-Doppler pipeline.

Every consumer (UI signal view, Python / Jupyter / MATLAB templates) must use
these functions so that bin <-> physical-unit mapping can never drift.

Formulas (DESIGN.md section 10.6)::

    R[k]     = k * c*fs/(2*S*Ns)                k = 0 .. Ns-1
    v[i]     = (i - Nc/2) * lambda/(2*Nc*Tc)    i = 0 .. Nc-1
    theta[k] = asin((k - Nfft/2) * lambda/(Nfft*d))

``fftshift`` moves the zero-frequency bin to 0-based index ``N/2``.
Windows are always *periodic* (DFT-even); never symmetric.
"""

from __future__ import annotations

import numpy as np

from .radar_config import RadarConfig

__all__ = [
    "periodic_window",
    "range_axis",
    "velocity_axis",
    "angle_axis",
    "index_to_range",
    "index_to_velocity",
    "index_to_angle",
    "assert_cube_shape",
    "empty_cube",
    "range_profile",
    "range_doppler_complex",
    "range_doppler_map",
    "angle_spectrum",
    "to_pointcloud_xy",
]


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def periodic_window(n: int, kind: str = "hann") -> np.ndarray:
    """Periodic (DFT-even) window of length ``n``.

    ``np.hanning(n)`` is *symmetric* and is wrong for FFT processing; this
    helper is the only place windows may be created.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("window length must be positive")
    if kind in ("rect", "boxcar", "none", "ones"):
        return np.ones(n, dtype=np.float64)
    t = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(t)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(t)
    if kind == "blackman":
        return 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2.0 * t)
    raise ValueError(f"unknown window kind: {kind!r}")


# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
def range_axis(cfg: RadarConfig) -> np.ndarray:
    """Range of every range-FFT bin, shape ``(Ns,)``, metres."""
    return np.arange(cfg.range_bins, dtype=np.float64) * cfg.range_bin_m


def velocity_axis(cfg: RadarConfig) -> np.ndarray:
    """Velocity of every doppler-FFT bin **after fftshift**, shape ``(Nc,)``."""
    i = np.arange(int(cfg.chirps_per_frame), dtype=np.float64)
    return (i - cfg.chirps_per_frame / 2.0) * cfg.velocity_resolution_mps


def angle_axis(n_fft: int, element_spacing_m: float, wavelength_m: float) -> np.ndarray:
    """Azimuth of every angle-FFT bin **after fftshift**, radians, ``(n_fft,)``."""
    n_fft = int(n_fft)
    k = np.arange(n_fft, dtype=np.float64)
    u = (k - n_fft / 2.0) / n_fft                       # cycles / element
    return np.arcsin(np.clip(u * wavelength_m / element_spacing_m, -1.0, 1.0))


def index_to_range(k: int | np.ndarray, cfg: RadarConfig) -> np.ndarray:
    return np.asarray(k, dtype=np.float64) * cfg.range_bin_m


def index_to_velocity(i: int | np.ndarray, cfg: RadarConfig) -> np.ndarray:
    return (np.asarray(i, dtype=np.float64) - cfg.chirps_per_frame / 2.0) * cfg.velocity_resolution_mps


def index_to_angle(k: int | np.ndarray, n_fft: int, cfg: RadarConfig) -> np.ndarray:
    u = (np.asarray(k, dtype=np.float64) - n_fft / 2.0) / n_fft
    return np.arcsin(np.clip(u * cfg.wavelength_m / cfg.element_spacing_m, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# cube helpers
# --------------------------------------------------------------------------- #
def expected_cube_shape(cfg: RadarConfig) -> tuple[int, int, int]:
    return (int(cfg.chirps_per_frame), int(cfg.virtual_elements), int(cfg.samples_per_chirp))


def assert_cube_shape(cube: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Guard the single-source-of-truth ``(Nc, Nv, Ns)`` convention."""
    arr = np.asarray(cube)
    expected = expected_cube_shape(cfg)
    if arr.shape != expected:
        raise ValueError(
            f"radar cube shape {arr.shape} != expected {expected} "
            f"(Nc, Nv, Ns) -- see DESIGN.md section 2.1"
        )
    return arr


def empty_cube(cfg: RadarConfig) -> np.ndarray:
    return np.zeros(expected_cube_shape(cfg), dtype=np.complex128)


# --------------------------------------------------------------------------- #
# canonical processing
# --------------------------------------------------------------------------- #
def range_profile(
    cube: np.ndarray,
    cfg: RadarConfig,
    *,
    window: str = "hann",
    coherent: bool = False,
) -> np.ndarray:
    """Range FFT magnitude, shape ``(Ns,)``.

    All ``Ns`` bins are kept: with complex (IQ) sampling the beat frequency is
    known to be non-negative, so the full ``fs`` span -- i.e. the whole
    ``c*fs/(2*S)`` range -- is unambiguous.
    """
    cube = assert_cube_shape(cube, cfg)
    w = periodic_window(cfg.samples_per_chirp, window)[None, None, :]
    spec = np.fft.fft(cube * w, axis=-1)
    if coherent:
        spec = spec.mean(axis=1)
    return np.sqrt((np.abs(spec) ** 2).sum(axis=(0, 1)))


def range_doppler_complex(
    cube: np.ndarray,
    cfg: RadarConfig,
    *,
    range_window: str = "hann",
    doppler_window: str = "hann",
) -> np.ndarray:
    """Complex range-Doppler cube, shape ``(Nc, Nv, Ns)``.

    The doppler axis is fftshifted (index ``Nc/2`` == 0 m/s), the range axis is
    not (index 0 == 0 m).  Each cell already carries the full coherent
    integration gain of both FFTs, so array snapshots for DOA must be taken
    from *this* cube rather than from a handful of raw chirps.
    """
    cube = assert_cube_shape(cube, cfg)
    wr = periodic_window(cfg.samples_per_chirp, range_window)[None, None, :]
    range_spec = np.fft.fft(cube * wr, axis=-1)
    wd = periodic_window(cfg.chirps_per_frame, doppler_window)[:, None, None]
    return np.fft.fftshift(np.fft.fft(range_spec * wd, axis=0), axes=0)


def range_doppler_map(
    cube: np.ndarray,
    cfg: RadarConfig,
    *,
    range_window: str = "hann",
    doppler_window: str = "hann",
) -> np.ndarray:
    """Non-coherently integrated range-Doppler power map, shape ``(Nc, Ns)``.

    Axis 0 is velocity (fftshifted, index ``Nc/2`` == 0 m/s) and axis 1 is
    range (index 0 == 0 m).
    """
    spec = range_doppler_complex(
        cube, cfg, range_window=range_window, doppler_window=doppler_window
    )
    return (np.abs(spec) ** 2).sum(axis=1)


def angle_spectrum(
    snapshot: np.ndarray,
    cfg: RadarConfig,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT beamforming of one array snapshot.

    Parameters
    ----------
    snapshot : (Nv,) complex -- the virtual array response at one RD cell.

    Returns
    -------
    power : (n_fft,) non-negative
    angle : (n_fft,) radians
    """
    snap = np.asarray(snapshot).reshape(-1)
    n_fft = int(n_fft or max(64, 8 * snap.size))
    w = periodic_window(snap.size, "hann")
    spec = np.fft.fftshift(np.fft.fft(snap * w, n_fft))
    return np.abs(spec) ** 2, angle_axis(n_fft, cfg.element_spacing_m, cfg.wavelength_m)


def to_pointcloud_xy(
    ranges: np.ndarray,
    azimuths_rad: np.ndarray,
    radar_pos: tuple[float, float],
    radar_yaw_rad: float = 0.0,
) -> np.ndarray:
    """Polar (range, azimuth) -> world XY, shape ``(N, 2)``.

    Azimuth is measured CCW from the radar's ``+X`` boresight rotated by
    ``radar_yaw_rad``.
    """
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    a = np.asarray(azimuths_rad, dtype=np.float64).reshape(-1) + float(radar_yaw_rad)
    return np.stack(
        [float(radar_pos[0]) + r * np.cos(a), float(radar_pos[1]) + r * np.sin(a)], axis=1
    )
