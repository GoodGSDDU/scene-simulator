"""Read a 2D Radar Scene Simulator dataset (``signal.bin`` + ``meta.json``).

Self-contained: only ``numpy`` is required.  Everything about the byte layout
comes from ``meta.json``, and the axis formulas are the contract from
``docs/DESIGN.md`` sections 2.1 / 10.6 -- keep them in sync if you change the
simulator.

Quick start
-----------

.. code-block:: python

    from radar_io import RadarFile

    rf = RadarFile("dataset/demo")
    print(rf.summary())
    cube = rf.frame(0)                          # (Nc, Nv, Ns) complex128
    rd_db, r_axis, v_axis = rf.range_doppler(0)  # (Nc, Ns) dB, metres, m/s

Conventions (single source of truth, mirrors DESIGN.md 2.1)
----------------------------------------------------------
* cube axes are ``(chirp, virtual_element, sample)``
* echo phase is ``4*pi*R/lambda`` -> phase grows with range
* positive radial velocity means **moving away**; after ``fftshift`` the peak
  of a receding target sits in the **upper** half of the range-Doppler map
* windows are always *periodic* (DFT-even); never symmetric
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def periodic_window(n: int, kind: str = "hann") -> np.ndarray:
    """Periodic (DFT-even) window -- ``np.hanning(n)`` would be symmetric."""
    if kind in ("rect", "boxcar", "none", "ones"):
        return np.ones(n)
    t = 2.0 * np.pi * np.arange(n) / n
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(t)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(t)
    if kind == "blackman":
        return 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2.0 * t)
    raise ValueError(f"unknown window {kind!r}")


# --------------------------------------------------------------------------- #
# unpacking
# --------------------------------------------------------------------------- #
def unpack_frame(raw: bytes, shape, layout: str) -> np.ndarray:
    """Bytes -> ``(Nc, Nv, Ns)`` complex128.

    ``ti_dca1000`` (default)
        per chirp, elements outer / samples inner, then every 2 complex samples
        packed as ``[Re(v0), Re(v1), Im(v0), Im(v1), ...]`` (TI's official
        ``readDCA1000.m`` layout).
    ``interleaved_iq``
        plain ``[Re, Im]`` per sample.
    """
    nc, nv, ns = (int(v) for v in shape)
    flat = nv * ns
    data = np.frombuffer(raw, dtype="<i2")
    if data.size != nc * flat * 2:
        raise ValueError(f"expected {nc * flat * 2} int16 values, got {data.size}")

    if layout == "interleaved_iq":
        arr = data.reshape(nc, nv, ns, 2).astype(np.float64)
        return arr[..., 0] + 1j * arr[..., 1]

    lin = data.reshape(nc, flat * 2).astype(np.float64)
    v = np.empty((nc, flat), dtype=np.complex128)
    v[:, 0::2] = lin[:, 0::4] + 1j * lin[:, 2::4]
    v[:, 1::2] = lin[:, 1::4] + 1j * lin[:, 3::4]
    return v.reshape(nc, nv, ns)


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
class RadarFile:
    """A dataset folder produced by ``DatasetWriter`` / the GUI export."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))
        self.signal = self.meta["signal"]
        self.radar = self.meta["radar"]

        self.layout = self.signal.get("layout", "interleaved_iq")
        self.shape = tuple(int(v) for v in self.signal["shape_per_frame"])
        self.frame_bytes = int(self.signal["frame_bytes"])
        self.num_frames = int(self.signal["num_frames"])
        self.signal_path = self.folder / self.signal.get("signal_file", "signal.bin")

        scales = self.signal.get("iq_scale_per_frame") or [self.signal.get("iq_scale", 1.0)]
        self.scales = [float(s) for s in scales]

        d = self.radar.get("derived", {})
        self.sample_rate = float(self.radar["sample_rate_hz"])
        self.slope = float(self.radar["slope_hz_per_s"])
        self.chirp_period = self._derived("chirp_period_s", self.radar)
        self.wavelength = float(d.get("wavelength_m", self.radar.get("wavelength_m", 0.0)))
        self.element_spacing = float(
            d.get("element_spacing_m", self.radar.get("element_spacing_m", 0.0))
        )
        self.chirps_per_tx = int(self.radar.get("chirps_per_tx", self.shape[0]))
        self.tx_chirp_period = float(
            d.get("tx_chirp_period_s", self.chirp_period)
        )
        self.tdm = bool(self.radar.get("tdm_interleave", False))
        self._handle = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _derived(key: str, radar: dict) -> float:
        if key in radar:
            return float(radar[key])
        return float(radar["ramp_time_s"]) + float(radar["idle_time_s"])

    @property
    def n_chirps(self) -> int:
        return self.shape[0]

    @property
    def n_elements(self) -> int:
        return self.shape[1]

    @property
    def n_samples(self) -> int:
        return self.shape[2]

    @property
    def frame_period(self) -> float:
        return float(self.meta.get("simulation", {}).get("frame_period_s", 0.0))

    # ------------------------------------------------------------------ #
    # axes -- DESIGN.md section 10.6
    # ------------------------------------------------------------------ #
    @property
    def range_bin_m(self) -> float:
        """c*fs / (2*S*Ns) -- equals c/(2*B_valid), NOT c/(2*S*Tr)."""
        return C_LIGHT * self.sample_rate / (2.0 * self.slope * self.n_samples)

    @property
    def max_range_m(self) -> float:
        """Complex sampling makes the whole fs span unambiguous."""
        return self.n_samples * self.range_bin_m

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength / (2.0 * self.n_chirps * self.chirp_period)

    @property
    def max_velocity_mps(self) -> float:
        return self.wavelength / (4.0 * self.tx_chirp_period)

    def range_axis(self) -> np.ndarray:
        """``R[k] = k * c*fs/(2*S*Ns)`` for all Ns bins, metres."""
        return np.arange(self.n_samples) * self.range_bin_m

    def velocity_axis(self) -> np.ndarray:
        """``v[i] = (i - Nc/2) * lambda/(2*Nc*Tc)``, m/s (fftshifted).

        Never build this with ``linspace``: the step would be wrong and an even
        ``Nc`` would not contain the zero-velocity bin.
        """
        i = np.arange(self.n_chirps)
        return (i - self.n_chirps / 2.0) * self.velocity_resolution_mps

    def angle_axis(self, n_fft: int = 180) -> np.ndarray:
        """``theta[k] = asin((k - Nfft/2) * lambda/(Nfft*d))``, radians."""
        k = np.arange(int(n_fft))
        u = (k - n_fft / 2.0) / n_fft
        return np.arcsin(np.clip(u * self.wavelength / self.element_spacing, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # frames
    # ------------------------------------------------------------------ #
    def raw_frame(self, index: int) -> bytes:
        """Read one frame's bytes with a single seek (never re-read the file)."""
        if not 0 <= index < self.num_frames:
            raise IndexError(f"frame {index} out of range (0..{self.num_frames - 1})")
        if self._handle is None:
            self._handle = self.signal_path.open("rb")
        self._handle.seek(index * self.frame_bytes)
        payload = self._handle.read(self.frame_bytes)
        if len(payload) != self.frame_bytes:
            raise IOError("truncated signal.bin")
        return payload

    def frame(self, index: int) -> np.ndarray:
        """One frame as ``(Nc, Nv, Ns)`` complex128 (scale already applied)."""
        cube = unpack_frame(self.raw_frame(index), self.shape, self.layout)
        scale = self.scales[min(index, len(self.scales) - 1)]
        return cube / scale

    def frames(self):
        """Iterate every frame lazily (constant memory)."""
        for index in range(self.num_frames):
            yield self.frame(index)

    # ------------------------------------------------------------------ #
    # standard processing
    # ------------------------------------------------------------------ #
    def range_profile(self, index: int = 0, *, window: str = "hann") -> np.ndarray:
        """Range FFT magnitude summed over chirps and elements, ``(Ns,)``."""
        cube = self.frame(index)
        w = periodic_window(self.n_samples, window)[None, None, :]
        spec = np.fft.fft(cube * w, axis=-1)
        return np.sqrt((np.abs(spec) ** 2).sum(axis=(0, 1)))

    def range_doppler_complex(
        self, index: int = 0, *, range_window: str = "hann",
        doppler_window: str = "hann",
    ) -> np.ndarray:
        """Complex range-Doppler cube ``(Nc, Nv, Ns)``.

        Take array snapshots for DOA from *this* cube: it already carries the
        full coherent integration gain of both FFTs.
        """
        cube = self.frame(index)
        wr = periodic_window(self.n_samples, range_window)[None, None, :]
        after_range = np.fft.fft(cube * wr, axis=-1)
        wd = periodic_window(self.n_chirps, doppler_window)[:, None, None]
        return np.fft.fftshift(np.fft.fft(after_range * wd, axis=0), axes=0)

    def range_doppler(
        self, index: int = 0, *, range_window: str = "hann",
        doppler_window: str = "hann", db: bool = True,
    ):
        """``(rd, range_axis, velocity_axis)`` with ``rd`` shaped ``(Nc, Ns)``."""
        cube = self.range_doppler_complex(
            index, range_window=range_window, doppler_window=doppler_window
        )
        power = (np.abs(cube) ** 2).sum(axis=1)
        rd = 10.0 * np.log10(np.maximum(power, 1e-30)) if db else power
        return rd, self.range_axis(), self.velocity_axis()

    # ------------------------------------------------------------------ #
    def truth(self, frame: int | None = None) -> dict[str, np.ndarray]:
        """Ground-truth point cloud (``pointcloud_gt.csv``), optionally filtered."""
        path = self.folder / "pointcloud_gt.csv"
        if not path.exists():
            return {}
        columns: dict[str, list] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if frame is not None and int(row["frame"]) != frame:
                    continue
                for key, value in row.items():
                    columns.setdefault(key, []).append(value)
        out: dict[str, np.ndarray] = {}
        for key, values in columns.items():
            if key == "path_type":
                out[key] = np.asarray(values, dtype="<U8")
            else:
                out[key] = np.asarray([float(v) for v in values])
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RadarFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = [
            f"dataset      : {self.folder}",
            f"layout       : {self.layout}",
            f"frames       : {self.num_frames}",
            f"shape (Nc,Nv,Ns): {self.shape}",
            f"frame bytes  : {self.frame_bytes}",
            f"iq scale     : {self.signal.get('iq_scale_mode')}",
            "",
            f"fs           : {self.sample_rate / 1e6:.3f} MHz",
            f"slope        : {self.slope / 1e12:.3f} THz/s",
            f"Tc           : {self.chirp_period * 1e6:.1f} us",
            f"lambda       : {self.wavelength * 1e3:.3f} mm",
            f"dR           : {self.range_bin_m:.4f} m",
            f"Rmax         : {self.max_range_m:.2f} m",
            f"vmax         : +-{self.max_velocity_mps:.3f} m/s",
            f"dv           : {self.velocity_resolution_mps:.4f} m/s",
        ]
        if self.tdm:
            lines.append(f"tdm          : chirps_per_tx={self.chirps_per_tx}, "
                         f"de-interleave before the doppler FFT!")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python radar_io.py DATASET_DIR")
    with RadarFile(sys.argv[1]) as reader:
        print(reader.summary())
        rd, r_axis, v_axis = reader.range_doppler(0)
        peak = np.unravel_index(int(np.argmax(rd)), rd.shape)
        print(f"\nframe 0 peak: R={r_axis[peak[1]]:.2f} m, "
              f"v={v_axis[peak[0]]:+.3f} m/s, {rd[peak]:.1f} dB")
