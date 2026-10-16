"""Dataset export: dual-layout int16 packing, ``meta.json``, ground-truth CSV.

Two binary layouts are supported (DESIGN.md section 8.2), both 4 bytes per
complex sample:

``interleaved_iq``
    ``for m, k, n:  int16 Re, int16 Im``

``ti_dca1000`` (default)
    Per chirp the virtual elements are laid out element-outer / sample-inner,
    then every two complex samples are packed into four int16:
    ``[Re(v0), Re(v1), Im(v0), Im(v1), ...]`` -- byte-compatible with TI's
    official ``readDCA1000.m``.

Writing is streaming: frames are appended as they are produced, so long runs
never need to be held in memory.  The IQ scale is chosen on the first frame
(``frame0`` mode) and then held constant, which keeps amplitudes comparable
across frames -- per-frame AGC would destroy the radar's amplitude
information.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .radar_config import RadarConfig
from .scene import Scene

__all__ = [
    "LAYOUTS",
    "SCALE_MODES",
    "FULL_SCALE",
    "quantize_frame",
    "pack_frame",
    "unpack_frame",
    "DatasetWriter",
    "write_truth_csv",
    "load_meta",
    "load_frame",
]

LAYOUTS = ("ti_dca1000", "interleaved_iq")
SCALE_MODES = ("frame0", "per_frame", "fixed")
FULL_SCALE = 32767.0


# --------------------------------------------------------------------------- #
# quantisation / packing
# --------------------------------------------------------------------------- #
def quantize_frame(
    cube: np.ndarray, scale: float, full_scale: float = FULL_SCALE
) -> tuple[np.ndarray, int]:
    """Scale and clip a complex cube to int16 ``(..., 2)`` ``[Re, Im]`` pairs."""
    scaled = np.asarray(cube) * float(scale)
    re = np.rint(scaled.real)
    im = np.rint(scaled.imag)
    clipped = int(np.count_nonzero(np.abs(re) > full_scale)) + int(
        np.count_nonzero(np.abs(im) > full_scale)
    )
    pairs = np.empty(scaled.shape + (2,), dtype=np.int16)
    pairs[..., 0] = np.clip(re, -full_scale, full_scale)
    pairs[..., 1] = np.clip(im, -full_scale, full_scale)
    return pairs, clipped


def pack_frame(pairs: np.ndarray, layout: str = "ti_dca1000") -> bytes:
    """Pack int16 ``(Nc, Nv, Ns, 2)`` pairs into the on-disk byte stream."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")
    arr = np.ascontiguousarray(pairs, dtype="<i2")
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"expected (Nc, Nv, Ns, 2) int16 pairs, got {arr.shape}")

    if layout == "interleaved_iq":
        return arr.tobytes()

    nc, nv, ns, _ = arr.shape
    flat = nv * ns
    if flat % 2 != 0:
        raise ValueError("ti_dca1000 layout needs an even Nv*Ns")
    lin = arr.reshape(nc, flat, 2).astype(np.int32)
    even = lin[:, 0::2, :]
    odd = lin[:, 1::2, :]
    out = np.empty((nc, flat * 2), dtype="<i2")
    out[:, 0::4] = even[:, :, 0]
    out[:, 1::4] = odd[:, :, 0]
    out[:, 2::4] = even[:, :, 1]
    out[:, 3::4] = odd[:, :, 1]
    return out.tobytes()


def unpack_frame(raw: bytes, shape: tuple[int, int, int], layout: str = "ti_dca1000") -> np.ndarray:
    """Inverse of :func:`pack_frame` (used by tests and by the core viewer)."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    nc, nv, ns = (int(v) for v in shape)
    flat = nv * ns
    data = np.frombuffer(raw, dtype="<i2")
    expected = nc * nv * ns * 2
    if data.size != expected:
        raise ValueError(f"expected {expected} int16 values, got {data.size}")

    if layout == "interleaved_iq":
        arr = data.reshape(nc, nv, ns, 2).astype(np.float64)
        return arr[..., 0] + 1j * arr[..., 1]

    lin = data.reshape(nc, flat * 2).astype(np.float64)
    v = np.empty((nc, flat), dtype=np.complex128)
    v[:, 0::2] = lin[:, 0::4] + 1j * lin[:, 2::4]
    v[:, 1::2] = lin[:, 1::4] + 1j * lin[:, 3::4]
    return v.reshape(nc, nv, ns)


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class DatasetWriter:
    """Streaming writer for ``signal.bin`` + ``meta.json`` + ``pointcloud_gt.csv``."""

    def __init__(
        self,
        out_dir: str | Path,
        cfg: RadarConfig,
        *,
        layout: str = "ti_dca1000",
        iq_scale_mode: str = "frame0",
        iq_scale: float | None = None,
        scene: Scene | None = None,
        simulation: dict[str, Any] | None = None,
        vehicle: dict[str, Any] | None = None,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")
        if iq_scale_mode not in SCALE_MODES:
            raise ValueError(f"unknown iq_scale_mode {iq_scale_mode!r}")
        if iq_scale_mode == "fixed" and iq_scale is None:
            raise ValueError("iq_scale_mode='fixed' requires an explicit iq_scale")
        cfg.validate()

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.layout = layout
        self.scale_mode = iq_scale_mode
        self.scale = float(iq_scale) if iq_scale is not None else None
        self.scene = scene
        self.simulation = dict(simulation or {})
        self.vehicle = dict(vehicle or {})

        self.signal_path = self.out_dir / "signal.bin"
        self._handle = self.signal_path.open("wb")
        self._frames: list[dict[str, Any]] = []
        self._scales: list[float] = []
        self._truth_rows: list[tuple[int, dict[str, np.ndarray]]] = []
        self.total_clipped = 0
        self._closed = False

    # ------------------------------------------------------------------ #
    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def add_frame(
        self,
        cube: np.ndarray,
        *,
        frame_meta: dict[str, Any] | None = None,
        truth: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("writer is already closed")
        cube = np.asarray(cube)
        expected = (int(self.cfg.chirps_per_frame), int(self.cfg.virtual_elements),
                    int(self.cfg.samples_per_chirp))
        if cube.shape != expected:
            raise ValueError(f"frame shape {cube.shape} != {expected}")

        if self.scale_mode == "per_frame" or self.scale is None:
            peak = float(np.max(np.abs(cube))) if cube.size else 0.0
            scale = (FULL_SCALE / peak) if peak > 0.0 else 1.0
            if self.scale_mode == "frame0" and self.scale is None:
                self.scale = scale
            frame_scale = scale
        else:
            frame_scale = self.scale

        pairs, clipped = quantize_frame(cube, frame_scale)
        payload = pack_frame(pairs, self.layout)
        self._handle.write(payload)

        self._scales.append(frame_scale)
        self.total_clipped += clipped
        index = len(self._frames)
        record: dict[str, Any] = {"frame": index, "byte_offset": index * len(payload)}
        record.update(frame_meta or {})
        self._frames.append(record)
        if truth is not None and int(np.size(truth.get("range_m", ()))) > 0:
            self._truth_rows.append((index, truth))
        return record

    # ------------------------------------------------------------------ #
    def close(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            return self.meta
        self._closed = True
        self._handle.close()

        total_samples = self.num_frames * int(self.cfg.samples_per_frame)
        clip_ratio = (
            self.total_clipped / (2.0 * total_samples) if total_samples else 0.0
        )
        meta: dict[str, Any] = {
            "format_version": "1.0",
            "generator": {"name": "2D Radar Scene Simulator", "version": "0.1.0"},
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "signal_file": "signal.bin",
            "signal": {
                "dtype": "int16",
                "endianness": "little",
                "layout": self.layout,
                "interleaved": "IQ" if self.layout == "interleaved_iq" else "IQ_pairs",
                "shape_per_frame": [
                    int(self.cfg.chirps_per_frame),
                    int(self.cfg.virtual_elements),
                    int(self.cfg.samples_per_chirp),
                ],
                "axis_order": ["chirp", "virtual_element", "sample"],
                "frame_bytes": int(self.cfg.frame_bytes),
                "num_frames": self.num_frames,
                "data_bytes": int(total_samples * 4),
                "iq_scale": float(self._scales[0]) if self._scales else 1.0,
                "iq_scale_mode": self.scale_mode,
                "iq_scale_per_frame": [float(s) for s in self._scales],
                "full_scale": FULL_SCALE,
                "count_clipped": int(self.total_clipped),
                "clip_ratio": float(clip_ratio),
            },
            "radar": {
                **self.cfg.to_dict(),
                "center_freq_hz": self.cfg.center_freq_hz,
                "wavelength_m": self.cfg.wavelength_m,
                "element_spacing_m": self.cfg.element_spacing_m,
                "tdm_interleave": self.cfg.mimo_mode == "tdm",
                "chirps_per_tx": int(self.cfg.chirps_per_tx),
                "derived": self.cfg.derived(),
            },
            "simulation": self.simulation,
            "vehicle": self.vehicle,
            "frames": self._frames,
        }
        if extra:
            meta.update(extra)
        if self.scene is not None:
            meta["scene_snapshot"] = self.scene.to_dict()

        (self.out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self._truth_rows:
            write_truth_csv(self.out_dir / "pointcloud_gt.csv", self._truth_rows)
        self._write_readme(meta)
        self.meta = meta
        return meta

    # ------------------------------------------------------------------ #
    def _write_readme(self, meta: dict[str, Any]) -> None:
        sig = meta["signal"]
        nc, nv, ns = (int(v) for v in sig["shape_per_frame"])
        if sig["layout"] == "interleaved_iq":
            parse = [
                "  raw  = np.memmap('signal.bin', dtype='<i2', mode='r')",
                f"  raw  = raw.reshape(-1, {nc}, {nv}, {ns}, 2)",
                "  cube = (raw[..., 0] + 1j * raw[..., 1]) / s",
            ]
            matlab = [
                "  raw  = fread(fid, Inf, 'int16=>double');",
                f"  raw  = reshape(raw, [2, {ns}, {nv}, {nc}, F]);   % column-major!",
                "  cube = (raw(1,:,:,:,:) + 1i*raw(2,:,:,:,:)) / s;",
            ]
        else:
            parse = [
                "  raw  = np.memmap('signal.bin', dtype='<i2', mode='r')",
                f"  raw  = raw.reshape(-1, {nc}, {nv} * {ns} * 2)   # ti_dca1000",
                "  v    = np.empty((raw.shape[0], %d), dtype=complex)" % (nv * ns),
                "  v[:, 0::2] = raw[:, :, 0::4] + 1j * raw[:, :, 2::4]",
                "  v[:, 1::2] = raw[:, :, 1::4] + 1j * raw[:, :, 3::4]",
                f"  cube = v.reshape(-1, {nc}, {nv}, {ns}) / s",
            ]
            matlab = [
                "  % same de-interleave as TI's official readDCA1000.m",
                "  raw = fread(fid, Inf, 'int16=>double');",
            ]
        lines = [
            "2D Radar Scene Simulator - dataset",
            "=================================",
            "",
            f"frames        : {sig['num_frames']}",
            f"shape/frame   : (Nc, Nv, Ns) = ({nc}, {nv}, {ns})",
            f"layout        : {sig['layout']}",
            f"frame bytes   : {sig['frame_bytes']}",
            f"iq_scale      : {sig['iq_scale']:.6g} ({sig['iq_scale_mode']})",
            "",
            "signal.bin has no header. Complex value = (Re + 1j*Im) / iq_scale,",
            "values are little-endian int16 in the order documented in meta.json.",
            "Frame f starts at byte offset f * frame_bytes (seek directly!).",
            "",
            "python:",
            "  import json, numpy as np",
            "  m = json.load(open('meta.json', encoding='utf-8'))",
            "  Nc, Nv, Ns = m['signal']['shape_per_frame']; s = m['signal']['iq_scale']",
            *parse,
            "",
            "matlab:",
            *matlab,
            "",
            "See templates/ for ready-to-run readers (Python / MATLAB) and a",
            "six-step Jupyter walkthrough.",
        ]
        (self.out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# ground truth csv
# --------------------------------------------------------------------------- #
_TRUTH_COLUMNS = (
    "frame", "target_id", "path_type", "range_m", "azimuth_deg",
    "radial_vel_mps", "amplitude_db", "rcs_eff", "snr_db",
)


def write_truth_csv(path: str | Path, rows: Iterable[tuple[int, dict[str, np.ndarray]]]) -> Path:
    """Write the ideal point cloud (one row per path per frame)."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_TRUTH_COLUMNS)
        for frame_index, truth in rows:
            count = int(np.size(truth.get("range_m", [])))
            for i in range(count):
                writer.writerow([
                    frame_index,
                    int(truth["owner"][i]),
                    str(truth["path_type"][i]),
                    f"{float(truth['range_m'][i]):.4f}",
                    f"{float(truth['azimuth_deg'][i]):.4f}",
                    f"{float(truth['radial_vel_mps'][i]):.4f}",
                    f"{float(truth['amplitude_db'][i]):.3f}",
                    f"{float(truth['rcs_eff'][i]):.6g}",
                    f"{float(truth['snr_db'][i]):.3f}",
                ])
    return path


# --------------------------------------------------------------------------- #
# reader helpers (shared by the core viewer; templates ship their own copy)
# --------------------------------------------------------------------------- #
def load_meta(folder: str | Path) -> dict[str, Any]:
    return json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))


def load_frame(folder: str | Path, index: int, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Read one frame as a ``(Nc, Nv, Ns)`` complex128 cube."""
    folder = Path(folder)
    meta = meta or load_meta(folder)
    sig = meta["signal"]
    frame_bytes = int(sig["frame_bytes"])
    total = int(sig["num_frames"])
    if not 0 <= index < total:
        raise IndexError(f"frame {index} out of range (0..{total - 1})")
    shape = tuple(int(v) for v in sig["shape_per_frame"])
    with (folder / sig.get("signal_file", "signal.bin")).open("rb") as handle:
        handle.seek(index * frame_bytes)
        raw = handle.read(frame_bytes)
    cube = unpack_frame(raw, shape, sig["layout"])          # type: ignore[arg-type]
    scales = sig.get("iq_scale_per_frame") or [sig.get("iq_scale", 1.0)]
    scale = float(scales[min(index, len(scales) - 1)])
    return cube / scale
