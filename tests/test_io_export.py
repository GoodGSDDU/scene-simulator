"""Export tests -- covers T13 (dual layout identity) and T9 (bit-exact reruns)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radar_sim.core.io_export import (
    FULL_SCALE,
    DatasetWriter,
    load_frame,
    load_meta,
    pack_frame,
    quantize_frame,
    unpack_frame,
)
from radar_sim.core.radar import synthesize_frame
from radar_sim.core.radar_config import RadarConfig
from radar_sim.core.raytrace import trace_paths
from radar_sim.core.scene import Primitive, Scene
from radar_sim.core.scatterers import build_scatterers

SMALL = dict(chirps_per_frame=16, samples_per_chirp=32, tx_elements=2, rx_elements=2,
             mimo_mode="ideal")


@pytest.fixture
def cfg() -> RadarConfig:
    return RadarConfig.iwr1843(**SMALL)


def _cube(cfg: RadarConfig, seed: int = 3, shift: float = 0.0) -> np.ndarray:
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": 8.0 + shift, "y": 3.0}, material_id="metal"))
    scene.add(Primitive("point", {"x": 14.0, "y": -4.0}, material_id="concrete"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, np.array([0.0, 0.0]), max_order=0)
    result = synthesize_frame(paths, sc, scene, cfg, rng=np.random.default_rng(seed))
    return result.cube


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #
def test_quantize_maps_to_int16_range(cfg: RadarConfig):
    cube = _cube(cfg)
    peak = float(np.max(np.abs(cube)))
    pairs, clipped = quantize_frame(cube, FULL_SCALE / peak)
    assert pairs.dtype == np.int16
    assert pairs.shape == cube.shape + (2,)
    assert clipped == 0
    assert int(np.abs(pairs).max()) <= int(FULL_SCALE)


def test_quantize_reports_clipping(cfg: RadarConfig):
    cube = _cube(cfg)
    _pairs, clipped = quantize_frame(cube, 10.0 * FULL_SCALE / float(np.max(np.abs(cube))))
    assert clipped > 0


def test_quantize_roundtrips_within_one_lsb(cfg: RadarConfig):
    cube = _cube(cfg)
    scale = FULL_SCALE / float(np.max(np.abs(cube)))
    pairs, _ = quantize_frame(cube, scale)
    restored = (pairs[..., 0] + 1j * pairs[..., 1]) / scale
    assert np.abs(restored - cube).max() <= 1.0 / scale


# --------------------------------------------------------------------------- #
# T13 -- the two layouts must describe identical data
# --------------------------------------------------------------------------- #
def test_ti_layout_byte_order_is_the_ti_documented_one():
    pairs = np.array([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=np.int16)
    raw = np.frombuffer(pack_frame(pairs, "ti_dca1000"), dtype="<i2")
    assert raw.tolist() == [1, 3, 2, 4, 5, 7, 6, 8]


def test_interleaved_layout_byte_order():
    pairs = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    raw = np.frombuffer(pack_frame(pairs, "interleaved_iq"), dtype="<i2")
    assert raw.tolist() == [1, 2, 3, 4]


def test_both_layouts_roundtrip_to_the_same_cube_t13(cfg: RadarConfig):
    cube = _cube(cfg)
    scale = FULL_SCALE / float(np.max(np.abs(cube)))
    pairs, _ = quantize_frame(cube, scale)

    ti = unpack_frame(pack_frame(pairs, "ti_dca1000"), cube.shape, "ti_dca1000")
    iq = unpack_frame(pack_frame(pairs, "interleaved_iq"), cube.shape, "interleaved_iq")
    assert np.array_equal(ti, iq), "the two layouts must describe identical data"
    # unpack_frame returns raw int16; the scale is applied by the caller
    assert np.abs(ti / scale - cube).max() <= 1.0 / scale


def test_both_layouts_have_the_same_size(cfg: RadarConfig):
    cube = _cube(cfg)
    pairs, _ = quantize_frame(cube, 1.0)
    assert len(pack_frame(pairs, "ti_dca1000")) == cfg.frame_bytes
    assert len(pack_frame(pairs, "interleaved_iq")) == cfg.frame_bytes


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        pack_frame(np.zeros((1, 1, 2, 2), dtype=np.int16), "nope")


def test_unpack_rejects_wrong_length(cfg: RadarConfig):
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x00", (4, 4, 8), "interleaved_iq")


# --------------------------------------------------------------------------- #
# writer / meta
# --------------------------------------------------------------------------- #
def test_writer_produces_meta_signal_and_readme(tmp_path, cfg: RadarConfig):
    scene = Scene.demo()
    writer = DatasetWriter(tmp_path / "out", cfg, layout="ti_dca1000", scene=scene,
                           simulation={"seed": 42, "duration_s": 1.0})
    for i in range(3):
        writer.add_frame(_cube(cfg, shift=0.1 * i), frame_meta={"t_s": 0.08 * i})
    meta = writer.close()

    assert (tmp_path / "out" / "signal.bin").stat().st_size == 3 * cfg.frame_bytes
    assert (tmp_path / "out" / "meta.json").exists()
    assert (tmp_path / "out" / "README.txt").exists()

    assert meta["signal"]["layout"] == "ti_dca1000"
    assert meta["signal"]["num_frames"] == 3
    assert meta["signal"]["shape_per_frame"] == [16, 4, 32]
    assert meta["signal"]["axis_order"] == ["chirp", "virtual_element", "sample"]
    assert meta["signal"]["frame_bytes"] == cfg.frame_bytes
    assert meta["signal"]["data_bytes"] == 3 * cfg.frame_bytes
    assert meta["radar"]["mimo_mode"] == "ideal"
    assert meta["radar"]["tdm_interleave"] is False
    assert meta["radar"]["derived"]["max_range_m"] == pytest.approx(cfg.max_range_m)
    assert meta["scene_snapshot"]["primitives"]
    assert [f["frame"] for f in meta["frames"]] == [0, 1, 2]
    assert meta["frames"][1]["byte_offset"] == cfg.frame_bytes

    on_disk = load_meta(tmp_path / "out")
    assert on_disk["signal"]["num_frames"] == 3
    json.dumps(on_disk)                                        # must serialise


def test_writer_records_frame0_scale_for_all_frames(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg, iq_scale_mode="frame0")
    for i in range(3):
        writer.add_frame(_cube(cfg, shift=0.5 * i))
    meta = writer.close()
    scales = meta["signal"]["iq_scale_per_frame"]
    assert len(scales) == 3
    assert 0.0 not in scales
    assert meta["signal"]["iq_scale"] == pytest.approx(scales[0])


def test_writer_per_frame_scale_mode(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg, iq_scale_mode="per_frame")
    for i in range(2):
        writer.add_frame(_cube(cfg, shift=3.0 * i))
    meta = writer.close()
    scales = meta["signal"]["iq_scale_per_frame"]
    assert scales[0] != pytest.approx(scales[1])                # different peaks


def test_writer_fixed_scale_requires_a_value(tmp_path, cfg: RadarConfig):
    with pytest.raises(ValueError):
        DatasetWriter(tmp_path / "never", cfg, iq_scale_mode="fixed")


def test_writer_rejects_unknown_scale_mode(tmp_path, cfg: RadarConfig):
    with pytest.raises(ValueError):
        DatasetWriter(tmp_path / "o", cfg, iq_scale_mode="magic")


def test_writer_rejects_wrong_frame_shape(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg)
    with pytest.raises(ValueError, match="frame shape"):
        writer.add_frame(np.zeros((4, 4, 32), dtype=complex))


def test_load_frame_seeks_the_right_frame(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg, iq_scale_mode="per_frame")
    cubes = [_cube(cfg, shift=2.0 * i) for i in range(3)]
    for cube in cubes:
        writer.add_frame(cube)
    writer.close()

    for i, expected in enumerate(cubes):
        loaded = load_frame(tmp_path / "o", i)
        assert loaded.shape == expected.shape
        # quantisation error only
        scale = FULL_SCALE / float(np.max(np.abs(expected)))
        assert np.abs(loaded - expected).max() <= 1.0 / scale


def test_load_frame_out_of_range(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg)
    writer.add_frame(_cube(cfg))
    writer.close()
    with pytest.raises(IndexError):
        load_frame(tmp_path / "o", 5)


def test_interleaved_layout_also_reads_back(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg, layout="interleaved_iq")
    cube = _cube(cfg)
    writer.add_frame(cube)
    meta = writer.close()
    assert meta["signal"]["layout"] == "interleaved_iq"
    loaded = load_frame(tmp_path / "o", 0)
    scale = FULL_SCALE / float(np.max(np.abs(cube)))
    assert np.abs(loaded - cube).max() <= 1.0 / scale


def test_truth_csv_is_written(tmp_path, cfg: RadarConfig):
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": 8.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, np.array([0.0, 0.0]), max_order=0)
    writer = DatasetWriter(tmp_path / "o", cfg, scene=scene)
    for _ in range(2):
        result = synthesize_frame(paths, sc, scene, cfg,
                                  rng=np.random.default_rng(1), include_truth=True)
        writer.add_frame(result.cube, truth=result.truth)
    writer.close()

    csv_path = tmp_path / "o" / "pointcloud_gt.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("frame,target_id,path_type,range_m,azimuth_deg")
    assert len(lines) == 3                                       # header + 2 frames


def test_writer_closed_twice_is_safe(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg)
    writer.add_frame(_cube(cfg))
    first = writer.close()
    second = writer.close()
    assert first is second


def test_add_frame_after_close_raises(tmp_path, cfg: RadarConfig):
    writer = DatasetWriter(tmp_path / "o", cfg)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.add_frame(_cube(cfg))


# --------------------------------------------------------------------------- #
# T9 -- determinism end to end
# --------------------------------------------------------------------------- #
def test_repeated_runs_are_bit_identical_t9(tmp_path, cfg: RadarConfig):
    def build(folder) -> bytes:
        scene = Scene.empty(60.0, 60.0)
        scene.add(Primitive("rect", {"x": 12.0, "y": 4.0, "w": 4.0, "h": 2.0},
                            material_id="metal"))
        scene.add(Primitive("point", {"x": 6.0, "y": -3.0}, material_id="wood"))
        sc = build_scatterers(scene)
        rng = np.random.default_rng(42)
        writer = DatasetWriter(folder, cfg, scene=scene)
        for frame in range(3):
            pos = np.array([0.2 * frame, 0.0])
            paths = trace_paths(scene, sc, pos, max_order=1)
            result = synthesize_frame(paths, sc, scene, cfg, rng=rng)
            writer.add_frame(result.cube)
        writer.close()
        return (folder / "signal.bin").read_bytes()

    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    assert a == b, "same scene + same seed must produce identical bytes"
    assert len(a) == 3 * cfg.frame_bytes


def test_different_seed_changes_the_bytes(tmp_path, cfg: RadarConfig):
    def build(folder, seed: int) -> bytes:
        scene = Scene.empty(60.0, 60.0)
        scene.add(Primitive("point", {"x": 6.0, "y": -3.0}, material_id="metal"))
        sc = build_scatterers(scene)
        paths = trace_paths(scene, sc, np.array([0.0, 0.0]), max_order=0)
        writer = DatasetWriter(folder, cfg, scene=scene)
        result = synthesize_frame(paths, sc, scene, cfg,
                                  rng=np.random.default_rng(seed))
        writer.add_frame(result.cube)
        writer.close()
        return (folder / "signal.bin").read_bytes()

    assert build(tmp_path / "a", 1) != build(tmp_path / "b", 2)
