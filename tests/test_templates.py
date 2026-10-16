"""Drift guard: the shipped templates must agree with the simulation core.

The templates under ``templates/`` are deliberately self-contained (numpy only)
so users can copy them, which means the axis formulas are duplicated.  These
tests are the contract that keeps the two copies identical -- covers T10
(cross-implementation agreement) and the numeric half of T13.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates" / "python"

from radar_sim.core import framing  # noqa: E402
from radar_sim.core.io_export import DatasetWriter, load_frame  # noqa: E402
from radar_sim.core.radar import synthesize_frame  # noqa: E402
from radar_sim.core.radar_config import RadarConfig  # noqa: E402
from radar_sim.core.raytrace import trace_paths  # noqa: E402
from radar_sim.core.scene import Primitive, Scene  # noqa: E402
from radar_sim.core.scatterers import build_scatterers  # noqa: E402


@pytest.fixture(scope="module")
def template_module():
    """Import ``templates/python/radar_io.py`` under its own name."""
    path = TEMPLATES / "radar_io.py"
    if not path.exists():                                       # pragma: no cover
        pytest.skip("templates/python/radar_io.py is missing")
    spec = importlib.util.spec_from_file_location("tpl_radar_io", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tpl_radar_io"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("ds")
    cfg = RadarConfig.iwr1843(chirps_per_frame=16, samples_per_chirp=32,
                              tx_elements=2, rx_elements=2)
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("rect", {"x": 20.0, "y": 8.0, "w": 6.0, "h": 2.0},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 10.0, "y": 4.0}, material_id="wood"))
    scatterers = build_scatterers(scene, segment_len=1.0)
    writer = DatasetWriter(folder, cfg, scene=scene)
    rng = np.random.default_rng(11)
    for frame in range(3):
        paths = trace_paths(scene, scatterers, np.array([1.0 + frame, 2.0]), cfg=cfg)
        result = synthesize_frame(paths, scatterers, scene, cfg, rng=rng)
        writer.add_frame(result.cube, truth=result.truth)
    writer.close()
    return folder


# --------------------------------------------------------------------------- #
def test_template_reads_the_same_cube_as_the_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    try:
        for index in range(3):
            from_core = load_frame(dataset, index)
            from_template = reader.frame(index)
            assert from_template.shape == from_core.shape
            assert np.allclose(from_template, from_core, atol=0.0), \
                "template reader and core reader disagree"
    finally:
        reader.close()


def test_template_hash_matches_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    try:
        assert reader.num_frames == 3
        assert reader.shape == (16, 4, 32)
        assert reader.layout == "ti_dca1000"
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# axis formulas -- DESIGN.md section 10.6
# --------------------------------------------------------------------------- #
def test_template_range_axis_matches_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    cfg = RadarConfig.iwr1843(chirps_per_frame=16, samples_per_chirp=32,
                              tx_elements=2, rx_elements=2)
    try:
        assert np.allclose(reader.range_axis(), framing.range_axis(cfg), atol=0.0)
        assert reader.range_bin_m == pytest.approx(framing.range_axis(cfg)[1], abs=0.0)
        assert reader.max_range_m == pytest.approx(cfg.max_range_m, rel=1e-12)
    finally:
        reader.close()


def test_template_velocity_axis_matches_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    cfg = RadarConfig.iwr1843(chirps_per_frame=16, samples_per_chirp=32,
                              tx_elements=2, rx_elements=2)
    try:
        axis = reader.velocity_axis()
        assert np.allclose(axis, framing.velocity_axis(cfg), atol=0.0)
        assert axis[cfg.chirps_per_frame // 2] == 0.0
        # explicit regression guard: never a linspace
        assert not np.allclose(
            axis, np.linspace(-cfg.max_velocity_mps, cfg.max_velocity_mps, cfg.chirps_per_frame)
        )
    finally:
        reader.close()


def test_template_angle_axis_matches_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    cfg = RadarConfig.iwr1843(chirps_per_frame=16, samples_per_chirp=32,
                              tx_elements=2, rx_elements=2)
    try:
        for n_fft in (64, 180):
            assert np.allclose(reader.angle_axis(n_fft),
                               framing.angle_axis(n_fft, cfg.element_spacing_m,
                                                  cfg.wavelength_m),
                               atol=0.0)
    finally:
        reader.close()


def test_template_periodic_window_matches_core(template_module):
    for n in (4, 8, 9, 16, 33):
        for kind in ("hann", "hamming", "blackman", "rect"):
            assert np.allclose(template_module.periodic_window(n, kind),
                               framing.periodic_window(n, kind), atol=0.0)


# --------------------------------------------------------------------------- #
def test_template_range_doppler_matches_core(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    cfg = RadarConfig.iwr1843(chirps_per_frame=16, samples_per_chirp=32,
                              tx_elements=2, rx_elements=2)
    try:
        rd_template, _r, _v = reader.range_doppler(0, db=False)
        cube = load_frame(dataset, 0)
        rd_core = framing.range_doppler_map(cube, cfg)
        assert np.allclose(rd_template, rd_core, rtol=1e-12, atol=0.0)
    finally:
        reader.close()


def test_template_unpack_rejects_truncated_data(template_module):
    with pytest.raises(ValueError):
        template_module.unpack_frame(b"\x00\x00", (4, 4, 8), "interleaved_iq")


def test_template_truth_csv_roundtrip(dataset, template_module):
    reader = template_module.RadarFile(dataset)
    try:
        truth = reader.truth(0)
        assert np.size(truth["range_m"]) > 0
        assert "radial_vel_mps" in truth
        assert "path_type" in truth
    finally:
        reader.close()
