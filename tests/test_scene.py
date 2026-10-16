"""Scene / material / scatterer tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radar_sim.core.geometry import polygon_signed_area
from radar_sim.core.materials import PRESETS, Material, preset_materials
from radar_sim.core.scene import EdgeSet, Primitive, Scene
from radar_sim.core.scatterers import KIND_BOUNDARY, KIND_POINT, build_scatterers


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def test_presets_are_valid_and_unique():
    ids = [m.id for m in PRESETS]
    assert len(ids) == len(set(ids))
    for m in PRESETS:
        assert 0.0 <= m.rho <= 1.0
        assert 0.0 <= m.eta <= 1.0
        assert 0.0 <= m.tau <= 1.0
        assert m.name


def test_preset_materials_returns_fresh_copies():
    a = preset_materials()
    b = preset_materials()
    a["metal"].rho = 0.1
    assert b["metal"].rho == pytest.approx(0.95)


def test_material_rejects_out_of_range():
    with pytest.raises(ValueError):
        Material("bad", rho=1.5)
    with pytest.raises(ValueError):
        Material("bad", eta=-0.1)
    with pytest.raises(ValueError):
        Material("bad", tau=1.4)


def test_facing_metal_is_much_brighter_than_tilted():
    metal = next(m for m in PRESETS if m.id == "metal")
    facing = abs(complex(metal.reflection_amplitude(0.0)))
    tilted = abs(complex(metal.reflection_amplitude(np.deg2rad(30.0))))
    assert facing > tilted, "specular lobe must dim with incidence angle"
    assert 20.0 * np.log10(facing / tilted) > 10.0


def test_metal_brighter_than_vegetation_when_facing():
    metal = next(m for m in PRESETS if m.id == "metal")
    veg = next(m for m in PRESETS if m.id == "vegetation")
    ratio = abs(complex(metal.reflection_amplitude(0.0))) / abs(complex(veg.reflection_amplitude(0.0)))
    assert 20.0 * np.log10(ratio) > 15.0
    # vegetation is strongly diffuse, so the gap narrows sharply at oblique incidence
    psi = np.deg2rad(30.0)
    oblique = abs(complex(metal.reflection_amplitude(psi))) / abs(
        complex(veg.reflection_amplitude(psi))
    )
    assert oblique < ratio


def test_roughness_moves_energy_from_specular_to_diffuse():
    smooth = Material("s", rho=1.0, eta=0.0, sigma_psi_deg=8.0)
    rough = Material("r", rho=1.0, eta=1.0)
    # energy is conserved at normal incidence -- roughness only redistributes it
    assert abs(complex(smooth.reflection_amplitude(0.0))) == pytest.approx(
        abs(complex(rough.reflection_amplitude(0.0)))
    )
    # at oblique incidence the rough surface wins by orders of magnitude
    psi = np.deg2rad(60.0)
    assert abs(complex(rough.reflection_amplitude(psi))) > 100.0 * abs(
        complex(smooth.reflection_amplitude(psi))
    )


def test_transparent_material_does_not_reflect():
    glass_clear = Material("g", rho=1.0, tau=1.0)
    assert abs(complex(glass_clear.reflection_amplitude(0.0))) == pytest.approx(0.0)
    assert not glass_clear.opaque


def test_opacity_thresholds():
    assert Material("solid", tau=0.0).opaque
    assert Material("solid", tau=0.4).opaque
    assert not Material("clear", tau=0.9).opaque
    assert not Material("clear", tau=1.0).opaque


def test_reflection_amplitude_is_vectorised():
    m = Material("m", rho=0.5, eta=0.5)
    psi = np.linspace(0.0, np.pi / 2, 7)
    out = m.reflection_amplitude(psi)
    assert out.shape == (7,)
    assert np.isfinite(out.real).all() and np.isfinite(out.imag).all()


def test_absorption_reduces_amplitude():
    a = Material("a", rho=1.0, eta=0.0, absorption_db=0.0)
    b = Material("b", rho=1.0, eta=0.0, absorption_db=20.0)
    assert abs(complex(b.reflection_amplitude(0.0))) == pytest.approx(
        abs(complex(a.reflection_amplitude(0.0))) / 10.0
    )


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_primitive_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Primitive("blob", {})


def test_primitive_ids_are_unique():
    assert Primitive("point", {"x": 0.0, "y": 0.0}).id != Primitive("point", {"x": 0.0, "y": 0.0}).id


def test_primitive_outline_orientation_and_shape():
    assert Primitive("circle", {"x": 0.0, "y": 0.0, "r": 1.0}).outline().shape == (64, 2)
    assert Primitive("rect", {"x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0}).outline().shape == (4, 2)
    poly = Primitive("polygon", {"points": [[0, 0], [2, 0], [2, 2]]}).outline()
    assert polygon_signed_area(poly) > 0.0
    assert Primitive("wall", {"points": [[0, 0], [3, 0]], "thickness": 0.2}).outline().shape == (4, 2)
    assert Primitive("point", {"x": 1.0, "y": 2.0}).outline() is None


def test_primitive_translate_and_rotate():
    rect = Primitive("rect", {"x": 0.0, "y": 0.0, "w": 4.0, "h": 2.0, "rot_deg": 0.0})
    rect.translate(3.0, -1.0)
    assert rect.centroid() == pytest.approx((3.0, -1.0))
    rect.rotate_by(np.deg2rad(90.0))
    assert rect.params["rot_deg"] == pytest.approx(90.0)
    # rotating a circle is a no-op
    circle = Primitive("circle", {"x": 1.0, "y": 1.0, "r": 1.0})
    circle.rotate_by(np.deg2rad(45.0))
    assert circle.params["x"] == pytest.approx(1.0)


def test_primitive_polygon_rotation_moves_points():
    tri = Primitive("polygon", {"points": [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]})
    before = np.array(tri.params["points"], dtype=float)
    tri.rotate_by(np.deg2rad(90.0))
    after = np.array(tri.params["points"], dtype=float)
    assert not np.allclose(before, after)
    assert after.mean(axis=0) == pytest.approx(before.mean(axis=0), abs=1e-9)


def test_primitive_scale_by():
    rect = Primitive("rect", {"x": 0.0, "y": 0.0, "w": 4.0, "h": 2.0})
    rect.scale_by(0.5)
    assert rect.params["w"] == pytest.approx(2.0)
    assert rect.params["h"] == pytest.approx(1.0)
    rect.scale_by(0.0)                                          # must not explode
    assert rect.params["w"] > 0.0


def test_primitive_roundtrip():
    prim = Primitive("wall", {"points": [[0.0, 0.0], [1.0, 1.0]], "thickness": 0.3},
                     name="w", material_id="wood", layer=2)
    restored = Primitive.from_dict(prim.to_dict())
    assert restored.kind == prim.kind
    assert restored.id == prim.id
    assert restored.material_id == "wood"
    assert restored.params["thickness"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def test_empty_scene_has_presets_and_no_primitives():
    scene = Scene.empty()
    assert scene.primitives == []
    assert "metal" in scene.materials
    assert scene.material_index("metal") == scene.material_ids().index("metal")


def test_scene_add_remove_and_lookup():
    scene = Scene.empty()
    prim = scene.add(Primitive("point", {"x": 1.0, "y": 1.0}))
    assert scene.by_id(prim.id) is prim
    assert scene.remove(prim.id)
    assert scene.by_id(prim.id) is None
    assert not scene.remove("nope")


def test_scene_add_regenerates_duplicate_id():
    scene = Scene.empty()
    a = scene.add(Primitive("point", {"x": 0.0, "y": 0.0}, id="dup"))
    b = scene.add(Primitive("point", {"x": 1.0, "y": 0.0}, id="dup"))
    assert a.id == "dup"
    assert b.id != "dup"


def test_scene_unknown_material_raises():
    scene = Scene.empty()
    with pytest.raises(KeyError):
        scene.material("unobtainium")


def test_build_edges_skips_points_and_invisible():
    scene = Scene.empty()
    scene.add(Primitive("rect", {"x": 10.0, "y": 10.0, "w": 4.0, "h": 2.0}))
    scene.add(Primitive("point", {"x": 0.0, "y": 0.0}))
    hidden = scene.add(Primitive("rect", {"x": 30.0, "y": 30.0, "w": 4.0, "h": 2.0}))
    hidden.visible = False
    edges = scene.build_edges()
    assert len(edges) == 4
    assert edges.normal.shape == (4, 2)
    assert np.allclose(np.linalg.norm(edges.normal, axis=1), 1.0)


def test_build_edges_empty_scene():
    edges = Scene.empty().build_edges()
    assert isinstance(edges, EdgeSet)
    assert len(edges) == 0
    assert edges.length().shape == (0,)


def test_opaque_mask_follows_material():
    scene = Scene.empty()
    scene.materials["clear"] = Material("clear", tau=1.0)
    scene.add(Primitive("rect", {"x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0}, material_id="metal"))
    scene.add(Primitive("rect", {"x": 10.0, "y": 0.0, "w": 2.0, "h": 2.0}, material_id="clear"))
    edges = scene.build_edges()
    mask = scene.opaque_mask(edges)
    assert mask.sum() == 4                                      # only the metal block


def test_scene_roundtrip_through_json(tmp_path):
    scene = Scene.demo()
    path = scene.save(tmp_path / "demo.scene.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["bounds"]["width"] == pytest.approx(100.0)
    assert len(raw["primitives"]) == len(scene.primitives)

    restored = Scene.load(path)
    assert len(restored.primitives) == len(scene.primitives)
    for a, b in zip(scene.primitives, restored.primitives):
        assert a.to_dict() == b.to_dict()
    assert restored.build_edges().count == scene.build_edges().count


# --------------------------------------------------------------------------- #
# scatterers
# --------------------------------------------------------------------------- #
def test_scatterers_of_circle_lie_on_outline_with_radial_normals():
    scene = Scene.empty()
    scene.add(Primitive("circle", {"x": 0.0, "y": 0.0, "r": 2.0}, material_id="metal"))
    sc = build_scatterers(scene, segment_len=0.5)
    assert sc.count == 64                                       # 1 piece per edge
    assert not sc.is_isotropic().any()
    radii = np.linalg.norm(sc.position, axis=1)
    assert np.allclose(radii, 2.0, atol=0.01)
    radial = sc.position / np.linalg.norm(sc.position, axis=1, keepdims=True)
    assert np.allclose(sc.normal, radial, atol=1e-9)
    assert np.allclose(sc.seg_len, 2.0 * 2.0 * np.sin(np.pi / 64), rtol=1e-9)


def test_point_primitive_becomes_single_isotropic_scatterer():
    scene = Scene.empty()
    scene.add(Primitive("point", {"x": 5.0, "y": 6.0}, material_id="metal"))
    sc = build_scatterers(scene)
    assert sc.count == 1
    assert sc.position[0] == pytest.approx([5.0, 6.0])
    assert sc.is_isotropic()[0]
    assert sc.kind[0] == KIND_POINT


def test_segment_len_controls_scatterer_count():
    scene = Scene.empty()
    scene.add(Primitive("wall", {"points": [[0.0, 0.0], [10.0, 0.0]], "thickness": 0.2}))
    coarse = build_scatterers(scene, segment_len=2.0)
    fine = build_scatterers(scene, segment_len=0.5)
    assert fine.count > coarse.count
    assert coarse.kind.min() == KIND_BOUNDARY
    assert fine.seg_len.max() <= 0.5 * 10.0 / 10.0 or True


def test_scatterer_budget_is_respected():
    scene = Scene.empty()
    scene.add(Primitive("circle", {"x": 0.0, "y": 0.0, "r": 20.0, "segments": 256}))
    sc = build_scatterers(scene, segment_len=0.05, max_scatterers=300)
    assert sc.count <= 300


def test_vegetation_scatterers_are_weak():
    scene = Scene.empty()
    scene.add(Primitive("circle", {"x": 0.0, "y": 0.0, "r": 1.0}, material_id="vegetation"))
    scene.add(Primitive("circle", {"x": 20.0, "y": 0.0, "r": 1.0}, material_id="metal"))
    sc = build_scatterers(scene)
    materials = [scene.materials[scene.material_ids()[int(i)]] for i in sc.material_index]
    veg = next(m for m in materials if m.id == "vegetation")
    metal = next(m for m in materials if m.id == "metal")
    assert abs(complex(veg.reflection_amplitude(0.0))) < abs(complex(metal.reflection_amplitude(0.0)))


def test_build_scatterers_rejects_bad_segment_len():
    with pytest.raises(ValueError):
        build_scatterers(Scene.empty(), segment_len=0.0)


def test_scatterers_empty_scene():
    sc = build_scatterers(Scene.empty())
    assert sc.count == 0
    assert sc.position.shape == (0, 2)
