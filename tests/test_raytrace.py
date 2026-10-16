"""Ray tracing tests -- occlusion (T6/T7) and image-method multipath (T8).

The multipath expectations are derived analytically in the docstrings so a
sign or image-method mistake cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core.radar_config import RadarConfig
from radar_sim.core.scene import Primitive, Scene
from radar_sim.core.scatterers import KIND_POINT, build_scatterers
from radar_sim.core.raytrace import trace_paths

RADAR = np.array([0.0, 0.0])


def _point_index(sc) -> int:
    return int(np.nonzero(sc.kind == KIND_POINT)[0][0])


def _flat_wall(x: float, y0: float, y1: float, material: str = "metal") -> Primitive:
    return Primitive(
        "wall", {"points": [[x, y0], [x, y1]], "thickness": 0.4},
        material_id=material, name="wall",
    )


# --------------------------------------------------------------------------- #
# direct paths
# --------------------------------------------------------------------------- #
def test_direct_path_range_and_angle():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 3.0, "y": 4.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    assert paths.count == 1
    assert paths.bounce[0] == 0
    assert paths.equiv_range[0] == pytest.approx(5.0)
    assert paths.theta_rad[0] == pytest.approx(np.arctan2(4.0, 3.0))
    assert paths.atten[0] == pytest.approx(1.0)
    assert paths.gain[0] == pytest.approx(1.0 + 0.0j)


def test_direct_path_radial_velocity_sign():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    # radar driving at +1 m/s straight at a static target -> approaching -> negative
    paths = trace_paths(scene, sc, RADAR, radar_vel=np.array([1.0, 0.0]), max_order=0)
    assert paths.v_eff[0] == pytest.approx(-1.0)
    # radar driving away -> positive (positive == moving away)
    paths = trace_paths(scene, sc, RADAR, radar_vel=np.array([-1.0, 0.0]), max_order=0)
    assert paths.v_eff[0] == pytest.approx(1.0)


def test_direct_path_incidence_cosine():
    """A point target is isotropic (cos psi forced to 1)."""
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 0.0, "y": 10.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    assert sc.is_isotropic()[paths.scatterer[0]]
    assert paths.cos_psi[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# T6 -- occlusion
# --------------------------------------------------------------------------- #
def test_opaque_wall_blocks_direct_path_t6():
    scene = Scene.empty(30.0, 30.0)
    scene.add(_flat_wall(5.0, -5.0, 5.0, material="metal"))
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    paths = trace_paths(scene, sc, RADAR, max_order=0)
    hits = (paths.bounce == 0) & (paths.scatterer == target)
    assert not hits.any(), "opaque wall must hide the target"


def test_target_is_visible_without_the_wall():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    assert paths.count == 1


# --------------------------------------------------------------------------- #
# T7 -- transmission through transparent material
# --------------------------------------------------------------------------- #
def test_transparent_wall_attenuates_but_does_not_block_t7():
    scene = Scene.empty(30.0, 30.0)
    scene.add(_flat_wall(5.0, -5.0, 5.0, material="glass"))     # tau = 0.85
    scene.add(Primitive("point", {"x": 10.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    paths = trace_paths(scene, sc, RADAR, max_order=0)
    hits = (paths.bounce == 0) & (paths.scatterer == target)
    assert hits.any(), "glass must not hide the target"
    atten = float(paths.atten[hits][0])
    assert atten < 1.0
    # crossing the slab means two edge crossings: sqrt(tau)^2 == tau
    assert atten == pytest.approx(0.85, abs=0.05)


# --------------------------------------------------------------------------- #
# T8 -- image-method multipath
# --------------------------------------------------------------------------- #
def test_order1_ghost_range_matches_image_method_t8():
    """radar (0,0), reflector edge y=4.9, target (5,3).

    image of the radar = (0, 9.8); one-way path = |(5,3) - (0,9.8)| = 8.4404 m
    versus a direct range of 5.8310 m -- the ghost is *farther*, as expected.
    """
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[0.0, 5.0], [10.0, 5.0]], "thickness": 0.2},
                        material_id="metal", name="reflector"))
    scene.add(Primitive("point", {"x": 5.0, "y": 3.0}, material_id="metal", name="target"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    paths = trace_paths(scene, sc, RADAR, max_order=1)
    direct = (paths.bounce == 0) & (paths.scatterer == target)
    ghost = (paths.bounce == 1) & (paths.scatterer == target)

    assert direct.any(), "direct path should still be clear"
    assert ghost.any(), "the plate should produce exactly one specular echo"
    direct_range = float(paths.equiv_range[direct][0])
    ghost_range = float(paths.equiv_range[ghost][0])
    assert direct_range == pytest.approx(5.8310, abs=0.01)
    assert ghost_range == pytest.approx(8.4404, abs=0.02)
    assert ghost_range > direct_range


def test_order1_departure_angle_points_at_the_reflection_spot():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[0.0, 5.0], [10.0, 5.0]], "thickness": 0.2},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 5.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    paths = trace_paths(scene, sc, RADAR, max_order=1)
    ghost = (paths.bounce == 1) & (paths.scatterer == target)
    # the departure azimuth must aim at the reflection point (3.6029, 4.9)
    expected = np.arctan2(4.9, 3.6029)
    assert float(paths.theta_rad[ghost][0]) == pytest.approx(expected, abs=1e-3)
    # and the path length equals leg1 + leg2
    leg1 = np.hypot(3.6029, 4.9)
    leg2 = np.hypot(5.0 - 3.6029, 3.0 - 4.9)
    assert float(paths.equiv_range[ghost][0]) == pytest.approx(leg1 + leg2, abs=1e-3)


def test_multipath_gain_follows_material():
    """Compare reflectors at nearly normal incidence.

    Geometry: reflector face y=4.9 spanning x in [-5, 5], radar (0, 0),
    target (0, 3) -- i.e. straight below the reflection point, so the
    incidence angle at the plate is ~0 and the specular lobe is fully excited.
    A tilted geometry would be unfair: the smooth metal lobe (sigma_psi = 6 deg)
    dies long before the rough/vegetation diffuse term does.
    """
    def ghost_gain(material: str) -> float:
        scene = Scene.empty(30.0, 30.0)
        scene.add(Primitive("wall", {"points": [[-5.0, 5.0], [5.0, 5.0]], "thickness": 0.2},
                            material_id=material))
        scene.add(Primitive("point", {"x": 0.0, "y": 3.0}, material_id="metal"))
        sc = build_scatterers(scene)
        target = _point_index(sc)
        paths = trace_paths(scene, sc, RADAR, max_order=1)
        ghost = (paths.bounce == 1) & (paths.scatterer == target)
        assert ghost.any()
        return float(np.abs(paths.gain[ghost][0]))

    metal = ghost_gain("metal")
    vegetation = ghost_gain("vegetation")
    assert metal == pytest.approx(0.95, abs=0.02)          # rho of a facing plate
    assert 20.0 * np.log10(metal / vegetation) > 10.0


def test_oblique_incidence_weakens_the_specular_ghost():
    """Lateral offset => ~18 deg incidence on the plate => the 6 deg specular
    lobe is almost gone, so the ghost collapses."""
    def ghost_gain(target_xy: tuple[float, float]) -> float:
        scene = Scene.empty(30.0, 30.0)
        scene.add(Primitive("wall", {"points": [[-5.0, 5.0], [5.0, 5.0]], "thickness": 0.2},
                            material_id="metal"))
        scene.add(Primitive("point", {"x": target_xy[0], "y": target_xy[1]},
                            material_id="metal"))
        sc = build_scatterers(scene)
        paths = trace_paths(scene, sc, RADAR, max_order=1)
        ghost = (paths.bounce == 1) & (paths.scatterer == _point_index(sc))
        assert ghost.any()
        return float(np.abs(paths.gain[ghost][0]))

    facing = ghost_gain((0.0, 3.0))
    oblique = ghost_gain((3.0, 0.5))
    assert oblique < 0.2 * facing


def test_max_order_zero_returns_only_direct():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[0.0, 5.0], [10.0, 5.0]], "thickness": 0.2},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 5.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    assert set(np.unique(paths.bounce)) <= {0}


def test_order2_path_exists_and_is_longer_than_order1_t8():
    """Corner reflector geometry, solved analytically.

    Wall A reflective edge x=7.9, wall B reflective edge y=7.9, target (3, 5).
      image1 = (15.8, 0)   mirror of radar across A
      image2 = (15.8, 15.8) mirror of image1 across B
      P2 crosses y=7.9 at (6.437, 7.9);  P1 crosses x=7.9 at (7.9, 6.665)
      one-way path = |(3,5) - (15.8,15.8)| = 16.747 m
    """
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[8.0, -10.0], [8.0, 10.0]], "thickness": 0.2},
                        material_id="metal", name="wallA"))
    scene.add(Primitive("wall", {"points": [[-10.0, 8.0], [10.0, 8.0]], "thickness": 0.2},
                        material_id="metal", name="wallB"))
    scene.add(Primitive("point", {"x": 3.0, "y": 5.0}, material_id="metal", name="target"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    paths = trace_paths(scene, sc, RADAR, max_order=2)
    double = (paths.bounce == 2) & (paths.scatterer == target)
    assert double.any(), "a corner must produce a two-bounce path"
    one = (paths.bounce == 1) & (paths.scatterer == target)
    direct = (paths.bounce == 0) & (paths.scatterer == target)

    double_range = float(paths.equiv_range[double].min())
    assert double_range == pytest.approx(16.747, abs=0.05)
    assert double_range > float(paths.equiv_range[one].min())
    assert double_range > float(paths.equiv_range[direct][0])


def test_counts_by_bounce_reports_groups():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[0.0, 5.0], [10.0, 5.0]], "thickness": 0.2},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 5.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=1)
    counts = paths.counts_by_bounce()
    assert 0 in counts
    assert counts.get(1, 0) >= 1
    assert sum(counts.values()) == paths.count


def test_max_multipath_paths_is_respected():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("wall", {"points": [[0.0, 5.0], [10.0, 5.0]], "thickness": 0.2},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 5.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=1, max_multipath_paths=3)
    assert (paths.bounce == 1).sum() <= 3


# --------------------------------------------------------------------------- #
# candidate-wall selection -- a nearby circle must not starve the long walls
# --------------------------------------------------------------------------- #
def test_long_wall_beats_a_nearby_circle_as_reflector():
    """Regression: ranking reflectors by distance let a 64-edge circle fill the
    quota, so a long metal wall produced no specular ghost at all."""
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("circle", {"x": 4.0, "y": 1.0, "r": 1.2}, material_id="wood"))
    scene.add(Primitive("wall", {"points": [[-5.0, 20.0], [25.0, 20.0]],
                                 "thickness": 0.3}, material_id="metal"))
    scene.add(Primitive("point", {"x": 8.0, "y": 16.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=1)
    assert (paths.bounce == 1).sum() >= 1, (
        "the long wall must still act as a specular reflector"
    )


def test_trace_supports_two_bounce_paths_in_a_corridor():
    """A real corridor must produce both single- and double-bounce ghosts."""
    scene = Scene.empty(40.0, 60.0)
    scene.add(Primitive("wall", {"points": [[8.0, 20.0], [32.0, 20.0]], "thickness": 0.4},
                        material_id="concrete"))
    scene.add(Primitive("wall", {"points": [[8.0, 40.0], [32.0, 40.0]], "thickness": 0.4},
                        material_id="metal"))
    scene.add(Primitive("wall", {"points": [[32.0, 20.0], [32.0, 40.0]], "thickness": 0.4},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 24.0, "y": 32.0}, material_id="metal"))
    sc = build_scatterers(scene, segment_len=1.0)
    cfg = RadarConfig()
    paths = trace_paths(scene, sc, np.array([7.5, 30.0]), cfg=cfg, max_order=2)
    counts = paths.counts_by_bounce()
    assert counts.get(1, 0) > 0
    assert counts.get(2, 0) > 0, "double bounce missing -- wall selection regressed"
    assert float(paths.equiv_range.max()) <= cfg.max_range_m


# --------------------------------------------------------------------------- #
# range gate
# --------------------------------------------------------------------------- #
def test_paths_beyond_max_range_are_dropped():
    """A long path folds back into the range window and would appear as a
    phantom target, so it must be gated out."""
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("wall", {"points": [[5.0, -20.0], [5.0, -12.0]], "thickness": 0.4},
                        material_id="metal"))
    scene.add(Primitive("point", {"x": 30.0, "y": 3.0}, material_id="metal"))
    sc = build_scatterers(scene)
    target = _point_index(sc)

    ungated = trace_paths(scene, sc, RADAR, max_order=0, max_range_m=None)
    gated = trace_paths(scene, sc, RADAR, max_order=0, max_range_m=10.0)

    assert ((ungated.bounce == 0) & (ungated.scatterer == target)).any()
    assert not ((gated.bounce == 0) & (gated.scatterer == target)).any(), \
        "the 30 m target must be gated out of a 10 m window"
    # everything left in the gated set must be inside the window (the wall's own
    # scatterers are all 13-21 m away, so the gated set may legitimately be empty)
    assert gated.count == 0 or float(gated.equiv_range.max()) <= 10.0


def test_default_gate_uses_the_config_max_range():
    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("point", {"x": 40.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    cfg = RadarConfig()
    near = trace_paths(scene, sc, RADAR, cfg=cfg)
    assert near.count == 1
    assert float(near.equiv_range[0]) == pytest.approx(40.0, abs=0.01)

    small = RadarConfig(**{**RadarConfig().to_dict(), "sample_rate_hz": 2.0e6})
    far = trace_paths(scene, sc, RADAR, cfg=small)
    assert far.count == 0, "40 m target must be gated out of a 10 m window"


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_empty_scatterer_set_returns_empty_paths():
    scene = Scene.empty()
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR)
    assert paths.count == 0
    assert paths.counts_by_bounce() == {}


def test_empty_scene_keeps_direct_paths():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 1.0, "y": 0.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=2)
    assert paths.count == 1
    assert paths.bounce[0] == 0


def test_select_and_concat_helpers():
    scene = Scene.empty(30.0, 30.0)
    scene.add(Primitive("point", {"x": 3.0, "y": 4.0}, material_id="metal"))
    scene.add(Primitive("point", {"x": 6.0, "y": 8.0}, material_id="metal"))
    sc = build_scatterers(scene)
    paths = trace_paths(scene, sc, RADAR, max_order=0)
    assert paths.count == 2
    first = paths.select(np.array([True, False]))
    assert first.count == 1
    assert first.concat(paths.select(np.array([False, True]))).count == 2
    from radar_sim.core.raytrace import EchoPaths

    assert EchoPaths.empty().concat(first).count == 1
    assert first.concat(EchoPaths.empty()).count == 1
