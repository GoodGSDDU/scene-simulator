"""Geometry unit tests (ray/segment intersection, mirroring, polygon normals)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_sim.core import geometry as geo


def test_ray_segment_intersection_basic():
    t, u, hit = geo.ray_segment_intersections(
        origin=np.array([0.0, 0.0]),
        dirs=np.array([[1.0, 0.0]]),
        a=np.array([[2.0, -1.0]]),
        b=np.array([[2.0, 1.0]]),
    )
    assert hit[0, 0]
    assert t[0, 0] == pytest.approx(2.0)
    assert u[0, 0] == pytest.approx(0.5)


def test_ray_segment_intersection_misses():
    # segment entirely behind the ray origin
    _, _, hit = geo.ray_segment_intersections(
        np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
        np.array([[-3.0, -1.0]]), np.array([[-3.0, 1.0]]),
    )
    assert not hit[0, 0]

    # segment beyond the segment ends
    _, _, hit = geo.ray_segment_intersections(
        np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
        np.array([[2.0, 3.0]]), np.array([[2.0, 4.0]]),
    )
    assert not hit[0, 0]


def test_ray_segment_intersection_parallel():
    _, _, hit = geo.ray_segment_intersections(
        np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
        np.array([[1.0, 1.0]]), np.array([[3.0, 1.0]]),
    )
    assert not hit[0, 0]


def test_ray_segment_intersection_vectorised():
    origin = np.array([0.0, 0.0])
    angles = np.deg2rad([0.0, 45.0, 90.0])
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    a = np.array([[5.0, -5.0]])
    b = np.array([[5.0, 5.0]])
    t, u, hit = geo.ray_segment_intersections(origin, dirs, a, b)
    assert hit.shape == (3, 1)
    # 0 deg and 45 deg hit the vertical segment; 90 deg runs parallel to it
    assert hit[0, 0] and hit[1, 0]
    assert not hit[2, 0]
    assert t[0, 0] == pytest.approx(5.0)
    assert t[1, 0] == pytest.approx(5.0 * np.sqrt(2.0))
    assert u[1, 0] == pytest.approx(1.0)                     # hits the endpoint
    assert np.isnan(t[2, 0])


def test_mirror_point_across_axis():
    a = np.array([-1.0, 0.0])
    b = np.array([1.0, 0.0])
    m = geo.mirror_point(np.array([1.0, 1.0]), a, b)
    assert m == pytest.approx([1.0, -1.0])


def test_mirror_points_batch_matches_scalar():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])                     # 45 degree line
    pts = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
    batch = geo.mirror_points(pts, a, b)
    for i in range(pts.shape[0]):
        assert batch[i] == pytest.approx(geo.mirror_point(pts[i], a, b))


def test_reflect_direction():
    d = np.array([1.0, -1.0]) / np.sqrt(2.0)
    n = np.array([0.0, 1.0])
    r = geo.reflect_direction(d, n)
    assert r == pytest.approx([1.0, 1.0] / np.sqrt(2.0))


def test_rect_polygon_is_ccw_and_normals_outward():
    poly = geo.rect_polygon(0.0, 0.0, 2.0, 2.0)
    assert poly.shape == (4, 2)
    assert geo.polygon_signed_area(poly) > 0.0

    a, b = geo.polygon_edges(poly)
    n = geo.edge_normals_outward(poly)
    for i in range(a.shape[0]):
        mid = 0.5 * (a[i] + b[i])
        assert not geo.point_in_polygon(mid + 0.02 * n[i], poly), "normal points inward"
        assert geo.point_in_polygon(mid - 0.02 * n[i], poly), "inward normal not inside"


def test_circle_polygon_area():
    poly = geo.circle_polygon(1.0, 2.0, 3.0, 1024)
    assert abs(geo.polygon_signed_area(poly)) == pytest.approx(np.pi * 9.0, rel=5e-5)


def test_rect_polygon_rotation_and_centre():
    poly = geo.rect_polygon(5.0, 7.0, 4.0, 2.0, np.deg2rad(30.0))
    assert poly.mean(axis=0) == pytest.approx([5.0, 7.0])
    # rotated bounding box must be larger than the axis-aligned one
    span = poly.max(axis=0) - poly.min(axis=0)
    assert span[0] > 4.0 and span[1] > 2.0


def test_wall_polygon_has_thickness():
    poly = geo.wall_polygon(np.array([[0.0, 0.0], [10.0, 0.0]]), 0.4)
    assert poly.shape == (4, 2)
    assert abs(geo.polygon_signed_area(poly)) == pytest.approx(10.0 * 0.4)
    ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
    assert ymax - ymin == pytest.approx(0.4)


def test_wall_polygon_rejects_single_point():
    with pytest.raises(ValueError):
        geo.wall_polygon(np.array([[0.0, 0.0]]), 0.3)


def test_subdivide_segment():
    centres, piece = geo.subdivide_segment(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.3)
    assert centres.shape == (4, 2)
    assert piece == pytest.approx(0.25)
    assert centres[0, 0] == pytest.approx(0.125)
    assert centres[-1, 0] == pytest.approx(0.875)


def test_subdivide_segment_single_piece():
    centres, piece = geo.subdivide_segment(np.array([0.0, 0.0]), np.array([0.1, 0.0]), 0.5)
    assert centres.shape == (1, 2)
    assert piece == pytest.approx(0.1)


def test_point_in_polygon():
    poly = geo.rect_polygon(0.0, 0.0, 4.0, 4.0)
    assert geo.point_in_polygon(np.array([0.0, 0.0]), poly)
    assert not geo.point_in_polygon(np.array([3.0, 0.0]), poly)


def test_point_segment_distance():
    assert geo.point_segment_distance(
        np.array([0.0, 1.0]), np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    ) == pytest.approx(1.0)
    assert geo.point_segment_distance(
        np.array([3.0, 0.0]), np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    ) == pytest.approx(1.0)


def test_unit_handles_zero_vector():
    out = geo.unit(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx([0.6, 0.8])
