"""Unit tests for ``rreve.utils.geometry``.

All oracles are hand-computed in a 10x10x10 cubic box unless stated otherwise.
"""

import numpy as np
import pytest

from rreve.utils import geometry as g

LAT = np.diag([10.0, 10.0, 10.0]).astype(float)


# --------------------------------------------------------------------------- #
# Wrapping
# --------------------------------------------------------------------------- #
def test_wrap_position_inside_box_unchanged():
    p = np.array([2.0, 3.0, 4.0])
    assert np.allclose(g.wrap_position(p, LAT), p)


def test_wrap_position_outside_box():
    # frac [1.2, -0.1, 0.5] -> [0.2, 0.9, 0.5] -> cart [2, 9, 5]
    p = np.array([12.0, -1.0, 5.0])
    assert np.allclose(g.wrap_position(p, LAT), [2.0, 9.0, 5.0])


def test_wrap_positions_batch():
    pts = np.array([[12.0, -1.0, 5.0], [2.0, 3.0, 4.0]])
    out = g.wrap_positions(pts, LAT, None)
    assert np.allclose(out, [[2.0, 9.0, 5.0], [2.0, 3.0, 4.0]])


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def test_direct_distance():
    assert g.calculate_direct_distance(np.zeros(3), np.array([3.0, 4.0, 0.0])) == 5.0


def test_pbc_distance_uses_minimum_image():
    # 0 and 9 along z in a box of 10 -> nearest image is 1.0 apart
    d = g.calculate_pbc_distance(np.zeros(3), np.array([0.0, 0.0, 9.0]), LAT)
    assert d == pytest.approx(1.0)


def test_pbc_distance_matches_direct_inside_box():
    a, b = np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.6])
    assert g.calculate_pbc_distance(a, b, LAT) == pytest.approx(1.6)


def test_pbc_distances_batch():
    p1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    p2 = np.array([[0.0, 0.0, 9.0], [1.0, 1.0, 2.6]])
    out = g.calculate_pbc_distances_batch(p1, p2, LAT, None)
    assert np.allclose(out, [1.0, 1.6])


def test_pbc_cv_distances_batch():
    center = np.array([5.0, 5.0, 5.0])
    verts = np.array([[5.0, 5.0, 6.6], [5.0, 6.6, 5.0]])
    out = g.calculate_pbc_cv_distances_batch(center, verts, LAT)
    assert np.allclose(out, [1.6, 1.6])


def test_pbc_dot_distances_combinations_count_and_value():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    out = g.calculate_pbc_dot_distances_combinations(pts, LAT)
    assert out.shape == (3,)  # C(3,2)
    assert sorted(np.round(out, 6)) == [1.0, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# Angles
# --------------------------------------------------------------------------- #
def test_direct_angle_right_angle():
    a = g.calculate_direct_angle(
        np.array([1.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])
    )
    assert a == pytest.approx(90.0)


def test_pbc_angle_right_angle():
    a = g.calculate_pbc_angle(
        np.array([1.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]), LAT
    )
    assert a == pytest.approx(90.0)


def test_pbc_angle_straight_across_boundary():
    # apex at origin, arms to +z and to the -z image of [0,0,9]
    a = g.calculate_pbc_angle(
        np.array([0.0, 0.0, 1.0]), np.zeros(3), np.array([0.0, 0.0, 9.0]), LAT
    )
    assert a == pytest.approx(180.0)


def test_pbc_angle_combinations_count():
    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    out = g.calculate_pbc_angle_combinations(pts, LAT)
    # n*(n-1)*(n-2)//2 for n=4 -> 12
    assert out.shape == (12,)
    assert np.all(out >= 0.0)


def test_pbc_cv_angle_combinations_regular_tetrahedron():
    s = 1.6 / np.sqrt(3.0)
    c = np.array([5.0, 5.0, 5.0])
    verts = c + s * np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    )
    angles = g.calculate_pbc_cv_angle_combinations(c, verts, LAT)
    assert angles.shape == (6,)  # C(4,2)
    assert np.allclose(angles, 109.4712, atol=1e-3)


def test_pbc_angle_around_is_stub():
    # the helper is an unfinished stub that returns None
    assert (
        g.calculate_pbc_angle_around(np.zeros(3), np.zeros(3), np.zeros(3), LAT)
        is None
    )


# --------------------------------------------------------------------------- #
# Coordinate transforms
# --------------------------------------------------------------------------- #
def test_cartesian_fractional_roundtrip():
    pos = np.array([1.23, 4.56, 7.89])
    frac = g.cartesian_to_fractional(pos, LAT)
    back = g.fractional_to_cartesian(frac, LAT)
    assert np.allclose(back, pos)


def test_cartesian_to_fractional_scaling():
    assert np.allclose(g.cartesian_to_fractional(np.array([5.0, 5.0, 5.0]), LAT), 0.5)


# --------------------------------------------------------------------------- #
# Gyration radius
# --------------------------------------------------------------------------- #
def test_gyration_radius_two_points():
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rg = g.calculate_gyration_radius(pts, np.zeros(3))
    assert rg == pytest.approx(1.0)


def test_gyration_radius_empty():
    rg = g.calculate_gyration_radius(np.empty((0, 3)), np.zeros(3))
    assert rg == 0.0


# --------------------------------------------------------------------------- #
# Histogram & neighbor filters
# --------------------------------------------------------------------------- #
def test_fast_histogram_binning():
    d = np.array([0.5, 1.5, 1.6, 9.9, 11.0])  # 11.0 outside r_max
    hist = g.fast_histogram(d, 10.0, 10)
    assert hist.sum() == 4
    assert hist[0] == 1  # 0.5
    assert hist[1] == 2  # 1.5, 1.6


def test_filter_neighbors_direct_batch():
    target = np.zeros(3)
    cands = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rcut2 = np.array([4.0, 4.0])  # cutoff 2.0
    keep, dists = g.filter_neighbors_direct_batch(target, cands, rcut2)
    assert list(keep) == [True, False]
    assert dists[0] == pytest.approx(1.0)


def test_filter_neighbors_pbc_batch_wraps():
    target = np.zeros(3)
    cands = np.array([[0.0, 0.0, 9.0]])  # nearest image 1.0 away
    inv = np.linalg.inv(LAT)
    keep, dists = g.filter_neighbors_pbc_batch(target, cands, LAT, inv, np.array([4.0]))
    assert keep[0]
    assert dists[0] == pytest.approx(1.0)


def test_filter_negative_cutoff_skipped():
    target = np.zeros(3)
    cands = np.array([[1.0, 0.0, 0.0]])
    keep, _ = g.filter_neighbors_direct_batch(target, cands, np.array([-1.0]))
    assert not keep[0]


# --------------------------------------------------------------------------- #
# Structure-factor components
# --------------------------------------------------------------------------- #
def test_calculate_components_shapes():
    q = np.array([0.1, 0.2])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    qcos, qsin = g.calculate_components(q, q, q, positions, None)
    assert qcos.shape == q.shape
    assert qsin.shape == q.shape


# --------------------------------------------------------------------------- #
# Polyhedricity / tetrahedricity metrics
# --------------------------------------------------------------------------- #
def test_tetrahedricity_regular_is_zero():
    # all 6 edges equal -> perfectly regular -> 0
    d = np.full(6, 2.5)
    assert g.calculate_tetrahedricity(d) == pytest.approx(0.0)


def test_tetrahedricity_irregular_positive():
    d = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    assert g.calculate_tetrahedricity(d) > 0.0


def test_tetrahedricity_angles_zero_at_ideal():
    angles = np.array([109.47, 109.47, 109.47])
    assert g.calculate_tetrahedricity_angles(angles, 109.47) == pytest.approx(0.0)


def test_tetrahedricity_angles_positive_off_ideal():
    angles = np.array([100.0, 120.0])
    assert g.calculate_tetrahedricity_angles(angles, 109.47) > 0.0


def test_square_based_pyramid_runs():
    d = np.array([1.0, 1.0, 1.0, 1.0, 1.4, 1.4])
    assert g.calculate_square_based_pyramid(d) >= 0.0


def test_triangular_bipyramid_runs():
    d = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.2])
    assert g.calculate_triangular_bipyramid(d) >= 0.0


def test_octahedricity_runs():
    d = np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4])
    assert g.calculate_octahedricity(d) >= 0.0


def test_warmup_jit_runs():
    # exercises every jitted entry point once; must not raise
    g.warmup_jit()
