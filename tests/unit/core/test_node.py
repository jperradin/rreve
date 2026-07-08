"""Unit tests for ``rreve.core.node.Node``."""

import numpy as np
import pytest

from rreve.core.node import Node, ATOMIC_DATA


def test_mass_and_correlation_length_lookup():
    n = Node(symbol="Si", node_id=0, position=np.zeros(3))
    mass, b = ATOMIC_DATA["Si"]
    assert n.mass == pytest.approx(mass)
    assert n.correlation_length == pytest.approx(b)


def test_unknown_symbol_defaults_to_zero_mass():
    n = Node(symbol="Xx", node_id=1, position=np.zeros(3))
    assert n.mass == 0.0
    assert n.correlation_length == 0.0


def test_post_init_defaults():
    n = Node(symbol="O", node_id=2, position=None)
    assert np.allclose(n.position, np.zeros(3))
    assert n.neighbors == []
    assert n.coordination == 0
    assert n.form == ""


def test_add_neighbor_and_coordination():
    a = Node(symbol="Si", node_id=0, position=np.zeros(3))
    b = Node(symbol="O", node_id=1, position=np.array([1.6, 0.0, 0.0]))
    a.add_neighbor(b)
    assert b in a.neighbors
    a.set_coordination(1)
    assert a.coordination == 1


def test_set_polyhedricity():
    n = Node(symbol="Si", node_id=0, position=np.zeros(3))
    n.set_polyhedricity(0.42)
    assert n.polyhedricity == pytest.approx(0.42)


def test_get_neighbors_positions_by_element():
    c = Node(symbol="Si", node_id=0, position=np.zeros(3))
    o1 = Node(symbol="O", node_id=1, position=np.array([1.0, 0.0, 0.0]))
    o2 = Node(symbol="O", node_id=2, position=np.array([0.0, 1.0, 0.0]))
    si = Node(symbol="Si", node_id=3, position=np.array([3.0, 0.0, 0.0]))
    for n in (o1, o2, si):
        c.add_neighbor(n)
    pos = c.get_neighbors_positions_by_element("O")
    assert pos.shape == (2, 3)


def test_calculate_angles_si_o_si():
    # Si - O - Si bent at 90 degrees: angle list should contain ~90
    center = Node(symbol="Si", node_id=0, position=np.array([0.0, 0.0, 0.0]))
    bridge = Node(symbol="O", node_id=1, position=np.array([1.0, 0.0, 0.0]))
    other = Node(symbol="Si", node_id=2, position=np.array([1.0, 1.0, 0.0]))
    bridge.add_neighbor(center)
    bridge.add_neighbor(other)
    center.add_neighbor(bridge)
    lattice = np.diag([10.0, 10.0, 10.0]).astype(float)
    center.calculate_angles(lattice)
    assert len(center.angles) == 1
    assert center.angles[0] == pytest.approx(90.0, abs=1e-6)


def test_wrap_position_staticmethod():
    lattice = np.diag([10.0, 10.0, 10.0]).astype(float)
    out = Node.wrap_position(np.array([12.0, 0.0, 0.0]), lattice)
    assert np.allclose(out, [2.0, 0.0, 0.0])


def test_node_ordering_and_str():
    # dataclass order compares (symbol, node_id, ...); same symbol -> by node_id
    a = Node(symbol="Si", node_id=0, position=np.zeros(3))
    b = Node(symbol="Si", node_id=1, position=np.zeros(3))
    assert a < b
    assert "Node 0" in str(a)
    assert "Si" in repr(a)
