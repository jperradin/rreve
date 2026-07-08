"""Unit tests for ``rreve.core.frame.Frame``."""

import numpy as np
import pytest

from rreve.core.frame import Frame, NodesData


def _raw_frame(symbols, positions, lattice=None, settings=None):
    lattice = np.diag([10.0, 10.0, 10.0]).astype(float) if lattice is None else lattice
    kwargs = dict(
        frame_id=0,
        nodes=[],
        lattice=lattice,
        _lattice_str="",
        _data={"symbol": symbols, "position": [np.asarray(p, float) for p in positions]},
    )
    if settings is not None:
        kwargs["_settings"] = settings
    return Frame(**kwargs)


def test_initialize_nodes_builds_node_list():
    f = _raw_frame(["Si", "O"], [[0, 0, 0], [1.6, 0, 0]])
    f.initialize_nodes()
    assert len(f) == 2
    assert f.nodes[0].symbol == "Si"
    assert f.nodes[1].node_id == 1


def test_initialize_nodes_length_mismatch_raises():
    f = Frame(
        frame_id=0,
        nodes=[],
        lattice=np.diag([10.0, 10.0, 10.0]),
        _lattice_str="",
        _data={"symbol": ["Si"], "position": [np.zeros(3), np.ones(3)]},
    )
    with pytest.raises(ValueError):
        f.initialize_nodes()


def test_get_nodes_by_element_and_unique():
    f = _raw_frame(["Si", "O", "O"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    f.initialize_nodes()
    assert len(f.get_nodes_by_element("O")) == 2
    assert set(f.get_unique_elements()) == {"Si", "O"}


def test_get_node_by_id():
    f = _raw_frame(["Si", "O"], [[0, 0, 0], [1, 0, 0]])
    f.initialize_nodes()
    assert f.get_node_by_id(1).symbol == "O"
    assert f.get_node_by_id(99) is None


def test_get_positions_and_by_element():
    f = _raw_frame(["Si", "O"], [[0, 0, 0], [1, 2, 3]])
    f.initialize_nodes()
    assert f.get_positions().shape == (2, 3)
    by_el = f.get_positions_by_element()
    assert np.allclose(by_el["O"], [[1, 2, 3]])


def test_nodes_data_populated_after_init():
    f = _raw_frame(["Si", "O"], [[0, 0, 0], [1.6, 0, 0]])
    f.initialize_nodes()
    assert isinstance(f.nodes_data, NodesData)
    assert "Si" in f.nodes_data.wrapped_positions
    assert "Si" in f.nodes_data.correlation_lengths


def test_set_lattice_validations():
    f = _raw_frame(["Si"], [[0, 0, 0]])
    with pytest.raises(ValueError):
        f.set_lattice(np.eye(2))  # wrong shape
    with pytest.raises(ValueError):
        f.set_lattice(np.zeros((3, 3)))  # singular
    good = np.diag([5.0, 5.0, 5.0])
    f.set_lattice(good)
    assert np.allclose(f.get_lattice(), good)


def test_num_nodes_and_repr():
    f = _raw_frame(["Si", "O"], [[0, 0, 0], [1, 0, 0]])
    f.initialize_nodes()
    assert f.get_num_nodes() == 2
    assert "Frame 0" in str(f)
    assert "lattice" in repr(f)


def test_frame_type_validation():
    with pytest.raises(TypeError):
        Frame(frame_id=0, nodes="notalist", lattice=np.eye(3), _lattice_str="")
    with pytest.raises(TypeError):
        Frame(frame_id=0, nodes=[], lattice=[[1, 0, 0]], _lattice_str="")
