"""Unit tests for the standard and fast neighbor searchers."""

import numpy as np
import pytest

from rreve.analysis.neighbor_searcher import NeighborSearcher
from rreve.analysis.fast_neighbor_searcher import FastNeighborSearcher


def _frame_no_neighbors(path, settings_factory, frame_factory, **kw):
    settings = settings_factory(path, **kw)
    frame = frame_factory(path, settings, with_neighbors=False)
    return frame, settings


@pytest.mark.parametrize("searcher_cls", [NeighborSearcher, FastNeighborSearcher])
def test_sio4_central_coordination(
    searcher_cls, sio4_path, settings_factory, frame_factory
):
    frame, settings = _frame_no_neighbors(sio4_path, settings_factory, frame_factory)
    searcher_cls(frame, settings).execute()
    si = frame.get_nodes_by_element("Si")[0]
    assert si.coordination == 4
    assert all(n.symbol == "O" for n in si.neighbors)
    # every Si-O distance ~1.6
    assert np.allclose(si.distances, 1.6, atol=1e-6)


def test_fast_and_standard_agree(sio4_path, settings_factory, frame_factory):
    f1, s1 = _frame_no_neighbors(sio4_path, settings_factory, frame_factory)
    f2, s2 = _frame_no_neighbors(sio4_path, settings_factory, frame_factory)
    NeighborSearcher(f1, s1).execute()
    FastNeighborSearcher(f2, s2).execute()
    for n1, n2 in zip(f1.nodes, f2.nodes):
        assert n1.coordination == n2.coordination
        assert sorted(n1.indices) == sorted(n2.indices)


def test_coordination_modes(sio4_path, settings_factory, frame_factory):
    # all_types: Si sees its 4 O; each O sees only the 1 Si
    frame, settings = _frame_no_neighbors(
        sio4_path, settings_factory, frame_factory, coordination_mode="all_types"
    )
    NeighborSearcher(frame, settings).execute()
    si = frame.get_nodes_by_element("Si")[0]
    assert si.coordination == 4

    # same_type: Si has no Si neighbor -> 0
    frame, settings = _frame_no_neighbors(
        sio4_path, settings_factory, frame_factory, coordination_mode="same_type"
    )
    NeighborSearcher(frame, settings).execute()
    si = frame.get_nodes_by_element("Si")[0]
    assert si.coordination == 0

    # explicit species mode "O": Si counts its O neighbors -> 4
    frame, settings = _frame_no_neighbors(
        sio4_path, settings_factory, frame_factory, coordination_mode="O"
    )
    NeighborSearcher(frame, settings).execute()
    si = frame.get_nodes_by_element("Si")[0]
    assert si.coordination == 4


def test_no_pbc_path(sio4_path, settings_factory, frame_factory):
    frame, settings = _frame_no_neighbors(
        sio4_path, settings_factory, frame_factory, apply_pbc=False
    )
    FastNeighborSearcher(frame, settings).execute()
    si = frame.get_nodes_by_element("Si")[0]
    assert si.coordination == 4
