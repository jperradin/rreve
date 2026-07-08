"""Shared fixtures for the rreve test suite.

Two environment concerns are handled here, before anything else:

* ``NUMBA_DISABLE_JIT=1`` is set *before* any ``rreve`` import. Most of
  ``utils.geometry`` and the analyzers run under ``@njit``; coverage.py cannot
  trace compiled code, so we run numba in pure-Python object mode. This also
  removes JIT compilation latency from the suite.
* ``os.get_terminal_size`` is monkeypatched (autouse) because several modules
  call it unconditionally to size tqdm bars, which raises ``OSError`` when the
  tests run without a controlling terminal.

Geometric fixtures use hand-chosen coordinates so the expected distances,
angles and tetrahedricity values are computable by hand.
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest

from rreve.config.settings import (
    AnalysisSettings,
    Cutoff,
    GeneralSettings,
    LatticeSettings,
    SettingsBuilder,
    Settings,
)
from rreve.core.node import Node
from rreve.io.reader.xyz_reader import XYZReader
from rreve.analysis.neighbor_searcher import NeighborSearcher
from rreve.analysis.fast_neighbor_searcher import FastNeighborSearcher

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULT_CUTOFFS = [
    Cutoff("Si", "Si", 3.50),
    Cutoff("Si", "O", 2.30),
    Cutoff("O", "O", 3.05),
]


# --------------------------------------------------------------------------- #
# Headless environment
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _fake_terminal(monkeypatch):
    """Make ``os.get_terminal_size`` safe under a non-tty test runner."""
    monkeypatch.setattr(
        os, "get_terminal_size", lambda *a, **k: os.terminal_size((80, 24))
    )


# --------------------------------------------------------------------------- #
# Data paths
# --------------------------------------------------------------------------- #
def _data_path(name):
    return os.path.join(DATA_DIR, name)


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def two_atoms_path():
    """2 atoms (Si, O) 1.6 apart in a box of 10. Bonded under the Si-O cutoff."""
    return _data_path("two_atoms.xyz")


@pytest.fixture
def two_frames_path():
    """Two-frame trajectory (2 atoms each)."""
    return _data_path("two_frames.xyz")


@pytest.fixture
def malformed_path():
    """Frame whose header lacks a ``Lattice="..."`` string. Reader error test."""
    return _data_path("malformed.xyz")


@pytest.fixture
def sio4_path():
    """One Si at the origin with 4 O on regular-tetrahedron vertices, edge-scaled
    so every Si-O distance is 1.6. Tetrahedricity ~0, vertex-center-vertex ~109.47."""
    return _data_path("sio4_single.xyz")


@pytest.fixture
def two_tetrahedra_path():
    """Two corner-sharing SiO4 units sharing one bridging O. Exercises the
    inter-tetrahedral (Si-O-Si) angle, connectivity and structural units."""
    return _data_path("two_tetrahedra.xyz")


@pytest.fixture
def sio2_path():
    """5 frames of 384-atom amorphous SiO2 (copied from the nexus dataset)."""
    return _data_path("sio2_384.xyz")


@pytest.fixture
def lammps_path():
    """Two-frame LAMMPS dump, 2 atoms each."""
    return _data_path("lammps_two.lammpstrj")


@pytest.fixture
def parser_dir():
    """Directory holding two .xyz files and a matching info.csv."""
    return _data_path("parser_dir")


# --------------------------------------------------------------------------- #
# Settings factory
# --------------------------------------------------------------------------- #
def make_settings(
    file_location,
    export_directory="rreve_test_export",
    cutoffs=None,
    apply_pbc=True,
    coordination_mode="different_type",
    use_fast_neighbor_search=True,
    range_of_frames=(0, -1),
    with_all=False,
    analysis=None,
    **analysis_flags,
):
    """Build a validated ``Settings`` the same way the quickstart scripts do.

    ``analysis_flags`` are forwarded to ``AnalysisSettings`` (e.g.
    ``with_tetrahedricity=True``). Pass a ready ``AnalysisSettings`` via
    ``analysis`` to override entirely.
    """
    cutoffs = list(DEFAULT_CUTOFFS if cutoffs is None else cutoffs)

    general = GeneralSettings(
        project_name="test",
        export_directory=export_directory,
        file_location=file_location,
        range_of_frames=range_of_frames,
        apply_pbc=apply_pbc,
        verbose=False,
        save_logs=False,
        save_performance=False,
        cutoffs=cutoffs,
        coordination_mode=coordination_mode,
        use_fast_neighbor_search=use_fast_neighbor_search,
    )
    if analysis is None:
        analysis = AnalysisSettings(with_all=with_all, **analysis_flags)

    return (
        SettingsBuilder()
        .with_general(general)
        .with_lattice(LatticeSettings(apply_custom_lattice=False))
        .with_analysis(analysis)
        .build()
    )


@pytest.fixture
def settings_factory():
    return make_settings


# --------------------------------------------------------------------------- #
# Frame factory
# --------------------------------------------------------------------------- #
def load_frame(file_location, settings, frame_id=0, with_neighbors=True):
    """Parse one frame and (optionally) run the neighbor search, exactly as the
    real pipeline does in ``main``."""
    reader = XYZReader(settings)
    reader.scan()
    frame = next(reader.parse(frame_id))
    frame.initialize_nodes()
    if with_neighbors:
        if settings.use_fast_neighbor_search:
            FastNeighborSearcher(frame, settings).execute()
        else:
            NeighborSearcher(frame, settings).execute()
    return frame


@pytest.fixture
def frame_factory():
    return load_frame


@pytest.fixture
def sio4_frame(sio4_path):
    settings = make_settings(sio4_path)
    return load_frame(sio4_path, settings)


@pytest.fixture
def two_tetrahedra_frame(two_tetrahedra_path):
    settings = make_settings(two_tetrahedra_path)
    return load_frame(two_tetrahedra_path, settings)


@pytest.fixture
def sio2_frame(sio2_path):
    settings = make_settings(sio2_path)
    return load_frame(sio2_path, settings)


# --------------------------------------------------------------------------- #
# Low-level builders (no I/O)
# --------------------------------------------------------------------------- #
@pytest.fixture
def cubic_lattice():
    return np.diag([10.0, 10.0, 10.0]).astype(float)


@pytest.fixture
def make_node():
    def _make(symbol="Si", node_id=0, position=(0.0, 0.0, 0.0)):
        return Node(
            symbol=symbol,
            node_id=node_id,
            position=np.asarray(position, dtype=float),
        )

    return _make
