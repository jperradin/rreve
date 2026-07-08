"""Lifecycle tests for every analyzer.

Each test enables one analyzer, runs ``analyze`` -> ``finalize`` /
``print_to_file`` on a small frame, and checks both the in-memory result and
the on-disk output. Physical correctness is checked where there is a clean
oracle (regular SiO4, bridging Si-O-Si).
"""

import os
import glob

import numpy as np
import pytest

from rreve.analysis.analyzer_factory import AnalyzerFactory
from rreve.analysis.analyzers.pair_distribution_function_analyzer import (
    PairDistributionFunctionAnalyzer,
)
from rreve.analysis.analyzers.bond_angular_distribution_analyzer import (
    BondAngularDistributionAnalyzer,
)
from rreve.analysis.analyzers.structural_units_analyzer import StructuralUnitsAnalyzer
from rreve.analysis.analyzers.connectivity_analyzer import ConnectivityAnalyzer
from rreve.analysis.analyzers.polyhedricity_analyzer import PolyhedricityAnalyzer
from rreve.analysis.analyzers.tetrahedricity_analyzer import TetrahedricityAnalyzer
from rreve.analysis.analyzers.neutron_structure_factor_fft_analyzer import (
    NeutronStructureFactorFFTAnalyzer,
)
from rreve.analysis.analyzers.neutron_structure_factor_analyzer import (
    NeutronStructureFactorAnalyzer,
)


def _run(analyzer, frame):
    analyzer.analyze(frame)
    analyzer.print_to_file()  # most analyzers call finalize() internally
    return analyzer.get_result()


def _files(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_registers_all(settings_factory, two_tetrahedra_path):
    factory = AnalyzerFactory(settings_factory(two_tetrahedra_path, with_all=True))
    assert factory.get_analyzer("PolyhedricityAnalyzer") is not None
    assert factory.get_analyzer("TetrahedricityAnalyzer") is not None
    assert factory.get_analyzer("DoesNotExist") is None


# --------------------------------------------------------------------------- #
# Pair distribution function
# --------------------------------------------------------------------------- #
def test_pdf(settings_factory, frame_factory, two_tetrahedra_path, tmp_path):
    settings = settings_factory(
        two_tetrahedra_path, export_directory=str(tmp_path),
        with_pair_distribution_function=True,
    )
    frame = frame_factory(two_tetrahedra_path, settings)
    result = _run(PairDistributionFunctionAnalyzer(settings), frame)
    assert result  # non-empty dict of g(r)
    assert os.path.exists(os.path.join(tmp_path, "pair_distribution_function.dat"))


# --------------------------------------------------------------------------- #
# Bond angular distribution
# --------------------------------------------------------------------------- #
def test_bad(settings_factory, frame_factory, sio2_path, tmp_path):
    # the default triplets include Si-Si-Si, so this needs a realistic network
    settings = settings_factory(
        sio2_path, export_directory=str(tmp_path),
        with_bond_angular_distribution=True,
    )
    frame = frame_factory(sio2_path, settings)
    result = _run(BondAngularDistributionAnalyzer(settings), frame)
    assert result
    assert os.path.exists(os.path.join(tmp_path, "bond_angular_distribution.dat"))


# --------------------------------------------------------------------------- #
# Structural units
# --------------------------------------------------------------------------- #
def test_structural_units(settings_factory, frame_factory, sio2_path, tmp_path):
    # print_to_file stacks every species pair (SiSi, SiO, OSi, OO); all must be
    # populated, which needs a realistic network rather than two tetrahedra
    settings = settings_factory(
        sio2_path, export_directory=str(tmp_path),
        with_structural_units=True,
    )
    frame = frame_factory(sio2_path, settings)
    analyzer = StructuralUnitsAnalyzer(settings)
    result = _run(analyzer, frame)
    assert result
    assert _files(str(tmp_path), "structural_units-*.dat")


# --------------------------------------------------------------------------- #
# Connectivity
# --------------------------------------------------------------------------- #
def test_connectivity(settings_factory, frame_factory, two_tetrahedra_path, tmp_path):
    settings = settings_factory(
        two_tetrahedra_path, export_directory=str(tmp_path),
        with_connectivity=True,
    )
    frame = frame_factory(two_tetrahedra_path, settings)
    result = _run(ConnectivityAnalyzer(settings), frame)
    assert result is not None
    assert os.path.exists(os.path.join(tmp_path, "connectivities.dat"))


# --------------------------------------------------------------------------- #
# Polyhedricity
# --------------------------------------------------------------------------- #
def test_polyhedricity_regular_sio4(settings_factory, frame_factory, sio4_path, tmp_path):
    settings = settings_factory(
        sio4_path, export_directory=str(tmp_path), with_polyhedricity=True
    )
    frame = frame_factory(sio4_path, settings)
    analyzer = PolyhedricityAnalyzer(settings)
    result = _run(analyzer, frame)
    assert result
    assert os.path.exists(os.path.join(tmp_path, "polyhedricity_proportions.dat"))
    assert os.path.exists(os.path.join(tmp_path, "polyhedricity_histograms.dat"))
    # the single Si is a (near) perfect tetrahedron
    assert result.get("4_fold", 0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tetrahedricity (incl. the cvc / inter-tetrahedral angle)
# --------------------------------------------------------------------------- #
def test_tetrahedricity_regular_sio4(settings_factory, frame_factory, sio4_path, tmp_path):
    settings = settings_factory(
        sio4_path, export_directory=str(tmp_path), with_tetrahedricity=True
    )
    frame = frame_factory(sio4_path, settings)
    analyzer = TetrahedricityAnalyzer(settings)
    result = _run(analyzer, frame)
    assert result.get("4_fold", 0) == pytest.approx(1.0)
    assert os.path.exists(os.path.join(tmp_path, "tetrahedricity_proportions.dat"))
    # the acvc (center-vertex-center) distribution file is produced
    assert os.path.exists(
        os.path.join(tmp_path, "tetrahedricity_distribution_acvc.dat")
    )


def test_tetrahedricity_acvc_bridging_angle(
    settings_factory, frame_factory, two_tetrahedra_path, tmp_path
):
    settings = settings_factory(
        two_tetrahedra_path, export_directory=str(tmp_path), with_tetrahedricity=True
    )
    frame = frame_factory(two_tetrahedra_path, settings)
    analyzer = TetrahedricityAnalyzer(settings)
    _run(analyzer, frame)
    data = np.loadtxt(os.path.join(tmp_path, "tetrahedricity_distribution_acvc.dat"))
    bins, hist = data[:, 0], data[:, 1]
    assert hist.sum() > 0  # the single bridging Si-O-Si angle was recorded
    peak = bins[np.argmax(hist)]
    assert 145.0 < peak < 160.0  # designed ~152 degrees


# --------------------------------------------------------------------------- #
# Neutron structure factor (FFT and legacy)
# --------------------------------------------------------------------------- #
def test_nsf_fft(settings_factory, frame_factory, two_tetrahedra_path, tmp_path):
    settings = settings_factory(
        two_tetrahedra_path, export_directory=str(tmp_path),
        with_neutron_structure_factor_fft=True,
    )
    frame = frame_factory(two_tetrahedra_path, settings)
    result = _run(NeutronStructureFactorFFTAnalyzer(settings), frame)
    assert result
    assert os.path.exists(os.path.join(tmp_path, "neutron_structure_factor_fft.dat"))


def test_nsf_legacy(settings_factory, frame_factory, sio4_path, tmp_path):
    # the legacy analyzer is disabled in get_analyzers() but still importable;
    # exercise its full path directly on the smallest frame.
    settings = settings_factory(sio4_path, export_directory=str(tmp_path))
    frame = frame_factory(sio4_path, settings)
    analyzer = NeutronStructureFactorAnalyzer(settings)
    result = _run(analyzer, frame)
    assert result
    assert os.path.exists(os.path.join(tmp_path, "neutron_structure_factor.dat"))


def test_nsf_finalize_without_frames_is_noop(settings_factory, sio4_path, tmp_path):
    settings = settings_factory(sio4_path, export_directory=str(tmp_path))
    analyzer = NeutronStructureFactorAnalyzer(settings)
    analyzer.finalize()  # no frames processed -> early return
    assert analyzer.get_result() is None
