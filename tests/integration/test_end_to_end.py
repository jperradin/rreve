"""End-to-end test: run ``main`` with every analyzer on a real SiO2 frame."""

import glob
import os

import pytest

from rreve.main import main


@pytest.fixture
def run_all(settings_factory, sio2_path, tmp_path):
    settings = settings_factory(
        sio2_path,
        export_directory=str(tmp_path),
        range_of_frames=(0, 0),  # single frame keeps the run fast
        with_all=True,
    )
    main(settings)
    # main joins project_name onto export_directory
    return os.path.join(str(tmp_path), "test")


def test_main_creates_output_directory(run_all):
    assert os.path.isdir(run_all)


@pytest.mark.parametrize(
    "filename",
    [
        "pair_distribution_function.dat",
        "bond_angular_distribution.dat",
        "connectivities.dat",
        "polyhedricity_proportions.dat",
        "polyhedricity_histograms.dat",
        "tetrahedricity_proportions.dat",
        "tetrahedricity_distribution_acvc.dat",
        "neutron_structure_factor_fft.dat",
    ],
)
def test_main_writes_expected_files(run_all, filename):
    path = os.path.join(run_all, filename)
    assert os.path.exists(path), f"missing {filename}"
    assert os.path.getsize(path) > 0


def test_main_writes_structural_units(run_all):
    assert glob.glob(os.path.join(run_all, "structural_units-*.dat"))
