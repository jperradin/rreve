"""Unit tests for ``rreve.config.settings``."""

import numpy as np
import pytest

from rreve.config.settings import (
    AnalysisSettings,
    BADAnalysisSettings,
    CONNAnalysisSettings,
    Cutoff,
    GeneralSettings,
    LatticeSettings,
    PDFAnalysisSettings,
    POLYAnalysisSettings,
    Settings,
    SettingsBuilder,
    SQFFTAnalysisSettings,
    STRUNITSAnalysisSettings,
    TETRAAnalysisSettings,
)

CUTOFFS = [Cutoff("Si", "Si", 3.5), Cutoff("Si", "O", 2.3), Cutoff("O", "O", 3.05)]


def _general(**kw):
    base = dict(
        project_name="p",
        export_directory="exp",
        file_location="/tmp/x.xyz",
        range_of_frames=(0, -1),
        apply_pbc=True,
        cutoffs=list(CUTOFFS),
    )
    base.update(kw)
    return GeneralSettings(**base)


# --------------------------------------------------------------------------- #
# Cutoff helpers
# --------------------------------------------------------------------------- #
def test_cutoff_str_and_get_distance():
    c = Cutoff("Si", "O", 2.3)
    assert c.get_distance() == 2.3
    assert "Si-O" in str(c)


def test_settings_get_cutoff_symmetric():
    s = Settings(cutoffs=list(CUTOFFS))
    assert s.get_cutoff("Si", "O") == 2.3
    assert s.get_cutoff("O", "Si") == 2.3  # symmetric lookup
    assert s.get_cutoff("Na", "Cl") is None


def test_settings_get_max_cutoff():
    s = Settings(cutoffs=list(CUTOFFS))
    assert s.get_max_cutoff() == 3.5


def test_output_directory_property():
    s = Settings(export_directory="out", project_name="run1")
    assert s.output_directory.endswith("out/run1")


def test_set_range_of_frames_valid_and_errors():
    s = Settings()
    s.set_range_of_frames(2, 5)
    assert s.range_of_frames == (2, 5)
    s.set_range_of_frames(3)  # end defaults to -1
    assert s.range_of_frames == (3, -1)
    with pytest.raises(ValueError):
        s.set_range_of_frames(-1)
    with pytest.raises(ValueError):
        s.set_range_of_frames(6, 4)


# --------------------------------------------------------------------------- #
# AnalysisSettings: with_all / exclude / include / enabled
# --------------------------------------------------------------------------- #
def test_with_all_activates_everything():
    a = AnalysisSettings(with_all=True)
    for name in (
        "pair_distribution_function",
        "bond_angular_distribution",
        "structural_units",
        "connectivity",
        "polyhedricity",
        "tetrahedricity",
    ):
        assert a.is_analyzer_enabled(name)


def test_exclude_and_include_analyzer():
    a = AnalysisSettings(with_all=True)
    a.exclude_analyzer("polyhedricity")
    assert not a.is_analyzer_enabled("polyhedricity")
    a.include_analyzer("polyhedricity")
    assert a.is_analyzer_enabled("polyhedricity")


def test_explicit_single_flag():
    a = AnalysisSettings(with_tetrahedricity=True)
    assert a.is_analyzer_enabled("tetrahedricity")
    assert not a.is_analyzer_enabled("connectivity")


def test_get_analyzers_list():
    a = AnalysisSettings(with_all=True)
    names = a.get_analyzers()
    assert "PolyhedricityAnalyzer" in names
    assert "TetrahedricityAnalyzer" in names
    assert "PairDistributionFunctionAnalyzer" in names


def test_get_analyzers_raises_on_legacy_nsf():
    a = AnalysisSettings(with_neutron_structure_factor=True)
    with pytest.raises(ValueError):
        a.get_analyzers()


def test_analysis_str_smoke():
    a = AnalysisSettings(with_all=True)
    a.bad_settings = BADAnalysisSettings(triplets_to_calculate=["O-Si-O"])
    a.pdf_settings = PDFAnalysisSettings(pairs_to_calculate=["O-O"])
    a.sqfft_settings = SQFFTAnalysisSettings()
    a.strunits_settings = STRUNITSAnalysisSettings(units_to_calculate=["SiO4"])
    a.connect_settings = CONNAnalysisSettings("Si", "O")
    a.poly_settings = POLYAnalysisSettings(central_species="Si")
    a.tetra_settings = TETRAAnalysisSettings(central_species="Si")
    out = str(a)
    assert "Analysis Settings" in out


def test_analysis_str_with_exclusions():
    a = AnalysisSettings(with_all=True)
    a.exclude_analyzer("connectivity")
    assert "exclude_analyzers" in str(a)


# --------------------------------------------------------------------------- #
# Sub-settings __str__
# --------------------------------------------------------------------------- #
def test_subsettings_str():
    assert "pdf_settings" in str(PDFAnalysisSettings())
    assert "bad_settings" in str(BADAnalysisSettings())
    assert "sqfft_settings" in str(SQFFTAnalysisSettings())
    assert "strunits_settings" in str(STRUNITSAnalysisSettings())
    assert "connect_settings" in str(CONNAnalysisSettings("Si", "O"))
    assert "poly_settings" in str(POLYAnalysisSettings())
    assert "tetra_settings" in str(TETRAAnalysisSettings())


def test_lattice_settings_str():
    assert "Lattice Settings" in str(LatticeSettings(apply_custom_lattice=False))
    custom = LatticeSettings(
        apply_custom_lattice=True, custom_lattice=np.diag([5.0, 5.0, 5.0])
    )
    assert "custom_lattice" in str(custom)


def test_settings_str_smoke():
    s = Settings(cutoffs=list(CUTOFFS))
    assert "General Settings" in str(s)


# --------------------------------------------------------------------------- #
# SettingsBuilder
# --------------------------------------------------------------------------- #
def test_builder_happy_path_creates_default_subsettings():
    s = (
        SettingsBuilder()
        .with_general(_general())
        .with_lattice(LatticeSettings(apply_custom_lattice=False))
        .with_analysis(AnalysisSettings(with_all=True))
        .build()
    )
    assert isinstance(s, Settings)
    # default sub-settings auto-created by the builder
    assert s.analysis.bad_settings is not None
    assert s.analysis.pdf_settings is not None
    assert s.analysis.tetra_settings is not None
    assert s.analysis.poly_settings is not None
    assert s.analysis.connect_settings is not None
    assert s.analysis.strunits_settings is not None
    assert s.analysis.sqfft_settings is not None


def test_builder_propagates_general_fields():
    s = (
        SettingsBuilder()
        .with_general(_general(coordination_mode="different_type", verbose=True))
        .with_analysis(AnalysisSettings(with_tetrahedricity=True))
        .build()
    )
    assert s.coordination_mode == "different_type"
    assert s.verbose is True
    assert s.get_cutoff("Si", "O") == 2.3


def test_builder_rejects_bad_types():
    b = SettingsBuilder()
    with pytest.raises(ValueError):
        b.with_general("notgeneral")
    with pytest.raises(ValueError):
        b.with_lattice("notlattice")
    with pytest.raises(ValueError):
        b.with_analysis("notanalysis")


def test_builder_general_validation_errors():
    with pytest.raises(ValueError):
        SettingsBuilder().with_general(_general(project_name=""))
    with pytest.raises(ValueError):
        SettingsBuilder().with_general(_general(file_location=""))
    with pytest.raises(ValueError):
        SettingsBuilder().with_general(_general(cutoffs=[]))
