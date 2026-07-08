"""Unit tests for ``rreve.io.reader.reader_factory.ReaderFactory``."""

import pytest

from rreve.io.reader.reader_factory import ReaderFactory
from rreve.io.reader.xyz_reader import XYZReader
from rreve.io.reader.lammps_reader import LAMMPSReader


def test_returns_xyz_reader(two_atoms_path, settings_factory):
    reader = ReaderFactory(settings_factory(two_atoms_path)).get_reader()
    assert isinstance(reader, XYZReader)


def test_returns_lammps_reader(lammps_path, settings_factory):
    reader = ReaderFactory(settings_factory(lammps_path)).get_reader()
    assert isinstance(reader, LAMMPSReader)


def test_missing_file_raises(settings_factory, tmp_path):
    settings = settings_factory(str(tmp_path / "absent.xyz"))
    with pytest.raises(ValueError):
        ReaderFactory(settings).get_reader()
