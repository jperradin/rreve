"""Unit tests for ``rreve.io.reader.xyz_reader.XYZReader``."""

import numpy as np
import pytest

from rreve.io.reader.xyz_reader import XYZReader


def test_detect():
    r = XYZReader.__new__(XYZReader)
    assert r.detect("traj.xyz")
    assert r.detect("TRAJ.XYZ")
    assert not r.detect("traj.lammpstrj")


def test_scan_indexes_frames(two_frames_path, settings_factory):
    reader = XYZReader(settings_factory(two_frames_path))
    indices = reader.scan()
    assert len(indices) == 2
    assert reader.num_frames == 2
    assert reader.is_indexed
    assert np.allclose(indices[0].lattice, np.diag([10.0, 10.0, 10.0]))
    assert indices[0].num_nodes == 2


def test_parse_yields_frame_with_data(two_atoms_path, settings_factory):
    reader = XYZReader(settings_factory(two_atoms_path))
    reader.scan()
    frame = next(reader.parse(0))
    assert frame.frame_id == 0
    assert frame._data["symbol"] == ["Si", "O"]
    assert np.allclose(frame._data["position"][1], [1.0, 1.0, 2.6])


def test_parse_autoscans_when_not_indexed(two_atoms_path, settings_factory):
    reader = XYZReader(settings_factory(two_atoms_path))
    frame = next(reader.parse(0))  # parse without explicit scan
    assert frame.get_num_nodes() == 0  # nodes not yet initialized
    assert len(frame._data["symbol"]) == 2


def test_scan_malformed_header_raises(malformed_path, settings_factory):
    reader = XYZReader(settings_factory(malformed_path))
    with pytest.raises(IOError):
        reader.scan()


def test_scan_missing_file_raises(settings_factory, tmp_path):
    missing = str(tmp_path / "nope.xyz")
    reader = XYZReader(settings_factory(missing))
    with pytest.raises(FileNotFoundError):
        reader.scan()
