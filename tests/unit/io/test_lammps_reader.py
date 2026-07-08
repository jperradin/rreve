"""Unit tests for ``rreve.io.reader.lammps_reader.LAMMPSReader``.

Only ``detect`` and ``scan`` are exercised: ``parse`` constructs ``Frame``
without the required ``_lattice_str`` argument, which is a separate known issue.
"""

import numpy as np

from rreve.io.reader.lammps_reader import LAMMPSReader


def test_detect_extensions():
    r = LAMMPSReader.__new__(LAMMPSReader)
    assert r.detect("dump.lammpstrj")
    assert r.detect("system.lammps")
    assert r.detect("data.data")
    assert not r.detect("traj.xyz")


def test_scan_indexes_frames(lammps_path, settings_factory):
    reader = LAMMPSReader(settings_factory(lammps_path))
    indices = reader.scan()
    assert reader.num_frames == 2
    assert len(indices) == 2
    assert indices[0].num_nodes == 2
    # box bounds 0..10 -> 10 along each axis
    assert np.allclose(indices[0].lattice, np.diag([10.0, 10.0, 10.0]))
    # column header parsed during scan
    assert reader.columns["type"] >= 0
    assert "x" in reader.columns
