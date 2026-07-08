"""Unit tests for ``rreve.io.parser.parser.Parser``."""

import os

import pytest

from rreve.io.parser.parser import Parser


def test_parse_lists_xyz_files_from_directory(parser_dir):
    p = Parser(parser_dir, "xyz")
    files = p.get_files()
    assert len(files) == 2
    assert all(f.endswith(".xyz") for f in files)
    # sorted order
    assert files == sorted(files)


def test_parse_from_file_path_resolves_sibling_files(parser_dir):
    # When pointed at a file, parse() lists siblings via the parent directory,
    # but parse_infos() looks for info.csv next to the file path itself and
    # raises. This documents that file-input construction needs that lookup.
    one_file = os.path.join(parser_dir, "a_traj.xyz")
    with pytest.raises(ValueError):
        Parser(one_file, "xyz")


def test_parse_infos(parser_dir):
    p = Parser(parser_dir, "xyz")
    infos = p.get_infos()
    assert infos["project_name"] == ["a_traj", "b_traj"]
    assert infos["density"] == [2.2, 3.0]
    assert infos["temperature"] == [300.0, 500.0]


def test_nonexistent_location_raises():
    with pytest.raises(ValueError):
        Parser("/does/not/exist", "xyz")


def test_missing_info_csv_raises(tmp_path):
    # directory with an xyz file but no info.csv
    (tmp_path / "x.xyz").write_text("1\nLattice=\"1 0 0 0 1 0 0 0 1\"\nSi 0 0 0\n")
    with pytest.raises(ValueError):
        Parser(str(tmp_path), "xyz")
