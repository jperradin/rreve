"""Unit tests for the writers and the ``WriterFactory``."""

import glob
import json
import os

import pytest

from rreve.io.writer.writer_factory import WriterFactory
from rreve.io.writer.logs_writer import LogsWriter
from rreve.io.writer.performance_writer import PerformanceWriter
from rreve.io.writer.multiple_files_summary_writer import MultipleFilesSummaryWriter
from rreve.io.writer.xyz_decorator import XYZDecorator
from rreve.utils.performance import Performance


@pytest.fixture
def export_settings(settings_factory, two_atoms_path, tmp_path):
    return settings_factory(two_atoms_path, export_directory=str(tmp_path))


def test_writer_factory_dispatch(export_settings):
    factory = WriterFactory(export_settings)
    assert isinstance(factory.get_writer("LogsWriter"), LogsWriter)
    assert isinstance(factory.get_writer("PerformanceWriter"), PerformanceWriter)
    assert isinstance(
        factory.get_writer("MultipleFilesSummaryWriter"), MultipleFilesSummaryWriter
    )
    assert isinstance(factory.get_writer("XYZDecorator"), XYZDecorator)
    assert factory.get_writer("Unknown") is None


def test_logs_writer_creates_file(export_settings):
    LogsWriter(export_settings).write()
    log_path = os.path.join(export_settings.export_directory, "log.txt")
    assert os.path.exists(log_path)
    assert os.path.getsize(log_path) > 0


def test_performance_writer_json(export_settings):
    perf = Performance(id="abc", name="run1")
    perf.add_metric("frames", 3)
    perf.record_history()
    PerformanceWriter(export_settings).write(perf)
    path = os.path.join(export_settings.export_directory, "performance_run1.json")
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["name"] == "run1"
    assert data["metrics"]["frames"] == 3


def test_multiple_files_summary_writer_noop(export_settings):
    # no cluster .dat files present -> writer runs without error and writes nothing
    MultipleFilesSummaryWriter(export_settings).write()


def _write_cluster_dat(directory, name):
    # format expected by the summary writer: type,concentration,avg,std
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write("# header\n")
        f.write("Si-O,0.50,3.20,0.10\n")
        f.write("O-O,0.50,2.10,0.05\n")


def test_multiple_files_summary_writer_all_mode(export_settings):
    d = export_settings.export_directory
    _write_cluster_dat(d, "average_cluster_size-run.dat")
    MultipleFilesSummaryWriter(export_settings, mode="all").write()
    assert os.path.exists(os.path.join(d, "average_cluster_size_summary.dat"))


def test_multiple_files_summary_writer_connectivity_mode(export_settings):
    d = export_settings.export_directory
    _write_cluster_dat(d, "correlation_length-run.dat")
    MultipleFilesSummaryWriter(export_settings, mode="connectivity").write()
    # one summary file per connectivity type
    assert glob.glob(os.path.join(d, "correlation_length_*_summary.dat"))


def test_xyz_decorator_writes_decorated_frame(
    export_settings, frame_factory, two_atoms_path
):
    frame = frame_factory(two_atoms_path, export_settings)
    XYZDecorator(export_settings).write(frame)
    out = os.path.join(
        export_settings.export_directory, "decorated_input_files", "two_atoms.xyz"
    )
    assert os.path.exists(out)
    content = open(out).read()
    assert "Properties=" in content
    assert "Si" in content and "O" in content
