"""Tests for ``rreve.utils.performance`` and ``rreve.utils.aesthetics``."""

import os

from rreve.utils.performance import Performance
from rreve.utils import aesthetics


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def test_performance_add_metric_and_history():
    perf = Performance(id="1", name="run")
    perf.add_metric("frames", 5)
    assert perf.metrics["frames"] == 5
    perf.execution_time_ms = 10.0
    perf.record_history()
    perf.execution_time_ms = 20.0
    perf.record_history()
    assert len(perf.history) == 2
    assert perf.get_average_execution_time() == 15.0


def test_performance_average_empty_history():
    perf = Performance(id="1", name="run")
    assert perf.get_average_execution_time() is None


def test_performance_str():
    perf = Performance(id="abc", name="run")
    assert "run" in str(perf)
    assert "abc" in str(perf)


# --------------------------------------------------------------------------- #
# Aesthetics
# --------------------------------------------------------------------------- #
def test_print_title(capsys):
    aesthetics.print_title("9.9.9")
    out = capsys.readouterr().out
    assert "9.9.9" in out


def test_print_title_to_file(tmp_path):
    path = str(tmp_path / "title.txt")
    aesthetics.print_title_to_file("1.2.3", path)
    assert os.path.exists(path)
    assert "1.2.3" in open(path).read()


def test_generate_color_gradient():
    assert aesthetics.generate_color_gradient(0) == [(255, 0, 0)]
    assert len(aesthetics.generate_color_gradient(1)) == 2
    grad = aesthetics.generate_color_gradient(5)
    assert len(grad) == 6
    assert all(len(c) == 3 for c in grad)


def test_remove_duplicate_lines(tmp_path):
    path = tmp_path / "dup.txt"
    path.write_text("a\nb\na\nc\nb\n")
    aesthetics.remove_duplicate_lines(str(path))
    lines = open(path).read().splitlines()
    assert lines == ["a", "b", "c"]
