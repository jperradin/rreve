"""Unit tests for ``rreve.core.system.System``."""

import pytest

from rreve.core.system import System
from rreve.io.reader.xyz_reader import XYZReader


def _system(path, settings_factory, **kw):
    settings = settings_factory(path, **kw)
    reader = XYZReader(settings)
    return System(reader, settings), settings


def test_get_num_frames(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    assert system.get_num_frames() == 2


def test_get_num_frames_cached(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    assert system.get_num_frames() == 2
    assert system.get_num_frames() == 2  # second call hits the cache branch


def test_iter_frames_yields_all(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    frames = list(system.iter_frames())
    assert len(frames) == 2
    assert [fr.frame_id for fr in frames] == [0, 1]


def test_iter_frames_respects_range(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory, range_of_frames=(1, 1))
    frames = list(system.iter_frames())
    assert len(frames) == 1
    assert frames[0].frame_id == 1


def test_load_frame_and_get_frame(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    assert system.load_frame(0) is True
    assert system.current_frame.frame_id == 0
    assert system.get_frame(1).frame_id == 1


def test_load_frame_negative_raises(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    with pytest.raises(ValueError):
        system.load_frame(-1)


def test_load_frame_out_of_range_returns_false(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory, range_of_frames=(0, 0))
    assert system.load_frame(1) is False


def test_dunder_iteration(two_frames_path, settings_factory):
    system, _ = _system(two_frames_path, settings_factory)
    count = sum(1 for _ in system)
    assert count == 2
