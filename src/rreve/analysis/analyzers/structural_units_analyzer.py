import numpy as np
import os
from itertools import product
from tqdm import tqdm
from typing import Dict, Optional, List
from natsort import natsorted

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...config.settings import Settings


class StructuralUnitsAnalyzer(BaseAnalyzer):
    """Optimized structural units analyzer."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.units: Optional[Dict[str, np.ndarray]] = {}
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None
        self.units_to_calculate = (
            self._settings.analysis.strunits_settings.units_to_calculate
            if self._settings.analysis.strunits_settings is not None
            else [
                "SiO4",
                "SiO5",
                "SiO6",
                "SiO7",
                "OSi1",
                "OSi2",
                "OSi3",
                "OSi4",
            ]
        )

    def analyze(self, frame: Frame) -> None:
        self._atoms_data = frame.nodes_data.wrapped_positions
        coordination_mode = self._settings.coordination_mode

        pairs, _ = self._get_pairs(
            list(self._atoms_data.keys()), [len(v) for v in self._atoms_data.values()]
        )

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(frame.nodes),
            desc="Calculating structural units ...",
            unit="atom",
            initial=0,
            total=len(frame.nodes),
            **progress_bar_kwargs,
        )
        for i, node in progress_bar:
            _counts = {}
            for neighbor in node.neighbors:
                _pair = node.symbol + neighbor.symbol
                if _pair not in _counts:
                    _counts[_pair] = 0
                _counts[_pair] += 1

            for key, value in _counts.items():
                if key + str(value) not in self.units:
                    self.units[key + str(value)] = 0
                self.units[key + str(value)] += 1

        self.frame_processed_count += 1

    def finalize(self) -> None:
        pairs, counts = self._get_pairs(
            list(self._atoms_data.keys()), [len(v) for v in self._atoms_data.values()]
        )
        for pair, count in zip(pairs, counts):
            for _pair in self.units.keys():
                if _pair.startswith(pair):
                    self.units[_pair] /= count * self.frame_processed_count

    def get_result(self) -> Dict[str, float]:
        return self.units

    def print_to_file(self) -> None:
        self.finalize()
        if self.units is None:
            return

        pairs, _ = self._get_pairs(
            list(self._atoms_data.keys()), [len(v) for v in self._atoms_data.values()]
        )

        for pair in pairs:
            output_path = os.path.join(
                self._settings.export_directory, f"structural_units-{pair}.dat"
            )

            keys = [p for p in self.units if p.startswith(pair)]

            keys = natsorted(keys)

            data = np.column_stack([self.units[k] for k in keys])

            np.savetxt(
                output_path,
                data,
                header=f"{keys}",
                delimiter="\t",
                fmt="%.5f",
                comments="# ",
            )

    def _get_pairs(self, species: List[str], counts: List[int]) -> List[str]:
        pairs = [f"{s1}{s2}" for s1, s2 in product(species, repeat=2)]
        counts = [s1 for s1, s2 in product(counts, repeat=2)]
        return pairs, counts
