import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Optional

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...config.settings import Settings


class ConnectivityAnalyzer(BaseAnalyzer):
    """Optimized connectivity analyzer."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.connectivity: Dict[str, int] = {
            "corner-sharing": 0,
            "edge-sharing": 0,
            "face-sharing": 0,
        }
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None

        self.networking_species = (
            self._settings.analysis.connect_settings.networking_species
            if self._settings.analysis.connect_settings is not None
            else "Si"
        )
        self.bridging_species = (
            self._settings.analysis.connect_settings.bridging_species
            if self._settings.analysis.connect_settings is not None
            else "O"
        )

    def analyze(self, frame: Frame) -> None:
        self._atoms_data = frame.nodes_data.wrapped_positions

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(frame.nodes),
            desc="Calculating connectivities ...",
            unit="atom",
            initial=0,
            total=len(frame.nodes),
            **progress_bar_kwargs,
        )
        for i, node in progress_bar:
            if node.symbol == self.networking_species:
                unique_bond = []
                for neighbor in node.neighbors:
                    if neighbor.symbol == self.bridging_species:
                        for second_neighbor in neighbor.neighbors:
                            if second_neighbor.symbol == self.networking_species:
                                unique_bond.append(second_neighbor.node_id)
                _, counts = np.unique(unique_bond, return_counts=True)
                for c in counts:
                    if c == 1:
                        self.connectivity["corner-sharing"] += 1
                    elif c == 2:
                        self.connectivity["edge-sharing"] += 1
                    elif c == 3:
                        self.connectivity["face-sharing"] += 1
                    else:
                        continue

        self.frame_processed_count += 1

    def finalize(self) -> None:
        specie_count = list([(k, len(v)) for k, v in self._atoms_data.items()])
        for s in specie_count:
            if s[0] == self.networking_species:
                count = s[1]

        for k, v in self.connectivity.items():
            self.connectivity[k] = v / (count * self.frame_processed_count)

    def get_result(self) -> Dict[str, float]:
        return self.connectivity

    def print_to_file(self) -> None:
        self.finalize()
        if self.connectivity is None:
            return

        output_path = os.path.join(
            self._settings.export_directory, "connectivities.dat"
        )

        keys = list(self.connectivity.keys())
        data = np.column_stack([self.connectivity[k] for k in keys])

        np.savetxt(
            output_path,
            data,
            header=f"{keys}",
            delimiter="\t",
            fmt="%.5f",
            comments="# ",
        )
