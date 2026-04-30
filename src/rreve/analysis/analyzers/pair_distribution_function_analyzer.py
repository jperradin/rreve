from logging import disable
import numpy as np
import os
from itertools import combinations
from tqdm import tqdm
from typing import Dict, Optional, List
from scipy.spatial import cKDTree
from numba_progress import ProgressBar

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...config.settings import Settings
from ...utils.geometry import (
    calculate_pbc_distances_batch,
    fast_histogram,
    cartesian_to_fractional,
)


class PairDistributionFunctionAnalyzer(BaseAnalyzer):
    """
    Optimized pair distribution function G(r) analyzer that maintains cKDTree
    efficiency while adding performance improvements.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.r: Optional[np.ndarray] = None
        self.gr: Optional[Dict[str, np.ndarray]] = None
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None
        self.gr_data: List[Dict[str, np.ndarray]] = []

        # PDFAnalysisSettings
        self.r_max: float = (
            self._settings.analysis.pdf_settings.r_max
            if self._settings.analysis.pdf_settings is not None
            else 10.0
        )
        self.bins: int = (
            self._settings.analysis.pdf_settings.bins
            if self._settings.analysis.pdf_settings is not None
            else 800
        )
        self.dr: float = (
            self.r_max / self.bins
            if self._settings.analysis.pdf_settings is not None
            else 10.0 / 800
        )
        self.pairs_to_calculate = (
            self._settings.analysis.pdf_settings.pairs_to_calculate
            if self._settings.analysis.pdf_settings is not None
            else ["O-O", "O-Si", "Si-Si", "total"]
        )

        # Pre-compute commonly used arrays
        self._shell_volumes: Optional[np.ndarray] = None
        self._r_mid: Optional[np.ndarray] = None
        self._r_bins: Optional[np.ndarray] = None

    def _initialize_arrays(self) -> None:
        """Pre-compute arrays that don't change between calculations."""
        if self._r_bins is None:
            self._r_bins = np.linspace(0, self.r_max, self.bins + 1)
            self._r_mid = (self._r_bins[:-1] + self._r_bins[1:]) / 2
            self._shell_volumes = (
                4.0 / 3.0 * np.pi * (self._r_bins[1:] ** 3 - self._r_bins[:-1] ** 3)
            )

    def _check_rmax(self, lattice: np.ndarray) -> None:
        """Ensures r_max is not larger than half the box size."""
        min_box_dim = np.min(np.linalg.norm(lattice, axis=1))
        if self.r_max > min_box_dim / 2.0:
            self.r_max = min_box_dim / 2.0
            self.bins = int(self.r_max / self.dr)
            # Reset cached arrays when r_max changes
            self._r_bins = None
            if self._settings.verbose:
                print(
                    f"Warning: r_max for PDF has been rescaled to half the smallest box dimension: {self.r_max:.3f}"
                )

    def analyze(self, frame: Frame) -> None:
        self._atoms_data = frame.nodes_data.wrapped_positions
        self._correlation_lengths: Dict[str, float] = (
            frame.nodes_data.correlation_lengths
        )
        self._check_rmax(lattice=frame.get_lattice())
        self._initialize_arrays()

        pairs = self._get_pairs(list(self._atoms_data.keys()))
        self.calculate_gr(pairs, frame)
        if self.gr is not None:
            self.gr_data.append(dict(self.gr))  # Make a copy
            self.frame_processed_count += 1

    def finalize(self) -> None:
        if self.frame_processed_count == 0:
            return

        # Vectorized averaging across frames
        if self.gr_data:
            keys = list(self.gr_data[0].keys())
            self.gr = {}

            for key in keys:
                # Stack all frames for this pair and compute mean
                values = np.array([frame_data[key] for frame_data in self.gr_data])
                self.gr[key] = np.mean(values, axis=0)

    def get_result(self) -> Dict[str, np.ndarray]:
        return self.gr if self.gr is not None else {}

    def print_to_file(self) -> None:
        self.finalize()
        if self.gr is None or self.r is None:
            return

        output_path = os.path.join(
            self._settings.export_directory, "pair_distribution_function.dat"
        )

        # Use numpy for efficient file writing
        keys = list(self.gr.keys())
        for key in keys:
            if key not in self.pairs_to_calculate:
                keys.remove(key)
        header = "r\t" + "\t".join(keys)
        data = np.column_stack([self.r] + [self.gr[key] for key in keys])

        np.savetxt(
            output_path, data, header=header, delimiter="\t", fmt="%.5f", comments="# "
        )

    def _get_pairs(self, species: List[str]) -> List[str]:
        pairs = [f"{s1}-{s2}" for s1, s2 in combinations(species, 2)]
        for s in species:
            pairs.append(f"{s}-{s}")
        return pairs

    def calculate_gr(self, pairs: List[str], frame: Frame) -> None:
        lattice = frame.get_lattice()
        box_volume = np.linalg.det(lattice)
        inv_lattice = np.linalg.inv(lattice)

        self.r = self._r_mid.copy()
        shell_volumes = self._shell_volumes

        self.gr = {}

        # Pre-compute fractional coordinates for all species
        positions_frac = {
            s: cartesian_to_fractional(p, lattice) for s, p in self._atoms_data.items()
        }

        # Optimized search radius calculation
        search_radius_frac = self.r_max * np.max(np.linalg.norm(inv_lattice, axis=1))

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": getattr(
                os, "get_terminal_size", lambda: type("", (), {"columns": 80})
            )().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(pairs),
            desc="Calculating distances ...",
            unit="pair",
            initial=0,
            total=len(pairs),
            **progress_bar_kwargs,
        )

        for p, pair in progress_bar:
            progress_bar.set_description(f"Calculating {pair} distances ...")
            s1, s2 = pair.split("-")
            pos1, pos2 = self._atoms_data[s1], self._atoms_data[s2]
            pos1_frac, pos2_frac = positions_frac[s1], positions_frac[s2]
            n1, n2 = len(pos1), len(pos2)

            if n1 == 0 or n2 == 0:
                self.gr[pair] = np.zeros(self.bins)
                continue

            # Use cKDTree for efficient neighbor finding
            tree2 = cKDTree(pos2_frac, boxsize=[1, 1, 1])

            if s1 == s2:
                if n1 < 2:
                    self.gr[pair] = np.zeros(self.bins)
                    continue

                # Optimized same-species calculation
                candidate_pairs = tree2.query_pairs(
                    r=search_radius_frac, output_type="ndarray"
                )

                if len(candidate_pairs) > 0:
                    # Batch process distances
                    pos1_batch = pos1[candidate_pairs[:, 0]]
                    pos2_batch = pos1[candidate_pairs[:, 1]]  # Same species
                    with ProgressBar(
                        total=pos1_batch.shape[0],
                        disable=not self._settings.verbose,
                        leave=False,
                        colour="#eaeaaa",
                        unit="atom",
                        desc="Calculating distances",
                    ) as progress:
                        distances = calculate_pbc_distances_batch(
                            pos1_batch, pos2_batch, lattice, progress
                        )
                    # Filter distances within r_max
                    valid_distances = distances[distances < self.r_max]
                else:
                    valid_distances = np.array([])

                norm_factor = (n1 * (n1 - 1) / 2.0) / box_volume
            else:
                # Different species calculation
                tree1 = cKDTree(pos1_frac, boxsize=[1, 1, 1])
                candidate_indices = tree1.query_ball_tree(tree2, r=search_radius_frac)

                # Collect pairs for batch processing
                pairs_list = []
                for i, indices_j in enumerate(candidate_indices):
                    for j in indices_j:
                        pairs_list.append((i, j))

                if pairs_list:
                    pairs_array = np.array(pairs_list)
                    pos1_batch = pos1[pairs_array[:, 0]]
                    pos2_batch = pos2[pairs_array[:, 1]]
                    with ProgressBar(
                        total=pos1_batch.shape[0],
                        disable=not self._settings.verbose,
                        leave=False,
                        colour="#eaeaaa",
                        unit="atom",
                        desc="Calculating distances",
                    ) as progress:
                        distances = calculate_pbc_distances_batch(
                            pos1_batch, pos2_batch, lattice, progress
                        )
                    valid_distances = distances[distances < self.r_max]
                else:
                    valid_distances = np.array([])

                norm_factor = (n1 * n2) / box_volume

            # Fast histogram calculation
            if len(valid_distances) > 0:
                hist = fast_histogram(valid_distances, self.r_max, self.bins)
            else:
                hist = np.zeros(self.bins, dtype=np.int64)

            # Vectorized normalization
            g_r = np.zeros(self.bins)
            ideal_counts = shell_volumes * norm_factor
            non_zero_mask = ideal_counts > 0
            g_r[non_zero_mask] = hist[non_zero_mask] / ideal_counts[non_zero_mask]

            self.gr[pair] = g_r

        # Optimized total G(r) calculation
        self._calculate_total_gr()

    def _calculate_total_gr(self) -> None:
        """Calculate total G(r) using Faber-Ziman formalism with vectorized operations."""
        cl = self._correlation_lengths
        species_list = sorted(self._atoms_data.keys())
        species_counts = {s: len(self._atoms_data[s]) for s in species_list}
        num_atoms = sum(species_counts.values())

        if num_atoms == 0:
            self.gr["total"] = np.zeros(self.bins)
            return

        concentrations = np.array([species_counts[s] / num_atoms for s in species_list])
        scattering_lengths = np.array([cl[s] for s in species_list])

        b_avg = np.dot(concentrations, scattering_lengths)
        b_avg_sq = b_avg**2

        # Vectorized total G(r) calculation
        total_gr = np.zeros(self.bins)

        # Same species contributions
        for i, s1 in enumerate(species_list):
            pair_key = f"{s1}-{s1}"
            if pair_key in self.gr:
                total_gr += (
                    concentrations[i] ** 2
                    * scattering_lengths[i] ** 2
                    * self.gr[pair_key]
                )

        # Different species contributions
        for i, s1 in enumerate(species_list):
            for j, s2 in enumerate(species_list[i + 1 :], i + 1):
                pair_key = f"{s1}-{s2}"
                if pair_key in self.gr:
                    total_gr += (
                        2
                        * concentrations[i]
                        * concentrations[j]
                        * scattering_lengths[i]
                        * scattering_lengths[j]
                        * self.gr[pair_key]
                    )

        if b_avg_sq > 0:
            total_gr /= b_avg_sq

        self.gr["total"] = total_gr
