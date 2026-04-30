from typing import List, Dict, Optional
from ...core.frame import Frame
from ...core.system import Settings
from .base_analyzer import BaseAnalyzer
from ...utils.aesthetics import remove_duplicate_lines
from ...utils.geometry import calculate_components

import numpy as np
import os
from datetime import datetime
from numba import njit, prange
from tqdm import tqdm
from dataclasses import dataclass, field
from itertools import combinations
from numba_progress import ProgressBar


@dataclass
class QVectors:
    """Holds reciprocal spaces q-vectors and their binning informations."""

    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    q_norm: np.ndarray  # 3D array of q-vector magnitudes
    q_bins: np.ndarray  # 1D array of q-bin edges


@dataclass
class SpeciesComponents:
    """Holds the intermediate sin/cos components for each atomic species."""

    qcos: dict[str, np.ndarray]
    qsin: dict[str, np.ndarray]
    correlation_lengths: dict[str, float]


class NeutronStructureFactorAnalyzer(BaseAnalyzer):
    """
    Computes the neutron structure factor for each pair of atoms.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.q = None
        self.nsf = None
        self._atoms_data: Optional(Dict[str, np.ndarray]) = None
        self.nsf_data = []

    def analyze(self, frame: Frame) -> None:
        self._atoms_data = frame.nodes_data.wrapped_positions
        self._correlation_lengths: Dict[str, float] = (
            frame.nodes_data.correlation_lengths
        )
        pairs = self._get_pairs(self._atoms_data.keys())
        self.calculate_neutron_structure_factor(pairs, frame)
        self.nsf_data.append(self.nsf)
        self.frame_processed_count += 1

    def finalize(self) -> None:
        if self.frame_processed_count == 0:
            return

        # Initialize a dictionary to hold the sum of structure factors
        nsf_sum = {
            key: np.zeros_like(self.nsf_data[0][key]) for key in self.nsf_data[0].keys()
        }

        # Sum the structure factors from all frames
        for nsf_frame in self.nsf_data:
            for key, value in nsf_frame.items():
                nsf_sum[key] += value

        # Calculate the average
        self.nsf = {
            key: value / self.frame_processed_count for key, value in nsf_sum.items()
        }

    def get_result(self) -> Dict[str, float]:
        return self.nsf

    def print_to_file(self) -> None:
        self.finalize()
        if self.nsf is None:
            return
        output_path = os.path.join(
            self._settings.export_directory, "neutron_structure_factor.dat"
        )
        with open(output_path, "w") as f:
            header = "# q (bin center) \t" + "\t".join(self.nsf.keys()) + "\n"
            f.write(header)
            for i in range(1, len(self.q)):
                line = f"{self.q[i]:2.5f}"
                for key in self.nsf.keys():
                    line += f"\t{self.nsf[key][i]:2.5f}"
                f.write(line + "\n")

    def _get_pairs(self, species: List[str]) -> List[str]:
        pairs = [f"{s1}-{s2}" for s1, s2 in combinations(species, 2)]
        for s in species:
            pairs.append(f"{s}-{s}")
        pairs.append("total")
        return pairs

    def calculate_neutron_structure_factor(
        self, pairs: list[str], frame: Frame
    ) -> None:
        """
        Calculate the neutron structure factor of the system by orchestrating
        calls to helper methods.

        Args:
            pairs (list[str]): A list of species pairs to analyze (e.g., ["a-a", "a-b", "total"]).
        """
        # 1. Generate q-vectors based on box dimensions
        q_vectors = self._generate_q_vectors(frame, q_max=10.0)

        # 2. Calculate cosine and sine components for each atomic species
        components = self._calculate_species_components(q_vectors)

        # 3. Calculate partial and total structure factors (unbinned)
        f_unbinned = self._calculate_all_partial_factors(pairs, components)

        # 4. Bin the structure factor results into a histogram using a vectorized approach
        structure_factor_binned = self._bin_structure_factor(f_unbinned, q_vectors)

        # 5. Store final results
        self.q = structure_factor_binned.pop("q")
        self.nsf = structure_factor_binned
        if not self._settings.verbose:
            print("Neutron structure factor calculation complete.")

    # --- Helper Methods ---

    def _progress_iterator(self, iterable, desc: str, unit: str = "it"):
        """Returns a tqdm iterator if not in quiet mode, otherwise returns the original iterable."""
        if self._settings.verbose:
            return tqdm(iterable, desc=desc, colour="#00ffff", leave=False, unit=unit)
        return iterable

    def _generate_q_vectors(
        self, frame: Frame, q_max: float = 10.0, q_step_scale: float = 6.0
    ) -> QVectors:
        """
        Generates reciprocal space q-vectors.

        Returns:
            QVectors: A dataclass containing qx, qy, qz meshes, and their norms.
        """
        box_dim = np.diag(frame.get_lattice())
        q_step = (2 * np.pi) / box_dim
        q_range_x = np.arange(q_step[0], q_step_scale, q_step[0])
        q_range_y = np.arange(q_step[1], q_step_scale, q_step[1])
        q_range_z = np.arange(q_step[2], q_step_scale, q_step[2])

        q_axis_x = np.concatenate([-q_range_x[::-1], [0], q_range_x])
        q_axis_y = np.concatenate([-q_range_y[::-1], [0], q_range_y])
        q_axis_z = np.concatenate([-q_range_z[::-1], [0], q_range_z])

        qx, qy, qz = np.meshgrid(q_axis_x, q_axis_y, q_axis_z, indexing="ij")

        q_norm = np.sqrt(qx**2 + qy**2 + qz**2)
        q_norm_unique = np.unique(q_norm)
        q_bins = q_norm_unique[q_norm_unique <= q_max]

        return QVectors(qx=qx, qy=qy, qz=qz, q_norm=q_norm, q_bins=q_bins)

    def _calculate_species_components(self, q_vectors: QVectors) -> SpeciesComponents:
        """
        Calculates the sum of sin(q*r) and cos(q*r) over all atoms of each species.
        """
        qsin, qcos, correlation_lengths = {}, {}, {}
        species_list = sorted(self._atoms_data.keys())

        for species in species_list:
            positions = self._atoms_data[species]

            with ProgressBar(
                total=len(positions),
                leave=False,
                colour="#00ffff",
                unit="atom",
                desc=f"Processing {species}",
                disable=not self._settings.verbose,
            ) as progress:
                cosd, sind = calculate_components(
                    q_vectors.qx, q_vectors.qy, q_vectors.qz, positions, progress
                )

            qcos[species], qsin[species] = cosd, sind

        return SpeciesComponents(
            qcos=qcos, qsin=qsin, correlation_lengths=self._correlation_lengths
        )

    def _calculate_all_partial_factors(
        self, pairs: list[str], components: SpeciesComponents
    ) -> dict:
        """
        Calculates the unbinned partial structure factors for each requested pair.
        """
        f = {}
        species_list = sorted(self._atoms_data.keys())
        species_counts = {s: len(self._atoms_data[s]) for s in species_list}
        number_of_atoms = np.sum(list(species_counts.values()))
        qcos, qsin = components.qcos, components.qsin

        iterator = self._progress_iterator(pairs, "Calculating partial factors", "pair")
        for pair in iterator:
            if pair == "total":
                continue

            species1, species2 = pair.split("-")
            if species1 == species2:
                f[pair] = (qcos[species1] ** 2 + qsin[species1] ** 2) / number_of_atoms
            else:
                A = (qcos[species1] + qcos[species2]) ** 2
                B = (qsin[species1] + qsin[species2]) ** 2
                C = qcos[species1] ** 2 + qsin[species1] ** 2
                D = qcos[species2] ** 2 + qsin[species2] ** 2
                f[pair] = (A + B - C - D) / (2 * number_of_atoms)

        if "total" in pairs:
            f["total"] = self._calculate_total_factor(
                f,
                components,
                species_list,
                list(species_counts.values()),
                number_of_atoms,
            )

        return f

    def _calculate_total_factor(
        self, f_partial, components, species_list, species_counts, num_atoms
    ):
        """Calculates the total structure factor from partial factors."""
        f_total = np.zeros_like(next(iter(f_partial.values())))
        cl = components.correlation_lengths

        for i, species in enumerate(species_list):
            f_total += cl[species] ** 2 * f_partial[f"{species}-{species}"]

        for s1, s2 in combinations(species_list, 2):
            pair_key = f"{s1}-{s2}" if f"{s1}-{s2}" in f_partial else f"{s2}-{s1}"
            f_total += 2 * cl[s1] * cl[s2] * f_partial[pair_key]

        normalization = sum(
            count * cl[species] ** 2
            for species, count in zip(species_list, species_counts)
        )
        f_total /= normalization / num_atoms
        return f_total

    def _bin_structure_factor(self, f_unbinned: dict, q_vectors: QVectors) -> dict:
        """
        Vectorized binning of structure factors by averaging over shells in q-space.
        """
        # 1. Flatten the 3D array of q-vector magnitudes.
        q_norm_flat = q_vectors.q_norm.ravel()
        q_bins = q_vectors.q_bins
        num_bins = len(q_bins) - 1

        # 2. Create a mask to identify all q-vectors that are NOT q=0.
        is_not_q_zero_mask = q_norm_flat > 1e-9

        # 3. Apply the mask to get the q-norms that we will actually bin.
        q_norms_to_bin = q_norm_flat[is_not_q_zero_mask]

        # 4. For these non-zero q-norms, determine which bin each one falls into.
        bin_indices = np.digitize(q_norms_to_bin, q_bins) - 1

        # 5. Create a new mask to select only the values that fall within our valid bin range (q <= q_max).
        # This mask has the same dimensions as `q_norms_to_bin`.
        is_in_valid_bin_mask = (bin_indices >= 0) & (bin_indices < num_bins)

        # 6. Get the final bin indices for the data points we are keeping.
        final_bin_indices = bin_indices[is_in_valid_bin_mask]

        # 7. Calculate the midpoints of the q-bins for the x-axis of the plot.
        q_midpoints = (q_bins[:-1] + q_bins[1:]) / 2
        binned_sf = {"q": q_midpoints}

        # 8. Count how many q-vectors fall into each bin.
        counts = np.bincount(final_bin_indices, minlength=num_bins)
        counts[counts == 0] = 1  # Avoid division by zero.

        # 9. Iterate through each partial/total structure factor and bin its values.
        iterator = self._progress_iterator(
            f_unbinned.items(), "Binning results", "pair"
        )
        for pair, unbinned_values in iterator:
            # a. Flatten the 3D S(q) data and select only the values for non-zero q.
            sf_to_bin = unbinned_values.ravel()[is_not_q_zero_mask]

            # b. From these, select only the S(q) values that correspond to valid bins.
            # `sf_to_bin` and `is_in_valid_bin_mask` have the same length, so this is a valid operation.
            sf_weights = sf_to_bin[is_in_valid_bin_mask]

            # c. Sum the S(q) values (weights) for each bin.
            sums = np.bincount(
                final_bin_indices,
                weights=sf_weights,
                minlength=num_bins,
            )

            # d. Calculate the average S(q) for each bin and store it.
            binned_sf[pair] = sums / counts

        return binned_sf
