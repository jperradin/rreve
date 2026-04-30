from typing import Optional, List, Dict
import numpy as np
import os
from dataclasses import dataclass
from itertools import combinations
from tqdm import tqdm
from scipy.stats import binned_statistic

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...config.settings import Settings


@dataclass
class QVectors:
    """Holds reciprocal spaces q-vectors and their binning informations."""

    q_norm: np.ndarray
    q_bins: np.ndarray


class NeutronStructureFactorFFTAnalyzer(BaseAnalyzer):
    """
    Computes the neutron structure factor using an efficient FFT-based method.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.q = None
        self.nsf = None
        self._atoms_data: Optional(Dict[str, nd.ndarray]) = None
        # For optimal FFT performance, grid size should be a power of 2
        self.grid_size = 128
        self.q_max = 10.0
        self.q_bins = 100
        self.nsf_data = []

    def analyze(self, frame: Frame) -> None:
        self._atoms_data = frame.nodes_data.wrapped_positions
        self._correlation_lengths: Dict[str, float] = (
            frame.nodes_data.correlation_lengths
        )

        # Dynamically adjust grid_size based on box dimensions and desired q_max
        L = np.max(np.diag(frame.get_lattice()))
        # Ensure grid is large enough to reach q_max
        required_grid_size = int(np.ceil(self.q_max * L / np.pi))
        # Use next power of 2 for FFT efficiency, but not smaller than the default
        self.grid_size = max(self.grid_size, 1 << (required_grid_size - 1).bit_length())

        pairs = self._get_pairs(list(self._atoms_data.keys()))
        self.calculate_s_q_fft(pairs, frame)
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

    def get_result(self) -> dict[str, float]:
        return self.nsf

    def print_to_file(self) -> None:
        self.finalize()
        if self.nsf is None:
            return
        output_path = os.path.join(
            self._settings.export_directory, "neutron_structure_factor_fft.dat"
        )
        with open(output_path, "w") as f:
            header = (
                f"# grid_size : {self.grid_size}\n"
                + "# q\t"
                + "\t".join(self.nsf.keys())
                + "\n"
            )
            f.write(header)
            for i in range(2, len(self.q)):
                # Skipping 2 values
                line = f"{self.q[i]:.5f}"
                for key in self.nsf.keys():
                    line += f"\t{self.nsf[key][i]:.5f}"
                f.write(line + "\n")

    def _get_pairs(self, species: list[str]) -> list[str]:
        pairs = [f"{s1}-{s2}" for s1, s2 in combinations(species, 2)]
        for s in species:
            pairs.append(f"{s}-{s}")
        pairs.append("total")
        return pairs

    def _progress_iterator(self, iterable, desc: str, unit: str = "it"):
        """Returns a tqdm iterator if not in quiet mode, otherwise returns the original iterable."""
        if self._settings.verbose:
            return tqdm(iterable, desc=desc, colour="#00ffff", leave=False, unit=unit)
        return iterable

    def calculate_s_q_fft(self, pairs: list[str], frame: Frame) -> None:
        """
        Calculates the neutron structure factor using an FFT-based method.
        """
        grid_shape = (self.grid_size, self.grid_size, self.grid_size)
        lattice = frame.get_lattice()
        box_size = np.diag(lattice)

        rho_q = {}
        iterator = self._progress_iterator(
            self._atoms_data.items(), "Processing species", "species"
        )
        for species, positions in iterator:
            density_grid, _ = np.histogramdd(
                positions, bins=grid_shape, range=[(0, L) for L in box_size]
            )
            rho_q[species] = np.fft.fftn(density_grid)

        q_vectors = self._generate_fft_q_vectors(lattice, grid_shape)

        # --- Calculate Total Structure Factor ---
        rho_q_total = np.zeros_like(next(iter(rho_q.values())), dtype=np.complex128)
        for species, rq in rho_q.items():
            rho_q_total += self._correlation_lengths[species] * rq

        num_atoms = sum(len(pos) for pos in self._atoms_data.values())
        # Normalization factor: <b^2>
        b_sq_avg = (
            sum(
                len(self._atoms_data[s]) * self._correlation_lengths[s] ** 2
                for s in self._atoms_data
            )
            / num_atoms
        )

        s_q_total_unbinned = (np.abs(rho_q_total) ** 2) / (num_atoms * b_sq_avg)

        # --- Calculate Partial Structure Factors ---
        f_unbinned = self._calculate_factors_from_rho_q(pairs, rho_q)

        # --- Binning ---
        # Shift the zero-frequency component to the center
        s_q_total_unbinned = np.fft.fftshift(s_q_total_unbinned)
        for pair in f_unbinned:
            f_unbinned[pair] = np.fft.fftshift(f_unbinned[pair])
        q_vectors.q_norm = np.fft.fftshift(q_vectors.q_norm)

        # Bin the total structure factor
        q_norm_flat = q_vectors.q_norm.ravel()
        q_bins = q_vectors.q_bins
        binned_total_sf, _, _ = binned_statistic(
            q_norm_flat, s_q_total_unbinned.ravel(), statistic="mean", bins=q_bins
        )

        # Bin the partials
        structure_factor_binned = self._bin_structure_factor(f_unbinned, q_vectors)
        structure_factor_binned["total"] = np.nan_to_num(binned_total_sf)

        self.q = structure_factor_binned.pop("q")
        self.nsf = structure_factor_binned

    def _generate_fft_q_vectors(
        self, lattice: np.ndarray, grid_shape: tuple
    ) -> QVectors:
        """Generates the q-vectors corresponding to the FFT grid."""
        box_size = np.diag(lattice)
        q_vecs = [
            np.fft.fftfreq(n, d=L / n) * 2 * np.pi for n, L in zip(grid_shape, box_size)
        ]
        qx, qy, qz = np.meshgrid(q_vecs[0], q_vecs[1], q_vecs[2], indexing="ij")
        q_norm = np.sqrt(qx**2 + qy**2 + qz**2)

        q_bins = np.linspace(0, self.q_max, self.q_bins)
        return QVectors(q_norm=q_norm, q_bins=q_bins)

    def _calculate_factors_from_rho_q(self, pairs: list[str], rho_q: dict) -> dict:
        """Calculates partial and total structure factors from Fourier-space densities."""
        f = {}
        species_list = sorted(self._atoms_data.keys())
        species_counts = {s: len(self._atoms_data[s]) for s in species_list}
        num_atoms = sum(species_counts.values())
        concentrations = {s: species_counts[s] / num_atoms for s in species_list}

        for pair in self._progress_iterator(
            pairs, "Calculating partial factors", "pair"
        ):
            if pair == "total":
                continue
            s1, s2 = pair.split("-")

            # Renormalize to get the Faber-Ziman partials
            if s1 == s2:
                f[pair] = (
                    np.abs(rho_q[s1]) ** 2 - species_counts[s1]
                ) / species_counts[s1]
            else:
                f[pair] = (rho_q[s1] * np.conj(rho_q[s2])).real / np.sqrt(
                    species_counts[s1] * species_counts[s2]
                )

        # Construct the partials the user expects
        s_user = {}
        for pair in f:
            s1, s2 = pair.split("-")
            if s1 == s2:
                s_user[pair] = concentrations[s1] * (1 + f[pair])
            else:
                s_user[pair] = (
                    np.sqrt(concentrations[s1] * concentrations[s2]) * f[pair]
                )

        if "total" in pairs:
            s_user["total"] = self._calculate_total_factor(
                s_user, species_list, species_counts, num_atoms
            )

        return s_user

    def _calculate_total_factor(
        self, f_partial, species_list, species_counts, num_atoms
    ):
        """Calculates the total structure factor from partial factors using the Faber-Ziman formalism."""
        f_total = np.zeros_like(next(iter(f_partial.values())))
        cl = self._correlation_lengths  # These are the neutron scattering lengths

        concentrations = {s: species_counts[s] / num_atoms for s in species_list}

        # Correct normalization for Faber-Ziman
        b_avg = sum(concentrations[s] * cl[s] for s in species_list)
        b_avg_sq = b_avg**2

        for s1 in species_list:
            for s2 in species_list:
                pair_key = f"{s1}-{s2}" if f"{s1}-{s2}" in f_partial else f"{s2}-{s1}"
                f_total += (
                    concentrations[s1]
                    * concentrations[s2]
                    * cl[s1]
                    * cl[s2]
                    * f_partial[pair_key]
                )

        if b_avg_sq > 0:
            f_total /= b_avg_sq

        return f_total

    def _bin_structure_factor(self, f_unbinned: dict, q_vectors: QVectors) -> dict:
        """Vectorized binning of structure factors by averaging over shells in q-space."""
        q_norm_flat = q_vectors.q_norm.ravel()
        q_bins = q_vectors.q_bins

        q_binned = (q_bins[:-1] + q_bins[1:]) / 2
        binned_sf = {"q": q_binned}

        for pair, unbinned_values in f_unbinned.items():
            binned_values, _, _ = binned_statistic(
                q_norm_flat, unbinned_values.ravel(), statistic="mean", bins=q_bins
            )
            binned_sf[pair] = np.nan_to_num(
                binned_values
            )  # Replace NaNs from empty bins with 0

        return binned_sf
