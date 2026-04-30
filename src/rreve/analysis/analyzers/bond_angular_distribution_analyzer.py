import numpy as np
import os
from itertools import combinations
from tqdm import tqdm
from typing import Dict, Optional, List

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...config.settings import Settings
from ...utils.geometry import calculate_pbc_angle


class BondAngularDistributionAnalyzer(BaseAnalyzer):
    """
    Computes the bond angular distribution for specified triplets in the system.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.angles: Optional[Dict[str, np.ndarray]] = None
        self.angle_data: list = []
        self.angle_average_data: list = []
        self.angle_most_probable_data: list = []
        self.mean_angles: Optional[Dict[str, np.ndarray]] = None
        self.mp_angles: Optional[Dict[str, np.ndarray]] = None
        self.std_angles: Optional[Dict[str, np.ndarray]] = None
        self.angle_max: float = 180.0

        # BADAnalysisSettings
        self.bins: int = (
            self._settings.analysis.bad_settings.bins
            if self._settings.analysis.bad_settings is not None
            else 800
        )
        self.d_angle: float = (
            self.angle_max / self.bins
            if self._settings.analysis.bad_settings is not None
            else 180.0 / 800
        )
        self.triplets_to_analyze = (
            self._settings.analysis.bad_settings.triplets_to_calculate
            if self._settings.analysis.bad_settings is not None
            else ["O-O-O", "O-Si-O", "Si-O-Si", "Si-Si-Si"]
        )

    def analyze(self, frame: Frame) -> None:
        angles_for_frame = {triplet: [] for triplet in self.triplets_to_analyze}

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(frame.nodes),
            desc="Calculating angles ...",
            unit="atom",
            initial=0,
            total=len(frame.nodes),
            **progress_bar_kwargs,
        )
        for i, node in progress_bar:
            if len(node.neighbors) < 2:
                continue

            for neighbor1, neighbor2 in combinations(node.neighbors, 2):
                if neighbor1.node_id != neighbor2.node_id:
                    for triplet_type in self.triplets_to_analyze:
                        neighbor_atom1, central_atom, neighbor_atom2 = (
                            triplet_type.split("-")
                        )

                        # Check if the triplet matches the one to be analyzed
                        if node.symbol == central_atom and (
                            (
                                neighbor1.symbol == neighbor_atom1
                                and neighbor2.symbol == neighbor_atom2
                            )
                            or (
                                neighbor1.symbol == neighbor_atom2
                                and neighbor2.symbol == neighbor_atom1
                            )
                        ):
                            angle = calculate_pbc_angle(
                                neighbor1.position,
                                node.position,
                                neighbor2.position,
                                frame.lattice,
                            )
                            angles_for_frame[triplet_type].append(angle)

        # Histogram the angles for the current frame
        histograms = {}
        averages = {}
        most_probable_angles = {}
        for triplet, angle_list in angles_for_frame.items():
            hist, _ = np.histogram(
                angle_list, bins=self.bins, range=(0, self.angle_max)
            )
            histograms[triplet] = hist
            # Calculate the average angle for this triplet
            if triplet == 'Si-O-Si':
                # Select angles in the range [110, 180]
                angle_list = [angle for angle in angle_list if 110 <= angle <= 180]
            averages[triplet] = np.mean(angle_list)
            # Find the most probable angle for this triplet
            if len(angle_list) > 0:
                if triplet == 'Si-O-Si':
                    # Select angles in the range [110, 180]
                    angle_list = [angle for angle in angle_list if 110 <= angle <= 180]
                    _hist, _ = np.histogram(
                        angle_list, bins=self.bins, range=(0, self.angle_max)
                    )
                    max_index = np.argmax(_hist)
                else:
                    max_index = np.argmax(hist)
                most_probable_angles[triplet] = (max_index + 0.5) * self.d_angle

        if histograms:
            self.angle_data.append(histograms)
            self.angle_average_data.append(averages)
            self.angle_most_probable_data.append(most_probable_angles)
            self.frame_processed_count += 1

    def finalize(self) -> None:
        if self.frame_processed_count == 0:
            return

        # Sum the histograms from all frames
        summed_hist = {
            triplet: np.zeros(self.bins) for triplet in self.triplets_to_analyze
        }

        for frame_hist in self.angle_data:
            for triplet, hist in frame_hist.items():
                summed_hist[triplet] += hist

        self.mean_angles = {}
        self.std_angles = {}
        self.mp_angles = {}
        for i in range(len(self.angle_average_data)):
            frame_average = self.angle_average_data[i]
            for triplet, this_average in frame_average.items():
                if triplet not in self.mean_angles:
                    self.mean_angles[triplet] = []
                    self.std_angles[triplet] = []
                    self.mp_angles[triplet] = []
                self.mean_angles[triplet].append(this_average)
                self.std_angles[triplet].append(this_average)
                self.mp_angles[triplet].append(
                    self.angle_most_probable_data[i][triplet]
                )
        # Calculate mean and std for each triplet
        for triplet in self.triplets_to_analyze:
            if triplet in self.mean_angles:
                if len(self.mean_angles[triplet]) > 0:
                    self.mean_angles[triplet] = np.mean(self.mean_angles[triplet])
                    self.mp_angles[triplet] = np.mean(self.mp_angles[triplet])
                else:
                    self.mean_angles[triplet] = 0.0
                    self.mp_angles[triplet] = 0.0
                if len(self.std_angles[triplet]) > 1:
                    self.std_angles[triplet] = np.std(self.std_angles[triplet])
                else:
                    self.std_angles[triplet] = 0.0
            else:
                self.mean_angles[triplet] = 0.0
                self.mp_angles[triplet] = 0.0
                self.std_angles[triplet] = 0.0
        
        # Normalize the distributions
        self.angles = {}
        for triplet, hist in summed_hist.items():
            total_angles = np.sum(hist)
            if total_angles > 0:
                self.angles[triplet] = hist / (
                    total_angles * self.d_angle
                )  # Normalize by integral
            else:
                self.angles[triplet] = np.zeros_like(hist)

    def get_result(self) -> Dict[str, np.ndarray]:
        return self.angles if self.angles is not None else {}

    def print_to_file(self) -> None:
        self.finalize()
        if self.angles is None:
            return
        output_path = os.path.join(
            self._settings.export_directory, "bond_angular_distribution.dat"
        )
        angle_bins = (
            np.linspace(0, self.angle_max, self.bins, endpoint=False)
            + self.d_angle / 2.0
        )

        with open(output_path, "w") as f:
            header = "# Angle\t" + "\t".join(self.angles.keys()) + "\n"
            f.write(header)
            for i in range(len(angle_bins)):
                line = f"{angle_bins[i]:.2f}"
                for triplet in self.angles.keys():
                    line += f"\t{self.angles[triplet][i]:.5f}"
                f.write(line + "\n")

        output_path = os.path.join(self._settings.export_directory, "mean_angles.dat")
        with open(output_path, "w") as f:
            header = "# Angle\t" + "\t".join(self.mean_angles.keys()) + "\n"
            f.write(header)
            line = ""
            for triplet in self.mean_angles.keys():
                line += f"{self.mean_angles[triplet]:.5f}\t"
            f.write(line.strip() + "\n")
            
        output_path = os.path.join(self._settings.export_directory, "mp_angles.dat")
        with open(output_path, "w") as f:
            header = "# Angle\t" + "\t".join(self.mp_angles.keys()) + "\n"
            f.write(header)
            line = ""
            for triplet in self.mp_angles.keys():
                line += f"{self.mp_angles[triplet]:.5f}\t"
            f.write(line.strip() + "\n")
    
        output_path = os.path.join(self._settings.export_directory, "std_angles.dat")
        with open(output_path, "w") as f:
            header = "# Angle\t" + "\t".join(self.std_angles.keys()) + "\n"
            f.write(header)
            line = ""
            for triplet in self.std_angles.keys():
                line += f"{self.std_angles[triplet]:.5f}\t"
            f.write(line.strip() + "\n")
