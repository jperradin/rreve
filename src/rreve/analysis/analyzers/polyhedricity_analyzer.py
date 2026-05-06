import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...core.node import Node
from ...config.settings import Settings
from ...utils.geometry import (
    calculate_octahedricity,
    calculate_pbc_angle_combinations,
    calculate_pbc_cv_distances_batch,
    calculate_pbc_dot_distances_combinations,
    calculate_square_based_pyramid,
    calculate_tetrahedricity,
    calculate_triangular_bipyramid,
)


class PolyhedricityAnalyzer(BaseAnalyzer):
    """Polyhedricity analyzer."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._local_settings = self._settings.analysis.poly_settings
        self.polyhedricity: Optional[Dict[str, np.ndarray]] = None
        self.distribution_cv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_vv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_a: Optional[Dict[str, np.ndarray]] = None
        self.proportion: Optional[Dict[str, float]] = None
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None
        self.central_nodes: List[Node] = []
        self.poly_data: List[Dict[str, np.ndarray]] = []
        self.dist_cv_data: List[Dict[str, np.ndarray]] = []
        self.dist_vv_data: List[Dict[str, np.ndarray]] = []
        self.dist_a_data: List[Dict[str, np.ndarray]] = []
        self.counts = {
            "4_fold": 0,
            "5_fold": 0,
            "6_fold": 0,
            "5_fold_sbp": 0,
            "5_fold_tbp": 0,
        }
        self.counts_distribution = {
            "4_fold_cv": 0,
            "5_fold_cv": 0,
            "6_fold_cv": 0,
            "4_fold_vv": 0,
            "5_fold_vv": 0,
            "6_fold_vv": 0,
            "4_fold_a": 0,
            "5_fold_a": 0,
            "6_fold_a": 0,
            "4_fold_a_cvn_vni": 0,
            "4_fold_a_cvn_ni": 0,
            "4_fold_a_cvn_ffi": 0,
        }

        # POLYAnalysisSettings
        self.central_species: str = (
            self._local_settings.central_species
            if self._local_settings is not None
            else "Si"
        )

        self.vertices_species: str = (
            self._local_settings.vertices_species
            if self._local_settings is not None
            else "O"
        )
        self.max_c: float = (
            self._local_settings.max_c if self._local_settings is not None else 0.5
        )
        self.print_forms = (
            self._local_settings.print_forms
            if self._local_settings is not None
            else True
        )
        self.calculate_distribution = (
            self._local_settings.calculate_distributions
            if self._local_settings is not None
            else True
        )
        if self.print_forms:
            filename = os.path.join(
                self._settings.export_directory, "lifetime_forms.dat"
            )
            with open(filename, "w") as f:
                f.write("# lifetime_forms\n")
            f.close()

        self._bins: Optional[np.ndarray] = None
        self._dbin: Optional[float] = None
        self._mid: Optional[np.ndarray] = None
        self._rcv: Optional[np.ndarray] = None
        self._dbincv: Optional[float] = None
        self._midrcv: Optional[np.ndarray] = None
        self._rvv: Optional[np.ndarray] = None
        self._dbinvv: Optional[float] = None
        self._midrvv: Optional[np.ndarray] = None
        self._ra: Optional[np.ndarray] = None
        self._dbina: Optional[float] = None
        self._midra: Optional[np.ndarray] = None

    def _initialize_arrays(self) -> None:
        """Pre-compute arrays that don't change between calculations."""
        if self._bins is None:
            # Polyhedricity
            self._bins = np.linspace(0, self.max_c, 1000)
            self._dbin = self.max_c / 1000
            self._mid = (self._bins[:-1] + self._bins[1:]) / 2
            self._hist_4_fold = np.zeros(len(self._bins))
            # all 5 folds considered as square based pyramid
            self._hist_SBP_pentahedricity = np.zeros(len(self._bins))
            # all 5 folds considered as triangular bipyramid
            self._hist_TBP_pentahedricity = np.zeros(len(self._bins))
            # minimum between square based pyramid vs triangular bipyramid
            self._hist_5_fold = np.zeros(len(self._bins))
            self._hist_sbp_pentahedricity = np.zeros(len(self._bins))
            self._hist_tbp_pentahedricity = np.zeros(len(self._bins))
            self._hist_6_fold = np.zeros(len(self._bins))

            # Distances center-vertices
            self._rcv = np.linspace(0, 3, 1000)
            self._dbincv = self._rcv[1] - self._rcv[0]
            self._midrcv = (self._rcv[:-1] + self._rcv[1:]) / 2
            self._rd_4_fold_cv = np.zeros(len(self._bins))
            self._rd_5_fold_cv = np.zeros(len(self._bins))
            self._rd_6_fold_cv = np.zeros(len(self._bins))

            # Distances vertices-vertices
            self._rvv = np.linspace(0, 5, 1000)
            self._dbinvv = self._rvv[1] - self._rvv[0]
            self._midrvv = (self._rvv[:-1] + self._rvv[1:]) / 2
            self._rd_4_fold_vv = np.zeros(len(self._bins))
            self._rd_5_fold_vv = np.zeros(len(self._bins))
            self._rd_6_fold_vv = np.zeros(len(self._bins))

            # Angles vertices-vertices-vertices
            self._a = np.linspace(0, 180, 1000)
            self._dbina = self._a[1] - self._a[0]
            self._mida = (self._a[:-1] + self._a[1:]) / 2
            self._ad_4_fold = np.zeros(len(self._bins))
            self._ad_5_fold = np.zeros(len(self._bins))
            self._ad_6_fold = np.zeros(len(self._bins))

            # Angles center-vertex-neighbor
            self._ad_4_fold_cvn_very_near_ideal = np.zeros(len(self._bins))
            self._ad_4_fold_cvn_near_ideal = np.zeros(len(self._bins))
            self._ad_4_fold_cvn_far_from_ideal = np.zeros(len(self._bins))

    def analyze(self, frame: Frame) -> None:
        self._initialize_arrays()
        self._atoms_data = frame.nodes_data.wrapped_positions
        self.central_nodes = frame.get_nodes_by_element(self.central_species)
        lattice = frame.get_lattice()
        N = len(self.central_nodes)
        self.polyhedricity = {}

        self.distribution_cv = {}
        self.distribution_vv = {}
        self.distribution_a = {}

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(self.central_nodes),
            desc="Calculating polyhedricity ...",
            unit="atom",
            initial=0,
            total=N,
            **progress_bar_kwargs,
        )
        for i, node in progress_bar:
            if node.coordination not in (4, 5, 6):
                node.form = str(node.coordination)
                continue
            pos_batch = node.get_neighbors_positions_by_element(self.vertices_species)
            distances = calculate_pbc_dot_distances_combinations(pos_batch, lattice)
            distances.sort()
            if self.calculate_distribution:
                distances_cv = calculate_pbc_cv_distances_batch(
                    node.position, pos_batch, lattice
                )
                distances_cv.sort()
                angles = calculate_pbc_angle_combinations(pos_batch, lattice)
                angles.sort()

            polyhedricity = self._calculate_polyhedricity(node, distances)

            m, n = polyhedricity

            bin_idx1 = int(m / self._dbin) + 1
            if np.isnan(n):
                bin_idx2 = 0  # bin_idx2 is not used in this case
            else:
                bin_idx2 = int(n / self._dbin) + 1
            max_bin = int(max(self._bins) / self._dbin)

            if node.coordination == 4:
                # Polyhedricity
                if bin_idx1 >= max_bin:
                    continue
                self._hist_4_fold[bin_idx1] += 1
                self.counts["4_fold"] += 1
                node.form = "4"
                node.polyhedricity = m
                node.calculate_angles(lattice)
                if self.calculate_distribution:
                    # Distances center-vertices
                    for r in distances_cv:
                        bin_idx = int(r / self._dbincv) + 1
                        if bin_idx < max_bin:
                            self._rd_4_fold_cv[bin_idx] += 1
                    self.counts_distribution["4_fold_cv"] += len(distances_cv)
                    # Distances vertices-vertices
                    for r in distances:
                        bin_idx = int(r / self._dbinvv) + 1
                        if bin_idx < max_bin:
                            self._rd_4_fold_vv[bin_idx] += 1
                    self.counts_distribution["4_fold_vv"] += len(distances)
                    # Angles vertices-vertices-vertices
                    for a in angles:
                        bin_idx = int(a / self._dbina) + 1
                        if bin_idx < max_bin:
                            self._ad_4_fold[bin_idx] += 1
                    self.counts_distribution["4_fold_a"] += len(angles)
                    # Angles center-vertex-neighbor
                    for a in node.angles:
                        bin_idx = int(a / self._dbina) + 1
                        if bin_idx < max_bin and m < 0.003:
                            self._ad_4_fold_cvn_very_near_ideal[bin_idx] += 1
                            self.counts_distribution["4_fold_a_cvn_vni"] += 1
                        if bin_idx < max_bin and m < 0.0045:
                            self._ad_4_fold_cvn_near_ideal[bin_idx] += 1
                            self.counts_distribution["4_fold_a_cvn_ni"] += 1
                        if bin_idx < max_bin and m > 0.0045:
                            self._ad_4_fold_cvn_far_from_ideal[bin_idx] += 1
                            self.counts_distribution["4_fold_a_cvn_ffi"] += 1

            if node.coordination == 5:
                if bin_idx1 >= max_bin:
                    continue
                self.counts["5_fold"] += 1
                self._hist_SBP_pentahedricity[bin_idx1] += 1
                if bin_idx2 >= max_bin:
                    continue
                self._hist_TBP_pentahedricity[bin_idx2] += 1
                if m < n:
                    self.counts["5_fold_sbp"] += 1
                    self._hist_5_fold[bin_idx1] += 1
                    self._hist_sbp_pentahedricity[bin_idx1] += 1
                    node.form = "5p"
                    node.polyhedricity = m
                else:
                    self.counts["5_fold_tbp"] += 1
                    self._hist_5_fold[bin_idx2] += 1
                    self._hist_tbp_pentahedricity[bin_idx2] += 1
                    node.form = "5b"
                    node.polyhedricity = n
                if self.calculate_distribution:
                    # Distances center-vertices
                    for r in distances_cv:
                        bin_idx = int(r / self._dbincv) + 1
                        if bin_idx < max_bin:
                            self._rd_5_fold_cv[bin_idx] += 1
                    self.counts_distribution["5_fold_cv"] += len(distances_cv)
                    # Distances vertices-vertices
                    for r in distances:
                        bin_idx = int(r / self._dbinvv) + 1
                        if bin_idx < max_bin:
                            self._rd_5_fold_vv[bin_idx] += 1
                    self.counts_distribution["5_fold_vv"] += len(distances)
                    # Angles vertices-vertices-vertices
                    for a in angles:
                        bin_idx = int(a / self._dbina) + 1
                        if bin_idx < max_bin:
                            self._ad_5_fold[bin_idx] += 1
                    self.counts_distribution["5_fold_a"] += len(angles)
            if node.coordination == 6:
                if bin_idx1 >= max_bin:
                    continue
                self.counts["6_fold"] += 1
                self._hist_6_fold[bin_idx1] += 1
                node.form = "6"
                node.polyhedricity = m
                if self.calculate_distribution:
                    # Distances center-vertices
                    for r in distances_cv:
                        bin_idx = int(r / self._dbincv) + 1
                        if bin_idx < max_bin:
                            self._rd_6_fold_cv[bin_idx] += 1
                    self.counts_distribution["6_fold_cv"] += len(distances_cv)
                    # Distances vertices-vertices
                    for r in distances:
                        bin_idx = int(r / self._dbinvv) + 1
                        if bin_idx < max_bin:
                            self._rd_6_fold_vv[bin_idx] += 1
                    self.counts_distribution["6_fold_vv"] += len(distances)
                    # Angles vertices-vertices-vertices
                    for a in angles:
                        bin_idx = int(a / self._dbina) + 1
                        if bin_idx < max_bin:
                            self._ad_6_fold[bin_idx] += 1
                    self.counts_distribution["6_fold_a"] += len(angles)

        self.polyhedricity["4_fold"] = self._hist_4_fold
        self.polyhedricity["5_fold"] = self._hist_5_fold
        self.polyhedricity["6_fold"] = self._hist_6_fold
        self.polyhedricity["5_fold_sbp"] = self._hist_sbp_pentahedricity
        self.polyhedricity["5_fold_tbp"] = self._hist_tbp_pentahedricity
        self.polyhedricity["5_fold_SBP"] = self._hist_SBP_pentahedricity
        self.polyhedricity["5_fold_TBP"] = self._hist_TBP_pentahedricity
        self.distribution_cv["4_fold_cv"] = self._rd_4_fold_cv
        self.distribution_cv["5_fold_cv"] = self._rd_5_fold_cv
        self.distribution_cv["6_fold_cv"] = self._rd_6_fold_cv
        self.distribution_vv["4_fold_vv"] = self._rd_4_fold_vv
        self.distribution_vv["5_fold_vv"] = self._rd_5_fold_vv
        self.distribution_vv["6_fold_vv"] = self._rd_6_fold_vv
        self.distribution_a["4_fold_a"] = self._ad_4_fold
        self.distribution_a["5_fold_a"] = self._ad_5_fold
        self.distribution_a["6_fold_a"] = self._ad_6_fold
        self.distribution_a["4_fold_a_cvn_ni"] = self._ad_4_fold_cvn_near_ideal
        self.distribution_a["4_fold_a_cvn_ffi"] = self._ad_4_fold_cvn_far_from_ideal

        self.poly_data.append(dict(self.polyhedricity))
        self.dist_cv_data.append(dict(self.distribution_cv))
        self.dist_vv_data.append(dict(self.distribution_vv))
        self.dist_a_data.append(dict(self.distribution_a))
        self.frame_processed_count += 1

        if self._local_settings.print_forms:
            output_file = os.path.join(
                self._settings.export_directory, "lifetime_forms.dat"
            )
            line = ""
            for node in self.central_nodes:
                line += f"{node.form:<3}"
            line += "\n"
            with open(output_file, "a") as f:
                f.write(line)
            f.close()

    def finalize(self) -> None:
        # Set the same normalization to all SiO5
        self.counts["5_fold_SBP"] = self.counts["5_fold"]
        self.counts["5_fold_TBP"] = self.counts["5_fold"]

        if self.poly_data:
            keys = list(self.poly_data[0].keys())
            self.polyhedricity = {}
            self.proportion = {}

            self.polyhedricity["bins"] = self._bins

            for key in keys:
                values = np.array([frame_data[key] for frame_data in self.poly_data])
                # Avoid zero division
                if self.counts[key] == 0:
                    self.counts[key] = 1
                # Use number of SiO5 to normalize.
                if key == "5_fold_sbp" or key == "5_fold_tbp":
                    self.polyhedricity[key] = (
                        np.sum(values, axis=0)
                        / self.counts["5_fold"]
                        / self._dbin
                        / self.frame_processed_count
                    )
                else:
                    self.polyhedricity[key] = (
                        np.sum(values, axis=0)
                        / self.counts[key]
                        / self._dbin
                        / self.frame_processed_count
                    )
                if key not in ["5_fold_SBP", "5_fold_TBP"]:
                    self.proportion[key] = self.counts[key] / (
                        len(self.central_nodes) * self.frame_processed_count
                    )
        if self.calculate_distribution:
            if self.dist_cv_data:
                keys = list(self.dist_cv_data[0].keys())
                self.distribution_cv = {}

                self.distribution_cv["bins"] = self._rcv

                for key in keys:
                    values = np.array(
                        [frame_data[key] for frame_data in self.dist_cv_data]
                    )
                    # Avoid zero division
                    if self.counts_distribution[key] == 0:
                        self.counts_distribution[key] = 1
                    self.distribution_cv[key] = (
                        np.sum(values, axis=0)
                        / self.counts_distribution[key]
                        / self._dbincv
                        / self.frame_processed_count
                    )
            if self.dist_vv_data:
                keys = list(self.dist_vv_data[0].keys())
                self.distribution_vv = {}

                self.distribution_vv["bins"] = self._rvv

                for key in keys:
                    values = np.array(
                        [frame_data[key] for frame_data in self.dist_vv_data]
                    )
                    # Avoid zero division
                    if self.counts_distribution[key] == 0:
                        self.counts_distribution[key] = 1
                    self.distribution_vv[key] = (
                        np.sum(values, axis=0) 
                        / self.counts_distribution[key]
                        / self._dbinvv
                        / self.frame_processed_count
                    )
            if self.dist_a_data:
                keys = list(self.dist_a_data[0].keys())
                self.distribution_a = {}

                self.distribution_a["bins"] = self._a

                for key in keys:
                    values = np.array(
                        [frame_data[key] for frame_data in self.dist_a_data]
                    )
                    # Avoid zero division
                    if self.counts_distribution[key] == 0:
                        self.counts_distribution[key] = 1
                    self.distribution_a[key] = (
                        np.sum(values, axis=0)
                        / self.counts_distribution[key]
                        / self._dbina
                        / self.frame_processed_count
                    )

    def get_result(self) -> Dict[str, float]:
        return self.proportion

    def print_to_file(self) -> None:
        self.finalize()
        if self.polyhedricity is None:
            return

        output_path1 = os.path.join(
            self._settings.export_directory, "polyhedricity_proportions.dat"
        )
        output_path2 = os.path.join(
            self._settings.export_directory, "polyhedricity_histograms.dat"
        )
        output_path3 = os.path.join(
            self._settings.export_directory, "polyhedricity_distribution_cv.dat"
        )
        output_path4 = os.path.join(
            self._settings.export_directory, "polyhedricity_distribution_vv.dat"
        )
        output_path5 = os.path.join(
            self._settings.export_directory, "polyhedricity_distribution_a.dat"
        )

        keys = list(self.proportion.keys())
        data = np.column_stack([self.proportion[k] for k in keys])

        np.savetxt(
            output_path1,
            data,
            header=f"{keys}",
            delimiter="\t",
            fmt="%.5f",
            comments="# ",
        )

        keys = list(self.polyhedricity.keys())
        data = np.column_stack([self.polyhedricity[k] for k in keys])

        np.savetxt(
            output_path2,
            data,
            header=f"{keys}",
            delimiter="\t",
            fmt="%.5f",
            comments="# ",
        )

        if self.calculate_distribution:
            keys = list(self.distribution_cv.keys())
            data = np.column_stack([self.distribution_cv[k] for k in keys])

            np.savetxt(
                output_path3,
                data,
                header=f"{keys}",
                delimiter="\t",
                fmt="%.5f",
                comments="# ",
            )

            keys = list(self.distribution_vv.keys())
            data = np.column_stack([self.distribution_vv[k] for k in keys])

            np.savetxt(
                output_path4,
                data,
                header=f"{keys}",
                delimiter="\t",
                fmt="%.5f",
                comments="# ",
            )

            keys = list(self.distribution_a.keys())
            data = np.column_stack([self.distribution_a[k] for k in keys])
            np.savetxt(
                output_path5,
                data,
                header=f"{keys}",
                delimiter="\t",
                fmt="%.5f",
                comments="# ",
            )

    def _calculate_polyhedricity(
        self, node: Node, distances: np.ndarray
    ) -> Tuple[float, float]:
        # Possible errors
        if node.coordination == 4 and len(distances) != 6:
            raise ValueError(
                f"4-fold units should have 6 distances, got {len(distances)}"
            )
        if node.coordination == 5 and len(distances) != 10:
            raise ValueError(
                f"5-fold units should have 10 distances, got {len(distances)}"
            )
        if node.coordination == 6 and len(distances) != 15:
            raise ValueError(
                f"6-fold units should have 15 distances, got {len(distances)}"
            )

        if node.coordination == 4 and len(distances) == 6:
            # Calculate tetrahedricity
            tetrahedricity = calculate_tetrahedricity(distances)
            return tetrahedricity, np.nan
        if node.coordination == 5 and len(distances) == 10:
            # Calculate square based pyramid polyhedricity
            sbp_polyhedricity = calculate_square_based_pyramid(distances)
            # Calculate triangular bipyramid polyhedricity
            tbp_polyhedricity = calculate_triangular_bipyramid(distances)
            return sbp_polyhedricity, tbp_polyhedricity
        if node.coordination == 6 and len(distances) == 15:
            # Calculate triangular bipyramid polyhedricity
            octahedricity = calculate_octahedricity(distances)
            return octahedricity, np.nan
