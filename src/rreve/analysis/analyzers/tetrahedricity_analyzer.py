import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Optional, List

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...core.node import Node
from ...config.settings import Settings
from ...utils.geometry import (
    calculate_pbc_angle_combinations,
    calculate_pbc_cv_angle_combinations,
    calculate_pbc_cv_distances_batch,
    calculate_pbc_dot_distances_combinations,
    calculate_tetrahedricity,
    calculate_tetrahedricity_angles,
)


class TetrahedricityAnalyzer(BaseAnalyzer):
    """Tetrahedricity analyzer.

    Measures the irregularity of a tetrahedron (4-fold unit) with respect to a
    regular tetrahedron through three independent methods:

    - ``vv``: based on the vertex-vertex distances (same metric as the
      ``PolyhedricityAnalyzer`` tetrahedricity).
    - ``cv``: based on the vertex-center-vertex angles, compared to the ideal
      tetrahedral angle (~109.47 deg).
    - ``vvv``: based on the vertex-vertex-vertex angles, compared to the ideal
      equilateral-triangle angle (60 deg).
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._local_settings = self._settings.analysis.tetra_settings
        self.tetrahedricity: Optional[Dict[str, np.ndarray]] = None
        self.distribution_cv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_vv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_acv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_avvv: Optional[Dict[str, np.ndarray]] = None
        self.proportion: Optional[Dict[str, float]] = None
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None
        self.central_nodes: List[Node] = []
        self.tetra_data: List[Dict[str, np.ndarray]] = []
        self.dist_cv_data: List[Dict[str, np.ndarray]] = []
        self.dist_vv_data: List[Dict[str, np.ndarray]] = []
        self.dist_acv_data: List[Dict[str, np.ndarray]] = []
        self.dist_avvv_data: List[Dict[str, np.ndarray]] = []
        self.counts = {
            "4_fold": 0,
        }
        self.counts_distribution = {
            "4_fold_cv": 0,
            "4_fold_vv": 0,
            "4_fold_acv": 0,
            "4_fold_avvv": 0,
        }

        # TETRAAnalysisSettings
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
            self._local_settings.max_c if self._local_settings is not None else 0.2
        )
        self.ideal_cv_angle: float = (
            self._local_settings.ideal_cv_angle
            if self._local_settings is not None
            else 109.47
        )
        self.ideal_vvv_angle: float = (
            self._local_settings.ideal_vvv_angle
            if self._local_settings is not None
            else 60.0
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
                self._settings.export_directory, "tetra_lifetime_forms.dat"
            )
            with open(filename, "w") as f:
                f.write("# tetra_lifetime_forms\n")
            f.close()

        self._bins: Optional[np.ndarray] = None
        self._dbin: Optional[float] = None

    def _initialize_arrays(self) -> None:
        """Pre-compute arrays that don't change between calculations."""
        if self._bins is None:
            # Tetrahedricity (irregularity) histograms
            self._bins = np.linspace(0, self.max_c, 1000)
            self._dbin = self.max_c / 1000
            self._mid = (self._bins[:-1] + self._bins[1:]) / 2
            self._hist_vv = np.zeros(len(self._bins))
            self._hist_cv = np.zeros(len(self._bins))
            self._hist_vvv = np.zeros(len(self._bins))

            # Distances center-vertices
            self._rcv = np.linspace(0, 3, 1000)
            self._dbincv = self._rcv[1] - self._rcv[0]
            self._midrcv = (self._rcv[:-1] + self._rcv[1:]) / 2
            self._rd_cv = np.zeros(len(self._bins))

            # Distances vertices-vertices
            self._rvv = np.linspace(0, 5, 1000)
            self._dbinvv = self._rvv[1] - self._rvv[0]
            self._midrvv = (self._rvv[:-1] + self._rvv[1:]) / 2
            self._rd_vv = np.zeros(len(self._bins))

            # Angles vertex-center-vertex and vertex-vertex-vertex
            self._a = np.linspace(0, 180, 1000)
            self._dbina = self._a[1] - self._a[0]
            self._mida = (self._a[:-1] + self._a[1:]) / 2
            self._ad_cv = np.zeros(len(self._bins))
            self._ad_vvv = np.zeros(len(self._bins))

    def analyze(self, frame: Frame) -> None:
        self._initialize_arrays()
        self._atoms_data = frame.nodes_data.wrapped_positions
        self.central_nodes = frame.get_nodes_by_element(self.central_species)
        lattice = frame.get_lattice()
        N = len(self.central_nodes)
        self.tetrahedricity = {}
        self.distribution_cv = {}
        self.distribution_vv = {}
        self.distribution_acv = {}
        self.distribution_avvv = {}

        progress_bar_kwargs = {
            "disable": not self._settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "blue",
        }

        progress_bar = tqdm(
            enumerate(self.central_nodes),
            desc="Calculating tetrahedricity ...",
            unit="atom",
            initial=0,
            total=N,
            **progress_bar_kwargs,
        )

        max_bin = int(max(self._bins) / self._dbin)

        for i, node in progress_bar:
            if node.coordination != 4:
                node.form = str(node.coordination)
                continue

            pos_batch = node.get_neighbors_positions_by_element(self.vertices_species)
            if len(pos_batch) != 4:
                node.form = str(node.coordination)
                continue

            # vertex-vertex distances
            distances = calculate_pbc_dot_distances_combinations(pos_batch, lattice)
            distances.sort()
            # vertex-center-vertex angles
            angles_cv = calculate_pbc_cv_angle_combinations(
                node.position, pos_batch, lattice
            )
            angles_cv.sort()
            # vertex-vertex-vertex angles
            angles_vvv = calculate_pbc_angle_combinations(pos_batch, lattice)
            angles_vvv.sort()

            tetra_vv = calculate_tetrahedricity(distances)
            tetra_cv = calculate_tetrahedricity_angles(angles_cv, self.ideal_cv_angle)
            tetra_vvv = calculate_tetrahedricity_angles(
                angles_vvv, self.ideal_vvv_angle
            )

            bin_vv = int(tetra_vv / self._dbin) + 1
            bin_cv = int(tetra_cv / self._dbin) + 1
            bin_vvv = int(tetra_vvv / self._dbin) + 1

            if bin_vv >= max_bin:
                continue

            self.counts["4_fold"] += 1
            self._hist_vv[bin_vv] += 1
            if bin_cv < max_bin:
                self._hist_cv[bin_cv] += 1
            if bin_vvv < max_bin:
                self._hist_vvv[bin_vvv] += 1

            node.form = "4"
            node.polyhedricity = tetra_vv

            if self.calculate_distribution:
                distances_cv = calculate_pbc_cv_distances_batch(
                    node.position, pos_batch, lattice
                )
                distances_cv.sort()
                # Distances center-vertices
                for r in distances_cv:
                    bin_idx = int(r / self._dbincv) + 1
                    if bin_idx < max_bin:
                        self._rd_cv[bin_idx] += 1
                self.counts_distribution["4_fold_cv"] += len(distances_cv)
                # Distances vertices-vertices
                for r in distances:
                    bin_idx = int(r / self._dbinvv) + 1
                    if bin_idx < max_bin:
                        self._rd_vv[bin_idx] += 1
                self.counts_distribution["4_fold_vv"] += len(distances)
                # Angles vertex-center-vertex
                for a in angles_cv:
                    bin_idx = int(a / self._dbina) + 1
                    if bin_idx < max_bin:
                        self._ad_cv[bin_idx] += 1
                self.counts_distribution["4_fold_acv"] += len(angles_cv)
                # Angles vertex-vertex-vertex
                for a in angles_vvv:
                    bin_idx = int(a / self._dbina) + 1
                    if bin_idx < max_bin:
                        self._ad_vvv[bin_idx] += 1
                self.counts_distribution["4_fold_avvv"] += len(angles_vvv)

        self.tetrahedricity["4_fold_vv"] = self._hist_vv
        self.tetrahedricity["4_fold_cv"] = self._hist_cv
        self.tetrahedricity["4_fold_vvv"] = self._hist_vvv
        self.distribution_cv["4_fold_cv"] = self._rd_cv
        self.distribution_vv["4_fold_vv"] = self._rd_vv
        self.distribution_acv["4_fold_acv"] = self._ad_cv
        self.distribution_avvv["4_fold_avvv"] = self._ad_vvv

        self.tetra_data.append(dict(self.tetrahedricity))
        self.dist_cv_data.append(dict(self.distribution_cv))
        self.dist_vv_data.append(dict(self.distribution_vv))
        self.dist_acv_data.append(dict(self.distribution_acv))
        self.dist_avvv_data.append(dict(self.distribution_avvv))
        self.frame_processed_count += 1

        if self.print_forms:
            output_file = os.path.join(
                self._settings.export_directory, "tetra_lifetime_forms.dat"
            )
            line = ""
            for node in self.central_nodes:
                line += f"{node.form:<3}"
            line += "\n"
            with open(output_file, "a") as f:
                f.write(line)
            f.close()

    def finalize(self) -> None:
        if self.tetra_data:
            keys = list(self.tetra_data[0].keys())
            self.tetrahedricity = {}
            self.proportion = {}

            self.tetrahedricity["bins"] = self._bins

            for key in keys:
                values = np.array([frame_data[key] for frame_data in self.tetra_data])
                self.tetrahedricity[key] = self._normalize_to_unit_area(
                    np.sum(values, axis=0), self._dbin
                )
            self.proportion["4_fold"] = self.counts["4_fold"] / (
                len(self.central_nodes) * self.frame_processed_count
            )

        if self.calculate_distribution:
            self._finalize_distribution(
                "distribution_cv", self.dist_cv_data, self._rcv, self._dbincv
            )
            self._finalize_distribution(
                "distribution_vv", self.dist_vv_data, self._rvv, self._dbinvv
            )
            self._finalize_distribution(
                "distribution_acv", self.dist_acv_data, self._a, self._dbina
            )
            self._finalize_distribution(
                "distribution_avvv", self.dist_avvv_data, self._a, self._dbina
            )

    def _finalize_distribution(
        self,
        attr: str,
        data: List[Dict[str, np.ndarray]],
        bins: np.ndarray,
        dbin: float,
    ) -> None:
        if not data:
            return
        keys = list(data[0].keys())
        result = {"bins": bins}
        for key in keys:
            values = np.array([frame_data[key] for frame_data in data])
            result[key] = self._normalize_to_unit_area(
                np.sum(values, axis=0), dbin
            )
        setattr(self, attr, result)

    @staticmethod
    def _normalize_to_unit_area(hist: np.ndarray, dbin: float) -> np.ndarray:
        """Normalize a histogram so its integral (area) equals 1."""
        area = np.sum(hist) * dbin
        if area > 0:
            return hist / area
        return hist

    def get_result(self) -> Dict[str, float]:
        return self.proportion

    def print_to_file(self) -> None:
        self.finalize()
        if self.tetrahedricity is None:
            return

        output_path1 = os.path.join(
            self._settings.export_directory, "tetrahedricity_proportions.dat"
        )
        output_path2 = os.path.join(
            self._settings.export_directory, "tetrahedricity_histograms.dat"
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

        keys = list(self.tetrahedricity.keys())
        data = np.column_stack([self.tetrahedricity[k] for k in keys])
        np.savetxt(
            output_path2,
            data,
            header=f"{keys}",
            delimiter="\t",
            fmt="%.5f",
            comments="# ",
        )

        if self.calculate_distribution:
            self._save_distribution(
                self.distribution_cv, "tetrahedricity_distribution_cv.dat"
            )
            self._save_distribution(
                self.distribution_vv, "tetrahedricity_distribution_vv.dat"
            )
            self._save_distribution(
                self.distribution_acv, "tetrahedricity_distribution_acv.dat"
            )
            self._save_distribution(
                self.distribution_avvv, "tetrahedricity_distribution_avvv.dat"
            )

    def _save_distribution(
        self, distribution: Optional[Dict[str, np.ndarray]], filename: str
    ) -> None:
        if distribution is None:
            return
        output_path = os.path.join(self._settings.export_directory, filename)
        keys = list(distribution.keys())
        data = np.column_stack([distribution[k] for k in keys])
        np.savetxt(
            output_path,
            data,
            header=f"{keys}",
            delimiter="\t",
            fmt="%.5f",
            comments="# ",
        )
