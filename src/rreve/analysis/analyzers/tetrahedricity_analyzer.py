import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Optional, List

from .base_analyzer import BaseAnalyzer
from ...core.frame import Frame
from ...core.node import Node
from ...config.settings import Settings
from ...utils.geometry import (
    calculate_pbc_angle,
    calculate_pbc_angle_combinations,
    calculate_pbc_cv_angle_combinations,
    calculate_pbc_cv_distances_batch,
    calculate_pbc_dot_distances_combinations,
    calculate_tetrahedricity,
    calculate_tetrahedricity_angles,
    calculate_errington_q,
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

    In addition, the Errington-Debenedetti orientational order parameter ``q``
    (Nature 409, 318 (2001)) is accumulated in two flavors, both Si-centred and
    built from the same 4 oxygen vertices:

    - ``q_vcv``: the original water q applied to the O-Si-O apex angles (ideal
      109.47 deg, cos = -1/3, prefactor 3/8).
    - ``q_vvv``: an adaptation to the O-O-O vertex-vertex-vertex angles (ideal
      60 deg, cos = 1/2, prefactor 1/7).

    Both are normalized so q = 1 for a perfect tetrahedron and <q> = 0 for
    randomly oriented neighbours. Note this is the *opposite* convention to the
    ``vv``/``vcv``/``vvv`` irregularity metrics above, which are 0 for a perfect
    tetrahedron.

    In addition to the per-tetrahedron metrics above, the analyzer accumulates
    the ``cvc`` (center-vertex-center) angle distribution: the inter-tetrahedral
    bridging angle (e.g. Si-O-Si), measured at each bridging vertex with the two
    arms pointing to the central atoms it connects.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._local_settings = self._settings.analysis.tetra_settings
        self.tetrahedricity: Optional[Dict[str, np.ndarray]] = None
        self.distribution_dcv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_dvv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_avcv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_avvv: Optional[Dict[str, np.ndarray]] = None
        self.distribution_acvc: Optional[Dict[str, np.ndarray]] = None
        self.distribution_q: Optional[Dict[str, np.ndarray]] = None
        self.proportion: Optional[Dict[str, float]] = None
        self._atoms_data: Optional[Dict[str, np.ndarray]] = None
        self.central_nodes: List[Node] = []
        self.tetra_data: List[Dict[str, np.ndarray]] = []
        self.dist_dcv_data: List[Dict[str, np.ndarray]] = []
        self.dist_dvv_data: List[Dict[str, np.ndarray]] = []
        self.dist_avcv_data: List[Dict[str, np.ndarray]] = []
        self.dist_avvv_data: List[Dict[str, np.ndarray]] = []
        self.dist_acvc_data: List[Dict[str, np.ndarray]] = []
        self.dist_q_data: List[Dict[str, np.ndarray]] = []
        self.counts = {
            "4_fold": 0,
        }
        self.counts_distribution = {
            "4_fold_dcv": 0,
            "4_fold_dvv": 0,
            "4_fold_avcv": 0,
            "4_fold_avvv": 0,
            "4_fold_acvc": 0,
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
        self.ideal_vcv_angle: float = (
            self._local_settings.ideal_vcv_angle
            if self._local_settings is not None
            else 109.47
        )
        self.ideal_vvv_angle: float = (
            self._local_settings.ideal_vvv_angle
            if self._local_settings is not None
            else 60.0
        )
        self.dcv_max: float = (
            self._local_settings.dcv_max
            if self._local_settings is not None
            else 3.0
        )
        self.dvv_max: float = (
            self._local_settings.dvv_max
            if self._local_settings is not None
            else 5.0
        )
        self.q_min: float = (
            self._local_settings.q_min
            if self._local_settings is not None
            else -0.5
        )
        self.q_max: float = (
            self._local_settings.q_max
            if self._local_settings is not None
            else 1.0
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
            self._hist_vcv = np.zeros(len(self._bins))
            self._hist_vvv = np.zeros(len(self._bins))

            # Distances center-vertices
            self._rdcv = np.linspace(0, self.dcv_max, 1000)
            self._dbindcv = self._rdcv[1] - self._rdcv[0]
            self._midrcv = (self._rdcv[:-1] + self._rdcv[1:]) / 2
            self._rd_dcv = np.zeros(len(self._bins))

            # Distances vertices-vertices
            self._rdvv = np.linspace(0, self.dvv_max, 1000)
            self._dbindvv = self._rdvv[1] - self._rdvv[0]
            self._midrvv = (self._rdvv[:-1] + self._rdvv[1:]) / 2
            self._rd_dvv = np.zeros(len(self._bins))

            # Angles vertex-center-vertex and vertex-vertex-vertex
            self._a = np.linspace(0, 180, 1000)
            self._dbina = self._a[1] - self._a[0]
            self._mida = (self._a[:-1] + self._a[1:]) / 2
            self._ad_vcv = np.zeros(len(self._bins))
            self._ad_vvv = np.zeros(len(self._bins))

            # Inter-tetrahedral center-vertex-center angles (bridging angle)
            self._ad_cvc = np.zeros(len(self._bins))

            # Errington-Debenedetti order parameter q (q = 1 perfect tetra)
            self._q = np.linspace(self.q_min, self.q_max, 1000)
            self._dbinq = self._q[1] - self._q[0]
            self._midq = (self._q[:-1] + self._q[1:]) / 2
            self._qd_vcv = np.zeros(len(self._q))
            self._qd_vvv = np.zeros(len(self._q))

    def analyze(self, frame: Frame) -> None:
        self._initialize_arrays()
        self._atoms_data = frame.nodes_data.wrapped_positions
        self.central_nodes = frame.get_nodes_by_element(self.central_species)
        lattice = frame.get_lattice()
        N = len(self.central_nodes)
        self.tetrahedricity = {}
        self.distribution_dcv = {}
        self.distribution_dvv = {}
        self.distribution_avcv = {}
        self.distribution_avvv = {}
        self.distribution_acvc = {}
        self.distribution_q = {}

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

            # Explicitly exclude the central atom from the vertex set, so that
            # dvv / avvv (and vv / vvv irregularity metrics) only ever involve
            # the 4 vertices — even when central_species == vertices_species or
            # the neighbor list ever contained self.
            vertex_neighbors = [
                n for n in node.neighbors
                if n.symbol == self.vertices_species and n.node_id != node.node_id
            ]
            if len(vertex_neighbors) != 4:
                node.form = str(node.coordination)
                continue
            pos_batch = np.array([n.position for n in vertex_neighbors])
            assert not np.any(np.all(pos_batch == node.position, axis=1)), (
                "Central atom leaked into vertex pos_batch"
            )

            # vertex-vertex distances
            distances = calculate_pbc_dot_distances_combinations(pos_batch, lattice)
            distances.sort()
            # vertex-center-vertex angles
            angles_vcv = calculate_pbc_cv_angle_combinations(
                node.position, pos_batch, lattice
            )
            angles_vcv.sort()
            # vertex-vertex-vertex angles
            angles_vvv = calculate_pbc_angle_combinations(pos_batch, lattice)
            angles_vvv.sort()

            tetra_vv = calculate_tetrahedricity(distances)
            tetra_vcv = calculate_tetrahedricity_angles(angles_vcv, self.ideal_vcv_angle)
            tetra_vvv = calculate_tetrahedricity_angles(
                angles_vvv, self.ideal_vvv_angle
            )
            # TODO: check len distances, angles 

            bin_vv = int(tetra_vv / self._dbin) + 1
            bin_vcv = int(tetra_vcv / self._dbin) + 1
            bin_vvv = int(tetra_vvv / self._dbin) + 1

            if bin_vv >= max_bin:
                continue

            self.counts["4_fold"] += 1
            self._hist_vv[bin_vv] += 1
            if bin_vcv < max_bin:
                self._hist_vcv[bin_vcv] += 1
            if bin_vvv < max_bin:
                self._hist_vvv[bin_vvv] += 1

            node.form = "4"
            node.polyhedricity = tetra_vv

            # Errington-Debenedetti order parameter q (1 = perfect tetrahedron):
            # q_vcv from the O-Si-O apex angles, q_vvv from the O-O-O angles.
            q_vcv = calculate_errington_q(angles_vcv, self.ideal_vcv_angle)
            q_vvv = calculate_errington_q(angles_vvv, self.ideal_vvv_angle)
            iq_vcv = int((q_vcv - self.q_min) / self._dbinq)
            if 0 <= iq_vcv < len(self._q):
                self._qd_vcv[iq_vcv] += 1
            iq_vvv = int((q_vvv - self.q_min) / self._dbinq)
            if 0 <= iq_vvv < len(self._q):
                self._qd_vvv[iq_vvv] += 1

            if self.calculate_distribution:
                distances_dcv = calculate_pbc_cv_distances_batch(
                    node.position, pos_batch, lattice
                )
                distances_dcv.sort()
                # Distances center-vertices
                for r in distances_dcv:
                    bin_idx = int(r / self._dbindcv) + 1
                    if bin_idx < max_bin:
                        self._rd_dcv[bin_idx] += 1
                self.counts_distribution["4_fold_dcv"] += len(distances_dcv)
                # Distances vertices-vertices
                for r in distances:
                    bin_idx = int(r / self._dbindvv) + 1
                    if bin_idx < max_bin:
                        self._rd_dvv[bin_idx] += 1
                self.counts_distribution["4_fold_dvv"] += len(distances)
                # Angles vertex-center-vertex
                for a in angles_vcv:
                    bin_idx = int(a / self._dbina) + 1
                    if bin_idx < max_bin:
                        self._ad_vcv[bin_idx] += 1
                self.counts_distribution["4_fold_avcv"] += len(angles_vcv)
                # Angles vertex-vertex-vertex
                for a in angles_vvv:
                    bin_idx = int(a / self._dbina) + 1
                    if bin_idx < max_bin:
                        self._ad_vvv[bin_idx] += 1
                self.counts_distribution["4_fold_avvv"] += len(angles_vvv)
                # Inter-tetrahedral center-vertex-center angles (bridging angle):
                # apex at each shared vertex, arms to the two central atoms it links.
                angles_cvc = []
                for vnode in vertex_neighbors:
                    for cn in vnode.neighbors:
                        if (
                            cn.symbol == self.central_species
                            and cn.node_id != node.node_id
                            and cn.coordination == 4
                        ):
                            angles_cvc.append(
                                calculate_pbc_angle(
                                    node.position,
                                    vnode.position,
                                    cn.position,
                                    lattice,
                                )
                            )
                for a in angles_cvc:
                    bin_idx = int(a / self._dbina) + 1
                    if bin_idx < max_bin:
                        self._ad_cvc[bin_idx] += 1
                self.counts_distribution["4_fold_acvc"] += len(angles_cvc)

        self.tetrahedricity["4_fold_vv"] = self._hist_vv
        self.tetrahedricity["4_fold_vcv"] = self._hist_vcv
        self.tetrahedricity["4_fold_vvv"] = self._hist_vvv
        self.distribution_dcv["4_fold_dcv"] = self._rd_dcv
        self.distribution_dvv["4_fold_dvv"] = self._rd_dvv
        self.distribution_avcv["4_fold_avcv"] = self._ad_vcv
        self.distribution_avvv["4_fold_avvv"] = self._ad_vvv
        self.distribution_acvc["4_fold_acvc"] = self._ad_cvc
        self.distribution_q["4_fold_q_vcv"] = self._qd_vcv
        self.distribution_q["4_fold_q_vvv"] = self._qd_vvv

        self.tetra_data.append(dict(self.tetrahedricity))
        self.dist_dcv_data.append(dict(self.distribution_dcv))
        self.dist_dvv_data.append(dict(self.distribution_dvv))
        self.dist_avcv_data.append(dict(self.distribution_avcv))
        self.dist_avvv_data.append(dict(self.distribution_avvv))
        self.dist_acvc_data.append(dict(self.distribution_acvc))
        self.dist_q_data.append(dict(self.distribution_q))
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

        # Errington q distributions are a core output, independent of the
        # optional distance/angle distributions.
        self._finalize_distribution(
            "distribution_q", self.dist_q_data, self._q, self._dbinq
        )

        if self.calculate_distribution:
            self._finalize_distribution(
                "distribution_dcv", self.dist_dcv_data, self._rdcv, self._dbindcv
            )
            self._finalize_distribution(
                "distribution_dvv", self.dist_dvv_data, self._rdvv, self._dbindvv
            )
            self._finalize_distribution(
                "distribution_avcv", self.dist_avcv_data, self._a, self._dbina
            )
            self._finalize_distribution(
                "distribution_avvv", self.dist_avvv_data, self._a, self._dbina
            )
            self._finalize_distribution(
                "distribution_acvc", self.dist_acvc_data, self._a, self._dbina
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

        self._save_distribution(
            self.distribution_q, "tetrahedricity_errington_q.dat"
        )

        if self.calculate_distribution:
            self._save_distribution(
                self.distribution_dcv, "tetrahedricity_distribution_dcv.dat"
            )
            self._save_distribution(
                self.distribution_dvv, "tetrahedricity_distribution_dvv.dat"
            )
            self._save_distribution(
                self.distribution_avcv, "tetrahedricity_distribution_avcv.dat"
            )
            self._save_distribution(
                self.distribution_avvv, "tetrahedricity_distribution_avvv.dat"
            )
            self._save_distribution(
                self.distribution_acvc, "tetrahedricity_distribution_acvc.dat"
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
