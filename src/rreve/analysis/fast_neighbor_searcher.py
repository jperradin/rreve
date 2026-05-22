"""
Optimized neighbor searcher using a precomputed cutoff lookup table and a
vectorized Numba PBC-distance kernel.

Drop-in replacement for ``NeighborSearcher``. Same outputs: populates
``Node.neighbors``, ``Node.distances``, ``Node.indices`` and the coordination
number / ovito selection string.

Improvements vs ``NeighborSearcher``:
  * Single ``inv_lattice`` computed once per frame (was: one matrix inverse
    per pair inside ``calculate_pbc_distance``).
  * Symbol -> int_id map + 2D ``rcut2_table`` so per-pair cutoff lookup is an
    array index instead of a dict scan over Cutoff objects.
  * Per-node candidates filtered through one Numba batch call rather than
    Python-per-pair calls.
  * Squared-distance comparison; ``sqrt`` only for kept neighbors.
"""

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

from ..core.node import Node
from ..core.frame import Frame
from ..config.settings import Settings
from ..utils.geometry import (
    cartesian_to_fractional,
    filter_neighbors_pbc_batch,
    filter_neighbors_direct_batch,
)


class FastNeighborSearcher:
    """Optimized drop-in replacement for ``NeighborSearcher``."""

    def __init__(self, frame: Frame, settings: Settings) -> None:
        self.frame: Frame = frame
        self.settings: Settings = settings
        self._nodes: List[Node] = frame.nodes
        self._lattice: np.ndarray = frame.lattice
        self._max_cutoff: float = self.settings.get_max_cutoff()

    def execute(self) -> None:
        if self.settings.wrap_position:
            positions = self.frame.get_wrapped_positions()
        else:
            positions = self.frame.get_positions()
        N = positions.shape[0]

        lattice = np.ascontiguousarray(self._lattice, dtype=np.float64)
        inv_lattice = np.ascontiguousarray(np.linalg.inv(lattice), dtype=np.float64)

        # KD-tree (broad-phase)
        if self.settings.apply_pbc:
            positions_frac = cartesian_to_fractional(positions, lattice)
            kdtree = cKDTree(positions_frac, boxsize=[1.0, 1.0, 1.0])
            query_positions = positions_frac
            search_radius = self._max_cutoff / np.linalg.norm(lattice, axis=0).max()
        else:
            kdtree = cKDTree(positions)
            query_positions = positions
            search_radius = self._max_cutoff

        # Per-pair squared cutoff lookup table
        sym_to_id, rcut2_table = self._build_cutoff_table()

        sym_ids = np.empty(N, dtype=np.int64)
        for i, n in enumerate(self._nodes):
            sym_ids[i] = sym_to_id.get(n.symbol, -1)

        cart_positions = np.ascontiguousarray(positions, dtype=np.float64)

        # Bulk broad-phase query (parallel where supported)
        try:
            all_candidates = kdtree.query_ball_point(
                query_positions, search_radius, workers=-1
            )
        except TypeError:
            all_candidates = kdtree.query_ball_point(query_positions, search_radius)

        progress_bar_kwargs = {
            "disable": not self.settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "green",
        }
        progress_bar = tqdm(
            range(N),
            desc="Fetching nearest neighbors (fast) ...",
            **progress_bar_kwargs,
        )

        apply_pbc = self.settings.apply_pbc

        for i in progress_bar:
            node = self._nodes[i]
            cand_idx = all_candidates[i]
            si = sym_ids[i]

            if not cand_idx or si < 0:
                self._assign_empty(node)
                self._calculate_coordination(node)
                continue

            # Drop self-interaction
            cand_arr = np.fromiter((j for j in cand_idx if j != i), dtype=np.int64)
            if cand_arr.size == 0:
                self._assign_empty(node)
                self._calculate_coordination(node)
                continue

            cand_syms = sym_ids[cand_arr]
            rcut2 = rcut2_table[si, cand_syms]  # negative entries = invalid pair
            cand_pos = cart_positions[cand_arr]

            if apply_pbc:
                keep, dists = filter_neighbors_pbc_batch(
                    cart_positions[i], cand_pos, lattice, inv_lattice, rcut2
                )
            else:
                keep, dists = filter_neighbors_direct_batch(
                    cart_positions[i], cand_pos, rcut2
                )

            kept_global = cand_arr[keep]
            kept_dists = dists[keep]

            node.neighbors = [self._nodes[j] for j in kept_global]
            node.distances = kept_dists.tolist()
            node.indices = [self._nodes[j].node_id for j in kept_global]
            node._ovito_selection_str = (
                "".join(f"ParticleIndex=={n.node_id}||" for n in node.neighbors)
                + "ParticleIndex=="
                + str(node.node_id)
            )

            self._calculate_coordination(node)

    @staticmethod
    def _assign_empty(node: Node) -> None:
        node.neighbors = []
        node.distances = []
        node.indices = []
        node._ovito_selection_str = "ParticleIndex==" + str(node.node_id)

    def _build_cutoff_table(self) -> Tuple[Dict[str, int], np.ndarray]:
        """Build (symbol -> int_id) map and a 2D squared-cutoff table.

        Returns:
            sym_to_id (Dict[str, int])
            rcut2_table (np.ndarray): (S, S) float64. Entries are squared cutoff
                distances; ``-1.0`` flags an invalid pair.
        """
        symbols: List[str] = []
        seen = set()
        for c in self.settings.cutoffs:
            for s in (c.type1, c.type2):
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)

        sym_to_id: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        S = len(symbols)
        table = np.full((S, S), -1.0, dtype=np.float64)
        for c in self.settings.cutoffs:
            i = sym_to_id[c.type1]
            j = sym_to_id[c.type2]
            d2 = float(c.distance) * float(c.distance)
            table[i, j] = d2
            table[j, i] = d2  # symmetric
        return sym_to_id, table

    def _calculate_coordination(self, node: Node) -> None:
        """Calculate the coordination number based on the configured mode."""
        mode = self.settings.coordination_mode

        if mode == "all_types":
            node.set_coordination(len(node.neighbors))
        elif mode == "same_type":
            node.set_coordination(
                len([n for n in node.neighbors if n.symbol == node.symbol])
            )
        elif mode == "different_type":
            node.set_coordination(
                len([n for n in node.neighbors if n.symbol != node.symbol])
            )
        else:
            node.set_coordination(len([n for n in node.neighbors if n.symbol == mode]))
