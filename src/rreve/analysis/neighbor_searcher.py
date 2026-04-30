import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
from typing import List

from ..core.node import Node
from ..core.frame import Frame
from ..config.settings import Settings
from ..utils.geometry import calculate_pbc_distance, cartesian_to_fractional


class NeighborSearcher:
    """
    A component responsible for finding and filtering neighbors for all nodes
    in a frame using a k-d tree algorithm
    """

    def __init__(self, frame: Frame, settings: Settings) -> None:
        """Initializes the NeighborSearcher"""
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

        # Build the k-d tree, handling periodic boundary conditions
        if self.settings.apply_pbc:
            positions_frac = cartesian_to_fractional(positions, self._lattice)
            kdtree = cKDTree(positions_frac, boxsize=[1, 1, 1])
            query_positions = positions_frac

            search_radius = (
                self._max_cutoff / np.linalg.norm(self._lattice, axis=0).max()
            )

        else:
            kdtree = cKDTree(positions)
            query_positions = positions
            search_radius = self._max_cutoff

        progress_bar_kwargs = {
            "disable": not self.settings.verbose,
            "leave": False,
            "ncols": os.get_terminal_size().columns,
            "colour": "green",
        }

        progress_bar = tqdm(
            range(len(self._nodes)),
            desc="Fetching nearest neighbors ...",
            **progress_bar_kwargs,
        )

        for i in progress_bar:
            node = self._nodes[i]

            # find candidate neighbors within the max cutoff radius
            indices = kdtree.query_ball_point(query_positions[i], search_radius)

            # refine neighbors with exact distance checks
            self._filter_and_assign_neighbors(node, indices)

            # calculate coordination number
            self._calculate_coordination(node)

    def _filter_and_assign_neighbors(
        self, node: Node, candidate_indices: List[int]
    ) -> None:
        """
        Filters candidate neighbors based on exact cutoffs and assigns them to the node.
        """
        new_neighbors = []
        new_distances = []

        node_pos = node.position
        node._ovito_selection_str = ""

        for neighbor_idx in candidate_indices:
            neighbor = self._nodes[neighbor_idx]

            # Skip self interaction
            if node.node_id == neighbor.node_id:
                continue

            # Check exact cutoff distance for this pair of node types
            rcut = self.settings.get_cutoff(node.symbol, neighbor.symbol)
            if rcut is None:
                continue

            # Calculate distance (PBC or direct)
            if self.settings.apply_pbc:
                dist = calculate_pbc_distance(
                    node_pos, neighbor.position, self._lattice
                )
            else:
                dist = np.linalg.norm(node_pos - neighbor.position)

            if dist <= rcut:
                new_neighbors.append(neighbor)
                new_distances.append(dist)
                node._ovito_selection_str += f"ParticleIndex=={neighbor.node_id}||"

        node.neighbors = new_neighbors
        node.distances = new_distances
        node.indices = [n.node_id for n in new_neighbors]
        node._ovito_selection_str += "ParticleIndex==" + str(node.node_id)

    def _calculate_coordination(self, node: Node) -> None:
        """
        Calculate the coordination number based on different modes:
            - all_types         : all atoms are considered
            - same_types        : every atoms of the same type are considered
            - different_type    : every atoms of different types are considered
            - <node_type>       : atoms of the specified node type are considered
        """

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
