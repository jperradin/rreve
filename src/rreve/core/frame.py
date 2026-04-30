import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from numba_progress import ProgressBar

from .node import Node
from ..utils.geometry import wrap_positions
from ..config.settings import Settings


@dataclass(slots=True)
class NodesData:
    """
    Holds node-related data for a frame.
    """

    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    wrapped_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    correlation_lengths: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Frame:
    """
    Reprensation of a frame of a trajectory

    Attributes:
    -----------
    frame_id : int
        Id of the frame
    nodes : List[Node]
        List of nodes in the frame
    lattice : np.ndarray
        Lattice of the frame
    nodes_data : NodesData
        Dataclass holding node data for the frame
    _data : Dict[str, np.ndarray]
        Internal data structure for node data (symbol, position)
    """

    frame_id: int
    nodes: List[Node]
    lattice: np.ndarray
    _lattice_str: str
    nodes_data: NodesData = field(default_factory=NodesData)
    _data: Dict[str, np.ndarray] | None = None
    _settings: Settings = field(default_factory=Settings)

    def __post_init__(self):
        """Initialisation after object creation"""
        if not isinstance(self.nodes, list):
            raise TypeError("nodes must be a list of Nodes")
        if self.lattice is not None and not isinstance(self.lattice, np.ndarray):
            raise TypeError("lattice must be a numpy array")

    def initialize_nodes(self) -> None:
        """Initialize the list of nodes in the frame"""
        id = 0
        symbols = self._data["symbol"]
        positions = self._data["position"]

        if len(symbols) != len(positions):
            raise ValueError("symbols and positions must have the same length")

        for symbol, position in zip(symbols, positions):
            self.nodes.append(Node(node_id=id, symbol=symbol, position=position))
            id += 1

        self._initalize_nodes_data()

    def _initalize_nodes_data(self) -> None:
        """Initializes the NodesData dataclass."""
        self.nodes_data.positions = self.get_positions_by_element()
        self.nodes_data.wrapped_positions = self.get_wrapped_positions_by_element()
        self.nodes_data.correlation_lengths = self.get_correlation_lengths()

    def set_lattice(self, lattice: np.ndarray) -> None:
        """Set the lattice of the frame"""
        if lattice.shape != (3, 3):
            raise ValueError("lattice must be a 3x3 numpy array")

        try:
            np.linalg.inv(lattice)
        except np.linalg.LinAlgError:
            raise ValueError("lattice must be a non-singular matrix")

        self.lattice = lattice

    def get_lattice(self) -> np.ndarray:
        """Get the lattice of the frame"""
        return self.lattice

    def get_unique_elements(self) -> np.ndarray:
        """Get the unique elements in the frame"""
        return np.unique([node.symbol for node in self.nodes])

    def get_correlation_lengths(self) -> Dict:
        """Get the correlation lengths of the species present in the system"""
        crl = {}
        for node in self.nodes:
            if node.symbol not in crl:
                crl[node.symbol] = node.correlation_length
        return crl

    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """Get an node by its id"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes_by_element(self, element: str) -> List[Node]:
        """Get nodes by their element"""
        nodes = [node for node in self.nodes if node.symbol == element]
        return nodes

    def get_positions(self) -> np.ndarray:
        """Get the positions of all nodes in the frame"""
        return np.array([node.position for node in self.nodes])

    def get_positions_by_element(self) -> Dict[str, np.ndarray]:
        """Get the positions of all nodes in the frame grouped by element"""
        wp = {}
        for species in self.get_unique_elements():
            if species not in wp:
                wp[species] = np.array(
                    [node.position for node in self.nodes if node.symbol == species]
                )
        return wp

    def get_wrapped_positions(self) -> np.ndarray:
        """Get the wrapped positions of all nodes in the frame"""
        with ProgressBar(
            total=len(self.get_positions()),
            disable=not self._settings.verbose,
            leave=False,
            colour="#eaeaaa",
            unit="atom",
            desc="Unwrapping positions",
        ) as progress:
            return wrap_positions(self.get_positions(), self.lattice, progress)

    def get_wrapped_positions_by_element(self) -> Dict[str, np.ndarray]:
        """Get the wrapped positions of all nodes in the frame grouped by element"""
        wrapped_positions = {}
        positions_by_element = self.get_positions_by_element()
        for species, positions in positions_by_element.items():
            with ProgressBar(
                total=len(positions),
                disable=not self._settings.verbose,
                leave=False,
                colour="#eaeaaa",
                unit="atom",
                desc=f"Wrapping {species} positions",
            ) as progress:
                wrapped_positions[species] = wrap_positions(
                    positions, self.lattice, progress
                )

        del positions, positions_by_element
        return wrapped_positions

    def get_nodes(self) -> List[Node]:
        """Get the nodes of the frame"""
        return self.nodes

    def get_num_nodes(self) -> int:
        """Get the number of nodes"""
        return len(self.nodes)

    def __len__(self) -> int:
        """Get the number of nodes in the frame"""
        return len(self.nodes)

    def __str__(self) -> str:
        return f"Frame {self.frame_id} (num_nodes={len(self.nodes)})"

    def __repr__(self) -> str:
        return f"Frame {self.frame_id} (num_nodes={len(self.nodes)})\n(first node: {(self.nodes[0].symbol, self.nodes[0].position) if len(self.nodes) > 0 else ''}\n(lattice=\n{self.lattice})\n"

    def __del__(self) -> None:
        del self.nodes
        del self.lattice
        del self._data
