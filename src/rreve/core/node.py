import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from ..utils.geometry import wrap_position, calculate_pbc_angle

# symbol -> (atomic_mass [u], neutron_scattering_length [fm])
NAN = float("nan")
ATOMIC_DATA: dict[str, tuple[float, float]] = {
    # fmt: off
    #  sym     mass             b (fm)
    "H":   (  1.008,           -3.7390),
    "He":  (  4.002602,         NAN   ),
    "Li":  (  6.94,            -1.90  ),
    "Be":  (  9.0121831,        7.79  ),
    "B":   ( 10.81,             5.30  ),
    "C":   ( 12.011,            6.6460),
    "N":   ( 14.007,            9.36  ),
    "O":   ( 15.999,            5.803 ),
    "F":   ( 18.998403163,      5.654 ),
    "Ne":  ( 20.1797,           4.566 ),
    "Na":  ( 22.98976928,       3.63  ),
    "Mg":  ( 24.305,            5.375 ),
    "Al":  ( 26.9815385,        3.449 ),
    "Si":  ( 28.085,            4.149 ),
    "P":   ( 30.973761998,      5.13  ),
    "S":   ( 32.06,             2.847 ),
    "Cl":  ( 35.45,             9.5770),
    "Ar":  ( 39.948,            1.909 ),
    "K":   ( 39.0983,           3.67  ),
    "Ca":  ( 40.078,            4.70  ),
    "Sc":  ( 44.955908,        12.29  ),
    "Ti":  ( 47.867,           -3.438 ),
    "V":   ( 50.9415,          -0.3824),
    "Cr":  ( 51.9961,           3.635 ),
    "Mn":  ( 54.938044,        -3.73  ),
    "Fe":  ( 55.845,            9.45  ),
    "Co":  ( 58.933194,         2.49  ),
    "Ni":  ( 58.6934,          10.3   ),
    "Cu":  ( 63.546,            7.718 ),
    "Zn":  ( 65.38,             5.680 ),
    "Ga":  ( 69.723,            7.288 ),
    "Ge":  ( 72.630,            8.185 ),
    "As":  ( 74.921595,         6.58  ),
    "Se":  ( 78.971,            7.970 ),
    "Br":  ( 79.904,            6.795 ),
    "Kr":  ( 83.798,            7.81  ),
    "Rb":  ( 85.4678,           7.09  ),
    "Sr":  ( 87.62,             7.02  ),
    "Y":   ( 88.90584,          7.75  ),
    "Zr":  ( 91.224,            7.16  ),
    "Nb":  ( 92.90637,          7.054 ),
    "Mo":  ( 95.95,             6.715 ),
    "Tc":  ( 97.90721,          NAN   ),
    "Ru":  (101.07,             7.03  ),
    "Rh":  (102.90550,          5.88  ),
    "Pd":  (106.42,             5.91  ),
    "Ag":  (107.8682,           5.922 ),
    "Cd":  (112.414,            4.87  ),
    "In":  (114.818,            4.065 ),
    "Sn":  (118.710,            6.225 ),
    "Sb":  (121.760,            5.57  ),
    "Te":  (127.60,             5.80  ),
    "I":   (126.90447,          5.28  ),
    "Xe":  (131.293,            4.92  ),
    "Cs":  (132.90545196,       5.42  ),
    "Ba":  (137.327,            5.07  ),
    "La":  (138.90547,          8.24  ),
    "Ce":  (140.116,            4.84  ),
    "Pr":  (140.90766,          4.58  ),
    "Nd":  (144.242,            7.69  ),
    "Pm":  (144.91276,          NAN   ),
    "Sm":  (150.36,             0.80  ),
    "Eu":  (151.964,            7.22  ),
    "Gd":  (157.25,             6.5   ),
    "Tb":  (158.92535,          7.38  ),
    "Dy":  (162.500,           16.9   ),
    "Ho":  (164.93033,          8.01  ),
    "Er":  (167.259,            7.79  ),
    "Tm":  (168.93422,          7.07  ),
    "Yb":  (173.054,           12.43  ),
    "Lu":  (174.9668,           7.21  ),
    "Hf":  (178.49,             7.7   ),
    "Ta":  (180.94788,          6.91  ),
    "W":   (183.84,             4.86  ),
    "Re":  (186.207,            9.2   ),
    "Os":  (190.23,            10.7   ),
    "Ir":  (192.217,           10.6   ),
    "Pt":  (195.084,            9.60  ),
    "Au":  (196.966569,         7.63  ),
    "Hg":  (200.592,           12.692 ),
    "Tl":  (204.38,            12.692 ),
    "Pb":  (207.2,              9.405 ),
    "Bi":  (208.98040,          8.532 ),
    "Po":  (208.98243,          NAN   ),
    "At":  (209.98715,          NAN   ),
    "Rn":  (222.01758,          NAN   ),
    "Fr":  (223.01974,          NAN   ),
    "Ra":  (226.02541,          NAN   ),
    "Ac":  (227.02775,          NAN   ),
    "Th":  (232.0377,          10.31  ),
    "Pa":  (231.03588,          9.1   ),
    "U":   (238.02891,          8.417 ),
    "Np":  (237.04817,          NAN   ),
    "Pu":  (244.06421,          NAN   ),
    "Am":  (243.06138,          NAN   ),
    "Cm":  (247.07035,          NAN   ),
    "Bk":  (247.07031,          NAN   ),
    "Cf":  (251.07959,          NAN   ),
    "Es":  (252.0830,           NAN   ),
    "Fm":  (257.09511,          NAN   ),
    "Md":  (258.09843,          NAN   ),
    "No":  (259.1010,           NAN   ),
    "Lr":  (262.110,            NAN   ),
    "Rf":  (267.122,            NAN   ),
    "Db":  (268.126,            NAN   ),
    "Sg":  (271.134,            NAN   ),
    "Bh":  (270.133,            NAN   ),
    "Hs":  (269.1338,           NAN   ),
    "Mt":  (278.156,            NAN   ),
    "Ds":  (281.165,            NAN   ),
    "Rg":  (281.166,            NAN   ),
    "Cn":  (285.177,            NAN   ),
    "Nh":  (286.182,            NAN   ),
    "Fl":  (289.190,            NAN   ),
    "Mc":  (289.194,            NAN   ),
    "Lv":  (293.204,            NAN   ),
    "Ts":  (293.208,            NAN   ),
    "Og":  (294.214,            NAN   ),
    # fmt: on
}


@dataclass(slots=True, order=True)
class Node:
    """Reprensation of a node

    Attributes:
    -----------
    symbol : str
        Symbol of the node
    node_id : int
        Id of the node (unique identifier, autoincremented)
    position : np.ndarray
        Position of the node
    parent : Optional['Node']
        Parent of the node (Optional)
    neighbors : List['Node']
        List of neighbors of the node (Optional)
    angles : Optional[List[float]]
        Angles with second neighbors of the node (Optional)
    distances : Optional[List[float]]
        Distances of the neighbors of the node (Optional)
    indices : Optional[List[int]]
        Indices of the neighbors of the node (Optional)
    mass : float
        Mass of the node (Optional)
    coordination : Optional[int]
        Coordination number of the node (Optional)
    other : Optional[List[str]]
        Other attributes of the node (Optional)
    """

    symbol: str
    node_id: int
    position: np.ndarray = field(compare=False, repr=False)
    neighbors: List["Node"] = field(default_factory=list, compare=False, repr=False)
    angles: Optional[List[float]] = field(default=None, compare=False, repr=False)
    distances: Optional[List[float]] = field(default=None, compare=False, repr=False)
    indices: Optional[List[int]] = field(default=None, compare=False, repr=False)
    mass: Optional[float] = field(default=None, compare=True, repr=True)
    correlation_length: Optional[float] = field(default=None, compare=True, repr=True)
    coordination: Optional[int] = field(default=None, compare=True, repr=True)
    polyhedricity: Optional[int] = field(default=-1, compare=True, repr=True)
    form: Optional[str] = field(default=None, compare=False, repr=False)
    _ovito_selection_str: Optional[str] = field(default=None, compare=False, repr=False)

    _next_id = 0
    _atomic_mass_cache = {}  # Cache for atomic masses
    _correlation_length_cache = {}  # Cache for correlation lengths

    def __post_init__(self):
        """Initialisation after object creation"""

        if self.position is None:
            object.__setattr__(self, "position", np.zeros(3))

        if self.node_id is None:
            object.__setattr__(self, "node_id", Node._next_id)
            Node._next_id += 1

        # Lazy load the mass
        if self.mass is None:
            if self.symbol in self._atomic_mass_cache:
                self.mass = self._atomic_mass_cache[self.symbol]
                self.correlation_length = self._correlation_length_cache[self.symbol]
            else:
                try:
                    mass, correlation_length = ATOMIC_DATA[self.symbol]
                    self.mass = mass
                    self._atomic_mass_cache[self.symbol] = mass
                    self.correlation_length = correlation_length
                    self._correlation_length_cache[self.symbol] = correlation_length
                except KeyError:
                    self.mass = 0.0
                    self.correlation_length = 0.0

        if self.coordination is None:
            object.__setattr__(self, "coordination", 0)

        if self.form is None:
            object.__setattr__(self, "form", "")

        if self.neighbors is None:
            object.__setattr__(self, "neighbors", [])

    @staticmethod
    def wrap_position(position: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """Wrap position in a periodic box defined by the lattice"""
        return wrap_position(position, lattice)

    def add_neighbor(self, node: "Node") -> None:
        """Add a node as a neighbor"""
        self.neighbors.append(node)

    def set_coordination(self, coordination: int) -> None:
        self.coordination = coordination

    def set_polyhedricity(self, polyhedricity: float) -> None:
        self.polyhedricity = polyhedricity

    def get_neighbors_positions_by_element(self, element: str) -> np.ndarray:
        n_positions = np.array(
            [n.position for n in self.neighbors if n.symbol == element]
        )
        return n_positions

    def calculate_angles(self, lattice) -> None:
        angles = []
        for n in self.neighbors:
            for nn in n.neighbors:
                if n.symbol == 'O' and nn.symbol == 'Si' and nn.node_id != self.node_id:
                    angle = calculate_pbc_angle(self.position, n.position, nn.position, lattice)
                    angles.append(angle)
        self.angles = np.array(angles)

    def __str__(self) -> str:
        return f"Node {self.node_id} ({self.symbol}) | coordination: {self.coordination} | neighbors: {len(self.neighbors)} | position: {self.position}"

    def __repr__(self) -> str:
        return self.__str__()
