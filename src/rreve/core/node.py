import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from ..utils.geometry import wrap_position

# Atomic data is now centralized in the Node's module
CHEMICAL_SYMBOLS = np.array(
    [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]
)

ATOMIC_NUMBERS = {symbol: Z for Z, symbol in enumerate(CHEMICAL_SYMBOLS)}

ATOMIC_MASSES = np.array(
    [
        1.008,
        4.002602,
        6.94,
        9.0121831,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998403163,
        20.1797,
        22.98976928,
        24.305,
        26.9815385,
        28.085,
        30.973761998,
        32.06,
        35.45,
        39.948,
        39.0983,
        40.078,
        44.955908,
        47.867,
        50.9415,
        51.9961,
        54.938044,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.630,
        74.921595,
        78.971,
        79.904,
        83.798,
        85.4678,
        87.62,
        88.90584,
        91.224,
        92.90637,
        95.95,
        97.90721,
        101.07,
        102.90550,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.710,
        121.760,
        127.60,
        126.90447,
        131.293,
        132.90545196,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        144.91276,
        150.36,
        151.964,
        157.25,
        158.92535,
        162.500,
        164.93033,
        167.259,
        168.93422,
        173.054,
        174.9668,
        178.49,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.966569,
        200.592,
        204.38,
        207.2,
        208.98040,
        208.98243,
        209.98715,
        222.01758,
        223.01974,
        226.02541,
        227.02775,
        232.0377,
        231.03588,
        238.02891,
        237.04817,
        244.06421,
        243.06138,
        247.07035,
        247.07031,
        251.07959,
        252.0830,
        257.09511,
        258.09843,
        259.1010,
        262.110,
        267.122,
        268.126,
        271.134,
        270.133,
        269.1338,
        278.156,
        281.165,
        281.166,
        285.177,
        286.182,
        289.190,
        289.194,
        293.204,
        293.208,
        294.214,
    ]
)

CORRELATION_LENGTHS = np.array(
    [
        -3.7390,
        np.nan,
        -1.90,
        7.79,
        5.30,
        6.6460,
        9.36,
        5.803,
        5.654,
        4.566,
        3.63,
        5.375,
        3.449,
        4.149,
        5.13,
        2.847,
        9.5770,
        1.909,
        3.67,
        4.70,
        12.29,
        -3.438,
        -0.3824,
        3.635,
        -3.73,
        9.45,
        2.49,
        10.3,
        7.718,
        5.680,
        7.288,
        8.185,
        6.58,
        7.970,
        6.795,
        7.81,
        7.09,
        7.02,
        7.75,
        7.16,
        7.054,
        6.715,
        np.nan,
        7.03,
        5.88,
        5.91,
        5.922,
        4.87,
        4.065,
        6.225,
        5.57,
        5.80,
        5.28,
        4.92,
        5.42,
        5.07,
        8.24,
        4.84,
        4.58,
        7.69,
        np.nan,
        0.80,
        7.22,
        6.5,
        7.38,
        16.9,
        8.01,
        7.79,
        7.07,
        12.43,
        7.21,
        7.7,
        6.91,
        4.86,
        9.2,
        10.7,
        10.6,
        9.60,
        7.63,
        12.692,
        12.692,
        9.405,
        8.532,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        10.31,
        9.1,
        8.417,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)


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
                    atomic_number = ATOMIC_NUMBERS[self.symbol]
                    mass = ATOMIC_MASSES[atomic_number]
                    correlation_length = CORRELATION_LENGTHS[atomic_number]
                    self.mass = mass
                    self._atomic_mass_cache[self.symbol] = mass
                    self.correlation_length = correlation_length
                    self._correlation_length_cache[self.symbol] = correlation_length
                except KeyError:
                    # Handle cases where the symbol is not found
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

    def __str__(self) -> str:
        return f"Node {self.node_id} ({self.symbol}) | coordination: {self.coordination} | neighbors: {len(self.neighbors)} | position: {self.position}"

    def __repr__(self) -> str:
        return self.__str__()
