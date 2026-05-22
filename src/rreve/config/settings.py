import os
from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Tuple, Optional, List, Set

@dataclass
class PDFAnalysisSettings:
    """Settings specific to the PairDistributionFunctionAnalyzer"""

    r_max: float = 10.0
    bins: int = 800
    pairs_to_calculate: List[str] = field(default_factory=lambda: [""])

    def __str__(self) -> str:
        line = "\t\t  |- pdf_settings:\n"
        line += f"\t\t    |- r_max = {self.r_max}\n"
        line += f"\t\t    |- bins = {self.bins}\n"
        line += f"\t\t    |- pairs_to_calculate = \n\t\t    {self.pairs_to_calculate}"
        return line


@dataclass
class BADAnalysisSettings:
    """Settings specific to the BondAngularDistributionAnalyzer"""

    bins: int = 800
    triplets_to_calculate: List[str] = field(default_factory=lambda: [""])

    def __str__(self) -> str:
        line = "\t\t  |- bad_settings:\n"
        line += f"\t\t    |- bins = {self.bins}\n"
        line += (
            f"\t\t    |- triplets_to_calculate = \n\t\t    {self.triplets_to_calculate}"
        )
        return line


@dataclass
class SQFFTAnalysisSettings:
    """Settings specific to the NeutronStructureFactorFFTAnalyzer"""

    grid_size_max: int = 512

    def __str__(self) -> str:
        line = "\t\t  |- sqfft_settings:\n"
        line += f"\t\t    |- grid_size_max = {self.grid_size_max}"
        return line


@dataclass
class STRUNITSAnalysisSettings:
    """Settings specific to the StructuralUnitsAnalyzer"""

    units_to_calculate: List[str] = field(default_factory=lambda: [""])

    def __str__(self) -> str:
        line = "\t\t  |- strunits_settings:\n"
        line += f"\t\t    |- units_to_calculate = \n\t\t    {self.units_to_calculate}"
        return line


@dataclass
class CONNAnalysisSettings:
    """Settings specific to the ConnectivityAnalyzer"""

    networking_species: str = ""
    bridging_species: str = ""

    def __str__(self) -> str:
        line = "\t\t  |- connect_settings:\n"
        line += f"\t\t    |- networking_species = {self.networking_species}\n"
        line += f"\t\t    |- bridging_species = {self.bridging_species}"
        return line


@dataclass
class POLYAnalysisSettings:
    """Settings specific to the PolyhedricityAnalyzer"""

    central_species: str = ""
    vertices_species: str = ""
    max_c: float = 0.1
    print_forms: bool = True
    calculate_distributions: bool = True

    def __str__(self) -> str:
        line = "\t\t  |- poly_settings:\n"
        line += f"\t\t    |- central_species = {self.central_species}\n"
        line += f"\t\t    |- vertices_species = {self.vertices_species}\n"
        line += f"\t\t    |- max_c = {self.max_c}\n"
        line += f"\t\t    |- print_forms = {self.print_forms}\n"
        line += f"\t\t    |- calculate_distributions = {self.calculate_distributions}"
        return line


@dataclass
class TETRAAnalysisSettings:
    """Settings specific to the TetrahedricityAnalyzer"""

    central_species: str = ""
    vertices_species: str = ""
    max_c: float = 0.1
    ideal_cv_angle: float = 109.47
    ideal_vvv_angle: float = 60.0
    print_forms: bool = True
    calculate_distributions: bool = True

    def __str__(self) -> str:
        line = "\t\t  |- tetra_settings:\n"
        line += f"\t\t    |- central_species = {self.central_species}\n"
        line += f"\t\t    |- vertices_species = {self.vertices_species}\n"
        line += f"\t\t    |- max_c = {self.max_c}\n"
        line += f"\t\t    |- ideal_cv_angle = {self.ideal_cv_angle}\n"
        line += f"\t\t    |- ideal_vvv_angle = {self.ideal_vvv_angle}\n"
        line += f"\t\t    |- print_forms = {self.print_forms}\n"
        line += f"\t\t    |- calculate_distributions = {self.calculate_distributions}"
        return line


@dataclass
class Cutoff:
    """
    Cutoff that contains all the cutoffs.

    Attributes:
    """

    type1: str
    type2: str
    distance: float

    def __str__(self) -> str:
        max_len = 5
        diff = max_len - len(self.type1) - 1 - len(self.type2)
        return f"{self.type1}-{self.type2}{' ' * diff} : distance = {self.distance}"

    def get_distance(self) -> float:
        return self.distance


@dataclass
class GeneralSettings:
    """
    General settings that contains all the general settings.

    Attributes:
    """

    # Name of the project
    project_name: str = "Project"
    # Directory to export results
    export_directory: str = "exports"
    # Path to the trajectory file
    file_location: str = ""
    # Range of frames to process (0 to -1 = all frames)
    range_of_frames: Tuple[int, int] = (0, -1)
    # Whether to apply periodic boundary conditions
    apply_pbc: bool = False
    # Whether to wrap positions systematically
    wrap_position: bool = True
    # Whether to print settings, progress bars and other information
    verbose: bool = False
    # Whether to save logs
    save_logs: bool = False
    # Whether to save performance
    save_performance: bool = False
    # Wheter to decorate inputs files with analysis outputs
    decorate_input_file: bool = False
    # Cutoffs for distance
    cutoffs: List[Cutoff] = field(default_factory=lambda: [])
    # Coordination mode
    coordination_mode: str = "all_types"
    # Whether to use the optimized fast neighbor searcher
    use_fast_neighbor_search: bool = True


@dataclass
class AnalysisSettings:
    """
    Analysis settings that contains all the analyzer settings.

    Attributes:
        overwrite: Whether to overwrite the existing file, if False, appends results to the file
        with_all: Whether to calculate all the properties
        exclude_analyzers: Set of analyzer names to exclude when with_all is True
        with_pair_distribution_function: Whether to calculate the pair distribution functions
        pdf_settings: Settings for pair distribution function analysis
        with_bond_angular_distribution: Whether to calculate the bond angular distribution
        bad_settings: Settings for bond angular distribution analysis
        with_neutron_structure_factor: Whether to calculate the neutron structure factor
        with_neutron_structure_factor_fft: Whether to calculate the neutron structure factor via fft
        sqfft_settings: Settings for neutron structure factor FFT analysis
        with_structural_units: Whether to calculate the structural units
        strunits_settings: Settings for structural units analysis
        with_connectivity: Whether to calculate the connectivities
        connect_settings: Settings for connectivity analysis
        with_polyhedricity: Whether to calculate the polyhedricity
        poly_settings: Settings for polyhedricity analysis
    """

    # Whether to overwrite the existing file, if False, appends results to the file
    overwrite: bool = True
    # Whether to calculate all the properties
    with_all: bool = False
    # Set of analyzer names to exclude when with_all is True
    exclude_analyzers: Set[str] = field(default_factory=set)
    # Whether to calculate the pair distribution functions.
    with_pair_distribution_function: bool = False
    pdf_settings: Optional["PDFAnalysisSettings"] = None
    # Whether to calculate the bond angular distribution
    with_bond_angular_distribution: bool = False
    bad_settings: Optional["BADAnalysisSettings"] = None
    # Whether to calculate the neutron structure factor
    with_neutron_structure_factor: bool = False
    # Whether to calculate the neutron structure factor via fft
    with_neutron_structure_factor_fft: bool = False
    sqfft_settings: Optional["SQFFTAnalysisSettings"] = None
    # Whether to calculate the structural units
    with_structural_units: bool = False
    strunits_settings: Optional["STRUNITSAnalysisSettings"] = None
    # Whether to calculate the conncetivities
    with_connectivity: bool = False
    connect_settings: Optional["CONNAnalysisSettings"] = None
    # Whether to calculate the polyhedricity
    with_polyhedricity: bool = False
    poly_settings: Optional["POLYAnalysisSettings"] = None
    # Whether to calculate the tetrahedricity
    with_tetrahedricity: bool = False
    tetra_settings: Optional["TETRAAnalysisSettings"] = None

    def __post_init__(self):
        """Post-initialization to handle with_all logic and exclusions."""
        if self.with_all:
            self._activate_all_analyzers()

    def _activate_all_analyzers(self):
        """Activate all analyzers except those in exclude_analyzers."""
        analyzer_mapping = {
            "pair_distribution_function": "with_pair_distribution_function",
            "bond_angular_distribution": "with_bond_angular_distribution",
            "neutron_structure_factor_fft": "with_neutron_structure_factor_fft",
            "structural_units": "with_structural_units",
            "connectivity": "with_connectivity",
            "polyhedricity": "with_polyhedricity",
            "tetrahedricity": "with_tetrahedricity",
        }

        for analyzer_key, attribute_name in analyzer_mapping.items():
            if analyzer_key not in self.exclude_analyzers:
                setattr(self, attribute_name, True)

    def is_analyzer_enabled(self, analyzer_type: str) -> bool:
        """
        Check if a specific analyzer is enabled, considering with_all and exclusions.

        Args:
            analyzer_type: Type of analyzer to check

        Returns:
            bool: True if the analyzer is enabled
        """
        analyzer_mapping = {
            "pair_distribution_function": self.with_pair_distribution_function,
            "bond_angular_distribution": self.with_bond_angular_distribution,
            "neutron_structure_factor_fft": self.with_neutron_structure_factor_fft,
            "structural_units": self.with_structural_units,
            "connectivity": self.with_connectivity,
            "polyhedricity": self.with_polyhedricity,
            "tetrahedricity": self.with_tetrahedricity,
        }

        if self.with_all:
            return analyzer_type not in self.exclude_analyzers

        return analyzer_mapping.get(analyzer_type, False)

    def exclude_analyzer(self, analyzer_name: str) -> "AnalysisSettings":
        """
        Add an analyzer to the exclusion list and deactivate it if with_all is True.

        Args:
            analyzer_name: Name of the analyzer to exclude

        Returns:
            AnalysisSettings: Self for method chaining
        """
        self.exclude_analyzers.add(analyzer_name)

        # If with_all is True, deactivate the specific analyzer
        if self.with_all:
            analyzer_mapping = {
                "pair_distribution_function": "with_pair_distribution_function",
                "bond_angular_distribution": "with_bond_angular_distribution",
                "neutron_structure_factor_fft": "with_neutron_structure_factor_fft",
                "structural_units": "with_structural_units",
                "connectivity": "with_connectivity",
                "polyhedricity": "with_polyhedricity",
                "tetrahedricity": "with_tetrahedricity",
            }

            if analyzer_name in analyzer_mapping:
                setattr(self, analyzer_mapping[analyzer_name], False)

        return self

    def include_analyzer(self, analyzer_name: str) -> "AnalysisSettings":
        """
        Remove an analyzer from the exclusion list and activate it if with_all is True.

        Args:
            analyzer_name: Name of the analyzer to include

        Returns:
            AnalysisSettings: Self for method chaining
        """
        self.exclude_analyzers.discard(analyzer_name)

        # If with_all is True, activate the specific analyzer
        if self.with_all:
            analyzer_mapping = {
                "pair_distribution_function": "with_pair_distribution_function",
                "bond_angular_distribution": "with_bond_angular_distribution",
                "neutron_structure_factor_fft": "with_neutron_structure_factor_fft",
                "structural_units": "with_structural_units",
                "connectivity": "with_connectivity",
                "polyhedricity": "with_polyhedricity",
                "tetrahedricity": "with_tetrahedricity",
            }

            if analyzer_name in analyzer_mapping:
                setattr(self, analyzer_mapping[analyzer_name], True)

        return self

    def get_analyzers(self) -> List[str]:
        """Get list of active analyzers."""
        analyzers = []

        if self.is_analyzer_enabled("pair_distribution_function"):
            analyzers.append("PairDistributionFunctionAnalyzer")

        if self.is_analyzer_enabled("bond_angular_distribution"):
            analyzers.append("BondAngularDistributionAnalyzer")

        if self.with_neutron_structure_factor:
            raise ValueError(
                "The NeutronStructureFactorAnalyzer is disabled, use NeutronStructureFactorFFTAnalyzer instead"
            )

        if self.is_analyzer_enabled("neutron_structure_factor_fft"):
            analyzers.append("NeutronStructureFactorFFTAnalyzer")

        if self.is_analyzer_enabled("structural_units"):
            analyzers.append("StructuralUnitsAnalyzer")

        if self.is_analyzer_enabled("connectivity"):
            analyzers.append("ConnectivityAnalyzer")

        if self.is_analyzer_enabled("polyhedricity"):
            analyzers.append("PolyhedricityAnalyzer")

        if self.is_analyzer_enabled("tetrahedricity"):
            analyzers.append("TetrahedricityAnalyzer")

        return analyzers

    def __str__(self) -> str:
        lines = []

        # Add exclude_analyzers info if with_all is True and there are exclusions
        if self.with_all and self.exclude_analyzers:
            lines.append(f"\t\t|- with_all: {self.with_all}")
            lines.append(f"\t\t|- exclude_analyzers: {list(self.exclude_analyzers)}")

        for key, value in self.__dict__.items():
            if key == "exclude_analyzers":
                continue
            if key == "with_all" and self.with_all and self.exclude_analyzers:
                continue

            if value is not None:
                if not self.with_all and key == "with_all":
                    continue
                if not self.is_analyzer_enabled("pair_distribution_function") and (
                    key == "with_pair_distribution_function" or key == "pdf_settings"
                ):
                    continue
                if not self.is_analyzer_enabled("bond_angular_distribution") and (
                    key == "with_bond_angular_distribution" or key == "bad_settings"
                ):
                    continue
                if (
                    not self.with_neutron_structure_factor
                    and key == "with_neutron_structure_factor"
                ):
                    continue
                if not self.is_analyzer_enabled("neutron_structure_factor_fft") and (
                    key == "with_neutron_structure_factor_fft"
                    or key == "sqfft_settings"
                ):
                    continue
                if not self.is_analyzer_enabled("structural_units") and (
                    key == "with_structural_units" or key == "strunits_settings"
                ):
                    continue
                if not self.is_analyzer_enabled("connectivity") and (
                    key == "with_connectivity" or key == "connect_settings"
                ):
                    continue
                if not self.is_analyzer_enabled("polyhedricity") and (
                    key == "with_polyhedricity" or key == "poly_settings"
                ):
                    continue
                if not self.is_analyzer_enabled("tetrahedricity") and (
                    key == "with_tetrahedricity" or key == "tetra_settings"
                ):
                    continue
                if (
                    self.is_analyzer_enabled("pair_distribution_function")
                ) and key == "pdf_settings":
                    lines.append(str(self.pdf_settings))
                    continue
                if (
                    self.is_analyzer_enabled("bond_angular_distribution")
                ) and key == "bad_settings":
                    lines.append(str(self.bad_settings))
                    continue
                if (
                    self.is_analyzer_enabled("neutron_structure_factor_fft")
                ) and key == "sqfft_settings":
                    lines.append(str(self.sqfft_settings))
                    continue
                if (
                    self.is_analyzer_enabled("structural_units")
                ) and key == "strunits_settings":
                    lines.append(str(self.strunits_settings))
                    continue
                if (
                    self.is_analyzer_enabled("connectivity")
                ) and key == "connect_settings":
                    lines.append(str(self.connect_settings))
                    continue
                if (
                    self.is_analyzer_enabled("polyhedricity")
                ) and key == "poly_settings":
                    lines.append(str(self.poly_settings))
                    continue
                if (
                    self.is_analyzer_enabled("tetrahedricity")
                ) and key == "tetra_settings":
                    lines.append(str(self.tetra_settings))
                    continue

                lines.append(f"\t\t|- {key}: {value}")

        output = """
        Analysis Settings:
        -----------------
{}
        """.format("\n".join(lines))
        return output


@dataclass
class LatticeSettings:
    """
    Lattice settings.

    TODO implement lattice fetcher from file
         implement the handling of lattice settings in the system

    Attributes:
        lattice_in_trajectory_file (bool): Whether the lattice is present in the trajectory file.
        lattice (np.ndarray): The lattice matrix.
        get_lattice_from_file (bool): Whether to get the lattice from a file.
        lattice_file_location (str): Location of the lattice file.
        apply_lattice_to_all_frames (bool): Whether to apply the lattice to all frames.
        apply_pbc (bool): Whether to apply periodic boundary conditions.
    """

    apply_custom_lattice: bool = False
    custom_lattice: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    )
    get_lattice_from_file: bool = False
    lattice_file_location: str = "./"
    apply_lattice_to_all_frames: bool = True

    def __str__(self) -> str:
        lines = []
        for key, value in self.__dict__.items():
            if value is not None:
                if not self.apply_custom_lattice and key == "apply_custom_lattice":
                    lines.append(f"\t\t|- {key}: {value}")
                    break
                elif key == "custom_lattice":
                    line1 = f"\t\t|- {key}:"
                    lx = np.array2string(
                        value[0],
                        separator=", ",
                        formatter={"float_kind": lambda x: f"{x}"},
                    )
                    ly = np.array2string(
                        value[1],
                        separator=", ",
                        formatter={"float_kind": lambda x: f"{x}"},
                    )
                    lz = np.array2string(
                        value[2],
                        separator=", ",
                        formatter={"float_kind": lambda x: f"{x}"},
                    )
                    lines.append(
                        f"{line1}\n\t\t\tlx = {lx}\n\t\t\tly = {ly}\n\t\t\tlz = {lz}"
                    )
                else:
                    lines.append(f"\t\t|- {key}: {value}")
        output = """

        Lattice Settings:
        -----------------
{}
        """.format("\n".join(lines))
        return output


@dataclass
class Settings:
    """Settings for the Reve package and it is constructed using the SettingsBuilder."""

    project_name: str = "default"
    export_directory: str = "export"
    file_location: str = "./"
    range_of_frames: Tuple[int, int] = (0, -1)
    apply_pbc: bool = True
    wrap_position: bool = True
    verbose: bool = False
    save_logs: bool = False
    save_performance: bool = False
    decorate_input_file: bool = False # NOTE: beta features works only with extended XYZ file format.
    coordination_mode: str = "all_types"
    use_fast_neighbor_search: bool = True
    general: GeneralSettings = field(default_factory=GeneralSettings)
    cutoffs: List[Cutoff] = field(default_factory=lambda: [])
    lattice: LatticeSettings = field(default_factory=LatticeSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)

    @property
    def output_directory(self) -> str:
        return os.path.join(self.export_directory, self.project_name)

    def get_cutoff(self, type1: str, type2: str) -> float | None:
        for cutoff in self.cutoffs:
            if cutoff.type1 == type1 and cutoff.type2 == type2:
                return cutoff.distance
            elif cutoff.type1 == type2 and cutoff.type2 == type1:
                return cutoff.distance
        return None

    def get_max_cutoff(self) -> float:
        return max(c.distance for c in self.cutoffs)

    def set_range_of_frames(self, start: int, end: Optional[int] = None):
        if end is None:
            end = -1
        if start < 0:
            raise ValueError("Start frame cannot be negative")
        if end != -1 and start > end:
            raise ValueError("Start frame cannot be greater than end frame")
        self.range_of_frames = (start, end)

    def __str__(self) -> str:
        lines = []
        for key, value in self.__dict__.items():
            if value is not None:
                if key == "general":
                    continue
                elif key == "lattice":
                    lines.append(f"\t{str(self.lattice)}")
                elif key == "analysis":
                    lines.append(f"\t{str(self.analysis)}")
                elif key == "cutoffs":
                    line1 = f"\t|- {key:}:"
                    for cutoff in value:
                        line1 += f"\n\t\t{str(cutoff)}"
                    lines.append(line1)
                else:
                    lines.append(f"\t|- {key}: {value}")

        output = """
        General Settings:
        ----------------
{}
        """.format("\n".join(lines))
        return output


class SettingsBuilder:
    def __init__(self):
        self._settings = Settings()  # Start with default settings

    def with_lattice(self, lattice: LatticeSettings):
        if not isinstance(lattice, LatticeSettings):
            raise ValueError(f"Invalid lattice settings: {lattice}")
        self._settings.lattice = lattice
        return self

    def with_general(self, general: GeneralSettings):
        if not isinstance(general, GeneralSettings):
            raise ValueError(f"Invalid general settings: {general}")
        if not general.project_name:
            raise ValueError(f"Invalid project name: {general.project_name}")
        if not general.export_directory:
            raise ValueError(f"Invalid export directory: {general.export_directory}")
        if not general.file_location:
            raise ValueError(f"Invalid file location: {general.file_location}")
        if not general.range_of_frames:
            raise ValueError(f"Invalid range of frames: {general.range_of_frames}")
        if not general.cutoffs:
            raise ValueError(f"Invalid cutoffs list: {general.cutoffs}")
        if general.apply_pbc is None:
            raise ValueError(f"Invalid apply pbc: {general.apply_pbc}")

        self._settings.project_name = general.project_name
        self._settings.export_directory = general.export_directory
        self._settings.file_location = general.file_location
        self._settings.range_of_frames = general.range_of_frames
        self._settings.apply_pbc = general.apply_pbc
        self._settings.cutoffs = general.cutoffs
        self._settings.coordination_mode = general.coordination_mode
        self._settings.use_fast_neighbor_search = general.use_fast_neighbor_search
        if general.verbose is not None:
            self._settings.verbose = general.verbose
        if general.save_logs is not None:
            self._settings.save_logs = general.save_logs
        if general.save_performance is not None:
            self._settings.save_performance = general.save_performance
        if general.decorate_input_file is not None:
            self._settings.decorate_input_file = general.decorate_input_file
        return self

    def with_analysis(self, analysis: AnalysisSettings):
        """Enhanced builder method that handles with_all and exclusions."""
        if not isinstance(analysis, AnalysisSettings):
            raise ValueError(f"Invalid analysis settings: {analysis}")

        # Handle settings creation for enabled analyzers
        if (
            analysis.is_analyzer_enabled("bond_angular_distribution")
            and analysis.bad_settings is None
        ):
            # Create a default BADAnalysisSettings object if not created
            bad_settings = BADAnalysisSettings(
                triplets_to_calculate=["O-O-O", "O-Si-O", "Si-O-Si", "Si-Si-Si"]
            )
            analysis.bad_settings = bad_settings

        if (
            analysis.is_analyzer_enabled("pair_distribution_function")
            and analysis.pdf_settings is None
        ):
            # Create a default PDFAnalysisSettings object if not created
            pdf_settings = PDFAnalysisSettings(
                pairs_to_calculate=["O-O", "O-Si", "Si-Si", "total"]
            )
            analysis.pdf_settings = pdf_settings

        if (
            analysis.is_analyzer_enabled("neutron_structure_factor_fft")
            and analysis.sqfft_settings is None
        ):
            # Create a SQFFTAnalysisSettings object if not created
            sqfft_settings = SQFFTAnalysisSettings()
            analysis.sqfft_settings = sqfft_settings

        if (
            analysis.is_analyzer_enabled("structural_units")
            and analysis.strunits_settings is None
        ):
            # Create a STRUNITSAnalysisSettings object if not created
            strunits_settings = STRUNITSAnalysisSettings(
                units_to_calculate=[
                    "SiO4",
                    "SiO5",
                    "SiO6",
                    "SiO7",
                    "OSi1",
                    "OSi2",
                    "OSi3",
                    "OSi4",
                ]
            )
            analysis.strunits_settings = strunits_settings

        if (
            analysis.is_analyzer_enabled("connectivity")
            and analysis.connect_settings is None
        ):
            # Create a CONNAnalysisSettings object if not created
            connect_settings = CONNAnalysisSettings(
                networking_species="Si",
                bridging_species="O",
            )
            analysis.connect_settings = connect_settings

        if (
            analysis.is_analyzer_enabled("polyhedricity")
            and analysis.poly_settings is None
        ):
            # Create a POLYAnalysisSettings object if not created
            poly_settings = POLYAnalysisSettings(
                central_species="Si",
                vertices_species="O",
                max_c=0.2,
                print_forms=False,
                calculate_distributions=True,
            )
            analysis.poly_settings = poly_settings

        if (
            analysis.is_analyzer_enabled("tetrahedricity")
            and analysis.tetra_settings is None
        ):
            # Create a TETRAAnalysisSettings object if not created
            tetra_settings = TETRAAnalysisSettings(
                central_species="Si",
                vertices_species="O",
                max_c=0.2,
                ideal_cv_angle=109.47,
                ideal_vvv_angle=60.0,
                print_forms=False,
                calculate_distributions=True,
            )
            analysis.tetra_settings = tetra_settings

        self._settings.analysis = analysis
        return self

    def build(self) -> Settings:
        return self._settings


__all__ = [
    Settings,
    SettingsBuilder,
    AnalysisSettings,
    LatticeSettings,
    Cutoff,
    GeneralSettings,
    PDFAnalysisSettings,
    BADAnalysisSettings,
    SQFFTAnalysisSettings,
    STRUNITSAnalysisSettings,
    CONNAnalysisSettings,
    POLYAnalysisSettings,
    TETRAAnalysisSettings,
]
