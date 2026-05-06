# Import necessary modules
from rreve import SettingsBuilder, main
import rreve.config.settings as c

# Path to the trajectory file
# path = "./examples/inputs/example-SiO2-1008at.xyz"
path = "./examples/inputs/example-SiO2-27216at.xyz"
# path = "./examples/inputs/example-SiO2-96000at.xyz"

# General settings
config_general = c.GeneralSettings(
    project_name="example-SiO2",  # Project name
    export_directory="./examples/outputs/",  # Export directory
    file_location=path,  # File location
    range_of_frames=(0, 5),  # Range of frames
    apply_pbc=True,  # Apply periodic boundary conditions
    verbose=False,  # Verbose mode (if True, print title, progress bars, etc.)
    save_logs=True,  # Save logs    (save logs to export_directory/logs.txt)
    save_performance=True,  # Save performance (save performance data to export_directory/performance...json)
    decorate_input_file=True,
    cutoffs=[
        c.Cutoff("Si", "Si", 3.50),
        c.Cutoff("Si", "O", 2.30),
        c.Cutoff("O", "O", 3.05),
    ],
    # coordination_mode="same_type",
    coordination_mode="different_type",
)

# Lattice settings
config_lattice = c.LatticeSettings(
    apply_custom_lattice=False,  # If False, read lattice from trajectory file
)

# Analysis settings
config_analysis = c.AnalysisSettings(
    with_all=False,
    with_polyhedricity=True,
    with_structural_units=True,
    with_bond_angular_distribution=True,
)

# config_analysis.exclude_analyzer("neutron_structure_factor_fft")

# Build Settings object
settings = (
    SettingsBuilder()
    .with_general(config_general)  # General settings \
    .with_lattice(config_lattice)  # Lattice settings \
    .with_analysis(config_analysis)  # Analysis settings \
    .build()  # Don't forget to build the settings object
)

# Run the main function to process the trajectory
main(settings)
