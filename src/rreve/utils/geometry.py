import numpy as np
from numpy.core.multiarray import dtype
from numba import njit, prange
from numba_progress import ProgressBar
from tqdm import tqdm


@njit(cache=True, fastmath=True)
def wrap_position(position: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Wrap position in a periodic lattice
    (ref: https://en.wikipedia.org/wiki/Fractional_coordinates#Relationship_between_fractional_and_Cartesian_coordinates)

    Args:
        position (np.ndarray): The position to wrap
        lattice (np.ndarray): The lattice of the system

    Returns:
        np.ndarray: The wrapped position
    """
    position = np.ascontiguousarray(position)
    lattice = np.ascontiguousarray(lattice)

    lattice_inv = np.linalg.inv(lattice)
    lattice_inv = np.ascontiguousarray(lattice_inv)

    fractional_position = np.dot(position, lattice_inv)
    fractional_position = np.ascontiguousarray(fractional_position)

    fractional_position -= np.floor(fractional_position)
    fractional_position = np.ascontiguousarray(fractional_position)

    wrapped_position = np.dot(fractional_position, lattice)
    wrapped_position = np.ascontiguousarray(wrapped_position)

    return wrapped_position


@njit(cache=True, fastmath=True, parallel=True)
def wrap_positions(
    positions: np.ndarray, lattice: np.ndarray, progress_proxy: ProgressBar
) -> np.ndarray:
    """Wrap positions in a periodic lattice
    (ref: https://en.wikipedia.org/wiki/Fractional_coordinates#Relationship_between_fractional_and_Cartesian_coordinates)

    Args:
        positions (np.ndarray): The positions to wrap
        lattice (np.ndarray): The lattice of the system

    Returns:
        np.ndarray: The wrapped positions
    """
    wrapped_positions = np.zeros_like(positions)
    lattice = np.ascontiguousarray(lattice)
    lattice_inv = np.linalg.inv(lattice)
    lattice_inv = np.ascontiguousarray(lattice_inv)

    for i in prange(positions.shape[0]):
        position = np.ascontiguousarray(positions[i])

        fractional_position = np.dot(position, lattice_inv)
        fractional_position = np.ascontiguousarray(fractional_position)

        fractional_position -= np.floor(fractional_position)
        fractional_position = np.ascontiguousarray(fractional_position)

        wrapped_position = np.dot(fractional_position, lattice)
        wrapped_position = np.ascontiguousarray(wrapped_position)

        wrapped_positions[i] = wrapped_position

        if progress_proxy is not None:
            progress_proxy.update(1)

    return wrapped_positions


@njit(cache=True, fastmath=True)
def calculate_direct_distance(position1: np.ndarray, position2: np.ndarray) -> float:
    """Return the distance for a given pair of positions in a direct space."""
    return np.linalg.norm(position1 - position2)


@njit(cache=True, fastmath=True)
def calculate_pbc_distance(
    position1: np.ndarray, position2: np.ndarray, lattice: np.ndarray
) -> float:
    """Return the minimum distance between two positions, taking into account periodic boundary conditions set by the lattice.
        (ref: https://en.wikipedia.org/wiki/Fractional_coordinates#Relationship_between_fractional_and_Cartesian_coordinates)

    Args:
        position1 (np.ndarray): The first position
        position2 (np.ndarray): The second position
        lattice (np.ndarray): The lattice of the system

    Returns:
        float: The minimum distance between the two positions
    """
    # Calculate the direct displacement vector
    direct_disp = position1 - position2

    # Get fractional coordinates in the lattice
    inv_lattice = np.linalg.inv(lattice)
    frac_disp = np.dot(inv_lattice, direct_disp)

    # Minimum image convention
    frac_disp -= np.round(frac_disp)

    min_disp = np.dot(frac_disp, lattice)

    return np.linalg.norm(min_disp)


@njit(cache=True, fastmath=True)
def calculate_direct_angle(
    position1: np.ndarray, position2: np.ndarray, position3: np.ndarray
) -> float:
    """Return the angle between three positions in a direct space."""
    angle_rad = np.arccos(
        np.dot((position1 - position2), (position3 - position2))
        / (
            np.linalg.norm(position1 - position2)
            * np.linalg.norm(position3 - position2)
        )
    )
    angle_deg = np.degrees(angle_rad)
    return angle_deg


@njit(cache=True, fastmath=True)
def calculate_pbc_angle(
    neighbor1_pos: np.ndarray,
    central_pos: np.ndarray,
    neighbor2_pos: np.ndarray,
    lattice: np.ndarray,
) -> float:
    """Return the angle formed by neighbor1-central-neighbor2 in a periodic space.

    Args:
        neighbor1_pos (np.ndarray): Position of the first neighbor
        central_pos (np.ndarray): Position of the central atom (vertex of the angle)
        neighbor2_pos (np.ndarray): Position of the second neighbor
        lattice (np.ndarray): The lattice matrix of the system (3x3)

    Returns:
        float: The angle formed at the central atom in degrees
    """
    # Calculate displacement vectors FROM central TO neighbors
    direct_disp1 = neighbor1_pos - central_pos
    direct_disp2 = neighbor2_pos - central_pos

    # Get fractional coordinates in the lattice
    inv_lattice = np.linalg.inv(lattice)
    frac_disp1 = np.dot(inv_lattice, direct_disp1)
    frac_disp2 = np.dot(inv_lattice, direct_disp2)

    # Apply minimum image convention
    frac_disp1 -= np.round(frac_disp1)
    frac_disp2 -= np.round(frac_disp2)

    # Convert back to Cartesian coordinates
    min_disp1 = np.dot(frac_disp1, lattice)
    min_disp2 = np.dot(frac_disp2, lattice)

    # Calculate the angle
    dot_product = np.dot(min_disp1, min_disp2)
    norm1 = np.linalg.norm(min_disp1)
    norm2 = np.linalg.norm(min_disp2)

    # Clamp to avoid numerical errors with arccos
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def cartesian_to_fractional(position: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert a Cartesian position to fractional coordinates in a periodic space.
    (ref: https://en.wikipedia.org/wiki/Fractional_coordinates#Relationship_between_fractional_and_Cartesian_coordinates)

    Args:
        position (np.ndarray): The Cartesian position
        lattice (np.ndarray): The lattice of the system

    Returns:
        np.ndarray: The fractional coordinates
    """
    return np.linalg.solve(lattice.T, position.T).T


def fractional_to_cartesian(position: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert a fractional position to Cartesian coordinates in a periodic space.
    (ref: https://en.wikipedia.org/wiki/Fractional_coordinates#Relationship_between_fractional_and_Cartesian_coordinates)

    Args:
        position (np.ndarray): The fractional position
        lattice (np.ndarray): The lattice of the system

    Returns:
        np.ndarray: The Cartesian coordinates
    """
    return np.dot(position, lattice)


@njit(cache=True, fastmath=True)
def calculate_gyration_radius(
    positions: np.ndarray, center_of_mass: np.ndarray
) -> float:
    """
    Calculates the gyration radius for a set of positions.

    Args:
        positions (np.ndarray): The positions of the cluster
        center_of_mass (np.ndarray): The center of mass of the cluster

    Returns:
        float: The gyration radius of the cluster
    """
    if positions.shape[0] == 0:
        return 0.0

    rg_squared = 0.0
    n_nodes = positions.shape[0]

    # Sum the squared distances from the center of mass
    for i in range(n_nodes):
        dx = positions[i, 0] - center_of_mass[0]
        dy = positions[i, 1] - center_of_mass[1]
        dz = positions[i, 2] - center_of_mass[2]
        rg_squared += dx**2 + dy**2 + dz**2

    # Return the root of the mean squared distance
    return np.sqrt(rg_squared / n_nodes)


@njit(cache=True, fastmath=True, parallel=True)
def calculate_pbc_distances_batch(
    pos1_batch: np.ndarray,
    pos2_batch: np.ndarray,
    lattice: np.ndarray,
    progress_proxy: ProgressBar,
) -> np.ndarray:
    """Calculate PBC distances for batches of position pairs."""
    inv_lattice = np.linalg.inv(lattice)
    distances = np.empty(pos1_batch.shape[0], dtype=np.float64)

    for i in prange(pos1_batch.shape[0]):
        # Calculate direct displacement
        direct_disp = pos1_batch[i] - pos2_batch[i]
        # Get fractional coordinates
        frac_disp = inv_lattice @ direct_disp
        # Minimum image convention
        frac_disp = frac_disp - np.round(frac_disp)
        min_disp = frac_disp @ lattice
        distances[i] = np.sqrt(np.sum(min_disp * min_disp))
        if progress_proxy is not None:
            progress_proxy.update(1)

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def calculate_pbc_cv_distances_batch(
    pos_center: np.ndarray,
    pos_vertices: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Calculate PBC distances for batches of position pairs."""
    inv_lattice = np.linalg.inv(lattice)
    distances = np.empty(pos_vertices.shape[0], dtype=np.float64)

    for i in prange(pos_vertices.shape[0]):
        # Calculate direct displacement
        direct_disp = pos_center - pos_vertices[i]
        # Get fractional coordinates
        frac_disp = inv_lattice @ direct_disp
        # Minimum image convention
        frac_disp = frac_disp - np.round(frac_disp)
        min_disp = frac_disp @ lattice
        distances[i] = np.sqrt(np.sum(min_disp * min_disp))

    return distances


@njit(cache=True, fastmath=True)
def calculate_pbc_dot_distances_combinations(
    pos_batch: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Calculate PBC distances for unique combinations of position pairs within a single batch."""
    inv_lattice = np.linalg.inv(lattice)
    # Calculate the number of unique combinations
    num_combinations = pos_batch.shape[0] * (pos_batch.shape[0] - 1) // 2
    distances = np.empty(num_combinations, dtype=np.float64)

    k = 0
    for i in range(pos_batch.shape[0]):
        for j in range(i + 1, pos_batch.shape[0]):
            # Calculate direct displacement
            direct_disp = pos_batch[i] - pos_batch[j]
            # Get fractional coordinates
            frac_disp = inv_lattice @ direct_disp
            # Minimum image convention
            frac_disp = frac_disp - np.round(frac_disp)
            min_disp = frac_disp @ lattice
            distances[k] = np.sqrt(np.sum(min_disp * min_disp))
            k += 1

    return distances


@njit(cache=True, fastmath=True)
def calculate_pbc_angle_combinations(
    pos_batch: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Calculate PBC angles for unique combinations of position pairs within a single batch.

    Args:
        pos_batch (np.ndarray): Positions
        lattice (np.ndarray): The lattice matrix of the system (3x3)

    Returns:
        angles: All the possible angles formed between atoms in degrees
    """
    n_atoms = pos_batch.shape[0]
    inv_lattice = np.linalg.inv(lattice)

    # Number of unique triples
    num_combinations = n_atoms * (n_atoms - 1) * (n_atoms - 2) // 2
    angles = np.empty(num_combinations, dtype=np.float64)

    k = 0
    for i in range(n_atoms):
        for j in range(n_atoms):
            if j == i:
                continue
            for l in range(j + 1, n_atoms):
                if l == i:
                    continue
                # Displacement vectors (central -> neighbor)
                direct_disp1 = pos_batch[j] - pos_batch[i]
                direct_disp2 = pos_batch[l] - pos_batch[i]

                # Convert to fractional coords
                frac_disp1 = inv_lattice @ direct_disp1
                frac_disp2 = inv_lattice @ direct_disp2

                # Apply minimum image convention
                frac_disp1 -= np.round(frac_disp1)
                frac_disp2 -= np.round(frac_disp2)

                # Back to Cartesian
                min_disp1 = frac_disp1 @ lattice
                min_disp2 = frac_disp2 @ lattice

                # Compute angle
                dot_product = np.dot(min_disp1, min_disp2)
                norm1 = np.linalg.norm(min_disp1)
                norm2 = np.linalg.norm(min_disp2)

                if norm1 > 1e-12 and norm2 > 1e-12:
                    cos_angle = dot_product / (norm1 * norm2)
                    # Clamp to avoid numerical issues
                    if cos_angle > 1.0:
                        cos_angle = 1.0
                    elif cos_angle < -1.0:
                        cos_angle = -1.0
                    angle_rad = np.arccos(cos_angle)
                    angles[k] = np.degrees(angle_rad)
                else:
                    # If one vector is zero (unlikely), assign NaN
                    angles[k] = np.nan

                k += 1

    return angles


@njit(cache=True, fastmath=True)
def fast_histogram(distances: np.ndarray, r_max: float, bins: int) -> np.ndarray:
    """Fast histogram calculation using numba."""
    hist = np.zeros(bins, dtype=np.int64)
    dr = r_max / bins

    for dist in distances:
        if 0 < dist < r_max:
            bin_idx = int(dist / dr)
            if bin_idx < bins:
                hist[bin_idx] += 1

    return hist


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def calculate_components(qx, qy, qz, positions, progress_proxy):
    """Loop to calculate cosine and sine components for S(q)."""
    qcos, qsin = np.zeros_like(qx), np.zeros_like(qx)
    for i in prange(len(positions)):
        pos = positions[i]
        dot_product = qx * pos[0] + qy * pos[1] + qz * pos[2]
        qcos += np.cos(dot_product)
        qsin += np.sin(dot_product)
        if progress_proxy is not None:
            progress_proxy.update(1)
    return qcos, qsin


@njit(nogil=True, cache=True, fastmath=True)
def calculate_tetrahedricity(distances: np.ndarray) -> float:
    squared_distances = distances**2

    mean_sqr_distance = np.mean(squared_distances)

    tetrahedricity = 0.0

    for i in range(len(distances)):
        for j in range(len(distances)):
            tetrahedricity += (distances[i] - distances[j]) ** 2

    tetrahedricity = tetrahedricity / (15 * mean_sqr_distance)

    return tetrahedricity


@njit(nogil=True, cache=True, fastmath=True)
def calculate_square_based_pyramid(distances: np.ndarray) -> float:
    _distances = np.copy(distances)
    _distances[-2] /= np.sqrt(2)
    _distances[-1] /= np.sqrt(2)

    sbp_polyhedricity = 0.0

    mean_sqr_distance = np.mean(_distances**2)

    for i in range(len(_distances)):
        for j in range(len(_distances)):
            sbp_polyhedricity += (_distances[i] - _distances[j]) ** 2

    sbp_polyhedricity = sbp_polyhedricity / (45 * mean_sqr_distance)

    return sbp_polyhedricity


@njit(nogil=True, cache=True, fastmath=True)
def calculate_triangular_bipyramid(distances: np.ndarray) -> float:
    _distances = np.copy(distances)
    _distances[-4] /= np.sqrt(3.0 / 2.0)
    _distances[-3] /= np.sqrt(3.0 / 2.0)
    _distances[-2] /= np.sqrt(3.0 / 2.0)
    _distances[-1] /= np.sqrt(2.0)

    tbp_polyhedricity = 0.0

    mean_sqr_distance = np.mean(_distances**2)

    for i in range(len(_distances)):
        for j in range(len(_distances)):
            tbp_polyhedricity += (_distances[i] - _distances[j]) ** 2

    tbp_polyhedricity = tbp_polyhedricity / (45 * mean_sqr_distance)

    return tbp_polyhedricity


@njit(nogil=True, cache=True, fastmath=True)
def calculate_octahedricity(distances: np.ndarray) -> float:
    _distances = np.copy(distances)
    _distances[-3] /= np.sqrt(2)
    _distances[-2] /= np.sqrt(2)
    _distances[-1] /= np.sqrt(2)

    octahedricity = 0.0

    mean_sqr_distance = np.mean(_distances**2)

    for i in range(len(_distances)):
        for j in range(len(_distances)):
            octahedricity += (_distances[i] - _distances[j]) ** 2

    octahedricity = octahedricity / (105 * mean_sqr_distance)

    return octahedricity


def warmup_jit():
    """
    Warms up the JIT-compiled functions by calling them with dummy data.
    """
    # Dummy data
    dummy_pos1 = np.array([1.0, 1.0, 1.0])
    dummy_pos2 = np.array([2.0, 2.0, 2.0])
    dummy_pos3 = np.array([3.0, 3.0, 3.0])
    dummy_positions = np.array([dummy_pos1, dummy_pos2, dummy_pos3])
    dummy_lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    dummy_center_of_mass = np.array([2.0, 2.0, 2.0])
    dummy_q = np.array([1.0, 1.0, 1.0])

    # progress bar
    progress_bar = tqdm(
        desc="Compiling jitted functions ...",
        total=17,
        colour="magenta",
        ascii=True,
        leave=True,
    )
    # Call JIT functions to compile them
    wrap_position(dummy_pos1, dummy_lattice)
    progress_bar.update(1)
    wrap_positions(dummy_positions, dummy_lattice, None)
    progress_bar.update(1)
    calculate_direct_distance(dummy_pos1, dummy_pos2)
    progress_bar.update(1)
    calculate_pbc_distance(dummy_pos1, dummy_pos2, dummy_lattice)
    progress_bar.update(1)
    calculate_direct_angle(dummy_pos1, dummy_pos2, dummy_pos3)
    progress_bar.update(1)
    calculate_pbc_angle(dummy_pos1, dummy_pos2, dummy_pos3, dummy_lattice)
    progress_bar.update(1)
    calculate_gyration_radius(dummy_positions, dummy_center_of_mass)
    progress_bar.update(1)
    calculate_pbc_distances_batch(dummy_positions, dummy_positions, dummy_lattice, None)
    progress_bar.update(1)
    calculate_pbc_cv_distances_batch(dummy_positions[0], dummy_positions, dummy_lattice)
    progress_bar.update(1)
    distances = calculate_pbc_dot_distances_combinations(dummy_positions, dummy_lattice)
    progress_bar.update(1)
    calculate_pbc_angle_combinations(dummy_positions, dummy_lattice)
    progress_bar.update(1)
    fast_histogram(np.array([1.0, 2.0, 3.0]), 10.0, 10)
    progress_bar.update(1)
    calculate_components(dummy_q, dummy_q, dummy_q, dummy_positions, None)
    progress_bar.update(1)
    calculate_tetrahedricity(distances)
    progress_bar.update(1)
    calculate_square_based_pyramid(distances)
    progress_bar.update(1)
    calculate_triangular_bipyramid(distances)
    progress_bar.update(1)
    calculate_octahedricity(distances)
    progress_bar.update(1)
    progress_bar.close()


__all__ = [
    "wrap_position",
    "wrap_positions",
    "calculate_direct_distance",
    "calculate_pbc_distance",
    "calculate_direct_angle",
    "calculate_pbc_angle",
    "calculate_gyration_radius",
    "calculate_pbc_distances_batch",
    "calculate_pbc_dot_distances_combinations",
    "fast_histogram",
    "calculate_components",
    "calculate_tetrahedricity",
    "calculate_square_based_pyramid",
    "calculate_triangular_bipyramid",
    "calculate_octahedricity",
    "warmup_jit",
]
