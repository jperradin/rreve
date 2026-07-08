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
def calculate_pbc_angle_around(
    self_pos: np.ndarray,
    pos_batch_n1: np.ndarray,
    pos_batch_n2: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Calculate PBC angles formed like : self - neighbor1 - neighbor2
    
    Args:
        self_pos (np.ndarray): Position 
        pos_batch_n1 (np.ndarray): Positions of first neighbors
        pos_batch_n2 (np.ndarray): Positions of second neighbors
        lattice (np.ndarray): The lattice matrix of the system (3x3)
    """

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
def filter_neighbors_pbc_batch(
    target: np.ndarray,
    cands: np.ndarray,
    lattice: np.ndarray,
    inv_lattice: np.ndarray,
    rcut2: np.ndarray,
):
    """Vectorized PBC-distance filter for one node against many candidates.

    Args:
        target (np.ndarray): (3,) cartesian position of the central node.
        cands (np.ndarray): (M, 3) cartesian positions of candidates.
        lattice (np.ndarray): (3, 3) lattice matrix.
        inv_lattice (np.ndarray): (3, 3) precomputed inverse lattice.
        rcut2 (np.ndarray): (M,) per-pair squared cutoff. Negative entries flag
            invalid pairs and are skipped.

    Returns:
        keep (np.ndarray): (M,) bool mask of accepted candidates.
        dists (np.ndarray): (M,) float64, distance for kept entries (0 elsewhere).
    """
    M = cands.shape[0]
    keep = np.zeros(M, dtype=np.bool_)
    dists = np.zeros(M, dtype=np.float64)
    tx = target[0]; ty = target[1]; tz = target[2]
    for i in range(M):
        rc2 = rcut2[i]
        if rc2 < 0.0:
            continue
        dx = cands[i, 0] - tx
        dy = cands[i, 1] - ty
        dz = cands[i, 2] - tz
        # Row convention: frac = d @ inv_lattice, cart = frac @ lattice
        fx = inv_lattice[0, 0] * dx + inv_lattice[1, 0] * dy + inv_lattice[2, 0] * dz
        fy = inv_lattice[0, 1] * dx + inv_lattice[1, 1] * dy + inv_lattice[2, 1] * dz
        fz = inv_lattice[0, 2] * dx + inv_lattice[1, 2] * dy + inv_lattice[2, 2] * dz
        fx -= np.round(fx); fy -= np.round(fy); fz -= np.round(fz)
        mx = lattice[0, 0] * fx + lattice[1, 0] * fy + lattice[2, 0] * fz
        my = lattice[0, 1] * fx + lattice[1, 1] * fy + lattice[2, 1] * fz
        mz = lattice[0, 2] * fx + lattice[1, 2] * fy + lattice[2, 2] * fz
        d2 = mx * mx + my * my + mz * mz
        if d2 <= rc2:
            keep[i] = True
            dists[i] = np.sqrt(d2)
    return keep, dists


@njit(cache=True, fastmath=True)
def filter_neighbors_direct_batch(
    target: np.ndarray,
    cands: np.ndarray,
    rcut2: np.ndarray,
):
    """Vectorized direct-distance (no PBC) filter, same contract as
    ``filter_neighbors_pbc_batch``."""
    M = cands.shape[0]
    keep = np.zeros(M, dtype=np.bool_)
    dists = np.zeros(M, dtype=np.float64)
    tx = target[0]; ty = target[1]; tz = target[2]
    for i in range(M):
        rc2 = rcut2[i]
        if rc2 < 0.0:
            continue
        dx = cands[i, 0] - tx
        dy = cands[i, 1] - ty
        dz = cands[i, 2] - tz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= rc2:
            keep[i] = True
            dists[i] = np.sqrt(d2)
    return keep, dists


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


@njit(cache=True, fastmath=True)
def calculate_pbc_cv_angle_combinations(
    center_pos: np.ndarray,
    pos_batch: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Calculate PBC vertex-center-vertex angles (apex at center).

    For each unique pair of vertices, return the angle vertex_i - center - vertex_j.

    Args:
        center_pos (np.ndarray): Position of the central atom (apex of the angle)
        pos_batch (np.ndarray): Positions of the vertices
        lattice (np.ndarray): The lattice matrix of the system (3x3)

    Returns:
        np.ndarray: All vertex-center-vertex angles in degrees
    """
    n_atoms = pos_batch.shape[0]
    inv_lattice = np.linalg.inv(lattice)

    num_combinations = n_atoms * (n_atoms - 1) // 2
    angles = np.empty(num_combinations, dtype=np.float64)

    k = 0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Displacement vectors (center -> vertex)
            direct_disp1 = pos_batch[i] - center_pos
            direct_disp2 = pos_batch[j] - center_pos

            frac_disp1 = inv_lattice @ direct_disp1
            frac_disp2 = inv_lattice @ direct_disp2

            frac_disp1 -= np.round(frac_disp1)
            frac_disp2 -= np.round(frac_disp2)

            min_disp1 = frac_disp1 @ lattice
            min_disp2 = frac_disp2 @ lattice

            dot_product = np.dot(min_disp1, min_disp2)
            norm1 = np.linalg.norm(min_disp1)
            norm2 = np.linalg.norm(min_disp2)

            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = dot_product / (norm1 * norm2)
                if cos_angle > 1.0:
                    cos_angle = 1.0
                elif cos_angle < -1.0:
                    cos_angle = -1.0
                angles[k] = np.degrees(np.arccos(cos_angle))
            else:
                angles[k] = np.nan

            k += 1

    return angles


@njit(nogil=True, cache=True, fastmath=True)
def calculate_tetrahedricity_angles(angles: np.ndarray, ideal_angle: float) -> float:
    """Irregularity of a tetrahedron from a set of angles vs an ideal angle.

    Mean squared relative deviation from the ideal angle (dimensionless), so a
    perfectly regular tetrahedron gives 0.

    Args:
        angles (np.ndarray): Angles in degrees (vertex-center-vertex or
            vertex-vertex-vertex)
        ideal_angle (float): Ideal angle in degrees (~109.47 for v-c-v, 60 for v-v-v)

    Returns:
        float: The angular irregularity
    """
    irregularity = 0.0
    for a in angles:
        irregularity += ((a - ideal_angle) / ideal_angle) ** 2
    irregularity = irregularity / len(angles)
    return irregularity


@njit(nogil=True, cache=True, fastmath=True)
def calculate_errington_q(angles: np.ndarray, ideal_angle: float) -> float:
    """Errington-Debenedetti tetrahedral order parameter q.

    Generalized form of the orientational order parameter of Errington &
    Debenedetti, Nature 409, 318-321 (2001) (itself a rescaling of Chau &
    Hardwick, Mol. Phys. 93, 511 (1998)):

        q = 1 - prefactor * sum_jk (cos(angle_jk) - cos(ideal_angle))^2

    The prefactor is self-normalizing so that q -> 1 for a perfect tetrahedron
    (every angle == ideal_angle) and <q> -> 0 for randomly oriented neighbours.
    For a sin-weighted random angle distribution <cos> = 0 and <cos^2> = 1/3, so
    <(cos - cos_ideal)^2> = 1/3 + cos_ideal^2 and the normalization over ``n``
    angles is prefactor = 1 / (n * (1/3 + cos_ideal^2)).

    This reproduces the published constants:
      * O-Si-O apex set: n = 6, ideal = 109.47 deg (cos = -1/3) -> prefactor 3/8
        (the original Errington-Debenedetti water q).
      * O-O-O vertex set: n = 12, ideal = 60 deg (cos = 1/2) -> prefactor 1/7
        (adaptation for the vertex-vertex-vertex angles of a SiO4 tetrahedron).

    Args:
        angles (np.ndarray): Angles in degrees (vertex-center-vertex or
            vertex-vertex-vertex).
        ideal_angle (float): Ideal angle in degrees (109.47 for O-Si-O, 60 for
            O-O-O).

    Returns:
        float: The order parameter q (1 = perfect tetrahedron, 0 = random).
    """
    cos_ideal = np.cos(np.radians(ideal_angle))
    n = len(angles)
    prefactor = 1.0 / (n * (1.0 / 3.0 + cos_ideal * cos_ideal))
    s = 0.0
    for a in angles:
        c = np.cos(np.radians(a))
        s += (c - cos_ideal) ** 2
    return 1.0 - prefactor * s


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
        total=21,
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
    dummy_inv_lattice = np.linalg.inv(dummy_lattice)
    dummy_rcut2 = np.array([4.0, 4.0, 4.0])
    filter_neighbors_pbc_batch(
        dummy_pos1, dummy_positions, dummy_lattice, dummy_inv_lattice, dummy_rcut2
    )
    progress_bar.update(1)
    filter_neighbors_direct_batch(dummy_pos1, dummy_positions, dummy_rcut2)
    progress_bar.update(1)
    calculate_components(dummy_q, dummy_q, dummy_q, dummy_positions, None)
    progress_bar.update(1)
    calculate_tetrahedricity(distances)
    progress_bar.update(1)
    cv_angles = calculate_pbc_cv_angle_combinations(
        dummy_positions[0], dummy_positions, dummy_lattice
    )
    progress_bar.update(1)
    calculate_tetrahedricity_angles(cv_angles, 109.47)
    progress_bar.update(1)
    # These polyhedricity metrics index the last 2-4 elements of the distance
    # vector (a 5-vertex pyramid has 10 edges, a 6-vertex octahedron 15), so warm
    # them up with a correctly sized dummy rather than the 3-element ``distances``.
    dummy_poly_distances = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    calculate_square_based_pyramid(dummy_poly_distances)
    progress_bar.update(1)
    calculate_triangular_bipyramid(dummy_poly_distances)
    progress_bar.update(1)
    calculate_octahedricity(dummy_poly_distances)
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
    "filter_neighbors_pbc_batch",
    "filter_neighbors_direct_batch",
    "calculate_components",
    "calculate_tetrahedricity",
    "calculate_pbc_cv_angle_combinations",
    "calculate_tetrahedricity_angles",
    "calculate_square_based_pyramid",
    "calculate_triangular_bipyramid",
    "calculate_octahedricity",
    "warmup_jit",
]
