"""Physical sanity checks on the amorphous SiO2 fixture.

These assert qualitative properties of glassy silica that any correct
neighbor-finding + analysis pipeline must reproduce.
"""

import numpy as np


def test_silicon_is_mostly_four_coordinated(sio2_frame):
    si_coords = [n.coordination for n in sio2_frame.get_nodes_by_element("Si")]
    mean_coord = np.mean(si_coords)
    # well-relaxed SiO2 glass: Si is predominantly SiO4
    assert 3.5 < mean_coord < 4.5
    fraction_four = np.mean([c == 4 for c in si_coords])
    assert fraction_four > 0.5


def test_oxygen_is_mostly_two_coordinated(sio2_frame):
    o_coords = [n.coordination for n in sio2_frame.get_nodes_by_element("O")]
    # bridging oxygens dominate -> coordination ~2
    assert 1.7 < np.mean(o_coords) < 2.3


def test_si_o_si_bridging_angle_is_physical(sio2_frame):
    # mean Si-O-Si angle in silica glass sits around 140-150 degrees
    angles = []
    lattice = sio2_frame.get_lattice()
    from rreve.utils.geometry import calculate_pbc_angle

    for o in sio2_frame.get_nodes_by_element("O"):
        si_neigh = [n for n in o.neighbors if n.symbol == "Si"]
        if len(si_neigh) == 2:
            angles.append(
                calculate_pbc_angle(
                    si_neigh[0].position, o.position, si_neigh[1].position, lattice
                )
            )
    assert len(angles) > 0
    assert 120.0 < np.mean(angles) < 165.0
