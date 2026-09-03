"""

Fiber Formation Index (FFI) Analysis for peptide self-assembly simulations.
Measures elongated fibrillar networks using inertia moments, cross-sectional uniformity, and fibrillar order.

calculate_ffi taken from 
Baskaran et al., "Shape Descriptors for Peptide Self-Assembly", Faraday Discussions, 2025.
DOI: 10.1039/D4FD00201F

Slightly modified. Here we also grab the largest cluster in the system
and calculate the FFI for that cluster, rather than the entire system.
We also select only the martini beads (BB, SC1, SC2, SC3) for the FFI calculation.
"""

import numpy as np
import MDAnalysis as mda
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from MDAnalysis.lib.distances import distance_array


DEFAULT_MIN_FIBER_SIZE=1000

def get_largest_cluster_beads(uni, chain_groups, frame=None, cutoff_distance=22.5):
    """
    Return the peptide beads belonging to the largest cluster at a single frame.

    Parameters
    ----------
    uni : mda.Universe
        Prebuilt MDAnalysis Universe.
    chain_groups : list
        List of residue groups (or AtomGroups), one per peptide chain.
    frame : int or None
        Frame to analyze. If None, uses the current trajectory frame.
    cutoff_distance : float
        Distance cutoff (Angstroms) for hierarchical clustering.

    Returns
    -------
    member_atoms : mda.AtomGroup
        All beads from the peptide chains in the largest cluster.
    member_chain_indices : np.ndarray
        Indices (into chain_groups) of the chains in the largest cluster.
    """
    if frame is not None:
        uni.trajectory[frame]

    coms = np.array([g.center_of_mass() for g in chain_groups])
    dist_matrix = distance_array(coms, coms, box=uni.dimensions)
    condensed = dist_matrix[np.triu_indices(len(coms), k=1)]
    Z = linkage(condensed, method='single')
    clustering = fcluster(Z, t=cutoff_distance, criterion='distance')

    # Find the label of the largest cluster (not just its size)
    labels, counts = np.unique(clustering, return_counts=True)
    largest_label = labels[np.argmax(counts)]
    member_chain_indices = np.where(clustering == largest_label)[0]

    # Concatenate the beads of all member chains into one AtomGroup
    member_atoms = chain_groups[member_chain_indices[0]].atoms
    for i in member_chain_indices[1:]:
        member_atoms = member_atoms + chain_groups[i].atoms

    return member_atoms, member_chain_indices


def calculate_ffi(universe, seq, min_fiber_size=DEFAULT_MIN_FIBER_SIZE,
                  first=0, last=None, skip=1):
    """Programmatic API for FFI analysis."""

    #peptides = u.select_atoms('all')
    peptides = universe.select_atoms('name BB SC1 SC2 SC3')

    n_chains = len(peptides.residues) // len(seq)
    chain_groups = [
        peptides.residues[len(seq)*i:len(seq)*(i+1)]
        for i in range(n_chains)
    ]
    
    frame_data = []
    frames = range(first, last or len(universe.trajectory), skip)

    for frame_number in frames:
        universe.trajectory[frame_number]
        chain_groups = [
            peptides.residues[len(seq)*i:len(seq)*(i+1)]
            for i in range(n_chains)
        ]

        _member_atoms, _member_chain_indices = get_largest_cluster_beads(universe, chain_groups, frame_number)

        ####
        ## SWAPPED peptides FOR _member_atoms
        # positions = peptides.positions
        # com = peptides.center_of_mass()
        positions = _member_atoms.positions
        com = _member_atoms.center_of_mass()

        centered = positions - com

        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]

        # Fibrillar condition: high elongation (l1 >> l2, l3) and solid core (l2 ~ l3)
        is_fiber = (l1 / max(l2, 1e-5) > 3.0) and (len(_member_atoms) >= min_fiber_size)
        fiber_count = 1 if is_fiber else 0

        frame_data.append({
            'frame': frame_number,
            'peptides': len(_member_atoms),
            'fiber_count': fiber_count,
            'total_peptides_in_fibers': len(_member_atoms) if is_fiber else 0,
            'avg_fiber_size': len(_member_atoms) if is_fiber else 0,
            'l1_div_l2':l1/max(l2, 1e-5)
        })

    df = pd.DataFrame(frame_data)
    return df
