"""
Contains functions for analyzing clustering behavior of peptides in 
MD simulations, particularly coarse-grained Martini simulations.
Clusters are defined using hierarchical clustering of peptide centers of mass,
with a specified distance cutoff. The main functions compute the average maximum
cluster size over the last 10 ns of a trajectory, and the time evolution of the
maximum cluster size.
"""

import mdtraj as mdt
import mdanalysis as mda
import numpy as np
import freesasa

from aggrepep.martini_sasa import get_martini_vdw_radii
from itertools import combinations
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import mode as stmode
from MDAnalysis.lib.distances import distance_array

def compute_max_cluster_size_over_slice(uni, chain_groups, start_frame, stop_frame, step=1,
                                        cutoff_distance=22.5):
    """
    Compute the average and std of the maximum cluster size over a frame slice.

    Parameters
    ----------
    uni : mda.Universe
        Prebuilt MDAnalysis Universe.
    chain_groups : list
        List of residue groups, one per peptide chain.
    start_frame, stop_frame, step : int
        Frame slice specification (applied as uni.trajectory[start_frame:stop_frame:step]).
    cutoff_distance : float
        Distance cutoff (Angstroms) for hierarchical clustering.

    Returns
    -------
    mean_max : float
        Mean maximum cluster size over the slice.
    std_max : float
        Standard deviation of the maximum cluster size over the slice.
    """
    max_sizes = []
    for ts in uni.trajectory[start_frame:stop_frame:step]:
        coms = np.array([g.center_of_mass() for g in chain_groups])
        dist_matrix = distance_array(coms, coms, box=uni.dimensions)
        condensed = dist_matrix[np.triu_indices(len(coms), k=1)]
        Z = linkage(condensed, method='single')
        clustering = fcluster(Z, t=cutoff_distance, criterion='distance')
        max_cluster_size = stmode(clustering).count
        max_sizes.append(max_cluster_size)

    max_sizes = np.array(max_sizes)
    return max_sizes.mean(), max_sizes.std()


def compute_avg_max_cluster_size(top_fpath, traj_fpath, sequence,
                                 cutoff_distance=22.5,
                                 frames_per_ns=500,
                                 N_blocks=10):
    """
    Expects CG sims.
    Computes the block-averaged maximum cluster size of peptides in the trajectory
    over the last 10 ns. Clusters are defined using hierarchical clustering with a
    cutoff distance (default 22.5 Angstroms).

    Parameters
    ----------
    top_fpath : str | path
        File path to topology.
    traj_fpath : str | path
        File path to trajectory.
    sequence : str
        Peptide sequence (used to determine chains).
    cutoff_distance : float
        Distance cutoff for defining clusters (Angstroms).
    frames_per_ns : int
        Number of frames per nanosecond. Default 500.
    N_blocks : int
        Number of blocks for block averaging. Default 10.

    Returns
    -------
    block_means : np.ndarray
        Per-block mean max cluster size, shape (N_blocks,).
    mean_clusters : float
        Mean across blocks.
    std_clusters : float
        Std across blocks.
    sem_clusters : float
        Standard error of the mean across blocks.
    """
    _seqL = len(sequence)
    uni = mda.Universe(top_fpath, traj_fpath)

    protein_atoms = uni.select_atoms("not name W WP WM NA+ CL-")
    nmol = len(protein_atoms.residues) // _seqL
    chain_groups = [
        protein_atoms.residues[_seqL * j:_seqL * (j + 1)]
        for j in range(nmol)
    ]

    # Analyze the last 10 ns
    N_frames = frames_per_ns * 10
    block_size = N_frames // N_blocks
    total_frames = len(uni.trajectory)
    slice_start = total_frames - N_frames

    block_means = np.zeros(N_blocks)
    for b in range(N_blocks):
        bstart = slice_start + b * block_size
        bstop = bstart + block_size
        mean_b, _std_b = compute_max_cluster_size_over_slice(
            uni, chain_groups,
            start_frame=bstart,
            stop_frame=bstop,
            step=1,
            cutoff_distance=cutoff_distance,
        )
        block_means[b] = mean_b

    mean_clusters = block_means.mean()
    std_clusters = block_means.std()
    sem_clusters = std_clusters / np.sqrt(N_blocks)

    return block_means, mean_clusters, std_clusters, sem_clusters

def compute_max_cluster_size_vs_time(top_fpath, traj_fpath, sequence,
                                     cutoff_distance=22.5,
                                     frames_per_ns=500,
                                     window_ns=1.0,
                                     start_frame=0,
                                     stop_frame=None,
                                     step=1):
    """
    Compute the maximum cluster size as a function of time over a trajectory slice,
    binned into fixed-width time windows.

    Parameters
    ----------
    top_fpath : str | path
        File path to topology.
    traj_fpath : str | path
        File path to trajectory.
    sequence : str
        Peptide sequence (used to determine chains).
    cutoff_distance : float
        Distance cutoff for hierarchical clustering (Angstroms).
    frames_per_ns : int
        Number of frames per nanosecond. Default 500.
    window_ns : float
        Width of each averaging window in nanoseconds. Default 1.0.
    start_frame, stop_frame, step : int
        Frame slice to analyze. stop_frame=None means end of trajectory.

    Returns
    -------
    times_ns : np.ndarray
        Center time (ns) of each window.
    means : np.ndarray
        Mean max cluster size in each window.
    stds : np.ndarray
        Std of max cluster size in each window.
    """
    _seqL = len(sequence)
    uni = mda.Universe(top_fpath, traj_fpath)

    protein_atoms = uni.select_atoms("not name W WP WM NA+ CL-")
    nmol = len(protein_atoms.residues) // _seqL
    chain_groups = [
        protein_atoms.residues[_seqL * j:_seqL * (j + 1)]
        for j in range(nmol)
    ]

    if stop_frame is None:
        stop_frame = len(uni.trajectory)

    window_size = int(round(window_ns * frames_per_ns / step))
    if window_size < 1:
        raise ValueError(
            f"window_ns={window_ns} with step={step} and frames_per_ns={frames_per_ns} "
            "produces a window smaller than 1 frame."
        )

    # Effective frame indices within [start_frame, stop_frame) at given step
    effective_frames = np.arange(start_frame, stop_frame, step)
    n_windows = len(effective_frames) // window_size

    times_ns = np.zeros(n_windows)
    means = np.zeros(n_windows)
    stds = np.zeros(n_windows)

    for w in range(n_windows):
        wstart_eff = w * window_size
        wstop_eff = wstart_eff + window_size
        wstart = effective_frames[wstart_eff]
        wstop = effective_frames[wstop_eff - 1] + step  # exclusive

        mean_w, std_w = compute_max_cluster_size_over_slice(
            uni, chain_groups,
            start_frame=wstart,
            stop_frame=wstop,
            step=step,
            cutoff_distance=cutoff_distance,
        )
        # Window center time, in ns
        center_frame = 0.5 * (wstart + wstop - step)
        times_ns[w] = center_frame / frames_per_ns
        means[w] = mean_w
        stds[w] = std_w

    return times_ns, means, stds