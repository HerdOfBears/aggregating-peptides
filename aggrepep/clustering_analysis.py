"""
Contains functions for analyzing clustering behavior of peptides in 
MD simulations, particularly coarse-grained Martini simulations.
Clusters are defined using hierarchical clustering of peptide centers of mass,
with a specified distance cutoff. The main functions compute the average maximum
cluster size over the last 10 ns of a trajectory, and the time evolution of the
maximum cluster size.

Half of the functions are for estimating max cluster size while
the other half is for estimating number of clusters.

"""

import mdtraj as mdt
import MDAnalysis as mda
import numpy as np
import freesasa

from aggrepep.martini_sasa import get_martini_vdw_radii
from itertools import combinations
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import mode as stmode
from MDAnalysis.lib.distances import distance_array

def compute_mu_i_t(uni, chain_groups, start_frame,i=1):
    """
    Compute the average cluster size M_i(t)/M_{i-1}(t) at a single time point t,
    using the i-th and (i-1)-th moments of the cluster size distribution.
    if i=1, this returns the number-average cluster size, 
        i.e. M_1(t)/M_0(t).
    if i=2, this returns the mass-average cluster size,
        i.e. M_2(t)/M_1(t).
    if i>2, this returns the ratio of the i-th to (i-1)-th moment,

    Parameters:
    -----------
    uni: mda.Universe
        Prebuilt MDAnalysis Universe.
    chain_groups: list
        List of residue groups, one per peptide chain.
    start_frame: int
        Frame index to analyze (applied as uni.trajectory[start_frame]).
    i: int
        Moment order to compute. i=1 for number-average, i=2 for mass-average, etc.

    Returns:
    --------
    mu_i_t: float
        The ratio M_i(t)/M_{i-1}(t) at the specified time point.
    """
    
    _M_imns1_t = compute_moment_i_of_cluster_size_distribution(
        uni, chain_groups, start_frame, start_frame+1, step=1, i=i-1
    )
    _M_i_t     = compute_moment_i_of_cluster_size_distribution(
        uni, chain_groups, start_frame, start_frame+1, step=1, i=i
    )

    return _M_i_t/_M_imns1_t

def compute_moment_i_of_cluster_size_distribution(uni, chain_groups, start_frame, stop_frame, step=1,
                                                cutoff_distance=22.5, i=2):
    """
    Compute the average of the i-th moment of the cluster size distribution
    over a frame slice.

    M_i(t) = sum_r (r^i * n_clusters_of_size_r(t))
    <M_i>  = (1/N_frames) sum_t^N_frames M_i(t)

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
    i : int
        Moment order to compute (e.g., i=2 for second moment).
    
    Returns
    -------
    time_avg_moment_i : float
        Time average of the i-th moment of the cluster size distribution over the slice.
    """
    n_chains = len(chain_groups)
    rmax = n_chains  # max possible cluster size is all chains in one cluster
    rmin = 1         # min possible cluster size is 1 (each chain separate)
    n_clusters_size_r = compute_num_clusters_of_each_size_over_slice(
        uni, chain_groups, start_frame, stop_frame, step, cutoff_distance
    ) # (N_frames, rmax) array of counts of clusters of size r in each frame

    # Compute the i-th moment for each frame, then average over frames for time avg
    moment_i = np.sum(n_clusters_size_r * np.arange(rmin, rmax + 1)**i, axis=1)
    time_avg_moment_i = np.mean(moment_i)

    return time_avg_moment_i

def compute_num_clusters_of_each_size_over_slice(uni, chain_groups, start_frame, stop_frame, step=1,
                                            cutoff_distance=22.5):
    """
    Compute the number of clusters of (size == r) for each value of r
    over a frame slice.

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
    n_clusters_r : np.ndarray
        Number of clusters of (size == r) in each frame of the slice.
    """
    n_clusters_r = []
    N_chains = len(chain_groups)

    # initialize n_clusters_r with zeros for all possible cluster sizes (1 to N_chains)
    n_clusters_r = np.zeros(((stop_frame - start_frame) // step, N_chains), dtype=int)

    # step through frames and compute number of clusters of each size
    for i, ts in enumerate(uni.trajectory[start_frame:stop_frame:step]):
        coms = np.array([g.center_of_mass() for g in chain_groups])
        dist_matrix = distance_array(coms, coms, box=uni.dimensions)
        condensed = dist_matrix[np.triu_indices(len(coms), k=1)]
        Z = linkage(condensed, method='single')

        clustering = fcluster(Z, t=cutoff_distance, criterion='distance')

        # Count clusters of size == r
        cluster_sizes = np.bincount(clustering)
        for r in range(1, N_chains + 1):
            n_clusters_r[i, r - 1] = np.sum(cluster_sizes == r)

    return np.array(n_clusters_r)

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
        wstop  = effective_frames[wstop_eff - 1] + step  # exclusive

        mean_w, std_w = compute_max_cluster_size_over_slice(
            uni, chain_groups,
            start_frame=wstart,
            stop_frame =wstop,
            step=step,
            cutoff_distance=cutoff_distance,
        )
        # Window center time, in ns
        center_frame = 0.5 * (wstart + wstop - step)
        times_ns[w] = center_frame / frames_per_ns
        means[   w] = mean_w
        stds[    w] = std_w

    return times_ns, means, stds

def estimate_cluster_size_auc(times, means, stds=None):
    """
    Trapezoidal rule area-under-curve. 
    means = f(t) points
    times = t points
    
    Parameters
    ----------
    times : np.ndarray
        Window center times (ns), as returned by compute_max_cluster_size_vs_time.
    means : np.ndarray
        Mean max cluster size per window.
    stds : np.ndarray, optional
        Std per window. If provided, an uncertainty on the AUC is returned,
        propagated assuming independent windows.

    Returns
    -------
    auc : float
        Area under the curve, in units of (cluster size) * ns.
    auc_err : float, optional
        Uncertainty on the AUC (only returned if stds is given).
    """
    times = np.asarray(times)
    means = np.asarray(means)

    if len(times) < 2:
        raise ValueError("Need at least two points to estimate an AUC.")

    auc = np.trapz(means, times)

    if stds is None:
        return auc

    # Trapezoidal rule: auc = sum_i 0.5 * (t[i+1] - t[i]) * (y[i] + y[i+1])
    # Each y[i] contributes with weight w[i] = 0.5 * (dt_left + dt_right),
    # where dt_left = t[i]-t[i-1] and dt_right = t[i+1]-t[i] (edges use one side).
    stds = np.asarray(stds)
    dt = np.diff(times)
    weights = np.zeros_like(times, dtype=float)
    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]
    weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    auc_err = np.sqrt(np.sum((weights * stds) ** 2))
    return auc, auc_err