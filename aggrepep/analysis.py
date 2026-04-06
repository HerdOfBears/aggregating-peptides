"""
analysis.py contains functions for analyzing trajectories of peptides
Some of the functions compute all-atom specific properties, while
others are only useful with coarse-grained sims.

- compute_beta_content_score computes the beta strand content of the 
initial configuration and the average over the last 10ns of
a trajectory. (all-atom only)

- compute_aggregation_propensity_sasa computes the SASA-based 
aggregation propensity (AP) score using freesasa. The AP_sasa score
originates from https://doi.org/10.1021/acs.jctc.3c00699 (CG)

- compute_aggregation_propensity_sasa_mdt computes the SASA-based 
aggregation propensity (AP) score using mdtraj.shrake_rupley. 
The AP_sasa score originates from https://doi.org/10.1021/acs.jctc.3c00699 (AA)


"""

import mdtraj as mdt
import numpy as np
import freesasa

from aggrepep.martini_sasa import get_martini_vdw_radii
from itertools import combinations


def compute_beta_content_score(top_fpath, traj_fpath, frames_per_ns=500):
    """
    compute_beta_content computes the beta strand content of the 
    initial configuration and the average over the last 10ns of
    a trajectory.
    
    :param top_fpath: Description
    :param traj_fpath: Description
    :param frames_per_ns: Description

    Returns:
    --------
    beta_content_score : float
        Fraction of beta strands found at the end of the sim
        compared to initial conformation.
    nEs_init : int
        Beta content of the initial conformation
    nEs : int
        average beta content over the last 10ns of the trajectory
    """
    frames_per_ns = int(frames_per_ns)
    traj = mdt.load(traj_fpath, top=top_fpath)
    traj_init = mdt.load(top_fpath)

    _dssp_results = mdt.compute_dssp(traj[-(frames_per_ns*10):])
    
    nEs = np.sum(_dssp_results=="E")/(frames_per_ns*10)

    _dssp_results_init = mdt.compute_dssp(traj_init)
    nEs_init = np.sum(_dssp_results_init=="E")

    beta_content_score = nEs/nEs_init if nEs_init > 0 else 0.0

    return beta_content_score, nEs_init, nEs 


def compute_aggregation_propensity_sasa(top_fpath, traj_fpath, frames_per_ns=500):
    """
    expects CG sims
    compute_aggregation_propensity_sasa computes the SASA-based 
    aggregation propensity (AP) score using freesasa. The AP_sasa score
    originates from https://doi.org/10.1021/acs.jctc.3c00699   
    
    Parameters:
    -----------
    top_fpath: 
        File path to topology file
    traj_fpath: Optional, 
        File path to trajectory file
    frames_per_ns : int
        How many simulation frames correspond to 1ns

    Returns:
    --------
    APsasa : float
        The aggregation propensity score

    SASA_init : float 
        the initial SASA score

    SASA_avg_fin10 : float
        The average SASA score over the last 10ns of the trajectory
    """
    traj = mdt.load(traj_fpath, top=top_fpath)
    traj_init = mdt.load(top_fpath)
    
    # grab protein atoms
    protein_atoms      = traj.topology.select("protein")
    protein_atoms_init = traj_init.topology.select("protein")

    traj = traj.atom_slice(protein_atoms)
    traj_init = traj_init.atom_slice(protein_atoms_init)
    _vdw_radii = get_martini_vdw_radii(top_fpath, martini_version="3.0")
    _vdw_radii = [x*10 for x in _vdw_radii] # must be in angstroms

    # set freesasa parameters
    # params = freesasa.Parameters(
    #     "algorithm": freesasa.ShrakeRupley,
    #     "probe-radius": 1.4, # in Angstroms
    #     "n-sphere-points": 960,
    # )
    params = freesasa.Parameters()
    params.setAlgorithm(freesasa.ShrakeRupley)
    params.setProbeRadius(1.4) # in Angstroms
    params.setNPoints(960)

    ######################
    # compute sasa for initial frame
    ######################
    xyz_init = traj_init.xyz[0] * 10 # convert nm to Angstroms
    # convert xyz to 3*N list of floats [x1, y1, z1, x2, y2, z2, ...]
    xyz_init_list = xyz_init.reshape(-1).tolist()

    sasa_init = freesasa.calcCoord(xyz_init_list, _vdw_radii, params)
    sasa_init_tot = sasa_init.totalArea()

    ######################
    # compute sasa for last 10ns of trajectory
    ######################
    xyz = traj.xyz[-(frames_per_ns*10):] * 10 # convert nm to Angstroms
    sasa_avg_fin10 = 0
    for frame in range(xyz.shape[0]):
        _xyz_frame = xyz[frame]
        _xyz_frame_list = _xyz_frame.reshape(-1).tolist()
        _sasa_frame = freesasa.calcCoord(_xyz_frame_list, _vdw_radii, params)
        sasa_avg_fin10 += _sasa_frame.totalArea()
    sasa_avg_fin10 /= xyz.shape[0]

    ap_score = sasa_init_tot/sasa_avg_fin10

    return ap_score, sasa_init_tot, sasa_avg_fin10

def compute_aggregation_propensity_sasa_mdt(top_fpath, traj_fpath, frames_per_ns=500):
    """
    expects all-atom sims
    compute_aggregation_propensity_sasa computes the SASA-based 
    aggregation propensity (AP) score using mdtraj.shrake_rupley. 
    The AP_sasa score originates from https://doi.org/10.1021/acs.jctc.3c00699

    Parameters:
    -----------
    top_fpath: 
        File path to topology file
    traj_fpath: Optional, 
        File path to trajectory file
    frames_per_ns : int
        How many simulation frames correspond to 1ns
        
    Returns:
    --------
    APsasa : float
        The aggregation propensity score
    
    SASA_init : float 
        the initial SASA score

    SASA_avg_fin10 : float
        The average SASA score over the last 10ns of the trajectory
    """
    traj = mdt.load(traj_fpath, top=top_fpath)
    traj_init = mdt.load(top_fpath)

    sasa_init = mdt.shrake_rupley(traj_init, mode="residue").sum()

    # average over time then sum the residue scores
    sasa_avg_fin10 = mdt.shrake_rupley(traj[-(frames_per_ns*10):], mode="residue").mean(axis=0).sum() 

    ap_score = sasa_init/sasa_avg_fin10

    return ap_score, sasa_init, sasa_avg_fin10

def _weight_distance(distance):
    """
    _weight_distance returns weights for varying distances
        distance <  4.0 angstrom: return 1.0
        distance > 12.0 angstrom: return 0.0
        between return exp(-(distance - 4))
    This weighting scheme, and the APcontact score it is
    used in, originate in the following paper

    Parameters:
    -----------
        distance: float or array
            Float (or Array) representing distance(s).
    
    Returns:
    --------
        weights: same shape as distances
            The weights corresponding to distances
    """

    d = np.asarray(distance)

    w = np.zeros_like(d, dtype=float)

    w[d < 4.0] = 1.0
    mid = (d >= 4.0) & (d <= 12.0)
    w[mid] = np.exp(-(d[mid] - 4.0))

    return w

def compute_aggregation_propensity_contact(top_fpath, traj_fpath, frames_per_ns=500, martini=False, seq_length=None):
    """
    Computes inter-peptide 'contact' derived aggregation propensity score, 
    as described in https://doi.org/10.1021/acs.jctc.3c00699.

    Essentially computes a weighted path through the inter-peptide distance matrix,
    where the weight is derived from the distance using the following scheme:
        distance <  4.0 angstrom: weight = 1.0
        distance > 12.0 angstrom: weight = 0.0
        between return exp(-(distance - 4))
        
    Parameters:
    -----------
        top_fpath: str|path
            file path to topology
        traj_fpath: str|path
            file path to trajectory
        frames_per_ns: int
            Number of frames per nanosecond. Default is 500
        martini: bool
            Whether or not analyzing a CG martini model
        seq_length: int | None
            Length of the peptide sequence (required if martini=True)
    
    Returns:
    --------
        aggregation_propensity_contact: float
            The inter-peptide 'contact' derived aggregation propensity score
    """

    if martini and seq_length is None:
        raise ValueError("seq_length must be provided if martini=True")

    traj      = mdt.load(traj_fpath, 
                         top=top_fpath)
    traj = traj[-(frames_per_ns*10):] # take last 10ns 

    traj_init = mdt.load(top_fpath)

    top = traj.topology
    chains = list(top.chains)
    n_chains = len(chains)
    n_frames = traj.n_frames

    # Pre-collect atom indices per chain (optionally heavy atoms only)
    if not martini:
        chain_atoms = [
            [a.index for a in chain.atoms if a.element.symbol != "H"]
            for chain in chains
        ]
    else:
        # for martini it has 1 chain for all proteins
        # so we use the seq length to grab residue objects, 
        # then grab all atoms in those residues
        n_chains = traj.n_residues // seq_length
        _residues = [x for x in traj.topology.residues]
        chain_atoms = []
        for i in range(n_chains):
            res_start = i * seq_length
            res_end = (i + 1) * seq_length
            residues = _residues[res_start:res_end]
            atoms = [a.index for r in residues for a in r.atoms]
            chain_atoms.append(atoms)

    ###################################################
    # compute distances between peptides using min atomic distances
    # (time, n_chains, n_chains)
    distance_matrix = np.zeros((n_frames, n_chains, n_chains), dtype=np.float32)

    for i, j in combinations(range(n_chains), 2):
        pairs = np.array([(ai, aj) for ai in chain_atoms[i] for aj in chain_atoms[j]], dtype=int)

        # (n_frames, n_pairs)
        d = mdt.compute_distances(traj, pairs)

        # per-frame minimum (n_frames,)
        min_d = d.min(axis=1).astype(np.float32)

        distance_matrix[:, i, j] = min_d
        distance_matrix[:, j, i] = min_d

    ###################################################
    # build 1 weighted path per peptide, over all frames
    ###################################################
    T, C, _ = distance_matrix.shape

    # prevent self-selection
    idx = np.arange(C)
    # dm[:, idx, idx] = np.inf

    # output: (time, starting_chain)
    paths = np.zeros((T, C), dtype=np.float32)

    for start in range(C):
        dm = distance_matrix.copy()

        dm[:, idx, idx] = np.inf

        current = np.full(T, start, dtype=int)

        # prevent revisiting the initial point
        dm[:, :, current] = np.inf
        
        acc = np.zeros(T, dtype=np.float32)
        for _ in range(C - 1):
            # select the row corresponding to the current peptide, for all frames
            rows = dm[np.arange(T), current, :]      # shape (T, C)

            mins = rows.min(axis=1)                   # (T,)
            next_idx = rows.argmin(axis=1)            # (T,)

            _w = _weight_distance(mins*10) # *10 to Ang --> nm
            acc += _w

            # # prevent revisiting selected peptides
            dm[np.arange(T)[:, None], np.arange(C)[None, :], next_idx[:, None]] = np.inf
            # dm[np.arange(T)[:, None], next_idx[:, None], np.arange(C)[None, :]] = np.inf

            current = next_idx

        paths[:, start] = acc / (C - 1)

    # max path at each frame
    _max_path_val = np.max(paths, axis=1) # (T,)

    # return average over time
    return np.mean(_max_path_val)

def compute_avg_max_cluster_size(top_fpath, traj_fpath, cutoff_distance=22.5,frames_per_ns=500):
    """
    Expects CG sims.
    Computes the average maximum cluster size of peptides in the trajectory.
    Clusters are defined using hierarchical clustering with a cutoff distance
    (default 22.5 Angstroms).

    Parameters:
    -----------
        top_fpath: str|path
            file path to topology
        traj_fpath: str|path
            file path to trajectory
        cutoff_distance: float
            distance cutoff for defining clusters (in Angstroms)
        frames_per_ns: int
            Number of frames per nanosecond. Default is 500
    """