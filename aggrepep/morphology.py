"""
A data-driven aggregation morphology characterization method, 
as per Statt, Kleeblatt, and Reinhart Soft Matter, 2021, 17, 7697–7707 (DOI: 10.1039/d1sm01012c)

Some of the functions in this file are adapted from original code provided by the authors of the above paper.

includes:
- GaussianHistogram
- SoftHistogram
- _coarsen_chains
    Default is to take each chain, find the COM of the first half of the chain and the
    second half of the chain. 
- _compute_threebody_features(p1, p2, p3)
    for three points p1,p2,p3 in 3D-space, compute
    d_jk: the distance between points j and k
    d_jk = |rk - rj| is the distance between neighbors 
    y_jik = arccos(rik \cdot rij) is the bond angle
    l_ik = d_ij + d_ik is the bond length
"""

import MDAnalysis as mda
from MDAnalysis.lib import distances, mdamath

import numpy as np
import torch
from torch import nn


def read_cg(top_filename, traj_filename, n_chains, frame=-1):
    """
    Assumes martini2 coarse-grained beads.
    Coarse-grain by splitting every chain into two contiguous halves,
    reducing each half to one bead.
    
    Assumes every chain has the same length `chain_len` and beads are
    ordered chain-by-chain (same layout the original stride code required).
    
    Parameters
    ----------
    top_filename : str
        path to topology file (e.g. .gro, .pdb) 
    traj_filename : str
        path to trajectory file (e.g. .xtc, .dcd)
    n_chains : nuumber of chains.
        used to compute chain length
                even L      -> two halves of L//2
                odd  L=2n+1 -> first half n beads, second half n+1 beads
    frame : int
        frame index to read from trajectory (default -1 = last frame)

    Returns
    -------
    xyz_cg : np.ndarray, shape (n_groups, 3)
        coarse-grained coordinates of the two halves of each chain
    box : np.ndarray, shape (6,)
        simulation box dimensions (xlo, xhi, ylo, yhi, zlo, zhi)
    types_cg : np.ndarray, shape (n_groups,)
        coarse-grained bead types (integer labels)
    """
    traj = mda.Universe(top_filename, traj_filename)
    traj.trajectory[int(frame)]              # -1 = last frame
    
    beads = traj.select_atoms('name BB SC1 SC2 SC3')
    xyz = beads.positions # Angstrom, MDAnalysis converts to angstrom when loading in 
    # xyz = xyz/10 # convert to nm
    _, types = np.unique(beads.types, return_inverse=True)   # beads, not all atoms
    box = traj.dimensions
    # box = box/10 # convert to nm
    
    f = distances.transform_RtoS(xyz, box)
    
    n_beads = len(f)
    chain_len = n_beads // n_chains
    if n_beads % chain_len != 0:
        raise ValueError(f"{n_beads} beads not divisible by chain_len={chain_len}")

    h1 = chain_len // 2          # first-half size  = n   (floor)
    h2 = chain_len - h1          # second-half size = n+1 (ceil); equal when even

    # per-bead group id: chain c -> group 2c (first h1 beads), group 2c+1 (next h2)
    pattern  = np.concatenate([np.zeros(h1, int), np.ones(h2, int)])
    group_id = np.tile(pattern, n_chains) + np.repeat(2 * np.arange(n_chains), chain_len)
    n_groups = 2 * n_chains

    # reference bead (first of each group) for minimum-image unwrapping
    ref = f[np.searchsorted(group_id, np.arange(n_groups))]
    disp = f - ref[group_id]
    disp -= np.round(disp)

    # sum displacements per group (no 1/N averaging, matching your original)
    f_cg = ref.copy()
    np.add.at(f_cg, group_id, disp)
    f_cg -= np.round(f_cg)
    xyz_cg = distances.transform_StoR(f_cg, box) + 0.5 * box[:3]

    # sum type codes per group, then relabel
    types_cg = np.zeros(n_groups, dtype=types.dtype)
    np.add.at(types_cg, group_id, types)
    _, types_cg = np.unique(types_cg, return_inverse=True)
    return xyz_cg, box, types_cg

def _coarsen_chains():
    pass

def _compute_threebody_features():
    pass


class GaussianHistogram(nn.Module):
    def __init__(self, bins, ranges, sigma, device='cpu'):
        super(GaussianHistogram, self).__init__()

        self.device = device

        # create sigma tensor of appropriate shape
        self.sigma = torch.tensor(sigma.reshape(3, 1, 1), dtype=torch.float, device=torch.device(device))

        # create bin vectors
        self.bins = bins
        rmin = torch.tensor(ranges[:, [0]], dtype=torch.float, device=torch.device(device))
        rmax = torch.tensor(ranges[:, [1]], dtype=torch.float, device=torch.device(device))
        delta = (rmax - rmin) / float(bins)
        self.centers = rmin + delta * (torch.arange(bins, device=torch.device(device)).float() + 0.5)
        self.delta = delta.reshape(3, 1, 1)

        # create centers grid
        self.xy = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            xv, yv = torch.meshgrid(self.centers[i], self.centers[j])
            xy = torch.vstack([xv.reshape(1, -1), yv.reshape(1, -1)])
            self.xy.append(torch.unsqueeze(xy, 2))

    def forward(self, x):
        eps = 1e-6

        # generate the histograms
        z = torch.zeros([3, self.bins ** 2], device=self.device)
        for k, ij in enumerate([(0, 1), (0, 2), (1, 2)]):
            # do the gaussian expansion
            y = torch.unsqueeze(x[ij, :], 1) - self.xy[k]
            y = torch.exp(-0.5 * (y / self.sigma[[ij]]) ** 2) / (self.sigma[[ij]] * np.sqrt(np.pi * 2)) * \
                self.delta[[ij]]
            y = y.prod(dim=0)
            z[k] = y.sum(dim=1)

        # normalize
        z /= torch.unsqueeze(z.sum(dim=-1) + eps, -1)

        return z

    def to(self, device, *args, **kwargs):
        super(GaussianHistogram, self).to(*args, **kwargs)
        self.sigma = self.sigma.to(device)
        self.delta = self.delta.to(device)
        self.xy = [x.to(device) for x in self.xy]

class SoftHistogram(nn.Module):
    def __init__(self, bins, ranges, sigma, vol_norm=False, device='cpu'):
        super(SoftHistogram, self).__init__()

        self.device = device

        # create sigma tensor of appropriate shape
        self.sigma = torch.tensor(sigma.reshape(3, 1, 1), dtype=torch.float, device=torch.device(device))

        # create bin vectors
        self.bins = bins
        rmin = torch.tensor(ranges[:, [0]], dtype=torch.float, device=torch.device(device))
        rmax = torch.tensor(ranges[:, [1]], dtype=torch.float, device=torch.device(device))
        delta = (rmax - rmin) / float(bins)
        self.centers = rmin + delta * (torch.arange(bins, device=torch.device(device)).float() + 0.5)
        self.delta = delta.reshape(3, 1, 1)

        # create centers grid
        self.xy = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            xv, yv = torch.meshgrid(self.centers[i], self.centers[j])
            xy = torch.vstack([xv.reshape(1, -1), yv.reshape(1, -1)])
            self.xy.append(torch.unsqueeze(xy, 2))

        v = 4 / 3 * np.pi * ((0.5 * self.centers[0] + 0.5 * self.delta[0, 0]) ** 3 -
                             (0.5 * self.centers[0] - 0.5 * self.delta[0, 0]) ** 3)
        if vol_norm:
            self.v = [torch.unsqueeze(v, 1) * v,
                      torch.unsqueeze(v, 1) * torch.ones_like(v),
                      torch.ones_like(torch.unsqueeze(v, 1)) * v]
        else:
            self.v = [torch.ones(bins**2) for _ in range(3)]

    def forward(self, x):
        eps = 1e-6

        u_idx = torch.triu_indices(x.shape[0], x.shape[0], +1)
        l_idx = torch.tril_indices(x.shape[0], x.shape[0], -1)
        idx = torch.hstack([u_idx, l_idx])

        # distance between, djk
        x_g = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
        d_mat = torch.sqrt(torch.pow(x_g, 2).sum(dim=2) + eps)  # avoid nan gradients
        dist_vec = d_mat[idx[0], idx[1]]

        # bond lengths, dik + dik
        x_l = torch.sqrt(torch.pow(x, 2).sum(dim=1) + eps)
        b_mat = torch.unsqueeze(x_l, 0) + torch.unsqueeze(x_l, 1)
        bond_vec = b_mat[idx[0], idx[1]]

        # bond angles
        x_n = x.norm(dim=1)[:, None]
        x_normed = x / torch.max(x_n, eps * torch.ones_like(x_n))
        a_mat = torch.mm(x_normed, x_normed.transpose(0, 1))
        a_vec = a_mat[idx[0], idx[1]]
        theta_vec = torch.acos(torch.clamp(a_vec, -1.0 + eps, 1.0 - eps))

        # concatenate the representations
        x = torch.vstack([dist_vec, bond_vec, theta_vec])

        # generate the histograms
        z = torch.zeros([3, self.bins ** 2], device=self.device)
        for k, ij in enumerate([(0, 1), (0, 2), (1, 2)]):
            # do the gaussian expansion
            y = torch.unsqueeze(x[ij, :], 1) - self.xy[k]
            y = torch.exp(-0.5 * (y / self.sigma[[ij]]) ** 2) / (self.sigma[[ij]] * np.sqrt(np.pi * 2)) * \
                self.delta[[ij]]
            y = y.prod(dim=0)
            y = y.sum(dim=1)
            z[k] = y * self.bins**3 / self.v[k].view(-1)

        # normalize
        z /= torch.unsqueeze(z.sum(dim=-1) + eps, -1)

        return z

    def to(self, device, *args, **kwargs):
        super(SoftHistogram, self).to(*args, **kwargs)
        self.sigma = self.sigma.to(device)
        self.delta = self.delta.to(device)
        self.xy = [x.to(device) for x in self.xy]
        self.v = [x.to(device) for x in self.v]

