"""
Compiles local umaps from different experiment output directories into
a single global umap embedding. 
use like
python scripts/run_compile_morphology.py -exptD /path/to/experiment1 -exptD /path/to/experiment2 -exptD ...
"""
import torch
import umap
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib import distances, mdamath

import os
import pickle as pkl
import tqdm
import argparse
import time

from aggrepep.morphology import SoftHistogram, GaussianHistogram, read_cg


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exptD","--experiment_dir", 
                        action="append", required=True, 
                        help="The dir(s) where simulation output dirs are stored. Each subdir should contain a 'cg' dir with 'replica_X' dirs with solvated.gro and prod.xtc files.")
    parser.add_argument("--out_dir", type=str, default=".", help="The dir where the compiled umap embedding will be saved.")
    parser.add_argument("--out_name", type=str, default="frame_embedding.pkl", help="The name of the output file for the compiled umap embedding.")
    parser.add_argument("--n_chains", type=int, default=64)
    args = parser.parse_args()
    params = vars(args)

    experiment_dirs = params["experiment_dir"]
    only_last_frame = True # of each replica

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = torch.device(device)
    print(device)

    t0 = time.time()

    n_cg = 10

    reinhart_cutoff = 6 # 2 times the cutoff of their LJ interaction; as per: DOI: 10.1039/d1sm01012c
    our_cutoff = 2.2    # 2 times the non-bonded interaction cutoff for martini_openmm
    our_cutoff = our_cutoff*10 # convert to Angstrom since MDAnalysis uses angstrom
    cutoff = our_cutoff
    print(f"using r_cut = {cutoff} Ang as local neighbourhood radius")

    n_species = 1
    bins = 12
    res = [bins*1.5]*3
    ranges = np.array([[0, 2 * cutoff], [0, 2 * cutoff], [0, np.pi]])
    sigma = np.array([ranges[0, 1]/res[0], ranges[1, 1]/res[1], ranges[2, 1]/res[2]])

    gsd_files = []
    expt_ids = []
    for experiment_dir in experiment_dirs:
        for _dir in os.listdir(experiment_dir):
            if "ipynb" in _dir: 
                continue
            if not os.path.isdir(os.path.join(experiment_dir, _dir)):
                continue
            
            for _replica in os.listdir(os.path.join(experiment_dir, f"{_dir}/cg/")):
                _replica_dir = os.path.join(experiment_dir, f"{_dir}/cg/{_replica}/")
                if not os.path.isdir(_replica_dir):
                    continue
                
                _top_file  = os.path.join(_replica_dir, "solvated.gro")
                _traj_file = os.path.join(_replica_dir, "prod.xtc")
                if not os.path.isfile(_top_file) or not os.path.isfile(_traj_file):
                    continue
                gsd_files.append((_top_file, _traj_file))
                expt_ids.append(_dir)
    print(f"found {len(gsd_files)} gsd files in {experiment_dirs}")

    traj = mda.Universe(gsd_files[0][0], gsd_files[0][1])

    timesteps = np.arange(len(traj.trajectory))
    # timesteps = [-1]

    bins = 12  # don't change this!
    n_chains=params["n_chains"]

    # reminder
    # all_H shape (n_timesteps, n_species*n_bins, bins, 3*bins)
    min_embd, max_embd = np.array([np.inf]*3), np.array([-np.inf]*3)
    for experiment_dir in experiment_dirs:
        if os.path.isfile(os.path.join(experiment_dir, "min_max_embd.pkl")):
            # load
            with open(os.path.join(experiment_dir, "min_max_embd.pkl"), 'rb') as fid:
                _min_embd, _max_embd = pkl.load(fid)

            for i in range(3):
                if _min_embd[i] < min_embd[i]:
                    min_embd[i] = _min_embd[i]
                if _max_embd[i] > max_embd[i]:
                    max_embd[i] = _max_embd[i]

    ##################################################
    ##################################################
    # Make and Embed the global histograms into a UMAP space
    ##################################################
    ##################################################
    print("making and embedding the global histograms for each timestep")
    super_reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=16, 
        min_dist=1, 
        random_state=0, 
        verbose=False
    )

    ###############################################
    # make the global histograms for each timestep
    ###############################################
    print("make the global histograms for each timestep")
    hbins = 36
    res = [hbins*0.5]*3
    # ranges = np.vstack([embedding.min(axis=0), embedding.max(axis=0)]).T

    # from reinhart:
    # ranges = np.array([[ 2.193795 , 10.049596 ], [ 2.736186 , 10.067611 ], [ 6.3948064,  9.304425 ]])

    # 
    ranges = np.vstack([min_embd, max_embd]).T
    sigma = np.array([ranges[0, 1]/res[0], ranges[1, 1]/res[1], ranges[2, 1]/res[2]])

    gh = GaussianHistogram(hbins, ranges, sigma, device=device)
    gh.to(device)

    fingerprints = []
    for j, top_and_traj_files in enumerate(gsd_files):
        _top_file = top_and_traj_files[0]
        _traj_file= top_and_traj_files[1]
        output_fname = _top_file.replace(".gro", "_Hist.pkl")
        output_embedding_fname = _top_file.replace(".gro", "_Hist_all_Zlocal.pkl")

        if not os.path.isfile(output_embedding_fname):
            print(f"skipping {output_embedding_fname} since it does not exist")
            continue
        
        with open(output_embedding_fname, 'rb') as fid:
            all_U = pkl.load(fid)

        if only_last_frame:
            print(f"embedding only the last frame of {output_embedding_fname}")
            lam = all_U[-1]
            X = torch.tensor(lam.T, device=device)
            yh = gh(X).to('cpu').detach().numpy()
            yh = [y.reshape(hbins, hbins).T for y in yh]
            yh = np.hstack([np.flipud(y) / y.sum() for y in yh])

            fingerprints.append(yh)
        else:
            for i,lam in enumerate(all_U):
                X = torch.tensor(lam.T, device=device)
                yh = gh(X).to('cpu').detach().numpy()
                yh = [y.reshape(hbins, hbins).T for y in yh]
                yh = np.hstack([np.flipud(y) / y.sum() for y in yh])

                fingerprints.append(yh)

    ###############################################
    # Embed the global histograms for each timestep
    ###############################################
    print("embedding the global histograms for each timestep")
    frame_embedding = super_reducer.fit_transform(
        np.array(fingerprints).reshape(len(fingerprints), -1)
    )

    if only_last_frame:
        print(f"writing {os.path.join(args.out_dir, args.out_name)}")
        results = {
            "frame_embedding": frame_embedding,
            "expt_ids": expt_ids
        }
        with open(os.path.join(args.out_dir, args.out_name), 'wb') as fid:
            pkl.dump(results, fid)
    else:
        results = {
            "frame_embedding": frame_embedding,
            "expt_ids": expt_ids,
            "timesteps": timesteps
        }
        with open(os.path.join(args.out_dir, args.out_name), 'wb') as fid:
            pkl.dump(results, fid)

    tf = time.time()
    print(f"total time taken: {tf-t0:.2f} seconds")
    print(f"done! wrote {os.path.join(args.out_dir, args.out_name)}")