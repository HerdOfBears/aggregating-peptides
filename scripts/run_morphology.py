
import torch
import umap
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib import distances, mdamath

import os
import pickle as pkl
import tqdm

from aggrepep.morphology import SoftHistogram, GaussianHistogram, read_cg

if __name__=="__main__":
    params = {}
    params["n_chains"] = 64
    params["experiment_dir"] = "../outputs/mj-Ceq45p2-neutral_term/"

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = torch.device(device)
    print(device)

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

    model = SoftHistogram(bins, ranges, sigma, device=device)
    model.to(device)


    # xyz_cg, box, types_cg = read_cg(
    #     "../outputs/mj-pt2-Ceq45p2-neutral_term/replica-1/solvated.gro",
    #     "../outputs/mj-pt2-Ceq45p2-neutral_term/replica-1/prod.xtc",
    #     64
    # )


    # target_dir = "../outputs/mj-pt2-Ceq45p2-neutral_term/"
    # target_dir = "../outputs/mj-Ceq45p2-neutral_term/mj8/cg/"
    experiment_dir = params["experiment_dir"] 
    gsd_files = []
    for _dir in os.listdir(experiment_dir):
        if not os.path.isdir(os.path.join(experiment_dir, _dir)):
            continue
        _replica_dir = os.path.join(experiment_dir, f"{_dir}/cg/replica-1/")

        _top_file = os.path.join(_replica_dir, "solvated.gro")
        _traj_file = os.path.join(_replica_dir, "prod.xtc")
        if not os.path.isfile(_top_file) or not os.path.isfile(_traj_file):
            continue
        gsd_files.append((_top_file, _traj_file))
    
    print(f"found {len(gsd_files)} gsd files in {experiment_dir}")

    traj = mda.Universe(gsd_files[0][0], gsd_files[0][1])

    timesteps = np.arange(len(traj.trajectory))
    # timesteps = [-1]

    ##################################################
    ##################################################
    # Calculate local neighbourhood histograms for each bead
    ##################################################
    ##################################################

    bins = 12  # don't change this!
    n_chains=params["n_chains"]

    for j, top_and_traj_files in enumerate(gsd_files):
        _top_file = top_and_traj_files[0]
        _traj_file= top_and_traj_files[1]
        output_fname = _top_file.replace(".gro", "_Hist.pkl")

        if os.path.isfile(output_fname.replace('.gsd', '_Hist.pkl')):
            print(f"skipping {output_fname} since it already exists")
            continue

        print('processing {:} of {:}, {:s}'.format(j+1, len(gsd_files), _top_file))
        xyz, box, types = read_cg(_top_file, _traj_file, n_chains, frame=0)
        all_H = np.zeros([len(timesteps), len(xyz), n_species * bins, 3 * bins])

        complete = True
        for ts, timestep in tqdm.tqdm(enumerate(timesteps), total=len(timesteps)):
            try:
                xyz, box, types = read_cg(_top_file, _traj_file, n_chains, frame=ts)
            except:
                complete = False
                break
            pairs, dists = distances.self_capped_distance(xyz, cutoff, box=box)

            def neighborhood(i):  # define neighborhood fetching function
                idx = np.argwhere(pairs == i)[:, 0]
                loc = np.unique(pairs[idx])
                f = distances.transform_RtoS(xyz[loc] - xyz[i], box)
                f -= np.round(f)
                r = distances.transform_StoR(f, box)
                t = types[loc]
                return r, t.reshape(-1, 1)

            # process the histograms
            idx = np.arange(xyz.shape[0])
            for k, i in enumerate(idx):
                r, t = neighborhood(i)
                data = torch.tensor(r, dtype=torch.float, requires_grad=False).to(device)
                output = model(data)
                z = output.to('cpu').detach().numpy()
                H = np.hstack([z[y, :].reshape(bins, bins) for y in range(3)])
                
                # shape of all_H is
                # (n_timesteps, n_species*n_bins, bins, 3*bins)
                all_H[ts, k] = H

        if complete:
            print(f"writing {output_fname}")
            with open(output_fname, 'wb') as fid:
                pkl.dump(all_H, fid)

    # reminder
    # all_H shape (n_timesteps, n_species*n_bins, bins, 3*bins)

    ##################################################
    ##################################################
    # Embed individual histograms into UMAP space 
    ##################################################
    ##################################################
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=10, 
        min_dist=0., 
        random_state=0, 
        verbose=False
    )

    local_time_step_idx = -1
    Z_local = reducer.fit_transform(
        all_H[local_time_step_idx,:].reshape(2*n_chains, -1)
    )

    min_embd, max_embd = np.array([np.inf]*3), np.array([-np.inf]*3)
    print(f"embedding all {len(all_H)} timesteps into UMAP space")
    print("reducer.fit_transform(), but might be wrong")
    for j, top_and_traj_files in enumerate(gsd_files):
        all_U = []

        _top_file = top_and_traj_files[0]
        _traj_file= top_and_traj_files[1]
        output_fname = _top_file.replace(".gro", "_Hist.pkl")
        output_embedding_fname = _top_file.replace(".gro", "_Hist_all_Zlocal.pkl")

        if not os.path.isfile(output_fname.replace('.gsd', '_Hist.pkl')):
            print(f"skipping {output_fname} since it does not exist")
            continue
        if os.path.isfile(output_embedding_fname):
            print(f"skipping {output_embedding_fname} since it already exists")
            continue
        
        for i, H in tqdm.tqdm(enumerate(all_H), total=len(all_H)):
            
            embedding = reducer.fit_transform(H.reshape(H.shape[0], -1))
            
            all_U.append(embedding)
            for j,_v in enumerate(np.min(embedding,axis=0)):
                if _v<min_embd[j]:
                    min_embd[j]=_v
            for j,_v in enumerate(np.max(embedding,axis=0)):
                if _v>max_embd[j]:
                    max_embd[j]=_v

        with open(output_embedding_fname, 'wb') as fid:
            pkl.dump(all_U, fid)
    
    print(min_embd)
    print(max_embd)
    if os.path.isfile(os.path.join(experiment_dir, "min_max_embd.pkl")):
        # load
        with open(os.path.join(experiment_dir, "min_max_embd.pkl"), 'rb') as fid:
            min_embd, max_embd = pkl.load(fid)
    else:
        # save
        with open(os.path.join(experiment_dir, "min_max_embd.pkl"), 'wb') as fid:
            pkl.dump((min_embd, max_embd), fid)

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

        for lam in all_U:

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

    with open(os.path.join(experiment_dir, "frame_embedding.pkl"), 'wb') as fid:
        pkl.dump(frame_embedding, fid)
