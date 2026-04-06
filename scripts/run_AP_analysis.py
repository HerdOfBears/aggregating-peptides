from aggrepep.analysis import (
    compute_beta_content_score, 
    compute_aggregation_propensity_contact, 
    compute_aggregation_propensity_sasa,
    compute_aggregation_propensity_sasa_mdt
)

import os
import argparse
import json
import pickle as pkl
import numpy as np

def main(params):
    wdir = params["wdir"]
    traj_fpath = os.path.join(wdir, params["traj"])
    top_fpath  = os.path.join(wdir, params["top"])
    
    if params['cg']:
        frames_per_ns = 10.0 # assuming 100 ps between frames during CG
    else:
        frames_per_ns = 500 
    print(f"Warning: frames_per_ns is set to {frames_per_ns}. Adjust if your trajectory has a different frame rate.")
    
    if params['cg']:
        beta_content_score  = -1
        agg_prop_score      = compute_aggregation_propensity_sasa(    top_fpath, traj_fpath, frames_per_ns=frames_per_ns)
    else:
        beta_content_score  = compute_beta_content_score(             top_fpath, traj_fpath, frames_per_ns=frames_per_ns)
        agg_prop_score      = compute_aggregation_propensity_sasa_mdt(top_fpath, traj_fpath, frames_per_ns=frames_per_ns)
    
    agg_prop_contact    = compute_aggregation_propensity_contact(top_fpath, traj_fpath, frames_per_ns=frames_per_ns)
    

    results = {
        "beta_content_score": beta_content_score,
        "agg_prop_contact": agg_prop_contact,
        "agg_prop_score": agg_prop_score
    }

    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence", required=True,
                        help="The sequence of the peptide.")
    parser.add_argument("--pepid", required=True, 
                        help="ID for the peptide")
    parser.add_argument("--wdir", required=True,
                        help="Working directory path. Either the path to pepid's parent dir or to pepid's dir" )
    parser.add_argument("--traj", required=False, default="prod.xtc",
                        help="Trajectory file name (default: prod.xtc)")
    parser.add_argument("--top", required=False, default="solvated.gro",
                        help="Topology file name (default: solvated.gro)")
    parser.add_argument("--rseed", default=42, required=False,
                        help="Random number seed")
    parser.add_argument("--dataset_file", required=False, type=str,
                        help="Path to file where analysis results can be appended.",
                        default=None)
    parser.add_argument("--cg", action="store_true", help="Whether the inputted top+traj are coarse-grained models or not.")
    args = parser.parse_args()
    params = vars(args)


    if params['pepid'] not in params['wdir']:
        params['wdir'] = os.path.join(params['wdir'], params['pepid'])

    results = main(params)

    # save results to pkl
    if params["cg"]:
        analysis_prefix = "cg_"    
    else:
        analysis_prefix = "aa_"

    with open(os.path.join(params['wdir'], f"{analysis_prefix}analysis_results.pkl"), "wb") as f:
        pkl.dump(results, f)

    if params["dataset_file"]:
        payload = f"{params['pepid']},{params['sequence']},{analysis_prefix[:2]},{results['beta_content_score']},{results['agg_prop_contact']},{results['agg_prop_score']}\n"
        with open(params['dataset_file'], 'a') as fobj:
            fobj.write(payload)
    