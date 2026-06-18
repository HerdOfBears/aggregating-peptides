import json
import shutil
import subprocess
import sys
from pathlib import Path

import os
import pickle as pkl
import argparse
import logging

import torch
import pandas as pd
import numpy as np
import MDAnalysis as mda

from openmm.app import PDBFile
from aggrepep.openmm_helpers import npt_production_run

from aggrepep.bayesian_optimization import MultiFidelityBO_Wu2019KG
from aggrepep.generative_model import GenerativeModelWrapper

from aggrepep.coagulation_theory import analyze_aggregation_trajectory
from aggrepep.analysis import (
    compute_beta_content_score, 
    compute_aggregation_propensity_contact, 
    compute_aggregation_propensity_sasa,
    compute_aggregation_propensity_sasa_mdt
)

def args_parser():
    pass

def coarse_grained_pw_pathway(sequence, pep_id, out_dir, params, replica_id=1):

    shutil.copytree(params["force_field_path"], out_dir / "martini")
    params["replica_id"] = replica_id
    ##################################################
    # Step 1: build structure from sequence using sequence_to_structure.py    
    ###################################################
    _output_fmt = "pdb"
    sequence_to_conformation_script = Path("scripts/sequence_to_structure.py")
    subprocess.run([
        sys.executable,  # Use the current Python interpreter
        str(sequence_to_conformation_script),
        "--sequence", sequence,
        "--sequence_id", pep_id,
        "--arrangement", params["arrangement"],
        "--output_dir", str(out_dir),
    ], check=True)

    ###################################################
    # Step 2: set up the CG PW simulation
    ###################################################
    logging.info(f"Setting up CG PW simulation for peptide {pep_id} in {out_dir}")
    cg_pw_setup_script = Path("bash_scripts/coarse_grain_pw_setup.sh")
    _monomer_pdb = f"{pep_id}.pdb"
    _seq_length = len(sequence)
    subprocess.run([
        str(cg_pw_setup_script),
        _monomer_pdb,
        str(_seq_length),
        str(out_dir),
        "y" if params["neutral_nterminus"] else "n",
        "y" if params["neutral_cterminus"] else "n"
    ], check=True)

    ###################################################
    # step 3: run a simulation 
    ###################################################
    logging.info(f"Running simulation for {pep_id} (replica {replica_id})")
    run_script = Path("scripts/test_run_martini_openmm.py")
    subprocess.run([
        sys.executable,  # Use the current Python interpreter
        str(run_script),
        "--gro", str(Path(out_dir) / "solvated.gro"),
        "--top", str(Path(out_dir) / "system.top"),
        "--wdir", str(out_dir),
        "--pw",
        "--params_file", str(Path("params.json"))
    ], check=True)

    ###################################################
    # step 4: run analysis 
    ###################################################
    _universe = mda.Universe(str(Path(out_dir) / "solvated.gro"), str(Path(out_dir) / "prod.xtc"))
    
    aggregation_results = analyze_aggregation_trajectory(_universe, sequence, params=params)

    # with open(Path(out_dir) / "analysis_results.json", "w") as f:
    #     json.dump(aggregation_results, f, indent=4)

    return aggregation_results

def all_atom_pathway(sequence, pep_id, out_dir, params, replica_id=1):
    

    ##################################################
    # Step 1: build structure from sequence using sequence_to_structure.py    
    # ##################################################
    _output_fmt = "pdb"
    sequence_to_conformation_script = Path("scripts/sequence_to_structure.py")
    subprocess.run([
        sys.executable,  # Use the current Python interpreter
        str(sequence_to_conformation_script),
        "--sequence", sequence,
        "--sequence_id", pep_id,
        "--arrangement", params["arrangement"],
        "--output_dir", str(out_dir),
        "--build_stacked_sheets"
    ], check=True)

    ##########################################
    # Step 2: Run equilibation: energy min, nvt and npt equilibrations
    ##########################################
    randomSeed=42
    jobPrefix=f"{pep_id}_parallel"
    jobName=f"{jobPrefix}_rs{randomSeed}"
    wDir=out_dir
    inputFile=f"{jobPrefix}.pdb"

    script_equilibration = Path("scripts/run_equilibration.py")
    subprocess.run([
        sys.executable,  # Use the current Python interpreter
        str(script_equilibration),
        "--pdb_file", inputFile,
        "--input_dir", wDir,
        "--output_dir", wDir,
        "--job_name", jobName,
        "--platform_name", params["platform"],
        "--params_file", "params.json",
        "--random_seed", str(randomSeed)
    ])

    ##########################################
    # Step 3: Run production
    ##########################################
    jobName=f"{jobPrefix}_rs{randomSeed}_"
    inputFile=f"{jobName}solvated_system._pbcFixed.pdb"
    checkpointFile=f"{jobName}nvt_npt_equilibrated_system.xml"
    paramsFile="params.json"
    params["job_name"] = f"{jobPrefix}_rs{randomSeed}_"
    params["output_dir"] = out_dir
    params[ "input_dir"] = out_dir
    params["checkpoint_fname"] = checkpointFile # <--the important one for npt production run function

    ###############
    # load equilibrated system file
    equilibrated_system_pdb = PDBFile( os.path.join(params['input_dir'], inputFile) )

    logging.info(f"Starting npt production run...")
    npt_production_run(equilibrated_system_pdb, params)
    logging.info(f"finished npt production run.")

    # script_production = Path("scripts/run_production.py")
    # subprocess.run([
    #     sys.executable,  # Use the current Python interpreter
    #     str(script_production),
    #     "--input_pdb", inputFile,
    #     "--checkpoint_fname", checkpointFile,
    #     "--input_dir", wDir,
    #     "--output_dir", wDir,
    #     "--job_name", jobName,
    #     "--params_file", paramsFile,
    #     "--platform_name", params["platform"],
    #     "--random_seed", str(randomSeed)
    # ], check=True)

    ##########################################
    # Step 4: Run analysis
    ##########################################
    trajFile=f"{jobName}npt_production_system_traj.dcd"
    topFile=f"{jobName}solvated_system._pbcFixed.pdb"
    trajFileNW=f"{jobName}npt_production_system_traj_pbcFixed.dcd"
    
    script_fix_pbcs = Path("scripts/fix_pbcs.py")
    subprocess.run([
        sys.executable,  # Use the current Python interpreter
        str(script_fix_pbcs),
        "--input",  str(Path(wDir) / trajFile),
        "--output", str(Path(wDir) / trajFileNW),
        "--topology", str(Path(wDir) / topFile)
    ], check=True)

    # script_AP_analysis = Path("scripts/run_AP_analysis.py")
    # analysisResults = params["all_atom_analysis_results"]
    # subprocess.run([
    #     sys.executable,  # Use the current Python interpreter
    #     str(script_AP_analysis),
    #     "--pepid", pep_id,
    #     "--sequence", sequence,
    #     "--wdir", wDir,
    #     "--traj", trajFileNW,
    #     "--top", f"{pep_id}_parallel.pdb",
    #     "--rseed", str(randomSeed),
    #     "--dataset_file", analysisResults
    # ], check=True)

    _top_fpath  = str(Path(wDir) / f"{pep_id}_parallel.pdb")
    _traj_fpath = str(Path(wDir) / trajFileNW)
    frames_per_ns = 500 # assuming 2 ps between frames during all-atom sim

    beta_content_score  = compute_beta_content_score(             _top_fpath, _traj_fpath, frames_per_ns=frames_per_ns)
    agg_prop_score      = compute_aggregation_propensity_sasa_mdt(_top_fpath, _traj_fpath, frames_per_ns=frames_per_ns)
    agg_prop_contact    = compute_aggregation_propensity_contact( _top_fpath, _traj_fpath, frames_per_ns=frames_per_ns)

    results = {
        "sequence": sequence,
        "pep_id": pep_id,
        "method": "aa",
        "replica_id": replica_id,
        "beta_content_score": beta_content_score,
        "agg_prop_contact": agg_prop_contact,
        "agg_prop_score": agg_prop_score
    }

    return results

def encode_initial_data(df, model):
    _sequences = df.sequence.to_list()
    _fidelities= df.fidelity.to_numpy().reshape(-1,1)
    _scores    = df.score.to_numpy().reshape(-1,1)

    _initial_train_set = model.encode_sequences_to_latent(_sequences)      # (n_sequences, n_reduced_dim)
    train_x =  np.concatenate([_initial_train_set, _fidelities], axis=1)   # (n_sequences, n_reduced_dim+1)
    train_y = _scores

    return train_x, train_y

def main(params):
    
    # df = pd.read_csv(params.input_file)
    # pep_ids     = df["pep-id"]
    # sequences   = df["sequence"]
    pep_ids = ["driver-test-mj10"]
    sequences = ["KGSGSNITINITI"]
    params["accelerator"] = "OpenCL"

    USE_BAYES_OPTIMIZATION = params["USE_BAYES_OPTIMIZATION"]
    SMOKE_TEST             = params["SMOKE_TEST"]
    USE_AA = False
    N_ITERATIONS = params["bayesian_optimization"]["iterations"] #len(pep_ids)
    
    ################################################
    # load generative model
    ################################################
    if USE_BAYES_OPTIMIZATION:
        generative_model = GenerativeModelWrapper(params["generative_model"], params["bayesian_optimization"])

    # encode initial dataset
    if USE_BAYES_OPTIMIZATION:
        df = pd.read_csv(params["bayesian_optimization"]["initial_dataset_fpath"])
        train_x, train_y = encode_initial_data(df, generative_model)
        train_x, train_obj = torch.from_numpy(train_x), torch.from_numpy(train_y)


    ################################################
    # --- initialize BO object with some initial data (can be empty) ---
    # the BO object owns the dataset and the surrogate model, 
    # and implements the BayesOpt logic
    # (except scoring the candidates, which is done below)
    ################################################
    if USE_BAYES_OPTIMIZATION:
        # train_x = torch.rand((2,6))  # 2 initial candidates, 5 design dimensions + 1 fidelity dimension
        # train_x[0, -1] = 0.0  # low-fidelity candidate
        # train_x[1, -1] = 1.0  # high-fidelity candidate
        # train_obj = torch.tensor(
        #     [
        #         [25.82],  # low-fidelity score for candidate 1; mj10, SzalaMendyk2023 kf score
        #         [0.9878 + 1.1592]   # high-fidelity score for candidate 2 ; mj11 0.9878 + 1.1592 (APcontact + APsasa)
        #     ]
        # )
        logging.info(f"Initializing Bayesian Optimization with train_x: {train_x.shape} and train_obj: {train_obj.shape}")
        BayesOpt = MultiFidelityBO_Wu2019KG(train_x, train_obj, params={"PROBLEM_DIM": 5, "SMOKE_TEST": SMOKE_TEST, "BATCH_SIZE": 1})
        pep_id_prefix = "bo-peptide-"
        if SMOKE_TEST:
            pep_id_prefix = "smoke-test-peptide-"
    else:
        pep_id_prefix = "peptide-"


    ################################################
    # Main loop over peptides | Main Bayesian Optimization loop (if BO enabled)
    ################################################
    cumulative_cost = 0.0
    bayesopt_results={
        "sequences":[],
        "fidelity" :[],
        "score"    :[],
        "cumulative_cost":[],
    }

    for i in range(N_ITERATIONS):
        if USE_BAYES_OPTIMIZATION:
            new_latent_point, cost = BayesOpt.suggest()
            new_scores = torch.empty((new_latent_point.shape[0], 1), dtype=torch.double)  # placeholder for scores of the new candidates
            logging.info(f"{new_latent_point.shape} candidate(s) suggested by BO with cost {cost:.2f}")

            cumulative_cost += cost

            pep_id = f"{pep_id_prefix}{i}"
            # pep_id = pep_ids[i]
            # seq    = sequences[i]

            if new_latent_point[0,BayesOpt.fidelity_col].item() >= 0.5:
                USE_AA=True
                logging.info(f"BO iteration {i+1}/{N_ITERATIONS} | Suggested high-fidelity evaluation")
            else:
                USE_AA=False
                logging.info(f"BO iteration {i+1}/{N_ITERATIONS} | Suggested low-fidelity evaluation")
        else:
            pep_id = pep_ids[i]
            seq    = sequences[i]

        ################################################
        # Generate sequence from suggested latent point
        ################################################
        if USE_BAYES_OPTIMIZATION:
            _seq_latent_point = new_latent_point[:,:BayesOpt.fidelity_col]
            _fidelity         = new_latent_point[:, BayesOpt.fidelity_col]

            seq = generative_model.decode_latent_point(_seq_latent_point.float())
            if len(seq) == 1:
                seq = seq[0]
            
            if SMOKE_TEST:
                USE_AA=True
                seq = "KGNITINITI"
                print("forcing high-fidelity")
                logging.info("hardcoded forcing of high-fidelity")
                logging.info(f"Using hard-coded mj seq-variant as smoke test sequence=KGNITINITI")

            logging.info(f"BO iteration {i+1}/{N_ITERATIONS} | decoded sequence = {seq}.")

        # logging.info(f"Processing peptide {pep_id} with sequence {seq}")
        logging.info(f"iteration {i+1}/{N_ITERATIONS} | Processing peptide {pep_id} with sequence {seq}")
        _odir = Path(params['wdir']) / f"{pep_id}"
        if not _odir.exists():
            _odir.mkdir(parents=True)

        ################################################
        # run either the CG PW pathway or the all-atom pathway
        ################################################
        if USE_AA:
            logging.info(f"Running all-atom pathway for peptide {pep_id}")
            _params = params["all_atom_simulation"]

            _replica_id = 1
            _odir = _odir / "aa" / f"replica-{_replica_id}"
            os.makedirs(_odir, exist_ok=True)

            all_atom_results = all_atom_pathway(   seq, pep_id, _odir, _params, replica_id=_replica_id)

            new_scores = torch.tensor([[all_atom_results["agg_prop_score"][0] + all_atom_results["agg_prop_contact"]]], dtype=torch.double)

            with open(_odir / "aa_analysis_results.pkl", "wb") as f:
                pkl.dump(all_atom_results, f)
        else:
            logging.info(f"Running coarse-grained pathway for peptide {pep_id}")
            _params = params["coarse_grained_martini"]

            _replica_id = 1
            _odir = _odir / "cg" / f"replica-{_replica_id}"
            os.makedirs(_odir, exist_ok=True)

            cg_results = coarse_grained_pw_pathway(seq, pep_id, _odir, _params, replica_id=_replica_id)

            new_scores = torch.tensor([[cg_results["SzalaMendyk2023"]["kf"]]], dtype=torch.double)

            with open(_odir / "cg_analysis_results.pkl", "wb") as f:
                pkl.dump(cg_results, f)
        
        ################################################
        # if using BO, register new observation (candidate + score) and refit the model
        ################################################
        if USE_BAYES_OPTIMIZATION:
            logging.info("Registering new observation to BO and refitting model...")
            # remember: the last column of new_latent_point is the fidelity level (0 for low-fidelity, 1 for high-fidelity)
            BayesOpt.register_observations(new_latent_point, new_scores)

            bayesopt_results["sequences"].append( seq )
            bayesopt_results["fidelity" ].append( _fidelity )
            bayesopt_results["score"    ].append( new_scores.item() )
            bayesopt_results["cumulative_cost"].append( cumulative_cost )
            logging.info(f"BO iteration {i+1}/{N_ITERATIONS} | {seq=}, {_fidelity=},score={ new_scores.item() }, {cumulative_cost=}")
    
    if USE_BAYES_OPTIMIZATION:
        bo_jobname = params["bayesian_optimization"]["jobname"]
        output_fpath = f"{params['wdir']}/{bo_jobname}_bo_results.pkl"
        logging.info(f"writing BayesOpt results to {output_fpath}")
        with open(output_fpath, "wb") as fobj:
            pkl.dump(bayesopt_results, fobj)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", required=True, help="Path to input CSV file containing peptide IDs and sequences")
    parser.add_argument("--wdir", required=True, help="Working directory path")
    parser.add_argument("--params_file", required=False, default=None, help="Path to JSON file containing parameters. Optional as parameters have defaults.")
    parser.add_argument("--smoke_test", action="store_true", help="Whether to run in smoke test mode (overrides parameters to use smaller numbers for quick testing)")
    parser.add_argument("--use_bayes_optimization", action="store_true", help="Whether to use Bayesian Optimization to suggest peptide sequences, or just run through the input file sequentially")
    
    params = parser.parse_args()
    params = vars(params)
    
    params["accelerator"] = "CPU"
    if params["params_file"]:
        with open(params["params_file"]) as f:
            file_params = json.load(f)

            for k, v in file_params.items():
                params[k] = v

    params["SMOKE_TEST"] = False
    if params["smoke_test"]:
        params["SMOKE_TEST"] = True
        params["coarse_grained_martini"]["SMOKE_TEST"] = True
        params["coarse_grained_martini"]["total_npt_time"] = 2.0
        
        params["all_atom_simulation"]["SMOKE_TEST"] = True
        params["all_atom_simulation"]["total_npt_time"] = 2.0

    if params["use_bayes_optimization"]:
        params["USE_BAYES_OPTIMIZATION"] = True
    else:
        params["USE_BAYES_OPTIMIZATION"] = False

    smoke_test = params["SMOKE_TEST"]

    os.makedirs("logs/", exist_ok=True)
    n_log_files = len(os.listdir("logs/"))

    log_prefix  = f"{n_log_files:04}"
    if smoke_test:
        log_prefix = f"smoke_test_{log_prefix}"
    else:
        log_prefix = f"bayesopt_run_{log_prefix}"

    log_fname = f"{log_prefix}.log"
    logging.basicConfig(
        filename=os.path.join("logs/", log_fname),
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"=================== Starting main loop {smoke_test=}===================")
    logging.info(f"")

    main(params)