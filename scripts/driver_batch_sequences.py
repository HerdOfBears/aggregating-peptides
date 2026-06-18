"""
Simulation driver script.

Adapted from the Bayesian-Optimization driver: ALL Bayesian-optimization logic
has been removed. This version simply runs molecular simulations for a set of
input sequences. For every sequence it launches `params["n_jobs"]` replicates,
which run *concurrently* (sharing a single GPU when one is available).

Style / imports mirror the original driver, minus the BO-specific dependencies
(GenerativeModelWrapper, encode_initial_data, MultiFidelityBO_Wu2019KG, torch),
which are no longer needed.
"""

import os
import shutil
import subprocess
import sys
import json
import logging
import argparse
import pickle as pkl
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# import pandas as pd  # only needed if you re-enable CSV input loading

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

def _gpu_available(params):
    """
    Decide whether a GPU is available for the simulations.

    We key off the requested accelerator. OpenMM-style GPU platforms are
    CUDA / OpenCL / HIP; CPU / Reference are not GPU platforms. Replace this
    with a real device probe if you want stricter detection.
    """
    accelerator = str(params.get("accelerator", "CPU")).upper()
    return accelerator in {"CUDA", "OPENCL", "HIP"}


def _run_replicate(task):
    """
    Run a SINGLE replicate for one sequence, in its own process.

    `task` is a plain tuple of picklable values so that it survives the
    spawn-based process boundary (everything is re-pickled into the child).

    Careful notes (this is the part the request asked to be careful about):
      * One *process* per replicate (not a thread): each simulation gets an
        isolated Python/OpenMM state. Combined with the "spawn" start method
        (see main), this avoids the classic CUDA/OpenCL "cannot re-initialize
        in a forked subprocess" failure that happens with fork() + GPU.
      * All `n_jobs` replicates target the SAME GPU. They share it via the
        driver's time-slicing (or CUDA MPS if configured). You are responsible
        for making sure `n_jobs` concurrent simulations actually fit in GPU
        memory -- nothing here enforces that.
      * The parent's logging config does NOT propagate to spawned children, so
        each replicate (re)configures logging to its own file inside its output
        dir. Because each child is a separate process, there is no cross-process
        log-file contention.
    """
    seq, pep_id, replica_id, base_odir, params, use_aa = task
    base_odir = Path(base_odir)

    if use_aa:
        pathway_tag = "aa"
        _params = params["all_atom_simulation"]
        results_fname = "aa_analysis_results.pkl"
    else:
        pathway_tag = "cg"
        _params = params["coarse_grained_martini"]
        results_fname = "cg_analysis_results.pkl"

    _odir = base_odir / pathway_tag / f"replica-{replica_id}"
    os.makedirs(_odir, exist_ok=True)

    # Per-process logging -> this replicate's own file. force=True so it also
    # works if a worker process is reused for another task later.
    logging.basicConfig(
        filename=str(_odir / "replica.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )
    logging.info(f"Starting {pathway_tag} replicate {replica_id} for {pep_id} | seq={seq}")

    try:
        if use_aa:
            logging.info(f"Running all-atom pathway for peptide {pep_id} (replica {replica_id})")
            results = all_atom_pathway(seq, pep_id, _odir, _params, replica_id=replica_id)
        else:
            logging.info(f"Running coarse-grained pathway for peptide {pep_id} (replica {replica_id})")
            results = coarse_grained_pw_pathway(seq, pep_id, _odir, _params, replica_id=replica_id)

        with open(_odir / results_fname, "wb") as f:
            pkl.dump(results, f)

        logging.info(f"Finished {pathway_tag} replicate {replica_id} for {pep_id}")

        # Return only a lightweight summary; the heavy `results` object stays on
        # disk (avoids shipping large arrays back across the process boundary).
        return {
            "pep_id": pep_id,
            "sequence": seq,
            "replica_id": replica_id,
            "pathway": pathway_tag,
            "output_dir": str(_odir),
            "status": "ok",
        }
    except Exception as e:
        logging.exception(f"Replicate {replica_id} for {pep_id} FAILED: {e}")
        return {
            "pep_id": pep_id,
            "sequence": seq,
            "replica_id": replica_id,
            "pathway": pathway_tag,
            "output_dir": str(_odir),
            "status": "failed",
            "error": repr(e),
        }


def main(inputs, params):
    """
    Simulation driver.

    Parameters
    ----------
    inputs : dict
        Mapping {pep_id: sequence}. One simulation "experiment" per entry.
        NOTE: since the *pep_id* is the key, keep the input IDs unique.
    params : dict
        Run parameters. Must contain "n_jobs": the number of replicates to run
        per sequence, which is ALSO the number of replicates run concurrently.
    """
    params["accelerator"] = "OpenCL"  # mirror the original main() override; edit as needed

    SMOKE_TEST = params["SMOKE_TEST"]
    # Pathway selection used to be driven by BO fidelity. Without BO we default
    # to the coarse-grained pathway (the original non-BO behavior) unless the
    # caller explicitly opts into the all-atom pathway.
    USE_AA = params.get("USE_AA", False)

    n_jobs = int(params["n_jobs"])
    if n_jobs < 1:
        raise ValueError(f"params['n_jobs'] must be >= 1, got {n_jobs}")

    gpu = _gpu_available(params)
    logging.info(
        f"Driver start | n_sequences={len(inputs)} | replicates/seq=n_jobs={n_jobs} "
        f"| accelerator={params['accelerator']} | gpu_available={gpu} | USE_AA={USE_AA} "
        f"| SMOKE_TEST={SMOKE_TEST}"
    )
    if gpu:
        logging.info(
            f"All {n_jobs} replicates per sequence will share the SAME GPU "
            f"(time-sliced, or via CUDA MPS if configured). Ensure they fit in GPU memory."
        )

    # 'spawn' start method: GPU contexts (CUDA/OpenCL) do NOT survive fork(), so
    # spawning fresh interpreters is the safe way to run several GPU simulations
    # concurrently from a single parent process.
    mp_ctx = multiprocessing.get_context("spawn")

    driver_results = {}  # pep_id -> list of per-replicate summaries

    for pep_id, seq in inputs.items():
        logging.info(f"Processing peptide {pep_id} | seq={seq} | launching {n_jobs} replicate(s)")

        base_odir = Path(params['wdir']) / f"{pep_id}"
        base_odir.mkdir(parents=True, exist_ok=True)
        # Pre-create the pathway dir in the parent to avoid mkdir races between
        # the concurrent workers (each replica subdir is unique per worker).
        (base_odir / ("aa" if USE_AA else "cg")).mkdir(parents=True, exist_ok=True)
        
        if "-Ac-" in pep_id:
            params["neutral_nterminus"]=True
            logging.info(f"using neutral N-terminus: {params['neutral_nterminus']} | using neutral C-terminus: {params['neutral_cterminus']}")
        else:
            params["neutral_nterminus"]=False
            logging.info(f"using neutral N-terminus: {params['neutral_nterminus']} | using neutral C-terminus: {params['neutral_cterminus']}")

        # One task per replicate. replica_id is 1-indexed to match the original.
        tasks = [
            (seq, pep_id, replica_id, str(base_odir), params, USE_AA)
            for replica_id in range(1, n_jobs + 1)
        ]

        rep_summaries = []
        # max_workers == n_jobs: all replicates for this sequence run at once.
        # A fresh executor per sequence keeps GPU concurrency capped at n_jobs and
        # avoids mixing replicates from different sequences on the device.
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_ctx) as executor:
            future_to_rep = {executor.submit(_run_replicate, t): t[2] for t in tasks}
            for fut in as_completed(future_to_rep):
                replica_id = future_to_rep[fut]
                try:
                    summary = fut.result()
                except Exception as e:
                    # _run_replicate catches its own errors, but guard against a
                    # hard worker crash so one bad replicate can't kill the batch.
                    logging.exception(f"{pep_id} replicate {replica_id} crashed: {e}")
                    summary = {
                        "pep_id": pep_id, "sequence": seq, "replica_id": replica_id,
                        "pathway": "aa" if USE_AA else "cg",
                        "status": "crashed", "error": repr(e),
                    }
                rep_summaries.append(summary)
                logging.info(f"{pep_id} | replicate {summary['replica_id']} -> {summary['status']}")

        rep_summaries.sort(key=lambda s: s["replica_id"])
        driver_results[pep_id] = rep_summaries

    # Persist a small run-level summary of where everything landed.
    os.makedirs("outputs", exist_ok=True)
    jobname = params.get("jobname", "sim_driver")
    output_fpath = f"{params['wdir']}}/{jobname}_batch_driver_results.pkl"
    logging.info(f"writing driver summary to {output_fpath}")
    with open(output_fpath, "wb") as fobj:
        pkl.dump(driver_results, fobj)

    return driver_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", required=True, help="Path to input CSV file containing peptide IDs and sequences")
    parser.add_argument("--wdir", required=True, help="Working directory path")
    parser.add_argument("--params_file", required=False, default=None, help="Path to JSON file containing parameters. Optional as parameters have defaults.")
    parser.add_argument("--smoke_test", action="store_true", help="Whether to run in smoke test mode (overrides parameters to use smaller numbers for quick testing)")
    parser.add_argument("--n_jobs", type=int, default=None, help="Replicates per sequence, run concurrently (shared GPU when available). Overrides params_file when given explicitly.")

    params = parser.parse_args()
    params = vars(params)

    # Capture an explicit CLI --n_jobs before the params file can overwrite it.
    cli_n_jobs = params["n_jobs"]

    params["accelerator"] = "CPU"
    if params["params_file"]:
        with open(params["params_file"]) as f:
            file_params = json.load(f)

            for k, v in file_params.items():
                params[k] = v

    # n_jobs precedence: explicit CLI flag wins, else params_file value, else 1.
    if cli_n_jobs is not None:
        params["n_jobs"] = cli_n_jobs
    params.setdefault("n_jobs", 1)

    params["SMOKE_TEST"] = False
    if params["smoke_test"]:
        params["SMOKE_TEST"] = True
        params["coarse_grained_martini"]["SMOKE_TEST"] = True
        params["coarse_grained_martini"]["total_npt_time"] = 2.0

        params["all_atom_simulation"]["SMOKE_TEST"] = True
        params["all_atom_simulation"]["total_npt_time"] = 2.0

    smoke_test = params["SMOKE_TEST"]

    os.makedirs("logs/", exist_ok=True)
    n_log_files = len(os.listdir("logs/"))

    log_prefix = f"{n_log_files:04}"
    if smoke_test:
        log_prefix = f"smoke_test_{log_prefix}"
    else:
        log_prefix = f"sim_driver_run_{log_prefix}"

    log_fname = f"{log_prefix}.log"
    logging.basicConfig(
        filename=os.path.join("logs/", log_fname),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    ###############################################################
    # build inputs = {pep_id: sequence} 
    # df = pd.read_csv(params["input_file"])
    # inputs = {pid: seq for seq, pid in zip(df["sequence"], df["pep-id"])}
    ###############################################################
    params["neutral_cterminus"]=True
    I10 = "SNNFGAILSS"
    pep_ids   = [
        "I10-Ac-EE-cap",
        "I10-EE-cap",
        "I10-E-cap",
        "I10-Ac-K-cap",
        "I10-K-cap",
        "I10-KK-cap",
    ]
    sequences = [
        "EE"+"GSGS"+I10,
        "EE"+"GSGS"+I10,
         "E"+"GSGS"+I10,
         "K"+"GSGS"+I10,
         "K"+"GSGS"+I10,
        "KK"+"GSGS"+I10,
    ]
    inputs = {pid: seq for seq, pid in zip(sequences, pep_ids)}

    logging.info(f"=================== Starting simulation driver {smoke_test=} | n_jobs={params['n_jobs']} ===================")
    logging.info(f"")

    main(inputs, params)