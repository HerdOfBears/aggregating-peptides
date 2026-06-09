# ruff: noqa: F405

from openmm.app import *
from openmm import *
from openmm.unit import *

import time
import os
import logging
import argparse
import datetime
import sys
import numpy as np
import json

from aggrepep.equilibration import run_equilibration


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str,  required=True, help='PDB file to simulate.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where the file exists.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where things will be saved")
    parser.add_argument("--job_name", type=str, required=True, help="short name to identify the 'job'")
    parser.add_argument("--platform_name", type=str, required=False, default="CPU", choices=["CPU","CUDA", "OpenCL"], help="run on 'CPU' or 'CUDA' or 'OpenCL'?")
    parser.add_argument("--params_file", type=str, required=False, default=None, help="Params file name residing in input_dir. Optional as parameters have defaults.")
    parser.add_argument("--random_seed", type=int, required=False, default=42, help="The random seed to set the integrator, barostat, and init velocities")
    args = parser.parse_args()

    params = {}
    params["nvt_equilibration_time"] = 50*picoseconds
    params["npt_equilibration_time"] = 50*picoseconds
    params["restraint_force_magnitude"] = 10.0

    params["random_seed"] = args.random_seed
    params["platform"] = args.platform_name # "CUDA", "OpenCL", "CPU"

    for i in range(Platform.getNumPlatforms()):
        platform = Platform.getPlatform(i)
        print(f"Platform {i}: {platform.getName()}")

    params["default_protonation"] = True

    params["checkpoint_fname"] = None#"energy_minimized_checkpoint.chk"
    params[ "input_dir"] = args.input_dir
    params["output_dir"] = args.output_dir 
    params["job_name"]   = args.job_name
    if params["job_name"][-1] != "_":
        params["job_name"] += "_"

    #####################################
    # if a json parameter file is provided, use the parameters from that.
    #####################################
    if args.params_file is not None:
        with open(args.params_file) as fobj:
            loaded_params_file = json.load(fobj)

        for k, v in loaded_params_file.items():
            params[k] = v
        params["params_file"] = args.params_file

    os.makedirs(params["output_dir"], exist_ok=True)

    pdb_file    = args.pdb_file
    prefix = ""
    log_dir = os.path.join(params["output_dir"], "logs/")
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.datetime.now()
    logfilename = f"{pdb_file}-sim.log"
    logging.basicConfig(
        filename= os.path.join(log_dir,logfilename),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt="%Y-%m-%d,%H:%M:%S",
        level=logging.INFO
    )
    logging.info(f"==================================== Job {pdb_file} ID  ====================================")
    logging.info(f"input arguments =\n{vars(args)}")
    logging.info("Starting equilibration...")
    
    pdb = PDBFile(
        os.path.join(params[ "input_dir"], pdb_file)
    )
    run_equilibration(pdb, params=params)    
