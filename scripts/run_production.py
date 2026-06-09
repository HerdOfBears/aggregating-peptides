# ruff: noqa: F405

from openmm.app import *
from openmm import *
from openmm.unit import *

from aggrepep.openmm_helpers import npt_production_run

import time
import os
import logging
import argparse
import datetime
import sys
import numpy as np
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb", required=True, help="the system to run in NPT ensemble. Must be solvated and equilibrated.")
    parser.add_argument("--input_dir", required=True, help="The dir where the above file resides")
    parser.add_argument("--checkpoint_fname", required=True, help="The equilibrated state.")
    parser.add_argument("--output_dir", required=True, help="The dir where files will be saved")
    parser.add_argument("--job_name", type=str, required=True, help="short name to identify the 'job'")
    parser.add_argument("--platform_name", type=str, required=False, default="CPU", choices=["CPU","CUDA", "OpenCL"], help="run on 'CPU' or 'CUDA' or 'OpenCL'?")
    parser.add_argument("--params_file", type=str, required=False, default=None, help="Params file name residing in input_dir. Optional as parameters have defaults.")
    parser.add_argument("--random_seed", type=int, required=False, default=42, help="The random seed to set the integrator, barostat, and init velocities")

    args = parser.parse_args()
    params={}

    for k,v in vars(args).items():
        params[k]=v
    params['npt_production_time'] = 5
    pdb_file = params['input_pdb']
    # params["output_dir"] = args.output_dir
    params["platform"] = "OpenCL"

    params['job_name'] = args.job_name
    if params["job_name"][-1] != "_":
        params["job_name"] += "_"

    # params["checkpoint_fname"] = "energy_minimized_checkpoint-post-nvt-npt-equil.chk"
    params["checkpoint_fname"] = params['job_name']+"nvt_npt_equilibrated_system.xml"
    params["checkpoint_fname"] = args.checkpoint_fname

    #####################################
    # if a json parameter file is provided, use the parameters from that.
    #####################################
    if args.params_file is not None:
        with open(args.params_file) as fobj:
            loaded_params_file = json.load(fobj)
            if "all_atom_simulation" in loaded_params_file:
                loaded_params_file = loaded_params_file["all_atom_simulation"]

        for k, v in loaded_params_file.items():
            params[k] = v
        params["params_file"] = args.params_file

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
    logging.info(f"==================================== Job {pdb_file} ID test ====================================")
    logging.info(f"input arguments =\n{params}")

    ###############
    # load equilibrated system file
    equilibrated_system_pdb = PDBFile( os.path.join(params['input_dir'], params['input_pdb']) )

    npt_production_run(equilibrated_system_pdb, params)
