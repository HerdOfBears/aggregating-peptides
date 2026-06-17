import json
import logging

from openmm.unit import *
from openmm import *
from openmm.app import *
from mdtraj.reporters import XTCReporter
from sys import stdout

import martini_openmm as martini
import numpy as np
import os
import math
import argparse
import logging

def main(params):

    test_dir = params["wdir"]
    platform = Platform.getPlatformByName(
        params.get("platform","OpenCL")
    )
    properties = {'Precision': 'double'}

    SAVE_EVERY = params.get("output_frequency", 5000)
    RANDOM_SEED = params.get("random_seed", 42)
    temperature= params.get("temperature", 310) # in Kelvin
    pressure   = params.get("pressure", 1.0) # in bar
    step_size  = params.get("time_step", 20) # in fs
    total_npt_time = params.get("total_npt_time", 100) # in ns

    _gro_fpath = os.path.join(test_dir,params["gro"]) if not os.path.isfile(params["gro"]) else params["gro"]
    _top_fpath = os.path.join(test_dir,params["top"]) if not os.path.isfile(params["top"]) else params["top"]
    conf = GromacsGroFile(
        _gro_fpath
    )
    box_vectors = conf.getPeriodicBoxVectors()
    
    logging.info(f"Loading topology from {_top_fpath} with box vectors {box_vectors}")
    top = martini.MartiniTopFile(
        _top_fpath,
        periodicBoxVectors=box_vectors,
        # defines=defines,
        epsilon_r=params["episilon_r"],
    )

    system = top.create_system()

    dt = step_size * femtosecond
    integrator = LangevinIntegrator(temperature * kelvin,
                                    10.0 / picosecond,
                                    dt)
    integrator.setRandomNumberSeed(RANDOM_SEED)

    logging.info(f"Creating Simulation object")
    simulation = Simulation(
        top.topology, 
        system, 
        integrator,
        platform, 
    )

    simulation.context.setPositions(conf.getPositions())

    logging.info(f"Minimizing energy...")
    _states = simulation.context.getState(getEnergy=True, getForces=True)
    for i in range(10):
        energiesI = _states.getPotentialEnergy()
        maxForceI = np.linalg.norm(_states.getForces().max())
        
        simulation.minimizeEnergy(maxIterations=100,tolerance=1.0)
        
        energiesF = _states.getPotentialEnergy()
        maxForceF = np.linalg.norm(_states.getForces().max())

        logging.info(f"{energiesI=} to {energiesF=} | {maxForceI=} to {maxForceF=}")
    
    _state_fpath = os.path.join(test_dir, "state.csv")
    simulation.reporters.append(StateDataReporter(_state_fpath, SAVE_EVERY,
                                                    step=True,
                                                    potentialEnergy=True,
                                                    totalEnergy=True,
                                                    density=True,
                                                    temperature=True,
                                                    volume=True,
                                                    speed=True,
                                                    remainingTime=True,
                                                    totalSteps=math.ceil(total_npt_time*nanosecond/dt)+math.ceil(2*nanosecond/dt)
        )
    )


    simulation.context.setVelocitiesToTemperature(temperature * kelvin)
    logging.info('Running NVT equilibration...')
    simulation.step(
        math.ceil(1*nanosecond/dt) #1ns
    ) 

    system.addForce(MonteCarloBarostat(pressure * bar, temperature * kelvin))
    # to update the simulation object to take in account the new system
    simulation.context.reinitialize(True)
    logging.info('Running NPT equilibration...')
    simulation.step(math.ceil(1*nanosecond/dt)) #1ns

    # save the trajectory in XTC format
    _xtc_fpath = os.path.join(test_dir, 'prod.xtc')
    xtc_reporter = XTCReporter(_xtc_fpath, SAVE_EVERY)
    simulation.reporters.append(xtc_reporter)

    # run simulation
    logging.info(f"Running simulation for {total_npt_time} ns...")
    checkpoint_interval=min(100, total_npt_time) # in ns
    _previous_time_interval = 0
    for _step in range( int(total_npt_time//checkpoint_interval) ):
        _npt_time_interval = int( checkpoint_interval )

        simulation.step( math.ceil( _npt_time_interval*nanosecond/dt) )

        simulation.saveState(      os.path.join(test_dir,f"prod_{_previous_time_interval}to{_npt_time_interval*(_step+1)}ns.state") )
        simulation.saveCheckpoint( os.path.join(test_dir,f"prod_{_previous_time_interval}to{_npt_time_interval*(_step+1)}ns.chk") )

        _previous_time_interval += _npt_time_interval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test run Martini OpenMM simulation")
    parser.add_argument("--gro", type=str, required=True, help="Path to the input GRO file")
    parser.add_argument("--top", type=str, required=True, help="Path to the input TOP file")
    parser.add_argument("--pw", action="store_true", help="Whether to run polarizable water simulation (if not set, will assume non-polarizable water simulation)")
    parser.add_argument("--wdir", type=str, default=".", help="Working directory for the test simulation")
    parser.add_argument("--params_file", type=str, required=False, default=None, help="Path to JSON file containing simulation parameters. Optional as parameters have defaults.")

    args = parser.parse_args()
    params=vars(args)
    if args.params_file is not None:
        with open(args.params_file) as fobj:
            loaded_params_file = json.load(fobj)

        if "coarse_grained_martini" in loaded_params_file:
            loaded_params_file = loaded_params_file["coarse_grained_martini"]

        for k, v in loaded_params_file.items():
            params[k] = v
        params["params_file"] = args.params_file

    if params["pw"]:
        params["episilon_r"] = 2.5
    else:
        params["episilon_r"] = 15

    # set up logging file
    logging.basicConfig(
        filename=os.path.join(params["wdir"], "run_martini_openmm.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main(params)