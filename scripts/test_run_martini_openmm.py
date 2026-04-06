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
    platform = Platform.getPlatformByName("OpenCL")
    properties = {'Precision': 'double'}
    SAVE_EVERY = 5_000 # steps

    _gro_fpath = os.path.join(test_dir,params["gro"]) if not os.path.isfile(params["gro"]) else params["gro"]
    _top_fpath = os.path.join(test_dir,params["top"]) if not os.path.isfile(params["top"]) else params["top"]
    conf = GromacsGroFile(
        _gro_fpath
    )
    box_vectors = conf.getPeriodicBoxVectors()
    
    top = martini.MartiniTopFile(
        _top_fpath,
        periodicBoxVectors=box_vectors,
        # defines=defines,
        epsilon_r=params["episilon_r"],
    )

    system = top.create_system()

    dt = 20 * femtosecond
    integrator = LangevinIntegrator(310 * kelvin,
                                    10.0 / picosecond,
                                    dt)
    integrator.setRandomNumberSeed(0)

    simulation = Simulation(
        top.topology, 
        system, 
        integrator,
        platform, 
    )

    simulation.context.setPositions(conf.getPositions())

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
                                                    totalSteps=math.ceil(100*nanosecond/dt)+math.ceil(2*nanosecond/dt)
        )
    )


    simulation.context.setVelocitiesToTemperature(310 * kelvin)
    logging.info('Running NVT equilibration...')
    simulation.step(
        math.ceil(1*nanosecond/dt) #1ns
    ) 

    system.addForce(MonteCarloBarostat(1 * bar, 310 * kelvin))
    # to update the simulation object to take in account the new system
    simulation.context.reinitialize(True)
    logging.info('Running NPT equilibration...')
    simulation.step(math.ceil(1*nanosecond/dt)) #1ns

    # save the trajectory in XTC format
    _xtc_fpath = os.path.join(test_dir, 'prod.xtc')
    xtc_reporter = XTCReporter(_xtc_fpath, SAVE_EVERY)
    simulation.reporters.append(xtc_reporter)

    # run simulation
    logging.info("Running simulation...")
    simulation.step(math.ceil(100*nanosecond/dt)) #25ns
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test run Martini OpenMM simulation")
    parser.add_argument("--gro", type=str, required=True, help="Path to the input GRO file")
    parser.add_argument("--top", type=str, required=True, help="Path to the input TOP file")
    parser.add_argument("--pw", action="store_true", help="Whether to run polarizable water simulation (if not set, will assume non-polarizable water simulation)")
    parser.add_argument("--wdir", type=str, default=".", help="Working directory for the test simulation")

    args = parser.parse_args()
    params=vars(args)

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