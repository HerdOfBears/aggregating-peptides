# ruff: noqa: F405

from openmm.app import *
from openmm import *
from openmm.unit import *

import sys
from .openmm_helpers import save_system_to_xml, \
    calculate_state_PE_and_maxForce, \
    calculate_actual_ionic_strength, \
    fix_pdb_periodic_boundaries_and_save, \
    add_backbone_posres, \
    change_protonation_batch

import time
import os
import logging
import argparse
import datetime
import numpy as np


def solvent_free_energy_minimization(pdb, params=None):
    """
    Perform energy minization on just the protein(s).
    No water molecules or ions are added. 
    """
    default_protonation = True
    temperature = 309.15 # human internal temperature in Kelvin
    step_size=2
    box_padding = 1.0*nanometer
    using_pbc = True # use periodic boundary conditions
    restrain_backbone = True # restrain the backbone during NVT and NPT equilibration

    forcefield_opt = "charmm36_2024"

    energy_tolerance = 10 # default of openmm

    output_dir = params["output_dir"]
    checkpoint_fname   = None
    state_output_fname = params["job_name"]+"state_preSolvent_eq.csv"
    if params is not None:
        temperature = params.get("temperature", temperature) # human internal temperature in Kelvin
        step_size = params.get("step_size", step_size) 
        
        box_padding = params.get("box_padding", box_padding)

        checkpoint_fname   = params.get("checkpoint_fname", checkpoint_fname)
        state_output_fname = params.get("state_output_fname", state_output_fname)
        output_dir         = params.get("output_dir",output_dir)

        forcefield_opt = params.get("forcefield_opt", forcefield_opt)

        energy_tolerance = params.get("energy_tolerance",energy_tolerance)

        platform_name = params["platform"]

    platform = Platform.getPlatformByName(platform_name)

    logging.info(f"{temperature=}, {step_size=} (femtoseconds), {using_pbc=}, {restrain_backbone=}")

    logging.info(f"Using {forcefield_opt=}")

    ##############
    # specify forcefield
    ##############
    forcefield = ForceField(f"{forcefield_opt}.xml", f"{forcefield_opt}/water.xml")

    ##############
    # build modeller object
    # delete crystallized water, may not be needed
    ##############
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()

    # if not default_protonation:
    #     pass
    # else:
    #     residues = modeller.addHydrogens(forcefield)

    # fix_pdb_periodic_boundaries_and_save(solvated_system_fpath)


    ##############
    # setup system and integrator
    ##############
    system = forcefield.createSystem(modeller.topology, 
                                    nonbondedMethod=PME, 
                                    nonbondedCutoff=1.0*nanometer, 
                                    constraints=HBonds
    )

    integrator = LangevinMiddleIntegrator(temperature*kelvin,
                                        1/picosecond, 
                                        step_size*femtoseconds
    )

    simulation = Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)

    ##############
    # local energy minimization
    ############## 
    logging.info("Minimizing energy...")

    _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
    logging.info(f"Pre  energy min: PE={_pe} kJ/mol and maxForce={_max_force}")

    simulation.minimizeEnergy(maxIterations=10_000)

    _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
    logging.info(f"Post energy min: PE={_pe} kJ/mol and maxForce={_max_force}")

    _output_fname = params['input_pdb'].replace(".pdb", "protein_min.pdb")
    _protein_energy_min_fpath = os.path.join( params['output_dir'], _output_fname)
    logging.info(f"Saving protein only minimization to: {_protein_energy_min_fpath}")
    PDBFile.writeFile(
        modeller.topology, 
        simulation.context.getState(getPositions=True).getPositions(), 
        open( _protein_energy_min_fpath,"w"), 
        keepIds=True
    )
    logging.info("Done.")

def staged_soft_energy_minimization(simulation:simulation.Simulation):
    """
    Performs up to four stages of 'soft' energy minimization.
    Tries to bring the potential energy below 0 and the 
    maximum force value below 15,000

    When a system has *very* high potential energy and force values, 
    we have to perform energy minimization with high thresholds 
    then progressively lower the threshold value. 

    Parameters
    ----------
    simulation: openmm.app.simulation.Simulation
        The system whose energy we want to minimize.

    Returns
    -------
    None
        Acts 'in-place' on the simulation.

    Notes:
    - This may not always succeed in sufficiently minimizing the energy.
    """
    _pe, _max_force = calculate_state_PE_and_maxForce(simulation)

    _stage_to_energy_tolerance = {
        "stage1": 10_000,
        "stage2":  1_000,
        "stage3":    100,
        "stage4":     10,
        "stage5":      5
    }
    _stage = "stage1"
    _max_iterations = 500

    _condn_pe = (_pe < 0 )
    _condn_max_force = (_max_force < 5_000)
    if (not _condn_pe) or (not _condn_max_force):
        logging.info(f"PE and maxForce still quite high. Doing soft energy minimization")
        logging.info(f"PE={_pe} kJ/mol and maxForce={_max_force}")

        _counter = 0
        while ((not _condn_pe) or (not _condn_max_force)):
            logging.info(f"staged soft energy minimization at stage={_stage}")
            t0 = time.time()
            energy_tolerance = _stage_to_energy_tolerance[_stage]

            simulation.minimizeEnergy(
                tolerance=energy_tolerance*(kilojoules_per_mole/nanometer),
                maxIterations=_max_iterations
            )

            logging.info(f"softer energy minimization duration: {round(time.time() - t0)}s per run of .minimizeEnergy ")

            _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
            logging.info(f"PE = {_pe} (kJ/mol), maxForce = {_max_force}")

            _condn_pe = (_pe < 0)
            _condn_max_force = (_max_force < 2_000)

            _counter += 1
            if _counter in [2,3]:
                _stage = "stage2"
            elif _counter in [4,5]:
                _max_iterations = 500
                _stage = "stage3"
            elif _counter in [6,7]:
                _max_iterations = 1000
                _stage = "stage4"
            elif _counter in [8,9]:
                _max_iterations = 1500
                _stage = "stage5"
            elif _counter > 9:
                logging.info(f"forceably leaving energy minimization loop {_counter=} > 8")
                break

def run_equilibration(pdb, params=None):
    """
    Runs NVT and NPT equilibration for a given PDBFile object . 
    Saves the equilibrated checkpoint file, and also a dcd
    """
    nvt_equilibration_time = 100*picoseconds
    npt_equilibration_time = 100*picoseconds
    temperature = 309.15 # human internal temperature in Kelvin
    step_size = 2 # in femtoseconds 

    box_padding = 1.0*nanometer
    using_pbc = True # use periodic boundary conditions
    restrain_backbone = True # restrain the backbone during NVT and NPT equilibration

    forcefield_opt = "charmm36_2024"
    add_solvent = True
    positive_ion = "Na+"
    negative_ion = "Cl-"

    energy_tolerance = 10 # default of openmm

    report_every_X_steps = 1000 # every 1000 steps
    output_dir = "./"
    checkpoint_fname   = None
    state_output_fname = params["job_name"]+"state_equilibration.csv"
    dcd_fname          = params["job_name"]+"nvt_npt_equilibrated_system.dcd"
    dcd_traj_fname     = params["job_name"]+"nvt_npt_equilibrated_system_traj.dcd"
    if params is not None:
        nvt_equilibration_time = params.get("nvt_equilibration_time", nvt_equilibration_time)
        npt_equilibration_time = params.get("npt_equilibration_time", npt_equilibration_time)
        temperature = params.get("temperature", temperature) # human internal temperature in Kelvin
        step_size = params.get("step_size", step_size) 
        
        box_padding = params.get("box_padding", box_padding)

        if not isinstance(box_padding, Quantity):
            box_padding = box_padding*nanometer

        report_every_X_steps = params.get("report_every_X_steps", report_every_X_steps) # every 1000 steps
        checkpoint_fname   = params.get("checkpoint_fname", checkpoint_fname)
        state_output_fname = params.get("state_output_fname", state_output_fname)
        dcd_fname          = params.get("dcd_fname", dcd_fname)
        dcd_traj_fname     = params.get("dcd_traj_fname",dcd_traj_fname)
        output_dir         = params.get("output_dir",output_dir)

        forcefield_opt = params.get("forcefield_opt", forcefield_opt)
        positive_ion = params.get("positive_ion", positive_ion)
        negative_ion = params.get("negative_ion", negative_ion)
        salt_concentration = params.get("salt_concentration", 0.0) # in moles/liter

        nonbonded_cutoff = params.get("nonbonded_cutoff", 1.0) # in nanometers

        energy_tolerance = params.get("energy_tolerance",energy_tolerance)

        default_protonation = params.get("default_protonation", True)

        platform_name = params["platform"]

    platform = Platform.getPlatformByName(platform_name)

    total_nvt_equilibriation_steps = int(nvt_equilibration_time / (step_size*femtoseconds) ) 
    total_npt_equilibriation_steps = int(npt_equilibration_time / (step_size*femtoseconds) ) 
    total_n_equilibriation_steps = int( (nvt_equilibration_time+npt_equilibration_time) / (step_size*femtoseconds) ) 
    logging.info(f"{temperature=}, {step_size=} (femtoseconds), {using_pbc=}, {restrain_backbone=}")
    logging.info(f"{add_solvent=}")
    logging.info(f"IONS: {positive_ion=}, {negative_ion=}, ionicStrength assumed 0, but will try neutralization")

    logging.info(f"total equilibriation time (NVT+NPT)    = {nvt_equilibration_time+npt_equilibration_time}")
    logging.info(f"total_n_equilibriation_steps (NVT, NPT) = {total_nvt_equilibriation_steps}, {total_npt_equilibriation_steps}")
    logging.info(f"total_n_equilibriation_steps (NVT+NPT) = {total_n_equilibriation_steps}")

    logging.info(f"Using {forcefield_opt=} and water = {forcefield_opt}/water")

    ##############
    # specify forcefield
    ##############
    forcefield = ForceField(f"{forcefield_opt}.xml", f"{forcefield_opt}/water.xml")

    ##############
    # build modeller object
    # delete crystallized water and add hydrogens, may not be needed
    ##############
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    # if not default_protonation:
    #     _target_residue_chain = ""
    #     _target_residue_id    = ""
    #     _target_residue_1char = ""
    #     _variants = change_protonation_batch(
    #         modeller.topology, 
    #         _target_residue_chain, 
    #         _target_residue_id, 
    #         _target_residue_1char
    #     )
    #     logging.info(
    #         f"Changing protonation of chain, resID, resName: {_target_residue_chain}, {_target_residue_id}, {_target_residue_1char}"
    #     )
    #     logging.info("Assuming we want the lower pH protonation.")
    #     residues = modeller.addHydrogens(forcefield, variants=_variants)
    # else:
    #     residues = modeller.addHydrogens(forcefield)

    ##############
    # add solvent (neutralizes the system)
    ##############
    if add_solvent:
        if isinstance(box_padding, Quantity):
            _condn = box_padding.value_in_unit(nanometer)<0
        else:
            _condn = box_padding < 0
        


        if _condn:
            logging.info(f"Using the box dimensions found in the topology: {modeller.topology.getUnitCellDimensions()}")
            modeller.addSolvent(
                forcefield,
                boxVectors=pdb.topology.getPeriodicBoxVectors(),
                positiveIon=positive_ion,
                negativeIon=negative_ion,
                neutralize=True,
                ionicStrength=salt_concentration*molar
            )
        else:
            modeller.addSolvent(
                forcefield, 
                padding=box_padding, 
                positiveIon=positive_ion,
                negativeIon=negative_ion,
                neutralize=True,
                ionicStrength=salt_concentration*molar
            )

        # calculate actual ionic strenght post neutralization
        calculate_actual_ionic_strength(modeller)

        solvated_system_fpath = os.path.join(
            params["output_dir"], 
            params["job_name"] + "solvated_system.pdb"
        )
        logging.info(f"Saving initial, solvated system to: {solvated_system_fpath}")
        PDBFile.writeFile(
            modeller.topology, 
            modeller.positions, 
            open(solvated_system_fpath,"w"), 
            keepIds=True
        )
        fix_pdb_periodic_boundaries_and_save(solvated_system_fpath, protein_only=False)


    ##############
    # setup system and integrator
    ##############
    system = forcefield.createSystem(modeller.topology, 
                                    nonbondedMethod=PME, 
                                    nonbondedCutoff=nonbonded_cutoff*nanometer, 
                                    constraints=HBonds
    )
    _system_output_fpath = os.path.join(params['output_dir'], params["job_name"]+"system.xml")
    save_system_to_xml(system, _system_output_fpath)

    integrator = LangevinMiddleIntegrator(temperature*kelvin,
                                        1/picosecond, 
                                        step_size*femtoseconds
    )
    integrator.setRandomNumberSeed(params["random_seed"])

    simulation = Simulation(modeller.topology, system, integrator, platform)

    if checkpoint_fname:
        logging.info(f"Loading checkpoint: {checkpoint_fname}")
        simulation.loadCheckpoint( os.path.join(params[ "output_dir"], checkpoint_fname) )
    else:
        simulation.context.setPositions(modeller.positions)

        ##############
        # local energy minimization
        ############## 
        logging.info("Minimizing energy...")

        _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
        logging.info(f"Initial PE={_pe} kJ/mol and maxForce={_max_force}")

        _condn_pe = (_pe < 0 )
        _condn_max_force = (_max_force < 150_000)
        if _condn_pe and _condn_max_force:
            simulation.minimizeEnergy(tolerance=energy_tolerance*(kilojoule/(nanometer*mole)))

            _pe, _max_force = calculate_state_PE_and_maxForce(simulation)

        else:# (not _condn_pe) and (not _condn_max_force):
            t0_overall = time.time()

            # perform longer, but softer energy minimization
            staged_soft_energy_minimization(simulation)    

            tf_overall = time.time()
            logging.info(f"Performed staged soft energy minimization in {round(tf_overall - t0_overall, 4)}s")

            _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
            logging.info(f"Post staged soft energy minimization PE={_pe} kJ/mol and maxForce={_max_force}")

        if _pe>0 or _max_force>15_000:
            raise ValueError(f"potential energy or maxForce still too high; {_pe=}, {_max_force=}")

        ##############
        # Save positions and a checkpoint of the system post energy minimization.
        ##############
        logging.info(f"Saving a checkpoint")
        positions = simulation.context.getState(getPositions=True).getPositions()

        # Save as pdb
        _post_energy_min_fpath = os.path.join(
            params["output_dir"], 
            params["job_name"] + 'system_post_energy_min.pdb'
        )
        PDBFile.writeFile(
            simulation.topology, 
            positions, 
            open(_post_energy_min_fpath, 'w')
        )

        simulation.saveCheckpoint( 
            os.path.join(
                params["output_dir"],
                params["job_name"] + 'energy_minimized_checkpoint.chk'
            ) 
        )
            
        logging.info(f"Post energy min PE={_pe} kJ/mol and maxForce={_max_force}")
    
    ##############
    # Reporters
    ##############
    # position reporter
    simulation.reporters.append(
        DCDReporter( os.path.join(output_dir,dcd_traj_fname), report_every_X_steps)
    )

    # state reporter 
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(output_dir, state_output_fname), 
            report_every_X_steps, 
            step=True, 
            time=True, 
            potentialEnergy=True, 
            temperature=True,
            volume=True, 
            density=True, 
            progress=True,
            remainingTime=True, 
            speed=True, 
            totalSteps=total_n_equilibriation_steps, 
            separator='\t'
        )
    )

    logging.info(f"Setting velocities with {temperature}kelvin, and random seed = {params['random_seed']}")
    simulation.context.setVelocitiesToTemperature(temperature*kelvin, params["random_seed"])

    ##############
    # Restrain protein backbone
    ##############
    if restrain_backbone:
        logging.info("Restrain protein backbone...")
        add_backbone_posres(system, pdb, params["restraint_force_magnitude"], periodic_boundaries=using_pbc)                
        simulation.context.reinitialize(preserveState=True) # reinitialize context with additional force

    ##############
    # run NVT equilibriation
    ##############
    logging.info(f"Running NVT equilibriation for {total_nvt_equilibriation_steps} steps...")
    simulation.step(total_nvt_equilibriation_steps)

    ##############
    # run NPT equilibriation
    ##############
    logging.info(f"Running NPT equilibriation for {total_npt_equilibriation_steps} steps...")
    _barostat = MonteCarloBarostat(1*bar, temperature*kelvin)
    _barostat.setRandomNumberSeed(params["random_seed"])
    system.addForce(
            _barostat
        )
    simulation.context.reinitialize(preserveState=True) # reinitialize context with additional force

    simulation.step(total_npt_equilibriation_steps)

    ##############
    # remove restraint if there is one
    ##############
    if restrain_backbone:
        logging.info("Removing backbone restraint...")
        simulation.context.setParameter('k', 0.0) # remove restraint

    ##############
    # save
    ##############
    _dcd_fpath = os.path.join(
        output_dir, 
        f"{dcd_fname.split('.')[0]}.pdb"
    )
    logging.info(f"Writing final positions of atoms to {_dcd_fpath}")
    PDBFile.writeFile(
        modeller.topology, 
        simulation.context.getState(getPositions=True).getPositions(), 
        open(_dcd_fpath,"w"), 
        keepIds=True
    )

    logging.info(f"Saving system state to {_dcd_fpath.split('.')[0]}.xml")
    simulation.saveState(
        _dcd_fpath.split(".")[0]+".xml"
    )

    # Save as binary checkpoint (compact)
    # _checkpoint_fpath = os.path.join(
    #     output_dir, 
    #     params["job_name"] + f"{checkpoint_fname.split('.')[0]}-post-nvt-npt-equil.chk")
    # simulation.saveCheckpoint(_checkpoint_fpath)
    # logging.info(f"Saved binary checkpoint to '{_checkpoint_fpath}'")

    logging.info("Done NVT/NPT equilibration")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str,  required=True, help='PDB file to simulate.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where the file exists.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where things will be saved")
    parser.add_argument("--job_name", type=str, required=True, help="short name to identify the 'job'")
    parser.add_argument("--platform_name", type=str, required=False, default="CPU", choices=["CPU","CUDA", "OpenCL"], help="run on 'CPU' or 'CUDA' or 'OpenCL'?")
    args = parser.parse_args()

    params = {}
    params["nvt_equilibration_time"] = 50*picoseconds
    params["npt_equilibration_time"] = 50*picoseconds
    params["restraint_force_magnitude"] = 10.0

    params["platform"] = args.platform_name # "CUDA", "OpenCL", "CPU"

    for i in range(Platform.getNumPlatforms()):
        platform = Platform.getPlatform(i)
        print(f"Platform {i}: {platform.getName()}")

    params["default_protonation"] = True

    params["checkpoint_fname"] = None#"energy_minimized_checkpoint.chk"
    params[ "input_dir"] = args.input_dir
    params["output_dir"] = args.output_dir 
    params["job_name"]   = args.job_name

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
