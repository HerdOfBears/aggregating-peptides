# ruff: noqa: F405

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

from mdtraj.reporters import DCDReporter as mdtDCDReporter

import math
import os
import logging
import numpy as np
import mdtraj as md
import MDAnalysis as mda

AMINO_ACIDS_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
    'U': 'SEC',  # Selenocysteine
    'O': 'PYL'   # Pyrrolysine
}
AMINO_ACIDS_3to1 = {v:k for k, v in AMINO_ACIDS_1to3.items()}


def pdbfixer_workflow(fixer):
    """
    Standard PDBFixer workflow.
    """
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    return fixer

def add_distance_bw_groups_restraint():
    """
    Apply a harmonic potential between two groups of atoms. The potential
    is dependent on the distance between each atom group's centre-of-mass.


    """
    pass

# Function to add backbone position restraints
def add_backbone_posres(system, pdb, restraint_force, periodic_boundaries=True):
    if periodic_boundaries:
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    else:
        force = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')        

    force_amount = restraint_force * kilocalories_per_mole/angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    # restrain all Calpha atoms
    for atom in pdb.topology.atoms():
        _carbon_condn = (atom.name.startswith("C"))
        _nitro_condn  = (atom.name.startswith("N"))
        _heavy_condn  = (atom.name in ('CA', 'C', 'N') )
        if _carbon_condn or _nitro_condn or _heavy_condn: # all heavy atoms
            force.addParticle(atom.index, pdb.positions[atom.index])

    system.addForce(force)

def calculate_state_PE_and_maxForce(simulation:simulation.Simulation):
    """
    Calculates the system's potential energy (PE) and max force.

    Returns:
        pe: the potential energy of the system
        max_force: the maximum force exerted on an atom
    """
    _state = simulation.context.getState(getEnergy=True, getForces=True)
    pe = _state.getPotentialEnergy()
    pe = pe._value
    _forces = _state.getForces(asNumpy=True)
    max_force = np.max(np.linalg.norm(_forces, axis=1))
    return pe, max_force

def save_system_to_xml(system:System, fpath:str) -> None:
    """
    Saves the entire system to xml.
    """

    with open(fpath, 'w') as output:
        output.write(XmlSerializer.serialize(system))
    logging.info(f"Saved system to {fpath}")

def load_system_from_xml(fpath:str) -> System:
    """
    Loads an OpenMM System object from a given filepath.
    """
    with open(fpath, 'r') as _input:
        system = XmlSerializer.deserialize( _input.read() )
    logging.info(f"Loaded system from {fpath}")
    
    return system

def calculate_actual_ionic_strength(modeller):
    """
    Given modeller object after addSolvent has been run, 
    compute the actual ionic strength by counting the number of 
    ions and water molecules.
    """
    # Calculate actual ionic strength
    logging.info("Computing actual ionic strength")
    n_na = 0
    n_cl = 0
    n_water = 0

    for residue in modeller.topology.residues():
        if residue.name == 'HOH':
            n_water += 1
        elif residue.name in ['NA', 'Na+']:
            n_na += 1
        elif residue.name in ['CL', 'Cl-']:
            n_cl += 1

    logging.info(f"\nSolvent composition:")
    logging.info(f"  Water molecules: {n_water}")
    logging.info(f"  Na+ ions: {n_na}")
    logging.info(f"  Cl- ions: {n_cl}")

    # Calculate box volume
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        a, b, c = box_vectors
        # Volume = a · (b × c) for triclinic box
        volume_nm3 = np.abs(
            np.dot(  a.value_in_unit(nanometers), 
            np.cross(b.value_in_unit(nanometers), 
                     c.value_in_unit(nanometers)
                )
            )
        )
        volume_liters = volume_nm3 * 1e-24  # nm^3 to liters
        
        logging.info(f"\nBox dimensions:")
        logging.info(f"  Volume: {volume_nm3:.3f} nm^3 = {volume_liters:.6e} L")
        
        # Calculate ionic strength
        # I = 0.5 * Σ(c_i * z_i²) where c is concentration in M, z is charge
        # For NaCl: I = 0.5 * ([Na+] * 1^2 + [Cl-] * 1^2) = [NaCl]
        
        avogadro = 6.022e23
        n_ion_pairs = min(n_na, n_cl)  # Paired ions
        
        concentration_M = (n_ion_pairs / avogadro) / volume_liters
        ionic_strength_M = concentration_M  # For 1:1 electrolyte
        
        logging.info(f"\nIonic strength:")
        logging.info(f"  Concentration: {concentration_M:.3f} M")
        logging.info(f"  Ionic strength: {ionic_strength_M:.3f} M")
        
        # Check for excess ions (for neutralization)
        excess_na = n_na - n_cl
        excess_cl = n_cl - n_na
        if excess_na > 0:
            logging.info(f"  +{excess_na} excess Na+ (for neutralization)")
        elif excess_cl > 0:
            logging.info(f"  +{excess_cl} excess Cl- (for neutralization)")
    else:
        logging.info("No periodic box vectors found")

    logging.info("Done computing actual ionic strength")

def fix_pdb_periodic_boundaries_and_save(pdb_file, output_file=None, topology_file=None, protein_only=True):
    """
    Fix periodic boundary conditions for a solvated PDB file.
    
    Parameters
    ----------
    pdb_file : str
        Path to input PDB file
    output_file : str, optional
        Path to output PDB file. If None, will append '_fixed' to input filename
    topology_file: str, optional (but sometimes REQUIRED)
        Path to file to use as a topology for .dcd files.
    protein_only: bool, optional
        Whether or not to ignore solvant.
        
    Returns
    -------
    str
        Path to the output file
        
    Examples
    --------
    >>> fix_pdb_periodic_boundaries('system.pdb', 'system_fixed.pdb')
    >>> fix_pdb_periodic_boundaries('solvated.pdb', anchor_molecules='resname LIG')
    """

    # check if the topology file is only protein. 
    # If not filter and save a new, protein-only file, then
    # use that one.
    if protein_only:
        _top = mda.Universe(topology_file)
        _new_topology_file = topology_file.replace(".pdb", "_protein.pdb")
        _top.select_atoms("protein").write(_new_topology_file)
        topology_file = _new_topology_file
        # _t = md.load(topology_file)
        # if _t.n_atoms > 30_000:
            # protein_idx = _t.topology.select("protein")
            # _t = _t.atom_slice(protein_idx)

    # Load the PDB file
    _file_suffix = pdb_file.split(".")[-1]
    if _file_suffix == "dcd":
        traj = md.load(pdb_file, top=topology_file)        
    else:
        traj = md.load(pdb_file)
    
    if protein_only:
        print(f"n atoms in traj: {traj.n_atoms} (pre  solvent removal)")
        traj = traj.remove_solvent()
        print(f"n atoms in traj: {traj.n_atoms} (post solvent removal)")

    # Get anchor atom indices
    if traj.n_chains>9:
        # artefact of investigating specific antibody.
        logging.warning("fixing periodic boundary conditions may have failed. N of chains in topology > 6")
        
        anchor_atoms = [set(traj.topology.chain(i).atoms) for i in range(traj.n_chains) if i==16]
        traj = traj.image_molecules(anchor_molecules=anchor_atoms, 
                                    inplace=False, 
                                    make_whole=True)
    else:
        traj = traj.image_molecules(inplace=False, 
                                    make_whole=True)
    
    # Generate output filename if not provided
    if output_file is None:
        if pdb_file.endswith( _file_suffix ):
            output_file = pdb_file.replace(_file_suffix, f"_pbcFixed.{_file_suffix}")
        else:
            output_file = pdb_file + f"_pbcFixed.{_file_suffix}"
    elif output_file.endswith(f"_pbcFixed.{_file_suffix}") is False:
        if "." in output_file:
            output_file = output_file.replace(_file_suffix, f"_pbcFixed.{_file_suffix}")
        else:
            output_file += "_pbcFixed.pdb"
    
    # Save the fixed structure
    traj.save(output_file)
    
    logging.info(f"Fixed PDB PBCs: {pdb_file} saved to: {output_file}")
    return output_file

def change_protonation_single(topology:Topology, chain_id:str, res_id:int, res_1letter:str) -> list:
    """
    Changes the protonation of residue. For a given chain, residue ID, and 
    residue letter abbreviation, returns a list of None the length of all 
    amino acids in the PDB. At the index location of the target residue will
    be the variant name.

    Args:
    -----------
    topology: Topology
        The topology of your protein(s)
    pdb: PDBFile ----DEPRECATED----
        The protein whose amino acid/residue you want changed
    chain_name: str in ['antigen', 'light', 'heavy']
        The Chain on which the target residue lives. 
    res_id: int
        The chain-specific index of the target residue, starting from 1
    res_1letter: str
        The 1 letter abbreviation of the target residue. 

    Returns:
    --------
    variants: list
        [None]*len(protein in pdb), with the residue variant at the 
        target residue's index location.  
    """

    variants = [None]* topology.getNumResidues()
    
    chain_to_chainID = {
        "antigen":"A",
        "light":"B",
        "heavy":"C"
    }
    residue_variants = {
        "H":"HIP", # histidine
        # "D":"ASH", # aspartic acid
        # "E":"GLU", # glutamic acid
    } 
    # if res_1letter not in residue_variants:
    #     return variants
    
    # residue_variant = residue_variants[res_1letter]
    
    for chain in topology.chains():
        if chain.id == chain_id:
            # _res = list(chain.residues())[int(res_id)-1]
            _res = [ _r for _r in chain.residues() if int(_r.id)==int(res_id)][0]
            _res_name = AMINO_ACIDS_3to1[_res.name]
            if _res_name not in residue_variants:
                continue
            residue_variant = residue_variants[_res_name]

            print(f"obtaining variant protonation for residue {_res}, {_res.id=}, {_res.index=}, chain ID = {chain_id}")
            _res_idx = _res.index
            variants[_res_idx] = residue_variant

            # for _res in chain.residues():
            #     # residue ID corresponds to what number (starting from 1) the 
            #     # residue is at.
            #     if int(_res.id) == int(res_id):
            #         # check if titratable residue
            #         _res_name = AMINO_ACIDS_3to1[_res.name]
            #         if _res_name not in residue_variants:
            #             continue

            #         _res_idx = _res.index
            #         variants[_res_idx] = residue_variant

    return variants

def change_protonation_batch(topology:Topology, residues_to_change:list[tuple]):
    """
    Changes the protonation states to 'lower pH' state for a list of tuples

    Args:
    -----
    topology: Topology
        The topology of the protein(s) you are modelling.
    pdb: PDBFile DEPRECATED
        The protein you are simulating. Chain ID, residues numbers, etc, 
        must match those in residues_to_change.
    residues_to_change : list[tuple]
        [ (chain_id, residue_id, residue_name) ]
        residue_name can be either 1 or 3 letter representation of the 20 standard AAs

    Returns:
    variants: list
        A list of None, except at the positions where there are residue changes.
        In line with OpenMM modeller's addHydrogen() method.
    """
    logging.info("Changing protonation of batch of residues")
    variants = [None]* topology.getNumResidues()

    for _tup in residues_to_change:
        _chain_id, _residue_id, _residue_name = _tup
        if _residue_name in AMINO_ACIDS_3to1:
            _residue_name = AMINO_ACIDS_3to1[_residue_name]
        
        _variants_single = change_protonation_single(topology, _chain_id, _residue_id, _residue_name)
        
        # merge into the main variants
        for j in range(len(_variants_single)):
            if variants[j] is not None:
                continue
            variants[j] = _variants_single[j]

    counter = 0
    for j in range(len(variants)):
        if variants[j] is not None:
            counter+=1
    logging.info(f"Changing protonation of {counter} residues.")

    return variants

def npt_production_run(system_pdb:pdbfile.PDBFile, params:dict):
    """
    set up and run the unrestrained NPT simulation ('production run')
    """
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
    checkpoint_fname   = params["checkpoint_fname"]
    state_output_fname = params["job_name"] + "state_production.csv"
    dcd_fname          = params["job_name"] + "npt_production_system.dcd"
    dcd_traj_fname     = params["job_name"] + "npt_production_system_traj.dcd"
    if params is not None:
        total_n_production_time = params.get("total_npt_time", 1) # ns

        temperature = params.get("temperature", temperature) # human internal temperature in Kelvin
        pressure    = params.get("pressure", 1.0) # in bar
        step_size = params.get("time_step", step_size) 

        total_n_production_steps = int(math.ceil( (total_n_production_time*nanoseconds)/(step_size*femtoseconds) ) )

        box_padding = params.get("box_padding", box_padding)

        report_every_X_steps = params.get("output_frequency", report_every_X_steps) # every 1000 steps
        checkpoint_fname   = params.get("checkpoint_fname", checkpoint_fname)
        state_output_fname = params.get("state_output_fname", state_output_fname)
        dcd_fname          = params.get("dcd_fname", dcd_fname)
        dcd_traj_fname     = params.get("dcd_traj_fname",dcd_traj_fname)
        output_dir         = params.get("output_dir",output_dir)

        if "force_field" in params:
            forcefield_opt = params.get("force_field", forcefield_opt)
        elif "forcefield_opt" in params:
            forcefield_opt = params.get("forcefield_opt", forcefield_opt)
        else:
            forcefield_opt = forcefield_opt
        
        positive_ion = params.get("positive_ion", positive_ion)
        negative_ion = params.get("negative_ion", negative_ion)
        salt_concentration = params.get("salt_concentration", 0.0) # in moles/liter

        nonbonded_cutoff = params.get("nonbonded_cutoff", 1.0) # in nanometers
        energy_tolerance = params.get("energy_tolerance",energy_tolerance)

        platform_name = params["platform"]

    platform = Platform.getPlatformByName(platform_name)

    ##############
    # specify forcefield
    ##############
    forcefield = ForceField(f"{forcefield_opt}.xml", f"{forcefield_opt}/water.xml")

    ##############
    # setup system and integrator
    ##############
    modeller = Modeller(system_pdb.topology, system_pdb.positions)
    # modeller.deleteWater()
    # modeller.addSolvent(
    #         forcefield, 
    #         padding=box_padding, 
    #         positiveIon=positive_ion,
    #         negativeIon=negative_ion,
    #         neutralize=True
    #     )

    

    # _topology = modeller.topology#system_pdb.getTopology()
    _topology = system_pdb.getTopology()
    _positions= modeller.positions#system_pdb.getPositions()

    system = forcefield.createSystem(_topology, 
                                    nonbondedMethod=PME, 
                                    nonbondedCutoff=nonbonded_cutoff*nanometer, 
                                    constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(temperature*kelvin,
                                        1/picosecond, 
                                        step_size*femtoseconds
    )
    integrator.setRandomNumberSeed(params["random_seed"])

    _barostat = MonteCarloBarostat(pressure*bar, temperature*kelvin)
    _barostat.setRandomNumberSeed(params["random_seed"])
    system.addForce(
            _barostat
        )
    add_backbone_posres(system, system_pdb, 0.0, periodic_boundaries=True)                
    # simulation.context.reinitialize(preserveState=True) # reinitialize context with additional force

    simulation = Simulation(_topology, system, integrator, platform)


    if checkpoint_fname:
        logging.info(f"loading checkpoint: {checkpoint_fname}")
        
        if checkpoint_fname.endswith(".chk"):
            simulation.loadCheckpoint( os.path.join(params['output_dir'], checkpoint_fname) )
        elif checkpoint_fname.endswith(".xml"):
            simulation.loadState( os.path.join(params['output_dir'], checkpoint_fname) )

        simulation.context.reinitialize(preserveState=True)

        # make sure restraints are turned off.
        _k = simulation.context.getParameter("k")
        if _k != 0.0:
            raise ValueError(f"{_k=} should be equal to zero so that no restraints are applied.")

    else:

        pass

    _k = simulation.context.getParameter("k")
    if _k != 0.0:
        raise ValueError(f"{_k=} should be equal to zero so that no restraints are applied.")
    ##############
    # Reporters
    ##############
    # position reporter
    # simulation.reporters.append(
    #     DCDReporter( os.path.join(output_dir, dcd_traj_fname), report_every_X_steps)
    # )
    mdtraj_topology = md.Topology.from_openmm(_topology)
    protein_indices = mdtraj_topology.select('protein')

    _dcd_fpath = os.path.join(output_dir, dcd_traj_fname)
    if os.path.exists(_dcd_fpath):
        _append = True
    else:
        _append = False

    _dcd_reporter = DCDReporter(
        _dcd_fpath, 
        report_every_X_steps, 
        atomSubset=protein_indices,
        append= _append
    ) 
    

    simulation.reporters.append(
        _dcd_reporter
    )
    # state reporter 
    _state_fpath = os.path.join(output_dir, state_output_fname) 
    if os.path.exists(_state_fpath):
        _append=True
    else:
        _append=False

    _state_reporter = StateDataReporter(
        _state_fpath, 
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
        totalSteps=total_n_production_steps, 
        separator='\t',
        append=_append
    )
    simulation.reporters.append(_state_reporter)

    simulation.reporters.append(
        CheckpointReporter(
            os.path.join(output_dir, params["job_name"]+"npt_state.xml"),
            reportInterval = math.ceil( (10*nanosecond)/(step_size*femtoseconds) ),
            writeState=True
        )
    )

    logging.info(f"Taking NPT Production steps...")
    simulation.step(total_n_production_steps)
    logging.info(f"Done.")
