import Bio.PDB
import PeptideBuilder
from PeptideBuilder import Geometry

import os
import time
import numpy as np
from io import StringIO, BytesIO

from aggrepep.helpers import biopython_to_pdbfixer_stringio, pdbfixer_workflow, rotate_around_length_axis, align_pc1_to_x_backbone

import mdtraj as mdt
from aggrepep.helpers import stack_chains as hstack_chains
from aggrepep.helpers import build_sandwich
from aggrepep.hydrophobicity import check_hydrophobic_burial
from aggrepep.minimization import setup_and_minimize
import logging
import argparse

import openmm as omm
import openmm.app as app
from openmm.app import PDBFile, Modeller
from openmm.unit import *

from aggrepep.helpers import check_for_overlap

def setup_rectangular_box(topology, positions, 
                          paddingFibreAxis, 
                          paddingPerpToFibreAxis,
                         fibreAxis="z"):
    """
    Set up a rectangular box for a stacked protein complex.
    
    Parameters:
    -----------
    positions: list
        Positions of all atoms in the system. 
        Expects Vec3D
    topology: Topology
        
    paddingFibreAxis : float
        Padding along the z-axis (stacking direction) in nm
    paddingPerpToFibreAxis : float
        Padding in x and y directions in nm
    fibreAxis : str
        the axis the fibre stacked along
        options: ['x','y','z']
    
    Returns:
    --------
    topology : openmm.app.Topology
        The topology with box vectors set
    positions : list
        Atomic positions
    box_vectors : tuple
        The box vectors (a, b, c)
    """
    if isinstance(positions, list):
        _incoming_unit = positions[0].unit
    else:
        _incoming_unit = positions.unit

    if fibreAxis not in ['x', 'y','z']:
        raise ValueError(f"{fibreAxis=} must be in ['x','y','z']")
    
    # Convert positions to numpy array for easier manipulation
    bb_names = {"N", "CA", "C"}#, "O"}
    bb_idx = [i for i, a in enumerate(topology.atoms()) if a.name in bb_names]
    if not bb_idx:
        raise ValueError("No backbone atoms (N, CA, C, O) found.")
    
    pos_array = []
    for i in bb_idx:
        _p = positions[i]
        pos_array.append( [_p.x, _p.y, _p.z] )
    pos_array = np.array(pos_array)
    # pos_array = np.array([[p.x, p.y, p.z] for p in positions])
    all_positions = np.array([[p.x, p.y, p.z] for p in positions])
    logging.info(len(all_positions))
    # Calculate the principal axes to find the stacking direction
    # Center the coordinates
    centroid = np.mean(pos_array, axis=0)
    centered = pos_array - centroid
    all_centered = all_positions - centroid
    
    # # Calculate covariance matrix and get eigenvectors
    # cov = np.cov(centered.T)
    # eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # # Sort by eigenvalues (largest first)
    # idx = eigenvalues.argsort()[::-1]
    # eigenvalues  = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]
    
    # # The first eigenvector points along the longest axis (stacking direction)
    # # We want this aligned with z-axis
    # rotation_matrix = eigenvectors.T
    
    # # Rotate positions to align stacking with z-axis
    # rotated = all_centered @ rotation_matrix.T
    rotated = all_centered
    
    # Get bounding box in rotated coordinates
    min_coords = np.min(rotated, axis=0)
    max_coords = np.max(rotated, axis=0)

    if fibreAxis=="z":
        padding_x = paddingPerpToFibreAxis
        padding_y = paddingPerpToFibreAxis
        padding_z = paddingFibreAxis
        max_fibreAxis = 0
        min_fibreAxis =1000
        for i in bb_idx:
            _z = rotated[i,2]
            if _z>max_fibreAxis:
                max_fibreAxis = _z
            if _z<min_fibreAxis:
                min_fibreAxis = _z
    elif fibreAxis=="x":
        padding_z = paddingPerpToFibreAxis
        padding_y = paddingPerpToFibreAxis
        padding_x = paddingFibreAxis
    elif fibreAxis=="y":
        padding_z = paddingPerpToFibreAxis
        padding_x = paddingPerpToFibreAxis
        padding_y = paddingFibreAxis
        
    
    # Calculate box dimensions with padding
    box_x = (max_coords[0] - min_coords[0]) + 2 * padding_x
    box_y = (max_coords[1] - min_coords[1]) + 2 * padding_y
    # box_z = (max_coords[2] - min_coords[2]) + 2 * padding_z
    box_z = (max_fibreAxis - min_fibreAxis) + 2 * padding_z

    # box_x = (max_coords[1] - min_coords[1]) + 2 * padding_x
    # box_y = (max_coords[2] - min_coords[2]) + 2 * padding_y
    # box_z = (max_coords[0] - min_coords[0]) + 2 * padding_z
    box_z = (max_fibreAxis - min_fibreAxis) + 2 * padding_z

    if fibreAxis=="z":
        box_xy = max(box_x, box_y)
    
    logging.info(f"Protein dimensions (before padding):")
    logging.info(f"  X: {max_coords[0] - min_coords[0]:.3f} nm")
    logging.info(f"  Y: {max_coords[1] - min_coords[1]:.3f} nm")
    logging.info(f"  Z: {max_coords[2] - min_coords[2]:.3f} nm (stacking direction)")
    logging.info(f"\nBox dimensions (with padding):")
    logging.info(f"  X: {box_xy:.3f} nm (padding: {padding_x} nm)")
    logging.info(f"  Y: {box_xy:.3f} nm (padding: {padding_y} nm)")
    logging.info(f"  Z: { box_z:.3f} nm (padding: {padding_z} nm)")
    
    # Center the protein in the box
    rotated -= min_coords
    rotated += np.array([padding_x, padding_y, padding_z])
    
    # Convert back to original reference frame (keep z-aligned)
    final_positions = rotated #@ rotation_matrix + centroid
    
    # Update positions
    new_positions = [omm.Vec3(p[0], p[1], p[2])* _incoming_unit for p in final_positions] 
    
    # Set box vectors (rectangular box, orthogonal)
    a = omm.Vec3(box_xy, 0, 0)# * omm.unit.nanometer
    b = omm.Vec3(0, box_xy, 0)# * omm.unit.nanometer
    c = omm.Vec3(0, 0, box_z)# * omm.unit.nanometer
    
    topology.setPeriodicBoxVectors([a, b, c]* omm.unit.nanometer)
    
    return topology, new_positions, (a, b, c)

def center_on_origin(topology, positions, tolerance=0.01):
    """
    Check if the structure is centered on the origin and center it if needed.
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        The molecular topology
    positions : list of Vec3
        Atomic positions with units
    tolerance : float, optional
        Distance threshold in nm to determine if structure is already centered
        Default is 0.01 nm (0.1 Angstrom)
    
    Returns:
    --------
    shifted_positions : list of Vec3 or None
        Shifted positions if centering was needed, None if already centered
    """
    if isinstance(positions, list):
        _incoming_unit = positions[0].unit
    else:
        _incoming_unit = positions.unit
        
    # Convert positions to numpy array (strip units for calculation)
    pos_array = np.array([[p.x, p.y, p.z] for p in positions])
    
    # Calculate centroid
    centroid = np.mean(pos_array, axis=0)
    distance_from_origin = np.linalg.norm(centroid)
    
    logging.info(f"\nCentroid position: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}) nm")
    logging.info(f"Distance from origin: {distance_from_origin:.4f} nm")
    
    # Check if already centered within tolerance
    if distance_from_origin < tolerance:
        logging.info(f"Structure is already centered (within {tolerance} nm tolerance)")
        return None
    
    # Need to center - shift all positions
    logging.info(f"Centering structure (shift needed: {distance_from_origin:.4f} nm)")
    shifted_array = pos_array - centroid
    
    # Convert back to OpenMM format with units
    shifted_positions = [omm.Vec3(p[0], p[1], p[2])* _incoming_unit for p in shifted_array]
    
    return shifted_positions

def openmm_to_mdtraj_topology(omm_top):
    return mdt.Topology.from_openmm(omm_top)

def main(params):

    sequence    = params["sequence"]
    sequence_id = params["sequence_id"]
    arrangement = params["arrangement"]
    output_dir  = params["output_dir"]

    if arrangement == "parallel":
        _phi, _psi = -120, 115 # from wikipedia page: https://en.wikipedia.org/wiki/Beta_sheet#Geometry
        _phi, _psi = -119, 113 # from https://bio.libretexts.org/Bookshelves/Biochemistry/Fundamentals_of_Biochemistry_(Jakubowski_and_Flatt)/01%3A_Unit_I-_Structure_and_Catalysis/04%3A_The_Three-Dimensional_Structure_of_Proteins/4.02%3A_Secondary_Structure_and_Loops
    elif arrangement == "antiparallel":
        _phi, _psi = -140, 135 # frmo wikipedia page: https://en.wikipedia.org/wiki/Beta_sheet#Geometry
        _phi, _psi = -139, 135 # from same bio.libretext

    phis = [_phi]*(len(sequence)-1)
    psis = [_psi]*(len(sequence)-1)


    structure = PeptideBuilder.make_structure(
        AA_chain=sequence,
        phi = phis, 
        psi_im1 = psis
    )

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save("temp.pdb")

    fixer = biopython_to_pdbfixer_stringio(structure)

    fixed = pdbfixer_workflow(fixer)

    topology  = fixed.topology
    positions = fixed.positions

    _, positions = setup_and_minimize(topology, positions)

    if params["build_stacked_sheets"]:
        positions = align_pc1_to_x_backbone(topology, positions)

        dtheta = 1.0
        angle_grid = np.arange(-185.0,185.0+dtheta, dtheta)

        spacing = 0.5 # nanometer
        spacing_grid   = np.arange(0.35,0.6+0.02, 0.05)
        layer_separation = 1.0 # nanometer
        layer_sep_grid = np.arange(0.8,1.2+0.05, 0.1)

        nEs = 0 # number of dssp codes indicating (beta) strand/bridge
        max_nEs = -1
        best_theta = -np.inf
        best_spacing = 0
        best_separa  = 0

        for _spacing in spacing_grid:
            logging.info(f"on {_spacing=}")
            for _layer_sep in layer_sep_grid:
                
                _t0 = time.time()
                for _theta in angle_grid:
                    
                    rotated_positions = positions#rotate_around_length_axis(topology, positions, (_theta/180)*np.pi)
                    
                    twist_top, twist_pos = build_sandwich(
                        topology, 
                        rotated_positions, 
                        num_sheets=2, 
                        nchains_per_sheet=2, 
                        spacing=_spacing, 
                        layer_separation=_layer_sep,
                        pattern=arrangement,
                        theta=_theta
                    )

                    _hydrophobic_burial = check_hydrophobic_burial(
                        twist_top, 
                        twist_pos, 
                        chain_sheet1="A", 
                        chain_sheet2="C",
                        exclude_termini=1
                    )
                    if _hydrophobic_burial["buried_score"]+_hydrophobic_burial["exposed_score"]>0:
                        if _hydrophobic_burial['buried_fraction']<0.5:
                            continue
                    
                    do_chainsAB_overlap = check_for_overlap(
                        twist_top, 
                        twist_pos, 
                        chain1="A",
                        chain2="B",
                        bbox_atoms="all",
                        tolerance=0.0,
                        verbose=False,
                        method="obb"
                    )
                
                    do_chainsAK_overlap = check_for_overlap(
                        twist_top, 
                        twist_pos, 
                        chain1="A",
                        chain2="C",
                        bbox_atoms="all",
                        tolerance=0.0,
                        verbose=False,
                        method="obb"
                    )
                    
                    if do_chainsAB_overlap or do_chainsAK_overlap:
                        continue
                    
                    twist_top, twist_pos = build_sandwich(
                        topology, 
                        rotated_positions, 
                        num_sheets=2, 
                        nchains_per_sheet=10, 
                        spacing=_spacing, 
                        layer_separation=_layer_sep,
                        pattern=arrangement,
                        theta=_theta
                    )
                    
                    twist_pos_lst = []
                    for _vec in twist_pos:
                        _lst = [_vec.x, _vec.y, _vec.z]
                        twist_pos_lst.append(_lst)
                
                    _traj = mdt.Trajectory(twist_pos_lst, openmm_to_mdtraj_topology(twist_top) )
                    
                    _dssp_results = mdt.compute_dssp(_traj)
                    
                    nEs = np.sum(_dssp_results=="E")
                    if nEs>=max_nEs:
                        logging.info(f"beta strand number: {nEs=}")
                        max_nEs=nEs
                        best_theta = _theta
                        best_spacing = _spacing
                        best_separa  = _layer_sep
                        logging.info(f"{best_theta=}, {best_spacing=}, {best_separa=}, {arrangement=}")


                _tf = time.time()
                logging.info(f"time to finish angle grid = {_tf-_t0}s")

        logging.info("Best values found:")
        logging.info(f"{max_nEs=}, {best_theta=}, {best_spacing=}, {best_separa=}, {arrangement=}")
        # rotated_positions = rotate_around_length_axis(topology, positions, (best_theta/180)*np.pi)
        twist_top, twist_pos = build_sandwich(
                topology, 
                rotated_positions, 
                num_sheets=2, 
                nchains_per_sheet=10, 
                spacing=best_spacing, 
                layer_separation=best_separa,
                pattern=arrangement,
                theta=best_theta
        )

        do_chainsAB_overlap = check_for_overlap(
            twist_top, 
            twist_pos, 
            chain1="A",
            chain2="B",
            bbox_atoms="all",
            tolerance=0.0,
            verbose=False,
            method="obb"
            
        )

        do_chainsAK_overlap = check_for_overlap(
            twist_top, 
            twist_pos, 
            chain1="A",
            chain2="K",
            bbox_atoms="all",
            tolerance=0.0,
            verbose=False,
            method="obb"
        )
        logging.info(f"{do_chainsAB_overlap=} | {do_chainsAK_overlap=}")
        new_twist_top, new_twist_pos, (boxA, boxB, boxC) = setup_rectangular_box(
            twist_top, 
            twist_pos, 
            (best_spacing/2)+0.015, 
            1.25
        )
        new_twist_pos = center_on_origin(new_twist_top, new_twist_pos)

        _hydrophobic_burial = check_hydrophobic_burial(twist_top, new_twist_pos, chain_sheet1="E", chain_sheet2="O",exclude_termini=1)

        logging.info("\nBox dimensions:")
        logging.info(new_twist_top.getUnitCellDimensions(),"\n")

        twist_pos_lst = []
        for _vec in new_twist_pos:
            _vec = _vec.value_in_unit(angstrom)
            _lst = [_vec.x, _vec.y, _vec.z]
            twist_pos_lst.append(_lst)
        logging.info("saving...")
        output_fname = f"{sequence_id}_{arrangement}.pdb"
        logging.info("to", os.path.join(output_dir,output_fname))
        PDBFile.writeFile(
            new_twist_top, 
            twist_pos_lst, 
            open( os.path.join(output_dir,output_fname), 'w'), 
            keepIds=True
        )
        logging.info("done")

    else:
        # save the minimized single monomer structure to a pdb
        with open(os.path.join(output_dir, f"{sequence_id}.pdb"), "w") as f:
            PDBFile.writeFile(topology, positions, f)
        logging.info(f"Saved minimized structure for {sequence_id} to {output_dir}/{sequence_id}.pdb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a structure from a sequence and arrangement")
    parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence (1-letter code)")
    parser.add_argument("--sequence_id", type=str, required=True, help="Identifier for the sequence (used in output filename)")
    parser.add_argument("--arrangement", type=str, choices=["parallel", "antiparallel"], default="parallel", help="Arrangement of beta strands")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output PDB files")
    parser.add_argument("--build_stacked_sheets", action="store_true", help="Whether to build stacked sheets (not implemented yet)")
    
    args = parser.parse_args()
    params = vars(args)

    if args.sequence_id not in args.output_dir:
        output_dir = os.path.join(args.output_dir, args.sequence_id)
        params["output_dir"] = output_dir
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(output_dir, "sequence_to_structure.log"),
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting sequence to structure generation for {args.sequence_id} with arrangement {args.arrangement}")
    logging.info(f"Input sequence: {args.sequence}")
    main(params)