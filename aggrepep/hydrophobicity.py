from typing import Literal, Dict, Any, Iterable

import numpy as np
import copy

from openmm.app import *
from openmm import *
from openmm.unit import *
from openmm import Vec3, unit
from openmm.app import Topology, PDBFile
from openmm.unit import nanometer, angstrom

from Bio.PDB import PDBIO
from pdbfixer import PDBFixer
from io import StringIO
from peptides import Peptide

def check_hydrophobic_burial(topology, positions, 
                             chain_sheet1="A", chain_sheet2="B",
                             hydrophobicity_scale="kyte_doolittle",
                             exclude_termini=2,
                             verbose=True):
    """
    Check if hydrophobic residues are buried inside a sandwich structure.
    
    Uses geometric criterion: a residue's beta-carbon (Cβ) must lie between 
    the two sheet planes to be considered "buried". Assumes peptides lie in 
    the xy-plane with sheets separated along the z-axis.
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        Topology containing both sheets
    positions : list of Vec3
        Positions of all atoms
    chain_sheet1 : str
        Chain ID from first sheet to analyze (default "A")
    chain_sheet2 : str
        Chain ID from second sheet (used to define opposite plane, default "B")
    hydrophobicity_scale : str
        choices=["kyte_doolittle", "eisenberg", "wimley_white"]
        Hydrophobicity scale to use (default "kyte_doolittle")
        - kyte_doolittle: Kyte-Doolittle scale (1982)
        - eisenberg: Eisenberg consensus scale (1984)
        - wimley_white: Wimley-White whole residue scale (1996)
    exclude_termini : int
        Number of residues to exclude from each terminus (default 2)
        Set to 0 to include all residues
    verbose : bool
        If True, print detailed analysis (default True)
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'buried_fraction': float, fraction of total hydrophobicity that is buried (0-1)
        - 'buried_score': float, sum of hydrophobicity scores for buried residues
        - 'exposed_score': float, sum of hydrophobicity scores for exposed residues
        - 'total_score': float, total hydrophobicity score (buried + exposed)
        - 'num_buried': int, number of residues with Cβ inside sandwich
        - 'num_exposed': int, number of residues with Cβ outside sandwich
        - 'num_total': int, total number of residues analyzed
        - 'residue_details': list of dict, per-residue information:
            - 'resname': residue name
            - 'resid': residue number
            - 'hydrophobicity': hydrophobicity score
            - 'is_buried': bool, whether Cβ is inside sandwich
            - 'ca_position': CA position
            - 'cb_position': Cβ position (or CA for glycine)
            - 'z_position': z-coordinate of Cβ
    
    Notes:
    ------
    - Glycine uses CA position instead of Cβ (since it has no Cβ)
    - Only hydrophobic residues (positive hydrophobicity scores) contribute
      to the buried/exposed fractions
    - Assumes sheets lie in xy-plane, separated along z-axis
    - The geometric criterion: Cβ is "buried" if its z-coordinate lies between
      the mean z-positions of the two chains
    - Requires peptides package: pip install peptides
    """
    
    if hydrophobicity_scale not in ["kyte_doolittle", "eisenberg", "wimley_white"]:
        raise ValueError(f"{hydrophobicity_scale=} must be in ['kyte_doolittle', 'eisenberg', 'wimley_white']")
    
    # Hydrophobicity scales
    # Kyte-Doolittle (1982) - higher = more hydrophobic
    KYTE_DOOLITTLE = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Eisenberg consensus (1984) - normalized scale
    EISENBERG = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    # Wimley-White whole residue (1996) - free energy of transfer to bilayer
    WIMLEY_WHITE = {
        'A': 0.17, 'R': 0.81, 'N': 0.42, 'D': 1.23, 'C': -0.24,
        'Q': 0.58, 'E': 2.02, 'G': 0.01, 'H': 0.17, 'I': -0.31,
        'L': -0.56, 'K': 0.99, 'M': -0.23, 'F': -1.13, 'P': 0.45,
        'S': 0.13, 'T': 0.14, 'W': -1.85, 'Y': -0.94, 'V': 0.07
    }
    
    # Select scale and invert if needed
    if hydrophobicity_scale == "kyte_doolittle":
        scale = KYTE_DOOLITTLE
        invert = False  # higher = more hydrophobic
    elif hydrophobicity_scale == "eisenberg":
        scale = EISENBERG
        invert = False  # higher = more hydrophobic
    else:  # wimley_white
        scale = WIMLEY_WHITE
        invert = True  # lower (more negative) = more hydrophobic, so invert
    
    # Find chains
    chain1_obj = None
    chain2_obj = None
    for chain in topology.chains():
        if chain.id == chain_sheet1:
            chain1_obj = chain
        if chain.id == chain_sheet2:
            chain2_obj = chain
    
    if chain1_obj is None:
        raise ValueError(f"Chain {chain_sheet1} not found in topology")
    if chain2_obj is None:
        raise ValueError(f"Chain {chain_sheet2} not found in topology")
    
    # Get mean z-position for both chains (using CA atoms)
    def get_mean_z(chain_obj):
        z_positions = []
        for atom in topology.atoms():
            if atom.residue.chain.id == chain_obj.id and atom.name == "CA":
                z_positions.append(positions[atom.index].y)
        if len(z_positions) == 0:
            raise ValueError(f"No CA atoms found for chain {chain_obj.id}")
        return np.mean(z_positions)
    
    z_sheet1 = get_mean_z(chain1_obj)
    z_sheet2 = get_mean_z(chain2_obj)
    
    # Ensure z_sheet1 < z_sheet2 for clarity
    if z_sheet1 > z_sheet2:
        z_sheet1, z_sheet2 = z_sheet2, z_sheet1
    
    z_min = z_sheet1
    z_max = z_sheet2
    
    # Analyze chain1 residues
    residues = list(chain1_obj.residues())
    num_residues = len(residues)
    
    # Apply termini exclusion
    if exclude_termini > 0:
        start_idx = exclude_termini
        end_idx = num_residues - exclude_termini
        residues_to_analyze = residues[start_idx:end_idx]
    else:
        residues_to_analyze = residues
    
    if len(residues_to_analyze) == 0:
        raise ValueError(f"No residues to analyze after excluding {exclude_termini} terminal residues")
    
    # Collect residue information
    residue_details = []
    buried_score = 0.0
    exposed_score = 0.0
    num_buried = 0
    num_exposed = 0
    
    for residue in residues_to_analyze:
        resname = residue.name
        resid = residue.id
        
        # Get single-letter code
        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        if resname not in three_to_one:
            continue  # Skip non-standard residues
        
        res_code = three_to_one[resname]
        
        # Get hydrophobicity score
        hydro_score = scale[res_code]
        if invert:
            hydro_score = -hydro_score
        
        # Find CA and CB atoms
        ca_pos = None
        cb_pos = None
        
        for atom in residue.atoms():
            pos = positions[atom.index]
            pos_array = np.array([pos.x, pos.y, pos.z])
            
            if atom.name == "CA":
                ca_pos = pos_array
            elif atom.name == "CB":
                cb_pos = pos_array
        
        if ca_pos is None:
            continue  # Skip if no CA found
        
        # Use CA for glycine (no CB)
        if cb_pos is None:
            cb_pos = ca_pos
        
        # Check if CB z-position is between the two sheets
        z_cb = cb_pos[1]
        is_buried = (z_min < z_cb < z_max)
        
        # Only count hydrophobic residues (positive scores)
        if hydro_score > 0:
            if is_buried:
                buried_score += hydro_score
                num_buried += 1
            else:
                exposed_score += hydro_score
                num_exposed += 1
        
        residue_details.append({
            'resname': resname,
            'resid': resid,
            'res_code': res_code,
            'hydrophobicity': hydro_score,
            'is_buried': is_buried,
            'ca_position': ca_pos,
            'cb_position': cb_pos,
            'z_position': z_cb
        })
    
    # Calculate fraction
    total_score = buried_score + exposed_score
    if total_score > 0:
        buried_fraction = buried_score / total_score
    else:
        buried_fraction = 0.0
    
    results = {
        'buried_fraction': buried_fraction,
        'buried_score': buried_score,
        'exposed_score': exposed_score,
        'total_score': total_score,
        'num_buried': num_buried,
        'num_exposed': num_exposed,
        'num_total': len(residues_to_analyze),
        'residue_details': residue_details,
        'hydrophobicity_scale': hydrophobicity_scale,
        'chain_analyzed': chain_sheet1,
        'z_sheet1': z_sheet1,
        'z_sheet2': z_sheet2,
        'z_sandwich_thickness': z_sheet2 - z_sheet1
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Hydrophobic Burial Analysis")
        print(f"{'='*70}")
        print(f"Chain analyzed: {chain_sheet1}")
        print(f"Reference chain: {chain_sheet2}")
        print(f"Hydrophobicity scale: {hydrophobicity_scale}")
        print(f"Terminal residues excluded: {exclude_termini} from each end")
        print(f"Total residues analyzed: {len(residues_to_analyze)}")
        
        print(f"\nSheet z-positions:")
        print(f"  Sheet 1 (mean z): {z_sheet1:.3f} nm")
        print(f"  Sheet 2 (mean z): {z_sheet2:.3f} nm")
        print(f"  Sandwich thickness: {z_sheet2 - z_sheet1:.3f} nm")
        print(f"  Burial criterion: {z_min:.3f} < z < {z_max:.3f}")
        
        print(f"\nHydrophobic residue statistics:")
        print(f"  Buried hydrophobic residues: {num_buried}")
        print(f"  Exposed hydrophobic residues: {num_exposed}")
        print(f"  Buried hydrophobicity score: {buried_score:.2f}")
        print(f"  Exposed hydrophobicity score: {exposed_score:.2f}")
        print(f"  Total hydrophobicity score: {total_score:.2f}")
        print(f"  Buried fraction: {buried_fraction:.3f} ({buried_fraction*100:.1f}%)")
        
        print(f"\nPer-residue breakdown:")
        print(f"  {'Res':<6} {'ID':<5} {'Hydro':<7} {'Buried':<8} {'Z-pos':<8}")
        print(f"  {'-'*45}")
        for res in residue_details:
            if res['hydrophobicity'] > 0:  # Only show hydrophobic
                burial_status = 'YES' if res['is_buried'] else 'NO'
                print(f"  {res['resname']:<6} {res['resid']:<5} "
                      f"{res['hydrophobicity']:>6.2f}  "
                      f"{burial_status:<8} "
                      f"{res['z_position']:>7.3f}")
        
        print(f"\nResult: ", end="")
        if buried_fraction > 0.7:
            print(f"EXCELLENT burial ({buried_fraction*100:.1f}% of hydrophobicity buried)")
        elif buried_fraction > 0.5:
            print(f"GOOD burial ({buried_fraction*100:.1f}% of hydrophobicity buried)")
        elif buried_fraction > 0.3:
            print(f"MODERATE burial ({buried_fraction*100:.1f}% of hydrophobicity buried)")
        else:
            print(f"POOR burial ({buried_fraction*100:.1f}% of hydrophobicity buried)")
        
        print(f"{'='*70}\n")
    
    return results