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

from PeptideBuilder import Geometry
import PeptideBuilder


BACKBONE_NAMES = {"N", "CA", "C", "O"}
ALPHABET = {i:chr(65+i) for i in range(26)}

def biopython_to_pdbfixer_stringio(structure):
    """Convert BioPython Structure to PDBFixer without temp file"""
    
    # Write BioPython structure to string
    io = PDBIO()
    io.set_structure(structure)
    
    pdb_string = StringIO()
    io.save(pdb_string)
    pdb_string.seek(0)  # Reset to beginning
    
    # Load with PDBFixer from string
    fixer = PDBFixer(pdbfile=pdb_string)
    
    return fixer

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

def _to_nm_array(positions) -> np.ndarray:
    """OpenMM positions (Quantity[List[Vec3]]) -> (N,3) float64 in nm."""
    # Positions may already be Quantity with length unit
    if isinstance(positions,list):
        _original_unit = positions[0].unit
    else:
        _original_unit = positions.unit

    pos_nm = unit.Quantity(positions, unit=positions[0].unit) if hasattr(positions[0], 'unit') else positions
    factor = 1.0 / unit.nanometer
    return np.array([[ (p.x*_original_unit*factor), (p.y*_original_unit*factor), (p.z*_original_unit*factor) ]
                     for p in positions], dtype=np.float64)

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    normalizes the given vector
    """
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def write_psf(topology, filename, title="PSF file generated from OpenMM"):
    """
    Write an OpenMM topology to a PSF file.
    
    Parameters
    ----------
    topology : openmm.app.Topology
        The OpenMM topology object
    filename : str
        Output PSF filename
    title : str, optional
        Title for the PSF file
    """
    
    with open(filename, 'w') as f:
        # Header
        f.write("PSF\n\n")
        f.write(f"       1 !NTITLE\n")
        f.write(f" REMARKS {title}\n\n")
        
        # Atoms section
        atoms = list(topology.atoms())
        f.write(f"{len(atoms):8d} !NATOM\n")
        
        for i, atom in enumerate(atoms, 1):
            segid = atom.residue.chain.id if atom.residue.chain.id else "A"
            # Convert residue id to integer (it may be a string)
            try:
                resid = int(atom.residue.id)
            except (ValueError, TypeError):
                resid = i  # Fallback to atom index if id is not numeric
            resname = atom.residue.name
            atomname = atom.name
            atomtype = atom.element.symbol if atom.element else "X"
            charge = 0.0  # OpenMM topology doesn't store charges
            mass = atom.element.mass.value_in_unit_system(openmm.unit.md_unit_system) if atom.element else 0.0
            
            # PSF format: atom_id segid resid resname atomname atomtype charge mass
            f.write(f"{i:8d} {segid:<4s} {resid:<4d} {resname:<4s} {atomname:<4s} "
                   f"{atomtype:<4s} {charge:14.6f} {mass:14.4f}           0\n")
        
        f.write("\n")
        
        # Bonds section
        bonds = list(topology.bonds())
        f.write(f"{len(bonds):8d} !NBOND: bonds\n")
        
        atom_indices = {atom: i+1 for i, atom in enumerate(atoms)}
        
        for i, bond in enumerate(bonds):
            atom1_idx = atom_indices[bond[0]]
            atom2_idx = atom_indices[bond[1]]
            f.write(f"{atom1_idx:8d}{atom2_idx:8d}")
            
            # 4 bonds per line
            if (i + 1) % 4 == 0:
                f.write("\n")
        
        if len(bonds) % 4 != 0:
            f.write("\n")
        f.write("\n")
        
        # Angles section (3-body)
        angles = []
        for atom in atoms:
            bonded = list(topology.bonds())
            neighbors = []
            for bond in bonded:
                if bond[0] == atom:
                    neighbors.append(bond[1])
                elif bond[1] == atom:
                    neighbors.append(bond[0])
            
            # Create angles for this central atom
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    angles.append((neighbors[i], atom, neighbors[j]))
        
        f.write(f"{len(angles):8d} !NTHETA: angles\n")
        for i, angle in enumerate(angles):
            idx1 = atom_indices[angle[0]]
            idx2 = atom_indices[angle[1]]
            idx3 = atom_indices[angle[2]]
            f.write(f"{idx1:8d}{idx2:8d}{idx3:8d}")
            
            if (i + 1) % 3 == 0:
                f.write("\n")
        
        if len(angles) % 3 != 0:
            f.write("\n")
        f.write("\n")
        
        # Dihedrals section (4-body) - simplified
        f.write(f"{0:8d} !NPHI: dihedrals\n\n")
        
        # Impropers section
        f.write(f"{0:8d} !NIMPHI: impropers\n\n")
        
        # Donors section
        f.write(f"{0:8d} !NDON: donors\n\n")
        
        # Acceptors section
        f.write(f"{0:8d} !NACC: acceptors\n\n")
        
        # NNB section
        f.write(f"{0:8d} !NNB\n\n")
        
        # Cross-terms
        f.write(f"{0:8d} !NCRTERM: cross-terms\n\n")
    print(f"Done writing psf to {filename=}")

def perpendicular_translation_vector(
    topology: Topology,
    positions,
    selection: Literal["CA", "backbone", "all"] = "CA",
    method:   Literal["pca", "end_to_end"] = "pca",
    prefer:   Literal["v1", "v2"] = "v1",
    distance: unit.Quantity = 0.0 * unit.nanometer,
) -> Dict[str, Any]:
    """
    Determine two orthonormal directions perpendicular to a coil's length axis and return one as a unit vector;
    optionally return a translation Vec3 of length `distance` along that unit vector.

    Args:
        topology: openmm.app.Topology for atom selection / residue order.
        positions: OpenMM positions (Quantity[List[Vec3]]).
        selection: "CA" (default), "backbone", or "all" atoms used to define the axis.
        method: "pca" (principal component of selected atoms) or "end_to_end" (N- to C-terminal Cα vector).
        prefer: choose which perpendicular vector to return ("v1" or "v2"); both are perpendicular to the axis.
        distance: length of desired translation; 0 nm returns no shift (still returns the unit vector).

    Returns:
        {
          "axis_unit": np.ndarray shape (3,),           # unit vector along coil length
          "perp1_unit": np.ndarray shape (3,),          # first perpendicular
          "perp2_unit": np.ndarray shape (3,),          # second perpendicular (perp to axis and perp1)
          "chosen_unit": np.ndarray shape (3,),         # perp1 or perp2 per `prefer`
          "translation_vec3": openmm.Vec3 * unit.nm     # translation of magnitude `distance` (0 if distance=0)
        }

    # ---------- Example usage ----------
    # out = perpendicular_translation_vector(top, positions, selection="CA", method="pca", prefer="v1", distance=0.5*unit.nanometer)
    # # Apply the translation to all atoms (or a subset):
    # shift = out["translation_vec3"]
    # new_positions = positions.__class__([p + shift for p in positions])  # preserves Quantity[List[Vec3]]

    """
    # --- select atoms ---
    if selection == "CA":
        sel_idx = [i for i, a in enumerate(topology.atoms()) if a.name == "CA"]
    elif selection == "backbone":
        sel_idx = [i for i, a in enumerate(topology.atoms()) if a.name in BACKBONE_NAMES]
    else:
        sel_idx = list(range(topology.getNumAtoms()))
    if len(sel_idx) < 2:
        raise ValueError("Need at least two atoms in selection to define an axis.")

    coords_nm = _to_nm_array(positions)
    X = coords_nm[sel_idx]

    # --- axis (length direction) ---
    if method == "pca":
        Xc = X - X.mean(axis=0, keepdims=True)
        C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        # eigh returns ascending eigenvalues -> last is the largest (principal component)
        vals, vecs = np.linalg.eigh(C)
        axis = vecs[:, -1]
    else:  # "end_to_end": Cα of first and last residue
        # map residue index -> its CA atom index if present
        res_to_ca = {}
        for i, a in enumerate(topology.atoms()):
            if a.name == "CA":
                res_to_ca.setdefault(a.residue.index, i)
        if len(res_to_ca) < 2:
            raise ValueError("end_to_end requires at least two residues with Cα atoms.")
        n_idx = res_to_ca[min(res_to_ca.keys())]
        c_idx = res_to_ca[max(res_to_ca.keys())]
        axis = coords_nm[c_idx] - coords_nm[n_idx]
    axis_unit = _normalize(axis)

    # --- build two perpendiculars (stable and deterministic) ---
    # Use a reference not nearly parallel to axis to seed cross products
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis_unit, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    perp1 = _normalize(np.cross(axis_unit, ref))
    perp2 = _normalize(np.cross(axis_unit, perp1))  # guaranteed orthogonal to both

    chosen = perp1 if prefer == "v1" else perp2

    # --- build translation Vec3 of required magnitude ---
    dist_nm = float(distance.value_in_unit(unit.nanometer)) if hasattr(distance, "unit") else float(distance)
    trans = Vec3(*(chosen * dist_nm)) * unit.nanometer

    return {
        "axis_unit":   axis_unit,
        "perp1_unit":  perp1,
        "perp2_unit":  perp2,
        "chosen_unit": chosen,
        "translation_vec3": trans,
    }

def find_length_axis(
    topology: Topology,
    positions,
    selection: Literal["CA","backbone","all"] = "CA",
    method: Literal["pca","end_to_end"] = "pca",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (point_nm, axis_unit) where axis_unit is peptide length direction; point_nm is centroid for pivot."""
    if selection == "CA":
        sel = [i for i,a in enumerate(topology.atoms()) if a.name == "CA"]
    elif selection == "backbone":
        sel = [i for i,a in enumerate(topology.atoms()) if a.name in BACKBONE]
    else:
        sel = list(range(topology.getNumAtoms()))
    if len(sel) < 2:
        raise ValueError("Need ≥2 atoms in selection.")
    X = _to_nm_array(positions)
    Y = X[sel]

    if method == "pca":
        Yc = Y - Y.mean(0)
        C = (Yc.T @ Yc) / max(len(Yc)-1, 1)
        _, vecs = np.linalg.eigh(C)
        axis = vecs[:, -1]
    else:
        # end-to-end using first and last residue CA if available; else first/last selected atom
        res_to_ca = {a.residue.index: i for i,a in enumerate(topology.atoms()) if a.name == "CA"}
        if len(res_to_ca) >= 2:
            n_idx, c_idx = min(res_to_ca.values()), max(res_to_ca.values())
        else:
            n_idx, c_idx = sel[0], sel[-1]
        axis = X[c_idx] - X[n_idx]

    axis_unit = _normalize(axis)
    point_nm  = Y.mean(0)  # centroid as a stable pivot
    return point_nm, axis_unit

def duplicate_chain(topology:Topology, positions:list, chain_id:str ='B'):
    """
    Duplicate the chain to create a new chain
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        Original topology
    positions : list of Vec3
        Original positions
    chain_id : str
        ID for the new chain
    
    Returns:
    --------
    new_topology : openmm.app.Topology
        Topology with duplicated chain
    new_positions : list of Vec3
        Positions with duplicated chain
    """
    # Create new topology with both chains
    new_topology = Topology()
    new_positions = list(positions)  # Copy original positions
    
    # Copy original chain
    for chain in topology.chains():
        new_chain = new_topology.addChain(chain.id)
        for residue in chain.residues():
            new_residue = new_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                new_topology.addAtom(atom.name, atom.element, new_residue)
    
    # Add bonds from original
    atoms_list = list(new_topology.atoms())
    for bond in topology.bonds():
        atom1_idx = bond[0].index
        atom2_idx = bond[1].index
        new_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
    
    # Duplicate chain
    original_chain = list(topology.chains())[0]
    duplicated_chain = new_topology.addChain(chain_id)
    
    atom_offset = topology.getNumAtoms()
    
    for residue in original_chain.residues():
        new_residue = new_topology.addResidue(residue.name, duplicated_chain)
        for atom in residue.atoms():
            new_topology.addAtom(atom.name, atom.element, new_residue)
            # Duplicate position
            new_positions.append(positions[atom.index])
    
    # Add bonds for duplicated chain
    for bond in topology.bonds():
        atom1_idx = bond[0].index + atom_offset
        atom2_idx = bond[1].index + atom_offset
        atoms_list = list(new_topology.atoms())
        new_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
    
    return new_topology, new_positions

def align_pc1_to_x_backbone(topology, positions, mass_weighted=False, length_unit=unit.nanometer):
    """
    Rotate all positions so the first principal component of BACKBONE atoms (N, CA, C, O) aligns with +X.
    Returns (new_positions, Rmat) where new_positions has same units as input.
    """
    pos_unit = positions.unit
    def as_nm(v):
        return [v.x, v.y, v.z]

    # Full coordinates (N x 3)
    # R_all = _to_nm_array(positions)
    R_all = np.array([as_nm(p) for p in positions], dtype=float)

    # Backbone indices
    bb_names = {"N", "CA", "C", "O"}
    bb_idx = [i for i, a in enumerate(topology.atoms()) if a.name in bb_names]
    if not bb_idx:
        raise ValueError("No backbone atoms (N, CA, C, O) found.")
    X_bb = R_all[bb_idx]

    # Weights
    if mass_weighted:
        w = []
        for i in bb_idx:
            a = list(topology.atoms())[i]
            m = a.element.mass.value_in_unit(unit.dalton) if a.element and a.element.mass is not None else 12.0
            w.append(m)
        w = np.asarray(w, dtype=float)
    else:
        w = np.ones(len(bb_idx), dtype=float)

    # Center on backbone centroid
    wsum = w.sum()
    centroid = (w[:, None] * X_bb).sum(axis=0) / wsum
    Xc_bb = X_bb - centroid

    # PCA on backbone
    cov = (Xc_bb * w[:, None]).T @ Xc_bb / wsum
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, np.argmax(evals)]
    pc1 /= np.linalg.norm(pc1)

    print(f"{pc1=}")
    _x = pc1[0]
    _y = pc1[1]
    _theta    = np.degrees( np.arctan(_y/_x) )
    print(_theta,    "angle between the monomer and the x-axis, projected in the xy plane")
    _theta = (-1)*_theta # we want to undo any rotation

    # Apply rotation about the backbone centroid to ALL atoms
    new_positions = center_chain_then_rotate_xy(topology, positions, rotation_angle=_theta)
    _tmp_new_positions = []
    for p in new_positions:
        _tmp_new_positions.append(p.value_in_unit(positions.unit))
    _tmp_new_positions = _tmp_new_positions*positions.unit

    # obtain new PC1 coordinates
    R_all = np.array([as_nm(p) for p in _tmp_new_positions], dtype=float)
    X_bb = R_all[bb_idx]
    Xc_bb = X_bb - centroid
    cov = (Xc_bb * w[:, None]).T @ Xc_bb / wsum
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, np.argmax(evals)]
    pc1 /= np.linalg.norm(pc1)
    print(np.degrees( np.arctan(pc1[2]/pc1[0]) ), "angle between the monomer and the x-axis, projected in the xz plane")

    # Apply rotation about the backbone centroid to ALL atoms
    new_positions = center_chain_then_rotate_xz(topology, _tmp_new_positions, rotation_angle=np.degrees( np.arctan(pc1[2]/pc1[0]) ))

    return new_positions

def translate_rotate_chain(positions, chain_indices, translation=None, rotation_matrix=None):
    """
    Translate and/or rotate a chain
    
    Parameters:
    -----------
    positions : list of Vec3
        All positions
    chain_indices : list of int
        Indices of atoms belonging to the chain to transform
    translation : Vec3 or array-like (3,)
        Translation vector in nanometers
    rotation_matrix : array-like (3, 3)
        Rotation matrix
    
    Returns:
    --------
    new_positions : list of Vec3
        Transformed positions
    """
    new_positions = copy.deepcopy(positions)
    
    # Convert positions to numpy array for the chain
    chain_positions = np.array([[pos.x, pos.y, pos.z] for i, pos in enumerate(positions) 
                                 if i in chain_indices])
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        rotation_matrix = np.array(rotation_matrix)
        chain_positions = chain_positions @ rotation_matrix.T
    
    # Apply translation if provided
    if translation is not None:
        if hasattr(translation, 'x'):  # Vec3 object
            trans = np.array([translation.x, translation.y, translation.z])
        else:
            trans = np.array(translation)
        chain_positions += trans
    
    # Update positions
    for idx, i in enumerate(chain_indices):
        new_positions[i] = Vec3(chain_positions[idx, 0], 
                                chain_positions[idx, 1], 
                                chain_positions[idx, 2]) * nanometer
    
    return new_positions

def compute_center_of_mass(topology:Topology, positions:list):
    """
    Compute the center of mass (COM) for an OpenMM system.

    Args:
        topology (openmm.app.Topology): System topology containing atom info.
        positions (openmm.unit.Quantity[list[Vec3]]): Atom positions.

    Returns:
        openmm.Vec3 * unit.nanometer: Center of mass position.
    """
    total_mass = 0.0 * unit.dalton
    weighted_pos = np.zeros(3) * unit.nanometer * unit.dalton

    for atom, pos in zip(topology.atoms(), positions):
        mass = atom.element.mass
        weighted_pos += pos * mass
        total_mass += mass

    return (weighted_pos / total_mass).in_units_of(unit.nanometer)

def center_chain_then_rotate_xz(topology:Topology, positions:list, rotation_angle:float=180)->list:
    """
    Translates a chain to the origin, rotates it rotation_angle degrees about y-axis, 
    then translates it back to where it was.
    """
    # deg -> rad
    _rotation_angle_rad = (rotation_angle/180)*np.pi
    _cos_a, _sin_a = np.cos(_rotation_angle_rad), np.sin(_rotation_angle_rad)

    _rotation_arr = np.array(
        [
            [     _cos_a, 0, _sin_a],
            [          0, 1,      0],
            [(-1)*_sin_a, 0, _cos_a],
        ]
    )
    
    _com_pos = compute_center_of_mass(topology, positions)
    new_positions = []
    for _pos in positions:
        _centered = _pos - _com_pos
        _rotated  = Vec3(*(_rotation_arr @ _centered._value).tolist() )*_centered.unit
        _new_pos  = _rotated + _com_pos

        new_positions.append(_new_pos)
    return new_positions

def center_chain_then_rotate_xy(topology:Topology, positions:list, rotation_angle:float=180)->list:
    """
    Translates a chain to the origin, rotates it rotation_angle degrees about z-axis, 
    then translates it back to where it was.
    """
    # deg -> rad
    _rotation_angle_rad = (rotation_angle/180)*np.pi
    _cos_a, _sin_a = np.cos(_rotation_angle_rad), np.sin(_rotation_angle_rad)

    _rotation_arr = np.array(
        [
            [_cos_a, (-1)*_sin_a, 0],
            [_sin_a,      _cos_a, 0],
            [     0,           0, 1],
        ]
    )
    
    _com_pos = compute_center_of_mass(topology, positions)
    new_positions = []
    for _pos in positions:
        _centered = _pos - _com_pos
        _rotated  = Vec3(*(_rotation_arr @ _centered._value).tolist() )*_centered.unit
        _new_pos  = _rotated + _com_pos

        new_positions.append(_new_pos)
    return new_positions

def rotate_around_length_axis(
    topology: Topology,
    positions,
    angle,  # float radians or Quantity in radians/degrees
    selection: Literal["CA","backbone","all"] = "CA",
    method: Literal["pca","end_to_end"] = "pca",
):
    """Rotate all atoms around the peptide’s length axis by `angle`; return new positions (Quantity[list[Vec3]])."""
    theta = float(angle.value_in_unit(unit.radian)) if hasattr(angle, "unit") else float(angle)
    print(f"{theta=}")
    pivot_nm, axis = find_length_axis(topology, positions, selection, method)
    k = _normalize(axis)

    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

    X = _to_nm_array(positions)
    Xc = X - pivot_nm
    Xr = (Xc @ R.T) + pivot_nm

    return unit.Quantity([Vec3(*row) for row in Xr], unit.nanometer)

def make_chain_id(chain_num:int)->str:
    """
    Given a chain number/index/count starting from 0, return a ID for that chain
    """
    _chain_letter_ix= chain_num%26
    _chain_letter = ALPHABET[_chain_letter_ix] # A, B, C
    _chain_letter_n = chain_num//26 +1 # 0_0,0_1,0_2,...,0_26, 1_1, 1_2, ..., 1_26, 2_1, ...
    # chain_id = f"{_chain_letter}{_chain_letter_n}"
    if _chain_letter_n>1:
        _chain_letter = _chain_letter.lower()
    chain_id = _chain_letter
    return chain_id

def make_sheet(topology, positions, 
               num_chains=8, 
               spacing=1.5,
               pattern="parallel",
               stack_along_axis="z",
               spacing_method="backbone",
               pattern_plane="xy",
               theta=0.0,
               verbose=False):
    """
    Stack multiple copies of a chain along the given direction with given
    spacing and chain patterning. 
    Parameters:
    -----------
    topology : openmm.app.Topology
        Original single chain topology
    positions : list of Vec3
        Original single chain positions
        Should be prepared s.t. its backbone is
        aligned with the x-axis.
    num_chains : int
        Number of chains to create (default 8)
    spacing : float
        Spacing between chains in nanometers (default 3.0)
    pattern : str
        choices=[parallel, antiparallel]
        Whether to make all of the coils parallel, or to flip every other one.
        Uses a custom function:
        rotated_positions = center_chain_then_rotate_xy(topology, positions, rotation_angle=180)
    stack_along_axis : str
        choices = ["x", "y", "z"]
        which axis to stack along.
    spacing_method : str
        choices=["backbone", "bounding_box"]
        Method to calculate interchain spacing (default "backbone")
    theta : float
        Amount to rotate the chain around the axis defined by its length
    Returns:
    --------
    final_topology : openmm.app.Topology
        Topology with all chains
    final_positions : list of Vec3
        Positions with all chains arranged in rectangle
    """
    if num_chains > 52:
        raise Warning(f"{num_chains=}>52=26+26. Some chains may have the same chain ID.")
    
    if stack_along_axis not in ["x", "y", "z"]:
        raise ValueError(f"{stack_along_axis=} must be in ['x','y','z']")
    
    if pattern not in ["parallel", "antiparallel"]:
        raise ValueError(f"{pattern=} must be in ['parallel', 'antiparallel']")
    
    if spacing_method not in ["backbone", "bounding_box"]:
        raise ValueError(f"{spacing_method=} must be in ['backbone', 'bounding_box']")
    
    # Calculate chain extent based on spacing_method
    if spacing_method == "backbone":
        # Use only backbone atoms
        bb_names = {"N", "CA", "C"}#, "O"}
        bb_idx = [i for i, a in enumerate(topology.atoms()) if a.name in bb_names]
        if not bb_idx:
            raise ValueError("No backbone atoms (N, CA, C, O) found.")
        
        if stack_along_axis == "x":
            coords = [positions[i].x for i in bb_idx]
        elif stack_along_axis == "y":
            coords = [positions[i].y for i in bb_idx]
        else:  # stack_along_axis == "z"
            coords = [positions[i].z for i in bb_idx]
    else:  # spacing_method == "bounding_box"
        # Use all atoms
        if stack_along_axis == "x":
            coords = [p.x for p in positions]
        elif stack_along_axis == "y":
            coords = [p.y for p in positions]
        else:  # stack_along_axis == "z"
            coords = [p.z for p in positions]
    
    chain_min = min(coords)
    chain_max = max(coords)
    chain_extent = chain_max - chain_min
    
    if verbose:
        print(f"Stacking {num_chains} chains along {stack_along_axis}-axis with {spacing} nm spacing")
        print(f"Pattern: {pattern}, Spacing method: {spacing_method}")
        print(f"Chain extent: {chain_extent:.3f} nm")
    
    final_topology = Topology()
    final_positions = []
    
    atoms_per_chain = topology.getNumAtoms()
    
    for chain_idx in range(num_chains):
        # Create chain
        chain_id = make_chain_id(chain_idx)
        new_chain = final_topology.addChain(chain_id)
        
        # Copy residues and atoms
        for residue in topology.residues():
            new_residue = final_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                final_topology.addAtom(atom.name, atom.element, new_residue)
        
        # Copy bonds
        atom_offset = chain_idx * atoms_per_chain
        atoms_list = list(final_topology.atoms())
        for bond in topology.bonds():
            atom1_idx = bond[0].index + atom_offset
            atom2_idx = bond[1].index + atom_offset
            final_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
        
        # Determine if this chain should be flipped (for antiparallel pattern)
        should_flip = (pattern == "antiparallel" and chain_idx % 2 == 1)

        # _rotated_positions = rotate_around_length_axis(topology, positions, ( theta/180.0)*np.pi)
        
        # Get positions for this chain (flip if needed)
        if should_flip:
            # _rotated_positions = rotate_around_length_axis(topology, positions, (2*theta/180.0)*np.pi)
            # print(f"positions         = {positions[:3]}")
            # print(f"rotated positions = {_rotated_positions[:3]}")
            # print()
            if   pattern_plane=="xy":
                chain_positions = center_chain_then_rotate_xy(topology, _rotated_positions)
            elif pattern_plane=="xz":
                chain_positions = center_chain_then_rotate_xz(topology, positions)
            chain_positions = rotate_around_length_axis(topology, chain_positions, (-2*theta/180.0)*np.pi)
        else:
            _rotated_positions = rotate_around_length_axis(topology, positions, ( theta/180.0)*np.pi)
            chain_positions = _rotated_positions
        
        # Calculate translation for this chain
        translation_distance = chain_idx * (chain_extent + spacing)
        
        if stack_along_axis == "x":
            translation = np.array([translation_distance, 0.0, 0.0])
        elif stack_along_axis == "y":
            translation = np.array([0.0, translation_distance, 0.0])
        else:  # stack_along_axis == "z"
            translation = np.array([0.0, 0.0, translation_distance])
        
        # Add translated positions
        for pos in chain_positions:
            new_pos = Vec3(pos.x + translation[0], 
                           pos.y + translation[1], 
                           pos.z + translation[2]) * nanometer
            final_positions.append(new_pos)
    
    return final_topology, final_positions

def build_sandwich(topology, positions,
                   num_sheets=2, nchains_per_sheet=8,
                   spacing=3.0, pattern="parallel",
                   fibre_axis="z",
                   stack_sheet_axis="y",
                   layer_separation=2.0,
                   spacing_method="backbone",
                   theta=0.0, 
                   verbose=False):
    """
    Create a sandwich structure with two rectangular sheets of chains facing each other.
    First creates one sheet using make_sheet, then duplicates it.
    Parameters:
    -----------
    topology : openmm.app.Topology
        Original single chain topology
    positions : list of Vec3
        Original single chain positions
    num_sheets : int
        Number of sheets (default 2)
    nchains_per_sheet : int
        Number of chains per sheet (default 8)
    spacing : float
        Spacing between chains within a sheet in nanometers (default 3.0)
    pattern : str
        choices=[parallel, antiparallel]
        Whether to make all of the coils parallel, or to flip every other one
    stack_along_axis : str
        choices = ["x", "z"]
        which axis to stack along
    layer_separation : float
        Distance between the two sheets in nanometers (default 2.0)
    spacing_method : str
        choices=["backbone", "bounding_box"]
        Method to calculate intersheet spacing (default "backbone")
    theta : float
        How much to rotate each chain around the axis defined by its length
    
    Returns:
    --------
    final_topology : openmm.app.Topology
        Topology with all chains from both sheets
    final_positions : list of Vec3
        Positions with all chains arranged in sandwich
    """
    if spacing_method not in ["backbone", "bounding_box"]:
        raise ValueError(f"{spacing_method=} must be in ['backbone', 'bounding_box']")
    
    # Build the first sheet
    # sheet1_topology, sheet1_positions = build_rectangle(
    #     topology, positions,
    #     num_rows=num_chains_per_row,
    #     num_chains_per_row=num_rows,
    #     spacing=spacing,
    #     pattern=pattern,
    #     stack_along_axis=fibre_axis
    # )
    sheet1_topology, sheet1_positions = make_sheet(
        topology, 
        positions,
        num_chains=nchains_per_sheet,
        spacing=spacing,
        pattern=pattern,
        pattern_plane="xz",
        stack_along_axis=fibre_axis,
        spacing_method=spacing_method,
        theta=theta
    )

    if stack_sheet_axis not in ["x","y", "z"]:
        raise ValueError(f"{stack_sheet_axis=} must be in ['x','y','z']")
    if stack_sheet_axis==fibre_axis:
        raise ValueError(f"{stack_sheet_axis=} and {fibre_axis=} must be different")
    
    translation_axis = stack_sheet_axis
    # Get the axis perpendicular to the sheet for translation
    # if stack_along_axis == "z":
    #     translation_axis = "x"  # Sheet is in y-z plane, translate along x
    # else:  # stack_along_axis == "x"
    #     translation_axis = "z"  # Sheet is in x-y plane, translate along z
    
    # Calculate sheet extent based on spacing_method
    if spacing_method == "backbone":
        # Use only backbone atoms
        bb_names = {"N", "CA", "C"}#, "O"}
        bb_idx = [i for i, a in enumerate(sheet1_topology.atoms()) if a.name in bb_names]
        if not bb_idx:
            raise ValueError("No backbone atoms (N, CA, C, O) found.")
        
        if translation_axis == "x":
            coords = [sheet1_positions[i].x for i in bb_idx]
        if translation_axis == "y":
            coords = [sheet1_positions[i].y for i in bb_idx]
        else:  # translation_axis == "z"
            coords = [sheet1_positions[i].z for i in bb_idx]
    else:  # spacing_method == "bounding_box"
        # Use all atoms
        if translation_axis == "x":
            coords = [p.x for p in sheet1_positions]
        if translation_axis == "y":
            coords = [p.y for p in sheet1_positions]
        else:  # translation_axis == "z"
            coords = [p.z for p in sheet1_positions]
    
    sheet1_min = min(coords)
    sheet1_max = max(coords)
    sheet1_extent = sheet1_max - sheet1_min
    
    # Create final topology
    final_topology = Topology()
    final_positions = []
    
    # Copy first sheet
    atoms_per_chain   = topology.getNumAtoms()
    
    chain_count = 0
    for chain in sheet1_topology.chains():
        new_chain = final_topology.addChain(make_chain_id(chain_count))
        
        for residue in chain.residues():
            new_residue = final_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                final_topology.addAtom(atom.name, atom.element, new_residue)
        
        chain_count += 1
    
    # Copy bonds from sheet1
    atoms_list = list(final_topology.atoms())
    for bond in sheet1_topology.bonds():
        atom1_idx = bond[0].index
        atom2_idx = bond[1].index
        final_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
    
    # Add sheet1 positions
    final_positions.extend(sheet1_positions)
    
    # Create second sheet (rotated and translated)
    # Add second sheet atoms
    for chain in sheet1_topology.chains():
        new_chain = final_topology.addChain(make_chain_id(chain_count))
        
        for residue in chain.residues():
            new_residue = final_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                final_topology.addAtom(atom.name, atom.element, new_residue)
        
        chain_count += 1
    
    # UPDATE atoms_list after adding second sheet's atoms
    atoms_list = list(final_topology.atoms())
    
    # Copy bonds from sheet2
    atom_offset = nchains_per_sheet * atoms_per_chain
    for bond in sheet1_topology.bonds():
        atom1_idx = bond[0].index + atom_offset
        atom2_idx = bond[1].index + atom_offset
        final_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
    
    # Rotate and translate each chain in sheet2
    for chain_idx in range(nchains_per_sheet):
        # Extract positions for this chain
        start_idx = chain_idx * atoms_per_chain
        end_idx   = start_idx + atoms_per_chain
        chain_positions = sheet1_positions[start_idx:end_idx]
        
        # Rotate chain by 180 degrees around its length axis
        rotated_positions = rotate_around_length_axis(topology, chain_positions, np.pi)
        
        # Translate the rotated chain
        for pos in rotated_positions:
            if translation_axis == "x":
                # Translate along x-axis
                translation = sheet1_extent + layer_separation
                new_pos = Vec3(pos.x + translation, pos.y, pos.z) * nanometer
            elif translation_axis == "y":
                # Translate along x-axis
                translation = sheet1_extent + layer_separation
                new_pos = Vec3(pos.x, pos.y + translation, pos.z) * nanometer
            else:  # translation_axis == "z"
                # Translate along z-axis
                translation = sheet1_extent + layer_separation
                new_pos = Vec3(pos.x, pos.y, pos.z + translation) * nanometer
            
            final_positions.append(new_pos)
    
    if verbose:
        print(f"Created sandwich with 2 sheets of {nchains_per_sheet=} chains each")
        print(f"Total chains: {chain_count}, Layer separation: {layer_separation} nm")
        print(f"Spacing method: {spacing_method}, Translation axis: {translation_axis}")
    
    return final_topology, final_positions

def stack_chains(topology, positions, num_chains=8, spacing=3.0, twist_angle=0, pattern="parallel", verbose=False):
    """
    Stack multiple copies of a chain vertically along the z-axis
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        Original single chain topology
    positions : list of Vec3
        Original single chain positions
    num_chains : int
        Number of chains to stack (default 8)
    spacing : float
        Vertical spacing between chains in nanometers (default 3.0)
    twist_angle : float (-180, 180)
        If non-zero, rotate each chain around z-axis by the twist angle, relative to the previous chain.
    pattern: str, ["parallel","antiparalle"]
    
    Returns:
    --------
    final_topology : openmm.app.Topology
        Topology with all stacked chains
    final_positions : list of Vec3
        Positions with all chains stacked vertically
    """
    if num_chains>52:
        raise Warning(f"{num_chains=}>52=26+26. Some chains may have the same chain ID.")

    final_topology = Topology()
    final_positions = []
    
    atoms_per_chain = topology.getNumAtoms()
    
    if verbose:
        print(f"Stacking {num_chains} chains with {spacing} nm vertical spacing, and twist {twist_angle}deg")
    
    if twist_angle!=0:
        _total_rotation_angle = twist_angle

    for chain_num in range(num_chains):
        # Create chain
        chain_id = make_chain_id(chain_num)
        if verbose:
            print(f"{chain_id=}")
        new_chain = final_topology.addChain(chain_id)
        
        # Copy residues and atoms
        for residue in topology.residues():
            new_residue = final_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                final_topology.addAtom(atom.name, atom.element, new_residue)
        
        # Copy bonds
        atom_offset = chain_num * atoms_per_chain
        atoms_list = list(final_topology.atoms())
        for bond in topology.bonds():
            atom1_idx = bond[0].index + atom_offset
            atom2_idx = bond[1].index + atom_offset
            final_topology.addBond(atoms_list[atom1_idx], atoms_list[atom2_idx])
        
        # Calculate z-translation for this chain
        z_offset = chain_num * spacing
        
        # Optional rotation around z-axis
        rotation_matrix = None
        if twist_angle!=0:
            # convert to radians
            angle = (twist_angle/180)*np.pi

            # compute rotation matrix
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
        
        # Add transformed positions
        _tmp_chain_new_positions = []
        for pos in positions:
            # Convert to numpy for rotation
            pos_array = np.array([pos.x, pos.y, pos.z])
            
            # Apply z-translation
            new_pos = Vec3(
                pos_array[0], 
                pos_array[1], 
                pos_array[2] + z_offset) * nanometer
            
            # final_positions.append(new_pos)
            _tmp_chain_new_positions.append(new_pos)

        # perform flip if antiparallel pattern
        if pattern=="antiparallel":
            _condn1 = (chain_num%2 == 1)
            if _condn1:
                # final_positions = center_chain_then_rotate_xy(topology, final_positions)
                _tmp_chain_new_positions = center_chain_then_rotate_xz(topology, _tmp_chain_new_positions)

        if ((twist_angle!=0) and (chain_num>0)):
            # final_positions = center_chain_then_rotate_xy(topology, final_positions, rotation_angle=twist_angle)
            _tmp_chain_new_positions = center_chain_then_rotate_xy(topology, _tmp_chain_new_positions, rotation_angle=_total_rotation_angle)
            _total_rotation_angle += twist_angle

        final_positions += _tmp_chain_new_positions

    return final_topology, final_positions


def check_for_overlap(topology, positions, chain1="A", chain2="B", 
                      bbox_atoms="all", tolerance=0.0, verbose=True):
    """
    Check if two chains' bounding boxes overlap.
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        Topology containing the chains
    positions : list of Vec3
        Positions of all atoms
    chain1 : str
        Chain ID of first chain (default "A")
    chain2 : str
        Chain ID of second chain (default "B")
    bbox_atoms : str
        choices=["all", "backbone", "heavy", "ca"]
        Which atoms to include in bounding box calculation:
        - "all": all atoms
        - "backbone": N, CA, C, O atoms
        - "heavy": all non-hydrogen atoms
        - "ca": only CA (alpha carbon) atoms
    tolerance : float
        Additional buffer distance in nanometers (default 0.0)
        Positive values expand bounding boxes, negative values shrink them
    verbose : bool
        If True, print detailed information about the bounding boxes
        and overlap status (default True)
    
    Returns:
    --------
    overlap : bool
        True if bounding boxes overlap, False otherwise
    
    Notes:
    ------
    - Returns False if either chain is not found in topology
    - Bounding boxes are axis-aligned (AABB - Axis-Aligned Bounding Box)
    - Tolerance can be used to check for near-misses or require separation
    """
    if bbox_atoms not in ["all", "backbone", "heavy", "ca"]:
        raise ValueError(f"{bbox_atoms=} must be in ['all', 'backbone', 'heavy', 'ca']")
    
    # Find the chains in topology
    chain1_obj = None
    chain2_obj = None
    for chain in topology.chains():
        if chain.id == chain1:
            chain1_obj = chain
        if chain.id == chain2:
            chain2_obj = chain
    
    if chain1_obj is None:
        if verbose:
            print(f"Warning: Chain {chain1} not found in topology")
        return False
    
    if chain2_obj is None:
        if verbose:
            print(f"Warning: Chain {chain2} not found in topology")
        return False
    
    # Helper function to filter atoms based on bbox_atoms option
    def should_include_atom(atom):
        if bbox_atoms == "all":
            return True
        elif bbox_atoms == "backbone":
            return atom.name in {"N", "CA", "C", "O"}
        elif bbox_atoms == "heavy":
            return atom.element.symbol != "H"
        elif bbox_atoms == "ca":
            return atom.name == "CA"
        return False
    
    # Extract positions for each chain
    def get_chain_positions(chain_obj):
        chain_positions = []
        for atom in topology.atoms():
            if atom.residue.chain.id == chain_obj.id and should_include_atom(atom):
                chain_positions.append(positions[atom.index])
        return chain_positions
    
    pos1 = get_chain_positions(chain1_obj)
    pos2 = get_chain_positions(chain2_obj)
    
    if not pos1:
        if verbose:
            print(f"Warning: No atoms found for chain {chain1} with bbox_atoms='{bbox_atoms}'")
        return False
    
    if not pos2:
        if verbose:
            print(f"Warning: No atoms found for chain {chain2} with bbox_atoms='{bbox_atoms}'")
        return False
    
    # Calculate bounding boxes
    def get_bbox(positions_list):
        x_coords = [p.x for p in positions_list]
        y_coords = [p.y for p in positions_list]
        z_coords = [p.z for p in positions_list]
        
        return {
            'min': np.array([min(x_coords), min(y_coords), min(z_coords)]),
            'max': np.array([max(x_coords), max(y_coords), max(z_coords)]),
            'center': np.array([
                (min(x_coords) + max(x_coords)) / 2,
                (min(y_coords) + max(y_coords)) / 2,
                (min(z_coords) + max(z_coords)) / 2
            ]),
            'size': np.array([
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords),
                max(z_coords) - min(z_coords)
            ])
        }
    
    bbox1 = get_bbox(pos1)
    bbox2 = get_bbox(pos2)
    
    # Apply tolerance (expand/shrink bounding boxes)
    bbox1_min = bbox1['min'] - tolerance
    bbox1_max = bbox1['max'] + tolerance
    bbox2_min = bbox2['min'] - tolerance
    bbox2_max = bbox2['max'] + tolerance
    
    # Check for overlap in each dimension
    # Bounding boxes overlap if they overlap in ALL three dimensions
    overlap_x = not (bbox1_max[0] < bbox2_min[0] or bbox2_max[0] < bbox1_min[0])
    overlap_y = not (bbox1_max[1] < bbox2_min[1] or bbox2_max[1] < bbox1_min[1])
    overlap_z = not (bbox1_max[2] < bbox2_min[2] or bbox2_max[2] < bbox1_min[2])
    
    overlap = overlap_x and overlap_y and overlap_z
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Bounding Box Overlap Check")
        print(f"{'='*60}")
        print(f"Chain 1: {chain1}, Chain 2: {chain2}")
        print(f"Atoms considered: {bbox_atoms}")
        print(f"Tolerance: {tolerance:.3f} nm")
        print(f"\nChain {chain1}:")
        print(f"  Atoms: {len(pos1)}")
        print(f"  Min: [{bbox1['min'][0]:.3f}, {bbox1['min'][1]:.3f}, {bbox1['min'][2]:.3f}] nm")
        print(f"  Max: [{bbox1['max'][0]:.3f}, {bbox1['max'][1]:.3f}, {bbox1['max'][2]:.3f}] nm")
        print(f"  Size: [{bbox1['size'][0]:.3f}, {bbox1['size'][1]:.3f}, {bbox1['size'][2]:.3f}] nm")
        print(f"  Center: [{bbox1['center'][0]:.3f}, {bbox1['center'][1]:.3f}, {bbox1['center'][2]:.3f}] nm")
        print(f"\nChain {chain2}:")
        print(f"  Atoms: {len(pos2)}")
        print(f"  Min: [{bbox2['min'][0]:.3f}, {bbox2['min'][1]:.3f}, {bbox2['min'][2]:.3f}] nm")
        print(f"  Max: [{bbox2['max'][0]:.3f}, {bbox2['max'][1]:.3f}, {bbox2['max'][2]:.3f}] nm")
        print(f"  Size: [{bbox2['size'][0]:.3f}, {bbox2['size'][1]:.3f}, {bbox2['size'][2]:.3f}] nm")
        print(f"  Center: [{bbox2['center'][0]:.3f}, {bbox2['center'][1]:.3f}, {bbox2['center'][2]:.3f}] nm")
        
        # Calculate distances between bounding boxes
        gap_x = max(0, max(bbox1_min[0], bbox2_min[0]) - min(bbox1_max[0], bbox2_max[0]))
        gap_y = max(0, max(bbox1_min[1], bbox2_min[1]) - min(bbox1_max[1], bbox2_max[1]))
        gap_z = max(0, max(bbox1_min[2], bbox2_min[2]) - min(bbox1_max[2], bbox2_max[2]))
        
        center_distance = np.linalg.norm(bbox1['center'] - bbox2['center'])
        
        print(f"\nOverlap Analysis:")
        print(f"  X-axis: {'OVERLAP' if overlap_x else 'NO OVERLAP'} (gap: {gap_x:.3f} nm)")
        print(f"  Y-axis: {'OVERLAP' if overlap_y else 'NO OVERLAP'} (gap: {gap_y:.3f} nm)")
        print(f"  Z-axis: {'OVERLAP' if overlap_z else 'NO OVERLAP'} (gap: {gap_z:.3f} nm)")
        print(f"  Center-to-center distance: {center_distance:.3f} nm")
        print(f"\nResult: {'BOUNDING BOXES OVERLAP' if overlap else 'NO OVERLAP'}")
        print(f"{'='*60}\n")
    
    return overlap

def check_for_overlap(topology, positions, chain1="A", chain2="B", 
                      bbox_atoms="all", tolerance=0.0, verbose=True,
                      method="bounding_box"):
    """
    Check if two chains' bounding volumes overlap.
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        Topology containing the chains
    positions : list of Vec3
        Positions of all atoms
    chain1 : str
        Chain ID of first chain (default "A")
    chain2 : str
        Chain ID of second chain (default "B")
    bbox_atoms : str
        choices=["all", "backbone", "heavy", "ca"]
        Which atoms to include in bounding volume calculation:
        - "all": all atoms
        - "backbone": N, CA, C, O atoms
        - "heavy": all non-hydrogen atoms
        - "ca": only CA (alpha carbon) atoms
    tolerance : float
        Additional buffer distance in nanometers (default 0.0)
        Positive values expand bounding volumes, negative values shrink them
    verbose : bool
        If True, print detailed information about the bounding volumes
        and overlap status (default True)
    method : str
        choices=["bounding_box", "parallelogram", "obb"]
        Method for overlap detection:
        - "bounding_box": Axis-aligned bounding box (AABB)
        - "parallelogram": Oriented bounding parallelogram aligned with chain axis
        - "obb": Oriented bounding box (full 3D orientation)
    
    Returns:
    --------
    overlap : bool
        True if bounding volumes overlap, False otherwise
    
    Notes:
    ------
    - Returns False if either chain is not found in topology
    - "parallelogram" method creates a bounding volume oriented along the
      backbone axis (typically x-axis for coiled coils)
    - "obb" method uses principal component analysis to find optimal orientation
    - Tolerance can be used to check for near-misses or require separation
    """
    if bbox_atoms not in ["all", "backbone", "heavy", "ca"]:
        raise ValueError(f"{bbox_atoms=} must be in ['all', 'backbone', 'heavy', 'ca']")
    
    if method not in ["bounding_box", "parallelogram", "obb"]:
        raise ValueError(f"{method=} must be in ['bounding_box', 'parallelogram', 'obb']")
    
    # Find the chains in topology
    chain1_obj = None
    chain2_obj = None
    for chain in topology.chains():
        if chain.id == chain1:
            chain1_obj = chain
        if chain.id == chain2:
            chain2_obj = chain
    
    if chain1_obj is None:
        if verbose:
            print(f"Warning: Chain {chain1} not found in topology")
        return False
    
    if chain2_obj is None:
        if verbose:
            print(f"Warning: Chain {chain2} not found in topology")
        return False
    
    # Helper function to filter atoms based on bbox_atoms option
    def should_include_atom(atom):
        if bbox_atoms == "all":
            return True
        elif bbox_atoms == "backbone":
            return atom.name in {"N", "CA", "C", "O"}
        elif bbox_atoms == "heavy":
            return atom.element.symbol != "H"
        elif bbox_atoms == "ca":
            return atom.name == "CA"
        return False
    
    # Extract positions for each chain
    def get_chain_positions(chain_obj):
        chain_positions = []
        for atom in topology.atoms():
            if atom.residue.chain.id == chain_obj.id and should_include_atom(atom):
                chain_positions.append(positions[atom.index])
        return chain_positions
    
    pos1 = get_chain_positions(chain1_obj)
    pos2 = get_chain_positions(chain2_obj)
    
    if not pos1:
        if verbose:
            print(f"Warning: No atoms found for chain {chain1} with bbox_atoms='{bbox_atoms}'")
        return False
    
    if not pos2:
        if verbose:
            print(f"Warning: No atoms found for chain {chain2} with bbox_atoms='{bbox_atoms}'")
        return False
    
    # Route to appropriate method
    if method == "bounding_box":
        return _check_aabb_overlap(pos1, pos2, chain1, chain2, tolerance, verbose)
    elif method == "parallelogram":
        return _check_parallelogram_overlap(topology, pos1, pos2, chain1, chain2, 
                                            tolerance, verbose)
    else:  # method == "obb"
        return _check_obb_overlap(pos1, pos2, chain1, chain2, tolerance, verbose)

def _check_aabb_overlap(pos1, pos2, chain1, chain2, tolerance, verbose):
    """Check axis-aligned bounding box overlap."""
    
    def get_bbox(positions_list):
        x_coords = [p.x for p in positions_list]
        y_coords = [p.y for p in positions_list]
        z_coords = [p.z for p in positions_list]
        
        return {
            'min': np.array([min(x_coords), min(y_coords), min(z_coords)]),
            'max': np.array([max(x_coords), max(y_coords), max(z_coords)]),
            'center': np.array([
                (min(x_coords) + max(x_coords)) / 2,
                (min(y_coords) + max(y_coords)) / 2,
                (min(z_coords) + max(z_coords)) / 2
            ]),
            'size': np.array([
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords),
                max(z_coords) - min(z_coords)
            ])
        }
    
    bbox1 = get_bbox(pos1)
    bbox2 = get_bbox(pos2)
    
    # Apply tolerance
    bbox1_min = bbox1['min'] - tolerance
    bbox1_max = bbox1['max'] + tolerance
    bbox2_min = bbox2['min'] - tolerance
    bbox2_max = bbox2['max'] + tolerance
    
    # Check for overlap in each dimension
    overlap_x = not (bbox1_max[0] < bbox2_min[0] or bbox2_max[0] < bbox1_min[0])
    overlap_y = not (bbox1_max[1] < bbox2_min[1] or bbox2_max[1] < bbox1_min[1])
    overlap_z = not (bbox1_max[2] < bbox2_min[2] or bbox2_max[2] < bbox1_min[2])
    
    overlap = overlap_x and overlap_y and overlap_z
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Axis-Aligned Bounding Box Overlap Check")
        print(f"{'='*60}")
        print(f"Chain 1: {chain1}, Chain 2: {chain2}")
        print(f"Tolerance: {tolerance:.3f} nm")
        print(f"\nChain {chain1}:")
        print(f"  Atoms: {len(pos1)}")
        print(f"  Min: [{bbox1['min'][0]:.3f}, {bbox1['min'][1]:.3f}, {bbox1['min'][2]:.3f}] nm")
        print(f"  Max: [{bbox1['max'][0]:.3f}, {bbox1['max'][1]:.3f}, {bbox1['max'][2]:.3f}] nm")
        print(f"  Size: [{bbox1['size'][0]:.3f}, {bbox1['size'][1]:.3f}, {bbox1['size'][2]:.3f}] nm")
        print(f"  Center: [{bbox1['center'][0]:.3f}, {bbox1['center'][1]:.3f}, {bbox1['center'][2]:.3f}] nm")
        print(f"\nChain {chain2}:")
        print(f"  Atoms: {len(pos2)}")
        print(f"  Min: [{bbox2['min'][0]:.3f}, {bbox2['min'][1]:.3f}, {bbox2['min'][2]:.3f}] nm")
        print(f"  Max: [{bbox2['max'][0]:.3f}, {bbox2['max'][1]:.3f}, {bbox2['max'][2]:.3f}] nm")
        print(f"  Size: [{bbox2['size'][0]:.3f}, {bbox2['size'][1]:.3f}, {bbox2['size'][2]:.3f}] nm")
        print(f"  Center: [{bbox2['center'][0]:.3f}, {bbox2['center'][1]:.3f}, {bbox2['center'][2]:.3f}] nm")
        
        gap_x = max(0, max(bbox1_min[0], bbox2_min[0]) - min(bbox1_max[0], bbox2_max[0]))
        gap_y = max(0, max(bbox1_min[1], bbox2_min[1]) - min(bbox1_max[1], bbox2_max[1]))
        gap_z = max(0, max(bbox1_min[2], bbox2_min[2]) - min(bbox1_max[2], bbox2_max[2]))
        center_distance = np.linalg.norm(bbox1['center'] - bbox2['center'])
        
        print(f"\nOverlap Analysis:")
        print(f"  X-axis: {'OVERLAP' if overlap_x else 'NO OVERLAP'} (gap: {gap_x:.3f} nm)")
        print(f"  Y-axis: {'OVERLAP' if overlap_y else 'NO OVERLAP'} (gap: {gap_y:.3f} nm)")
        print(f"  Z-axis: {'OVERLAP' if overlap_z else 'NO OVERLAP'} (gap: {gap_z:.3f} nm)")
        print(f"  Center-to-center distance: {center_distance:.3f} nm")
        print(f"\nResult: {'BOUNDING BOXES OVERLAP' if overlap else 'NO OVERLAP'}")
        print(f"{'='*60}\n")
    
    return overlap

def _check_parallelogram_overlap(topology, pos1, pos2, chain1, chain2, tolerance, verbose):
    """
    Check oriented parallelogram overlap.
    
    Creates a parallelogram aligned with the chain's backbone axis (typically x-axis).
    The parallelogram is defined by:
    - Length along the backbone axis
    - Width in the perpendicular directions
    
    Uses Separating Axis Theorem (SAT) for overlap detection.
    """
    
    def get_chain_axis_and_bounds(positions_list):
        """Get the backbone axis direction and bounding parallelogram."""
        # Convert positions to numpy array
        coords = np.array([[p.x, p.y, p.z] for p in positions_list])
        
        # Calculate center
        center = np.mean(coords, axis=0)
        
        # Get backbone direction (assume aligned with x-axis for coiled coils)
        # Or use PCA to find principal axis
        centered = coords - center
        
        # Use PCA to find principal axis (backbone direction)
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Principal axis (backbone direction)
        axis_main = eigenvectors[:, 0]
        axis_perp1 = eigenvectors[:, 1]
        axis_perp2 = eigenvectors[:, 2]
        
        # Project coordinates onto each axis
        proj_main = np.dot(centered, axis_main)
        proj_perp1 = np.dot(centered, axis_perp1)
        proj_perp2 = np.dot(centered, axis_perp2)
        
        # Get extents along each axis
        half_length = (np.max(proj_main) - np.min(proj_main)) / 2
        half_width1 = (np.max(proj_perp1) - np.min(proj_perp1)) / 2
        half_width2 = (np.max(proj_perp2) - np.min(proj_perp2)) / 2
        
        return {
            'center': center,
            'axes': [axis_main, axis_perp1, axis_perp2],
            'half_extents': np.array([half_length, half_width1, half_width2]),
            'eigenvalues': eigenvalues
        }
    
    # bb_names = {"N", "CA", "C", "O"}
    # bb1_idx = [i for i, a in enumerate(topology.atoms()) if a.name in bb_names]
    # if not bb1_idx:
    #     raise ValueError("No backbone atoms (N, CA, C, O) found.")

    obb1 = get_chain_axis_and_bounds( pos1 )
    obb2 = get_chain_axis_and_bounds( pos2 )
    
    # Apply tolerance to half extents
    obb1['half_extents'] += tolerance
    obb2['half_extents'] += tolerance
    
    # Separating Axis Theorem (SAT) for OBB overlap
    def sat_overlap_test(obb1, obb2):
        """
        Use Separating Axis Theorem to test overlap between two OBBs.
        Returns True if overlap, False otherwise.
        """
        # Test all 15 potential separating axes:
        # 3 face normals from obb1
        # 3 face normals from obb2
        # 9 cross products of edges
        
        axes_to_test = []
        
        # Face normals from both OBBs
        axes_to_test.extend(obb1['axes'])
        axes_to_test.extend(obb2['axes'])
        
        # Cross products of edges
        for i in range(3):
            for j in range(3):
                axis = np.cross(obb1['axes'][i], obb2['axes'][j])
                # Skip if cross product is near zero (parallel axes)
                if np.linalg.norm(axis) > 1e-6:
                    axes_to_test.append(axis / np.linalg.norm(axis))
        
        # Test each axis
        for axis in axes_to_test:
            axis = axis / np.linalg.norm(axis)  # Normalize
            
            # Project both OBBs onto this axis
            # For OBB1
            center_proj1 = np.dot(obb1['center'], axis)
            radius1 = 0
            radius2 = 0
            for i in range(3):
                radius1 = radius1 + abs(np.dot(obb1['axes'][i], axis)) * obb1['half_extents'][i]
                radius2 = radius2 + abs(np.dot(obb2['axes'][i], axis)) * obb2['half_extents'][i]
            
            # For OBB2
            center_proj2 = np.dot(obb2['center'], axis)
            
            # Check if projections overlap
            distance = abs(center_proj1 - center_proj2)
            if distance > radius1 + radius2:
                # Found a separating axis - no overlap
                return False
        
        # No separating axis found - must overlap
        return True
    
    overlap = sat_overlap_test(obb1, obb2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Oriented Parallelogram Overlap Check")
        print(f"{'='*60}")
        print(f"Chain 1: {chain1}, Chain 2: {chain2}")
        print(f"Tolerance: {tolerance:.3f} nm")
        print(f"\nChain {chain1}:")
        print(f"  Atoms: {len(pos1)}")
        print(f"  Center: [{obb1['center'][0]:.3f}, {obb1['center'][1]:.3f}, {obb1['center'][2]:.3f}] nm")
        print(f"  Half extents: [{obb1['half_extents'][0]:.3f}, {obb1['half_extents'][1]:.3f}, {obb1['half_extents'][2]:.3f}] nm")
        print(f"  Main axis: [{obb1['axes'][0][0]:.3f}, {obb1['axes'][0][1]:.3f}, {obb1['axes'][0][2]:.3f}]")
        print(f"  Eigenvalues: [{obb1['eigenvalues'][0]:.3f}, {obb1['eigenvalues'][1]:.3f}, {obb1['eigenvalues'][2]:.3f}]")
        
        print(f"\nChain {chain2}:")
        print(f"  Atoms: {len(pos2)}")
        print(f"  Center: [{obb2['center'][0]:.3f}, {obb2['center'][1]:.3f}, {obb2['center'][2]:.3f}] nm")
        print(f"  Half extents: [{obb2['half_extents'][0]:.3f}, {obb2['half_extents'][1]:.3f}, {obb2['half_extents'][2]:.3f}] nm")
        print(f"  Main axis: [{obb2['axes'][0][0]:.3f}, {obb2['axes'][0][1]:.3f}, {obb2['axes'][0][2]:.3f}]")
        print(f"  Eigenvalues: [{obb2['eigenvalues'][0]:.3f}, {obb2['eigenvalues'][1]:.3f}, {obb2['eigenvalues'][2]:.3f}]")
        
        center_distance = np.linalg.norm(obb1['center'] - obb2['center'])
        axis_angle = np.arccos(np.clip(np.dot(obb1['axes'][0], obb2['axes'][0]), -1.0, 1.0))
        
        print(f"\nOverlap Analysis:")
        print(f"  Center-to-center distance: {center_distance:.3f} nm")
        print(f"  Angle between main axes: {np.degrees(axis_angle):.1f}°")
        print(f"\nResult: {'PARALLELOGRAMS OVERLAP' if overlap else 'NO OVERLAP'}")
        print(f"{'='*60}\n")
    
    return overlap


def _check_obb_overlap(pos1, pos2, chain1, chain2, tolerance, verbose):
    """
    Check oriented bounding box (OBB) overlap.
    This is an alias for parallelogram check but emphasizes full 3D orientation.
    """
    return _check_parallelogram_overlap(None, pos1, pos2, chain1, chain2, tolerance, verbose)