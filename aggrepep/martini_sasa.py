import mdtraj as mdt
import numpy as np

import os
import warnings
from itertools import combinations

_MARTINI_VDW_RADII = {
    "R" :0.264, #nm
    "S" :0.230, #nm
    "T" :0.191, #nm
    "VS":None,
}

def get_martini_vdw_radii(top_fpath: str, martini_version: str | float) -> dict:
    """
    get_martini_vdw_radii returns a dictionary mapping Martini bead 
    types to their corresponding van der Waals radii in nanometers. 

    Parameters:
    -----------
        top_fpath: str
            The path to the topology file.
        martini_version: str|float
            The version of the Martini force field (e.g., "2.2", "3.0")
    
    """

    topology = mdt.load(top_fpath).topology
    # select protein 
    protein_atoms = topology.select("protein")
    topology = topology.subset(protein_atoms)

    _top_dir = os.path.dirname(top_fpath)
    itp_fpath = os.path.join(_top_dir, "molecule_0.itp") # assuming molecule_0.itp
    bead_to_beadtype, seq_length = _get_bead_type_from_itp(itp_fpath)
    print(bead_to_beadtype)
    atom_radii = []
    for atom in topology.atoms:
        if atom.name.startswith("VS") or atom.element.symbol == "VS":
            atom_radii.append(0.0)
            continue 
        _seqN = atom.residue.resSeq-seq_length*int((atom.residue.resSeq-1)/seq_length)
        _name = f"{atom.residue.name}{_seqN}-{atom.name}" # PHE1-BB, GLY2-BB, etc.
        # _name = f"{atom.residue.name}-{atom.name}" # PHE-BB, GLY-BB, etc.
        _beadtype = bead_to_beadtype.get(_name, None)
        if _beadtype is None:
            warnings.warn(f"Bead {_name} not found in .itp mapping. Defaulting to 0.264 nm.", UserWarning)
            atom_radii.append(0.264) # default radius for (R)egular beads
        else:
            _bead_size = _beadtype[0] # e.g., "P2" -> "P", "Qd" -> "Q", "SC5" -> "S"
            if _bead_size in ["S", "T"]:
                atom_radii.append(_MARTINI_VDW_RADII[_bead_size])
            else: # Regular bead, "R"
                atom_radii.append(_MARTINI_VDW_RADII["R"])
    return atom_radii

def _get_bead_type_from_itp(itp_fpath):
    """
    Uses the molecule_0.itp file to map bead names to their corresponding bead types.
    E.g. for FT10 peptide the first few atom lines are:
        PHE1-BB BB boron B
        PHE1-SC1 SC1 sulfur S
        PHE1-SC2 SC2 sulfur S
        PHE1-SC3 SC3 sulfur S
        GLY2-BB BB boron B
    and we map "PHE1-BB" to "BB", "PHE1-SC1" to "SC1", etc. 
    This allows us to assign vdW radii based on bead type.
    """
    # place warning that we are relying on itp file.
    warnings.warn("Relying on .itp file to map bead names to types. Should be made more flexible.", UserWarning)

    length=-1
    bead_to_beadtype = {}
    with open(itp_fpath, 'r') as f:
        lines = f.readlines()
        in_atoms_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("[ atoms ]"):
                in_atoms_section = True
                continue
            if in_atoms_section:
                if line.startswith("[") or not line:
                    break  # end of atoms section
                parts = line.split()
                if len(parts) >= 5:
                    bead_type = parts[1]  # e.g., "P2", "Qd", "SC5"
                    res_n     = parts[2]  # e.g., "1"
                    res_name  = parts[3]  # e.g., "ALA", "PHE"
                    bead_name = parts[4]  # e.g., "BB", "SC1"
                    _full_bead_name = f"{res_name}{res_n}-{bead_name}"
                    # _full_bead_name = f"{res_name}-{bead_name}"
                    bead_to_beadtype[_full_bead_name] = bead_type # ex GLY2-BB : Nda
                    length = max(length, int(res_n))
    return bead_to_beadtype, length
