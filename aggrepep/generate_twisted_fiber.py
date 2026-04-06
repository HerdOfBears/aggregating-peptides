import Bio.PDB
from PeptideBuilder import Geometry
import PeptideBuilder

from openmm.app import PDBFile

from aggrepep.helpers import biopython_to_pdbfixer_stringio, \
                                pdbfixer_workflow, \
                                stack_chains

if __name__=="__main__":

    # Define your sequence
    sequence = "GKIIKLKASLKLL"  # Your sequence here
    #sequence = "RADARADARADARADA"
    #sequence = "AGGREGATE"

    geo = Geometry.geometry(sequence[0])
    structure = PeptideBuilder.initialize_res(geo)
    for aa in sequence[1:]:
        structure = PeptideBuilder.add_residue(structure, aa)

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    # io.save("AGGREGATEpeptide.pdb")

    fixer = biopython_to_pdbfixer_stringio(structure)

    fixed = pdbfixer_workflow(fixer)

    topology  = fixed.topology
    positions = fixed.positions

    print("Warning: should optimize monomer's rotation about its longest axis before stacking")
    twist_top, twist_pos = stack_chains(topology, positions, num_chains=35, spacing=2.0, twist_angle=-10)
    twist_pos_lst = []
    for _vec in twist_pos:
        _lst = [_vec.x, _vec.y, _vec.z]
        twist_pos_lst.append(_lst)

    print("saving...")
    PDBFile.writeFile(
        twist_top, 
        twist_pos_lst, 
        open(f'peptide_{sequence}_twisted10deg.pdb', 'w'), 
        keepIds=True
    )
    print("done")