# ruff: noqa: F405

import mdtraj as md
import numpy as np
import logging

import sys
from structSim.openmm_helpers import fix_pdb_periodic_boundaries_and_save


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(   "--input", required=True, help="File to have image molecule")
    parser.add_argument(  "--output", required=False, default=None, help="File name for saving")
    parser.add_argument("--topology", required=False, default=None, help="File for topology")
    parser.add_argument("--remove_solvent", required=False, default="yes", help="Whether to remove solvent")

    args = parser.parse_args()

    # Simple command line interface
    if len(sys.argv) < 2:
        print("Usage: python script.py input.pdb [output.pdb] [anchor_selection]")
        print("\nExamples:")
        print("  python script.py system.pdb")
        print("  python script.py system.pdb system_fixed.pdb")
        print("  python script.py system.pdb system_fixed.pdb 'resname LIG'")
        sys.exit(1)
    
    input_pdb  = args.input#sys.argv[1]
    output_pdb = args.output#sys.argv[2] if len(sys.argv) > 2 else None
    topology   = args.topology#sys.argv[3] if len(sys.argv) > 3 else 'protein'
    remove_solvent = args.remove_solvent
    if remove_solvent=="yes":
        remove_solvent=True
    else:
        remove_solvent=False
    protein_only = remove_solvent
    fix_pdb_periodic_boundaries_and_save(input_pdb, output_pdb, topology_file=topology, protein_only=protein_only)