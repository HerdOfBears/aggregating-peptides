#!/bin/bash
set -eou pipefail

inPSF=$1
inPDB=$2
outName=$3
padding=$4

vmd -dispdev text -e bash_scripts/solvate_with_vmd.tcl -args \
    -psf "$inPSF" -pdb "$inPDB" -o "$outName" -axis z -pad "$padding" 