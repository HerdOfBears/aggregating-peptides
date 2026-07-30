#!/bin/bash
set -eou pipefail

inPDB=$1
outPDB=$2

vmd -dispdev text -e bash_scripts/aa_neutralize_termini.tcl -args "$inPDB" "$outPDB"