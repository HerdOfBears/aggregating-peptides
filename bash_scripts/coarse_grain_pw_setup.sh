#!/bin/bash
set -eou pipefail

PROTEIN_PDB=$1
NRES=$2

# take as argument or default to current directory
WDIR=${3:-$(pwd)}
cd $WDIR

# usage error message
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <protein_pdb_file> <number_of_residues> [working_directory]"
    exit 1
fi

VENV_DIR=$HOME/projects/venvs

SECONDARY_STRUCTURE=$(printf 'E%.0s' $(seq 1 $NRES))

# elastic network params
EN_lower=0
EN_upper=0.85

# insane/simulation box params
NMOL=50
BOX_L=15
SALT_CONCENTRATION=0.15

# file names
PROTEIN_NAME="${PROTEIN_PDB%.*}"
PROTEIN_CG_PDB=$PROTEIN_NAME"_cg.pdb"
PROTEIN_ONLY_TOP=$PROTEIN_NAME"_only.top"
SYSTEM_GRO="system.gro"
SOL_GRO="solvated.gro"
SYSTEM_TOP="system.top"

#############################################
# start virtual environment
source $VENV_DIR/venv-cg/bin/activate

echo "ignoring hydrogens in inputted structure (martinize2 -ignh ...)"
#############################################
# martinize with Elastic Network model
# CHANGED: -ff martini22 (Martini 2.2 protein params, compatible with 2.3P polarizable water)
martinize2 -f $PROTEIN_PDB -x $PROTEIN_CG_PDB -o $PROTEIN_ONLY_TOP \
  -ff martini22 -p backbone -ss "$SECONDARY_STRUCTURE" \
  -elastic -el $EN_lower -eu $EN_upper \
  -noscfix \
  -ignh

#############################################
# make copies of the coarse-grained chain
gmx insert-molecules -ci $PROTEIN_CG_PDB -nmol $NMOL -box $BOX_L $BOX_L $BOX_L -o $SYSTEM_GRO

#############################################
# copy the .top file from martinize and prepare one for the solvated system
cp $PROTEIN_ONLY_TOP $SYSTEM_TOP
sed -i "s/^molecule_0 .*/molecule_0         $NMOL/" $SYSTEM_TOP

# CHANGED for polarizable water: 
# martini_v2.3P.itp contains the FULL interaction matrix plus polarizable water definitions
# It replaces both the main FF itp and the solvents itp
# Ion itp is still separate
sed -i 's/ AC1 / C1 /g' molecule_0.itp
sed -i 's/ AC2 / C2 /g' molecule_0.itp

sed -i 's|#include "martini.itp"|#include "martini/martini_v2.3P.itp"\n#include "martini/martini_v2.0_ions.itp"|' $SYSTEM_TOP
sed -i '/./,$!d' $SYSTEM_TOP

#############################################
# solvate with insane
# CHANGED: -sol PW (polarizable water, 3 beads per water unit: W, WP, WM)
insane -f $SYSTEM_GRO -o $SOL_GRO \
  -pbc cubic \
  -salt $SALT_CONCENTRATION \
  -sol PW \
  -d 0 2>&1 | tail -n3 >> $SYSTEM_TOP

# CHANGED: In Martini 2, ion moleculetype names in the itp are typically NA+ and CL-
# which should match what insane outputs — so renaming may NOT be needed.
# Check your martini_v2.0_ions.itp to confirm the moleculetype names.
# If they are "NA+" and "CL-", comment out the following two lines:
# sed -i "s/^NA+ /NA /" $SYSTEM_TOP
# sed -i "s/^CL- /CL /" $SYSTEM_TOP
