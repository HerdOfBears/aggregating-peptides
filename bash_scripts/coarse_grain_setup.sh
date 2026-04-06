#!/bin/bash

set -eou pipefail

PROTEIN_PDB=$1
NRES=$2

# usage error message
if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <protein_pdb_file> <number_of_residues>"
	exit 1
fi
SECONDARY_STRUCTURE=$(printf 'E%.0s' $(seq 1 $NRES))

VENV_DIR=$HOME/projects/venvs

# elastic network params
EN_lower=0
EN_upper=0.85

# insane/simulation box params
NMOL=50 # number of protein chains to insert
BOX_L=15 # nm, box length in each dimension
SALT_CONCENTRATION=0.15 # auto neutralizes as well

# file names
PROTEIN_NAME="${PROTEIN_PDB%.*}" # "myprotein" from "myprotein.pdb"
PROTEIN_CG_PDB=$PROTEIN_NAME"_cg.pdb"
PROTEIN_ONLY_TOP=$PROTEIN_NAME"_only.top"

SYSTEM_GRO="system.gro"
SOL_GRO="solvated.gro"
SYSTEM_TOP="system.top"

#############################################
# start virtual environment
source $VENV_DIR/venv-cg/bin/activate

#############################################
# martinize with Elastic Network model
martinize2 -f $PROTEIN_PDB -x $PROTEIN_CG_PDB -o $PROTEIN_ONLY_TOP \
	-ff martini3001 -p backbone -ss "$SECONDARY_STRUCTURE" -elastic -el $EN_lower -eu $EN_upper

#############################################
# make copies of the coarse-grained chain
gmx insert-molecules -ci $PROTEIN_CG_PDB -nmol $NMOL -box $BOX_L $BOX_L $BOX_L -o $SYSTEM_GRO


#############################################
# copy the .top file from martinize and prepare one for the solvated system
cp $PROTEIN_ONLY_TOP $SYSTEM_TOP

# add the number of protein chains to the system.top file
sed -i "s/^molecule_0 .*/molecule_0         $NMOL/" $SYSTEM_TOP
sed -i 's|#include "martini.itp"|#include "martini/martini_v3.0.0.itp"\n#include "martini/martini_v3.0.0_solvents_v1.itp"\n#include "martini/martini_v3.0.0_ions_v1.itp"|' $SYSTEM_TOP
sed -i '/./,$!d' $SYSTEM_TOP

#############################################
# solvate with insane
# insane -f $SYSTEM_GRO -o $SOL_GRO -p $SYSTEM_TOP \
insane -f $SYSTEM_GRO -o $SOL_GRO \
	-pbc cubic \
	-salt $SALT_CONCENTRATION \
	-sol W \
	-d 0 2>&1 | tail -n3 >> $SYSTEM_TOP
	# -box $BOX_L,$BOX_L,$BOX_L \
#	-charge 11 # only need when using 
sed -i "s/^NA+ /NA /" $SYSTEM_TOP
sed -i "s/^CL- /CL /" $SYSTEM_TOP

# edit the system.top file to have the correct number of protein chains
# sed -i "s/^Protein .*/Protein         $NMOL/" $SYSTEM_TOP


