#!/bin/bash
set -eou pipefail

PROTEIN_PDB=$1
NRES=$2

# take as argument or default to current directory
OLD_DIR=$(pwd)
WDIR=${3:-$(pwd)}
AcNtermini=${4:-"n"} # whether to use an effective Acetylated N-termini or not
amidateCtermini=${5:-"n"} # whether to use an effective C-terminus amidation or not
NMOL=${6:-64}
BOX_L=${7:-13.3}

cd $WDIR

SCRIPT_DIR=$OLD_DIR"/scripts"
INITIAL_CONFORMATION="monodisperse" # [monodisperse, random]

# usage error message
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <protein_pdb_file> <number_of_residues> <working_directory> <whether_to_neutralize_Nterminus> <whether_to_neutralize_Cterminus>"
    exit 1
fi
if [ ! -f "$PROTEIN_PDB" ]; then
    echo "Error: Protein PDB file '$PROTEIN_PDB' not found."
    exit 1
fi
# check initial conformaiton choice
if [ "$INITIAL_CONFORMATION" != "monodisperse" ] && [ "$INITIAL_CONFORMATION" != "random" ]; then
    echo "Error: Invalid initial conformation choice '$INITIAL_CONFORMATION'. Must be 'monodisperse' or 'random'."
    exit 1
fi

VENV_DIR=$HOME/projects/venvs
VENV_DIR=$HOME/venvs

SECONDARY_STRUCTURE=$(printf 'E%.0s' $(seq 1 $NRES))

# elastic network params
EN_lower=0
EN_upper=0.85

# insane/simulation box params
#NMOL=64
#BOX_L=13.3
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
common_args=(-f "$PROTEIN_PDB" -x "$PROTEIN_CG_PDB" -o "$PROTEIN_ONLY_TOP"
             -ff martini22 -p backbone -ss "$SECONDARY_STRUCTURE"
             -elastic -el $EN_lower -eu $EN_upper -noscfix -ignh)

if [[ "$AcNtermini" == "y" && "$amidateCtermini" == "y" ]]; then
    martinize2 "${common_args[@]}" -nt          # neutral termini at both ends
elif [[ "$AcNtermini" == "y" ]]; then
    martinize2 "${common_args[@]}" -nter NH2-ter
elif [[ "$amidateCtermini" == "y" ]]; then
    martinize2 "${common_args[@]}" -cter COOH-ter
else
    martinize2 "${common_args[@]}"
fi

#############################################
# make copies of the coarse-grained chain
if [ "$INITIAL_CONFORMATION" == "monodisperse" ]; then
    echo "initial conformation: monodisperse (copies of the same chain)"
    # center the monomer at the origin (required for -ip absolute positions)
    gmx editconf -f $PROTEIN_CG_PDB -o ${PROTEIN_NAME}_centered.pdb -center 0 0 0

    # generate lattice positions
    python $SCRIPT_DIR/generate_lattice_points.py $NMOL $BOX_L positions.dat

    # place monomers at lattice points
    gmx insert-molecules -ci ${PROTEIN_NAME}_centered.pdb \
      -ip positions.dat \
      -nmol $NMOL \
      -box $BOX_L $BOX_L $BOX_L \
      -dr 0.1 0.1 0.1 \
      -o $SYSTEM_GRO
elif [ "$INITIAL_CONFORMATION" == "random" ]; then
    echo "initial conformation: random (copies of randomly rotated chains)"
    gmx insert-molecules -ci $PROTEIN_CG_PDB -nmol $NMOL -box $BOX_L $BOX_L $BOX_L -o $SYSTEM_GRO
else
    echo "Error: Invalid initial conformation choice '$INITIAL_CONFORMATION'. Must be 'monodisperse' or 'random'."
    exit 1
fi

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
  -x $BOX_L -y $BOX_L -z $BOX_L \
  -d 0 2>&1 | tail -n3 >> $SYSTEM_TOP

# CHANGED: In Martini 2, ion moleculetype names in the itp are typically NA+ and CL-
# which should match what insane outputs — so renaming may NOT be needed.
# Check your martini_v2.0_ions.itp to confirm the moleculetype names.
# If they are "NA+" and "CL-", comment out the following two lines:
# sed -i "s/^NA+ /NA /" $SYSTEM_TOP
# sed -i "s/^CL- /CL /" $SYSTEM_TOP
