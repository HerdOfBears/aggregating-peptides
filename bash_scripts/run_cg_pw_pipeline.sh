#!/bin/bash

# Pipeline script to set up a coarse-grained protein-water simulation
# Usage: ./setup_cg_pw_sim.sh <sequence> <sequence_id>

set -eou pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <sequence> <sequence_id>"
    exit 1
fi

SEQUENCE=$1
SEQUENCE_ID=$2

VENV_DIR=$HOME/projects/venvs
source $VENV_DIR/venv-cg/bin/activate

# Step 1: Generate structure from sequence using sequence_to_structure.py
python scripts/sequence_to_structure.py \
                --sequence "$SEQUENCE" \
                --sequence_id "$SEQUENCE_ID" \
                --arrangement "parallel" \
                --output_dir "outputs/"

# Step 2: set up the CG PW simulation
echo "Setting up CG PW simulation for $SEQUENCE_ID"
echo "pwd="$(pwd)
cp -r outputs/martini outputs/$SEQUENCE_ID
bash bash_scripts/coarse_grain_pw_setup.sh \
    "$SEQUENCE_ID.pdb" \
    ${#SEQUENCE} \
    "outputs/$SEQUENCE_ID"
echo "CG PW simulation setup complete for $SEQUENCE_ID"
echo "pwd="$(pwd)

# step 3: run a test simulation 
echo "Running test simulation for $SEQUENCE_ID"

python scripts/test_run_martini_openmm.py \
    --gro "outputs/$SEQUENCE_ID/solvated.gro" \
    --top "outputs/$SEQUENCE_ID/system.top" \
    --wdir "outputs/$SEQUENCE_ID/"

echo "CG PW simulation complete for $SEQUENCE_ID"