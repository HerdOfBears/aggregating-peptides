#!/bin/bash

# Pipeline script to set up all-atom 20-mer stack of protein in water simulation
# Usage: ./run_aa_pipeline.sh <sequence> <sequence_id>

set -eou pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <sequence> <sequence_id>"
    exit 1
fi

SEQUENCE=$1
SEQUENCE_ID=$2

platformName="OpenCL"
VENV_DIR=$HOME/projects/venvs
source $VENV_DIR/venv-cg/bin/activate

##########################################
# Step 1: Generate structure from sequence using sequence_to_structure.py
##########################################
python scripts/sequence_to_structure.py \
                --sequence "$SEQUENCE" \
                --sequence_id "$SEQUENCE_ID" \
                --arrangement "parallel" \
                --output_dir "outputs/" \
                --build_stacked_sheets

##########################################
# Step 2: Run equilibation: energy min, nvt and npt equilibrations
##########################################
randomSeed=42
jobPrefix=$SEQUENCE_ID"_parallel"
jobName=$jobPrefix"_rs"$randomSeed
wDir=outputs/$SEQUENCE_ID
inputFile=$jobPrefix".pdb"

python ~/additional_repos/structure-simulation/scripts/run_equilibration.py \
        --pdb_file $inputFile \
        --input_dir $wDir \
        --output_dir $wDir \
        --job_name $jobName \
        --platform_name $platformName \
        --params_file params.json \
        --random_seed $randomSeed

##########################################
# Step 3: Run production
##########################################
jobName=$jobPrefix"_rs"$randomSeed"_"
inputFile=$jobName"solvated_system._pbcFixed.pdb"
checkpointFile=$jobName"nvt_npt_equilibrated_system.xml"
paramsFile=params.json
python ~/additional_repos/structure-simulation/scripts/run_production.py \
        --input_pdb $inputFile \
        --checkpoint_fname $checkpointFile \
        --input_dir $wDir \
        --output_dir $wDir \
        --job_name $jobName \
        --params_file $paramsFile \
        --platform_name $platformName \
        --random_seed $randomSeed

##########################################
# Step 4: Run analysis
##########################################
trajFile=$jobName"npt_production_system_traj.dcd"
topFile=$jobName"solvated_system._pbcFixed.pdb"
trajFileNW=$jobName"npt_production_system_traj_pbcfix.dcd"

# fix pbcs
echo "fixing pbcs.."
python scripts/fix_pbcs.py \
        --input $wDir/$trajFile \
        --output $wDir/$trajFileNW \
        --topology $topFile
echo "done fixing pbcs"
echo "running analysis.."
# run analysis
python scripts/run_AP_analysis.py \
    --pepid $SEQUENCE_ID \
    --sequence $SEQUENCE \
    --wdir $wDir \
    --traj $trajFileNW \
    --top  $SEQUENCE_ID"_parallel.pdb" \
    --rseed $randomSeed \
    --dataset_file $analysisResults
echo "Done all-atom for $SEQUENCE_ID"
