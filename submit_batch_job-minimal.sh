#!/bin/bash
#SBATCH --job-name=JOB_NAME 	 # adjust this for yourself 
#SBATCH --account=ACCOUNT_NAME   # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-01:05:00        # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=1        # number of cpus needed
#SBATCH --gpus=h100:1            # gpus needed
#SBATCH --mem=20G                # adjust this according to the memory you need

# replace USER and EXPERIMENT_NAME with what you want
WDIR=/scratch/USER/EXPERIMENT_NAME/

inputFile="data/input_files/sequences-morphology-experiment.csv"
paramsFile="params-morph-Ceq30p2.json"

echo "wdir="$WDIR
echo "paramsFile="$paramsFile
echo "inputFile="$inputFile

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6
module load vmd/1.9.4a57
module load gromacs/2026.1 openmm/8.4.0

# the same virtual environment as specified in the coarse_grained_pw_setup bash script
source $HOME/venvs/venv-cg/bin/activate
pip install .

echo "starting batch of jobs"
python scripts/driver_batch_sequences.py --input_file $inputFile --wdir $WDIR --params_file $paramsFile --n_jobs 5

