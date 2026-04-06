#!/bin/bash

set -eou pipefail

SEQUENCE=$1
SEQUENCE_ID=$2
USE_AA=${3:-"n"} # "y" for all-atom, "n" for coarse-grained

# check that USE_AA is in "y" or "n"
if [[ "$USE_AA" != "y" && "$USE_AA" != "n" ]]; then
    echo "Error: USE_AA must be 'y' or 'n'"
    exit 1
fi

echo "Running pipeline for sequence ID: $SEQUENCE_ID, all-atom: $USE_AA"
echo "Sequence: $SEQUENCE"
if [[ "$USE_AA" == "y" ]]; then
    echo "Running all-atom pipeline..."
    bash bash_scripts/run_aa_pipeline.sh "$SEQUENCE" "$SEQUENCE_ID"
else
    echo "Running coarse-grained pipeline..."
    bash bash_scripts/run_cg_pipeline.sh "$SEQUENCE" "$SEQUENCE_ID"
fi