#!/bin/bash

# Utility script to run pairs of sequences 
# through the pipeline in parallel on 2 GPUs
set -eou pipefail

USE_AA="n" # "y" for all-atom, "n" for coarse-grained
pairs=("SEQ1:ID1" "SEQ2:ID2" "SEQ3:ID3" "SEQ4:ID4") # example seq:ID pairs


for ((i=0; i<${#pairs[@]}; i+=2)); do
        # grab first seq:ID pair
        pair0="${pairs[$i]}"

        # submit to first GPU
        CUDA_VISIBLE_DEVICES=0 bash bash_scripts/run_pipeline.sh "${pair0%%:*}" "${pair0##*:}" "$USE_AA" &
        pid0=$!

        # check if there are more seq:ID pairs
        if [[ $((i+1)) -lt ${#pairs[@]} ]]; then
                # if there is, do the same thing but put on second GPU
                pair1="${pairs[$i+1]}"
                CUDA_VISIBLE_DEVICES=1 bash bash_scripts/run_pipeline.sh "${pair1%%:*}" "${pair1##*:}" "$USE_AA" &
                pid1=$!

                # Capture exit codes without triggering set -e
                wait $pid0; status0=$?
                wait $pid1; status1=$?

                if [[ $status0 -ne 0 || $status1 -ne 0 ]]; then
                        echo "ERROR: job failed — gpu0: $status0, gpu1: $status1" >&2
                        exit 1
                fi
                echo "Done: $pair0 and $pair1"
        else
                # if there isn't, wait for the  first GPU to be done
                wait $pid0; status0=$?
                if [[ $status0 -ne 0 ]]; then
                        echo "ERROR: job failed — gpu0: $status0" >&2
                        exit 1
                fi
                echo "Done: $pair0"
        fi
done
