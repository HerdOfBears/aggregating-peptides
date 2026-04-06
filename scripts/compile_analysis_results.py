import pickle as pkl
import numpy as np
import os
import json
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", required=True, type=str,
                        help="The amino acid sequence of the peptide.")
    parser.add_argument("--pepid", required=True, type=str,
                        help="The ID of the peptide.")
    parser.add_argument("--wdir", required=False, type=str,
                        help="The working directory where analysis results are \
                            or where the pepid/ directory is.",
                        default="outputs/")
    args = parser.parse_args()

    params=vars(args)

    
    