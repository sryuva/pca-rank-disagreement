#!/usr/bin/env bash
# Reproduce every result, table and figure in the note (about 5 minutes on a laptop CPU).
set -e
cd "$(dirname "$0")"
python src/experiment.py                 # main 72-condition grid + subspace-distance study
python src/experiment.py --pilot         # Section 4.1 geometric-spectrum pilot
python src/checks.py        | tee results/checks_output.txt          # Section 3 + Davis-Kahan column
python src/split_analysis.py | tee results/split_analysis_output.txt # Table 3
python src/plots.py                      # Figures 1-3
