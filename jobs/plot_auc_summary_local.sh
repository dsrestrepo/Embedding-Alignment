#!/bin/bash

# Base directory where metrics.json files are stored
# Adjust this to point to your actual results directory
# Defaulting to the path found in run_alignment.sh for the cluster, but mapped to local if possible
# or just a relative path if you ran it locally.

# Example: If you downloaded the results to a folder named 'Results_Alignment'
# BASE_DIR="Results_Alignment"

# Default based on previous context:
BASE_DIR="Images/Alignment_early_linear_5runs_batch512"

# Output directory for the summary plot
OUTPUT_DIR="Images/Summary"
mkdir -p "$OUTPUT_DIR"

echo "Running plot_auc_summary.py..."
echo "Looking for metrics in: $BASE_DIR"

python scripts/plot_auc_summary.py \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Done! Check $OUTPUT_DIR for the 4x4 summary plot."
