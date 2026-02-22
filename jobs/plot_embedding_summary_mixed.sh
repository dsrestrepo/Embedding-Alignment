#!/bin/bash

# Base directory for embeddings
# Adjust to your actual path
EMBEDDINGS_BASE="/Users/davidrestrepo/Embeddings_Alignemt/Embeddings_vlm"

# Output directory for mixed summary
OUTPUT_DIR="Images/Embedding_Plots/Summary_Mixed"
mkdir -p "$OUTPUT_DIR"

echo "Running plot_embedding_summary_mixed.py..."

python scripts/plot_embedding_summary_mixed.py \
    --base_dir "$EMBEDDINGS_BASE" \
    --output_dir "$OUTPUT_DIR"

echo "Done! Check $OUTPUT_DIR for results."
