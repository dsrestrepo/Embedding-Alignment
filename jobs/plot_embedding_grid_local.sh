#!/bin/bash

# Base directory for embeddings
# Adjust to your actual path
EMBEDDINGS_BASE="/Users/davidrestrepo/Embeddings_Alignemt/Embeddings_vlm"

# Output directory
OUTPUT_DIR="Images/Summary_Grids"
mkdir -p "$OUTPUT_DIR"

echo "Running plot_embedding_grid.py..."

python scripts/plot_embedding_grid.py \
    --base_dir "$EMBEDDINGS_BASE" \
    --output_dir "$OUTPUT_DIR"

echo "Done! Check $OUTPUT_DIR for results."
