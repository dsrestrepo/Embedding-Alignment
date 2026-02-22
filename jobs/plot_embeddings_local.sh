#!/bin/bash

# Base directory for embeddings
# Adjust this if your folder structure is different
EMBEDDINGS_BASE="/Users/davidrestrepo/Embeddings_Alignemt/Embeddings_vlm"

# Dataset names corresponding to folders in EMBEDDINGS_BASE
DATASETS=("Recipes5k" "daquar" "coco-qa" "fakeddit" "brset" "ham10000" "mimic" "mbrset")

# Construct the arguments for the python script
PATHS_ARG=""
DATASETS_ARG=""

for dataset in "${DATASETS[@]}"; do
    PATHS_ARG+="${EMBEDDINGS_BASE}/${dataset}/ "
    DATASETS_ARG+="${dataset} "
done

# Files and Backbones need to match in length and order
FILES="embeddings_clip.csv embeddings_siglip.csv embeddings_medsiglip.csv embeddings_biomedclip.csv"
BACKBONES="CLIP SigLIP MedSigLIP BioMedCLIP"

# Output directory
OUTPUT_DIR="Images/Embedding_Plots"
mkdir -p "$OUTPUT_DIR"

echo "Running plot_embeddings.py..."
# Print the command for debugging (optional)
# echo "python scripts/plot_embeddings.py --paths $PATHS_ARG --files $FILES --datasets $DATASETS_ARG --backbones $BACKBONES --output_dir \"$OUTPUT_DIR\" --shifts -1 -0.5 0 0.5 1"

python scripts/plot_embeddings.py \
    --paths $PATHS_ARG \
    --files $FILES \
    --datasets $DATASETS_ARG \
    --backbones $BACKBONES \
    --output_dir "$OUTPUT_DIR" \
    --shifts 0

echo "Done! Check $OUTPUT_DIR for results."
