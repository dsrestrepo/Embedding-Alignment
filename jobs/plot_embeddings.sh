#!/bin/bash
#SBATCH --job-name=plot_embeddings
#SBATCH --output=outputs/plot_embeddings.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32000
#SBATCH --time=3:00:00

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate the Conda environment
source activate base_ml

# Default Arguments
# Example of multiple datasets:
# PATHS=("Embeddings_vlm/brset/" "Embeddings_vlm/mimic/")
# FILES=("embeddings_clip.csv" "embeddings_clip.csv")
# DATASETS=("BRSET" "MIMIC")

# For now, keeping single example which works with new list support
EMBEDDINGS_PATHS="Embeddings_vlm/brset/"
FILE="embeddings_clip.csv"
DATASET="BRSET"
# Using a base output directory structure
OUTPUT_DIR="Images/Alignment"

# Create output dir
mkdir -p "$OUTPUT_DIR"

python scripts/plot_embeddings.py \
    --path "$EMBEDDINGS_PATHS" \
    --file "$FILE" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --shifts -1 0 1
