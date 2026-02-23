#!/bin/bash
#SBATCH --job-name=make_gif
#SBATCH --output=outputs/make_gif.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32000
#SBATCH --time=1:00:00

# Load the Anaconda module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Base directory for embeddings on cluster
EMBEDDINGS_BASE="/gpfs/workdir/restrepoda/Embeddings_vlm"

# Output directory
OUTPUT_DIR="Images/Summary_gif"
mkdir -p "$OUTPUT_DIR"

echo "Running make_gif_summary.py..."

python scripts/make_gif_summary.py \
    --base_dir "$EMBEDDINGS_BASE" \
    --output_dir "$OUTPUT_DIR"
