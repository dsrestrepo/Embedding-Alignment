#!/bin/bash
#SBATCH --job-name=plot_embedding_summary_mixed
#SBATCH --output=outputs/plot_embedding_summary_mixed.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32000
#SBATCH --time=3:00:00

# Load the Anaconda module
#module load anaconda3/2024.06/gcc-13.2.0
#module load cuda/12.2.1/gcc-11.2.0
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Base directory for embeddings on cluster
EMBEDDINGS_BASE="/gpfs/workdir/restrepoda/Embeddings_vlm"

# Output directory for mixed summary
OUTPUT_DIR="Images/Summary"
mkdir -p "$OUTPUT_DIR"

echo "Running plot_embedding_summary_mixed.py..."

python scripts/plot_embedding_summary_mixed.py \
    --base_dir "$EMBEDDINGS_BASE" \
    --output_dir "$OUTPUT_DIR"
