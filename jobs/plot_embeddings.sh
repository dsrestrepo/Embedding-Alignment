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

# Example using space-separated strings (as per python script logic)
EMBEDDINGS_PATHS="/gpfs/workdir/restrepoda/Embeddings_vlm/Recipes5k/ /gpfs/workdir/restrepoda/Embeddings_vlm/daquar/ /gpfs/workdir/restrepoda/Embeddings_vlm/coco-qa/ /gpfs/workdir/restrepoda/Embeddings_vlm/fakeddit/ /gpfs/workdir/restrepoda/Embeddings_vlm/brset/ /gpfs/workdir/restrepoda/Embeddings_vlm/ham10000/ /gpfs/workdir/restrepoda/Embeddings_vlm/mimic/ /gpfs/workdir/restrepoda/Embeddings_vlm/mbrset/"
FILES="embeddings_biomedclip.csv  embeddings_clip.csv  embeddings_medsiglip.csv  embeddings_siglip.csv"
DATASET="Recipes5k daquar coco-qa fakeddit brset ham10000 mimic mbrset"
BACKBONE="CLIP SigLIP MedSigLIP BioMedCLIP"

# Using a base output directory structure
OUTPUT_DIR="Images/Embedding_Plots"

# Create output dir
mkdir -p "$OUTPUT_DIR"

python scripts/plot_embeddings.py \
    --paths $EMBEDDINGS_PATHS \
    --files $FILES \
    --datasets $DATASET \
    --backbones $BACKBONE \
    --output_dir "$OUTPUT_DIR" \
    --shifts -1 -0.5 0 0.5 1
