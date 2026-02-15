#!/bin/bash
#SBATCH --job-name=auto_align
#SBATCH --output=outputs/auto_align.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=64000
#SBATCH --time=24:00:00

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate the Conda environment
source activate base_ml

# Default Arguments (modify as needed or pass via sbatch --export)
# Example: sbatch --export=ALL,DATASET="MIMIC",FILE="embeddings_clip_mimic.csv" jobs/auto_align.sh

# To run multiple datasets, separate with spaces in quotes
EMBEDDINGS_PATHS="/gpfs/workdir/restrepoda/Embeddings_vlm/mimic/" 

FILES="embeddings_biomedclip.csv  embeddings_clip.csv  embeddings_medsiglip.csv  embeddings_siglip.csv"

DATASET="mimic"

BACKBONE="BioMedCLIP CLIP MedSigLIP SigLIP "
# Multilabel flags corresponding to DATASET

MULTILABEL="False" 

LABEL_COLUMN="disease_label" 

MODEL_TYPE="early" # Options: 'early', 'late', 'both'
HIDDEN=0 # For multiple hidden layers: "128 64" ... for linear only: "0"
EPOCHS=100
N_RUNS=5
OUTPUT_DIR="Images/Auto_Alignment_early_linear_5runs"
BATCH_SIZE=512

mkdir -p "$OUTPUT_DIR"

python scripts/auto_align.py \
    --paths $EMBEDDINGS_PATHS \
    --files $FILES \
    --datasets $DATASET \
    --backbones $BACKBONE \
    --label_columns $LABEL_COLUMN \
    --multilabels $MULTILABEL \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --n_runs $N_RUNS \
    --model_type $MODEL_TYPE \
    --hidden $HIDDEN \
    --batch_size $BATCH_SIZE \
    --initial_lambda 0.0 \
    --save_plots
