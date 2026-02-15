#!/bin/bash
#SBATCH --job-name=run_alignment
#SBATCH --output=outputs/run_alignment.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpua100
#SBATCH --mem=64000
#SBATCH --time=24:00:00

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate the Conda environment
source activate base_ml

# Default Arguments (modify as needed or pass via sbatch --export)
# Example: sbatch --export=ALL,DATASET="MIMIC",FILE="embeddings_clip_mimic.csv" jobs/run_alignment.sh

# To run multiple datasets, separate with spaces in quotes
#EMBEDDINGS_PATHS="/gpfs/workdir/restrepoda/Embeddings_vlm/brset/ /gpfs/workdir/restrepoda/Embeddings_vlm/ham10000/ /gpfs/workdir/restrepoda/Embeddings_vlm/mimic/ /gpfs/workdir/restrepoda/Embeddings_vlm/mbrset/" #"/gpfs/workdir/restrepoda/Embeddings_vlm/Recipes5k/ /gpfs/workdir/restrepoda/Embeddings_vlm/daquar/ /gpfs/workdir/restrepoda/Embeddings_vlm/coco-qa/ /gpfs/workdir/restrepoda/Embeddings_vlm/fakeddit/ /gpfs/workdir/restrepoda/Embeddings_vlm/brset/ /gpfs/workdir/restrepoda/Embeddings_vlm/ham10000/ /gpfs/workdir/restrepoda/Embeddings_vlm/mimic/ /gpfs/workdir/restrepoda/Embeddings_vlm/mbrset/"
EMBEDDINGS_PATHS="/gpfs/workdir/restrepoda/Embeddings_vlm/mimic/" #"/gpfs/workdir/restrepoda/Embeddings_vlm/Recipes5k/ /gpfs/workdir/restrepoda/Embeddings_vlm/daquar/ /gpfs/workdir/restrepoda/Embeddings_vlm/coco-qa/ /gpfs/workdir/restrepoda/Embeddings_vlm/fakeddit/ /gpfs/workdir/restrepoda/Embeddings_vlm/brset/ /gpfs/workdir/restrepoda/Embeddings_vlm/ham10000/ /gpfs/workdir/restrepoda/Embeddings_vlm/mimic/ /gpfs/workdir/restrepoda/Embeddings_vlm/mbrset/"

FILES="embeddings_biomedclip.csv  embeddings_clip.csv  embeddings_medsiglip.csv  embeddings_siglip.csv"

#DATASET="brset ham10000 mimic mbrset" #"Recipes5k daquar coco-qa fakeddit brset ham10000 mimic mbrset"
DATASET="mimic"

BACKBONE="BioMedCLIP CLIP MedSigLIP SigLIP "
# Multilabel flags corresponding to DATASET (Recipes5k=False, daquar=True, coco-qa=False, fakeddit=False, brset=False, ham10000=False, mimic=True, mbrset=False)

#MULTILABEL="False False True False" #"False True False False False False True False"
MULTILABEL="False" #Test mimic on multiclass


#LABEL_COLUMN="DR_2 dx disease_label DR_2" #"class answer answers 2_way_label DR_2 dx disease_label DR_2"
LABEL_COLUMN="disease_label" #Test mimic on multiclass

MODEL_TYPE="early" # Options: 'early', 'late', 'both'
HIDDEN=0 # For multiple hidden layers: "128 64" ... for linear only: "0"
EPOCHS=100
N_RUNS=5
OUTPUT_DIR="Images/Alignment_early_linear_5runs_batch512"
BATCH_SIZE=512

mkdir -p "$OUTPUT_DIR"

python scripts/run_alignment.py \
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
    --batch_size $BATCH_SIZE