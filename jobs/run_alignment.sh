#!/bin/bash
#SBATCH --job-name=run_alignment
#SBATCH --output=outputs/run_alignment.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
EMBEDDINGS_PATHS="/gpfs/workdir/restrepoda/Embeddings_vlm/Recipes5k/ /gpfs/workdir/restrepoda/Embeddings_vlm/daquar/ /gpfs/workdir/restrepoda/Embeddings_vlm/coco-qa/ /gpfs/workdir/restrepoda/Embeddings_vlm/brset/ /gpfs/workdir/restrepoda/Embeddings_vlm/ham10000/ /gpfs/workdir/restrepoda/Embeddings_vlm/mimic/ /gpfs/workdir/restrepoda/Embeddings_vlm/mbrset/"
FILE="embeddings_clip.csv embeddings_clip.csv embeddings_clip.csv embeddings_clip.csv embeddings_clip.csv embeddings_clip.csv embeddings_clip.csv"
DATASET="Recipes5k daquar coco-qa brset ham10000 mimic mbrset"
# Multilabel flags corresponding to DATASET (Recipes5k=False, daquar=True, coco-qa=False, brset=False, ham10000=False, mimic=True, mbrset=False)
MULTILABEL="False True False False False True False"
LABEL_COLUMN="class answer answers DR_2 dx disease_label DR_2"
OUTPUT_DIR="Images/Alignment"

mkdir -p "$OUTPUT_DIR"

python scripts/run_alignment.py \
    --path $EMBEDDINGS_PATHS \
    --file $FILE \
    --dataset $DATASET \
    --label_column $LABEL_COLUMN \
    --multilabel $MULTILABEL \
    --output_dir "$OUTPUT_DIR" \
    --epochs 100 \
    --model_type both
