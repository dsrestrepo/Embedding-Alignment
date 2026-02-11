#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --output=outputs/download_datasets.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long       
#SBATCH --mem=32000                

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0

# Activate the Conda environment
source activate base_ml

echo "Starting download job..."

DATA_DIR="/gpfs/workdir/restrepoda/datasets/"
mkdir -p "$DATA_DIR"


# To run all datasets (uncomment if you want to download everything, but be cautious about time and storage requirements)
# python scripts/download_datasets.py --dataset all --data_dir "$DATA_DIR"

# 1. Fakeddit (20% subset)
echo "Running Fakeddit (0.2)..."
python scripts/download_datasets.py --dataset fakeddit --fakeddit_subset 0.2 --fakeddit_download_mode "url" --data_dir "$DATA_DIR"

# 2. COCO-QA
#echo "Running COCO-QA..."
#python scripts/download_datasets.py --dataset coco-qa --data_dir "$DATA_DIR"

# 3. Recipes5k
# echo "Running Recipes5k..."
# python scripts/download_datasets.py --dataset recipes5k --data_dir "$DATA_DIR"

# 4. DAQUAR
# echo "Running DAQUAR..."
# python scripts/download_datasets.py --dataset daquar --data_dir "$DATA_DIR"

# 5. BRSET
# echo "Running BRSET..."
# python scripts/download_datasets.py --dataset brset --data_dir "$DATA_DIR"

# 6. Satellite
# echo "Running Satellite..."
# python scripts/download_datasets.py --dataset satellite --data_dir "$DATA_DIR"

# 7. MIMIC
# echo "Running MIMIC..."
# python scripts/download_datasets.py --dataset mimic --data_dir "$DATA_DIR"

# 8. mBRSET
# echo "Running mBRSET..."
# python scripts/download_datasets.py --dataset mbrset --data_dir "$DATA_DIR"

# 9. HAM10000 (Requires manual download, this will only preprocess if files exist)
#echo "Running HAM10000..."
#python scripts/download_datasets.py --dataset ham10000 --data_dir "$DATA_DIR"


echo "All requested downloads completed."
