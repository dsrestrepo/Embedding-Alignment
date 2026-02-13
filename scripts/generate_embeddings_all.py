import subprocess
import os
import sys

# Configuration
# The root directory where datasets are stored
DATASET_ROOT = '/gpfs/workdir/restrepoda/datasets'
# DATASET_ROOT = './datasets' # Uncomment for local testing

# The root directory where outputs will be saved
OUTPUT_ROOT = '/gpfs/workdir/restrepoda/Embeddings_vlm'

# Training parameters
BATCH_SIZE = 256
NPROC_PER_NODE = 2

# Dataset configurations
# 'path': relative path from DATASET_ROOT
# 'image_col': column name for images
# 'text_col': column name for text
DATASET_CONFIGS = {
    'daquar': {'path': 'daquar', 'image_col': 'image_id', 'text_col': 'question', 'image_dir': 'images'},
    'coco-qa': {'path': 'coco-qa', 'image_col': 'image_id', 'text_col': 'questions', 'image_dir': 'images'},
    'fakeddit': {'path': 'fakeddit', 'image_col': 'id', 'text_col': 'text', 'image_dir': 'images'},
    'Recipes5k': {'path': 'Recipes5k', 'image_col': 'image', 'text_col': 'ingredients', 'image_dir': 'images'},
    'brset': {'path': 'BRSET/brset', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images'},
    'ham10000': {'path': 'HAM10000', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images'},
    'mimic': {'path': 'MIMIC/mimic', 'image_col': 'path_preproc', 'text_col': 'text', 'image_dir': '.'},
    'mbrset': {'path': 'mBRSET/mbrset', 'image_col': 'file', 'text_col': 'text', 'image_dir': 'images'},
}

# List of datasets to process (add/remove as needed)
DATASETS_TO_RUN = [
    'Recipes5k',    # ✅
    'daquar',       # ✅
    'coco-qa',      # ✅
    'brset',        # ✅
    'ham10000',     # ✅
    'mimic',        # ✅
    'mbrset',       # ✅ 
    'fakeddit',     # ✅
]

# List of models to use
MODELS_TO_RUN = ['CLIP', 'MedSigLip', 'SigLip', 'biomedclip'] #['CLIP', 'MedSigLip', 'SigLip', 'biomedclip', 'Unimed']

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def main():
    # Ensure subprocess is run from the project root where src/ matches
    if not os.path.exists('src/vlm_embeddings.py'):
        print("Error: src/vlm_embeddings.py not found. Please run this script from the project root.")
        return

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for dataset_name in DATASETS_TO_RUN:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Warning: Configuration for {dataset_name} not found. Skipping.")
            continue

        config = DATASET_CONFIGS[dataset_name]
        dataset_path = os.path.join(DATASET_ROOT, config['path'])
        
        # Check availability (warn only)
        if not os.path.exists(dataset_path):
             print(f"Warning: Dataset path {dataset_path} does not exist (or is not accessible). Skipping {dataset_name}?")
             pass

        for model in MODELS_TO_RUN:
            output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
            output_file = f'embeddings_{model.lower()}.csv'
            
            # Construct command
            cmd = (
                f"torchrun --nproc_per_node={NPROC_PER_NODE} src/vlm_embeddings.py "
                f"--classifier {model} "
                f"--batch_size {BATCH_SIZE} "
                f"--dataset_path {dataset_path} "
                f"--image_col {config['image_col']} "
                f"--text_col {config['text_col']} "
                f"--output_dir {output_dir} "
                f"--output_file {output_file} "
                f"--image_dir {config['image_dir']}"
            )

            print(f"\n--- Processing {dataset_name} with {model} ---")
            run_command(cmd)

if __name__ == "__main__":
    main()
