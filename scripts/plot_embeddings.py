import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
from src.data_utils import preprocess_data
from utils import visualize_embeddings, normalize_embeddings, modify_and_normalize_embeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings and variance for given backbones.")
    # Changed path, file, dataset to support multiple inputs using nargs='+'
    # We will assume they are passed in corresponding order or user iterates in job script.
    # To support "multiple datasets at once" in one run, we can accept lists.
    
    parser.add_argument('--path', type=str, nargs='+', required=True, help="List of paths to the embeddings files")
    parser.add_argument('--file', type=str, nargs='+', required=True, help="List of filenames of the embeddings CSVs")
    parser.add_argument('--dataset', type=str, nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for plots")
    parser.add_argument('--shifts', type=float, nargs='+', default=[-1, 0, 1], help="Lambda shift values to plot")
    
    args = parser.parse_args()
    
    # Check if lengths match
    if not (len(args.path) == len(args.file) == len(args.dataset)):
        raise ValueError("Arguments --path, --file, and --dataset must have the same number of elements.")
    
    for i in range(len(args.dataset)):
        path = args.path[i]
        file = args.file[i]
        dataset = args.dataset[i]
        
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset}")
        print(f"{'='*50}\n")
        
        # Load Data
        full_path = os.path.join(path, file)
        print(f"Loading data from {full_path}")
        df = pd.read_csv(full_path)
    
        # Simple cleanup if needed, based on notebook
        if 'image_id' in df.columns:
            df.drop(columns=['image_id'], inplace=True, errors='ignore')
        if 'text' in df.columns:
            df.drop(columns=['text'], inplace=True, errors='ignore')

        text_columns = [column for column in df.columns if 'text_emb_' in column]
        image_columns = [column for column in df.columns if 'img_emb_' in column] 
        
        print(f"Found {len(text_columns)} text columns and {len(image_columns)} image columns.")

        # Normalize initial embeddings
        df[text_columns] = normalize_embeddings(df[text_columns].values)
        df[image_columns] = normalize_embeddings(df[image_columns].values)


        dataset_output_dir = os.path.join(args.output_dir, f"{dataset}_CLIP")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for lambda_shift in args.shifts:
            print(f"Processing shift: {lambda_shift}")
            text_embeddings = df[text_columns].values
            image_embeddings = df[image_columns].values
            
            # Modify and normalize
            text_shifted, image_shifted = modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift)
            
            # Visualize and print variance
            visualize_embeddings(
                text_shifted, 
                image_shifted, 
                f'Embeddings with Lambda Shift {lambda_shift} for {dataset}', 
                lambda_shift, 
                dataset, 
                save=True, 
                var=True,
                output_dir=dataset_output_dir
            )

if __name__ == "__main__":
    main()
