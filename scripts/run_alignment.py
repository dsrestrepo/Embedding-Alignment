import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_utils import split_data, process_labels
from src.datasets import VQADataset
from src.classifiers import train_early_fusion, train_late_fusion
from utils import plot_results, visualize_embeddings, normalize_embeddings, modify_and_normalize_embeddings

def main():
    parser = argparse.ArgumentParser(description="Run embedding alignment training.")
    # Changed to support multiple inputs using nargs='+'
    parser.add_argument('--path', type=str, nargs='+', required=True, help="List of paths to the embeddings files")
    parser.add_argument('--file', type=str, nargs='+', required=True, help="List of filenames of the embeddings CSVs")
    parser.add_argument('--dataset', type=str, nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--label_column', type=str, nargs='+', required=True, help="List of column names for labels")
    parser.add_argument('--multilabel', type=str, nargs='+', required=True, help="List of boolean flags (True/False) for multilabel classification")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for results")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--val_size', type=float, default=0.1, help="Validation set size (fraction of train)")
    parser.add_argument('--model_type', type=str, default='both', choices=['early', 'late', 'both'], help="Model type to train")
    
    args = parser.parse_args()
    
    # Check if lengths match
    if not (len(args.path) == len(args.file) == len(args.dataset) == len(args.label_column) == len(args.multilabel)):
        raise ValueError("Arguments --path, --file, --dataset, --label_column, and --multilabel must have the same number of elements.")

    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Lambda shift values
    lambda_shift_values = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Define the full set of disease label columns for MIMIC
    disease_cols = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Lung Opacity',
        'No Finding',
        'Pleural Effusion',
        'Pleural Other',
        'Pneumonia',
        'Pneumothorax'
    ]

    for i in range(len(args.dataset)):
        path = args.path[i]
        file = args.file[i]
        dataset = args.dataset[i]
        label_column = args.label_column[i]
        multilabel_arg = args.multilabel[i]
        
        # Convert string arg to boolean
        is_multilabel = str(multilabel_arg).lower() == 'true'
        
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset} (Multilabel: {is_multilabel})")
        print(f"{'='*50}\n")
        
        # Load Data
        full_path = os.path.join(path, file)
        print(f"Loading data from {full_path}")
        df = pd.read_csv(full_path)
        
        if 'image_id' in df.columns:
            df.drop(columns=['image_id'], inplace=True, errors='ignore')
        if 'text' in df.columns:
            df.drop(columns=['text'], inplace=True, errors='ignore')

        text_columns = [column for column in df.columns if 'text_emb_' in column]
        image_columns = [column for column in df.columns if 'img_emb_' in column]        
        
        # Normalize initial embeddings
        df[text_columns] = normalize_embeddings(df[text_columns].values)
        df[image_columns] = normalize_embeddings(df[image_columns].values)
        
        df_shifted = df.copy()
        results = {}
        
        # Dataset specific output dir
        dataset_output_dir = os.path.join(args.output_dir, f"{dataset}_CLIP")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for lambda_shift in lambda_shift_values:
            print('#'*50, f' Shift {lambda_shift} ', '#'*50)
            
            text_embeddings = df[text_columns].values
            image_embeddings = df[image_columns].values
            
            # Modify and normalize
            text_shifted, image_shifted = modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift)
            
            df_shifted[text_columns] = text_shifted
            df_shifted[image_columns] = image_shifted
            
            # Save aligned embeddings
            embedding_output_file = os.path.join(dataset_output_dir, f"embeddings_shift_{lambda_shift}.csv")
            df_shifted.to_csv(embedding_output_file, index=False)
            print(f"Saved aligned embeddings to {embedding_output_file}")

            # Visualize (optional in alignment run, but good for verification)
            visualize_embeddings(
                text_shifted, 
                image_shifted, 
                f'Embeddings with Lambda Shift {lambda_shift} for {dataset}', 
                lambda_shift, 
                dataset,
                save=True,
                var=False,
                output_dir=dataset_output_dir
            )
            
            # Split data with validation set
            train_df, val_df, test_df = split_data(df_shifted, val_size=args.val_size)
            
            # --- DAQUAR Specific Filter ---
            if dataset.lower() == 'daquar':
                print(f"Applying {dataset} filter: keeping classes with >= 100 samples in test set.")
                if label_column in test_df.columns:
                    # Count occurrences in test set
                    # Note: Labels might be lists or strings. process_labels handles list splitting.
                    # Assuming 'answer' column contains the raw labels.
                    # Use helper to normalize label column for counting
                    temp_test_series = test_df[label_column]
                    
                    # Check if values are lists (or string rep of lists)
                    first_val = temp_test_series.iloc[0] if len(temp_test_series) > 0 else ""
                    if isinstance(first_val, str) and ',' in first_val:
                         # e.g. "dog, cat"
                         temp_test_series = temp_test_series.str.split(',').explode().str.strip()
                    
                    label_counts = temp_test_series.value_counts()
                    valid_classes = label_counts[label_counts >= 100].index.tolist()
                    
                    print(f"Found {len(valid_classes)} valid classes with >= 100 samples in test set.")
                    
                    # Filter function
                    def filter_rows(row, valid):
                        val = row[label_column]
                        if isinstance(val, str):
                            parts = [p.strip() for p in val.split(',')]
                            return any(p in valid for p in parts)
                        return val in valid

                    # Apply filter to all splits
                    # Re-assignment is necessary
                    train_df = train_df[train_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                    if val_df is not None:
                        val_df = val_df[val_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                    test_df = test_df[test_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                    
                    print(f"Filtered sizes - Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, Test: {len(test_df)}")
                else:
                    print(f"Warning: Label column {label_column} not found in test set, skipping filter.")
            # ------------------------------

            # Process Labels
            # If dataset is MIMIC, use the disease_cols list as the label column
            current_label_col = label_column
            if dataset.lower() == 'mimic':
                 if set(disease_cols).issubset(train_df.columns):
                      current_label_col = disease_cols
                 else:
                      print(f"Warning: Not all disease columns found in {dataset} dataframe. Using provided label_column: {label_column}")
            
            train_labels, mlb, train_target_columns = process_labels(train_df, col=current_label_col)
            val_labels = process_labels(val_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb) if val_df is not None else None
            test_labels = process_labels(test_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb)
            
            train_dataset = VQADataset(train_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=train_labels)
            val_dataset = VQADataset(val_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=val_labels) if val_df is not None else None
            test_dataset = VQADataset(test_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2) # Use val_loader for monitoring if implemented in train loop, else utilize for early stopping etc.
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            
            # Note: train_early_fusion and train_late_fusion currently take train_loader and test_loader. 
            text_input_size = len(text_columns)
            image_input_size = len(image_columns)
            
            # Determine output size and multilabel
            if hasattr(train_dataset, 'labels'):
                if len(train_dataset.labels.shape) > 1:
                     output_size = train_dataset.labels.shape[1]
                else:
                     output_size = 1
            else:
                output_size = 1
            
            multilabel = is_multilabel

            if args.model_type in ['early', 'both']:
                print("Training Early Fusion Model:")
                # Pass patience and val_loader for early stopping
                accuracy, precision, recall, f1, best = train_early_fusion(
                    train_loader, test_loader, text_input_size, image_input_size, output_size, 
                    num_epochs=args.epochs, multilabel=multilabel, report=True, V=False,
                    val_loader=val_loader, patience=5
                )
                print(f"Best Accuracy: {best['Acc']}")
                print(f"Best Macro-F1: {best['Macro-F1']}")
                results[f"early_({lambda_shift})"] = best

            if args.model_type in ['late', 'both']:
                print("Training Late Fusion Model:")
                accuracy, precision, recall, f1, best = train_late_fusion(
                    train_loader, test_loader, text_input_size, image_input_size, output_size, 
                    num_epochs=args.epochs, multilabel=multilabel, report=True, V=False,
                    val_loader=val_loader, patience=5
                )
                print(f"Best Accuracy: {best['Acc']}")
                print(f"Best Macro-F1: {best['Macro-F1']}")
                results[f"late_({lambda_shift})"] = best

        # Save results metrics to JSON
        metrics_file = os.path.join(dataset_output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        print(f"Saved metrics to {metrics_file}")

        # Save results
        plot_results(results, lambda_shift_values, dataset, output_dir=dataset_output_dir)

if __name__ == "__main__":
    main()
