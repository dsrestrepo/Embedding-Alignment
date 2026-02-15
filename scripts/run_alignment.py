import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import torch
import random
import statistics
from torch.utils.data import DataLoader

from src.data_utils import split_data, process_labels
from src.datasets import VQADataset
from src.classifiers import train_early_fusion, train_late_fusion
from utils import plot_results, visualize_embeddings, normalize_embeddings, modify_and_normalize_embeddings

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run embedding alignment training.")
    # Changed to support multiple inputs using nargs='+' and nested loops
    parser.add_argument('--paths', type=str, nargs='+', required=True, help="List of paths to the embeddings files (one per dataset)")
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--label_columns', type=str, nargs='+', required=True, help="List of column names for labels (one per dataset)")
    parser.add_argument('--multilabels', type=str, nargs='+', required=True, help="List of boolean flags (True/False) for multilabel classification (one per dataset)")
    
    parser.add_argument('--files', type=str, nargs='+', required=True, help="List of filenames of the embeddings CSVs (one per backbone)")
    parser.add_argument('--backbones', type=str, nargs='+', required=True, help="List of backbone names (e.g. CLIP, SigLIP)")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for results")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")
    parser.add_argument('--val_size', type=float, default=0.1, help="Validation set size (fraction of train)")
    parser.add_argument('--model_type', type=str, default='both', choices=['early', 'late', 'both'], help="Model type to train")
    
    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs to average results")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed")
    parser.add_argument('--hidden', type=int, nargs='+', default=[128], help="List of hidden layer sizes (e.g. 128 64). Pass 0 for no hidden layers (linear probe).")

    args = parser.parse_args()
    
    # Process hidden argument
    if len(args.hidden) == 1 and args.hidden[0] == 0:
         hidden_config = 0 # Linear Probe
    elif len(args.hidden) == 1:
         hidden_config = args.hidden[0]
    else:
         hidden_config = args.hidden
    
    # Set global seed
    set_seed(args.seed)
    
    # Check if lengths match for dataset properties
    if not (len(args.paths) == len(args.datasets) == len(args.label_columns) == len(args.multilabels)):
        raise ValueError("Arguments --paths, --datasets, --label_columns, and --multilabels must have the same number of elements.")
    
    # Check if lengths match for backbone properties
    if len(args.files) != len(args.backbones):
        raise ValueError("Arguments --files and --backbones must have the same number of elements.")

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
    lambda_shift_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Define the full set of disease label columns for MIMIC
    disease_cols_full = [
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

    # Using Chexpert's common 6 diseases for simplicity and to ensure they are present in the dataset, as MIMIC can be sparse for some conditions
    disease_cols = [
        "Atelectasis", 
        "Cardiomegaly", 
        "Consolidation", 
        "Edema",
        "No Finding", 
        "Pleural Effusion"
    ]

    for i in range(len(args.datasets)):
        dataset = args.datasets[i]
        path = args.paths[i]
        label_column = args.label_columns[i]
        multilabel_arg = args.multilabels[i]
        
        # Convert string arg to boolean
        is_multilabel = str(multilabel_arg).lower() == 'true'

        for j in range(len(args.backbones)):
            backbone = args.backbones[j]
            file = args.files[j]
            
            print(f"\n{'='*50}")
            print(f"Processing Dataset: {dataset} (Multilabel: {is_multilabel}) | Backbone: {backbone}")
            print(f"{'='*50}\n")
            
            # Load Data
            full_path = os.path.join(path, file)
            print(f"Loading data from {full_path}")
            df = pd.read_csv(full_path)

            if dataset.lower() == 'mimic' and is_multilabel:
                # Filter out rows with -1 in any of the disease columns and fill NaNs with 0
                df = df[(df[disease_cols] != -1).all(axis=1)]
                df[disease_cols] = df[disease_cols].fillna(0)
            
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
            
            # Dataset and backbone specific output dir
            dataset_output_dir = os.path.join(args.output_dir, f"{dataset}/{backbone}")
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            for lambda_shift in lambda_shift_values:
                print('#'*50, f' Shift {lambda_shift} ', '#'*50)
                
                text_embeddings = df[text_columns].values
                image_embeddings = df[image_columns].values
                
                # Modify and normalize
                text_shifted, image_shifted = modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift)
                
                df_shifted[text_columns] = text_shifted
                df_shifted[image_columns] = image_shifted
                
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
                
                # Collections to store results from multiple runs
                early_fusion_runs = []
                late_fusion_runs = []
                
                for run_idx in range(args.n_runs):
                    print(f"\n--- Run {run_idx+1}/{args.n_runs} for shift {lambda_shift} ---")
                    current_seed = args.seed + run_idx
                    set_seed(current_seed)
                
                    # Split data with validation set
                    train_df, val_df, test_df = split_data(df_shifted, val_size=args.val_size, random_state=current_seed)
                    
                    # --- DAQUAR Specific Filter since it has too many classes
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
                    # ------------------------------ END DAQUAR Filter ------------------------------

                    # Process Labels
                    # If dataset is MIMIC, use the disease_cols list as the label column
                    current_label_col = label_column
                    if dataset.lower() == 'mimic' and is_multilabel:
                        if set(disease_cols).issubset(train_df.columns):
                            current_label_col = disease_cols
                        else:
                            print(f"Warning: Not all disease columns found in {dataset} dataframe. Using provided label_column: {label_column} and setting multilabel to False.")
                            is_multilabel = False  # Fallback to non-multilabel if disease columns are missing
                    
                    train_labels, mlb, train_target_columns = process_labels(train_df, col=current_label_col)
                    val_labels = process_labels(val_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb) if val_df is not None else None
                    test_labels = process_labels(test_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb)
                    
                    train_dataset = VQADataset(train_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=train_labels)
                    val_dataset = VQADataset(val_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=val_labels) if val_df is not None else None
                    test_dataset = VQADataset(test_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=test_labels)
                    
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True) if val_dataset else None
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
                    
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
                            val_loader=val_loader, patience=10, hidden=hidden_config
                        )
                        print(f"Best Accuracy: {best['Acc']}")
                        print(f"Best Macro-F1: {best['Macro-F1']}")
                        early_fusion_runs.append(best)
                        # results[f"early_({lambda_shift})"] = best

                    if args.model_type in ['late', 'both']:
                        print("Training Late Fusion Model:")
                        accuracy, precision, recall, f1, best = train_late_fusion(
                            train_loader, test_loader, text_input_size, image_input_size, output_size, 
                            num_epochs=args.epochs, multilabel=multilabel, report=True, V=False,
                            val_loader=val_loader, patience=10, hidden=hidden_config
                        )
                        print(f"Best Accuracy: {best['Acc']}")
                        print(f"Best Macro-F1: {best['Macro-F1']}")
                        late_fusion_runs.append(best)
                        # results[f"late_({lambda_shift})"] = best

                # Aggregate metrics function
                def aggregate_metrics(runs):
                    if not runs: return {'mean': {}, 'std': {}}
                    agg = {'mean': {}, 'std': {}}
                    
                    # Based on structure: best['Macro-F1']['Acc'], best['Macro-F1']['F1'], best['Macro-F1']['Auc']
                    # We use the 'Macro-F1' top-key as the source of truth for the metrics at the best epoch logic
                    
                    values_acc = [r['Macro-F1']['Acc'] for r in runs if 'Macro-F1' in r]
                    values_f1 = [r['Macro-F1']['F1'] for r in runs if 'Macro-F1' in r]
                    values_auc = [r['Macro-F1']['Auc'] for r in runs if 'Macro-F1' in r]
                    
                    # Handle None or NaN
                    values_acc = [v for v in values_acc if v is not None and not np.isnan(v)]
                    values_f1 = [v for v in values_f1 if v is not None and not np.isnan(v)]
                    values_auc = [v for v in values_auc if v is not None and not np.isnan(v)]

                    if values_acc:
                        agg['mean']['Acc'] = statistics.mean(values_acc)
                        agg['std']['Acc'] = statistics.stdev(values_acc) if len(values_acc) > 1 else 0
                    
                    if values_f1:
                        agg['mean']['Macro-F1'] = statistics.mean(values_f1)
                        agg['std']['Macro-F1'] = statistics.stdev(values_f1) if len(values_f1) > 1 else 0
                    
                    if values_auc:
                        agg['mean']['AUC'] = statistics.mean(values_auc)
                        agg['std']['AUC'] = statistics.stdev(values_auc) if len(values_auc) > 1 else 0
                        
                    return agg

                if args.model_type in ['early', 'both']:
                     results[f"early_({lambda_shift})"] = aggregate_metrics(early_fusion_runs)
                
                if args.model_type in ['late', 'both']:
                     results[f"late_({lambda_shift})"] = aggregate_metrics(late_fusion_runs)

            # Save results metrics to JSON
            metrics_file = os.path.join(dataset_output_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(results, f, cls=NumpyEncoder, indent=4)
            print(f"Saved metrics to {metrics_file}")

            # Save results
            plot_results(results, lambda_shift_values, dataset, output_dir=dataset_output_dir)

if __name__ == "__main__":
    main()
