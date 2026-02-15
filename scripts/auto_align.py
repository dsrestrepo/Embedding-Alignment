import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import torch
import random
import statistics
import json
from torch.utils.data import DataLoader

from src.data_utils import split_data, process_labels
from src.datasets import VQADataset
from src.classifiers import EarlyFusionModel, LateFusionModel
from src.auto_alignment import train_auto_align
from utils import normalize_embeddings

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_gap_vector(text_embeddings, image_embeddings):
    """
    Calculate the gap vector (mean image - mean text).
    """
    gap_vector = np.mean(image_embeddings, axis=0) - np.mean(text_embeddings, axis=0)
    return gap_vector

def main():
    parser = argparse.ArgumentParser(description="Run auto-alignment training with learnable lambda.")
    
    parser.add_argument('--paths', type=str, nargs='+', required=True, help="List of paths to the embeddings files")
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--label_columns', type=str, nargs='+', required=True, help="List of column names for labels")
    parser.add_argument('--multilabels', type=str, nargs='+', required=True, help="List of flags for multilabel classification")
    
    parser.add_argument('--files', type=str, nargs='+', required=True, help="List of filenames of the embeddings CSVs")
    parser.add_argument('--backbones', type=str, nargs='+', required=True, help="List of backbone names")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for results")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")
    parser.add_argument('--val_size', type=float, default=0.1, help="Validation set size")
    parser.add_argument('--model_type', type=str, default='both', choices=['early', 'late', 'both'], help="Model type to train")
    
    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed")
    parser.add_argument('--hidden', type=int, nargs='+', default=[128], help="List of hidden layer sizes (0 for linear probe)")
    
    parser.add_argument('--initial_lambda', type=float, default=0.0, help="Initial value for lambda")
    parser.add_argument('--save_plots', action='store_true', help="Save GIF and plots of embedding evolution")

    args = parser.parse_args()
    
    # Process hidden argument
    if len(args.hidden) == 1 and args.hidden[0] == 0:
         hidden_config = 0 # Linear Probe
    elif len(args.hidden) == 1:
         hidden_config = args.hidden[0]
    else:
         hidden_config = args.hidden
    
    set_seed(args.seed)
    
    # JSON Encoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
            
    # MIMIC Settings
    disease_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"
    ]

    for i in range(len(args.datasets)):
        dataset = args.datasets[i]
        path = args.paths[i]
        label_column = args.label_columns[i]
        multilabel_arg = args.multilabels[i]
        
        is_multilabel = str(multilabel_arg).lower() == 'true'

        for j in range(len(args.backbones)):
            backbone = args.backbones[j]
            file = args.files[j]
            
            print(f"\n{'='*50}")
            print(f"Auto-Align Processing: {dataset} | Backbone: {backbone}")
            print(f"{'='*50}\n")
            
            # Load Data
            full_path = os.path.join(path, file)
            print(f"Loading data from {full_path}")
            df = pd.read_csv(full_path)

            if dataset.lower() == 'mimic' and is_multilabel:
                df = df[(df[disease_cols] != -1).all(axis=1)]
                df[disease_cols] = df[disease_cols].fillna(0)
            
            # Cleanup
            df.drop(columns=['image_id', 'text'], inplace=True, errors='ignore')

            text_columns = [col for col in df.columns if 'text_emb_' in col]
            image_columns = [col for col in df.columns if 'img_emb_' in col]        
            
            # Normalize initial embeddings (Important before gap calc)
            df[text_columns] = normalize_embeddings(df[text_columns].values)
            df[image_columns] = normalize_embeddings(df[image_columns].values)
            
            dataset_output_dir = os.path.join(args.output_dir, f"{dataset}/{backbone}")
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Helper to allow aggregating results
            aggregated_results = {} 
            
            # Run Loop
            early_fusion_runs = []
            late_fusion_runs = []

            for run_idx in range(args.n_runs):
                print(f"\n--- Run {run_idx+1}/{args.n_runs} ---")
                current_seed = args.seed + run_idx
                set_seed(current_seed)
            
                # Split Data
                train_df, val_df, test_df = split_data(df, val_size=args.val_size, random_state=current_seed)
                
                # DAQUAR Filter logic
                if dataset.lower() == 'daquar':
                    if label_column in test_df.columns:
                        temp_test = test_df[label_column]
                        if len(temp_test) > 0 and isinstance(temp_test.iloc[0], str) and ',' in temp_test.iloc[0]:
                             temp_test = temp_test.str.split(',').explode().str.strip()
                        lc = temp_test.value_counts()
                        valid_classes = lc[lc >= 100].index.tolist()
                        
                        def filter_rows(row, valid):
                            val = row[label_column]
                            if isinstance(val, str):
                                parts = [p.strip() for p in val.split(',')]
                                return any(p in valid for p in parts)
                            return val in valid

                        train_df = train_df[train_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                        if val_df is not None: val_df = val_df[val_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                        test_df = test_df[test_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]

                # Calculate Gap Vector on Training Data
                # Note: We must ensure text_columns/image_columns order is consistent
                train_text = train_df[text_columns].values
                train_image = train_df[image_columns].values
                gap_vector = calculate_gap_vector(train_text, train_image)
                print(f"Gap Vector calculated on training set (shape: {gap_vector.shape})")

                # Process Labels
                current_label_col = label_column
                if dataset.lower() == 'mimic' and is_multilabel:
                    if set(disease_cols).issubset(train_df.columns):
                        current_label_col = disease_cols
                    else:
                        is_multilabel = False

                train_labels, mlb, train_target_columns = process_labels(train_df, col=current_label_col)
                val_labels = process_labels(val_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb) if val_df is not None else None
                test_labels = process_labels(test_df, col=current_label_col, train_columns=train_target_columns, mlb=mlb)
                
                train_dataset = VQADataset(train_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=train_labels)
                val_dataset = VQADataset(val_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=val_labels) if val_df is not None else None
                test_dataset = VQADataset(test_df, text_columns, image_columns, label_column, mlb, train_target_columns, labels=test_labels)
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) if val_dataset else None
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
                
                text_input_size = len(text_columns)
                image_input_size = len(image_columns)
                
                if hasattr(train_dataset, 'labels'):
                    output_size = train_dataset.labels.shape[1] if len(train_dataset.labels.shape) > 1 else 1
                else:
                    output_size = 1
                
                # Viz Data (Use validation set subset for visualization)
                viz_data = None
                viz_output_dir = None
                if args.save_plots and run_idx == 0: # Only save visual evolution for the first run
                    viz_output_dir = os.path.join(dataset_output_dir, f"viz_run_{run_idx}")
                    os.makedirs(viz_output_dir, exist_ok=True)
                    # Prepare viz data numpy arrays
                    if val_dataset:
                        viz_df_sub = val_df.sample(min(1000, len(val_df)), random_state=current_seed)
                    else:
                        viz_df_sub = train_df.sample(min(1000, len(train_df)), random_state=current_seed)
                    
                    viz_text = viz_df_sub[text_columns].values
                    viz_image = viz_df_sub[image_columns].values
                    viz_data = (viz_text, viz_image)

                if args.model_type in ['early', 'both']:
                    print("Training Auto-Align Early Fusion Model:")
                    acc, prec, rec, f1, best, l_hist = train_auto_align(
                        EarlyFusionModel,
                        train_loader, test_loader, text_input_size, image_input_size, output_size, gap_vector,
                        num_epochs=args.epochs, multilabel=is_multilabel, report=True, V=False,
                        val_loader=val_loader, patience=10, hidden=hidden_config,
                        viz_data=viz_data, output_dir=viz_output_dir, dataset_name=dataset
                    )
                    early_fusion_runs.append(best)

                if args.model_type in ['late', 'both']:
                    print("Training Auto-Align Late Fusion Model:")
                    acc, prec, rec, f1, best, l_hist = train_auto_align(
                        LateFusionModel,
                        train_loader, test_loader, text_input_size, image_input_size, output_size, gap_vector,
                        num_epochs=args.epochs, multilabel=is_multilabel, report=True, V=False,
                        val_loader=val_loader, patience=10, hidden=hidden_config,
                        viz_data=viz_data, output_dir=viz_output_dir, dataset_name=dataset
                    )
                    late_fusion_runs.append(best)

            # Aggregate Runs function
            def aggregate_metrics(runs):
                if not runs: return {'mean': {}, 'std': {}}
                agg = {'mean': {}, 'std': {}}
                
                # We need to look up keys inside best dictionary
                # Structure: best['Macro-F1']['Acc'], best['Best_Lambda']
                
                keys_to_avg = ['Acc', 'F1', 'Auc']
                
                # Handling nested structure from run_alignment.py
                # Reference: best['Macro-F1'] = {'Acc': ..., 'F1': ...}
                
                # Let's aggregate the top level scalar 'Best_Lambda'
                lambdas = [r.get('Best_Lambda', 0) for r in runs]
                agg['mean']['Best_Lambda'] = statistics.mean(lambdas)
                agg['std']['Best_Lambda'] = statistics.stdev(lambdas) if len(lambdas) > 1 else 0

                # Aggregate metrics based on 'Macro-F1' category
                values_acc = [r['Macro-F1']['Acc'] for r in runs if 'Macro-F1' in r]
                values_f1 = [r['Macro-F1']['F1'] for r in runs if 'Macro-F1' in r]
                values_auc = [r['Macro-F1']['Auc'] for r in runs if 'Macro-F1' in r]

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
                 aggregated_results["early"] = aggregate_metrics(early_fusion_runs)
            
            if args.model_type in ['late', 'both']:
                 aggregated_results["late"] = aggregate_metrics(late_fusion_runs)
            
            # Save results
            metrics_file = os.path.join(dataset_output_dir, "auto_align_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(aggregated_results, f, cls=NumpyEncoder, indent=4)
            print(f"Saved aggregated metrics to {metrics_file}")
            
            # Note: We can't reuse plot_results from utils easily because it expects results[f"early_({lambda})"]
            # But here we assume lambda is learned. We don't sweep.
            
if __name__ == "__main__":
    main()
