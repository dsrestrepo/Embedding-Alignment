import os
import argparse
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

from src.data_utils import preprocess_data, process_labels, split_data
from src.datasets import VQADataset
from src.classifiers import train_early_fusion, train_late_fusion

from utils import normalize_embeddings, modify_and_normalize_embeddings, visualize_embeddings, plot_results, update_column_names

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Embedding Alignment and Plotting")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings CSV file")
    parser.add_argument("--output_dir", type=str, default="Images/Alignment", help="Output directory for results")
    parser.add_argument("--backbone", type=str, default="start", help="Backbone name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--label_column", type=str, default="DR_2", help="Column name for labels")
    parser.add_argument("--lambda_shifts", type=str, default="-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1", help="Comma separated list of lambda shifts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multilabel", action='store_true', help="Whether to use multilabel classification")

    args = parser.parse_args()

    # Set up output directory
    model_output_dir = os.path.join(args.output_dir, args.backbone) # Using backbone as "Model" folder name proxy, or we can add a Model arg
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"Loading embeddings from {args.embeddings_file}")
    df = pd.read_csv(args.embeddings_file)
    
    # Dataset specific preprocessing (if needed, try to keep generic)
    if 'DR_3' in df.columns:
        df.DR_3 = df.DR_3.astype(str)


    text_columns = [column for column in df.columns if 'text_emb_' in column]
    image_columns = [column for column in df.columns if 'img_emb_' in column] 
    
    # Normalize initial embeddings
    print("Normalizing embeddings...")
    df[text_columns] = normalize_embeddings(df[text_columns].values)
    df[image_columns] = normalize_embeddings(df[image_columns].values)

    df_shifted = df.copy()
    results = {}
    
    # Lambda shift values
    lambda_shift_values = [float(x) for x in args.lambda_shifts.split(',')]

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
    
    for lambda_shift in lambda_shift_values:
        print('#'*50, f' Shift {lambda_shift} ', '#'*50)
        
        # Extract embeddings
        text_embeddings = df[text_columns].values
        image_embeddings = df[image_columns].values
        
        # Modify and normalize embeddings
        text_embeddings_shifted, image_embeddings_shifted = modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift)
        
        # Update column names
        if text_embeddings.shape[1] < text_embeddings_shifted.shape[1]:
            text_columns_updated = update_column_names(text_columns, text_embeddings_shifted.shape[1])
        else: 
            text_columns_updated = text_columns
            
        if image_embeddings.shape[1] < image_embeddings_shifted.shape[1]: 
            image_columns_updated = update_column_names(image_columns, image_embeddings_shifted.shape[1])
        else:
            image_columns_updated = image_columns
            
        current_df = df.copy()
        
        # Assign shifted embeddings. If shape differs, we need new column names.
        if text_embeddings_shifted.shape[1] != len(text_columns):
             # We need new column names
             temp_text_cols = [f"text_shift_{i}" for i in range(text_embeddings_shifted.shape[1])]
             current_df = current_df.drop(columns=text_columns)
             current_df = pd.concat([current_df, pd.DataFrame(text_embeddings_shifted, columns=temp_text_cols)], axis=1)
             current_text_cols = temp_text_cols
        else:
             current_df[text_columns] = text_embeddings_shifted
             current_text_cols = text_columns

        if image_embeddings_shifted.shape[1] != len(image_columns):
             temp_image_cols = [f"image_shift_{i}" for i in range(image_embeddings_shifted.shape[1])]
             current_df = current_df.drop(columns=image_columns)
             current_df = pd.concat([current_df, pd.DataFrame(image_embeddings_shifted, columns=temp_image_cols)], axis=1)
             current_image_cols = temp_image_cols
        else:
             current_df[image_columns] = image_embeddings_shifted
             current_image_cols = image_columns

        # Visualize
        visualize_embeddings(
            text_embeddings_shifted, 
            image_embeddings_shifted, 
            f'Embeddings with Lambda Shift {lambda_shift} for {args.dataset}', 
            lambda_shift, 
            args.dataset, 
            output_dir=model_output_dir,
            var=True # Calculate variance
        )
        
        # Split Data
        train_df, val_df, test_df = split_data(current_df, val_size=args.validation_split, random_state=args.seed, stratify_col=args.label_column if not args.multilabel else None)
        
        # Process Labels
        # Assuming generic process_labels fits
        label_col = args.label_column
        
        # Process Labels
        # If dataset is MIMIC, use the disease_cols list as the label column
        
        current_label_col = label_col
        if args.dataset.lower() == 'mimic':
             if set(disease_cols).issubset(train_df.columns):
                  current_label_col = disease_cols
             else:
                  print(f"Warning: Not all disease columns found in {args.dataset} dataframe. Using provided label_column: {label_col}")

        train_labels, mlb, train_label_columns = process_labels(train_df, col=current_label_col)
        
        # For val and test we reuse mlb (train_target_columns unused in val/test logic effectively as mlb handles valid classes)
        # but process_labels signature for non-None mlb accepts train_columns
        val_labels = process_labels(val_df, col=current_label_col, train_columns=train_label_columns, mlb=mlb) if val_df is not None else None
        test_labels = process_labels(test_df, col=current_label_col, train_columns=train_label_columns, mlb=mlb)
            
        # VQADataset
        train_dataset = VQADataset(train_df, current_text_cols, current_image_cols, label_col, mlb, train_label_columns, labels=train_labels)
        val_dataset = VQADataset(val_df, current_text_cols, current_image_cols, label_col, mlb, train_label_columns, labels=val_labels) if val_df is not None else None
        test_dataset = VQADataset(test_df, current_text_cols, current_image_cols, label_col, mlb, train_label_columns, labels=test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        text_input_size = len(current_text_cols)
        image_input_size = len(current_image_cols)
        # output_size = len(mlb.classes_) if len(mlb.classes_) > 2 else 1
        # Correct output size from dataset labels shape which is already processed
        if hasattr(train_dataset, 'labels'):
            if len(train_dataset.labels.shape) > 1:
                output_size = train_dataset.labels.shape[1]
            else:
                output_size = 1
        else:
            output_size = 1
        
        multilabel = args.multilabel
        
        # Train Early Fusion
        print("Training Early Fusion Model:")
        res_early = train_early_fusion(
            train_loader, test_loader, 
            text_input_size, image_input_size, output_size, 
            num_epochs=args.num_epochs, multilabel=multilabel, report=True, V=False,
            val_loader=val_loader, patience=5
        )
        results[f"early_({lambda_shift})"] = res_early
        
        # Train Late Fusion
        print("Training Late Fusion Model:")
        res_late = train_late_fusion(
            train_loader, test_loader, 
            text_input_size, image_input_size, output_size, 
            num_epochs=args.num_epochs, multilabel=multilabel, report=True, V=False,
            val_loader=val_loader, patience=5
        )
        results[f"late_({lambda_shift})"] = res_late

    # Plot Metrics
    plot_results(results, lambda_shift_values, args.dataset, output_dir=model_output_dir)
    print("Finished.")

if __name__ == "__main__":
    main()
