from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

def interpolate_embeddings(smaller_embeddings, target_length):
    """Interpolate embeddings to match a target length."""
    interpolated_embeddings = np.zeros((smaller_embeddings.shape[0], target_length))
    for i in range(smaller_embeddings.shape[0]):
        interp_func = interp1d(np.linspace(0, 1, smaller_embeddings.shape[1]), smaller_embeddings[i, :])
        interpolated_embeddings[i, :] = interp_func(np.linspace(0, 1, target_length))
    return interpolated_embeddings

def normalize_embeddings(embeddings):
    """Normalize embeddings to the unit sphere."""
    return normalize(embeddings, axis=1, norm='l2')

def modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift):
    """Shift and normalize embeddings."""
    # Check and match dimensions
    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        print('Warning: Text and image embeddings have different dimensions. Interpolating to match dimensions.')
        if text_embeddings.shape[1] > image_embeddings.shape[1]:
            image_embeddings = interpolate_embeddings(image_embeddings, text_embeddings.shape[1])
        else:
            text_embeddings = interpolate_embeddings(text_embeddings, image_embeddings.shape[1])
    
    # Calculate the original gap vector
    gap_vector = np.mean(image_embeddings, axis=0) - np.mean(text_embeddings, axis=0)
    
    # Shift embeddings
    text_embeddings_shifted = text_embeddings + (lambda_shift/2) * gap_vector
    image_embeddings_shifted = image_embeddings - (lambda_shift/2) * gap_vector
    
    # Normalize to the unit sphere
    text_embeddings_shifted = normalize_embeddings(text_embeddings_shifted)
    image_embeddings_shifted = normalize_embeddings(image_embeddings_shifted)
    
    return text_embeddings_shifted, image_embeddings_shifted


def visualize_embeddings(text_embeddings, image_embeddings, title, lambda_shift, DATASET, save=True, var=False, output_dir=None):
    """Visualize embeddings in 2D and 3D, including the unit circle and sphere."""
    if output_dir is None:
        output_dir = f'Images/{DATASET}'

    pca = PCA(n_components=2)
    all_embeddings = np.concatenate([text_embeddings, image_embeddings])
    reduced_embeddings = pca.fit_transform(all_embeddings)
    
    # Split reduced embeddings back
    reduced_text_embeddings = reduced_embeddings[:len(text_embeddings)]
    reduced_image_embeddings = reduced_embeddings[len(text_embeddings):]
    if var:
        # Calculate and print the variance for each modality in the PCA-transformed space
        text_embeddings_variance = np.var(reduced_text_embeddings, axis=0)
        image_embeddings_variance = np.var(reduced_image_embeddings, axis=0)
        # Calculate the mean variance across PCA components
        mean_variance_text = np.mean(text_embeddings_variance)
        mean_variance_image = np.mean(image_embeddings_variance)

        # Print the mean variance
        print("Mean Variance of PCA-transformed text embeddings:", mean_variance_text)
        print("Mean Variance of PCA-transformed image embeddings:", mean_variance_image)

    # Plotting in 2D with unit circle
    plt.figure(figsize=(10, 6))
    #circle = plt.Circle((0, 0), 1, color='green', fill=False)
    #plt.gca().add_artist(circle)
    plt.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], label='Text Embeddings', alpha=0.5)
    plt.scatter(reduced_image_embeddings[:, 0], reduced_image_embeddings[:, 1], label='Image Embeddings', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title(title + ' in 2D')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    img_path_2d = os.path.join(output_dir, f'2d_shift({lambda_shift}).pdf')
    if save:
        os.makedirs(os.path.dirname(img_path_2d), exist_ok=True)
        plt.savefig(img_path_2d)
    plt.show()

    # Plotting in 3D with unit sphere
    fig = plt.figure(figsize=(10, 10))  # Corrected figsize
    ax = fig.add_subplot(111, projection='3d')
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(all_embeddings)
    reduced_text_embeddings_3d = reduced_embeddings_3d[:len(text_embeddings)]
    reduced_image_embeddings_3d = reduced_embeddings_3d[len(text_embeddings):]
    
    ax.scatter(reduced_text_embeddings_3d[:, 0], reduced_text_embeddings_3d[:, 1], reduced_text_embeddings_3d[:, 2], label='Text Embeddings', alpha=0.5)
    ax.scatter(reduced_image_embeddings_3d[:, 0], reduced_image_embeddings_3d[:, 1], reduced_image_embeddings_3d[:, 2], label='Image Embeddings', alpha=0.5)
    
    # Draw a unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #ax.plot_wireframe(x, y, z, color="r", alpha=0.1)
    
    ax.set_title(title + ' in 3D')
    ax.set_xlabel('PCA Component 1', labelpad=10)
    ax.set_ylabel('PCA Component 2', labelpad=10)
    ax.set_zlabel('PCA Component 3', labelpad=10)
    plt.legend()
    
    img_path_3d = os.path.join(output_dir, f'3d_shift({lambda_shift}).pdf')
    if save:
        os.makedirs(os.path.dirname(img_path_3d), exist_ok=True)
        plt.savefig(img_path_3d)
    plt.show()

def plot_results(results, lambda_shift_values, DATASET, output_dir=None):
    if output_dir is None:
        output_dir = f'Images/{DATASET}'

    # Check if we have aggregated results (mean/std) or single run
    first_key = f'early_({lambda_shift_values[0]})'
    if first_key not in results:
        # Fallback for late fusion only scenario or different naming
        potential_keys = [k for k in results.keys() if f'({lambda_shift_values[0]})' in k]
        if potential_keys:
            first_key = potential_keys[0]
            
    is_aggregated = 'mean' in results.get(first_key, {})

    if is_aggregated:
        early_f1_mean = [results.get(f'early_({lambda_shift})', {}).get('mean', {}).get('Macro-F1', 0) for lambda_shift in lambda_shift_values]
        early_f1_std = [results.get(f'early_({lambda_shift})', {}).get('std', {}).get('Macro-F1', 0) for lambda_shift in lambda_shift_values]
        
        late_f1_mean = [results.get(f'late_({lambda_shift})', {}).get('mean', {}).get('Macro-F1', 0) for lambda_shift in lambda_shift_values]
        late_f1_std = [results.get(f'late_({lambda_shift})', {}).get('std', {}).get('Macro-F1', 0) for lambda_shift in lambda_shift_values]

        early_acc_mean = [results.get(f'early_({lambda_shift})', {}).get('mean', {}).get('Acc', 0) for lambda_shift in lambda_shift_values]
        early_acc_std = [results.get(f'early_({lambda_shift})', {}).get('std', {}).get('Acc', 0) for lambda_shift in lambda_shift_values]
        
        late_acc_mean = [results.get(f'late_({lambda_shift})', {}).get('mean', {}).get('Acc', 0) for lambda_shift in lambda_shift_values]
        late_acc_std = [results.get(f'late_({lambda_shift})', {}).get('std', {}).get('Acc', 0) for lambda_shift in lambda_shift_values]
        
        early_auc_mean = [results.get(f'early_({lambda_shift})', {}).get('mean', {}).get('AUC', 0) for lambda_shift in lambda_shift_values]
        early_auc_std = [results.get(f'early_({lambda_shift})', {}).get('std', {}).get('AUC', 0) for lambda_shift in lambda_shift_values]
        
        late_auc_mean = [results.get(f'late_({lambda_shift})', {}).get('mean', {}).get('AUC', 0) for lambda_shift in lambda_shift_values]
        late_auc_std = [results.get(f'late_({lambda_shift})', {}).get('std', {}).get('AUC', 0) for lambda_shift in lambda_shift_values]
    else:
        # Extracting F1 and Accuracy values for early and late fusion models (Backward compatibility)
        early_f1_mean = [results[f'early_({lambda_shift})']['Macro-F1']['F1'] if f'early_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        early_f1_std = None
        late_f1_mean = [results[f'late_({lambda_shift})']['Macro-F1']['F1'] if f'late_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        late_f1_std = None

        early_acc_mean = [results[f'early_({lambda_shift})']['Acc']['Acc'] if f'early_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        early_acc_std = None
        late_acc_mean = [results[f'late_({lambda_shift})']['Acc']['Acc'] if f'late_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        late_acc_std = None
        
        early_auc_mean = [results[f'early_({lambda_shift})']['AUC']['Auc'] if f'early_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        early_auc_std = None
        late_auc_mean = [results[f'late_({lambda_shift})']['AUC']['Auc'] if f'late_({lambda_shift})' in results else 0 for lambda_shift in lambda_shift_values]
        late_auc_std = None

    # Determine which models have data
    has_early = any(v != 0 for v in early_f1_mean)
    has_late = any(v != 0 for v in late_f1_mean)
    
    if not has_early and not has_late:
        print("No results to plot.")
        return

    # Function to plot with or without error bars
    def plot_metric(ax, x, y_mean, y_std, label, color):
        valid_indices = [i for i, val in enumerate(y_mean) if val != 0]
        if not valid_indices:
            return

        x_filtered = [x[i] for i in valid_indices]
        y_mean_filtered = [y_mean[i] for i in valid_indices]
        if y_std:
            y_std_filtered = [y_std[i] for i in valid_indices]
        else:
            y_std_filtered = None

        if y_std_filtered is not None and any(v > 0 for v in y_std_filtered):
            ax.errorbar(x_filtered, y_mean_filtered, yerr=y_std_filtered, fmt='-o', label=label, color=color, capsize=5)
        else:
            ax.plot(x_filtered, y_mean_filtered, marker='o', linestyle='-', label=label, color=color)
        ax.legend()
        ax.grid(True)

    def generate_plot(model_name, f1_mean, f1_std, acc_mean, acc_std, auc_mean, auc_std, color, filename_suffix):
        figsize = (7, 15)
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        axs = axs.reshape(3)

        # Plot F1 Score
        ax = axs[0]
        plot_metric(ax, lambda_shift_values, f1_mean, f1_std, f'{model_name} F1 Score', color)
        ax.set_title(f'{model_name} F1 Score - {DATASET}')
        ax.set_xlabel('Lambda Shift')
        ax.set_ylabel('F1 Score')

        # Plot Accuracy
        ax = axs[1]
        plot_metric(ax, lambda_shift_values, acc_mean, acc_std, f'{model_name} Accuracy', color)
        ax.set_title(f'{model_name} Accuracy - {DATASET}')
        ax.set_xlabel('Lambda Shift')
        ax.set_ylabel('Accuracy')

        # Plot AUC
        ax = axs[2]
        plot_metric(ax, lambda_shift_values, auc_mean, auc_std, f'{model_name} AUC', color)
        ax.set_title(f'{model_name} AUC - {DATASET}')
        ax.set_xlabel('Lambda Shift')
        ax.set_ylabel('AUC')

        plt.tight_layout()
        
        img_path_metrics = os.path.join(output_dir, f'Metrics_{filename_suffix}.pdf')
        os.makedirs(os.path.dirname(img_path_metrics), exist_ok=True)
        plt.savefig(img_path_metrics)
        plt.close(fig)

    if has_early:
        generate_plot('Early Fusion', early_f1_mean, early_f1_std, early_acc_mean, early_acc_std, early_auc_mean, early_auc_std, 'b', 'Early')

    if has_late:
        generate_plot('Late Fusion', late_f1_mean, late_f1_std, late_acc_mean, late_acc_std, late_auc_mean, late_auc_std, 'r', 'Late')
    
    
def update_column_names(columns, new_size):
    """Update column names based on the new size of the embeddings."""
    prefix = columns[0].split('_')[0]  # Extracts 'text' or 'image' from the first column name
    new_columns = [f"{prefix}_{i+1}" for i in range(new_size)]
    return new_columns