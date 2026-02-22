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


def calculate_spherical_statistics(embeddings):
    """
    Calculate spherical statistics for a set of embeddings.
    
    Args:
        embeddings: numpy array of shape (N, D), assumed to be L2 normalized.
        
    Returns:
        mean_resultant_length (R): Measure of concentration (0 to 1). Higher = more concentrated (stronger cone).
        spherical_variance (V): Measure of dispersion (0 to 1). Higher = more dispersed. V = 1 - R.
    """
    # Calculate the resultant vector (sum of all vectors)
    resultant_vector = np.sum(embeddings, axis=0)
    
    # Calculate the length of the resultant vector
    R_magnitude = np.linalg.norm(resultant_vector)
    
    # Mean Resultant Length (R_bar)
    n = embeddings.shape[0]
    mean_resultant_length = R_magnitude / n
    
    # Spherical Variance
    spherical_variance = 1 - mean_resultant_length
    
    return mean_resultant_length, spherical_variance

def visualize_embeddings(text_embeddings, image_embeddings, title, lambda_shift, DATASET, save=True, var=False, output_dir=None, max_samples=5000):
    """Visualize embeddings in 2D and 3D, including the unit circle and sphere."""
    if output_dir is None:
        output_dir = f'Images/{DATASET}'
    
    # Subsample if there are too many points
    if len(text_embeddings) > max_samples:
        print(f"Subsampling {len(text_embeddings)} data points to {max_samples} for cleaner visualization.")
        indices = np.random.choice(len(text_embeddings), max_samples, replace=False)
        text_embeddings = text_embeddings[indices]
        image_embeddings = image_embeddings[indices]
    
    # Set plot style for academic papers
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

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

    # ---------------- 2D PLOT ----------------
    fig_2d = plt.figure(figsize=(8, 8))
    ax2 = plt.gca()

    # Draw connections
    for i in range(len(reduced_text_embeddings)):
        ax2.plot([reduced_text_embeddings[i, 0], reduced_image_embeddings[i, 0]],
                 [reduced_text_embeddings[i, 1], reduced_image_embeddings[i, 1]],
                 color='gray', alpha=0.2, linewidth=0.5, zorder=1)

    ax2.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], 
                label='Text', alpha=0.7, s=30, edgecolors='w', linewidth=0.5, zorder=2, c='#1f77b4')
    ax2.scatter(reduced_image_embeddings[:, 0], reduced_image_embeddings[:, 1], 
                label='Image', alpha=0.7, s=30, edgecolors='w', linewidth=0.5, zorder=2, c='#ff7f0e')
    
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plot_title = title + ' (2D PCA)'
    ax2.set_title(plot_title, pad=20)
        
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    
    # Center the plot
    max_range = np.max(np.abs(reduced_embeddings)) * 1.1
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    img_path_2d = os.path.join(output_dir, f'2d_shift({lambda_shift}).pdf')
    if save:
        os.makedirs(os.path.dirname(img_path_2d), exist_ok=True)
        plt.savefig(img_path_2d, bbox_inches='tight', dpi=300)
    plt.close(fig_2d)

    # ---------------- 3D PLOT ----------------
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(all_embeddings)
    
    # Normalize the 3D embeddings to lie on the unit sphere
    reduced_embeddings_3d = normalize_embeddings(reduced_embeddings_3d)

    reduced_text_embeddings_3d = reduced_embeddings_3d[:len(text_embeddings)]
    reduced_image_embeddings_3d = reduced_embeddings_3d[len(text_embeddings):]

    fig_3d = plt.figure(figsize=(10, 10))
    ax3 = fig_3d.add_subplot(111, projection='3d')
    
    # Clean up 3D pane
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.grid(False) # Remove grid lines for cleaner look
    
    # Draw a unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = 0.98 * np.outer(np.cos(u), np.sin(v))
    y = 0.98 * np.outer(np.sin(u), np.sin(v))
    z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_wireframe(x, y, z, color="gray", alpha=0.05, linewidth=0.5)

    # Draw geodesic arcs between corresponding text and image embeddings
    for i in range(len(reduced_text_embeddings_3d)):
        p1 = reduced_text_embeddings_3d[i]
        p2 = reduced_image_embeddings_3d[i]
        
        # Create interpolation points
        num_points = 20
        t_values = np.linspace(0, 1, num_points)
        # Linear interpolation
        interp = np.outer(1 - t_values, p1) + np.outer(t_values, p2)
        # Normalize to project onto sphere surface
        interp_norm = np.linalg.norm(interp, axis=1, keepdims=True)
        interp_normalized = interp / interp_norm
        
        ax3.plot(interp_normalized[:, 0], interp_normalized[:, 1], interp_normalized[:, 2],
                color='gray', alpha=0.2, linewidth=0.5)

    ax3.scatter(reduced_text_embeddings_3d[:, 0], reduced_text_embeddings_3d[:, 1], reduced_text_embeddings_3d[:, 2], 
               label='Text', alpha=0.8, s=20, depthshade=True, c='#1f77b4')
    ax3.scatter(reduced_image_embeddings_3d[:, 0], reduced_image_embeddings_3d[:, 1], reduced_image_embeddings_3d[:, 2], 
               label='Image', alpha=0.8, s=20, depthshade=True, c='#ff7f0e')

    plot_title_3d = title + ' (3D Spherical Projection)'
    ax3.set_title(plot_title_3d, pad=20)
        
    ax3.set_xlabel('PC 1')
    ax3.set_ylabel('PC 2')
    ax3.set_zlabel('PC 3')
    
    # Set consistent view limits
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    
    ax3.legend(loc='upper right')
        
    img_path_3d = os.path.join(output_dir, f'3d_shift({lambda_shift}).pdf')
    if save:
        os.makedirs(os.path.dirname(img_path_3d), exist_ok=True)
        plt.savefig(img_path_3d, bbox_inches='tight', dpi=300)
    plt.close(fig_3d)

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