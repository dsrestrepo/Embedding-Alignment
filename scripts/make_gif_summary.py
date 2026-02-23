
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import shutil

# Add src to python path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import calculate_spherical_statistics, normalize_embeddings, modify_and_normalize_embeddings

# Configuration
EMBEDDINGS_BASE = "Embeddings_vlm"
OUTPUT_DIR = "Images/Summary_gif"
TEMP_DIR = "Images/Summary_gif/temp_frames"
MAX_SAMPLES = 20000

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})




def load_embeddings(dataset, model, run="run_1"):
    """Load text and image embeddings from CSV."""
    filename = f"embeddings_{model.lower()}.csv"
    path1 = os.path.join(EMBEDDINGS_BASE, dataset, filename)
    path2 = os.path.join(EMBEDDINGS_BASE, dataset, model, run, "embeddings.csv")

    if os.path.exists(path1):
        embeddings_path = path1
    elif os.path.exists(path2):
        embeddings_path = path2
    else:
        print(f"File not found: {path1}")
        return None, None
        
    # print(f"Loading embeddings from {embeddings_path}...")
    try:
        df = pd.read_csv(embeddings_path)
        text_cols = [c for c in df.columns if c.startswith("text_embedding") or c.startswith("text_emb_") or c.startswith("text_emb_")]
        img_cols = [c for c in df.columns if c.startswith("image_embedding") or c.startswith("img_emb_") or c.startswith("img_emb_")]
        
        if not text_cols or not img_cols:
            return None, None
            
        text_emb = df[text_cols].values
        img_emb = df[img_cols].values
        
        # Initial normalization
        text_emb = normalize_embeddings(text_emb)
        img_emb = normalize_embeddings(img_emb)
        
        return text_emb, img_emb
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

def plot_frame(ax, text_emb, img_emb, dataset, model, lambda_val, dim='2d', plot_lines=True):
    """
    Plot embeddings for a specific frame (lambda value).
    """
    
    # Combined for PCA
    all_emb = np.concatenate([text_emb, img_emb])
    
    if dim == '2d':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X=all_emb)
        
        red_text = reduced[:len(text_emb)]
        red_img = reduced[len(text_emb):]
        
        # Connections
        if plot_lines:
            for i in range(min(len(red_text), 1000)): 
                ax.plot([red_text[i, 0], red_img[i, 0]],
                        [red_text[i, 1], red_img[i, 1]],
                        color='gray', alpha=0.1, linewidth=0.3, zorder=1)

        ax.scatter(red_text[:, 0], red_text[:, 1], 
                  label='Text', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#1f77b4')
        ax.scatter(red_img[:, 0], red_img[:, 1], 
                  label='Image', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#ff7f0e')
        
        ax.text(0.5, 0.90, f"Lambda: {lambda_val:.1f}", transform=ax.transAxes, ha='center', va='top', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Remove axis ticks for cleaner look in grid
        ax.set_xticks([])
        ax.set_yticks([])

    elif dim == '3d':
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(X=all_emb)
        reduced = normalize_embeddings(reduced)
        
        red_text = reduced[:len(text_emb)]
        red_img = reduced[len(text_emb):]
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        # Wireframe sphere - DARKER and BOLD
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
        # Changed color to 'black' and increased linewidth/alpha
        ax.plot_wireframe(x, y, z, color="black", alpha=0.2, linewidth=0.8)
        
        # Connections (Geodesic Approximation)
        if plot_lines:
            for i in range(min(len(red_text), 500)): 
                 p1 = red_text[i]
                 p2 = red_img[i]
                 t_values = np.linspace(0, 1, 10)
                 interp = np.outer(1 - t_values, p1) + np.outer(t_values, p2)
                 interp_norm = np.linalg.norm(interp, axis=1, keepdims=True)
                 interp_normalized = interp / interp_norm
                 ax.plot(interp_normalized[:, 0], interp_normalized[:, 1], interp_normalized[:, 2],
                         color='gray', alpha=0.1, linewidth=0.3)

        ax.scatter(red_text[:, 0], red_text[:, 1], red_text[:, 2], 
                  label='Text', alpha=0.6, s=5, c='#1f77b4', depthshade=True)
        ax.scatter(red_img[:, 0], red_img[:, 1], red_img[:, 2], 
                  label='Image', alpha=0.6, s=5, c='#ff7f0e', depthshade=True)
                  
        # Use text2D for stats in 3D plots to avoid overwriting the title
        # Dist = 8 makes the camera closer (default is usually 10), making the sphere appear larger
        ax.dist = 8 
        ax.text2D(0.5, 0.95, f"Lambda: {lambda_val:.1f}", transform=ax.transAxes, ha='center', va='top', fontsize=12)
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])


def create_mixed_grid_frame(datasets, models, lambda_val, output_dir, frame_idx, dim='2d', plot_lines=True, data_cache=None):
    print(f"Generating Frame {frame_idx} (Lambda={lambda_val:.1f})...")
    
    fig = plt.figure(figsize=(16, 16))
    
    if dim == '2d':
        axes = fig.subplots(4, 4)
    else:
        # For 3D we create subplots iteratively
        pass
    
    ds_names = {
        "mimic": "MIMIC-CXR", "mbrset": "mBRSET",
        "coco-qa": "COCO-QA", "fakeddit": "Fakeddit"
    }
    
    model_names = {
        "CLIP": "CLIP\n(General)", "SigLIP": "SigLIP\n(General)", 
        "BioMedCLIP": "BioMedCLIP\n(Medical)", "MedSigLIP": "MedSigLIP\n(Medical)"
    }
    
    # We use a fixed seed for subsampling locally to ensure consistency across frames if we did subsampling inside loop
    # But better to subsample ONCE and store in cache.
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            if dim == '3d':
                ax = fig.add_subplot(4, 4, i * 4 + j + 1, projection='3d')
            else:
                ax = axes[i, j]
            
            # Retrieve from cache
            if data_cache and (dataset, model) in data_cache:
                text_emb, img_emb = data_cache[(dataset, model)]
            else:
                # Should not happen if pre-loaded
                continue

            # Apply shift
            # We copy to avoid modifying the cached original
            if lambda_val > 0:
                t_shifted, i_shifted = modify_and_normalize_embeddings(text_emb, img_emb, lambda_val)
            else:
                t_shifted, i_shifted = text_emb, img_emb
            
            # Plot
            plot_frame(ax, t_shifted, i_shifted, dataset, model, lambda_val, dim=dim, plot_lines=plot_lines)

            if i == 0:
                ax.set_title(model_names.get(model, model), fontweight='bold', pad=15, fontsize=16)
            
            if j == 0:
                if dim == '2d':
                    ax.set_ylabel(f"{ds_names.get(dataset, dataset).upper()}", fontweight='bold', fontsize=16, labelpad=10)
                else:
                    # For 3D, put label on left
                    ax.text2D(-0.1, 0.5, f"{ds_names.get(dataset, dataset).upper()}", 
                             transform=ax.transAxes, fontweight='bold', fontsize=14, rotation='vertical', va='center')

    plt.tight_layout()
    if dim == '3d':
        # Tighter spacing for 3D to make plots larger
        plt.subplots_adjust(top=0.90, left=0.05, right=0.99, hspace=0.1, wspace=0.1)
    else:
        plt.subplots_adjust(top=0.90, left=0.08, right=0.98, hspace=0.2, wspace=0.1)
    
    from matplotlib.lines import Line2D
    # Vertical Separator
    line_v = Line2D([0.525, 0.525], [0.02, 0.94], transform=fig.transFigure, color="black", linestyle="--", linewidth=1.5)
    fig.add_artist(line_v)
    
    if dim == '3d':
        y_line_pos = 0.50 # Moved up slightly for 3D to avoid overlap
    else:
        y_line_pos = 0.48 # Kept same for 2D

    line_h = Line2D([0.02, 0.98], [y_line_pos, y_line_pos], transform=fig.transFigure, color="black", linestyle="-", linewidth=2.0)
    fig.add_artist(line_h)

    fig.text(0.30, 0.96, "Generalist Models", ha='center', fontsize=20, fontweight='bold', color='#1f77b4')
    fig.text(0.75, 0.96, "Medical Models", ha='center', fontsize=20, fontweight='bold', color='#d62728')
    
    # Side headers
    fig.text(0.02, 0.72, "Medical", rotation=90, va='center', ha='center', fontsize=18, fontweight='bold', color='#444444')
    fig.text(0.02, 0.25, "Natural", rotation=90, va='center', ha='center', fontsize=18, fontweight='bold', color='#444444')

    # Global Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', label='Image Emb.', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Text Emb.', markersize=10)
    ]
    
    fig.legend(handles=legend_elements, loc='center', ncol=1, fontsize=12, 
              bbox_to_anchor=(0.525, y_line_pos),
              frameon=True, framealpha=1.0, edgecolor='black')

    # Add Lambda title
    fig.suptitle(f"Embedding Alignment (Lambda = {lambda_val})", fontsize=24, fontweight='bold', y=1.02)

    filename = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filename


def main():
    # Use global variable directly
    global EMBEDDINGS_BASE
    
    parser = argparse.ArgumentParser(description="Generate Mixed Embedding Grid GIF")
    parser.add_argument('--base_dir', type=str, default=EMBEDDINGS_BASE, help="Base directory containing dataset folders")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    EMBEDDINGS_BASE = args.base_dir
    
    if args.output_dir:
        gif_output_dir = args.output_dir
    else:
        gif_output_dir = OUTPUT_DIR

    temp_dir_2d = os.path.join(gif_output_dir, "temp_frames_2d")
    temp_dir_3d = os.path.join(gif_output_dir, "temp_frames_3d")

    os.makedirs(gif_output_dir, exist_ok=True)
    os.makedirs(temp_dir_2d, exist_ok=True)
    os.makedirs(temp_dir_3d, exist_ok=True)

    datasets_plot = ["mimic", "mbrset", "coco-qa", "fakeddit"]
    models = ["CLIP", "SigLIP", "BioMedCLIP", "MedSigLIP"]
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Pre-load data to ensure consistent subsampling
    print("Pre-loading data...")
    data_cache = {}
    for ds in datasets_plot:
        for model in models:
            t, i = load_embeddings(ds, model)
            if t is not None:
                # Subsample ONCE here
                if len(t) > MAX_SAMPLES:
                    indices = np.random.choice(len(t), MAX_SAMPLES, replace=False)
                    t = t[indices]
                    i = i[indices]
                data_cache[(ds, model)] = (t, i)
                print(f"Loaded {ds}-{model}: {len(t)} samples")
    
    # 2D GIF
    filenames_2d = []
    print("Generating 2D frames...")
    for idx, l in enumerate(lambdas):
        fname = create_mixed_grid_frame(datasets_plot, models, l, temp_dir_2d, idx, dim='2d', plot_lines=True, data_cache=data_cache)
        filenames_2d.append(fname)
        
    print("Creating 2D GIF...")
    gif_path_2d = os.path.join(gif_output_dir, "alignment_animation_2d.gif")
    with imageio.get_writer(gif_path_2d, mode='I', duration=0.8) as writer:
        for filename in filenames_2d:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"2D GIF saved to {gif_path_2d}")

    # 3D GIF
    filenames_3d = []
    print("Generating 3D frames...")
    for idx, l in enumerate(lambdas):
        fname = create_mixed_grid_frame(datasets_plot, models, l, temp_dir_3d, idx, dim='3d', plot_lines=True, data_cache=data_cache)
        filenames_3d.append(fname)
        
    print("Creating 3D GIF...")
    gif_path_3d = os.path.join(gif_output_dir, "alignment_animation_3d.gif")
    with imageio.get_writer(gif_path_3d, mode='I', duration=0.8) as writer:
        for filename in filenames_3d:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"3D GIF saved to {gif_path_3d}")
    
    # Cleanup
    shutil.rmtree(temp_dir_2d)
    shutil.rmtree(temp_dir_3d)

if __name__ == "__main__":
    main()
