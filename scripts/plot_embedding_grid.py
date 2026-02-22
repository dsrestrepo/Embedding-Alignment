import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Add src to python path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import calculate_spherical_statistics, normalize_embeddings

# Configuration
EMBEDDINGS_BASE = "Embeddings_vlm"
OUTPUT_DIR = "Images/Embedding_Plots/Summary_Grids"
MAX_SAMPLES = 5000

# Set plot style for academic papers with customized fonts/sizes
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
    # Attempt to load from various potential path structures
    # 1. Standard: Embeddings_vlm/dataset/embeddings_model.csv (based on previous script analysis)
    # The file name format in plot_embeddings.py was 'embeddings_model.csv' or similar.
    # In generate_embeddings.py: output_file = f'embeddings_{model.lower()}.csv'
    
    filename = f"embeddings_{model.lower()}.csv"
    
    # Path possibility 1: Embeddings_vlm/dataset/embeddings_model.csv
    path1 = os.path.join(EMBEDDINGS_BASE, dataset, filename)
    
    # Path possibility 2: Embeddings_vlm/dataset/model/run/embeddings.csv (older structure?)
    path2 = os.path.join(EMBEDDINGS_BASE, dataset, model, run, "embeddings.csv")

    if os.path.exists(path1):
        embeddings_path = path1
    elif os.path.exists(path2):
        embeddings_path = path2
    else:
        # Fallback to check if user provided a specific file name mapping in previous scripts
        # Let's try flexible matching if needed, but for now specific is better.
        print(f"File not found: {path1}")
        return None, None
        
    print(f"Loading embeddings from {embeddings_path}...")
    try:
        df = pd.read_csv(embeddings_path)
        
        # Identify text and image columns
        text_cols = [c for c in df.columns if c.startswith("text_embedding") or c.startswith("text_emb_") or c.startswith("text_emb_")]
        img_cols = [c for c in df.columns if c.startswith("image_embedding") or c.startswith("img_emb_") or c.startswith("img_emb_")]
        
        if not text_cols or not img_cols:
            print(f"Error: Could not identify embedding columns in {embeddings_path}")
            return None, None
            
        text_emb = df[text_cols].values
        img_emb = df[img_cols].values
        
        # Normalize
        text_emb = normalize_embeddings(text_emb)
        img_emb = normalize_embeddings(img_emb)
        
        return text_emb, img_emb
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

def plot_embeddings_on_ax(ax, text_emb, img_emb, title, dim='2d', stats=None):
    """
    Plot embeddings on a specific axis.
    Does NOT save figure or close it.
    """
    
    # Subsample if needed for performance/visuals
    if len(text_emb) > MAX_SAMPLES:
        indices = np.random.choice(len(text_emb), MAX_SAMPLES, replace=False)
        text_emb = text_emb[indices]
        img_emb = img_emb[indices]

    # Combine for PCA
    all_emb = np.concatenate([text_emb, img_emb])
    
    # Calculate R and V stats if not provided
    if stats is None:
        text_R, text_V = calculate_spherical_statistics(text_emb)
        img_R, img_V = calculate_spherical_statistics(img_emb)
    else:
        text_R, text_V = stats.get('text_R', 0), stats.get('text_V', 0)
        img_R, img_V = stats.get('img_R', 0), stats.get('img_V', 0)

    # Title with statistics
    # Format carefully to fit in small subplots
    stats_str = f"T: R={text_R:.2f} | I: R={img_R:.2f}"
    
    if dim == '2d':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X=all_emb)
        
        red_text = reduced[:len(text_emb)]
        red_img = reduced[len(text_emb):]
        
        # Draw connections
        for i in range(min(len(red_text), 1000)): # Limit connections for speed if dense
            ax.plot([red_text[i, 0], red_img[i, 0]],
                    [red_text[i, 1], red_img[i, 1]],
                    color='gray', alpha=0.1, linewidth=0.3, zorder=1)

        ax.scatter(red_text[:, 0], red_text[:, 1], 
                  label='Text', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#1f77b4')
        ax.scatter(red_img[:, 0], red_img[:, 1], 
                  label='Image', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#ff7f0e')
        
        # Add stats inside the plot to save space? Or subtitle?
        # Subplot title is handled by caller usually, but we can add stats here
        ax.text(0.5, 0.98, stats_str, transform=ax.transAxes, ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel('PC 1', fontsize=8)
        # ax.set_ylabel('PC 2', fontsize=8) # Save space?
        ax.grid(True, linestyle='--', alpha=0.3)
        # Legend only on first plot perhaps? Or simplified
        # ax.legend(loc='upper right', fontsize=6)
        
        # Center the plot
        max_range = np.max(np.abs(reduced)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')
        
    elif dim == '3d':
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(X=all_emb)
        # Normalize to sphere for 3D viz
        reduced = normalize_embeddings(reduced)
        
        red_text = reduced[:len(text_emb)]
        red_img = reduced[len(text_emb):]
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.05, linewidth=0.5)
        
        # Connections (Geodesic Arcs)
        for i in range(min(len(red_text), 500)): 
             p1 = red_text[i]
             p2 = red_img[i]
             t_values = np.linspace(0, 1, 20) # More points for smooth curve
             
             # Linear interpolation
             interp = np.outer(1 - t_values, p1) + np.outer(t_values, p2)
             
             # Project to sphere surface (L2 normalization)
             interp_norm = np.linalg.norm(interp, axis=1, keepdims=True)
             interp_normalized = interp / interp_norm
             
             ax.plot(interp_normalized[:, 0], interp_normalized[:, 1], interp_normalized[:, 2],
                     color='gray', alpha=0.1, linewidth=0.3)

        ax.scatter(red_text[:, 0], red_text[:, 1], red_text[:, 2], 
                  label='Text', alpha=0.6, s=5, c='#1f77b4', depthshade=True)
        ax.scatter(red_img[:, 0], red_img[:, 1], red_img[:, 2], 
                  label='Image', alpha=0.6, s=5, c='#ff7f0e', depthshade=True)
                  
        ax.set_title(stats_str, fontsize=8)
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

def create_4x4_grid_figure(datasets, models, output_dir, dim='2d'):
    """
    Creates a 4x4 grid figure.
    Rows: Datasets
    Cols: Models
    """
    print(f"Generating 4x4 {dim.upper()} Grid...")
    
    # Adjust size for 4x4
    fig = plt.figure(figsize=(16, 16))
    
    # GridSpec logic or just subplots
    # If 3D, we need proper projection spec
    if dim == '2d':
        axes = fig.subplots(4, 4) # Returns 4x4 array
    
    # Dataset Display Names
    ds_names = {
        "mimic": "MIMIC-CXR", "ham10000": "HAM10000", 
        "brset": "BRSET", "mbrset": "mBRSET",
        "coco-qa": "COCO-QA"
    }
    
    # Model Display Names
    model_names = {
        "CLIP": "CLIP\n(General)", "SigLIP": "SigLIP\n(General)", 
        "BioMedCLIP": "BioMedCLIP\n(Medical)", "MedSigLIP": "MedSigLIP\n(Medical)"
    }
    
    # Cache data to avoid reloading if possible, though strict loop is fine
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            
            # Setup Axis
            if dim == '3d':
                ax = fig.add_subplot(4, 4, i * 4 + j + 1, projection='3d')
            else:
                ax = axes[i, j]
            
            print(f"Processing {dataset} - {model}...")
            text_emb, img_emb = load_embeddings(dataset, model)
            
            if text_emb is not None:
                plot_embeddings_on_ax(ax, text_emb, img_emb, f"{dataset}-{model}", dim=dim)
            else:
                ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            # Axis Labels / Titles logic similar to AUC plot
            # Top Row: Model Titles
            if i == 0:
                ax.set_title(model_names.get(model, model), fontweight='bold', pad=15, fontsize=14)
            
            # Left Column: Dataset Titles
            if j == 0:
                if dim == '2d':
                    ax.set_ylabel(f"{ds_names.get(dataset, dataset).upper()}", fontweight='bold', fontsize=14, labelpad=10)
                else:
                    ax.set_zlabel(f"{ds_names.get(dataset, dataset).upper()}", fontweight='bold', fontsize=12)

    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08, right=0.98, hspace=0.3, wspace=0.2)
    
    # Add Separator Line (Visual)
    from matplotlib.lines import Line2D
    # Vertical line splitting Generalist (cols 0,1) and Medical (cols 2,3)
    # Roughly at 0.53 figure coordinate X
    line = Line2D([0.525, 0.525], [0.02, 0.94], transform=fig.transFigure, color="black", linestyle="--", linewidth=1.5)
    fig.add_artist(line)

    # Super Headers
    fig.text(0.30, 0.96, "Generalist Models", ha='center', fontsize=18, fontweight='bold', color='#1f77b4')
    fig.text(0.75, 0.96, "Medical Models", ha='center', fontsize=18, fontweight='bold', color='#d62728')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"Medical_Datasets_Embedding_Grid_4x4_{dim}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")


def main():
    global EMBEDDINGS_BASE, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="Generate 4x4 Embedding Grid Plots")
    parser.add_argument('--base_dir', type=str, default=EMBEDDINGS_BASE, help="Base directory containing dataset folders")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # Update global configs with args
    EMBEDDINGS_BASE = args.base_dir
    OUTPUT_DIR = args.output_dir

    # Define the datasets and models for the 4x4 grid
    # Matching plot_auc_summary.py
    datasets = ["mimic", "ham10000", "brset", "mbrset"]
    models = ["CLIP", "SigLIP", "BioMedCLIP", "MedSigLIP"]
    
    # Generate 2D Grid
    create_4x4_grid_figure(datasets, models, OUTPUT_DIR, dim='2d')
    
    # Optional: Generate 3D Grid (might be too crowded for 4x4 but available)
    # create_4x4_grid_figure(datasets, models, OUTPUT_DIR, dim='3d')

if __name__ == "__main__":
    main()
