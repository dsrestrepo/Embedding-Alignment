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
OUTPUT_DIR = "Images/Embedding_Plots/Summary_Mixed"
MAX_SAMPLES = 5000

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
        
    print(f"Loading embeddings from {embeddings_path}...")
    try:
        df = pd.read_csv(embeddings_path)
        text_cols = [c for c in df.columns if c.startswith("text_embedding") or c.startswith("text_emb_") or c.startswith("text_emb_")]
        img_cols = [c for c in df.columns if c.startswith("image_embedding") or c.startswith("img_emb_") or c.startswith("img_emb_")]
        
        if not text_cols or not img_cols:
            return None, None
            
        text_emb = df[text_cols].values
        img_emb = df[img_cols].values
        
        text_emb = normalize_embeddings(text_emb)
        img_emb = normalize_embeddings(img_emb)
        
        return text_emb, img_emb
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

def plot_and_collect_stats(ax, text_emb, img_emb, dataset, model, stats_collection, dim='2d'):
    """
    Plot embeddings and collect stats for tables.
    """
    # Calculate R and V stats (using full data before subsampling)
    text_R, text_V = calculate_spherical_statistics(text_emb)
    img_R, img_V = calculate_spherical_statistics(img_emb)
    
    # Store stats
    if dataset not in stats_collection:
        stats_collection[dataset] = {}
    stats_collection[dataset][model] = {
        'text_R': text_R, 'text_V': text_V,
        'img_R': img_R, 'img_V': img_V
    }

    # Subsample for plotting
    if len(text_emb) > MAX_SAMPLES:
        indices = np.random.choice(len(text_emb), MAX_SAMPLES, replace=False)
        text_emb = text_emb[indices]
        img_emb = img_emb[indices]
    
    # Combine for PCA
    all_emb = np.concatenate([text_emb, img_emb])
    
    # Stats string
    stats_str = f"T: R={text_R:.2f} | I: R={img_R:.2f}"

    if dim == '2d':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X=all_emb)
        
        red_text = reduced[:len(text_emb)]
        red_img = reduced[len(text_emb):]
        
        # Connections
        for i in range(min(len(red_text), 1000)): 
            ax.plot([red_text[i, 0], red_img[i, 0]],
                    [red_text[i, 1], red_img[i, 1]],
                    color='gray', alpha=0.1, linewidth=0.3, zorder=1)

        ax.scatter(red_text[:, 0], red_text[:, 1], 
                  label='Text', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#1f77b4')
        ax.scatter(red_img[:, 0], red_img[:, 1], 
                  label='Image', alpha=0.6, s=10, edgecolors='w', linewidth=0.2, zorder=2, c='#ff7f0e')
        
        ax.text(0.5, 0.98, stats_str, transform=ax.transAxes, ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        max_range = np.max(np.abs(reduced)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')

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
                  
        ax.set_title(stats_str, fontsize=8)
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])


def generate_latex_tables(stats_collection, output_dir):
    """
    Generates two LaTeX tables for Mean Resultant Length (R):
    1. Table for Text Embeddings (T)
    2. Table for Image Embeddings (I)
    """
    latex_content = []
    
    # Define groups
    medical_datasets = ["mimic", "ham10000", "brset", "mbrset"]
    natural_datasets = ["coco-qa", "fakeddit", "Recipes5k", "daquar"]
    models = ["CLIP", "SigLIP", "BioMedCLIP", "MedSigLIP"]
    
    # helper for names
    ds_nice = {
        "mimic": "MIMIC-CXR", "ham10000": "HAM10000", 
        "brset": "BRSET", "mbrset": "mBRSET",
        "coco-qa": "COCO-QA", "fakeddit": "Fakeddit",
        "Recipes5k": "Recipes5k", "daquar": "DAQUAR"
    }

    # Helper function to generate a table for a specific modality
    def create_modality_table(modality_key, modality_name, label):
        latex_content.append(f"% Table: Mean Resultant Length (R) - {modality_name}")
        latex_content.append("\\begin{table}[h]") 
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{Mean Resultant Length ($R$) - {modality_name} Embeddings (Cone Strength)}}")
        
        # Columns: Dataset | CLIP | SigLIP | BioMedCLIP | MedSigLIP
        latex_content.append("\\begin{tabular}{lcccc}") 
        latex_content.append("\\toprule")
        
        # Header Row
        mod_header = ["\\textbf{Dataset}"]
        for m in models:
            mod_header.append(f"\\textbf{{{m}}}")
        latex_content.append(" & ".join(mod_header) + " \\\\")
        latex_content.append("\\midrule")
        
        # Function to print dataset group rows
        def print_group_rows(d_list, group_label):
            latex_content.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group_label}}}}} \\\\")
            latex_content.append("\\midrule")
            
            for ds in d_list:
                row = [ds_nice.get(ds, ds)]
                for model in models:
                    if ds in stats_collection and model in stats_collection[ds]:
                        s = stats_collection[ds][model]
                        # Use text_R or img_R based on modality_key
                        val = s[f'{modality_key}_R']
                        row.append(f"{val:.3f}")
                    else:
                        row.append("-")
                latex_content.append(" & ".join(row) + " \\\\")

        print_group_rows(medical_datasets, "Medical Datasets")
        latex_content.append("\\midrule")
        print_group_rows(natural_datasets, "Natural Datasets")

        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\end{table}")
        latex_content.append("\n\n") # Spacing between tables

    # Generate Table for Text Embeddings
    create_modality_table('text', 'Text', 'tab:cone_strength_text')
    
    # Generate Table for Image Embeddings
    create_modality_table('img', 'Image', 'tab:cone_strength_image')
    
    out_file = os.path.join(output_dir, "all_datasets_tables_latex.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_content))
    print(f"Saved Full LaTeX tables to {out_file}")


def collect_all_stats(models):
    """
    Collects stats for ALL datasets, regardless of what is plotted in the mixed grid.
    """
    datasets = ["mimic", "ham10000", "brset", "mbrset", "coco-qa", "fakeddit", "Recipes5k", "daquar"]
    stats_collection = {}
    
    print("Collecting stats for ALL datasets for LaTeX tables...")
    for ds in datasets:
        if ds not in stats_collection:
            stats_collection[ds] = {}
        for model in models:
            print(f"Loading stats for {ds} - {model}...")
            text_emb, img_emb = load_embeddings(ds, model)
            if text_emb is not None:
                # Reuse the plotting function logic just for stats collection
                text_R, text_V = calculate_spherical_statistics(text_emb)
                img_R, img_V = calculate_spherical_statistics(img_emb)
                
                stats_collection[ds][model] = {
                    'text_R': text_R, 'text_V': text_V,
                    'img_R': img_R, 'img_V': img_V
                }
    return stats_collection

def create_mixed_grid(datasets, models, output_dir, dim='2d'):
    print(f"Generating Mixed 4x4 {dim.upper()} Grid...")
    
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
    
    stats_collection = {}
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            
            if dim == '3d':
                ax = fig.add_subplot(4, 4, i * 4 + j + 1, projection='3d')
            else:
                ax = axes[i, j]
                
            print(f"Processing {dataset} - {model}...")
            text_emb, img_emb = load_embeddings(dataset, model)
            
            if text_emb is not None:
                # We can reuse the stats collecting plotting, although we already computed full stats, 
                # we do it again here for the labels inside the plot (which use subsampled data? No, full data)
                # Let's just collect again, it's safer than passing the huge dict around
                plot_and_collect_stats(ax, text_emb, img_emb, dataset, model, stats_collection, dim=dim)
            else:
                ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            if i == 0:
                ax.set_title(model_names.get(model, model), fontweight='bold', pad=15, fontsize=14)
            
            if j == 0:
                if dim == '2d':
                    ax.set_ylabel(f"{ds_names.get(dataset, dataset).upper()}", fontweight='bold', fontsize=14, labelpad=10)
                else:
                    # For 3D, put label on left
                    ax.text2D(-0.1, 0.5, f"{ds_names.get(dataset, dataset).upper()}", 
                             transform=ax.transAxes, fontweight='bold', fontsize=12, rotation='vertical', va='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08, right=0.98, hspace=0.3, wspace=0.2)
    
    from matplotlib.lines import Line2D
    # Vertical Separator
    line_v = Line2D([0.525, 0.525], [0.02, 0.94], transform=fig.transFigure, color="black", linestyle="--", linewidth=1.5)
    fig.add_artist(line_v)
    
    # Horizontal Separator (between row 1 (mbrset) and 2 (coco-qa))
    # Y = 0.5 roughly
    line_h = Line2D([0.02, 0.98], [0.5, 0.5], transform=fig.transFigure, color="black", linestyle="-", linewidth=2.0)
    fig.add_artist(line_h)

    fig.text(0.30, 0.96, "Generalist Models", ha='center', fontsize=18, fontweight='bold', color='#1f77b4')
    fig.text(0.75, 0.96, "Medical Models", ha='center', fontsize=18, fontweight='bold', color='#d62728')
    
    # Side headers
    # Slightly adjusted X position due to rotated label
    fig.text(0.02, 0.72, "Medical", rotation=90, va='center', ha='center', fontsize=16, fontweight='bold', color='#444444')
    fig.text(0.02, 0.25, "Natural", rotation=90, va='center', ha='center', fontsize=16, fontweight='bold', color='#444444')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"Mixed_Datasets_Embedding_Grid_4x4_{dim}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mixed Embedding Grid Plots")
    parser.add_argument('--base_dir', type=str, default=EMBEDDINGS_BASE, help="Base directory containing dataset folders")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # Update global configs with args
    EMBEDDINGS_BASE = args.base_dir
    OUTPUT_DIR = args.output_dir

    # 2 Medical + 2 Natural for the PLOT
    datasets_plot = ["mimic", "mbrset", "coco-qa", "fakeddit"]
    models = ["CLIP", "SigLIP", "BioMedCLIP", "MedSigLIP"]
    
    # Generate Stats for ALL datasets (Tables)
    print("Collecting comprehensive stats...")
    full_stats = collect_all_stats(models)
    generate_latex_tables(full_stats, OUTPUT_DIR)
    
    # 2D Grid Plot
    create_mixed_grid(datasets_plot, models, OUTPUT_DIR, dim='2d')
    
    # 3D Grid Plot
    create_mixed_grid(datasets_plot, models, OUTPUT_DIR, dim='3d')
