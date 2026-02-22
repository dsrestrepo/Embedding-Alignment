import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot 4x4 AUC summary grid.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory containing the dataset/model folders")
    parser.add_argument('--output_dir', type=str, default="Images/Summary", help="Output directory for the summary plot")
    args = parser.parse_args()

    # Define the 4 Medical Datasets (Rows)
    datasets = ["mimic", "ham10000", "brset", "mbrset"]
    dataset_titles = ["MIMIC-CXR", "HAM10000", "BRSET", "mBRSET"]

    # Define the 4 Models (Columns) - Grouped: Generalist first, then Medical
    models = ["CLIP", "SigLIP", "BioMedCLIP", "MedSigLIP"]
    model_titles = ["CLIP\n(General)", "SigLIP\n(General)", "BioMedCLIP\n(Medical)", "MedSigLIP\n(Medical)"]

    # Lambda shifts (assuming standard set used in experiments)
    lambda_shifts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Setup the plot styled for paper
    # using a style that looks good for scientific papers
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'grid.linestyle': '--',
        'grid.alpha': 0.5
    })

    # Do not share axes to allow independent scaling
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=False, sharey=False)
    
    # Adjust layout to make room for titles and spacing
    # wspace increased to accommodate independent y-axis labels
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.98, hspace=0.5, wspace=0.35)
    
    # Iterate Datasets on Rows (i) and Models on Columns (j)
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            # Construct path to metrics.json
            metrics_path = os.path.join(args.base_dir, dataset, model, "metrics.json")
            
            auc_means = []
            auc_stds = []
            found_data = False

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        data = json.load(f)
                    
                    found_data = True
                    # Extract AUC data for 'early' fusion
                    for shift in lambda_shifts:
                        # Key format in metrics.json: "early_({shift})"
                        key = f"early_({shift})"
                        if key not in data:
                             key = f"early_({int(shift)})" if shift == int(shift) else f"early_({shift})"
                        
                        if key in data and 'mean' in data[key]:
                            val = data[key]['mean'].get('AUC', 0)
                            # If value is 0 (missing run?), handle it? Assuming 0 means real 0 or bug.
                            auc_means.append(val)
                            auc_stds.append(data[key]['std'].get('AUC', 0))
                        else:
                            auc_means.append(0)
                            auc_stds.append(0)
                            
                except Exception as e:
                    print(f"Error reading {metrics_path}: {e}")
            
            # Determine Color Scheme
            # Generalist (Indices 0, 1): Blue/Cyan tones
            # Medical (Indices 2, 3): Green/Red tones or distinct
            if j < 2:
                line_color = '#1f77b4' # Blue
                marker_style = 'o'
            else:
                line_color = '#d62728' # Red
                marker_style = 's'

            if found_data and any(x > 0 for x in auc_means):
                # Plot with error bars
                ax.errorbar(lambda_shifts, auc_means, yerr=auc_stds, fmt=f'-{marker_style}', 
                            color=line_color, ecolor='gray', capsize=3, markersize=5, 
                            linewidth=1.5, label='AUC')
                
                # Highlight max point
                max_idx = np.argmax(auc_means)
                ax.plot(lambda_shifts[max_idx], auc_means[max_idx], color='gold', marker='*', 
                        markersize=12, markeredgecolor='black', markeredgewidth=0.5, label='Max', zorder=10)
                
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes, color='gray')

            # Force X axis to 0-1
            ax.set_xlim(-0.05, 1.05)
            # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticks([0, 0.5, 1.0]) # Minimal ticks to avoid clutter if small, or full ticks?
            # User asked for "number in the axis for all plots", let's give standard ticks
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.2)], fontsize=9)
            
            # Formatting labels
            # Column titles (Models) on the first row
            if i == 0:
                ax.set_title(model_titles[j], fontweight='bold', pad=15, fontsize=14)
            
            # Row titles (Datasets) on the first column
            if j == 0:
                ax.set_ylabel(f"{dataset_titles[i]}\nAUC", fontweight='bold', fontsize=14, labelpad=10)
            else:
                # Add Y label just for scale context but maybe no text to save space? 
                # User asked for numbers on axes.
                # Just scale numbers are autoshown by sharey=False
                pass
            
            # X labels on every plot (User requested axis numbers for all plots)
            # ax.set_xlabel("Lambda") 
            
            # Grid
            ax.grid(True, linestyle=':', alpha=0.6)

    # Add a visual separator line between Generalist (col 1) and Medical (col 2)
    # Coordinates in figure fraction. 
    # Left 2 cols take roughly left half.
    # We can draw a vertical line.
    from matplotlib.lines import Line2D
    # Approximate middle of the figure logic
    line = Line2D([0.535, 0.535], [0.08, 0.93], transform=fig.transFigure, color="black", linestyle="--", linewidth=2)
    fig.add_artist(line)

    # Super Titles for Groups
    fig.text(0.32, 0.96, "Generalist Models", ha='center', fontsize=16, fontweight='bold', color='#1f77b4')
    fig.text(0.76, 0.96, "Medical Models", ha='center', fontsize=16, fontweight='bold', color='#d62728')

    # Common X label at bottom
    fig.text(0.5, 0.02, "Lambda Shift (τ)", ha='center', fontsize=16, fontweight='bold')

    # Overall Figure Title disabled to make it cleaner for paper (titles usually in caption)
    # fig.suptitle("Embedding Alignment Evolution: AUC vs Lambda Shift", fontsize=18, y=1.02)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "Medical_Datasets_AUC_Grid_4x4.pdf")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Summary plot saved to {out_path}")

if __name__ == "__main__":
    main()
