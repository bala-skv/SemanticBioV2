#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_heatmaps():
    results_path = Path("runs/pilot_trace_results.json")
    if not results_path.exists():
        print(f"Cannot find {results_path}")
        return
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    conditions = list(data.keys())
    if len(conditions) == 0:
        return
        
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 8))
    if len(conditions) == 1:
        axes = [axes]
        
    for ax, cond in zip(axes, conditions):
        heatmap_data = np.array(data[cond]) # shape (28, 12)
        # Flip the array so layer 0 is at the bottom
        heatmap_data = np.flip(heatmap_data, axis=0)
        
        sns.heatmap(heatmap_data, ax=ax, cmap="RdBu_r", center=0, vmin=-1.0, vmax=1.0,
                    cbar_kws={'label': 'Recovery'}, xticklabels=2, yticklabels=2)
        
        ax.set_title(cond)
        ax.set_xlabel("Head ID")
        ax.set_ylabel("Layer ID")
        
        # Adjust y-axis ticks since we flipped it (0 at bottom)
        ax.set_yticks(np.arange(0.5, 28.5, 2))
        ax.set_yticklabels(np.arange(27, -1, -2))

    plt.tight_layout()
    out_file = "runs/pilot_heatmaps.png"
    plt.savefig(out_file, dpi=300)
    print(f"Heatmaps saved to {out_file}")

if __name__ == "__main__":
    plot_heatmaps()