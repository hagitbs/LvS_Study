import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_consolidated_analysis(df_doc, doc_id, secondary_col, filename):
    # 1. Sort by Element (Labels) alphabetically as requested
    df_doc = df_doc.sort_values("element", ascending=True).reset_index(drop=True)
    labels = [str(e)[:8] for e in df_doc["element"]]
    x = np.arange(len(df_doc))
    
    # 2. Define Scaling Factor
    # This brings values like 0.002 up to 0.10 so they share the axis with bars
    scaling_factor = 1
    scaled_values = df_doc[secondary_col].values * scaling_factor

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 3. Plot Bars (Proportions)
    ax.bar(x, df_doc["expected"], width=0.7, color='#D3D3D3', label='Expected', alpha=0.8, zorder=2)
    ax.bar(x, df_doc["observed"], width=0.3, color="#8bacc3", label='Observed', alpha=1.0, zorder=3)
    
    # 4. Plot Arrows with Lines (Scaled secondary metric)
    for i, val in enumerate(scaled_values):
        color = 'green' if val >= 0 else 'red'
        marker = '^' if val >= 0 else 'v'
        
        # Draw vertical shaft from 0
        ax.vlines(x[i], 0, val, color=color, linestyle='-', linewidth=0.01, alpha=0.7, zorder=4)
        # Draw arrow head at scaled position
        ax.scatter(x[i], val, marker=marker, color=color, s=80, edgecolors='black', zorder=5)

    # 5. Axis Constraints & Pulling X-axis up
    # We use a very small negative padding to reduce the blank area at bottom
    y_max = max(df_doc["expected"].max(), df_doc["observed"].max(), scaled_values.max()) * 1.1
    ax.set_ylim(-0.05, y_max) 
    
    # 6. Formatting
    ax.set_ylabel(f"Share / Scaled {secondary_col} (x{scaling_factor})", fontsize=12, fontweight='bold')
    ax.set_title(f"Consolidated Analysis: {doc_id}", fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.axhline(0, color='black', linewidth=0.001, zorder=1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='#D3D3D3', lw=6, label='Expected Share'),
        Line2D([0], [0], color='#8bacc3', lw=4, label='Observed Share'), 
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label=f'{secondary_col} (+)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label=f'{secondary_col} (-)'),
        Line2D([0], [0], color='white', linestyle=':', lw=0.001, label='')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    plt.tight_layout()
    # Pull the labels closer to the chart area
    plt.subplots_adjust(bottom=0.2) 
    
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

# --- Run Loop ---
df = pd.read_csv('/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis_cod101.csv')
out_dir = "/Users/hagitbenshoshan/Documents/PHD/Market/all_years_lvs"

for yr in sorted(df['document'].unique()):
    subset = df[df['document'] == yr]
    plot_consolidated_analysis(subset, yr, 'gap_val', f"{out_dir}/gap_{yr}.png") 