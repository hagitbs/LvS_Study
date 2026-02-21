import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9_\-\.]", "", s)

def plot_final_analysis(df_doc, doc_id, secondary_col, filename):
    # 1. Sort by Element (Labels) alphabetically
    df_doc = df_doc.sort_values("element", ascending=True).reset_index(drop=True)    
    labels = [str(e)[:8] for e in df_doc["element"]]
    x = np.arange(len(df_doc))
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # --- CONSOLIDATION & SCALING ---
    # Scaling factor to make tiny LvS/Gap values visible on the same axis as bars
    scaling_factor = 50 
    scaled_values = df_doc[secondary_col].values * scaling_factor

    # 2. PRIMARY AXIS (Proportions)
    # INCREASE WIDTH to 1.0 to eliminate the gap between categories
    bar_width_expected = 1.0  # Full width eliminates the gap
    bar_width_observed = 0.4  # Inner bar

    # Bars (left axis)
    #ax1.bar(x, df_doc["expected"], width=0.75, color="grey", alpha=0.35, label="Expected")
    #ax1.bar(x, df_doc["observed"], width=0.45, color="tab:blue", alpha=0.95, label="Observed")
    ax1.set_ylabel("Share (observed / expected)")
    #ax1.set_ylim(y1min, y1max)
    
    ax1.bar(x, df_doc["expected"], width=bar_width_expected, color="grey", alpha=0.35, 
            edgecolor='white', linewidth=0.1, label='Expected', zorder=2)
    
    ax1.bar(x, df_doc["observed"], width=bar_width_observed, color="tab:blue", alpha=0.95,
            edgecolor='navy', linewidth=0.1, label='Observed',  zorder=3)
    
    # 3. SECONDARY MEASURE (Arrows) on the SAME axis
    for i, val in enumerate(scaled_values):
        color = 'green' if val >= 0 else 'red'
        marker = '^' if val >= 0 else 'v'
        # Vertical shaft
        ax1.vlines(x[i], 0, val, color=color, linestyle='None', linewidth=0.01, alpha=0.7, zorder=4)
        # Arrow head
        ax1.scatter(x[i], val, marker=marker, color=color, s=30, 
                    edgecolors='black', linewidths=0.001, zorder=5)

    # 4. AXIS CONSTRAINTS & REDUCING BLANK AREA
    y_max = max(df_doc["expected"].max(), df_doc["observed"].max(), scaled_values.max()) * 1.15
    # Small negative value pulls the X-axis labels closer to the baseline
    ax1.set_ylim(-0.025, y_max)
    
    # 5. FORMATTING & AESTHETICS
    ax1.set_ylabel(f"Share / Scaled {secondary_col} (x{scaling_factor})", fontsize=12, fontweight='bold')
    ax1.set_title(f"Country Analysis: {doc_id}", fontsize=16, fontweight='bold', pad=25)
    
    # Setting ha='right' and rotation=45 pulls labels tight to the axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha='right', fontsize=9)
    
    # Visible zero baseline
    ax1.axhline(0, color='black', linewidth=0.2, zorder=1)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='#D3D3D3', lw=6, label='Expected'),
        Line2D([0], [0], color='#8bacc3', lw=4, label='Observed'),
        Line2D([0], [0], marker='^', color='green', markersize=8, label=f'{secondary_col} (+)', linestyle='None'),
        Line2D([0], [0], marker='v', color='red', markersize=8, label=f'{secondary_col} (-)', linestyle='None'),
        Line2D([0], [0], color='white', linestyle=':', lw=0.001, label='') 
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9) 

    # Tighten layout and adjust bottom margin to remove blank space
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)

# --- EXECUTION ---
 
def plot_document (df,dataset,docs):

    out_dir = f"results/{dataset}/all_test_lvs"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print(docs)
    docs_list = docs['document'].dropna().unique()
    for dc in docs_list:
        subset = df[df['document'] == dc]
        # This now runs on a single Y-axis with scaled LvS values
        plot_final_analysis(subset, dc, 'LvS', os.path.join(out_dir, f"LvSt_{dc}.png"))