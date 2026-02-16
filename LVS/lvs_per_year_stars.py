'''
df = pd.read_csv('/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis_cod1.csv')   # <-- set this 
df_1994 = df[df['document'] == 1994]

# Version 1: LvS
plot_1994_constrained(df_1994, 'LvS', 'doc_1994_lvs_final.png')

# Version 2: gap_val
#plot_1994_constrained(df_1994, 'gap_val', 'doc_1994_gap_final.png')

 for doc in documents:
    sub = df[df['document'] == doc]
    plot_final_analysis(sub, doc, 'LvS', out_dir) 

print(f"Generated {len(documents)} plots in {out_dir}")
# Load data
df = pd.read_csv('/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis_cod101.csv')
out_dir = "/Users/hagitbenshoshan/Documents/PHD/Market/all_years_lvs"
'''
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def safe_name(s: str) -> str:
    """Cleans strings for filenames."""
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9_\-\.]", "", s)

def plot_final_analysis(df_doc, doc_id, secondary_col, filename):
    # 1. Pre-process Data: Sort by expected and shorten labels
    #df_doc = df_doc.sort_values("expected", ascending=False).reset_index(drop=True)
    df_doc = df_doc.sort_values("element", ascending=True).reset_index(drop=True)    
    labels = [str(e)[:8] for e in df_doc["element"]]
    x = np.arange(len(df_doc))
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 2. PRIMARY AXIS (Proportions): Overlapping Bars
    # Expected bar (Wider, Grey)
    ax1.bar(x, df_doc["expected"], width=0.7, color='#D3D3D3',  
            label='Expected', alpha=0.8, zorder=2)
    # Observed bar (Inner, Blue)
    ax1.bar(x, df_doc["observed"], width=0.3, color="#8bacc3",  
            label='Observed', alpha=1.0, zorder=3)
    
    # 3. SECONDARY AXIS (LvS/Gap): Arrows and Trend Line
    ax2 = ax1.twinx()
    sec_values = df_doc[secondary_col].values
    
    # Black dotted line connecting the markers
    if len(x) > 1:
        ax2.plot(x, sec_values, color='black', linestyle=':', linewidth=0.01, zorder=4, alpha=0.8)
    
    # Arrow markers: Blue Up for positive, Red Down for negative
    for i, val in enumerate(sec_values):
        color = 'green' if val >= 0 else 'red'
        marker = '^' if val >= 0 else 'v'
        ax2.scatter(x[i], val, marker=marker, color=color, s=60, alpha=1.0,
                    edgecolors='black', linewidths=0.5 , zorder=5)

    # 4. AXIS CONSTRAINTS: Start right axis at -0.001 and align zeros
    y2_min = -0.0005
    y2_max = max(sec_values.max(), 0.001) * 1.2
    
    y1_max = max(df_doc["expected"].max(), df_doc["observed"].max()) * 1.2
    # Calculate y1_min to force the zero lines of both axes to overlap perfectly
    y1_min = (y2_min * y1_max) / y2_max
    
    ax1.set_ylim(y1_min, y1_max)
    ax2.set_ylim(y2_min, y2_max)
    
    # 5. FORMATTING & AESTHETICS
    ax1.set_ylabel("Proportion (Share)", fontsize=12, fontweight='bold')
    ax2.set_ylabel(f"{secondary_col} (Labels Hidden)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Country Analysis: {secondary_col} Version", fontsize=16, fontweight='bold', pad=25)
    
    # Rotate shortened x-labels 90 degrees
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
    
    # Hide the tick labels on the right axis
    ax2.set_yticklabels([])
    
    # Draw the unified zero baseline
    ax1.axhline(0, color='black', linewidth=0.001, zorder=1)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='#D3D3D3', lw=6, label='Expected Share'),
        Line2D([0], [0], color='#1f77b4', lw=4, label='Observed Share'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label=f'{secondary_col} (+)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label=f'{secondary_col} (-)'),
        Line2D([0], [0], color='white', linestyle=':', lw=0.001, label='')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9) 
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)

# --- EXECUTION ---
# Load your dataset
out_dir = "/Users/hagitbenshoshan/Documents/PHD/Market/all_years_lvs"
df = pd.read_csv('/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis_cod101.csv')

# Filter for the specific year 1994
# df_1994 = df[df['document'] == 1994]

# Generate both versions

#plot_final_analysis(df_1994, 1994, 'gap_val', 'doc_1994_gap_final.png')
# Run Loop
years = sorted(df['document'].unique())
for yr in years:
    subset = df[df['document'] == yr]
    plot_final_analysis(subset, yr, 'LvS', os.path.join(out_dir, f"LvS_{str(yr)}.png"))

print(f"Generated {len(years)} plots in '{out_dir}' with reduced secondary axis proportion.")