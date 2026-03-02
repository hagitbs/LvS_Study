import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import StringIO
import os
import configparser
import pandas as pd
import altair as alt
from LPA import Corpus
import lvs_per_country
import lvs_per_document

dataset='demo'

df = pd.read_csv = pd.read_csv(f"results/{dataset}/df_merged.csv") 
keys = df['key'].unique()

# Layout: grid of subplots
n_keys = len(keys)
ncols = 5
nrows = int(np.ceil(n_keys / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle('Expected vs Observed  |  LvS Signal', 
             color='#e0e0e0', fontsize=15, fontweight='bold',
             fontfamily='monospace', y=1.01)

axes_flat = axes.flatten() if nrows > 1 else axes.flatten()

BAR_WIDTH = 0.5
BLUE = '#3b8bff'
GREY = '#888888'

for idx, key in enumerate(keys):
    ax = axes_flat[idx]
    ax2 = ax.twinx()

    sub = df[df['key'] == key].reset_index(drop=True)
    x = np.arange(len(sub))
    labels = sub['element_observed'].tolist()

    # --- Background ---
    ax.set_facecolor('#141414')
    ax2.set_facecolor('#141414')

    # --- Wide grey bar (expected) ---
    ax.bar(x, sub['expected'], width=BAR_WIDTH, color=GREY,
           alpha=0.55, zorder=2, label='Expected')

    # --- Thin blue bar (observed) on top / overlapping ---
    ax.bar(x, sub['observed'], width=BAR_WIDTH * 0.45, color=BLUE,
           alpha=0.9, zorder=3, label='Observed')

    # --- LvS markers on secondary y-axis ---
    for i, (lvs_val) in enumerate(sub['LvS']):
        color = '#22c55e' if lvs_val >= 0 else '#ef4444'
        marker = '^' if lvs_val >= 0 else 'v'
        ax2.scatter(i, lvs_val, marker=marker, color=color,
                    s=90, zorder=5, linewidths=0.5,
                    edgecolors='white' if lvs_val >= 0 else '#ff8888')
        # Label: above marker if positive, below if negative
        va = 'bottom' if lvs_val >= 0 else 'top'
        offset = 0.003 if lvs_val >= 0 else -0.003
        ax2.text(i, lvs_val + offset, f'{lvs_val:.3f}',
                 ha='center', va=va, fontsize=6.5,
                 color=color, fontfamily='monospace', zorder=6)

    # ---------------------------------------------------------------
    # Align zeros on both axes using a shared zero-fraction strategy
    # ---------------------------------------------------------------
    # Step 1: determine ranges from data
    left_max = max(sub['expected'].max(), sub['observed'].max()) * 1.15 + 1e-9
    lvs_vals  = sub['LvS'].values
    lvs_neg   = abs(min(lvs_vals.min(), 0)) * 1.20 + 1e-9   # magnitude below 0
    lvs_pos   =     max(lvs_vals.max(), 0)  * 1.20 + 1e-9   # magnitude above 0

    # Step 2: pick a common zero-fraction that fits both axes
    # zero_frac = neg_range / total_range
    # For left axis bars (all ≥ 0): we WANT zero_frac so the chart still
    # looks good. Use the fraction dictated by the LvS range:
    total_lvs  = lvs_neg + lvs_pos
    zero_frac  = lvs_neg / total_lvs          # fraction of height below zero

    # Step 3: derive left axis limits so zero sits at zero_frac
    # zero_frac = (0 - left_min) / (left_max - left_min)
    # → left_min = -zero_frac * left_max / (1 - zero_frac)
    if zero_frac < 1:
        left_min = -zero_frac / (1 - zero_frac) * left_max
    else:
        left_min = -left_max   # fallback

    ax.set_ylim(left_min, left_max)

    # Step 4: derive right axis limits so zero sits at the SAME zero_frac
    # zero_frac = lvs_neg / total_lvs  (already guaranteed by construction)
    ax2.set_ylim(-lvs_neg, lvs_pos)

    # --- Zero line on ax2 (and ax) ---
    ax2.axhline(0, color='#666', linewidth=1.0, linestyle='--', zorder=1)
    ax.axhline(0,  color='#666', linewidth=1.0, linestyle='--', zorder=1)

    # --- Styling ax (left, bars) ---
    ax.set_title(key, color='#e0e0e0', fontsize=11,
                 fontfamily='monospace', fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right',
                       fontsize=8, color='#aaaaaa', fontfamily='monospace')
    ax.tick_params(axis='y', colors='#888888', labelsize=8)
    ax.yaxis.label.set_color('#888')
    ax.set_ylabel('Share', color='#888', fontsize=8, fontfamily='monospace')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#2a2a2a', linewidth=0.5, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))

    # --- Styling ax2 (right, LvS) ---
    ax2.tick_params(axis='y', colors='#888888', labelsize=8)
    ax2.set_ylabel('LvS', color='#888', fontsize=8, fontfamily='monospace')
    ax2.spines['bottom'].set_color('#333')
    ax2.spines['right'].set_color('#333')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

# Hide unused subplots
for idx in range(n_keys, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# --- Global legend ---
legend_elements = [
    mpatches.Patch(color=GREY, alpha=0.6, label='Expected'),
    mpatches.Patch(color=BLUE, alpha=0.9, label='Observed'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#22c55e',
               markersize=9, label='LvS positive', linewidth=0),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#ef4444',
               markersize=9, label='LvS negative', linewidth=0),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           facecolor='#1a1a1a', edgecolor='#333',
           labelcolor='#cccccc', fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('results/{dataset}/chart.png', dpi=150, 
            bbox_inches='tight', facecolor='#0f0f0f')
print("Saved.")
