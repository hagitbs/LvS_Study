import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_document(df_doc, doc_id, secondary_col, filename):
    # Sort and clean data
    df_doc = df_doc.sort_values("expected", ascending=False).reset_index(drop=True)
    x = np.arange(len(df_doc), dtype=float)
    ys = df_doc["LvS"].to_numpy(dtype=float)

    # 1. Define Y2 (LvS) limits to ensure symmetry or sufficient padding for labels
    lvs_min = float(ys.min())
    lvs_max = float(ys.max())
    # Ensure we always show some range even if data is flat
    y2min = min(lvs_min * 1.3, -0.1) 
    y2max = max(lvs_max * 1.3, 0.1)

    # 2. Define Y1 (Bars) limits
    # We keep Y1 mostly positive but add a small buffer at the bottom 
    # so the zero line isn't cut off.
    y1max = float(max(df_doc["expected"].max(), df_doc["observed"].max()) * 1.2)
    y1min = y1max * (y2min / y2max) # This mathematically aligns the zeros

    fig, ax1 = plt.subplots(figsize=(11.6, 6.3))

    # --- PRIMARY AXIS (Bars) ---
    ax1.bar(x, df_doc["expected"], width=0.75, color="grey", alpha=0.3, label="Expected")
    ax1.bar(x, df_doc["observed"], width=0.45, color="tab:blue", alpha=0.8, label="Observed")
    ax1.set_ylabel("Share (observed / expected)")
    ax1.set_ylim(y1min, y1max)
    
    # Optional: Hide negative tick labels for the bar chart since bars aren't negative
    ticks = ax1.get_yticks()
    ax1.set_yticks([t for t in ticks if t >= 0])

    # --- SECONDARY AXIS (LvS Stars) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("LvS Value")
    ax2.set_ylim(y2min, y2max)

    # Plot stars with conditional coloring
    neg = ys < 0
    ax2.scatter(x[~neg], ys[~neg], marker="*", s=200, color="green", 
                edgecolor="black", zorder=10, label="LvS (Positive)")
    ax2.scatter(x[neg], ys[neg], marker="*", s=200, color="red", 
                edgecolor="black", zorder=10, label="LvS (Negative)")

    # Curved Dash line
    x_dense = np.linspace(x.min(), x.max(), 400)
    try:
        from scipy.interpolate import make_interp_spline
        y_dense = make_interp_spline(x, ys, k=3)(x_dense)
        ax2.plot(x_dense, y_dense, linestyle="--", linewidth=1.5, color="black", alpha=0.4, zorder=5)
    except:
        ax2.plot(x, ys, linestyle="--", color="black", alpha=0.4)

    # --- REFINED ZERO LINE ---
    ax1.axhline(0, color='black', linewidth=1.2, zorder=3)

    # Annotations with dynamic offsetting
    for xi, lvs in zip(x, ys):
        offset = (y2max - y2min) * 0.03
        va = "bottom" if lvs >= 0 else "top"
        ax2.text(xi, lvs + (offset if lvs >= 0 else -offset), 
                 f"{lvs:+.3f}", ha="center", va=va, 
                 fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_doc["element"], rotation=35, ha="right")
    ax1.set_title(f"Analysis for: {doc_id}", pad=20, fontsize=14)
    
    # Unified Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=4, frameon=False)

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)