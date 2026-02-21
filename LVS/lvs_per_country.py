import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s if s else "Unknown Country"

def   plot_final_analysis(df_doc, doc_id, secondary_col, filename):


    # Sort order (you can switch to Observed or alphabetical)
    df_doc = df_doc.sort_values("expected", ascending=False).reset_index(drop=True)

    x = np.arange(len(df_doc), dtype=float)
    ys = df_doc["LvS"].to_numpy(dtype=float)

    # Right axis limits (LvS)
    lvs_min = float(df_doc["LvS"].min())
    lvs_max = float(df_doc["LvS"].max())
    y2min = min(lvs_min * 1.25, -0.02)
    y2max = max(lvs_max * 1.15, 0.02)

    # Align zeros across axes
    f2 = (-y2min) / (y2max - y2min)
    y1max = float(max(df_doc["expected"].max(), df_doc["observed"].max()) * 1.18)
    y1min = (f2 * y1max) / (f2 - 1)

    fig, ax1 = plt.subplots(figsize=(11.6, 6.3))

    # Bars (left axis)
    ax1.bar(x, df_doc["expected"], width=0.75, color="grey", alpha=0.35, label="Expected")
    ax1.bar(x, df_doc["observed"], width=0.45, color="tab:blue", alpha=0.95, label="Observed")
    ax1.set_ylabel("Share (observed / expected)")
    ax1.set_ylim(y1min, y1max)

    # Stars + sash (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("LvS")
    ax2.set_ylim(y2min, y2max)

    neg = ys < 0
    pos = ~neg

    ax2.scatter(x[neg], ys[neg], marker="*", s=180, color="red",
                edgecolor="black", linewidth=0.6, zorder=8, label="LvS (negative)")
    ax2.scatter(x[pos], ys[pos], marker="*", s=180, color="green",
                edgecolor="black", linewidth=0.6, zorder=8, label="LvS (positive)")

    # Curved Dash line connecting stars
    x_dense = np.linspace(x.min(), x.max(), 400)
    try:
        from scipy.interpolate import make_interp_spline
        y_dense = make_interp_spline(x, ys, k=3)(x_dense)
    except Exception:
        y_dense = np.interp(x_dense, x, ys)

    ax2.plot(x_dense, y_dense, linestyle="--", linewidth=1.0,
             color="grey", alpha=0.6, zorder=7)

    # Shared zero line
    ax1.axhline(0, linewidth=1)

    # LvS value annotations
    for xi, lvs in zip(x, ys):
        dy = 0.015 if lvs >= 0 else -0.015
        va = "bottom" if lvs >= 0 else "top"
        ax2.text(xi, lvs + dy, f"{lvs:+.3f}", ha="center", va=va, fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_doc["element"], rotation=35, ha="right")

    ax1.set_title(
        f"Country: {doc_id} — Expected (grey) vs Observed (blue) + LvS (stars)"
    )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
 

# --- EXECUTION ---
 
def plot_document (df,dataset,docs):

    out_dir = f"results/{dataset}/country_lvs"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print(docs)
    docs_list = docs['document'].dropna().unique()
    for dc in docs_list:
        subset = df[df['document'] == dc]
        # This now runs on a single Y-axis with scaled LvS values
        plot_final_analysis(subset, dc, 'LvS', os.path.join(out_dir, f"LvSc_{dc}.png"))