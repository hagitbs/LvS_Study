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
    return s if s else "UnknownCountry"

def plot_country_panel(df_country: pd.DataFrame, document: str, out_path: str , entity:str) :
    # Sort order (you can switch to Observed or alphabetical)
    df_country = df_country.sort_values("expected", ascending=False).reset_index(drop=True)
    print(df_country.head(4) )
    print( df_country.columns)
    #df_country= df_country[df_country['element_observed']] != 'USA' # condition 
    #df_country = df_country.drop(columns=['USA'])
    df_country=df_country[df_country['element_observed']!='USA'] 
    x = np.arange(len(df_country), dtype=float)
    ys = df_country["LvS"].to_numpy(dtype=float)

 
 

    fig, ax1 = plt.subplots(figsize=(11.6, 6.3))

    # Bars (left axis)
    ax1.bar(x, df_country["expected"], width=0.55, color="grey", alpha=0.35, label="Expected")
    ax1.bar(x, df_country["observed"], width=0.35, color="tab:blue", alpha=0.95, label="Observed")
    ax1.set_ylabel("Share (Observed / Expected)")
    ax1.set_ylim(0, 0.15)
    '''
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
    ''' 
    # Shared zero line
    ax1.axhline(0, linewidth=0.1)

    
    # LvS value annotations
    for xi, lvs in zip(x, ys):
        dy = 0.015 if lvs >= 0 else -0.015
        va = "bottom" if lvs >= 0 else "top"
        #ax2.text(xi, lvs + dy, f"{lvs:+.3f}", ha="center", va=va, fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_country["element"], rotation=90, ha="center", fontsize=7)

    ax1.set_title(
        f"country: {entity} — Expected (grey) vs Observed (blue)"
    )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1  , l1 , frameon=False, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================
# MAIN: read + pivot + plot
# =========================
dataset='demo'
csv_path = f"results/{dataset}/df_merged.csv"   # <-- set this
out_dir = f"results/{dataset}/industry_figs_no_star3"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path) 
print (df)  
#rename

  
# One figure per country
for document, sub in df.groupby("document", sort=True):
    out_path = os.path.join(out_dir, f"{safe_name(document)}_expected_observed_lvs.png")
    entity=safe_name(document)
    plot_country_panel(sub, df, out_path, entity)

print("Saved figures to:", out_dir)