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

def plot_country_panel(df_country: pd.DataFrame, country: str, out_path: str):
    # Sort order (you can switch to Observed or alphabetical)
    df_country = df_country.sort_values("Expected", ascending=False).reset_index(drop=True)

    x = np.arange(len(df_country), dtype=float)
    ys = df_country["LvS"].to_numpy(dtype=float)

    # Right axis limits (LvS)
    lvs_min = float(df_country["LvS"].min())
    lvs_max = float(df_country["LvS"].max())
    y2min = min(lvs_min * 1.25, -0.02)
    y2max = max(lvs_max * 1.15, 0.02)

    # Align zeros across axes
    f2 = (-y2min) / (y2max - y2min)
    y1max = float(max(df_country["Expected"].max(), df_country["Observed"].max()) * 1.18)
    y1min = (f2 * y1max) / (f2 - 1)

    fig, ax1 = plt.subplots(figsize=(11.6, 6.3))

    # Bars (left axis)
    ax1.bar(x, df_country["Expected"], width=0.55, color="grey", alpha=0.35, label="Expected")
    ax1.bar(x, df_country["Observed"], width=0.35, color="tab:blue", alpha=0.95, label="Observed")
    ax1.set_ylabel("Share (Observed / Expected)")
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
    ax1.set_xticklabels(df_country["Element"], rotation=35, ha="right")

    ax1.set_title(
        f"Country: {country} — Expected (grey) vs Observed (blue) + LvS (stars)"
    )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================
# MAIN: read + pivot + plot
# =========================
csv_path = "/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis2.csv"   # <-- set this
out_dir = "/Users/hagitbenshoshan/Documents/PHD/Market/Industry_figs"
os.makedirs(out_dir, exist_ok=True)

df_long = pd.read_csv(csv_path) 
print (df_long) 

# Normalize column names (handles minor spacing/casing differences)
df_long = df_long.rename(columns={
    "Measure Names": "Measure",
    "Measure Values": "Value"
})

required = {"Country", "Element", "Measure", "Value"}
missing = required - set(df_long.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(df_long.columns)}")

df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long = df_long.dropna(subset=["Country", "Element", "Measure", "Value"])

# Pivot to wide: Observed / Expected / LvS columns
df_wide = (df_long
           .pivot_table(index=["Country", "Element"], columns="Measure", values="Value", aggfunc="mean")
           .reset_index())

# If your measures are named Freq instead of Observed, rename here:
rename_map = {"Freq": "Observed", "Observed": "Observed", "Expected": "Expected", "LvS": "LvS"}
df_wide = df_wide.rename(columns={k: v for k, v in rename_map.items() if k in df_wide.columns})

needed = {"Observed", "Expected", "LvS"}
missing_measures = needed - set(df_wide.columns)
if missing_measures:
    raise ValueError(f"After pivot, missing measures: {missing_measures}. Columns now: {list(df_wide.columns)}")

df_wide = df_wide.dropna(subset=["Observed", "Expected", "LvS"])
 
print (df_wide)
# One figure per country
for country, sub in df_wide.groupby("Country", sort=True):
    out_path = os.path.join(out_dir, f"{safe_name(country)}_expected_observed_lvs.png")
    plot_country_panel(sub, country, out_path)

print("Saved figures to:", out_dir)