import pandas as pd, numpy as np, matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

raw = """Element\tFreq\tExpected\tLvS
Electricity\t0.011596348\t0.104765751\t-0.030957628
Food\t0\t0.118241915\t-0.059120957
Insurance\t0\t0.151073334\t-0.075536667
Oil&Gas\t0.035529238\t0.212548344\t-0.050532726
Pharmaceuticals\t0\t0.092007699\t-0.046003849
Real Estate\t0.010115963\t0.041962579\t-0.007544452
Retail\t0\t0.102219889\t-0.051109944
Technology\t0.942758451\t0.177180488\t0.207189163
"""
df = pd.read_csv(StringIO(raw), sep="\t").sort_values("Expected", ascending=False).reset_index(drop=True)

x = np.arange(len(df), dtype=float)
ys = df["LvS"].to_numpy(dtype=float)

# Right axis limits
lvs_min = float(df["LvS"].min())
lvs_max = float(df["LvS"].max())
y2min = min(lvs_min * 1.25, -0.02)
y2max = max(lvs_max * 1.15, 0.02)

# Align zeros across axes
f2 = (-y2min) / (y2max - y2min)
y1max = float(max(df["Expected"].max(), df["Freq"].max()) * 1.18)
y1min = (f2 * y1max) / (f2 - 1)

fig, ax1 = plt.subplots(figsize=(11.6, 6.3))

# Bars (left axis)
ax1.bar(x, df["Expected"], width=0.75, color="grey", alpha=0.35, label="Expected")
ax1.bar(x, df["Freq"], width=0.45, color="tab:blue", alpha=0.95, label="Observed (Freq)")
ax1.set_ylabel("Share (Freq / Expected)")
ax1.set_ylim(y1min, y1max)

# Stars + sash (right axis)
ax2 = ax1.twinx()
ax2.set_ylabel("LvS")
ax2.set_ylim(y2min, y2max)

neg = ys < 0
pos = ~neg

# Plot stars by sign
ax2.scatter(
    x[neg], ys[neg],
    marker="*", s=180, color="red",
    edgecolor="black", linewidth=0.6, zorder=8,
    label="LvS (negative)"
)
ax2.scatter(
    x[pos], ys[pos],
    marker="*", s=180, color="green",
    edgecolor="black", linewidth=0.6, zorder=8,
    label="LvS (positive)"
)

# Sash curve (keep single curve; choose neutral dark line)
x_dense = np.linspace(x.min(), x.max(), 400)
try:
    from scipy.interpolate import make_interp_spline
    y_dense = make_interp_spline(x, ys, k=3)(x_dense)
except Exception:
    y_dense = np.interp(x_dense, x, ys)

ax2.plot(x_dense, y_dense, linestyle="--", linewidth=2.0, color="black", alpha=0.6, zorder=7, label="LvS sash")

# Shared zero line
ax1.axhline(0, linewidth=1)

# Annotations
for xi, lvs in zip(x, ys):
    dy = 0.015 if lvs >= 0 else -0.015
    va = "bottom" if lvs >= 0 else "top"
    ax2.text(xi, lvs + dy, f"{lvs:+.3f}", ha="center", va=va, fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(df["Element"], rotation=35, ha="right")
ax1.set_title("Dual Y-Axis (Zero-Aligned): Bars + LvS Stars (Red=Negative, Green=Positive) + Trend Line")

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right")

fig.tight_layout()

out_path = "dual_axis_zero_aligned_expected_freq_lvs.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

str(out_path), (y1min, y1max, y2min, y2max, f2)

