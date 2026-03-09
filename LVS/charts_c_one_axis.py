import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

dataset = 'demo'
df = pd.read_csv(f"results/{dataset}/df_merged.csv")
keys = df['key'].unique()

output_dir = f"results/{dataset}/charts_c_one_axis"
os.makedirs(output_dir, exist_ok=True) 

for key in keys:
    df_key = df[(df['key'] == key) & (df['element'] != 'USA')].copy()
    df_key = df_key.sort_values('expected', ascending=False).reset_index(drop=True)

    elements = df_key['element'].tolist()
    expected = df_key['expected'].tolist()
    observed = df_key['observed'].tolist()
    lvs = df_key['LvS'].tolist()

    n = len(elements)
    x = np.arange(n) * 0.6  # compress spacing between elements

    bar_width_wide = 0.4    # expected (grey, wide)
    bar_width_narrow = 0.18  # observed (blue, narrow)

    fig, ax1 = plt.subplots(figsize=(max(6, n * 0.55), 6))

    # Wide grey bars: expected
    ax1.bar(x, expected, width=bar_width_wide, color='#d1d5db', label='Expected', zorder=2)

    # Narrow blue bars: observed
    ax1.bar(x, observed, width=bar_width_narrow, color='#3b82f6', label='Observed', zorder=3)

    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements, rotation=90, ha='center', fontsize=9)
    ax1.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # LvS markers only (no line)
    for i, lvs_val in enumerate(lvs):
        color = "#151916" if lvs_val >= 0 else '#ef4444'
        marker = '^' if lvs_val >= 0 else 'v'
        ax1.scatter(x[i], lvs_val, color=color, marker=marker, s=80, zorder=5)

    # Legend
    from matplotlib.lines import Line2D
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.8)

    plt.title(f'Expected vs Observed — Key: {key}', fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()

    safe_key = str(key).replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"chart_{safe_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("Done.")
