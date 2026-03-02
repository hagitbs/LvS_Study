'''
I need to write a python code to design a bar chart per key , that shows 'expected' versus 'observed' for each element . I want to read the data from a csv file ,

dataset='demo'

df = pd.read_csv = pd.read_csv(f"results/{dataset}/df_merged.csv") 
keys = df['key'].unique()
the relevant columns are 

key,element,expected,LvS,observed
the bar chart is wide in grey for expected col , and narrow blue for observed . in dual axes show the value of LvS , while you align the '0'  on the same Y .  the 'LvS' signal 

        color = '#22c55e' if lvs_val >= 0 else '#ef4444'
        marker = '^' if lvs_val >= 0 else 'v'
write all in Python  
''' 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

dataset = 'demo'
df = pd.read_csv(f"results/{dataset}/df_merged.csv")
keys = df['key'].unique()

output_dir = f"results/{dataset}/charts"
os.makedirs(output_dir, exist_ok=True)


def align_dual_axes(ax1, ax2):
    """Align zero on both y-axes."""
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    # Fraction of the range where zero sits
    frac1 = -y1_min / (y1_max - y1_min) if y1_max != y1_min else 0.5
    frac2 = -y2_min / (y2_max - y2_min) if y2_max != y2_min else 0.5

    # Use the larger fraction to ensure zero is within both ranges
    frac = max(frac1, frac2)
    frac = np.clip(frac, 0.1, 0.9)

    ax1.set_ylim(-frac * (y1_max - y1_min) / (1 - frac + frac),
                 (1 - frac) * (y1_max - y1_min) / (1 - frac + frac))

    # Recompute ranges to align zeros
    range1 = max(abs(y1_min), abs(y1_max)) / frac if frac > 0 else max(abs(y1_min), abs(y1_max))
    range2 = max(abs(y2_min), abs(y2_max)) / frac if frac > 0 else max(abs(y2_min), abs(y2_max))

    ax1.set_ylim(-range1 * frac, range1 * (1 - frac))
    ax2.set_ylim(-range2 * frac, range2 * (1 - frac))


for key in keys:
    df_key = df[df['key'] == key].copy().reset_index(drop=True)

    elements = df_key['element'].tolist()
    expected = df_key['expected'].tolist()
    observed = df_key['observed'].tolist()
    lvs = df_key['LvS'].tolist()

    n = len(elements)
    x = np.arange(n)

    bar_width_wide = 0.55   # expected (grey, wide)
    bar_width_narrow = 0.25  # observed (blue, narrow)

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.9), 6))

    # Wide grey bars: expected
    ax1.bar(x, expected, width=bar_width_wide, color='#d1d5db', label='Expected', zorder=2)

    # Narrow blue bars: observed
    ax1.bar(x, observed, width=bar_width_narrow, color='#3b82f6', label='Observed', zorder=3)

    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements, rotation=45, ha='right', fontsize=9)
    ax1.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # Secondary axis: LvS
    ax2 = ax1.twinx()
    ax2.set_ylabel('LvS', fontsize=11)
    ax2.plot(x, lvs, color='#6b7280', linewidth=1.2, linestyle='--', zorder=4, alpha=0.6)

    for i, lvs_val in enumerate(lvs):
        color = '#22c55e' if lvs_val >= 0 else '#ef4444'
        marker = '^' if lvs_val >= 0 else 'v'
        ax2.scatter(x[i], lvs_val, color=color, marker=marker, s=80, zorder=5)

    ax2.axhline(0, color='#9ca3af', linewidth=0.6, linestyle=':', zorder=1)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.2f}'))

    # Align zeros
    ax1.set_ylim(bottom=min(min(expected), min(observed), 0))
    ax2.set_ylim(bottom=min(min(lvs), 0))
    align_dual_axes(ax1, ax2)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lvs_handle = Line2D([0], [0], color='#6b7280', linestyle='--', linewidth=1.2, label='LvS')
    ax1.legend(handles=handles1 + [lvs_handle], loc='upper left', fontsize=9, framealpha=0.8)

    plt.title(f'Expected vs Observed — Key: {key}', fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()

    safe_key = str(key).replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"chart_{safe_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("Done.")
