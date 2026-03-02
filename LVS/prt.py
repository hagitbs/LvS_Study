import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import StringIO

csv_data = """key,element_observed,LvS,element_expected,expected,document,element,frequency_in_document,doc_total,observed,gap_val
Electricity,Saudi Arabia,-0.019078288928215392,Saudi Arabia,0.038156577856430784,Electricity,Saudi Arabia,0.0,2226610000000.0,0.0,-0.038156577856430784
Food,Saudi Arabia,-0.008446145928102673,Saudi Arabia,0.038156577856430784,Food,Saudi Arabia,15700000000.0,2211748080000.0,0.007098457614576069,-0.031058120241854715
Insurance,Saudi Arabia,-0.019078288928215392,Saudi Arabia,0.038156577856430784,Insurance,Saudi Arabia,0.0,3510225350000.0,0.0,-0.038156577856430784
Oil&Gas,Saudi Arabia,0.08183905110626377,Saudi Arabia,0.038156577856430784,Oil&Gas,Saudi Arabia,1963890000000.0,6614459610000.0,0.2969086086837561,0.25875203082732534
Pharmaceuticals,Saudi Arabia,-0.019078288928215392,Saudi Arabia,0.038156577856430784,Pharmaceuticals,Saudi Arabia,0.0,5393512270000.0,0.0,-0.038156577856430784
Real Estate,Saudi Arabia,-0.019078288928215392,Saudi Arabia,0.038156577856430784,Real Estate,Saudi Arabia,0.0,1782132340000.0,0.0,-0.038156577856430784
Retail,Saudi Arabia,-0.01571338088195676,Saudi Arabia,0.038156577856430784,Retail,Saudi Arabia,6750000000.0,5419264170000.0,0.0012455565531141104,-0.03691102130331667
Technology,Saudi Arabia,-0.019078288928215392,Saudi Arabia,0.038156577856430784,Technology,Saudi Arabia,0.0,23817025780000.0,0.0,-0.038156577856430784
Electricity,Switzerland,-0.012644495358061045,Switzerland,0.039699023007236,Electricity,Switzerland,8120000000.0,2226610000000.0,0.0036467993945953716,-0.036052223612640634
Food,Switzerland,0.025098170995476312,Switzerland,0.039699023007236,Food,Switzerland,334740000000.0,2211748080000.0,0.15134635043969383,0.11164732743245782
Insurance,Switzerland,0.0026522064739568883,Switzerland,0.039699023007236,Insurance,Switzerland,237400000000.0,3510225350000.0,0.06763098557190922,0.027931962564673216
Oil&Gas,Switzerland,-0.017495625861867575,Switzerland,0.039699023007236,Oil&Gas,Switzerland,5060000000.0,6614459610000.0,0.0007649906868204461,-0.038934032320415556
Pharmaceuticals,Switzerland,0.005989312077389994,Switzerland,0.039699023007236,Pharmaceuticals,Switzerland,456410000000.0,5393512270000.0,0.08462203795079157,0.044923014943555564
Real Estate,Switzerland,-0.012193465335005989,Switzerland,0.039699023007236,Real Estate,Switzerland,7140000000.0,1782132340000.0,0.004006436469246723,-0.03569258653798928
Retail,Switzerland,-0.019849511503618,Switzerland,0.039699023007236,Retail,Switzerland,0.0,5419264170000.0,0.0,-0.039699023007236
Technology,Switzerland,-0.010451541031389313,Switzerland,0.039699023007236,Technology,Switzerland,132770000000.0,23817025780000.0,0.005574583544830844,-0.03412443946240516
Electricity,Spain,0.010205009167501847,Spain,0.011435311210851649,Electricity,Spain,116260000000.0,2226610000000.0,0.05221390364724851,0.040778592436396865
Food,Spain,-0.0026023038264620436,Spain,0.011435311210851649,Food,Spain,4520000000.0,2211748080000.0,0.0020436323833047027,-0.009391678827546946"""

df = pd.read_csv(StringIO(csv_data))
keys = df['key'].unique()

# Layout: grid of subplots
n_keys = len(keys)
ncols = 3
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

    # --- Zero line on ax2 ---
    ax2.axhline(0, color='#444', linewidth=0.7, linestyle='--', zorder=1)

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
plt.savefig('chart.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f0f')
print("Saved.")
plt.show()
#