import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def symmetric_ylim(values, pad_ratio=0.08, min_range=1e-9):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)

    max_abs = float(np.max(np.abs(vals)))
    max_abs = max(max_abs, min_range)
    pad = max_abs * pad_ratio
    L = max_abs + pad
    return (-L, L)


def plot_key_bars(df_key: pd.DataFrame, key_value: str, out_dir: str):

    # 🔹 Remove USA
    df_key = df_key[df_key["element"] != "USA"].copy()

    # 🔹 Sort by expected descending
    df_key = df_key.sort_values(by="expected", ascending=False)

    x = np.arange(len(df_key))
    expected = df_key["expected"].astype(float).to_numpy()
    observed = df_key["observed"].astype(float).to_numpy()
    lvs = df_key["LvS"].astype(float).to_numpy()
    gap = df_key["gap_val"].astype(float).to_numpy()   # 🆕 NEW COLUMN

    fig, ax = plt.subplots(figsize=(18, 7))

    # Wide grey bars (expected)
    ax.bar(x, expected, width=0.80, color="#9ca3af", alpha=0.9,
           label="expected", zorder=1)

    # Narrow blue bars (observed)
    ax.bar(x, observed, width=0.35, color="#2563eb", alpha=0.95,
           label="observed", zorder=2)

    ax.set_title(f"Expected vs Observed + LvS + Gap — key = {key_value}")
    ax.set_xlabel("element")
    ax.set_ylabel("value")
    ax.axhline(0, linewidth=1.0, color="black", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(df_key["element"].tolist(), rotation=60, ha="right")

    # 🔵 Secondary axis
    ax2 = ax.twinx()
    ax2.set_ylabel("LvS / gap")

    # 🔺 LvS markers
    for i, lvs_val in enumerate(lvs):
        color = "#22c55e" if lvs_val >= 0 else "#ef4444"
        marker = "^" if lvs_val >= 0 else "v"

        ax2.scatter(
            x[i],
            lvs_val,
            s=110,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            zorder=4
        )

    # 🆕 ⚫ GAP circles
    ax2.scatter(
        x,
        gap,
        s=70,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        marker="o",
        label="gap_val",
        zorder=3
    )

    # Align zero on both axes
    ax.set_ylim(*symmetric_ylim(np.r_[expected, observed]))
    ax2.set_ylim(*symmetric_ylim(np.r_[lvs, gap]))

    # Legend
    bar_handles, bar_labels = ax.get_legend_handles_labels()

    lvs_pos = plt.Line2D([0], [0], marker="^", color="w",
                         markerfacecolor="#22c55e",
                         markeredgecolor="black",
                         markersize=9, label="LvS ≥ 0")

    lvs_neg = plt.Line2D([0], [0], marker="v", color="w",
                         markerfacecolor="#ef4444",
                         markeredgecolor="black",
                         markersize=9, label="LvS < 0")

    gap_proxy = plt.Line2D([0], [0], marker="o", color="black",
                           markerfacecolor="white",
                           markersize=8, label="gap_val")

    ax.legend(handles=bar_handles + [lvs_pos, lvs_neg, gap_proxy],
              loc="upper left")

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(key_value))
    out_path = os.path.join(out_dir, f"{safe_key}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    dataset = "demo"
    in_path = f"results/{dataset}/df_merged.csv"
    out_dir = f"results/{dataset}/charts_by_key_plus"

    df = pd.read_csv(in_path)

    required = {"key", "element", "expected", "LvS", "observed", "gap_val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    for col in ["expected", "observed", "LvS", "gap_val"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keys = df["key"].dropna().unique()

    for k in keys:
        df_key = df[df["key"] == k]
        plot_key_bars(df_key, k, out_dir)


if __name__ == "__main__":
    main()