import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------
def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))


def clip_scale_sizes(values, s_min=30, s_max=220, q_low=0.05, q_high=0.95):
    """
    Map |values| to marker sizes with robust clipping (quantile-based).
    Returns array of sizes in [s_min, s_max].
    """
    v = np.asarray(values, dtype=float)
    a = np.abs(v)
    a = np.where(np.isfinite(a), a, 0.0)

    lo = float(np.quantile(a, q_low)) if a.size else 0.0
    hi = float(np.quantile(a, q_high)) if a.size else 1.0
    if hi <= lo:
        return np.full_like(a, (s_min + s_max) / 2.0, dtype=float)

    a_clip = np.clip(a, lo, hi)
    t = (a_clip - lo) / (hi - lo)
    return s_min + t * (s_max - s_min)


def lvs_color_marker(lvs_val: float):
    if lvs_val >= 0:
        return "#22c55e", "^"  # green, up
    return "#ef4444", "v"     # red, down


# ---------- plotting ----------
def plot_lvs_dumbbell_for_key(
    df_key: pd.DataFrame,
    key_value: str,
    out_dir: str,
    top_k: int = 25,                 # show top-K by expected (rest excluded)
    drop_element: str = "USA",        # remove USA
    annotate_k_absences: int = 3,     # annotate top negative LvS
    figsize=(12, 9),
):
    """
    Recommended LvS visualization:
    - Horizontal layout (readable labels)
    - Top-K by expected
    - Dumbbell (expected -> observed) connector encodes gap
    - LvS encoded as triangle at observed position (color+direction by sign, size by |LvS|)
    - Optional annotation of strongest absences (most negative LvS)
    """
    required = {"element", "expected", "observed", "LvS"}
    missing = required - set(df_key.columns)
    if missing:
        raise ValueError(f"Missing columns for key={key_value}: {sorted(missing)}")

    d = df_key.copy()
    d = d[d["element"].astype(str) != drop_element].copy()

    # numeric coercion
    for col in ["expected", "observed", "LvS"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["element", "expected", "observed", "LvS"]).copy()
    if d.empty:
        return None

    # Sort by expected DESC and take Top-K
    d["element"] = d["element"].astype(str)
    d = d.sort_values("expected", ascending=False, kind="mergesort")
    if top_k is not None and top_k > 0 and len(d) > top_k:
        d = d.head(top_k)

    # Reverse for horizontal plot: largest at top
    d = d.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(d))
    expected = d["expected"].to_numpy(float)
    observed = d["observed"].to_numpy(float)
    lvs = d["LvS"].to_numpy(float)
    #gap = d["gap_val"].to_numpy(float)
    labels = d["element"].tolist()

    # Marker sizes from |LvS|
    lvs_sizes = clip_scale_sizes(lvs, s_min=40, s_max=240)

    # X-limits: robust padding
    all_x = np.r_[expected, observed]
    xmin = float(np.nanmin(all_x))
    xmax = float(np.nanmax(all_x))
    pad = (xmax - xmin) * 0.08 if xmax > xmin else 0.5
    xlim = (xmin - pad, xmax + pad)

    fig, ax = plt.subplots(figsize=figsize)

    # ---- style defaults (publication-friendly) ----
    expected_color = "#d1d5db"  # light gray
    observed_color = "#2563eb"  # blue
    connector_color = "#6b7280" # mid gray

    # ---- dumbbell connectors (expected -> observed) ----
    for yi, x1, x2 in zip(y, expected, observed):
        ax.plot([x1, x2], [yi, yi], linewidth=2.0, color=connector_color, alpha=0.55, zorder=1)

    # endpoints (dots)
    ax.scatter(expected, y, s=55, color=expected_color, edgecolors="none", zorder=2, label="expected")
    ax.scatter(observed, y, s=55, color=observed_color, edgecolors="none", zorder=3, label="observed")

    # ---- LvS markers at observed positions ----
    # (triangles encode sign; size encodes magnitude)
    for yi, x_obs, lvs_val, s in zip(y, observed, lvs, lvs_sizes):
        c, m = lvs_color_marker(lvs_val)
        ax.scatter(
            x_obs, yi,
            s=s,
            marker=m,
            c=c,
            edgecolors="black",
            linewidths=0.6,
            zorder=4
        )

    # ---- optional gap circles at midpoint (subtle, neutral) ----
    # If you still want to show gap_val explicitly as circles, put them at midpoint to avoid clutter.
    # Circle size encodes |gap_val| (optional, mild).
    #gap_sizes = clip_scale_sizes(gap, s_min=20, s_max=90)
    #mid = (expected + observed) / 2.0
    '''ax.scatter(
        mid, y,
        s=gap_sizes,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        alpha=0.9,
        zorder=2.5,
        label="gap_val (midpoint)"
    )
    '''
    # ---- annotate strongest absences (most negative LvS) ----
    if annotate_k_absences and annotate_k_absences > 0:
        neg_idx = np.where(np.isfinite(lvs) & (lvs < 0))[0]
        if neg_idx.size:
            # most negative first
            top_neg = neg_idx[np.argsort(lvs[neg_idx])][:annotate_k_absences]
            for idx in top_neg:
                ax.text(
                    observed[idx] + (xlim[1] - xlim[0]) * 0.01,
                    y[idx],
                    f"LvS={lvs[idx]:.3g}",
                    va="center",
                    ha="left",
                    fontsize=10
                )

    # ---- axes, grid, labels ----
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_xlim(*xlim)
    ax.set_xlabel("Value (expected / observed)", fontsize=12)
    ax.set_title(f"Expected vs Observed (dumbbell) + LvS — Key: {key_value}", fontsize=14)

    # subtle grid
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    # tidy spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend with LvS explanation proxies
    lvs_pos = plt.Line2D([0], [0], marker="^", color="w",
                         markerfacecolor="#22c55e", markeredgecolor="black",
                         markersize=9, label="LvS ≥ 0")
    lvs_neg = plt.Line2D([0], [0], marker="v", color="w",
                         markerfacecolor="#ef4444", markeredgecolor="black",
                         markersize=9, label="LvS < 0")
    '''gap_proxy = plt.Line2D([0], [0], marker="o", color="black",
                           markerfacecolor="white", markersize=7,
                           label="gap_val (midpoint)")
    '''

    handles, labels0 = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [lvs_pos, lvs_neg], #, gap_proxy],
              loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_filename(key_value)}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    dataset = "demo"
    in_path = f"results/{dataset}/df_merged.csv"
    out_dir = f"results/{dataset}/charts_by_key_dumbbell"

    df = pd.read_csv(in_path)

    # Ensure required columns exist
    required = {"key", "element", "expected", "observed", "LvS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Generate one chart per key
    keys = df["key"].dropna().unique()
    print(f"Keys: {len(keys)} -> output: {out_dir}")

    for k in keys:
        df_key = df[df["key"] == k]
        out = plot_lvs_dumbbell_for_key(
            df_key=df_key,
            key_value=k,
            out_dir=out_dir,
            top_k=25,                 # adjust (15–30 recommended)
            drop_element="USA",
            annotate_k_absences=3,
            figsize=(12, 9),
        )
        if out:
            print("Saved:", out)


if __name__ == "__main__":
    main()