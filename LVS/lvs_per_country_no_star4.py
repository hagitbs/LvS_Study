import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_p_q_with_jsd_stars_from_column(
    df: pd.DataFrame,
    document: str,
    sort_by: str = "q",          # "q", "p", "element", "abs_diff", "jsd_nats"
    use_log_y: bool = False,
    output_path: str | None = None,
    jsd_fmt: str = "{:.2e}",     # formatting for label text
):
    required = {"document", "element", "p", "q", "jsd_nats"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df[df["document"] == document].copy()
    if d.empty:
        raise ValueError(f"No rows for document={document!r}")

    # Ensure numeric
    for col in ["p", "q", "jsd_nats"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    # Optional sort for readability (still shows ALL elements)
    d["abs_diff"] = (d["p"] - d["q"]).abs()
    if sort_by in ("p", "q", "jsd_nats"):
        d = d.sort_values(sort_by, ascending=False)
    elif sort_by == "abs_diff":
        d = d.sort_values("abs_diff", ascending=False)
    elif sort_by == "element":
        d = d.sort_values("element")
    else:
        raise ValueError("sort_by must be one of: 'q','p','element','abs_diff','jsd_nats'")

    x = np.arange(len(d))
    width = 0.45

    fig_w = max(14, min(80, 0.25 * len(d)))
    plt.figure(figsize=(fig_w, 6))

    plt.bar(x - width/2, d["q"].to_numpy(), width=width, label="Expected (q)")
    plt.bar(x + width/2, d["p"].to_numpy(), width=width, label="Observed (p)")

    plt.title(f"Expected (q) vs Observed (p) — {document}")
    plt.xlabel("Element (country)")
    plt.ylabel("Probability / normalized weight")
    plt.xticks(x, d["element"].astype(str), rotation=90, ha="center", fontsize=8)

    if use_log_y:
        plt.yscale("symlog", linthresh=1e-12)

    # ⭐ Star position: above the taller of the two bars per element
    y_top = np.maximum(d["p"].to_numpy(), d["q"].to_numpy())
    if use_log_y:
        y_star = y_top + np.maximum(1e-12, 0.15 * y_top)
    else:
        y_star = y_top + np.maximum(1e-12, 0.02 * (y_top.max() if y_top.max() > 0 else 1.0))

    plt.scatter(x, y_star, marker="*", s=70)

    # Print the value from your CSV's jsd_nats column above each star
    for xi, yi, jsd_val in zip(x, y_star, d["jsd_nats"].to_numpy()):
        plt.text(
            xi, yi,
            jsd_fmt.format(jsd_val),
            rotation=90, ha="center", va="bottom", fontsize=7
        )

    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


 
# plot_p_q_with_jsd_stars_from_column(df, "Acute_hepatitis", sort_by="q", use_log_y=True)

import os
df = pd.read_csv ("/Users/hagitbenshoshan/Documents/PHD/Market/Data/vis_cod2.csv" )   # <-- set this 
out_dir = "/Users/hagitbenshoshan/Documents/PHD/Market/cod_figs_2"
os.makedirs(out_dir, exist_ok=True)
plot_p_q_with_jsd_stars_from_column(df, "Acute_hepatitis", sort_by="q", use_log_y=True) # create folder "plots" first