from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 1e-2


def load(results_dir=Path("results")):
    frames = []
    for path in results_dir.glob("ablation_threshold_*.csv"):
        try: threshold = float(path.stem.removeprefix("ablation_threshold_"))
        except ValueError: continue
        df = pd.read_csv(path)
        df["threshold"] = threshold
        frames.append(df)
    return pd.concat(frames).sort_values(["threshold", "train_num", "seed"])


def summarize(df):
    rows = []
    for t, g in df.groupby("threshold"):
        err = g["acc"] - g["pred_acc"]
        rows.append({
            "threshold": t,
            "r": np.corrcoef(g["pred_acc"], g["acc"])[0, 1],
            "mae": err.abs().mean(),
            "frac_acc_above": (err > 1e-9).mean(),  # Acc > Acc-hat: Acc-hat too pessimistic
            "frac_acc_below": (err < -1e-9).mean(),  # Acc < Acc-hat: Acc-hat too optimistic
            "mean_rqi": g["rqi"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    df = load()
    # threshold_P only enters RQI / Acc-hat, not training, so Acc must not change with it
    acc_spread = df.groupby(["train_num", "seed"])["acc"].agg(np.ptp).max()
    print(f"max spread of Acc across thresholds (should be 0): {acc_spread:.2e}")

    summary = summarize(df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    thresholds = sorted(df["threshold"].unique())
    fig = plt.figure(figsize=(13, 6.2))
    grid = fig.add_gridspec(2, len(thresholds), height_ratios=[1, 1.15])

    for i, t in enumerate(thresholds):
        ax = fig.add_subplot(grid[0, i])
        g = df[df["threshold"] == t]
        ax.scatter(g["pred_acc"], g["acc"], s=10, color="blue", alpha=0.25, edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", color="red", lw=1)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal", title=f"threshold_P = {t:g}" + (" (default)" if t == DEFAULT_THRESHOLD else ""))
        ax.title.set_fontsize(9)
        ax.set_xlabel(r"$\widehat{\rm Acc}$")
        if i == 0: ax.set_ylabel("Acc")
        else: ax.set_yticklabels([])

    ax = fig.add_subplot(grid[1, : len(thresholds) // 2])
    ax.plot(summary["threshold"], summary["mae"], "o-", color="black", label="MAE |Acc − Acc-hat|")
    ax.plot(summary["threshold"], summary["frac_acc_above"], "s--", color="tab:blue", label=r"fraction Acc > $\widehat{\rm Acc}$")
    ax.plot(summary["threshold"], summary["frac_acc_below"], "^--", color="tab:orange", label=r"fraction Acc < $\widehat{\rm Acc}$")
    ax.axvline(DEFAULT_THRESHOLD, color="grey", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="threshold_P", ylim=(0, None))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, len(thresholds) // 2 :])
    ax.plot(summary["threshold"], summary["r"], "o-", color="black", label="Pearson r(Acc, Acc-hat)")
    ax.plot(summary["threshold"], summary["mean_rqi"], "s--", color="tab:green", label="mean final RQI")
    ax.axvline(DEFAULT_THRESHOLD, color="grey", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="threshold_P", ylim=(0, 1.02))
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path("figures") / "ablation_threshold.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    summary.to_csv(Path("results") / "ablation_threshold_summary.csv", index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
