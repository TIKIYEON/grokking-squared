import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fig3(df, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [
        ("train_frac", "acc", "training data fraction", "Acc", "(a)"),
        ("train_frac", "pred_acc", "training data fraction", r"$\widehat{\rm Acc}$", "(b)"),
        ("pred_acc", "acc", r"$\widehat{\rm Acc}$", "Acc", "(c)"),
    ]
    for ax, (x, y, xlabel, ylabel, tag) in zip(axes, panels):
        ax.scatter(df[x], df[y], s=18, color="blue", alpha=0.25, edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", color="red", lw=1)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=xlabel, ylabel=ylabel, title=tag)
        ax.set_aspect("equal")
    axes[0].text(0.55, 0.42, "Memorization", color="red", rotation=45, fontsize=9)

    r = np.corrcoef(df["pred_acc"], df["acc"])[0, 1]
    mae = (df["acc"] - df["pred_acc"]).abs().mean()
    axes[2].text(0.04, 0.9, f"r = {r:.3f}\nMAE = {mae:.3f}\nn = {len(df)}", fontsize=9, va="top")
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = args.out or Path("figures") / f"fig3_{args.csv.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_fig3(df, args.title).savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
