import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# optimizer families: colour + marker, colourblind-safe (Okabe-Ito blue / orange). Filled = reached, open = not reached.
STYLE = {
    "adamw": {"color": "#0072B2", "marker": "o", "label": "AdamW"},
    "adam": {"color": "#0072B2", "marker": "o", "label": "Adam"},
    "sgd": {"color": "#E69F00", "marker": "s", "label": "SGD"},
    "gd": {"color": "#E69F00", "marker": "s", "label": "GD"},
}


def critical_fraction(lam, level=0.5):
    """Smallest training fraction at which at least `level` of the random training sets have lambda3 > 0."""
    p_pos = lam.assign(pos=lam["lambda3"] > 0).groupby("train_frac")["pos"].mean()
    return p_pos[p_pos >= level].index.min()


def fit_slope(runs, n_boot=2000, seed=0):
    """OLS slope of log10(steps to RQI > 0.95) vs log10(lambda3) over runs that reached it, with a bootstrap 95% CI.
    Also returns the lambda3 range of the fitted runs (for drawing the fit line)."""
    r = runs[(runs["reached"] == 1) & (runs["lambda3"] > 0)]
    if len(r) < 3:
        return {"slope": np.nan, "intercept": np.nan, "ci": (np.nan, np.nan), "n": len(r), "lam_range": (np.nan, np.nan)}
    x, y = np.log10(r["lambda3"].to_numpy()), np.log10(r["rqi_step"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    boots = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return {"slope": slope, "intercept": intercept, "ci": tuple(np.percentile(boots, [2.5, 97.5])), "n": len(x),
            "lam_range": (r["lambda3"].min(), r["lambda3"].max())}


def load_authors_fig4(toy_dir=HERE.parent / "toy"):
    """The authors' released Fig. 4b runs (toy/results/fig4, train_add with 10^4 steps) with lambda3 of their training sets.

    The files hold train_add's iter_rqi: 0-based, and steps - 1 = 9999 when RQI > 0.95 was never reached.
    """
    from lambda3 import effective_theory, train_set

    rows = []
    for f in sorted((toy_dir / "results" / "fig4").glob("rqistep_num_*_seed_*.txt")):
        n, s = map(int, re.findall(r"num_(\d+)_seed_(\d+)", f.name)[0])
        it = int(np.loadtxt(f))
        rows.append({"train_num": n, "seed": s, "iter_rqi": it, "reached": int(it < 9999), "rqi_step": it + 1 if it < 9999 else np.nan,
                     "steps": 10000, "eta_repr": 1e-3, **effective_theory(train_set(n, s))})
    return pd.DataFrame(rows)


def theory_line(ax, lam, eta, **kw):
    xs = np.logspace(np.log10(lam.min()) - 0.1, np.log10(lam.max()) + 0.1, 50)
    ax.plot(xs, 1 / (2 * xs * eta), **{"color": "black", "ls": "--", "lw": 1, "label": r"theory $t = 1/(2\lambda_3\eta)$", **kw})


def fit_line(ax, fit, **kw):
    """The log-log fit, drawn only over the lambda3 range of the runs it was fitted to."""
    if not np.isfinite(fit["slope"]): return
    xs = np.logspace(np.log10(fit["lam_range"][0]), np.log10(fit["lam_range"][1]), 50)
    ax.plot(xs, 10 ** fit["intercept"] * xs ** fit["slope"], **{"ls": ":", "lw": 1.5,
            "label": f"fit: slope {fit['slope']:.2f} [{fit['ci'][0]:.2f}, {fit['ci'][1]:.2f}]", **kw})


def plot_fig5a(ax, lam):
    ax.scatter(lam["train_frac"], lam["lambda3"], s=10, color="blue", alpha=0.08, edgecolors="none", label=f"random training sets ({lam.groupby('train_num').size().iloc[0]} per fraction)")
    mean = lam.groupby("train_frac")["lambda3"].mean()
    ax.plot(mean.index, mean.values, color="black", lw=1.5, label="mean")
    fc = critical_fraction(lam)
    ax.axvline(fc, color="red", ls="--", lw=1, label=rf"P($\lambda_3>0$) $\geq$ 0.5 at {fc:.2f}")
    ax.set(xlim=(0, 1), ylim=(0, None), xlabel="training data fraction", ylabel=r"grokking rate $\lambda_3$", title="(a)")
    ax.legend(fontsize=8, loc="upper left")


def plot_fig5b(ax, runs, authors=None):
    """Paper-style 5(b): blue dots reached, red triangles not reached (at the step cap). Optionally overlays the
    authors' released Fig. 4b runs (open black diamonds)."""
    eta = runs["eta_repr"].iloc[0]
    steps = runs["steps"].max()
    runs = runs[runs["lambda3"] > 0]
    hit, miss = runs[runs["reached"] == 1], runs[runs["reached"] == 0]

    ax.scatter(hit["lambda3"], hit["rqi_step"], s=18, color="blue", alpha=0.6, edgecolors="none", label=f"RQI > 0.95 reached ({len(hit)})")
    ax.scatter(miss["lambda3"], miss["steps"], s=28, marker="^", color="red", alpha=0.6, edgecolors="none", label=f"not reached in {steps:.0e} steps ({len(miss)})")
    if authors is not None:
        a = authors[authors["lambda3"] > 0]
        ah = a[a["reached"] == 1]
        ax.scatter(a["lambda3"], np.where(a["reached"] == 1, a["rqi_step"], a["steps"]), s=40, marker="D", facecolors="none", edgecolors="black", lw=0.8,
                   label=rf"authors' toy/results/fig4, $\lambda_3>0$ ({len(ah)} reached, {len(a) - len(ah)} not)")
    theory_line(ax, runs["lambda3"], eta)
    fit_line(ax, fit_slope(runs), color="grey")
    ax.set(xscale="log", yscale="log", ylim=(None, 12 * steps), xlabel=r"grokking rate $\lambda_3$", ylabel="steps to RQI > 0.95", title=rf"(b)  $\eta$ = {eta:g}")
    ax.legend(fontsize=7.5, loc="upper right")


def plot_steps_vs_lambda3(ax, runs, style, eta, cap, title=None, theory=True, slope_ref=False):
    """5(b)-style panel for one optimizer: filled markers reached, open markers at the step cap not reached, and the
    log-log fit with bootstrap CI. `theory` draws t = 1/(2 lambda3 eta) (only meaningful when eta is a gradient-descent
    step on the effective loss scale); `slope_ref` draws a grey slope -1 reference through the median reached point."""
    st = STYLE[style]
    runs = runs[runs["lambda3"] > 0]
    hit, miss = runs[runs["reached"] == 1], runs[runs["reached"] == 0]
    ax.scatter(hit["lambda3"], hit["rqi_step"], s=18, marker=st["marker"], color=st["color"], alpha=0.6, edgecolors="none",
               label=f"{st['label']}: reached ({len(hit)}/{len(runs)})")
    if len(miss):
        ax.scatter(miss["lambda3"], np.full(len(miss), cap), s=28, marker=st["marker"], facecolors="none", edgecolors=st["color"], lw=0.8, alpha=0.7,
                   label=f"{st['label']}: not reached in {cap:.0e} ({len(miss)})")
    if theory:
        theory_line(ax, runs["lambda3"], eta)
    fit = fit_slope(runs)
    fit_line(ax, fit, color=st["color"])
    if slope_ref and np.isfinite(fit["slope"]):
        xs = np.logspace(np.log10(fit["lam_range"][0]), np.log10(fit["lam_range"][1]), 50)
        x0, y0 = hit["lambda3"].median(), hit["rqi_step"].median()
        ax.plot(xs, y0 * (xs / x0) ** -1, color="grey", ls="-", lw=1, alpha=0.8, label="slope −1 reference (through median)")
    top = 40 * cap if len(miss) else 12 * hit["rqi_step"].max()
    ax.set(xscale="log", yscale="log", ylim=(None, top), xlabel=r"grokking rate $\lambda_3$", ylabel="steps to RQI > 0.95",
           title=title or rf"{st['label']}, $\eta$ = {eta:g}")
    ax.legend(fontsize=7.5, loc="upper right")
    return fit


def plot_fig5(lam, runs, authors=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    plot_fig5a(axes[0], lam)
    plot_fig5b(axes[1], runs, authors)
    fig.tight_layout()
    return fig


def plot_ablation(adamw, sgd):
    """AdamW vs SGD on the embeddings (decoder: AdamW in both). No theory line for SGD: its eta is a step size on the
    network loss, not on the effective loss, so 1/(2 lambda3 eta) has no meaning there."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (style, d) in zip(axes, [("adamw", adamw), ("sgd", sgd)]):
        eta = d["eta_repr"].iloc[0]
        plot_steps_vs_lambda3(ax, d, style, eta=eta, cap=d["steps"].iloc[0], theory=(style == "adamw"), slope_ref=(style == "sgd"),
                              title=rf"{STYLE[style]['label']} on embeddings, $\eta_{{\rm repr}}$ = {eta:g}")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="figures/fig5.png and figures/fig5_ablation.png from the CSVs in results/")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    R = args.results
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_fig5(pd.read_csv(R / "lambda3.csv"), pd.read_csv(R / "runs.csv"), authors=pd.read_csv(R / "runs_authors.csv")).savefig(args.figures / "fig5.png", dpi=200)
    plot_ablation(pd.read_csv(R / "ablation_adamw.csv"), pd.read_csv(R / "ablation_sgd.csv")).savefig(args.figures / "fig5_ablation.png", dpi=200)
    print(f"saved {args.figures / 'fig5.png'} and {args.figures / 'fig5_ablation.png'}")


if __name__ == "__main__":
    main()
