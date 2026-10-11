#!/usr/bin/env python
"""Plot the Figure 6(b) results.

Use our saved runs by default, or --source authors-data for Liu et al.'s results.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE_H5 = ROOT / "phase_diagram_plot/data_add_r.h5"
FRESH_RESULTS = HERE / "results/fig6b"
eta_decs = 10 ** np.linspace(-5, -2, num=16)
wds = np.arange(11)
PHASES = ["comprehension", "grokking", "memorization", "confusion"]
palette = np.array([[160, 218, 57], [253, 231, 37], [70, 50, 126], [30, 30, 30]])


def validate_grid(lrs, weight_decays):
    lrs, weight_decays = np.asarray(lrs, dtype=float), np.asarray(weight_decays, dtype=float)
    if (lrs.shape != eta_decs.shape or weight_decays.shape != wds.shape
            or not np.allclose(lrs, eta_decs, rtol=1e-12, atol=0)
            or not np.array_equal(weight_decays, wds)):
        raise ValueError("Figure 6(b) requires 16 decoder learning rates 10**linspace(-5,-2,16) "
                         "and 11 weight decays 0..10; refusing a different grid.")
    return lrs, weight_decays


def validate_times(train, test, steps):
    for name, values in (("train", train), ("validation", test)):
        if np.shape(values) != (len(wds), len(eta_decs)):
            raise ValueError(f"{name} crossing array must have shape (11, 16), got {np.shape(values)}")
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values > steps - 1).any() or (values != np.floor(values)).any()):
            raise ValueError(f"{name} crossing times must be integers in 0..{steps - 1}")


def read_manifest(resdir):
    path = Path(resdir) / "manifest.json"
    with path.open() as f:
        manifest = json.load(f)
    for key in ("eta_decs", "wds", "steps"):
        if key not in manifest:
            raise ValueError(f"{path} is missing required field {key!r}")
    lrs, weight_decays = validate_grid(manifest["eta_decs"], manifest["wds"])
    steps = int(manifest["steps"])
    if steps != manifest["steps"] or steps < 2:
        raise ValueError(f"Invalid training budget in {path}")
    if "n_runs" in manifest and manifest["n_runs"] != lrs.size * weight_decays.size:
        raise ValueError(f"n_runs in {path} disagrees with its grid")
    return manifest, lrs, weight_decays, steps


def load(resdir):
    """Read our results with rows for weight decay and columns for learning rate."""
    _, lrs, weight_decays, steps = read_manifest(resdir)
    paths = [(Path(resdir) / (f"{kind}step_eta_{lr:.5f}_wd_{wd:.5f}.txt"))
             for kind in ("train", "test") for wd in weight_decays for lr in lrs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete fresh sweep: {len(missing)} crossing files missing; "
                                f"first missing file: {missing[0]}. No reference data will be substituted.")
    values = []
    for path in paths:
        value = np.asarray(np.loadtxt(path))
        if value.size != 1:
            raise ValueError(f"Expected one crossing time in {path}, found {value.size}")
        values.append(float(value.reshape(-1)[0]))
    train, test = np.asarray(values).reshape(2, len(weight_decays), len(lrs))
    validate_times(train, test, steps)
    return train, test


def load_reference(path=REFERENCE_H5):
    """Read the authors' saved results."""
    with h5py.File(path, "r") as f:
        lrs, weight_decays = validate_grid(f["learning_rate"][()], f["weight_decay"][()])
        train, test = f["train_steps"][()].T, f["test_steps"][()].T
    validate_times(train, test, 100000)
    return train, test, lrs, weight_decays


def classify(train_iterss, test_iterss, steps, threshold=1e3):
    """Authors' exact phase rule: input [wd, lr], output [lr, wd]."""
    return (
        np.transpose((train_iterss < (steps - 1)) * (test_iterss < (steps - 1))
                     * ((test_iterss - train_iterss) < threshold)) * 0
        + np.transpose((train_iterss < (steps - 1)) * (test_iterss < (steps - 1))
                       * ((test_iterss - train_iterss) > threshold)) * 1
        + np.transpose((train_iterss < (steps - 1))
                       * (test_iterss == (steps - 1))) * 2
        + np.transpose((train_iterss == (steps - 1))
                       * (test_iterss == (steps - 1))) * 3
    )


def render(I, shp, scale=100):
    """Smooth the boundaries using the authors' plotting method."""
    id_mat = np.eye(4)
    RGB = id_mat[I]
    RGBt = np.transpose(RGB, (2, 0, 1))
    RGBt = torch.nn.functional.interpolate(
        torch.tensor(np.array([RGBt]).astype(np.float64)), scale_factor=scale, mode="bicubic")
    RGBt = np.transpose(RGBt[0], (1, 2, 0)).detach().numpy()
    return np.transpose(
        np.argmin(np.mean((RGBt.reshape(shp[0] * shp[1] * scale ** 2, 4)[np.newaxis, :]
                           - id_mat[:, np.newaxis, :]) ** 2, axis=2),
                  axis=0).reshape(shp[1] * scale, shp[0] * scale))


def format_axes(ax, lrs, weight_decays, scale=1, *, labelsize=14, ticksize=12):
    ticks = [int(np.flatnonzero(np.isclose(lrs, x, rtol=1e-12, atol=0))[0])
             for x in (1e-5, 1e-4, 1e-3, 1e-2)]
    offset = scale / 2 if scale > 1 else 0
    ax.set_xticks([i * scale + offset for i in ticks], ["1e-5", "1e-4", "1e-3", "1e-2"])
    ax.set_yticks([i * scale + offset for i in (0, 5, 10)], ["0", "5", "10"])
    ax.set_xlabel("decoder learning rate", fontsize=labelsize)
    ax.set_ylabel("weight decay", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticksize)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resdir", type=Path, default=FRESH_RESULTS)
    ap.add_argument("--source", choices=("fresh", "authors-data"), default="fresh")
    ap.add_argument("--reference", action="store_true", help="alias for --source authors-data")
    ap.add_argument("--reference-h5", type=Path, default=REFERENCE_H5)
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--steps", type=int, help="optional check against the source budget; never changes it")
    ap.add_argument("--threshold", type=float, default=1e3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    source = "authors-data" if args.reference else args.source
    if source == "authors-data":
        train, test, lrs, weight_decays = load_reference(args.reference_h5)
        steps = 100000
        source_path = os.path.relpath(args.reference_h5.resolve(), ROOT)
        outdir = args.outdir or HERE / "runs/authors_figures"
        tag = args.tag or "authors_fig6b"
        source_label = "Results from Liu et al."
    else:
        _, lrs, weight_decays, steps = read_manifest(args.resdir)
        train, test = load(args.resdir)
        source_path = os.path.relpath(args.resdir.resolve(), ROOT)
        outdir = args.outdir or HERE / "runs/figures"
        tag = args.tag or "fig6b"
        source_label = "Our reproduced results"
    if args.steps is not None and args.steps != steps:
        raise ValueError(f"Requested --steps {args.steps} disagrees with source budget {steps}")
    if not np.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    phase = classify(train, test, steps, args.threshold).T
    rendered = render(phase.T, phase.shape)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"{tag}_phase_data.npz", eta_decs=lrs, wds=weight_decays,
             steps=steps, threshold=args.threshold, train_iters=train, test_iters=test,
             phase_wd_by_lr=phase, source=source, source_path=source_path)
    with (outdir / f"{tag}_phase_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weight_decay", "decoder_lr", "steps_to_train_acc90", "steps_to_val_acc90",
                         "delay", "phase", "source"])
        for i, wd in enumerate(weight_decays):
            for j, lr in enumerate(lrs):
                writer.writerow([f"{wd:g}", f"{lr:.17g}", int(train[i, j]), int(test[i, j]),
                                 int(test[i, j] - train[i, j]), PHASES[phase[i, j]], source])
    Image.fromarray(palette[rendered].astype(np.uint8)).save(outdir / f"{tag}_panel.png")
    handles = [Patch(facecolor=palette[k] / 255, label=PHASES[k]) for k in range(4)]
    for raw in (False, True):
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.imshow(palette[phase if raw else rendered], interpolation="nearest", aspect="auto")
        format_axes(ax, lrs, weight_decays, scale=1 if raw else 100)
        ax.set_title(f"Figure 6(b): addition, regression\n{source_label} ({steps:,} steps)",
                     fontsize=16, pad=12)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  ncol=2, frameon=False, fontsize=14)
        suffix = "_rawgrid" if raw else ""
        # Disabling PDF compression avoids incorrect colors with these versions.
        with matplotlib.rc_context({"pdf.compression": 0}):
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"{tag}{suffix}.{ext}", bbox_inches="tight", dpi=200)
        plt.close(fig)
    print(f"source: {source}; grid: {len(weight_decays)} weight decays x {len(lrs)} learning rates")
    print("grid cells per phase:", {p: int((phase == i).sum()) for i, p in enumerate(PHASES)})
    print("wrote figures and all 176 crossing records to", outdir)


if __name__ == "__main__":
    main()
