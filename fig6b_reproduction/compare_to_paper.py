#!/usr/bin/env python
"""Compare our Figure 6(b) results with Liu et al.'s saved data.

Check that the authors' data recreate the published image before comparing runs.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image

from plot_fig6b import (HERE, ROOT, REFERENCE_H5, PHASES, classify, format_axes,
                       load_reference, palette, render, validate_grid, validate_times)


def phases_from_image(path, n_wd=11, n_lr=16, frac=0.4):
    """Read labels near the centre of each grid cell."""
    if n_wd < 1 or n_lr < 1 or not 0 < frac <= 1:
        raise ValueError("Positive grid dimensions and 0 < frac <= 1 are required")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    height, width = rgb.shape[:2]
    out = np.zeros((n_wd, n_lr), dtype=int)
    for i in range(n_wd):
        for j in range(n_lr):
            y0, y1 = i * height / n_wd, (i + 1) * height / n_wd
            x0, x1 = j * width / n_lr, (j + 1) * width / n_lr
            dy, dx = (y1 - y0) * (1 - frac) / 2, (x1 - x0) * (1 - frac) / 2
            block = rgb[int(y0 + dy):int(y1 - dy), int(x0 + dx):int(x1 - dx)].reshape(-1, 3)
            if not block.size:
                raise ValueError("Raster resolution is too small for this sampling grid")
            distance = ((block[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
            out[i, j] = np.bincount(distance.argmin(1), minlength=4).argmax()
    return out


def verify_reference_image(reference, published_path):
    published_rgb = np.asarray(Image.open(published_path).convert("RGB"))
    colors = np.unique(published_rgb.reshape(-1, 3), axis=0)
    if colors.shape != palette.shape or set(map(tuple, colors)) != set(map(tuple, palette)):
        raise ValueError("Published panel must contain exactly the authors' four palette colors")
    sampled = phases_from_image(published_path, *reference.shape)
    labels_equal = int((sampled == reference).sum())
    image = Image.fromarray(palette[render(reference.T, reference.shape)].astype(np.uint8))
    reconstructed = np.asarray(image.resize((published_rgb.shape[1], published_rgb.shape[0]),
                                            Image.Resampling.NEAREST))
    exact = np.all(reconstructed == published_rgb, axis=2)
    pixels_equal, pixels_total = int(exact.sum()), int(exact.size)
    if labels_equal != reference.size or pixels_equal != pixels_total:
        raise ValueError(f"Reference verification failed: {labels_equal}/{reference.size} native labels "
                         f"and {pixels_equal}/{pixels_total} pixels agree with the published raster")
    return sampled, reconstructed, pixels_equal, pixels_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, default=HERE / "figures/fig6b_phase_data.npz")
    ap.add_argument("--published", type=Path, default=HERE / "reference/published_fig6b_panel.png")
    ap.add_argument("--reference-h5", type=Path, default=REFERENCE_H5)
    ap.add_argument("--outdir", type=Path, default=HERE / "runs/figures")
    ap.add_argument("--tag", default="fig6b")
    args = ap.parse_args()
    torch.set_num_threads(1)
    with np.load(args.ours, allow_pickle=False) as data:
        required = {"eta_decs", "wds", "steps", "threshold", "train_iters", "test_iters",
                    "phase_wd_by_lr", "source", "source_path"}
        missing = required - set(data.files)
        # Check that both results use the same grid.
        lrs, weight_decays = validate_grid(data["eta_decs"], data["wds"])
        if missing:
            raise ValueError(f"Phase artifact is missing provenance/data fields: {sorted(missing)}")
        steps, threshold = int(data["steps"]), float(data["threshold"])
        if steps != 100000 or threshold != 1000:
            raise ValueError("Paper comparison requires budget 100000 and delay threshold 1000")
        train, test = data["train_iters"].copy(), data["test_iters"].copy()
        validate_times(train, test, steps)
        ours = data["phase_wd_by_lr"].copy()
        if ours.shape != train.shape or not np.array_equal(ours, classify(train, test, steps, threshold).T):
            raise ValueError("Saved phase labels do not agree with the saved crossing times")
        ours = ours.astype(np.int64, copy=False)
        source, source_path = str(data["source"].item()), str(data["source_path"].item())
        if source not in ("fresh", "authors-data"):
            raise ValueError(f"Unknown phase artifact source {source!r}")
    ref_train, ref_test, ref_lrs, ref_wds = load_reference(args.reference_h5)
    if not (np.allclose(lrs, ref_lrs, rtol=1e-12, atol=0) and np.array_equal(weight_decays, ref_wds)):
        raise ValueError("Reproduction and published reference have different hyperparameter coordinates")
    published = classify(ref_train, ref_test, 100000, 1000).T
    sampled, reconstructed, pixel_equal, pixel_total = verify_reference_image(published, args.published)
    agree = published == ours
    confusion = np.array([[int(((published == i) & (ours == j)).sum()) for j in range(4)]
                          for i in range(4)])
    counts_paper = {name: int((published == k).sum()) for k, name in enumerate(PHASES)}
    counts_ours = {name: int((ours == k).sum()) for k, name in enumerate(PHASES)}
    print(f"input source: {source} ({source_path})")
    if source == "authors-data":
        print("This is an authors-data reconstruction check; it is not a fresh-training result.")
    print(f"published native labels verified: {int((sampled == published).sum())}/{published.size}")
    print(f"published reference raster verified: {pixel_equal}/{pixel_total} pixels")
    print(f"matching classifications: {int(agree.sum())}/{agree.size} = {100 * agree.mean():.2f}%")
    print("paper phase counts:", counts_paper)
    print("input phase counts:", counts_ours)
    print("confusion matrix (rows=paper, columns=input; C/G/M/X):")
    print(confusion)
    # Write the comparison after checking the inputs.
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / f"{args.tag}_comparison.npz", published=published, ours=ours, agree=agree,
             eta_decs=lrs, wds=weight_decays, steps=steps, threshold=threshold,
             source=source, source_path=source_path, confusion_matrix=confusion)
    with (args.outdir / f"{args.tag}_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weight_decay", "decoder_lr", "paper_phase", "input_phase", "agree", "source"])
        for i, wd in enumerate(weight_decays):
            for j, lr in enumerate(lrs):
                writer.writerow([f"{wd:g}", f"{lr:.17g}", PHASES[published[i, j]],
                                 PHASES[ours[i, j]], int(agree[i, j]), source])
    report = {"source": source, "source_path": source_path, "grid_shape_wd_by_lr": list(ours.shape),
              "steps": steps, "threshold": threshold, "matching_cells": int(agree.sum()),
              "total_cells": int(agree.size), "agreement_fraction": float(agree.mean()),
              "reference_matching_pixels": pixel_equal, "reference_total_pixels": pixel_total,
              "paper_phase_counts": counts_paper, "input_phase_counts": counts_ours,
              "confusion_matrix_paper_by_input": confusion.tolist()}
    (args.outdir / f"{args.tag}_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    Image.fromarray(reconstructed).save(args.outdir / f"{args.tag}_authors_reference_panel.png")
    input_title = "Our reproduced\nresults" if source == "fresh" else "Authors' data\nreconstruction"
    # Keep the same panel shape as in the paper.
    panel_aspect = reconstructed.shape[0] / reconstructed.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 16.6))
    for ax, phase, title in ((axes[0], published, "Published Figure 6(b)\nData from Liu et al. [1]"),
                              (axes[1], ours, input_title)):
        ax.imshow(palette[render(phase.T, phase.shape)], interpolation="nearest", aspect="auto")
        ax.set_box_aspect(panel_aspect)
        format_axes(ax, lrs, weight_decays, scale=100, labelsize=22, ticksize=20)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=22, pad=10)
    axes[2].imshow((~agree).astype(int), interpolation="nearest", aspect="auto", vmin=0, vmax=1,
                   cmap=matplotlib.colors.ListedColormap(["#e8e8e8", "#c0392b"]))
    axes[2].set_box_aspect(panel_aspect)
    format_axes(axes[2], lrs, weight_decays, labelsize=22, ticksize=20)
    axes[2].set_title(f"{int((~agree).sum())}/{agree.size} cells differ\n({100 * agree.mean():.1f}% match)",
                      fontsize=22, pad=10)
    fig.legend(handles=[Patch(facecolor=palette[k] / 255, label=PHASES[k]) for k in range(4)],
               loc="lower center", ncol=2, frameon=False, fontsize=18)
    fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=1.8)
    # Disabling PDF compression avoids incorrect colors with these versions.
    with matplotlib.rc_context({"pdf.compression": 0}):
        for ext in ("png", "pdf"):
            fig.savefig(args.outdir / f"{args.tag}_comparison.{ext}", bbox_inches="tight", dpi=200)
            shutil.copyfile(args.outdir / f"{args.tag}_comparison.{ext}",
                            args.outdir / f"{args.tag}_comparison_stacked.{ext}")
    plt.close(fig)
    print("wrote comparison and 176 coordinate records to", args.outdir)


if __name__ == "__main__":
    main()
