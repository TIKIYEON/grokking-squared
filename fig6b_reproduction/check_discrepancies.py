"""Repeat the two checks discussed in the Figure 6(b) report.

New runs are saved under runs/checks. The submitted results are only read.
"""

import argparse
import contextlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "phase_diagram_plot/data_add_r.h5"
RESULTS = HERE / "results/fig6b"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def cell(lr_index, weight_decay):
    with h5py.File(REFERENCE, "r") as data:
        learning_rate = float(data["learning_rate"][lr_index])
        crossings = {
            "iter_train": int(data["train_steps"][lr_index, weight_decay]),
            "iter_test": int(data["test_steps"][lr_index, weight_decay]),
        }
    return learning_rate, crossings


def original_run(learning_rate, weight_decay):
    with (RESULTS / "runs.jsonl").open() as file:
        for line in file:
            run = json.loads(line)
            if (
                run["eta_dec"] == learning_rate
                and run["weight_decay_dec"] == weight_decay
                and run["status"] == "ok"
            ):
                return {
                    key: run[key]
                    for key in (
                        "iter_train", "iter_test", "reached_train", "reached_test",
                        "final_acc_train", "final_acc_test", "steps", "seed",
                    )
                }
    raise ValueError("The selected cell was not found in the submitted runs.")


def compare_saved_curves(curves, learning_rate, weight_decay):
    """Compare measurements at the steps shared with the saved full run."""
    path = RESULTS / "curves" / (
        f"curves_eta_{learning_rate:.5f}_wd_{weight_decay:.5f}.npz"
    )
    comparison = {}
    with np.load(path) as saved:
        steps, new_indices, saved_indices = np.intersect1d(
            curves["step"], saved["step"], return_indices=True
        )
        comparison["shared_measurements"] = int(len(steps))
        comparison["last_shared_step"] = int(steps[-1]) if len(steps) else None
        comparison["arrays"] = {
            key: {
                "equal": bool(np.array_equal(curves[key][new_indices], saved[key][saved_indices])),
                "matching": int(np.sum(curves[key][new_indices] == saved[key][saved_indices])),
                "total": int(len(steps)),
            }
            for key in saved.files
        }
    return comparison


def threshold_intervals(curves):
    """List consecutive steps where training accuracy reached 90%."""
    steps = curves["step"][curves["acc_train"] >= 0.9]
    if not len(steps):
        return []
    groups = np.split(steps, np.flatnonzero(np.diff(steps) > 1) + 1)
    return [
        {"first_step": int(group[0]), "last_step": int(group[-1]), "count": int(len(group))}
        for group in groups
    ]


def run_check(outdir, name, learning_rate, weight_decay, steps, curve_every, threads, rqi, state=False):
    from notebook_source import _run

    torch.set_num_threads(threads)
    print(f"Running {name}: {steps:,} steps, {threads} PyTorch thread(s).", flush=True)
    start = time.monotonic()
    with (outdir / f"{name}.log").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = _run(
                learning_rate, weight_decay, steps, curve_every, 58, 10000,
                include_rqi=rqi, capture_state=state,
            )
    curves = result.pop("curves")
    model_state = result.pop("_state", None)
    result.update(
        eta_dec=learning_rate,
        weight_decay_dec=weight_decay,
        steps=steps,
        seed=58,
        curve_every=curve_every,
        threads=threads,
        include_rqi=rqi,
        runtime_seconds=time.monotonic() - start,
        saved_curve_comparison=compare_saved_curves(curves, learning_rate, weight_decay),
    )
    if curve_every == 1:
        result["train_at_or_above_90_count"] = int(np.sum(curves["acc_train"] >= 0.9))
        result["train_at_or_above_90_intervals"] = threshold_intervals(curves)
        result["max_train_accuracy"] = float(curves["acc_train"].max())
    np.savez_compressed(outdir / f"{name}_curves.npz", **curves)
    write_json(outdir / f"{name}.json", result)
    print(
        f"{name}: train crossing {result['iter_train']}, "
        f"validation crossing {result['iter_test']}. "
        f"An unreached threshold is saved as {steps - 1}.",
        flush=True,
    )
    return result, model_state, curves


def brief_crossing(outdir):
    learning_rate, reference = cell(9, 6)
    saved_run = original_run(learning_rate, 6)
    short, short_state, short_curves = run_check(
        outdir, "without_rqi", learning_rate, 6, 5000, 1, 1, False, state=True
    )
    full, full_state, full_curves = run_check(
        outdir, "with_rqi", learning_rate, 6, 5000, 1, 1, True, state=True
    )
    comparison = {
        "published_crossings": reference,
        "submitted_full_run": saved_run,
        "crossings_without_rqi": [short["iter_train"], short["iter_test"]],
        "crossings_with_rqi": [full["iter_train"], full["iter_test"]],
        "model_tensors_equal": {
            key: bool(torch.equal(value, short_state["model"][key]))
            for key, value in full_state["model"].items()
        },
        "embeddings_equal": bool(torch.equal(full_state["latent"], short_state["latent"])),
        "targets_equal": bool(torch.equal(full_state["y_templates"], short_state["y_templates"])),
        "train_split_equal": bool(np.array_equal(full_state["train_id"], short_state["train_id"])),
        "validation_split_equal": bool(np.array_equal(full_state["test_id"], short_state["test_id"])),
        "curves_equal": {
            key: bool(np.array_equal(value, short_curves[key]))
            for key, value in full_curves.items()
        },
        "matching_accuracy_measurements": int(
            np.sum(full_curves["acc_train"] == short_curves["acc_train"])
            + np.sum(full_curves["acc_test"] == short_curves["acc_test"])
        ),
        "total_accuracy_measurements": 10000,
        "train_at_or_above_90_count": short["train_at_or_above_90_count"],
        "train_at_or_above_90_intervals": short["train_at_or_above_90_intervals"],
        "saved_full_run_prefix": short["saved_curve_comparison"],
    }
    write_json(outdir / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


def threads_check(outdir, threads, rqi):
    learning_rate, reference = cell(0, 8)
    saved_run = original_run(learning_rate, 8)
    name = f"{'with' if rqi else 'without'}_rqi_threads{threads}"
    result, _, _ = run_check(
        outdir, name, learning_rate, 8, 100000, 100, threads, rqi
    )
    comparison = {
        "published_crossings": reference,
        "submitted_full_run": saved_run,
        "rerun": result,
    }
    write_json(outdir / f"{name}_comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = parser.add_subparsers(dest="case", required=True)
    brief = cases.add_parser(
        "brief-crossing",
        help="Repeat 5,000 steps at weight decay 6, with and without the RQI calculation.",
    )
    threads = cases.add_parser(
        "threads",
        help="Repeat 100,000 steps at weight decay 8 using one or two PyTorch threads.",
    )
    threads.add_argument("--threads", type=int, choices=(1, 2), default=2)
    threads.add_argument("--rqi", action="store_true", help="Include the authors' RQI calculation.")
    for command in (brief, threads):
        command.add_argument(
            "--outdir", type=Path,
            help="Output folder. Default: runs/checks/<case> beside this script.",
        )
    args = parser.parse_args()
    outdir = (args.outdir or HERE / "runs/checks" / args.case).resolve()
    if outdir == RESULTS.resolve() or RESULTS.resolve() in outdir.parents:
        parser.error("Use a separate output folder so the submitted results stay unchanged.")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "brief-crossing":
        brief_crossing(outdir)
    else:
        threads_check(outdir, args.threads, args.rqi)
    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
