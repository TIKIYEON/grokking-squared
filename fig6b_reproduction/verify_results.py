#!/usr/bin/env python
"""Check the saved Figure 6(b) runs without changing them."""

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = ROOT / "phase_diagram_plot/data_add_r.h5"
NOTEBOOK = ROOT / "phase_diagram_plot/phase_diagram.ipynb"
PHASES = ["comprehension", "grokking", "memorization", "confusion"]
CONFIG_KEYS = (
    "format_version", "steps", "seed", "curve_every", "eta_decs", "wds",
    "torch_threads", "torch_version", "numpy_version", "python", "source",
    "adapter_sha256",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tag(lr, wd):
    return f"eta_{lr:.5f}_wd_{wd:.5f}"


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def integer(value):
    return type(value) is int


def finite_number(value):
    return type(value) in (int, float) and np.isfinite(value)


def classify(train, validation, budget=100000, threshold=1000):
    """Apply the authors' classification rule."""
    end = budget - 1
    phase = np.zeros(train.shape, dtype=np.int64)
    phase[(train < end) & (validation < end) & (validation - train > threshold)] = 1
    phase[(train < end) & (validation == end)] = 2
    phase[(train == end) & (validation == end)] = 3
    return phase


def source_integrity(manifest, errors):
    """Check the authors' source files against the saved run information."""
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    source = "\n".join("".join(notebook["cells"][i]["source"]) for i in (0, 1))
    saved_source = manifest.get("source", {})
    checks = {
        "notebook": (hashlib.sha256(notebook_bytes).hexdigest(), saved_source.get("notebook_sha256")),
        "notebook_cells_0_1": (hashlib.sha256(source.encode()).hexdigest(), saved_source.get("source_sha256")),
        "reference_hdf5": (sha256(REFERENCE), manifest.get("grid_reference_sha256")),
    }
    result = {}
    for name, (actual, expected) in checks.items():
        result[name] = {"sha256": actual, "expected_sha256": expected, "matches": actual == expected}
        if actual != expected:
            errors.append(f"Source hash mismatch: {name}")
    result["saved_script_hashes"] = {
        "adapter_sha256": manifest.get("adapter_sha256"),
        "runner_sha256": manifest.get("runner_sha256"),
        "checked_against_original_scripts": False,
        "note": (
            "These identify the scripts used when the results were saved. "
            "They are recorded here but are not checked against the original files. "
            "The submitted historical runs were completed before the packaging changes."
        ),
    }
    return result


def check_cell(resdir, lr, wd, manifest, expected_steps, errors, ambiguities):
    name = tag(lr, wd)
    prefix = f"wd={wd:g}, lr={lr:.17g}: "

    def require(condition, message):
        if not condition:
            errors.append(prefix + message)

    record_path = resdir / "records" / f"{name}.json"
    record = json.loads(record_path.read_text())
    expected_metadata = {
        "status": "ok", "eta_dec": lr, "weight_decay_dec": wd,
        "steps": manifest["steps"], "seed": manifest["seed"],
        "curve_every": manifest["curve_every"], "torch_threads": 1,
        "config_id": manifest["config_id"], "provenance": manifest["source"],
    }
    for key, expected in expected_metadata.items():
        require(record.get(key) == expected, f"record {key} differs from manifest/grid")
    require(finite_number(record.get("runtime_s")) and record["runtime_s"] > 0,
            "invalid runtime_s")
    end = manifest["steps"] - 1
    for kind, crossing_key in (("train", "iter_train"), ("test", "iter_test")):
        crossing = record.get(crossing_key)
        require(integer(crossing) and 0 <= crossing <= end, f"invalid {crossing_key}: {crossing!r}")
        txt = np.asarray(np.loadtxt(resdir / f"{kind}step_{name}.txt"))
        require(txt.size == 1 and np.isfinite(txt).all()
                and float(txt.reshape(-1)[0]) == crossing,
                f"{kind} crossing text file differs from record or is invalid")

    with np.load(resdir / "curves" / f"curves_{name}.npz", allow_pickle=False) as archive:
        expected_keys = {"step", "acc_train", "acc_test", "loss_train"}
        require(set(archive.files) == expected_keys, "unexpected/missing curve array keys")
        curves = {key: archive[key].copy() for key in archive.files}
    if set(curves) != expected_keys:
        return record
    shapes_ok = all(value.shape == expected_steps.shape for value in curves.values())
    require(shapes_ok, "curve arrays must each contain the expected 1001 samples")
    if not shapes_ok:
        return record
    require(curves["step"].dtype.kind in "iu", "step array is not an integer array")
    require(np.array_equal(curves["step"], expected_steps), "wrong sampled steps or missing final step")
    for key in expected_keys - {"step"}:
        require(curves[key].dtype.kind == "f", f"{key} is not a floating-point array")
        require(np.isfinite(curves[key]).all(), f"nonfinite {key}; no phase will be assigned")
    require((curves["loss_train"] >= 0).all(), "negative MSE loss")
    final_loss = record.get("final_loss_train")
    require(finite_number(final_loss) and final_loss >= 0, "invalid/nonfinite final_loss_train")
    require(curves["loss_train"][-1] == final_loss, "final loss differs from last curve sample")

    for kind, denominator in (("train", 45), ("test", 10)):
        accuracy = curves[f"acc_{kind}"]
        reached = record.get(f"reached_{kind}")
        crossing = record.get(f"iter_{kind}")
        final_accuracy = record.get(f"final_acc_{kind}")
        require(type(reached) is bool, f"reached_{kind} must be a boolean")
        require(np.all((accuracy >= 0) & (accuracy <= 1)), f"{kind} accuracy is outside [0,1]")
        require(np.allclose(accuracy * denominator, np.rint(accuracy * denominator), rtol=0, atol=1e-10),
                f"{kind} accuracy is not a multiple of 1/{denominator}")
        require(finite_number(final_accuracy) and 0 <= final_accuracy <= 1,
                f"invalid final_acc_{kind}")
        require(accuracy[-1] == final_accuracy, f"final {kind} accuracy differs from final curve sample")
        if type(reached) is not bool or not integer(crossing) or not 0 <= crossing <= end:
            continue
        if not reached:
            require(crossing == end, f"never-reached {kind} must use the budget-1 sentinel")
            require(np.all(accuracy < 0.9), f"never-reached {kind} has sampled accuracy >= 0.9")
        else:
            require(np.all(accuracy[expected_steps < crossing] < 0.9),
                    f"sampled {kind} accuracy reaches 0.9 before its reported first crossing")
            at_crossing = expected_steps == crossing
            if at_crossing.any():
                require(np.all(accuracy[at_crossing] >= 0.9),
                        f"sample at exact {kind} crossing is below 0.9")
            if crossing == end:
                ambiguities.append(dict(weight_decay=wd, decoder_lr=lr, metric=kind,
                                        first_crossing=crossing, reached=True))

    # The saved counter uses every step. Samples only give bounds for it.
    below = record.get("val_below_last10000")
    tail_length = min(10000, manifest["steps"])
    tail_start = manifest["steps"] - tail_length
    require(integer(below) and 0 <= below <= tail_length, "invalid exact validation-tail counter")
    tail_mask = expected_steps >= tail_start
    tail_steps = expected_steps[tail_mask]
    tail_accuracy = curves["acc_test"][tail_mask]
    lower = int((tail_accuracy < 0.9).sum())
    upper = tail_length - int((tail_accuracy >= 0.9).sum())
    crossing = record.get("iter_test")
    if record.get("reached_test") is False:
        lower = upper = tail_length
    elif record.get("reached_test") is True and integer(crossing) and tail_start <= crossing <= end:
        lower = crossing - tail_start + int(((tail_steps >= crossing) & (tail_accuracy < 0.9)).sum())
        if crossing not in tail_steps:
            upper -= 1  # The unsampled crossing itself must be a successful validation step.
    require(integer(below) and lower <= below <= upper,
            f"validation-tail counter {below!r} contradicts measured bounds [{lower}, {upper}]")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resdir", type=Path, default=HERE / "results/fig6b")
    parser.add_argument("--outdir", type=Path, default=HERE / "runs/checks")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    for protected in (ROOT / "phase_diagram_plot", HERE / "results", args.resdir.resolve()):
        if args.outdir == protected.resolve() or protected.resolve() in args.outdir.parents:
            parser.error(f"Validation reports must not be written inside protected {protected}")
    resdir = args.resdir.resolve()
    start = time.monotonic()
    manifest_path = resdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with h5py.File(REFERENCE, "r") as reference:
        rates = reference["learning_rate"][:]
        decays = reference["weight_decay"][:]
        reference_train = reference["train_steps"][:].T
        reference_test = reference["test_steps"][:].T
    coordinates = [(float(lr), float(wd)) for wd in decays for lr in rates]
    names = {tag(lr, wd) for lr, wd in coordinates}
    expected_files = {
        "records": {resdir / "records" / f"{name}.json" for name in names},
        "crossings": {resdir / f"{kind}step_{name}.txt" for name in names for kind in ("train", "test")},
        "curves": {resdir / "curves" / f"curves_{name}.npz" for name in names},
    }
    missing = [str(path) for paths in expected_files.values() for path in sorted(paths) if not path.is_file()]
    if missing:
        print(f"Incomplete sweep: {len(missing)} required artifacts missing. No validation report written.")
        print("First missing artifact:", missing[0])
        raise SystemExit(2)

    errors = []
    actual_files = {
        "records": set((resdir / "records").glob("*.json")),
        "crossings": set(resdir.glob("trainstep_*.txt")) | set(resdir.glob("teststep_*.txt")),
        "curves": set((resdir / "curves").glob("*.npz")),
    }
    counts = {key: len(paths) for key, paths in actual_files.items()}
    for key in expected_files:
        if actual_files[key] != expected_files[key]:
            errors.append(f"Unexpected {key} artifacts: " + str(sorted(str(p) for p in actual_files[key] - expected_files[key])))
    expectations = {
        "format_version": 2, "steps": 100000, "seed": 58, "curve_every": 100,
        "torch_threads": 1, "n_runs": 176, "successful_runs": 176,
        "data_origin": "fresh_training", "eta_repr": 0.001,
        "weight_decay_repr": 0.0, "train_num": 45, "test_num": 10,
        "p": 10, "repr_dim": 1, "output_dim": 30, "width": 200, "batch_size": 45,
        "batch_order": "fixed original train_id order",
        "threshold_train_acc": 0.9, "threshold_test_acc": 0.9,
        "delay_threshold": 1000, "evaluation": "post-update for both training and validation",
        "early_stopping": False,
    }
    for key, value in expectations.items():
        if manifest.get(key) != value:
            errors.append(f"Manifest {key} must equal {value!r}; found {manifest.get(key)!r}")
    if len(coordinates) != 176 or len(names) != 176:
        errors.append("Reference grid must contain 176 unique cells and filenames")
    if not np.array_equal(manifest.get("eta_decs"), rates) or not np.array_equal(manifest.get("wds"), decays):
        errors.append("Manifest grid differs from released HDF5 coordinates")
    config = {key: manifest.get(key) for key in CONFIG_KEYS}
    recomputed_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if manifest.get("config_id") != recomputed_id:
        errors.append("Manifest config_id does not match its configuration fields")
    if not finite_number(manifest.get("completed_unix")):
        errors.append("Manifest does not record completed_unix")

    try:
        sources = source_integrity(manifest, errors)
    except Exception as exc:
        errors.append(f"Cannot verify source integrity: {type(exc).__name__}: {exc}")
        sources = {"error": str(exc)}
    expected_steps = np.r_[np.arange(0, 100000, 100, dtype=np.int64), 99999]
    ambiguities = []
    records = []
    for lr, wd in coordinates:
        try:
            records.append(check_cell(resdir, lr, wd, manifest, expected_steps, errors, ambiguities))
        except Exception as exc:
            errors.append(f"wd={wd:g}, lr={lr:.17g}: cannot validate artifacts: {type(exc).__name__}: {exc}")

    comparison = None
    if not errors:
        train = np.asarray([record["iter_train"] for record in records]).reshape(11, 16)
        validation = np.asarray([record["iter_test"] for record in records]).reshape(11, 16)
        fresh_phase = classify(train, validation)
        reference_phase = classify(reference_train, reference_test)
        # Check that the plotting code assigns the same labels.
        from plot_fig6b import classify as plotting_classify
        if not np.array_equal(fresh_phase, plotting_classify(train, validation, 100000).T):
            errors.append("Independent phase classification differs from plot_fig6b.classify")
        confusion = np.zeros((4, 4), dtype=np.int64)
        np.add.at(confusion, (reference_phase.ravel(), fresh_phase.ravel()), 1)
        cells = []
        for index, ((lr, wd), record) in enumerate(zip(coordinates, records)):
            i, j = divmod(index, 16)
            cells.append(dict(
                weight_decay=wd, decoder_lr=lr,
                iter_train=record["iter_train"], iter_test=record["iter_test"],
                reached_train=record["reached_train"], reached_test=record["reached_test"],
                final_acc_train=record["final_acc_train"], final_acc_test=record["final_acc_test"],
                val_below_last10000=record["val_below_last10000"],
                reference_iter_train=int(reference_train[i, j]), reference_iter_test=int(reference_test[i, j]),
                phase=PHASES[fresh_phase[i, j]], reference_phase=PHASES[reference_phase[i, j]],
                agrees=bool(fresh_phase[i, j] == reference_phase[i, j])))
        comparison = dict(
            phase_order=PHASES, shape=[11, 16],
            fresh_counts=np.bincount(fresh_phase.ravel(), minlength=4).tolist(),
            reference_counts=np.bincount(reference_phase.ravel(), minlength=4).tolist(),
            matching_cells=int((fresh_phase == reference_phase).sum()), total_cells=176,
            agreement_fraction=float((fresh_phase == reference_phase).mean()),
            confusion_matrix_reference_rows_fresh_columns=confusion.tolist(),
            fresh_phase_wd_by_lr=fresh_phase.tolist(), reference_phase_wd_by_lr=reference_phase.tolist(),
            exact_train_crossings=int((train == reference_train).sum()),
            exact_test_crossings=int((validation == reference_test).sum()),
            never_reached_train=sum(not record["reached_train"] for record in records),
            never_reached_test=sum(not record["reached_test"] for record in records),
            exact_delay_1000_count=int(((train < 99999) & (validation < 99999) & (validation - train == 1000)).sum()),
            validation_only_fallthrough_count=int(((train == 99999) & (validation < 99999)).sum()),
            mismatches=[cell for cell in cells if not cell["agrees"]], cells=cells)

    report = dict(
        passed=not errors, complete=True, results_directory=str(resdir),
        config_id=manifest.get("config_id"), recomputed_config_id=recomputed_id,
        manifest_sha256=sha256(manifest_path), counts=counts,
        sampled_steps_per_cell=1001, expected_sampled_steps=expected_steps.tolist(),
        source_integrity=sources,
        last_step_reached_sentinel_ambiguities=ambiguities,
        phase_comparison=comparison, errors=errors,
        limits=[
            "Accuracy was saved every 100 steps, so these curves can miss brief changes between samples.",
            "The last-10000-step counter is checked against the saved accuracy samples and first crossing times.",
            "A first crossing at the last step uses the same number as never reaching 90%; reached flags let us detect this.",
        ],
        elapsed_seconds=time.monotonic() - start)
    summary = [
        f"passed: {report['passed']}",
        f"saved results: {counts['records']} records, {counts['crossings']} crossing files, {counts['curves']} curves",
        "curve samples per cell: 1001 (step 0, every 100 steps, and step 99999)",
        f"config_id: {manifest.get('config_id')}",
        "Authors' notebook and H5 hashes checked against the saved manifest.",
        "Saved runner and adapter hashes identify the original scripts; those files were not checked.",
        f"actual last-step crossings sharing sentinel 99999: {len(ambiguities)}",
    ]
    if comparison is not None:
        summary += [
            "phase order: comprehension, grokking, memorization, confusion",
            f"our classification counts: {comparison['fresh_counts']}",
            f"reference phase counts: {comparison['reference_counts']}",
            f"matching classifications: {comparison['matching_cells']}/176 = {100 * comparison['agreement_fraction']:.4f}%",
            "confusion matrix (reference rows, our columns): "
            + json.dumps(comparison['confusion_matrix_reference_rows_fresh_columns']),
            f"exact crossing times: train {comparison['exact_train_crossings']}/176, validation {comparison['exact_test_crossings']}/176",
        ]
    else:
        summary.append("Phase comparison withheld because numerical or integrity validation failed.")
    if errors:
        summary += ["errors:"] + ["  " + error for error in errors]
    summary += ["limits:"] + ["  " + limit for limit in report["limits"]]
    text = "\n".join(summary) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results_validation.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
    (args.outdir / "results_validation.txt").write_text(text)
    print(text, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
