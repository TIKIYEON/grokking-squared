#!/usr/bin/env python
"""Check that skipping the RQI calculation leaves training results unchanged."""

import sys

sys.dont_write_bytecode = True

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from notebook_source import NOTEBOOK, _run, source_provenance, train_addition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "runs/source_check")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    notebook_hash_before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    log = io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        reference = _run(
            1e-4, 10, 100, 1, 58, 10000, include_rqi=True, capture_state=True
        )
        adapted = _run(
            1e-4, 10, 100, 1, 58, 10000, include_rqi=False, capture_state=True
        )
        control = train_addition(1e-4, 0, steps=2500, curve_every=100)

    reference_state, adapted_state = reference.pop("_state"), adapted.pop("_state")
    model_checks = {
        key: torch.equal(value, adapted_state["model"][key])
        for key, value in reference_state["model"].items()
    }
    curve_checks = {
        key: np.array_equal(values, adapted["curves"][key])
        for key, values in reference["curves"].items()
    }
    metric_keys = [
        "iter_train", "iter_test", "reached_train", "reached_test",
        "final_acc_train", "final_acc_test", "final_loss_train", "val_below_last10000",
    ]
    metric_checks = {key: reference[key] == adapted[key] for key in metric_keys}

    data_path = NOTEBOOK.parent / "data_add_r.h5"
    with h5py.File(data_path, "r") as data:
        lr_index = int(np.flatnonzero(np.isclose(data["learning_rate"][:], 1e-4, rtol=1e-12, atol=0))[0])
        wd_index = int(np.flatnonzero(data["weight_decay"][:] == 0)[0])
        expected_train = int(data["train_steps"][lr_index, wd_index])
        expected_test = int(data["test_steps"][lr_index, wd_index])

    checks = {
        "model_tensors": model_checks,
        "latent_bitwise_equal": torch.equal(reference_state["latent"], adapted_state["latent"]),
        "targets_bitwise_equal": torch.equal(reference_state["y_templates"], adapted_state["y_templates"]),
        "train_ids_equal": np.array_equal(reference_state["train_id"], adapted_state["train_id"]),
        "test_ids_equal": np.array_equal(reference_state["test_id"], adapted_state["test_id"]),
        "curves": curve_checks,
        "metrics": metric_checks,
        "control_crossings_equal_hdf5": (
            control["iter_train"] == expected_train and control["iter_test"] == expected_test
        ),
        "notebook_unchanged": notebook_hash_before == hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    passed = (
        all(model_checks.values())
        and all(curve_checks.values())
        and all(metric_checks.values())
        and all(value for value in checks.values() if isinstance(value, bool))
    )
    report = {
        "passed": passed,
        "command": " ".join(sys.argv),
        "provenance": source_provenance(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "reference_comparison": {
            "steps": 100,
            "eta_dec": 1e-4,
            "weight_decay_dec": 10,
            "model_tensors_equal": sum(model_checks.values()),
            "model_tensors_total": len(model_checks),
            "accuracy_entries_equal": int(sum(
                np.sum(reference["curves"][key] == adapted["curves"][key])
                for key in ("acc_train", "acc_test")
            )),
            "accuracy_entries_total": 200,
        },
        "control": {
            "steps": 2500,
            "eta_dec": 1e-4,
            "weight_decay_dec": 0,
            "expected_iter_train": expected_train,
            "expected_iter_test": expected_test,
            **{key: control[key] for key in metric_keys},
        },
        "checks": checks,
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.outdir / "source_validation.log").write_text(log.getvalue())
    (args.outdir / "source_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.outdir / "source_control_curves.npz", **control["curves"])
    summary = (
        f"passed: {passed}\n"
        f"Original RQI loop versus production adapter, 100 steps:\n"
        f"  model tensors bitwise equal: {sum(model_checks.values())}/{len(model_checks)}\n"
        f"  latent bitwise equal: {checks['latent_bitwise_equal']}\n"
        f"  accuracy entries exact: {report['reference_comparison']['accuracy_entries_equal']}/200\n"
        f"  all curve arrays exact: {all(curve_checks.values())}\n"
        f"2500-step control (lr=1e-4, wd=0):\n"
        f"  measured crossings: {control['iter_train']}, {control['iter_test']}\n"
        f"  released HDF5 crossings: {expected_train}, {expected_test}\n"
        f"notebook unchanged: {checks['notebook_unchanged']}\n"
    )
    (args.outdir / "source_validation.txt").write_text(summary)
    print(summary, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
