"""Run the training function from Liu et al.'s notebook.

Keep the unused input templates and encoder initialization because they draw
random numbers and affect the model's starting values.
"""

import sys

sys.dont_write_bytecode = True

import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "phase_diagram_plot/phase_diagram.ipynb"
)
EXPECTED_SOURCE_SHA256 = (
    "849ba9b003624dca489a2bd20523148766bba268f670b056c669b479b90534fd"
)
TRANSFORMATIONS = [
    "Use code cells 0 and 1 with the original initialization and training calculations.",
    "Skip unused scipy imports, but keep the random input templates and unused encoder.",
    "Skip the RQI calculation unless requested for a check.",
    "Return results instead of writing to the notebook's output folder.",
    "Set how often to print progress and save accuracy and loss measurements.",
    "Use the requested seed and keep train_num=45 and wd1_mode=0.",
]


@lru_cache(maxsize=1)
def _source() -> tuple[str, dict[str, Any]]:
    raw = NOTEBOOK.read_bytes()
    notebook = json.loads(raw)
    if any(notebook["cells"][i]["cell_type"] != "code" for i in (0, 1)):
        raise RuntimeError("The notebook's first two cells changed.")
    source = "\n".join("".join(notebook["cells"][i]["source"]) for i in (0, 1))
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Notebook training code differs from the expected version: " + source_hash
        )
    provenance = {
        "source_path": str(NOTEBOOK.relative_to(NOTEBOOK.parents[1])),
        "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": source_hash,
        "source_cells": [0, 1],
        "transformations": TRANSFORMATIONS,
        "validation_accuracy_timing": "after both optimizer steps",
        "training_loss_timing": "before both optimizer steps",
        "threshold_train_acc": 0.9,
        "threshold_test_acc": 0.9,
        "weight_decay_reprs": 0.0,
        "eta_reprs": 0.001,
        "train_num": 45,
    }
    return source, provenance


def source_provenance() -> dict[str, Any]:
    """Record which notebook code and changes were used."""
    return json.loads(json.dumps(_source()[1]))


def _assignment_to(statement: ast.stmt, name: str) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    )


def _is_savetxt(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "np"
        and statement.value.func.attr == "savetxt"
    )


@lru_cache(maxsize=2)
def _compiled_train(include_rqi: bool):
    tree = ast.parse(_source()[0])
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("scipy") for alias in node.names)
        )
    ]
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    ]
    if len(functions) != 1:
        raise RuntimeError("Expected exactly one original train function.")
    function = functions[0]
    loop_indices = [
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "epoch"
    ]
    if len(loop_indices) != 1:
        raise RuntimeError("Expected exactly one original epoch loop.")
    loop_index = loop_indices[0]
    loop = function.body[loop_index]
    rqi_indices = [
        i for i, statement in enumerate(loop.body) if _assignment_to(statement, "latent_scale")
    ]
    if len(rqi_indices) != 1:
        raise RuntimeError("Could not identify the original RQI diagnostic block.")
    if not include_rqi:
        loop.body = loop.body[: rqi_indices[0]]

    # Record results after the original training and accuracy calculations.
    loop.body += ast.parse(
        """
if epoch % _audit_curve_every == 0 or epoch == epochs - 1:
    _audit_curves["step"].append(epoch)
    _audit_curves["acc_train"].append(float(acc_nn_train))
    _audit_curves["acc_test"].append(float(acc_nn_test))
    _audit_curves["loss_train"].append(float(loss_train.detach().cpu()))
if epoch >= max(0, epochs - 10000):
    _audit_val_below_last10000 += int(acc_nn_test < 0.9)
"""
    ).body

    save_indices = [i for i, statement in enumerate(function.body) if _is_savetxt(statement)]
    if len(save_indices) != 3 or save_indices[0] <= loop_index:
        raise RuntimeError("Expected the notebook's three output-file writes after training.")
    original_tail = function.body[loop_index + 1 : save_indices[0]]
    function.body = (
        function.body[:loop_index]
        + ast.parse(
            """
log = _audit_log_freq
_audit_curves = {"step": [], "acc_train": [], "acc_test": [], "loss_train": []}
_audit_val_below_last10000 = 0
"""
        ).body
        + [loop]
        + original_tail
        + ast.parse(
            """
_audit_result = {
    "iter_train": int(iter_train),
    "iter_test": int(iter_test),
    "reached_train": bool(reach_acc_train),
    "reached_test": bool(reach_acc_test),
    "final_acc_train": float(acc_nn_train),
    "final_acc_test": float(acc_nn_test),
    "final_loss_train": float(loss_train.detach().cpu()),
    "val_below_last10000": int(_audit_val_below_last10000),
    "curves": {
        key: np.asarray(values, dtype=np.int64 if key == "step" else np.float64)
        for key, values in _audit_curves.items()
    },
}
if _audit_capture_state:
    _audit_result["_state"] = {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "latent": latent.detach().cpu().clone(),
        "train_id": train_id.copy(),
        "test_id": test_id.copy(),
        "y_templates": y_templates.detach().cpu().clone(),
    }
return _audit_result
"""
        ).body
    )
    ast.fix_missing_locations(tree)
    return compile(tree, str(NOTEBOOK) + "[cells 0,1; adapted]", "exec")


def _run(
    eta_dec: float,
    weight_decay_dec: float,
    steps: int,
    curve_every: int,
    seed: int,
    log_freq: int,
    *,
    include_rqi: bool = False,
    capture_state: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(eta_dec) or eta_dec <= 0:
        raise ValueError("eta_dec must be finite and positive")
    if not math.isfinite(weight_decay_dec) or weight_decay_dec < 0:
        raise ValueError("weight_decay_dec must be finite and nonnegative")
    for name, value in (("steps", steps), ("curve_every", curve_every), ("log_freq", log_freq)):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    namespace: dict[str, Any] = {}
    exec(_compiled_train(include_rqi), namespace)
    namespace.update(
        seed=int(seed),
        _audit_curve_every=int(curve_every),
        _audit_log_freq=int(log_freq),
        _audit_capture_state=capture_state,
    )
    result = namespace["train"]([float(eta_dec), float(weight_decay_dec), int(steps)])
    result["provenance"] = source_provenance()
    return result


def train_addition(
    eta_dec: float,
    weight_decay_dec: float,
    steps: int = 100000,
    curve_every: int = 100,
    seed: int = 58,
    log_freq: int = 10000,
) -> dict[str, Any]:
    """Train the addition model on CPU.

    Check for 90% accuracy every step. If it is never reached, store steps - 1.
    Save curves every ``curve_every`` steps and at the final step. Training
    loss is measured before the update. The caller sets the CPU thread count.
    """
    return _run(eta_dec, weight_decay_dec, steps, curve_every, seed, log_freq)
