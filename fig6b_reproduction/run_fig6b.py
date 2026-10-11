#!/usr/bin/env python
"""Train all 176 settings for Figure 6(b).

Read the learning rates and weight decays from the authors' H5 file.
Save new runs separately from the results submitted with the report.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from multiprocessing import get_context

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT
REFERENCE = UPSTREAM / "phase_diagram_plot/data_add_r.h5"
SOURCE_COMMIT = "0229df94de69b8384e560367280a43a238112bf5"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid():
    with h5py.File(REFERENCE, "r") as h5:
        rates = np.asarray(h5["learning_rate"], dtype=float)
        decays = np.asarray(h5["weight_decay"], dtype=float)
    if rates.shape != (16,) or not np.allclose(rates, 10 ** np.linspace(-5, -2, 16), rtol=1e-14, atol=0):
        raise ValueError("Unexpected published learning-rate grid")
    if not np.array_equal(decays, np.arange(11)):
        raise ValueError("Unexpected published weight-decay grid")
    if len({tag(lr, wd) for lr in rates for wd in decays}) != 176:
        raise ValueError("Rounded output filenames collide")
    return rates, decays


def tag(lr, wd):
    return f"eta_{lr:.5f}_wd_{wd:.5f}"


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def worker(job):
    lr, wd, config, output = job
    import torch
    from notebook_source import train_addition
    torch.set_num_threads(1)
    out = Path(output)
    name = tag(lr, wd)
    start = time.time()
    try:
        with (out / "run_logs" / f"{name}.log").open("w", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"Run lr={lr:.17g}, wd={wd:g}, steps={config['steps']}", flush=True)
                result = train_addition(
                    eta_dec=lr, weight_decay_dec=wd, steps=config["steps"],
                    curve_every=config["curve_every"], seed=config["seed"], log_freq=10000)
                print("Completed:", result["iter_train"], result["iter_test"], flush=True)
        curves = result.pop("curves")
        record = safe_json(result)
        record.update(status="ok", eta_dec=lr, weight_decay_dec=wd,
                      steps=config["steps"], seed=config["seed"],
                      curve_every=config["curve_every"], config_id=config["config_id"],
                      runtime_s=time.time() - start, host=socket.gethostname(),
                      pid=os.getpid(), torch_threads=torch.get_num_threads())
        for metric, key in (("train", "iter_train"), ("test", "iter_test")):
            target = out / f"{metric}step_{name}.txt"
            tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
            np.savetxt(tmp, np.array([record[key]], dtype=np.int64))
            os.replace(tmp, target)
        target = out / "curves" / f"curves_{name}.npz"
        tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **curves)
        os.replace(tmp, target)
        # Save this last so an interrupted run will be retried.
        atomic_json(out / "records" / f"{name}.json", record)
        return record
    except Exception:
        record = dict(status="failed", eta_dec=lr, weight_decay_dec=wd,
                      runtime_s=time.time() - start, config_id=config["config_id"],
                      traceback=traceback.format_exc())
        atomic_json(out / "failures" / f"{name}.json", record)
        return record


def completed_record(out, lr, wd, config):
    name = tag(lr, wd)
    record_path = out / "records" / f"{name}.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    expected = dict(status="ok", eta_dec=lr, weight_decay_dec=wd,
                    config_id=config["config_id"], steps=config["steps"])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible completed run: {record_path}")
    for metric, key in (("train", "iter_train"), ("test", "iter_test")):
        path = out / f"{metric}step_{name}.txt"
        if not path.exists() or float(np.loadtxt(path)) != record[key]:
            raise ValueError(f"Missing/inconsistent completed output: {path}")
    path = out / "curves" / f"curves_{name}.npz"
    with np.load(path) as data:
        if not {"step", "acc_train", "acc_test", "loss_train"}.issubset(data.files):
            raise ValueError(f"Incomplete curve archive: {path}")
        if not len(data["step"]) or any(len(data[k]) != len(data["step"]) for k in data.files):
            raise ValueError(f"Invalid curve lengths: {path}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--curve-every", type=int, default=100)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs/fig6b")
    parser.add_argument("--limit", type=int, help="Run only this many unfinished settings")
    args = parser.parse_args()
    if min(args.steps, args.workers, args.curve_every) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Steps, workers, curve-every and limit must be positive")
    out = args.outdir.resolve()
    for protected in (ROOT / "phase_diagram_plot", HERE / "results", HERE / "checks"):
        if out == protected.resolve() or protected.resolve() in out.parents:
            parser.error(f"Original files are read-only: choose an output outside {protected}")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".sweep.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another sweep already owns this output directory")
    for name in ("records", "curves", "run_logs", "failures"):
        (out / name).mkdir(exist_ok=True)

    import torch
    from notebook_source import source_provenance
    rates, decays = grid()
    config = dict(format_version=2, steps=args.steps, seed=args.seed,
                  curve_every=args.curve_every, eta_decs=rates.tolist(), wds=decays.tolist(),
                  torch_threads=1, torch_version=torch.__version__, numpy_version=np.__version__,
                  python=platform.python_version(), source=source_provenance(),
                  adapter_sha256=sha256(HERE / "notebook_source.py"))
    config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_id") != config["config_id"]:
            parser.error("Directory has a different configuration/source; use a new output directory")
    else:
        manifest = dict(
            **config, figure="Figure 6(b): addition regression",
            data_origin="fresh_training", upstream_commit=SOURCE_COMMIT,
            grid_reference=str(REFERENCE.relative_to(ROOT)), grid_reference_sha256=sha256(REFERENCE),
            n_runs=176, workers=args.workers, host=socket.gethostname(), platform=platform.platform(),
            started_unix=time.time(), runner_sha256=sha256(__file__),
            eta_repr=0.001, weight_decay_repr=0.0, train_num=45, test_num=10,
            p=10, repr_dim=1, output_dim=30, width=200, batch_size=45,
            batch_order="fixed original train_id order",
            threshold_train_acc=0.9, threshold_test_acc=0.9, delay_threshold=1000,
            evaluation="post-update for both training and validation", early_stopping=False)
        atomic_json(manifest_path, manifest)
    records, jobs = [], []
    for wd in decays:
        for lr in rates:
            record = completed_record(out, float(lr), float(wd), config)
            if record is None:
                jobs.append((float(lr), float(wd), config, str(out)))
            else:
                records.append(record)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    start = time.time()
    failures = []
    print(f"Figure 6(b): 176 settings x {args.steps} steps; "
          f"{len(records)} complete, {len(jobs)} queued, {args.workers} workers", flush=True)
    with (out / "runs.jsonl").open("a", buffering=1) as log:
        with get_context("spawn").Pool(args.workers) as pool:
            for record in pool.imap_unordered(worker, jobs):
                log.write(json.dumps(record, allow_nan=False) + "\n")
                (records if record["status"] == "ok" else failures).append(record)
                progress = dict(completed=len(records), expected=176, complete=len(records) == 176,
                                failed_this_invocation=len(failures), config_id=config["config_id"],
                                elapsed_this_invocation_s=time.time() - start, updated_unix=time.time())
                atomic_json(out / "progress.json", progress)
                print(f"[{len(records):3d}/176] {record['status']} lr={record['eta_dec']:.8g} "
                      f"wd={record['weight_decay_dec']:g} train={record.get('iter_train')} "
                      f"val={record.get('iter_test')} runtime={record['runtime_s']:.1f}s", flush=True)
                if record["status"] != "ok":
                    print(record["traceback"], flush=True)
    progress = dict(completed=len(records), expected=176, complete=len(records) == 176,
                    failed_this_invocation=len(failures), config_id=config["config_id"],
                    elapsed_this_invocation_s=time.time() - start, updated_unix=time.time())
    atomic_json(out / "progress.json", progress)
    print(json.dumps(progress), flush=True)
    if failures:
        raise SystemExit(1)
    if len(records) == 176:
        manifest.update(completed_unix=time.time(), successful_runs=176)
        atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
