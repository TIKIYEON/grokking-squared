import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOY_DIR = HERE.parent / "toy"
sys.path.insert(0, str(HERE))
from lambda3 import parse_int_list  # noqa: E402

# CPU only, one thread per worker. Set before the pool spawns so workers inherit it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

# dense sweep: train sizes 20, 24, ..., 52, 54 (fractions 0.36 ... 0.98) x seeds 0-19
DEFAULT_SIZES = [*range(20, 53, 4), 54]
DEFAULT_SEEDS = list(range(20))
# columns that identify a run; a row with the same values already in --out is skipped (resume)
KEY = ["train_num", "seed", "init_seed", "repr_opt", "eta_repr", "eta_dec", "steps", "early_stop"]
COLUMNS = KEY + ["train_frac", "lambda3", "dof", "n_parallelograms", "reached", "rqi_step", "final_rqi", "steps_run", "seconds"]


def use_authors_code():
    sys.path.insert(0, str(TOY_DIR))


def needs_mod(job):
    """The authors' toy/train_add.py is used unless an option only train_add_mod.py has is requested."""
    return job["repr_opt"] != "adamw" or job["init_seed"] is not None or bool(job["early_stop"])


def run_key(row):
    return tuple("" if row[k] in (None, "") else f"{float(row[k]):.6g}" if k.startswith("eta") else str(row[k]) for k in KEY)


def run_one(job):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from lambda3 import effective_theory, train_set

    kwargs = dict(train_num=job["train_num"], seed=job["seed"], steps=job["steps"], eff_steps=1,
                  eta_reprs=job["eta_repr"], eta_dec=job["eta_dec"], device="cpu")
    if needs_mod(job):
        from train_add_mod import train_add
        kwargs.update(repr_opt=job["repr_opt"], init_seed=job["init_seed"], early_stop=bool(job["early_stop"]))
    else:
        use_authors_code()
        from train_add import train_add

    t0 = time.time()
    with open(os.devnull, "w") as devnull:
        stdout, sys.stdout = sys.stdout, devnull
        try: dic = train_add(**kwargs)
        finally: sys.stdout = stdout

    theory = effective_theory(train_set(job["train_num"], job["seed"], p=dic["p"]), p=dic["p"])
    assert theory["dof"] == int(dic["dof"]) - 2, f"rebuilt training set disagrees with train_add ({job})"
    # first step with RQI > threshold, counted as number of updates; train_add's iter_rqi
    # falls back to steps-1 when never reached, so read it off the trajectory instead
    hit = np.flatnonzero(dic["rqi"] > dic["threshold_rqi"])
    return {
        **{k: ("" if job[k] is None else job[k]) for k in KEY},
        "train_frac": dic["train_ratio"],
        **theory,
        "reached": int(len(hit) > 0),
        "rqi_step": int(hit[0]) + 1 if len(hit) else "",
        "final_rqi": dic["rqi"][-1],
        "steps_run": len(dic["rqi"]),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Fig. 5(b) training runs, one per (size, seed); resumable")
    ap.add_argument("--sizes", type=parse_int_list, default=DEFAULT_SIZES, help="train sizes out of 55, e.g. 20,24,28 or 20-54")
    ap.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS, help="e.g. 0-19")
    ap.add_argument("--steps", type=int, default=10000, help="training steps (the cap, with --early-stop)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "runs.csv")
    ap.add_argument("--repr-opt", choices=["adamw", "sgd"], default="adamw", help="optimizer for the embeddings (the decoder keeps AdamW)")
    ap.add_argument("--eta-repr", type=float, default=1e-3)
    ap.add_argument("--eta-dec", type=float, default=1e-4)
    ap.add_argument("--init-seed", choices=["none", "size_seed"], default="none",
                    help="none: train_add's seeding (initial embedding depends on the seed only); size_seed: init_seed = 1000*size + seed")
    ap.add_argument("--early-stop", action="store_true", help="stop at the first step with RQI > 0.95")
    args = ap.parse_args()

    jobs = [{"train_num": n, "seed": s, "init_seed": None if args.init_seed == "none" else 1000 * n + s, "repr_opt": args.repr_opt,
             "eta_repr": args.eta_repr, "eta_dec": args.eta_dec, "steps": args.steps, "early_stop": int(args.early_stop)}
            for n in args.sizes for s in args.seeds]
    done = set()
    if args.out.exists():
        with open(args.out, newline="") as f:
            done = {run_key(r) for r in csv.DictReader(f)}
    todo = [j for j in jobs if run_key(j) not in done]
    code = "train_add_mod.py" if needs_mod(jobs[0]) else "the authors' toy/train_add.py (unmodified)"
    print(f"{len(jobs)} runs with {code}, {len(jobs) - len(todo)} already in {args.out}, {len(todo)} to do on {args.workers} workers", flush=True)
    if not todo: return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool, open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0: writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            writer.writerow(row)
            f.flush()
            status = f"step {row['rqi_step']}" if row["reached"] else f"not reached (RQI {row['final_rqi']:.2f})"
            print(f"[{i}/{len(todo)} {(time.time() - t0) / 60:.1f} min] n={row['train_num']} seed={row['seed']} "
                  f"lambda3={row['lambda3']:.3f} {status} ({row['seconds']}s)", flush=True)
    print(f"saved {len(todo)} new rows to {args.out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
