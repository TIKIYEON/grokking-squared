import argparse
import ast
import csv
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOY_DIR = HERE.parent / "toy"

# defaults from toy/produce_figure_data.py
DEFAULT_TRAIN_NUMS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 54]
DEFAULT_SEEDS = [0, 1, 2]


def use_authors_code():
    sys.path.insert(0, str(TOY_DIR))


def parse_int_list(s):
    """'5,10,15' or '3-54' or '0-9,20' -> list of ints."""
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_overrides(items):
    overrides = {}
    for item in items:
        key, val = item.split("=", 1)
        try: overrides[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError): overrides[key] = val
    return overrides


def run_one(job):
    train_num, seed, steps, overrides = job
    import torch

    torch.set_num_threads(1)
    use_authors_code()
    from train_add import train_add

    t0 = time.time()
    with open(os.devnull, "w") as devnull:
        stdout, sys.stdout = sys.stdout, devnull
        try: dic = train_add(train_num=train_num, seed=seed, steps=steps, eff_steps=1, **overrides)
        finally: sys.stdout = stdout
    row = {
        "train_num": train_num,
        "seed": seed,
        "train_frac": dic["train_ratio"],
        "acc": dic["acc"][-1],
        "pred_acc": dic["pred_acc"],
        "ideal_acc": dic["ideal_acc"],
        "train_acc": dic["acc_train"][-1],
        "test_acc": dic["acc_test"][-1],
        "rqi": dic["rqi"][-1],
        "dof": int(dic["dof"]) - 2,
    }
    print(f"train_num={train_num:2d} seed={seed:2d}  Acc={row['acc']:.3f} Acc_hat={row['pred_acc']:.3f}  ({time.time() - t0:.1f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-nums", type=parse_int_list, default=DEFAULT_TRAIN_NUMS)
    ap.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override train_add keyword arguments (ablations)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "replication.csv")
    args = ap.parse_args()

    use_authors_code()
    overrides = parse_overrides(args.set)
    jobs = [(n, s, args.steps, overrides) for n in args.train_nums for s in args.seeds]
    print(f"{len(jobs)} runs on {args.workers} workers, overrides={overrides}")

    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
