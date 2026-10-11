import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lambda3 import P, build_A, train_set

HERE = Path(__file__).resolve().parent
TOY_DIR = HERE.parent / "toy"


def initial_embedding(seed, p=P):
    """Embedding train_add initialises for `seed`, before any update.

    train_add calls torch.manual_seed(seed) and makes no other torch random call before
    `torch.rand(p, 1) - 1/2`, so the initial embedding depends on the seed only, not on train_num.
    """
    import torch

    torch.manual_seed(seed)
    return (torch.rand(p, 1) - 1 / 2).numpy()[:, 0].astype(np.float64)


def parallelograms(p=P):
    """train_add's list of all permissible parallelograms [i, n, j, m] with E_i + E_n = E_j + E_m."""
    return np.array([[i, n, j, m] for i in range(p) for j in range(i + 1, p) for m in range(j, p) for n in range(m + 1, p) if i + n - j - m == 0])


def rqi(E, par, threshold_P=0.01):
    """train_add's RQI for a batch of 1D embeddings E (K, p): normalize with mean and std (ddof=1), then count
    parallelograms with squared error < threshold_P."""
    En = (E - E.mean(1, keepdims=True)) / E.std(1, ddof=1, keepdims=True)
    d = En[:, par[:, 0]] + En[:, par[:, 1]] - En[:, par[:, 2]] - En[:, par[:, 3]]
    return (d ** 2 < threshold_P).mean(1)


def check_rqi_definition(train_num=50, seed=2, steps=1500):
    """Our numpy RQI must reproduce the authors' train_add per-step RQI on a real trajectory."""
    import contextlib
    import io

    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(TOY_DIR))
    from train_add import train_add

    with contextlib.redirect_stdout(io.StringIO()):
        dic = train_add(train_num=train_num, seed=seed, steps=steps, eff_steps=1)
    ours = rqi(dic["repr_nn"][:, :, 0].astype(np.float64), parallelograms())
    mismatch = int((np.abs(ours - dic["rqi"]) > 1e-12).sum())
    print(f"RQI definition check (train_num={train_num}, seed={seed}, {steps} steps): {mismatch} mismatching steps")
    return mismatch


def simulate(H, E0, opt, eta, max_steps, threshold_rqi=0.95, beta1=0.9, beta2=0.999, eps=1e-8):
    """Descend l_eff = l0 / Z0 with l0 = E^T H E (H = A^T A / |P0(D)|) and Z0 = sum_i E_i^2, for a batch of runs.

    Same effective loss as the authors' Task 3 in train_add (there over the full P0), here over P0(D).
    Returns (steps to RQI > threshold, counted as updates; 0 where never reached) and the final RQI.
    """
    par = parallelograms(H.shape[1])
    E = E0.copy()
    m, v = np.zeros_like(E), np.zeros_like(E)
    step_hit = np.zeros(len(E), dtype=int)
    active = np.ones(len(E), dtype=bool)
    final = np.zeros(len(E))
    for t in range(1, max_steps + 1):
        a = np.flatnonzero(active)
        Ea, Ha = E[a], H[a]
        HE = np.einsum("kij,kj->ki", Ha, Ea)
        l0 = (Ea * HE).sum(1, keepdims=True)
        Z0 = (Ea ** 2).sum(1, keepdims=True)
        grad = 2 * HE / Z0 - 2 * l0 * Ea / Z0 ** 2
        if opt == "gd":
            E[a] = Ea - eta * grad
        else:
            m[a] = beta1 * m[a] + (1 - beta1) * grad
            v[a] = beta2 * v[a] + (1 - beta2) * grad ** 2
            E[a] = Ea - eta * (m[a] / (1 - beta1 ** t)) / (np.sqrt(v[a] / (1 - beta2 ** t)) + eps)
        r = rqi(E[a], par)
        final[a] = r
        hit = a[r > threshold_rqi]
        step_hit[hit] = t
        active[hit] = False
        if not active.any(): break
    return step_hit, final


def main():
    ap = argparse.ArgumentParser(description="Effective-loss dynamics (no network), GD vs Adam, on the training sets of results/runs.csv")
    ap.add_argument("--runs", type=Path, default=HERE / "results" / "runs.csv")
    ap.add_argument("--opts", default="gd,adam")
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "effective_dynamics.csv")
    args = ap.parse_args()

    assert check_rqi_definition() == 0, "numpy RQI does not reproduce train_add's RQI"

    runs = pd.read_csv(args.runs)
    runs = runs[runs["lambda3"] > 0].reset_index(drop=True)
    H, E0 = [], []
    for n, s in zip(runs["train_num"], runs["seed"]):
        A = build_A(train_set(n, s))
        H.append(A.T @ A / len(A))
        E0.append(initial_embedding(s))
    H, E0 = np.array(H), np.array(E0)

    done = set(pd.read_csv(args.out)["opt"]) if args.out.exists() else set()
    for opt in args.opts.split(","):
        if opt in done:
            print(f"{opt}: already in {args.out}, skipped")
            continue
        t0 = time.time()
        step_hit, final = simulate(H, E0, opt, args.eta, args.max_steps)
        secs = time.time() - t0
        rows = [{"opt": opt, "eta": args.eta, "max_steps": args.max_steps, "train_num": n, "seed": s, "lambda3": lam,
                 "z0_init": float((e0 ** 2).sum()), "reached": int(k > 0), "rqi_step": k if k > 0 else "", "final_rqi": fr}
                for n, s, lam, e0, k, fr in zip(runs["train_num"], runs["seed"], runs["lambda3"], E0, step_hit, final)]
        new = not args.out.exists()
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new: w.writeheader()
            w.writerows(rows)
        hit = step_hit > 0
        x, y = np.log10(runs["lambda3"].to_numpy()[hit]), np.log10(step_hit[hit])
        print(f"{opt}: {hit.sum()}/{len(hit)} reached RQI > 0.95, slope {np.polyfit(x, y, 1)[0]:.3f}, "
              f"median t*2*lambda3*eta {np.median(step_hit[hit] * 2 * runs['lambda3'].to_numpy()[hit] * args.eta):.3f} ({secs:.0f}s)")


if __name__ == "__main__":
    main()
