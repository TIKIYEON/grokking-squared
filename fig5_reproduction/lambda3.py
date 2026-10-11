import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# defaults of toy/train_add.py::train_add
P = 10
OUTPUT_DIM = 30


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


def train_set(train_num, seed, p=P, output_dim=OUTPUT_DIM):
    """Indices (into D0) of the training set that train_add(train_num=..., seed=...) draws.

    Replays train_add's numpy RNG calls: np.random.seed(seed), the y_templates draw, then
    np.random.choice. RandomState(seed) is the generator behind the legacy np.random.* functions.
    """
    rng = np.random.RandomState(seed)
    rng.normal(0, 1, size=(2 * p - 1, output_dim))
    return rng.choice(p * (p + 1) // 2, train_num, replace=False)


def build_A(train_id, p=P):
    """One row per parallelogram in P0(D): E_i + E_j - E_m - E_n = 0 for (i,j),(m,n) in D, i+j = m+n."""
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    D = [pairs[k] for k in sorted(train_id)]
    rows = []
    for a in range(len(D)):
        for b in range(a + 1, len(D)):
            (i, j), (m, n) = D[a], D[b]
            if i + j == m + n:
                r = np.zeros(p)
                r[i] += 1; r[j] += 1; r[m] -= 1; r[n] -= 1
                rows.append(r)
    return np.array(rows).reshape(-1, p)


def effective_theory(train_id, p=P, tol=1e-10):
    """lambda3, dof and |P0(D)| of a training set.

    A has one row per parallelogram in P0(D), so l0 = |A E|^2 / |P0(D)|. lambda3 is the 3rd-smallest
    eigenvalue of H = A^T A / |P0(D)|; the first two are 0 (translation and scaling of a linear
    representation). The paper does not spell the normalisation out; this convention is inferred:
    - on the full dataset (all 55 pairs, |P0| = 70) it gives 30/70 = 0.4286, the top of the paper's Fig. 5a (~0.42);
    - gradient descent on l0, dE/dt = -eta dl0/dE = -2 eta H E, relaxes the slowest non-trivial mode as
      exp(-2 lambda3 eta t), which is the paper's theory line t = 1/(2 lambda3 eta).

    dof follows train_add: nullity of A^T A minus translation and scaling.
    """
    A = build_A(train_id, p)
    if len(A) == 0:
        return {"lambda3": 0.0, "dof": p - 2, "n_parallelograms": 0}
    gram = A.T @ A
    dof = int((np.linalg.eigvalsh(gram) < 1e-8).sum()) - 2
    lam3 = float(np.linalg.eigvalsh(gram / len(A))[2])
    return {"lambda3": lam3 if lam3 > tol else 0.0, "dof": dof, "n_parallelograms": len(A)}


def main():
    ap = argparse.ArgumentParser(description="Fig. 5(a): lambda3 of random training sets, numpy only")
    ap.add_argument("--train-nums", type=parse_int_list, default=list(range(1, 55)))
    ap.add_argument("--seeds", type=parse_int_list, default=list(range(200)), help="one training set per seed, drawn as in train_add")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "lambda3.csv")
    args = ap.parse_args()

    all_num = P * (P + 1) // 2
    rows = [{"train_num": n, "seed": s, "train_frac": n / all_num, **effective_theory(train_set(n, s))}
            for n in args.train_nums for s in args.seeds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
