# Reproducing Figure 3 of Liu et al. (2022), "Towards Understanding Grokking"

Replication track: the authors' released code ([ejmichaud/grokking-squared](https://github.com/ejmichaud/grokking-squared),
commit `0229df9`, which this fork is based on) is used unmodified. `run_fig3.py` imports
`toy/train_add.py::train_add` from this repository and calls it with the same settings as `toy/produce_figure_data.py`
(p = 10, 1D embeddings, 3-layer tanh MLP decoder of width 200, MSE loss, AdamW with
η_repr = 1e-3 and η_dec = 1e-4, 5000 steps). Everything runs on CPU and takes about 8 s per run
on an M2.

## Files

- `run_fig3.py` runs `train_add` for every (train size, seed) combination in parallel and writes one
  CSV row per run. Options:
  - `--train-nums`, `--seeds`: lists like `5,10,15` or ranges like `3-54` (defaults: the authors'
    11 train sizes and seeds 0–2)
  - `--steps`: training steps per run (default 5000)
  - `--workers`: parallel processes (default: CPU count − 2)
  - `--set KEY=VALUE ...`: override any other `train_add` keyword, e.g. `threshold_P=1e-3`
  - `--out`: output CSV (default `results/replication.csv`)
- `plot_fig3.py CSV [--out PNG] [--title TEXT]` draws Fig. 3 (a)–(c) from a `run_fig3.py` CSV.
  The default output is `figures/fig3_<csv name>.png`.
- `plot_threshold_ablation.py` reads `results/ablation_threshold_*.csv` and writes
  `figures/ablation_threshold.png` and `results/ablation_threshold_summary.csv` (Ablation 2).
- `../toy/` holds the authors' code. `run_fig3.py` imports `train_add` from there.

## Setup

```bash
python3 -m venv .venv            # from the repo root
.venv/bin/pip install torch numpy matplotlib pandas
cd fig3_reproduction
```

## Runs

```bash
# 1) exact replication of the released script: 11 train sizes x 3 seeds (~40 s)
../.venv/bin/python run_fig3.py --out results/replication.csv
../.venv/bin/python plot_fig3.py results/replication.csv

# 2) denser sweep: train sizes 3..54 x 10 seeds (~10 min)
../.venv/bin/python run_fig3.py --train-nums 3-54 --seeds 0-9 --out results/dense.csv
../.venv/bin/python plot_fig3.py results/dense.csv
```

The ablations below use `--steps` and `--set KEY=VALUE`, which override any `train_add` keyword.

Columns in each CSV: `train_frac` (|D|/|D0|), `acc` (Acc, empirical accuracy on the full dataset),
`pred_acc` (Acc-hat, accuracy predicted from the learned representation), plus `train_acc`,
`test_acc`, `rqi` and `dof` (degrees of freedom of the representation left by the training set,
excluding translation and scaling).

## Notes

- The replication is bit-exact: all 33 (train_num, seed) runs give the same Acc and Acc-hat as the
  result files the authors shipped in `../toy/results/fig3/`.
- In the authors' `../toy/Figure3.ipynb`, the x-axis of panel (c) is labelled "training data fraction".
  It should be Acc-hat, which `plot_fig3.py` uses.
- At large train sizes, Acc falls below Acc-hat. Ablation 1 traces this to under-training.

## Ablation 1: training steps

In the dense sweep, Acc falls below Acc-hat for most runs with 47–54 training samples, and some of
those points even fall below the memorization line in Fig. 3a, which the paper does not show.
Hypothesis: Acc-hat assumes the decoder fits the training set perfectly, and with batch size 45 and
η_dec = 1e-4 the decoder has not finished fitting after 5000 steps. To test it, the same runs
(train sizes 45–54, seeds 0–9) are retrained for 20,000 steps, with everything else unchanged:

```bash
../.venv/bin/python run_fig3.py --train-nums 45-54 --seeds 0-9 --steps 20000 \
    --out results/ablation_steps_20k.csv
```

| steps (train sizes 45–54, 100 runs) | train acc < 100% | Acc < Acc-hat | below memorization line | MAE  | mean Acc |
|-------------------------------------|------------------|---------------|-------------------------|------|----------|
| 5,000 (released script)             | 81               | 80            | 22                      | 0.037 | 0.949   |
| 20,000                              | 0                | 1             | 0                       | 0.0002 | 0.986  |

- The hypothesis holds: with enough training, every run fits its training set, and Acc and Acc-hat
  agree almost exactly.
- The deviation at high training fractions is therefore an artefact of the 5000-step budget in the
  released script, not a failure of the representation-based prediction.
- Acc-hat describes a decoder that has fully fit the training set. Before that point, Acc can fall
  short of it, so Fig. 3 implicitly assumes training has converged.

## Ablation 2: parallelogram tolerance `threshold_P`

Acc-hat counts a held-out pair as predictable if it forms a parallelogram with a training pair,
i.e. `mean((E_i + E_j − E_m − E_n)^2) < threshold_P` on the normalized embeddings. The paper writes
this tolerance as δ and says it "can be taken to be zero". The code uses 0.01. The sweep covers
`threshold_P` from 1e-5 to 1 (52 train sizes x 3 seeds each):

```bash
for t in 1e-5 1e-4 1e-3 1e-2 1e-1 1; do
  ../.venv/bin/python run_fig3.py --train-nums 3-54 --seeds 0-2 --set threshold_P=$t \
      --out results/ablation_threshold_$t.csv
done
../.venv/bin/python plot_threshold_ablation.py   # -> figures/ablation_threshold.png
```

| threshold_P | r(Acc, Acc-hat) | MAE  | Acc > Acc-hat | Acc < Acc-hat |
|-------------|-----------------|------|---------------|---------------|
| 1e-5        | 0.967           | 0.070 | 74%          | 4%            |
| 1e-4        | 0.985           | 0.052 | 71%          | 5%            |
| 1e-3        | 0.993           | 0.031 | 53%          | 11%           |
| 1e-2 (default) | 0.996        | 0.017 | 30%          | 19%           |
| 1e-1        | 0.995           | 0.018 | 6%           | 42%           |
| 1           | 0.978           | 0.054 | 0%           | 64%           |

- Acc itself is identical across thresholds (the tolerance does not enter training), so only Acc-hat
  changes.
- Correlation stays high (r ≥ 0.97) at every threshold, so the qualitative claim (structure in the
  representation predicts generalization) is robust.
- The tight agreement in Fig. 3c depends on the tolerance. Below the default, Acc-hat
  underestimates Acc: approximate parallelograms that the decoder already exploits are not counted.
  Above it, Acc-hat overestimates Acc: parallelograms the decoder cannot actually use are counted.
  The default 1e-2 is about where the two errors balance. With δ = 0, as the paper's text suggests,
  Acc-hat would be systematically too low.
- Part of the "Acc < Acc-hat" share at the default comes from the under-training at large train
  sizes (Ablation 1).
