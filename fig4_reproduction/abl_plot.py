import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
tns, seeds, N = [20, 35, 45], range(5), 55
fig, ax = plt.subplots(figsize=(6, 4.5))
for off, (d, lab, col) in enumerate([("results/fig4/", "eta=1e-3", "C0"), ("results/fig4_eta3/", "eta=3e-3", "C1")]):
    for k, t in enumerate(tns):
        y = np.array([float(np.loadtxt(f"{d}rqistep_num_{t}_seed_{s}.txt")) for s in seeds])
        x = k + (off - 0.5) * 0.25
        c = y < 9999
        ax.scatter([x]*c.sum(), y[c], color=col, label=lab if k == 0 else None)
        ax.scatter([x]*(~c).sum(), y[~c], marker="^", color="red")
        ax.hlines(np.median(y), x - 0.1, x + 0.1, color=col)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{t/N:.2f}" for t in tns])
ax.set_yscale("log"); ax.set_xlabel("training data fraction"); ax.set_ylabel("steps to RQI>0.95")
ax.legend(); plt.tight_layout(); plt.savefig("ablation_eta.png", dpi=150)

