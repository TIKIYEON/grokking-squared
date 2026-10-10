import os
import sys

import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toy")
    ),
)

from produce_figure_data import HiddenPrints
from matplotlib import pyplot as plt
from train_add import train_add

train_nums = np.arange(1, 19) * 3
seeds = 100
probs = []

for train_num in train_nums:
    success_count = 0
    for seed in np.arange(seeds):
        with HiddenPrints():
            dic = train_add(train_num=train_num, seed=seed, steps=1, eff_steps=1)
        success_count += dic["dof"] == 2
    probs.append(success_count / seeds)
    print(train_num, probs[-1], flush=True)

fontsize = 15
plt.plot(train_nums / dic["all_num"], probs)
plt.xlabel("training data fraction", fontsize=fontsize)
plt.ylabel("Probability(linear structure)", fontsize=fontsize)
plt.title("Figure 4 (a)", fontsize=fontsize)
plt.savefig("fig4a_repro.png", dpi=150)