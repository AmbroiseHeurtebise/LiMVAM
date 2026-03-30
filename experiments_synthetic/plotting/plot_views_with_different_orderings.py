import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# matplotlib style
fontsize = 20
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 100
ci = 95
metric = "success_P"
if metric == "error_B":
    estimator = np.median
elif metric in ["error_P_exact", "error_P_spearmanr", "success_P"]:
    estimator = np.mean

# read dataframe
simulation_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic")
results_dir = simulation_dir / "results/results_views_with_different_orderings"
save_name = f"DataFrame_with_{nb_seeds}_seeds_new_error"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# add a column
df["success_P"] = 1 - df["error_P_exact"]

fig, ax = plt.subplots(figsize=(9, 5))

# plot
sns.lineplot(
    data=df, x="n_views_different_orderings", y=metric, linewidth=2.5,
    estimator=estimator, errorbar=('ci', ci), label="DirectLiMVAM"
)

# ax.set_yscale("log")
if metric == "error_B":
    eps = 0.01
    ax.set_ylim(-eps, 0.3 + eps)
    ax.set_yticks([0, 0.3])
    ax.set_yticks(np.arange(0, 0.4, 0.1), minor=True)
elif metric == "success_P":
    ax.set_yticks([0, 1])
    ax.set_yticks(np.arange(0, 1.2, 0.2), minor=True)
    ax.set_yticklabels(["0%", "100%"])
    
ax.grid(which="both", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Number of views with a different causal ordering")
if metric == "error_B":
    ax.set_ylabel(r"Error on $B^i$")
elif metric == "error_P_exact":
    ax.set_ylabel(r"Error on $P$")
elif metric == "error_P_spearmanr":
    ax.set_ylabel(r"Spearman-r correlation on $P$")
elif metric == "success_P":
    ax.set_ylabel(r"Success rate on $P$")

# caption
if metric == "success_P":
    caption = (
        r"Caption: Data are generated with $m=20$ views in total, and" + "\n"
        r"the experiment is ran 100 times. The $y$-axis represents the" + "\n"
        "percentage of runs where the ordering is fully recovered."
    )
else:
    caption = (
        "Caption: Performance of DirectLiMVAM when a subset of\n"
        "views has a different causal ordering. Data are generated\n"
        r"with $m=20$ views in total, and the experiment is ran" + "\n"
        "with 100 different seeds."
    )
fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)


# save figure
figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_views_with_different_orderings.pdf", bbox_inches="tight")
plt.savefig(figures_dir / f"simulation_views_with_different_orderings.png", bbox_inches="tight", dpi=300)
