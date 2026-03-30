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

# read dataframe
simulation_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic")
results_dir = simulation_dir / "results/results_views_with_different_orderings"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

fig, ax = plt.subplots(figsize=(9, 5))

# plot
sns.lineplot(
    data=df, x="n_views_different_orderings", y="error_B", linewidth=2.5,
    estimator=np.median, errorbar=('ci', 95), label="DirectLiMVAM"
)

# ax.set_yscale("log")
eps = 0.01
ax.set_ylim(-eps, 0.3 + eps)
ax.set_yticks([0, 0.3])
ax.set_yticks(np.arange(0, 0.4, 0.1), minor=True)
ax.grid(which="both", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Number of views with a different causal ordering")
ax.set_ylabel(r"Error on $B^i$")

# caption
caption = (
    "Caption: Performance of DirectLiMVAM when a subset of\n"
    "views has a different causal ordering. Data are generated\n"
    r"with $m=20$ views in total."
)
fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)


# save figure
figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_views_with_different_orderings.pdf", bbox_inches="tight")
