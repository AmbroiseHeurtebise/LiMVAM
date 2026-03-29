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
results_dir = simulation_dir / "results/results_cross_view_correlations_decrease"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

fig, ax = plt.subplots(figsize=(9, 5))

# plot
sns.lineplot(
    data=df, x="cross_view_correlations_alpha", y="error_B", linewidth=2.5,
    estimator=np.median, errorbar=('ci', 95), label="DirectLiMVAM"
)

# ax.set_yscale("log")
ax.grid(linewidth=0.5, alpha=0.5)
ax.set_xlabel("Level of cross-view correlations")
ax.set_ylabel(r"Error on $B^i$")

# caption
caption = (
    "Cross-view correlation level is 0 when views are uncorrelated,\n"
    "and 1 when cross-view correlations are drawn from a product\n"
    "of standard Gaussians."
)
fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)


# save figure
figures_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/figures")
plt.savefig(figures_dir / f"simulation_cross_view_correlations_decrease.pdf", bbox_inches="tight")
