import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path


# matplotlib and seaborn style
fontsize = 15
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
sns.set(style="white")  # "whitegrid"
plt.rcParams.update(rc)

# parameters 
nb_seeds = 50

# read dataframe
simulation_dir = Path("/storage/store2/work/aheurteb/LiMVAM/simulation_studies")
results_dir = simulation_dir / "results/results_timepoints_in_xaxis/shared_P"
save_name = f"DataFrame_with_{nb_seeds}_seeds_and_time"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# Remove MICaDo-MVICA curve, and keep only data for n=1000 and nb_gaussian_disturbances=2
df_filtered = df[
    (df["n"] == 1000) 
    & (df["nb_gaussian_disturbances"] == 2) 
    & (df["ica_algo"] != "multiviewica")]

# Use color palette and loop through methods
palette_sns = sns.color_palette()
palette = {
    'shica_ml': palette_sns[0],
    'shica_j': palette_sns[1],
    'lingam': palette_sns[2],
    'multi_group_direct_lingam': palette_sns[3],
    'pairwise': palette_sns[4],
    'mv_notears': palette_sns[5],
}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3.2))

for method, group_df in df_filtered.groupby("ica_algo"):
    sns.scatterplot(
        data=group_df,
        x="error_B",
        y="execution_time",
        color=palette.get(method, "gray"),
        alpha=0.2,
        label=None
    )
    sns.lineplot(
        data=group_df,
        x="error_B",
        y="execution_time",
        color=palette.get(method, "gray"),
        label=None,
        errorbar=None,
        estimator="median",
        sort=True,
        linewidth=2
    )

# legend
legend_styles = [
    Line2D([0], [0], color=palette_sns[0], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette_sns[1], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette_sns[2], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette_sns[3], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette_sns[4], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette_sns[5], linewidth=2.5, linestyle='-'),
]
labels = ['MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM', 'PairwiseLiMVAM', 'MV-NOTEARS']
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.12), loc="center",
    ncol=2, borderaxespad=0., fontsize=fontsize
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Error on $B^i$", fontsize=fontsize, family="serif")
plt.ylabel("Fitting time (in s)", fontsize=fontsize, family="serif")
ax.tick_params(which='major', bottom=True, left=True, length=4, width=0.8, color='black')
ax.tick_params(which='minor', bottom=True, left=True, length=2.2, width=0.8, color='black')
plt.tight_layout()

# save figure
figures_dir = simulation_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / f"simulation_execution_time.pdf", bbox_inches="tight")
plt.show()
