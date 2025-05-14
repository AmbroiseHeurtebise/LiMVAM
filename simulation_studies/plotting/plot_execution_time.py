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
sns.set(style="white")
plt.rcParams.update(rc)

# parameters 
nb_seeds = 200

# read dataframe
simulation_dir = Path("/storage/store2/work/aheurteb/LiMVAM/simulation_studies")
results_dir = simulation_dir / "results/results_execution_time"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir / save_name
df = pd.read_csv(save_path)

# Use color palette and markers
palette_sns = sns.color_palette()
palette = {
    'pairwise': palette_sns[0],
    'shica_ml': palette_sns[1],
    'shica_j': palette_sns[2],
    'lingam': palette_sns[3],
    'multi_group_direct_lingam': palette_sns[4],
}
marker_styles = {
    'pairwise': 'o',
    'shica_ml': 'X',
    'shica_j': 's',
    'lingam': 'P',
    'multi_group_direct_lingam': 'D',
}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3.2))

for method, group_df in df.groupby("ica_algo"):
    # cloud of points
    sns.scatterplot(
        data=group_df,
        x="error_B",
        y="execution_time",
        color=palette.get(method, "gray"),
        alpha=0.2,
        label=None
    )
    
    # median
    x_median = group_df["error_B"].median()
    y_median = group_df["execution_time"].median()
    
    # error bars
    x_low = group_df["error_B"].quantile(0.16)
    x_high = group_df["error_B"].quantile(0.84)
    y_low = group_df["execution_time"].quantile(0.16)
    y_high = group_df["execution_time"].quantile(0.84)
    xerr = [[x_median - x_low], [x_high - x_median]]
    yerr = [[y_median - y_low], [y_high - y_median]]
    
    # markersize
    if method == "shica_j":
        markersize = 9
    elif method == "multi_group_direct_lingam":
        markersize = 8
    else:
        markersize = 10

    # markers and error bars
    ax.errorbar(
        x_median,
        y_median,
        xerr=xerr,
        yerr=yerr,
        fmt=marker_styles.get(method),
        markersize=markersize,
        color=palette.get(method, "gray"),
        capsize=5,
        elinewidth=2,
        markerfacecolor=palette.get(method, "gray"),
        markeredgecolor='white'
    )

# legend
legend_styles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=palette_sns[0], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='X', color='w',
           markerfacecolor=palette_sns[1], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor=palette_sns[2], markeredgecolor='white', markersize=9),
    Line2D([0], [0], marker='P', color='w',
           markerfacecolor=palette_sns[3], markeredgecolor='white', markersize=10),
    Line2D([0], [0], marker='D', color='w',
           markerfacecolor=palette_sns[4], markeredgecolor='white', markersize=8),
]
labels = ['PRaLiNE', 'MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.12), loc="center",
    ncol=2, borderaxespad=0., fontsize=fontsize
)

plt.xscale("log")
plt.yscale("log")
ax.set_xlim([10**(-4.1), 10**0.1])
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
