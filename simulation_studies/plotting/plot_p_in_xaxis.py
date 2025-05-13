import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 15
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 20
metric = "error_B"  # or "error_T", "error_P_exact", "error_P_spearmanr", "amari_distance"
beta1 = 1.5
beta2 = 2.5

# read dataframe
beta1_str = str(beta1).replace('.', '')
beta2_str = str(beta2).replace('.', '')
results_dir = "/storage/store2/work/aheurteb/LiMVAM/simulation_studies/results/results_p_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_beta_{beta1_str}_{beta2_str}_time_and_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# number of views and disturbances
m_list = np.sort(np.unique(df["m"]))
p_list = np.sort(np.unique(df["p"]))

# metric name
if metric == "error_B":
    metric_name = r"Error on $B^i$"
elif metric == "error_T":
    metric_name = r"Error on $T^i$"
elif metric == "error_P_exact":
    metric_name = r"Error on $P$"
elif metric == "error_P_spearmanr":
    metric_name = "Spearman's rank\ncorrelation on" + r" $P$"
elif metric == "amari_distance":
    metric_name = "Amari distance"

# labels, dashes and curves order
# labels = ['PairwiseLiMVAM', 'MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
labels = ['PRaLiNE', 'MICaDo-ML', 'MICaDo-J', 'ICA-LiNGAM', 'MultiGroupDirectLiNGAM']
dashes = ['', (2, 2), (2, 2), (2, 2), (2, 2)]
hue_order = ["pairwise", "shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]
# remove MVICA LiNGAM and MV-NOTEARS curves
filtered_df = df[(df["ica_algo"] != "multiviewica") & (df["ica_algo"] != "mv_notears")]

# subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 4.8), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    m = m_list[i]
    data = filtered_df[filtered_df["m"] == m]
    sns.lineplot(
        data=data, x="p", y=metric, linewidth=2.5, hue="ica_algo", ax=ax, estimator=np.median,
        errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
        dashes=dashes, markers=True)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"m = {m}", fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
xlabel = fig.supxlabel(r"Number of disturbances $p$", fontsize=fontsize)
ylabel = fig.supylabel(metric_name, fontsize=fontsize)
xlabel.set_position((0.5, 0.07))
ylabel.set_position((0.032, 0.5))
ax.set_xticks(p_list)
# if include_multiviewica:
#     ymin, ymax = ax.get_ylim()
#     ax.set_ylim(ymin, 1e2)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# legend
palette = sns.color_palette()
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='--', marker='X', 
           markeredgecolor="white", markersize=7),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle='--', marker='s', 
           markeredgecolor="white", markersize=5),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle='--', marker='P', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[4], linewidth=2.5, linestyle='--', marker='D', 
           markeredgecolor="white", markersize=5),
]
ncol = 3
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
    ncol=ncol, borderaxespad=0., fontsize=fontsize
)

# # caption
# caption = (
#     "Caption: Separation performance of ICA-LiNGAM, MultiGroupDirectLiNGAM, and two versions \n"
#     "of our method when we vary the number of views $m$ and disturbances $p$. The disturbances are \n"
#     "evenly divided into sub-Gaussian, Gaussian, and super-Gaussian groups."
# )
# fig.text(0.5, -0.05, caption, ha='center', va='center', fontsize=fontsize)

# save figure
figures_dir = Path("/storage/store2/work/aheurteb/LiMVAM/simulation_studies/figures")
plt.savefig(figures_dir / f"simulation_p_in_xaxis.pdf", bbox_inches="tight")
plt.show()
