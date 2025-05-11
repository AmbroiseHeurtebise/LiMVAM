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
nb_seeds = 50
metric = "error_B"  # or "error_T", "error_P_exact", "error_P_spearmanr", "amari_distance"

# read dataframe
results_dir = "/Users/ambroiseheurtebise/Desktop/LiMVAM/simulation_studies/results/results_noise_diversity/"
# results_dir = "/storage/store2/work/aheurteb/MICaDo/simulation_studies/results/results_noise_diversity/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

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
labels = ['MICaDo-ML']
dashes = ['']
hue_order = ["shica_ml"]

# plot
fig, ax = plt.subplots(figsize=(6, 2.7))
sns.lineplot(
    data=df, x="nb_equal_variances", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
    errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
    dashes=dashes, markers=True)
ax.set_yscale("log")
ax.set_xlabel("Number of views with equal variances", fontsize=fontsize)
ax.xaxis.set_label_coords(0.5, -0.17)
ax.set_ylabel(metric_name, fontsize=fontsize)
ax.yaxis.set_label_coords(-0.155, 0.5)
ax.grid(which='both', linewidth=0.5, alpha=0.5)
ax.get_legend().remove()

# legend
palette = sns.color_palette()[:2]
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-', marker='X', 
           markeredgecolor="white", markersize=7),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 0.99), loc="center",
    ncol=2, borderaxespad=0., fontsize=fontsize
)

# caption
caption = (
    "Caption: Data are generated with $m=5$ views and $p=4$\ndisturbances, "
    "consisting of 2 Gaussian and 2 non-Gaussian \n"
    "disturbances. We vary the number of views in which the \n"
    "2 Gaussian disturbances have equal variances. The error \n"
    "increases abruptly only when variances are equal in all \n"
    "views, which justifies Assumption 1."
)
fig.text(0.5, -0.42, caption, ha='center', va='center', fontsize=fontsize)

# save figure
figures_dir = Path("/Users/ambroiseheurtebise/Desktop/LiMVAM/simulation_studies/figures")
# figures_dir = Path("/storage/store2/work/aheurteb/MICaDo/simulation_studies/figures")
plt.savefig(figures_dir / f"simulation_noise_diversity.pdf", bbox_inches="tight")
plt.show()
