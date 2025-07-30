# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# %%
# Parameters
n_subjects = 98
n_runs = 50
ica_algo = "shica_ml"
n_arrows = 10
random_state = 42
rng = np.random.RandomState(random_state)

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/LiMVAM/real_data_experiments")
results_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects}_subjects_{n_runs}_runs_{ica_algo}")

P_total = np.load(results_dir / "P_total.npy")
T_total = np.load(results_dir / "T_total.npy")
B_total = np.load(results_dir / "B_total.npy")
with open(results_dir / f"labels.pkl", "rb") as f:
    labels = pickle.load(f)

# %%
# matplotlib style
fontsize = 26
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# %%
# Compute Pearson coefficients for the Bi matrices
spearmanr_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        B1_median = np.median(B_total[i], axis=0)
        B2_median = np.median(B_total[j], axis=0)
        # rho, p_value = spearmanr(B1_median.flatten(), B2_median.flatten())
        rho, p_value = pearsonr(B1_median.flatten(), B2_median.flatten())
        spearmanr_matrix[i, j] = rho
np.fill_diagonal(spearmanr_matrix, 0)

# Compute the average Pearson coefficient
upper_triangular_values = spearmanr_matrix[np.triu_indices(n_runs, k=1)]
avg_corr = np.mean(upper_triangular_values)

# Plot obtained coefficients
fig, ax = plt.subplots(figsize=(5, 5))
norm = TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
im = plt.imshow(spearmanr_matrix, norm=norm, cmap="coolwarm")
# plt.title(f"Average = {avg_corr:.2f}")
# ax.set_xticks(np.arange(n_runs))
# ax.set_yticks(np.arange(n_runs))
ax.set_xlabel("Runs")
ax.set_ylabel("Runs")
ax.set_xticks([])
ax.set_yticks([])

# colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(["-1", "0", "1"])

save = True
if save:
    figures_dir = "/storage/store2/work/aheurteb/LiMVAM/real_data_experiments/6_figures//"
    plt.savefig(figures_dir + f"pearson_coefs_B.pdf", bbox_inches="tight")
plt.show()

print(f"Average = {avg_corr:.2f}")

# %%
# Plot median matrix T (should be lower triangular)
T_avg = np.mean(np.abs(T_total), axis=(0, 1))
plt.imshow(T_avg)
plt.colorbar()
plt.title("Average absolute value lower triangular matrix T")
plt.show()

# %%
# Plot median adjacency matrix
B_median = np.median(B_total, axis=(0, 1))
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=min(-1, np.min(B_median)), vmax=max(1, np.max(B_median)), vcenter=0)
plt.imshow(B_median, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title("Median adjacency matrix B")
label_names = [label.name for label in labels]
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
# Only keep the most important effects
M = np.abs(B_median)
indices = np.argsort(M.flatten())[::-1]
ranked_flat = np.zeros(M.size)
ranked_flat[indices] = np.arange(M.size)
B_avg_rank = ranked_flat.reshape(M.shape)
B_avg_subset = B_median * (B_avg_rank < n_arrows)

fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=min(-1, np.min(B_avg_subset)), vmax=max(1, np.max(B_avg_subset)), vcenter=0)
plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Median adjacency matrix B ({n_arrows} highest effects)")
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
# random select 6 subjects
random_state = 46
rng = np.random.RandomState(random_state)
idx = rng.choice(B_total.shape[1], size=6, replace=False)
# idx[2] = 44
B_median_2 = np.median(B_total, axis=(0))  # shape (49, 10, 10)
B_subset = B_median_2[idx]

# %%
# Plot function
def plot_B(B_median, n_arrows=10):
    # Only keep the most important effects
    M = np.abs(B_median)
    indices = np.argsort(M.flatten())[::-1]
    ranked_flat = np.zeros(M.size)
    ranked_flat[indices] = np.arange(M.size)
    B_avg_rank = ranked_flat.reshape(M.shape)
    B_avg_subset = B_median * (B_avg_rank < n_arrows)
    
    fig, ax = plt.subplots()
    if np.min(B_avg_subset) < 0:
        norm = TwoSlopeNorm(vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset), vcenter=0)
    else:
        norm = TwoSlopeNorm(
            vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset),
            vcenter=(np.min(B_avg_subset)+np.max(B_avg_subset))/2)
    plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
    plt.colorbar()
    plt.title("Median adjacency matrix B")
    label_names = [label.name for label in labels]
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    plt.grid()
    plt.show()

# %%
# plot 6 causal effect matrices
go = False
if go:
    for i in range(6):
        print(i)
        plot_B(B_subset[i])

# %%
# Compute Spearman's rank correlation between the P's
spearmanr_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        sigma1 = np.argmax(P_total[i], axis=1)
        sigma2 = np.argmax(P_total[j], axis=1)
        rho, p_value = pearsonr(sigma1, sigma2)
        spearmanr_matrix[i, j] = rho
np.fill_diagonal(spearmanr_matrix, 0)

# Compute the average Pearson coefficient
upper_triangular_values = spearmanr_matrix[np.triu_indices(n_runs, k=1)]
avg_corr = np.mean(upper_triangular_values)

# Plot obtained coefficients
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
plt.imshow(spearmanr_matrix, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Average = {avg_corr:.2f}")
# ax.set_xticks(np.arange(n_runs))
# ax.set_yticks(np.arange(n_runs))
ax.set_xlabel("Runs")
ax.set_ylabel("Runs")

save = True
if save:
    figures_dir = "/storage/store2/work/aheurteb/LiMVAM/real_data_experiments/6_figures//"
    plt.savefig(figures_dir + f"spearmanr_coefs_P.pdf", bbox_inches="tight")
plt.show()

# %%
# Histogram of the correlations
fig, ax = plt.subplots(figsize=(11, 4))
nbins = 50
bins = np.linspace(-1, 1, nbins + 1)
n, bins, patches = ax.hist(upper_triangular_values, bins=bins, edgecolor='black')
# colors
bin_centers = 0.5 * (bins[:-1] + bins[1:])
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.cm.coolwarm
for center, patch in zip(bin_centers, patches):
    color = cmap(norm(center))
    patch.set_facecolor(color)
# vline and labels
ax.vlines(x=0, ymin=0, ymax=n.max(), ls="--", colors="black", lw=2.5)
# ax.set_xlabel("Spearman's rank correlation")
ax.set_xlabel("")
# label = ax.set_ylabel("Number of pairs\nof permutations")
# label.set_position((0., 0.42))
ax.set_xticks([-1, 0, 1])
ax.set_yticks([])
# Add a text box for the mean
textstr = f"Average = {avg_corr:.2f}"
props = dict(boxstyle="round", facecolor="white", alpha=0.8)
ax.text(0.4, 0.8, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right', bbox=props)
# save
save = True
if save:
    # fig.subplots_adjust(left=0.15, bottom=0.15, top=1.1)
    figures_dir = "/storage/store2/work/aheurteb/LiMVAM/real_data_experiments/6_figures//"
    plt.savefig(figures_dir + f"histogram_spearmanr_coefs_P.pdf", bbox_inches="tight")
plt.show()

# %%
