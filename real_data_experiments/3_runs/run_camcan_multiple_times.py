import numpy as np
import pickle
from pathlib import Path
import os
from time import time
from joblib import Parallel, delayed
from limvam.micado import micado
from limvam.praline import praline


# Limit the number of jobs
N_JOBS = 10
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=1"
)

# Parameters
n_runs = 50
# keep_subjects_rate = 1 / 2  # only keep 50% of the subjects
n_subjects_batch = 30  # only keep 30 subjects
ica_algo = "shica_ml"
steps = 1000
lr = 1e-2

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/LiMVAM/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/aparc_sub_152_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# Load labels
with open(load_dir / f"labels.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Get all 38 labels
i = 0
while len(labels_list[i]) < 38:
    i+=1
labels = labels_list[i]

# Keep only 10 regions
selected_label_names = [
    'superiortemporal_3-lh',
    'superiortemporal_5-rh',
    'pericalcarine_1-lh',
    'pericalcarine_4-rh',
    'postcentral_6-lh',
    'postcentral_8-lh',
    'postcentral_7-rh',
    'postcentral_8-rh',
    'precentral_11-lh',
    'precentral_7-rh',
]
n_labels = len(selected_label_names)

# Only keep subjects who have all these regions available
X = []
for X_current, labels_current in zip(X_list, labels_list):
    label_names_current = {label.name for label in labels_current}
    if all(name in label_names_current for name in selected_label_names):
        label_to_row = {label.name: row for label, row in zip(labels_current, X_current)}
        filtered_X = np.array([label_to_row[name] for name in selected_label_names])
        X.append(filtered_X)
X = np.array(X)  # shape (98, 10, 1760)
labels = [label for label in labels if label.name in selected_label_names]
n_subjects_full = len(X)

# Run our method ``n_runs`` times
# n_subjects_batch = int(n_subjects_full * keep_subjects_rate)
B_total = np.zeros((n_runs, n_subjects_batch, n_labels, n_labels))
T_total = np.zeros((n_runs, n_subjects_batch, n_labels, n_labels))
P_total = np.zeros((n_runs, n_labels, n_labels))

def single_run(i):
    rng = np.random.RandomState(i)
    subjects_idx = rng.choice(n_subjects_full, size=n_subjects_batch, replace=False)
    X_subset = X[subjects_idx]
    
    if ica_algo == "shica_ml":
        B, T, P, _, _ = micado(X_subset, ica_algo=ica_algo, random_state=i)
    elif ica_algo == "pairwise":
        B, T, P = praline(X, steps=steps, lr=lr)
    return B, T, P

start = time()

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(single_run)(i) for i in range(n_runs)
)

execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# Unpack the results
for i, (B, T, P) in enumerate(results):
    B_total[i] = B
    T_total[i] = T
    P_total[i] = P

# Save data
save_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects_full}_subjects_{n_runs}_runs_{ica_algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "B_total.npy", B_total)
np.save(save_dir / "T_total.npy", T_total)
np.save(save_dir / "P_total.npy", P_total)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels, f)
