import numpy as np
import pickle
from time import time
from pathlib import Path
import os
from limvam.micado import micado
from limvam.praline import praline


# Limit the number of jobs
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=1"
)

# Parameters
n_subjects = 152
parcellation = "aparc_sub"
n_labels = 38
subset = False
group = 1
ica_algo = "pairwise"
random_state = 42
steps = 1000  # for PRaLiNE
lr = 1e-2  # for PRaLiNE

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/LiMVAM/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/{parcellation}_{n_subjects}_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# Load labels
with open(load_dir / f"labels.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Only keep subjects that have ``n_labels`` labels
if parcellation == "aparc":
    X = []
    for i in range(len(labels_list)):
        if len(labels_list[i]) == n_labels:
            X.append(X_list[i])
            labels = labels_list[i]
    X = np.array(X)
    n_subjects_full = len(X)  # may be lower than n_subjects

    # Remove subjects who have a strange envelope
    if subset:
        idx_remove = [
            8, 11, 17, 32, 36, 66, 71, 95, 99, 126, 133, 146, 3, 5, 6, 7, 9, 10, 23, 27, 28, 31, 33, 37, 38,
            44,47, 57, 65, 76, 78, 79, 96, 97, 105, 106, 112, 116, 117, 119, 125, 128, 132, 141, 143, 145]
        idx = list(set(np.arange(n_subjects_full)) - set(idx_remove))
        X = X[idx]
        n_subjects_full = len(X)
elif parcellation =="aparc_sub":
    # Get all 38 labels
    i = 0
    while len(labels_list[i]) < n_labels:
        i+=1
    labels = labels_list[i]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

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
    
    if group == 1:
        n_subjects_full = len(X) // 2
        X = X[:n_subjects_full]
    elif group == 2:
        n_subjects_full = len(X) - len(X) // 2
        X = X[n_subjects_full:]

# Apply causal discovery method
if ica_algo == "shica_ml":
    start = time()
    B, T, P, _, _ = micado(
        X, ica_algo=ica_algo, new_find_order_function=False, random_state=random_state)
    execution_time = time() - start
elif ica_algo == "pairwise":
    B, T, P = praline(X, steps=steps, lr=lr)

print(f"The method took {execution_time:.2f} s.")

# Save data
if group == 1:
    group_suffix = "_group1_"
elif group == 2:
    group_suffix = "_group2_"
else:
    group_suffix = ""
save_dir = Path(expes_dir / f"4_results/{parcellation}_{n_subjects_full}_subjects{group_suffix}_{ica_algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "P.npy", P)
np.save(save_dir / "T.npy", T)
np.save(save_dir / "B.npy", B)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels, f)
