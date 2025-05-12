import numpy as np
import pandas as pd
import os
from itertools import product
from joblib import Parallel, delayed
from utils import run_experiment


# limit the number of jobs
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# fixed parameters
m = 5
p = 4
shared_causal_ordering = True

# varying parameters
nb_gaussian_disturbances_list = [0, 2, 4]
nb_seeds = 50
random_state_list = np.arange(nb_seeds)
n_list = np.logspace(2, 4, 21, dtype=int)
algo_list = [
    "multiviewica", "shica_j", "shica_ml", "multi_group_direct_lingam", "lingam",
    "pairwise", "mv_notears"]

# run experiment
nb_expes = len(nb_gaussian_disturbances_list) * len(random_state_list) * len(n_list) * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        nb_gaussian_disturbances=nb_gaussian_disturbances,
        random_state=random_state,
        ica_algo=ica_algo,
        shared_causal_ordering=shared_causal_ordering,
    ) for n, nb_gaussian_disturbances, random_state, ica_algo
    in product(n_list, nb_gaussian_disturbances_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/LiMVAM/simulation_studies/results/"
if shared_causal_ordering:
    parent_dir = "results_timepoints_in_xaxis/shared_P/"
else:
    parent_dir = "results_timepoints_in_xaxis/multiple_Pi/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_and_time"
save_path = results_dir + parent_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
