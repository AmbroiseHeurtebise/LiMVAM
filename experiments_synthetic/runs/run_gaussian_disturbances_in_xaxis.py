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
m = 10
p = 8
n = 1000
new_find_order_function = False

# varying parameters
nb_gaussian_disturbances_list = np.arange(p+1)
nb_seeds = 50
random_state_list = np.arange(nb_seeds)
algo_list = ["ica_limvam_j", "ica_limvam_ml", "multi_group_direct_lingam", "lingam"]

# run experiment
nb_expes = len(nb_gaussian_disturbances_list) * len(random_state_list) * len(algo_list)
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
        new_find_order_function=new_find_order_function,
    ) for nb_gaussian_disturbances, random_state, ica_algo
    in product(nb_gaussian_disturbances_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/MICaDo/experiments_synthetic/results/"
parent_dir = "results_gaussian_disturbances_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_and_7_metrics"
save_path = results_dir + parent_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
