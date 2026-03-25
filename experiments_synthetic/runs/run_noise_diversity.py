import numpy as np
import pandas as pd
import os
from time import time
from itertools import product
from joblib import Parallel, delayed
from utils import run_experiment


# limit number of jobs
N_JOBS = 8
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# fixed parameters
m = 5
p = 4
n = 1000
nb_gaussian_disturbances = 2
noise_level = 1.
use_scale_D = True

# varying parameters
nb_equal_variances_list = np.arange(m+1)
nb_seeds = 50
random_state_list = np.arange(nb_seeds)
algo_list = ["ica_limvam_j", "ica_limvam_ml", "pairwise_limvam"]

# run experiment
nb_expes = len(nb_equal_variances_list) * len(random_state_list) * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
start = time()
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        noise_level=noise_level,
        nb_gaussian_disturbances=nb_gaussian_disturbances,
        nb_equal_variances=nb_equal_variances,
        random_state=random_state,
        ica_algo=ica_algo,
        use_scale_D=use_scale_D,
    ) for nb_equal_variances, random_state, ica_algo
    in product(nb_equal_variances_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)
execution_time = time() - start
print(f"The experiment took {execution_time:.2f} s.")

# save dataframe
results_dir = "/storage/store2/work/aheurteb/LiMVAM/simulation_studies/results/results_noise_diversity/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
