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
m = 8
p = 6
n = 1000
noise_level = 1.
density = "sub_gauss_super"
beta1 = 1.5
beta2 = 2.5
use_scale_D = True

# varying parameters
shared_causal_ordering_list = [True, False]
nb_zeros_Ti_list = np.arange(p * (p - 1) // 2 + 1)
nb_seeds = 30
random_state_list = np.arange(nb_seeds)
algo_list = ["shica_j", "shica_ml", "pairwise"]

# run experiment
nb_expes = len(shared_causal_ordering_list) * len(nb_zeros_Ti_list) * len(random_state_list) \
    * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
start = time()
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        noise_level=noise_level,
        density=density,
        beta1=beta1,
        beta2=beta2,
        nb_zeros_Ti=nb_zeros_Ti,
        shared_causal_ordering=shared_causal_ordering,
        random_state=random_state,
        ica_algo=ica_algo,
        use_scale_D=use_scale_D,
    ) for shared_causal_ordering, nb_zeros_Ti, random_state, ica_algo
    in product(shared_causal_ordering_list, nb_zeros_Ti_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)
execution_time = time() - start
print(f"The experiment took {execution_time:.2f} s.")

# save dataframe
results_dir = "/storage/store2/work/aheurteb/LiMVAM/simulation_studies/results/results_sparsity_of_Ti/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_time_and_scale"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
