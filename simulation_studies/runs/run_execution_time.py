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
m = 6
p = 5
n = 1000
noise_level = 1.
density = "sub_gauss_super"
beta1 = 1
beta2 = 3
betas_evenly_spaced = True
shared_causal_ordering = True
use_scale_D = True

# varying parameters
nb_seeds = 200
random_state_list = np.arange(nb_seeds)
algo_list = [
    "shica_j", "shica_ml", "multi_group_direct_lingam", "lingam", "pairwise", "direct_limvam"]

# run experiment
nb_expes = len(random_state_list) * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
start = time()
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        density=density,
        beta1=beta1,
        beta2=beta2,
        betas_evenly_spaced=betas_evenly_spaced,
        random_state=random_state,
        ica_algo=ica_algo,
        noise_level=noise_level,
        shared_causal_ordering=shared_causal_ordering,
        use_scale_D=use_scale_D,
    ) for random_state, ica_algo
    in product(random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)
execution_time = time() - start
print(f"The experiment took {execution_time:.2f} s.")

# save dataframe
results_dir = "/storage/store4/work/aheurteb/LiMVAM/simulation_studies/results/results_execution_time/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_new_methods"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
