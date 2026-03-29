import numpy as np
import pandas as pd
import os
from time import time
from itertools import product
from joblib import Parallel, delayed
from utils import run_experiment


# limit number of jobs
N_JOBS = 20
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)


# fixed parameters
m = 6
p = 5
n = 1000
density = "sub_gauss_super"
algo = "direct_limvam"

# varying parameters
cross_view_correlations_alpha_list = np.linspace(0, 1, 21, endpoint=True)
# cross_view_correlations_alpha_list = np.concatenate((np.array([0]), np.logspace(-2, 0, 11, endpoint=True)))
nb_seeds = 100
random_state_list = np.arange(nb_seeds)

# run experiment
nb_expes = len(random_state_list) * len(cross_view_correlations_alpha_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
start = time()
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        density=density,
        algo=algo,
        random_state=random_state,
        cross_view_correlations_alpha=alpha,
    ) for random_state, alpha
    in product(random_state_list, cross_view_correlations_alpha_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)
execution_time = time() - start
print(f"The experiment took {execution_time:.2f} s.")

# save dataframe
results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_cross_view_correlations_decrease/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
