import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from time import time
import os
from itertools import product
from joblib import Parallel, delayed
from limvam.directlingam_extension import estimate_causal_order

from sparse_shift.utils import dag2cpdag
from sparse_shift.methods import MinChange

# Monkey-patch the function 'kci', which is outdated
from causallearn.utils.cit import CIT

def _kci_wrapper(data, X, Y, S, **kwargs):
    kci_obj = CIT(data, method="kci", **kwargs)
    return kci_obj(X, Y, S)

import sparse_shift.testing as sst
import sparse_shift.methods as sm
sst.kci = _kci_wrapper
sm.kci  = _kci_wrapper


# Wrapper around Perry's method (MSS with the KCI estimator)
def mss_kci_wrapped(X, B, use_oracle_cpdag=False):
    m, p, n = X.shape
    X_reshaped = [Xi.T for Xi in X]
    
    # get the true DAG (a square matrix with zeros and ones)
    true_dag = (B[0] != 0) * 1
    
    # get the CPDAG
    if use_oracle_cpdag:
        cpdag = dag2cpdag(true_dag)
    else:
        # cpdag is a matrix of ones with zeros on the diagonal
        cpdag = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
    
    # default parameters of the MSS method
    default_hyperparams = {
        'alpha': 0.05,
        'scale_alpha': True,
        'test': 'kci',
        'test_kwargs': {
            "KernelX": "GaussianKernel",
            "KernelY": "GaussianKernel",
            "KernelZ": "GaussianKernel",
        }
    }

    # instantiate MSS
    mss = MinChange(cpdag=cpdag, **default_hyperparams)

    # add environments one by one
    for Xi in X_reshaped:
        mss.add_environment(Xi)
    
    # recover the optimal CPDAG
    min_cpdag = mss.get_min_cpdag(False)
    
    # binary score: 1 if the DAG is recovered, 0 otherwise
    score = (min_cpdag.T == true_dag).all() * 1
    
    return score


def generate_data_with_interventions_on_Bi_and_fixed_variance(
    m,
    p,
    n,
    nb_interventions,  # should be between 0 and p*(p-1)//2
    rng=None,
    disturbances="gaussian",
):
    """
    Generate data according to the model xi = Bi xi + ei, where the Bi are DAG matrices 
    and the disturbances ei are correlated across views.
    """
    # covariance of the disturbances
    # for each view and each variable, the variance is fixed to 1
    M = rng.randn(p, m, m)  # or rng.standard_normal((p, m, m))
    Sigmas = np.zeros((p, m, m))
    for j in range(p):
        # Sigmas[j] = M[j] @ M[j].T / m
        # np.fill_diagonal(Sigmas[j], 1)
        S = M[j] @ M[j].T / m
        d = np.sqrt(np.diag(S))
        Sigmas[j] = S / np.outer(d, d)
    
    # disturbances
    E = np.zeros((m, p, n))
    for j in range(p):
        if disturbances == "gaussian":
            E[:, j] = rng.multivariate_normal(mean=np.zeros(m), cov=Sigmas[j], size=(n,)).T
        elif disturbances == "student_t":
            E[:, j] = multivariate_t.rvs(loc=np.zeros(m), shape=Sigmas[j], df=1, size=n, random_state=rng).T
        else:
            raise ValueError("The parameter 'disturbances' should be 'gaussian' or 'student_t'.")

    # strictly lower triangular matrices T  
    T0 = rng.normal(size=(p, p))
    T0[np.triu_indices(p)] = 0
    T = np.array([T0] * m)
    for i in range(m):
        interventions_indices = rng.choice(p*(p-1)//2, size=nb_interventions, replace=False)
        tril_indices = np.tril_indices(p, k=-1)
        for j in interventions_indices:
            T[i][tril_indices[0][j], tril_indices[1][j]] = rng.randn()
    
    # causal order P
    order = rng.permutation(p)
    P = np.eye(p)[order]

    # causal effect matrices B
    B = P.T @ T @ P
    
    # mixing matrices
    A = np.linalg.inv(np.eye(p) - B)
    
    # observations
    X = np.array([Ai @ Ei for Ai, Ei in zip(A, E)])
    return X, B, T, P, order


def run_experiment(
    m,
    p,
    n,
    nb_interventions,
    method="directlimvam",
    random_state=None,
    disturbances="gaussian",
):
    # generate data
    rng = np.random.RandomState(random_state)
    X, B, T, P, order = generate_data_with_interventions_on_Bi_and_fixed_variance(
        m, p, n, nb_interventions, rng, disturbances)
    
    # MSS
    if method == "mss":
        start = time()
        score = mss_kci_wrapped(X, B, use_oracle_cpdag=False)
        execution_time = time() - start
    
    # DirectLiMVAM
    if method == "directlimvam":
        start = time()
        order_estimated = estimate_causal_order(X)
        score = (order == order_estimated).all() * 1
        execution_time = time() - start
    
    # output
    output = {
        "m": m,
        "p": p,
        "n": n,
        "nb_interventions": nb_interventions,
        "method": method,
        "random_state": random_state,
        "disturbances": disturbances,
        "score": score,
        "execution_time": execution_time,
    }
    return output


# limit number of jobs
N_JOBS = 20
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# parameters
m = 5
p = 3
n = 200
disturbances = "gaussian"

method_list = ["directlimvam", "mss"]
max_interventions = p * (p-1) // 2
nb_interventions_list = np.arange(max_interventions+1)
nb_seeds = 50
random_state_list = np.arange(nb_seeds)

# run experiment
nb_expes = len(method_list) * len(nb_interventions_list) * len(random_state_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
start = time()
dict_res = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        nb_interventions=nb_interventions,
        method=method,
        random_state=random_state,
        disturbances=disturbances,
    ) for nb_interventions, random_state, method
    in product(nb_interventions_list, random_state_list, method_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)
execution_time = time() - start
print(f"The experiment took {execution_time:.2f} s.")

# save dataframe
results_dir = "/storage/store4/work/aheurteb/LiMVAM/simulation_studies/results/results_comparison_with_perry/"
save_name = f"DataFrame_with_m{m}_p{p}_n{n}_seeds{nb_seeds}"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
