import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_t, pearsonr
from itertools import product
from joblib import Parallel, delayed
from limvam import direct_limvam


def generate_data(
    m,
    p,
    n,
    rng=None,
    disturbances="gaussian",
    cross_view_correlations=True,
    cross_variable_correlations=True,
    cross_view_variable_diversity=True,
):
    """
    Generate data according to the model xi = Bi xi + ei, where the Bi are DAG matrices 
    and the disturbances ei are correlated across views.
    
    Parameters
    ----------
    m : Number of views (int)
    p : Number of components (int)
    n : Number of samples (int)
    rng : Random seed 
    disturbances : Type of density (either 'gaussian' or 'student_t')

    Returns
    -------
    X : Training data (ndarray of shape (m, p, n))
    B : DAG matrices (ndarray of shape (m, p, p))
    T : Strictly lower triangular matrices (ndarray of shape (m, p, p))
    P : Permutation matrix (ndarray of shape (p, p))
    order : Permutation (array of shape (p,))
    """
    # variance of the disturbances
    M = rng.randn(p, m, m)
    Sigmas = np.zeros((p, m, m))
    for j in range(p):
        S = M[j] @ M[j].T
        D = np.diag(1 / np.sqrt(np.diag(S)))
        Sigmas[j] = D @ S @ D
        if not cross_view_correlations:
            Sigmas[j] *= np.eye(m)
        elif not cross_view_variable_diversity:
            Sigmas[j] = abs(Sigmas[j, 0, 1])
            np.fill_diagonal(Sigmas[j], 1)
    
    # disturbances
    E = np.zeros((m, p, n))
    for j in range(p):
        if disturbances == "gaussian":
            E[:, j] = rng.multivariate_normal(mean=np.zeros(m), cov=Sigmas[j], size=(n,)).T
        elif disturbances == "student_t":
            E[:, j] = multivariate_t.rvs(
                loc=np.zeros(m), shape=Sigmas[j], df=4, size=n, random_state=rng).T
        else:
            raise ValueError("The parameter 'disturbances' should be 'gaussian' or 'student_t'.")

    # strictly lower triangular matrices T
    T = rng.normal(size=(m, p, p))
    for i in range(m):
        T[i][np.triu_indices(p, k=0)] = 0
    if not cross_variable_correlations:
        T = np.zeros((m, p, p))
    elif not cross_view_variable_diversity:
        for i in range(1, m):
            T[i] = T[0]
    
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


def power_mean(x, r=2):
    return (np.mean(x**r))**(1/r)


def correlation_vector(x, y):
    """
    Compute correlation vector between x and y.
    x, y: arrays of shape (m, n)
    Returns correlation vector
    """
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)

    num = np.sum(x * y, axis=1)
    den = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1))

    return num / den


def correlation_matrix(x, y):
    """
    Compute correlation matrix between x and y.
    x, y: arrays of shape (m, n)
    Returns correlation matrix
    """
    cov = x @ y.T / x.shape[1]
    std_x = x.std(axis=1, ddof=1)
    std_y = y.std(axis=1, ddof=1)
    corr = cov / np.outer(std_x, std_y)
    return corr


def compute_residuals_with_univariate_OLS(x, y):
    """
    Regress y on x for multiple views and return residuals.
    x, y: arrays of shape (m, n)
    Returns residuals of shape (m, n)
    """
    cov = np.mean(x * y, axis=1)  # shape (m,)
    var_x = np.mean(x * x, axis=1)  # shape (m,)
    beta = cov / var_x
    return y - beta[:, None] * x


def get_score_assumption_bivariate(x, y):
    """Evaluate if the "correlation and diversity" assumption is fulfilled 
    for only 2 variables x and y. The function assumes x -> y.

    x, y: arrays of shape (m, n)

    Returns a scalar score between 0 and 1
    """
    # correlation between variables
    corr_xy = correlation_vector(x, y)  # shape (m,)
    score_corr_xy = np.cbrt(power_mean(corr_xy, r=2))
    
    # compute residuals r = y - b * x
    r = compute_residuals_with_univariate_OLS(x, y)  # shape (m, n)
    
    # correlation across views
    corr_x = correlation_matrix(x, x)  # shape (m, m)
    corr_r = correlation_matrix(r, r)  # shape (m, m)
    score_corr_x = np.cbrt(power_mean(corr_x[np.triu_indices(m, k=1)], r=2))
    score_corr_r = np.cbrt(power_mean(corr_r[np.triu_indices(m, k=1)], r=2))
    
    # diversity between variables
    score_div_xy = np.cbrt(2 * np.std(np.abs(corr_xy)))  # score is between 0 and 1
    
    # diversity across views
    corr_diff = np.abs(np.abs(corr_x) - np.abs(corr_r))
    score_div_diff = np.cbrt(2 * np.std(corr_diff[np.triu_indices(m, k=1)]))
    
    # aggregate scores
    score_corr = score_corr_xy * ((score_corr_x + score_corr_r) / 2)
    score_div = (score_div_xy + score_div_diff) / 2
    
    norm_constant = 1 / 2  # XXX arbitrary for now
    final_score = score_corr * score_div / norm_constant

    return final_score


def get_score_assumption(X, order=None):
    """Evaluate if the "correlation and diversity" assumption is fulfilled for p variables.

    X: array of shape (m, p, n)

    Returns a scalar score between 0 and 1
    """
    p = X.shape[1]
    
    if order is None:
        _, _, P = direct_limvam(X)
        order = np.argmax(P, axis=1)
    
    R = X.copy()
    scores = np.zeros((p-1))
    remaining_indices = list(np.arange(p))
    for i in range(p-1):
        cause_id = order[i]
        effect_id = order[i+1]
        scores[i] = get_score_assumption_bivariate(R[:, cause_id], R[:, effect_id])
        remaining_indices.remove(cause_id)
        for j in remaining_indices:
            R[:, j] = compute_residuals_with_univariate_OLS(R[:, cause_id], R[:, j])
    
    return scores, order


def compute_error_P(P1, P2, method="exact"):
    if method == "exact":
        return 1 - (P1 == P2).all()
    else:
        return pearsonr(np.argmax(P1, axis=1), np.argmax(P2, axis=1))[0]


def run_experiment(
    m, 
    p, 
    n, 
    random_state=None, 
    disturbances="gaussian", 
    cross_view_correlations=True, 
    cross_variable_correlations=True,
):
    rng = np.random.RandomState(random_state)
    
    # generate data
    X, B, _, P, _ = generate_data(
        m, 
        p, 
        n, 
        rng=rng, 
        disturbances=disturbances, 
        cross_view_correlations=cross_view_correlations, 
        cross_variable_correlations=cross_variable_correlations,
    )
    
    # run DirectLiMVAM
    B_est, _, P_est = direct_limvam(X)
    order_est = np.argmax(P_est, axis=1)
    
    # get assumption score
    scores, _ = get_score_assumption(X, order=order_est)
    score_assump = np.mean(scores)
    
    # get errors for B and P
    error_B = np.mean((B_est - B) ** 2)
    error_P_exact = compute_error_P(P, P_est, method="exact")
    corr_P_spearmanr = compute_error_P(P, P_est, method="spearmanr")
    
    output = {
        "cross_view_correlations": cross_view_correlations,
        "cross_variable_correlations": cross_variable_correlations,
        "random_state": random_state,
        "score_assump": score_assump,
        "error_B": error_B,
        "corr_P_spearmanr": corr_P_spearmanr,
        "error_P_exact": error_P_exact,
    }
    
    return output


# limit number of jobs
N_JOBS = 20
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# parameters
N_JOBS = 4
m = 10
p = 5
n = 1000
disturbances = "gaussian"
nb_seeds = 100
random_state_list = np.arange(nb_seeds)

# run experiment multiple times
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        random_state=random_state, 
        disturbances=disturbances, 
        cross_view_correlations=cross_view_correlations, 
        cross_variable_correlations=cross_variable_correlations,
    ) for cross_view_correlations, cross_variable_correlations, random_state
    in product([True, False], [True, False], random_state_list)
)

df = pd.DataFrame(dict_res)

# save
results_dir = "/storage/store4/work/aheurteb/LiMVAM/experiments_synthetic/results/results_diversity_correlation_assumption/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
