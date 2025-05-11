import numpy as np
from scipy.stats import gennorm, pearsonr
from time import time
from picard import amari_distance
import lingam
from limvam.micado import micado
from limvam.praline import praline
from limvam.caramel import caramel
import warnings
from sklearn.exceptions import ConvergenceWarning


# function that samples data according to our model
# we use similar parameters as in Fig. 2 of the ShICA paper
def sample_data(
    m,
    p,
    n,
    noise_level=1.,
    density="gauss_super",
    beta1=1,
    beta2=3,
    nb_zeros_Ti=0,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    random_state=None,
    shared_causal_ordering=True,
    use_scale_D=False,
):
    rng = np.random.RandomState(random_state)
    if density == "gauss_super":
        # sources
        S_ng = rng.laplace(size=(p-nb_gaussian_disturbances, n))
        S_g = rng.normal(size=(nb_gaussian_disturbances, n))
        S = np.vstack((S_ng, S_g))
        # noise variances
        sigmas = np.ones((m, p)) * 1 / 2
        if nb_gaussian_disturbances != 0:
            sigmas[:, -nb_gaussian_disturbances:] = rng.uniform(size=(m, nb_gaussian_disturbances))
            if nb_equal_variances > 0:
                indices = rng.choice(m, size=nb_equal_variances, replace=False)
                sigmas[indices, -nb_gaussian_disturbances:] = sigmas[indices, -nb_gaussian_disturbances][:, np.newaxis]
    elif density == "sub_gauss_super":
        # sources
        S1 = gennorm.rvs(beta1, size=(p//3, n), random_state=random_state)
        S2 = gennorm.rvs(2, size=(p-2*(p//3), n), random_state=random_state)  # Gaussian
        S3 = gennorm.rvs(beta2, size=(p//3, n), random_state=random_state)
        S = np.vstack((S1, S2, S3))
        # noise variances
        sigmas = rng.uniform(size=(m, p))
    else:
        raise ValueError("The parameter 'density' should be either 'gauss_super' or 'sub_gauss_super'")

    # noise
    N = noise_level * rng.normal(scale=sigmas[:, :, np.newaxis], size=(m, p, n))

    # scale and disturbances
    if use_scale_D:
        D = rng.uniform(low=0.1, high=3, size=(m, p))
        E = D[:, :, None] * S + N
    else:
        E = S + N

    # causal effect matrices T
    T = rng.normal(size=(m, p, p))
    for i in range(m):
        T[i][np.triu_indices(p, k=0)] = 0
        lower_tri_indices = np.tril_indices(p, k=-1)
        zero_indices = rng.choice(len(lower_tri_indices[0]), size=nb_zeros_Ti, replace=False)
        T[i][lower_tri_indices[0][zero_indices], lower_tri_indices[1][zero_indices]] = 0

    # causal order
    if shared_causal_ordering:
        P = np.eye(p)[rng.permutation(p)]
    else:
        P = np.array([np.eye(p)[rng.permutation(p)] for _ in range(m)])

    # causal effect matrices B
    if shared_causal_ordering:
        B = P.T @ T @ P
    else:
        B = np.array([Pi.T @ Ti @ Pi for Pi, Ti in zip(P, T)])

    # mixing matrices
    A = np.linalg.inv(np.eye(p) - B)

    # observations
    X = np.array([Ai @ Ei for Ai, Ei in zip(A, E)])
    return X, B, T, P, A


def run_experiment(
    m,
    p,
    n,
    noise_level=1.,
    density="gauss_super",
    beta1=1,
    beta2=3,
    nb_zeros_Ti=0,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    ica_algo="shica_ml",
    random_state=None,
    shared_causal_ordering=True,
    new_find_order_function=False,
    use_scale_D=False,
):
    if density == "sub_gauss_super":
        nb_gaussian_disturbances = p - 2 * (p // 3)
    # generate observations X, causal order(s) P, and causal effects B and T
    X, B, T, P, A = sample_data(
        m=m,
        p=p,
        n=n,
        noise_level=noise_level,
        density=density,
        beta1=beta1,
        beta2=beta2,
        nb_zeros_Ti=nb_zeros_Ti,
        nb_gaussian_disturbances=nb_gaussian_disturbances,
        nb_equal_variances=nb_equal_variances,
        random_state=random_state,
        shared_causal_ordering=shared_causal_ordering,
        use_scale_D=use_scale_D,
    )

    # apply either our method, Multi Group DirectLiNGAM, or LiNGAM
    if ica_algo in ["multiviewica", "shica_j", "shica_ml"]:
        start = time()
        # apply our main function to retrieve B, T, P, and W;
        B_estimates, T_estimates, P_estimate, _, W_estimates = micado(
            X, shared_causal_ordering=shared_causal_ordering, ica_algo=ica_algo,
            random_state=random_state, new_find_order_function=new_find_order_function,
            return_full=True)
        execution_time = time() - start
        if not shared_causal_ordering:
            P_estimates = P_estimate  # shape (m, p, p)
    elif ica_algo == "multi_group_direct_lingam":
        start = time()
        # apply Multi Group DirectLiNGAM to retrieve B, T, P, and W
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model = lingam.MultiGroupDirectLiNGAM()
            model.fit(list(np.swapaxes(X, 1, 2)))
        execution_time = time() - start
        # causal order P
        P_estimate = np.eye(p)[model.causal_order_]
        # causal effect matrices B and T
        B_estimates = np.array(model.adjacency_matrices_)
        T_estimates = P_estimate @ B_estimates @ P_estimate.T
        # reconstruct what would be unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    elif ica_algo == "lingam":
        start = time()
        # apply LiNGAM to retrieve B, T, P, and W
        B_estimates = []
        T_estimates = []
        P_estimates = []
        model = lingam.ICALiNGAM()
        for i in range(m):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(np.swapaxes(X[i], 0, 1))
            # causal order P
            P_estimate = np.eye(p)[model.causal_order_]
            P_estimates.append(P_estimate)
            # causal effect matrices B and T
            B_estimate = np.array(model._adjacency_matrix)
            B_estimates.append(B_estimate)
            T_estimate = P_estimate @ B_estimate @ P_estimate.T
            T_estimates.append(T_estimate)
        execution_time = time() - start
        B_estimates = np.array(B_estimates)
        T_estimates = np.array(T_estimates)
        P_estimates = np.array(P_estimates)  # shape (m, p, p) and not (p, p)
        # reconstruct unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    elif ica_algo == "pairwise":
        start = time()
        B_estimates, T_estimates, P_estimate = praline(X, steps=1000, lr=1e-2)
        execution_time = time() - start
    elif ica_algo == "mv_notears":
        start = time()
        B_estimates, T_estimates, P_estimate, _ = caramel(
            X, lambda_pen=1., shared_causal_ordering=shared_causal_ordering, 
            use_callback=False)
        execution_time = time() - start
        if not shared_causal_ordering:
            P_estimates = P_estimate
    else:
        raise ValueError("Wrong ica_algo.")
    
    # errors
    def compute_error_P(P1, P2, method="exact"):
        if method == "exact":
            error_P = 1 - (P1 == P2).all()
            return error_P
        else:
            corr = pearsonr(np.argmax(P1, axis=1), np.argmax(P2, axis=1))[0]
            return corr

    if shared_causal_ordering:
        # P has shape (p, p)
        if ica_algo == "lingam":
            # P_estimates has shape (m, p, p)
            # error_P = np.mean([1 - (Pe == P).all() for Pe in P_estimates])
            error_P_spearmanr = np.mean(
                [compute_error_P(Pe, P, method="spearmanr") 
                 for Pe in P_estimates])
            error_P_exact = np.mean(
                [compute_error_P(Pe, P, method="exact") 
                 for Pe in P_estimates])
        else:
            # P_estimate has shape (p, p)
            # error_P = 1 - (P_estimate == P).all()
            error_P_spearmanr = compute_error_P(P_estimate, P, method="spearmanr")
            error_P_exact = compute_error_P(P_estimate, P, method="exact")
    else:
        # P has shape (m, p, p)
        if ica_algo == "multi_group_direct_lingam" or ica_algo == "pairwise":
            # P_estimate has shape (p, p)
            # error_P = np.mean([1 - (P_estimate == Pi).all() for Pi in P])
            error_P_spearmanr = np.mean(
                [compute_error_P(P_estimate, Pi, method="spearmanr") for Pi in P])
            error_P_exact = np.mean(
                [compute_error_P(P_estimate, Pi, method="exact") for Pi in P])
        else:
            # P_estimates has shape (m, p, p)
            # error_P = np.mean([1 - (Pe == Pi).all() for Pe, Pi in zip(P_estimates, P)])
            error_P_spearmanr = np.mean(
                [compute_error_P(Pe, Pi, method="spearmanr")
                 for Pe, Pi in zip(P_estimates, P)])
            error_P_exact = np.mean(
                [compute_error_P(Pe, Pi, method="exact")
                 for Pe, Pi in zip(P_estimates, P)])
    error_B = np.mean((B_estimates - B) ** 2)
    error_B_abs = np.mean(np.abs(B_estimates - B))
    error_T = np.mean((T_estimates - T) ** 2)
    error_T_abs = np.mean(np.abs(T_estimates - T))
    amari = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_estimates, A)])
    
    # output
    output = {
        "m": m,
        "p": p,
        "n": n,
        "noise_level": noise_level,
        "ica_algo": ica_algo,
        "nb_gaussian_disturbances": nb_gaussian_disturbances,
        "nb_equal_variances": nb_equal_variances,
        "nb_zeros_Ti": nb_zeros_Ti,
        "shared_causal_ordering": shared_causal_ordering,
        "random_state": random_state,
        "error_B": error_B,
        "error_B_abs": error_B_abs,
        "error_T": error_T,
        "error_T_abs": error_T_abs,
        "error_P_spearmanr": error_P_spearmanr,
        "error_P_exact": error_P_exact,
        "amari_distance": amari,
        "execution_time": execution_time,
    }
    return output
