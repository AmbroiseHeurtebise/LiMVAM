import numpy as np
from numpy.linalg import cholesky
from scipy.stats import gennorm, pearsonr, multivariate_t
from time import time
from picard import amari_distance
import lingam
from limvam.micado import micado
from limvam.praline import praline
from limvam.caramel import caramel
from limvam.directlingam_extension import directlingam_extension
import warnings
from sklearn.exceptions import ConvergenceWarning


# non-linear activation: f = (1 - alpha) * identity + alpha * logistic
def f1(x):
    return x

def f2(x):
    return 2 / (1 + np.exp(-x)) - 1

def f(x, alpha):
    return (1 - alpha) * f1(x) + alpha * f2(x)


def rmultivariate_powerexp(rng, n, mean, Sigma, beta):
    """
    Multivariate generalized Gaussian (power-exponential) with SciPy-like convention:
      - beta = 2 -> multivariate Gaussian
      - beta = 1 -> Laplace-like
      - beta < 2 -> heavier tails; beta > 2 -> lighter tails

    Construction (elliptical):
      X = mean + A (U * T),  with  A A^T = Sigma
      U ~ uniform on the unit sphere S^{m-1}
      Draw S ~ Gamma(shape=m/beta, scale=2), then set T = S**(1/beta)

    Notes:
      - Sigma acts as a *scatter* matrix. For beta != 2, Cov(X) (if it exists) is a scalar multiple of Sigma.
      - Returns array of shape (m, n).
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")

    mean = np.asarray(mean)
    m = mean.size
    A = cholesky(Sigma)

    # Directions: U ~ uniform on the unit sphere in R^m
    Z = rng.standard_normal(size=(n, m))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    U = Z / norms  # each row has unit norm

    # Radii that match SciPy's gennorm convention (beta=2 -> Gaussian)
    # With our mapping, T^{beta} ~ Gamma(shape=m/beta, scale=2)  => T = S**(1/beta)
    S = rng.gamma(shape=m / beta, scale=2.0, size=n)
    T = S ** (1.0 / beta)

    X = mean + (U * T[:, None]) @ A.T
    return X.T


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
    betas_evenly_spaced=False,
    nb_zeros_Ti=0,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    random_state=None,
    shared_causal_ordering=True,
    use_scale_D=False,
    non_linearity_alpha=None,
    use_shared_disturbances=True,
):
    rng = np.random.RandomState(random_state)
    
    if use_shared_disturbances:
        # scales
        if use_scale_D:
            # D = rng.uniform(low=0.1, high=3, size=(m, p))
            D = rng.uniform(low=0.5, high=2, size=(m, p))
        else:
            D = np.ones((m, p))
        
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
                    if use_scale_D:
                        # hardcode for p=4
                        new_values = D[:, 3] / D[:, 2] * sigmas[:, 2]
                        if len(indices) > 0:
                            for idx in indices:
                                sigmas[idx, 3] = new_values[idx]
                    else:
                        sigmas[indices, -nb_gaussian_disturbances:] = sigmas[indices, -nb_gaussian_disturbances][:, np.newaxis]
        elif density == "sub_gauss_super":
            # sources
            if betas_evenly_spaced:
                betas = np.linspace(beta1, beta2, p)
                S = np.zeros((p, n))
                for j in range(p):
                    S[j] = gennorm.rvs(betas[j], size=n, random_state=random_state)
            else:
                S1 = gennorm.rvs(beta1, size=(p//3, n), random_state=random_state)
                S2 = gennorm.rvs(2, size=(p-2*(p//3), n), random_state=random_state)  # Gaussian
                S3 = gennorm.rvs(beta2, size=(p//3, n), random_state=random_state)
                S = np.vstack((S1, S2, S3))
            S = S[rng.permutation(p)]
            # noise variances
            sigmas = rng.uniform(size=(m, p))
        else:
            raise ValueError("The parameter 'density' should be either 'gauss_super' or 'sub_gauss_super'")

        # noise
        N = noise_level * rng.normal(scale=sigmas[:, :, np.newaxis], size=(m, p, n))

        # disturbances
        E = D[:, :, None] * S + N
    else:
        # variance of the disturbances
        M = rng.randn(p, m, m)
        Sigmas = np.zeros((p, m, m))
        for j in range(p):
            Sigmas[j] = M[j] @ M[j].T + m * np.eye(m)

        # disturbances
        E = np.zeros((m, p, n))
        if density == "gauss_super":
            if nb_gaussian_disturbances == p:
                for j in range(p):
                    E[:, j] = rng.multivariate_normal(
                        mean=np.zeros(m), cov=Sigmas[j], size=(n,)).T
            elif nb_gaussian_disturbances == 0:
                for j in range(p):
                    E[:, j] = multivariate_t.rvs(
                        loc=np.zeros(m), shape=Sigmas[j], df=10, size=n, random_state=rng).T
            else:
                for j in range(nb_gaussian_disturbances):
                    E[:, j] = rng.multivariate_normal(
                        mean=np.zeros(m), cov=Sigmas[j], size=(n,)).T
                for j in range(nb_gaussian_disturbances, p):
                    E[:, j] = multivariate_t.rvs(
                        loc=np.zeros(m), shape=Sigmas[j], df=10, size=n, random_state=rng).T
                perm = rng.permutation(p)
                E = E[:, perm, :]
        elif density == "sub_gauss_super":
            if betas_evenly_spaced:
                betas = np.linspace(beta1, beta2, p)
                for j in range(p):
                    E[:, j] = rmultivariate_powerexp(
                        rng, n=n, mean=np.zeros(m), Sigma=Sigmas[j], beta=betas[j])
            else:
                for j in range(p//3):
                    E[:, j] = rmultivariate_powerexp(
                        rng, n=n, mean=np.zeros(m), Sigma=Sigmas[j], beta=beta1)
                for j in range(p//3, p-p//3):
                    E[:, j] = rmultivariate_powerexp(
                        rng, n=n, mean=np.zeros(m), Sigma=Sigmas[j], beta=2)
                for j in range(p-p//3, p):
                    E[:, j] = rmultivariate_powerexp(
                        rng, n=n, mean=np.zeros(m), Sigma=Sigmas[j], beta=beta2)
            perm = rng.permutation(p)
            E = E[:, perm, :]
        else:
            raise ValueError("The parameter 'density' should be either 'gauss_super' or 'sub_gauss_super'")

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
    
    # non-linear activation
    if non_linearity_alpha is not None:
        X = f(X, alpha=non_linearity_alpha)

    return X, B, T, P, A


def run_experiment(
    m,
    p,
    n,
    noise_level=1.,
    density="gauss_super",
    beta1=1,
    beta2=3,
    betas_evenly_spaced=False,
    nb_zeros_Ti=0,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    ica_algo="shica_ml",
    random_state=None,
    shared_causal_ordering=True,
    use_scale_D=False,
    non_linearity_alpha=None,
    use_shared_disturbances=True,
):
    if use_shared_disturbances & (density == "sub_gauss_super"):
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
        betas_evenly_spaced=betas_evenly_spaced,
        nb_zeros_Ti=nb_zeros_Ti,
        nb_gaussian_disturbances=nb_gaussian_disturbances,
        nb_equal_variances=nb_equal_variances,
        random_state=random_state,
        shared_causal_ordering=shared_causal_ordering,
        use_scale_D=use_scale_D,
        non_linearity_alpha=non_linearity_alpha,
        use_shared_disturbances=use_shared_disturbances,
    )

    # apply either our method, Multi Group DirectLiNGAM, or LiNGAM
    if ica_algo in ["multiviewica", "shica_j", "shica_ml"]:
        start = time()
        # apply our main function to retrieve B, T, P, and W;
        B_estimates, T_estimates, P_estimate, _, W_estimates = micado(
            X, shared_causal_ordering=shared_causal_ordering, ica_algo=ica_algo,
            random_state=random_state, return_full=True)
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
        # reconstruct what would be unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    elif ica_algo == "direct_limvam":
        start = time()
        B_estimates, T_estimates, P_estimate = directlingam_extension(X)
        execution_time = time() - start
        # reconstruct what would be unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    elif ica_algo == "mv_notears":
        start = time()
        B_estimates, T_estimates, P_estimate, _ = caramel(
            X, lambda_pen=1., shared_causal_ordering=shared_causal_ordering, 
            use_callback=False)
        execution_time = time() - start
        if not shared_causal_ordering:
            P_estimates = P_estimate
        # reconstruct unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
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
        "non_linearity_alpha": non_linearity_alpha,
        "use_shared_disturbances": use_shared_disturbances,
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
