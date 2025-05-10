# %%
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
from jax import flatten_util
from scipy.optimize import fmin_l_bfgs_b


def negative_log_likelihood(W, Sigma, X):
    # Compute useful quantities
    Y = jnp.array([Wi @ Xi for Wi, Xi in zip(W, X)])        # shape (m, p, n)
    C = jnp.sum(1 / Sigma, axis=0)                          # shape (p,)
    S_avg = _compute_S_avg(Sigma, Y, C)                     # shape (m, p)
    R = Y - S_avg                                           # shape (m, p, n)
    
    # Compute the NLL
    loss = -jnp.sum(jnp.linalg.slogdet(W)[1])
    loss += jnp.sum(jnp.log(Sigma)) / 2
    loss += _weighted_norm(R, Sigma) / 2
    loss += jnp.sum(jnp.log(C)) / 2
    loss -= _log_pdf(S_avg, C)
    
    return loss


def _compute_S_avg(Sigma, Y, C):
    weights = 1 / Sigma                                     # shape (m, p)
    weighted_Y = weights[:, :, None] * Y                    # shape (m, p, n)
    S_avg = jnp.sum(weighted_Y, axis=0) / C[:, None]        # shape (m, p)
    return S_avg 


def _weighted_norm(R, Sigma):
    """
    R : array of shape (m, p, n)
    Sigma : array of shape (m, p)
    """
    inv_Sigma = 1.0 / Sigma                                 # shape (m, p)
    squared_R = R ** 2                                      # shape (m, p, n)
    weighted = squared_R * inv_Sigma[:, :, None]            # broadcasting to (m, p, n)
    return weighted.sum() / R.shape[2]
 

def _log_pdf(S_avg, C):
    """
    S_avg : array of shape (p, n)
    C : array of shape (p,)
    """
    n_components, _ = S_avg.shape
    S_T = S_avg.T                                           # shape (n, p)

    diag_Cinv = 1.0 / C
    # Construct diagonal covariances
    V1_diag = 0.5 + diag_Cinv
    V2_diag = 1.5 + diag_Cinv
    
    # Log-pdf for each Gaussian component (n,)
    logp1 = multivariate_normal.logpdf(
        S_T, mean=jnp.zeros(n_components), cov=jnp.diag(V1_diag))
    logp2 = multivariate_normal.logpdf(
        S_T, mean=jnp.zeros(n_components), cov=jnp.diag(V2_diag))

    # Log of mixture density using logsumexp for numerical stability
    # log(p1 + p2) = logsumexp([logp1, logp2])
    log_mixture = logsumexp(jnp.stack([logp1, logp2]), axis=0)

    return jnp.mean(log_mixture)


# %%
import numpy as np
from scipy.stats import logistic


def sample_mixture(size, rng=None):
    """Sample from 0.5 * (N(0, 0.5) + N(0, 1.5))"""
    binary_choice = rng.randint(0, 2, size=size)
    stds = np.where(binary_choice == 0, np.sqrt(0.5), np.sqrt(1.5))
    samples = rng.normal(loc=0.0, scale=1.0, size=size) * stds
    return samples


def generate_data(
    n_views,
    n_components,
    n_samples,
    random_state=None,
    which_density="logistic",
):
    rng = np.random.RandomState(random_state)

    # sources S
    if which_density == "logistic":
        S = logistic.rvs(loc=0, scale=1, size=(n_components, n_samples), random_state=random_state)
    elif which_density == "laplace":
        S = rng.laplace(0, 1, size=(n_components, n_samples))
    elif which_density == "gaussian":
        S = rng.randn(n_components, n_samples)
    elif which_density == "mixture":
        S = sample_mixture(size=(n_components, n_samples), rng=rng)
    else:
        raise ValueError("The parameter 'which_density' should be either 'logistic', 'laplace', or 'gaussian'.")

    # noise N
    Sigma = rng.uniform(low=0.1, high=3, size=(n_views, n_components))
    N = rng.normal(scale=Sigma[:, :, None], size=(n_views, n_components, n_samples))
    
    # disturbances E
    E = np.array([S + Ni for Ni in N])

    # mixing matrices A
    A = rng.randn(n_views, n_components, n_components)

    # observations X
    X = np.array([Ai @ Ei for Ai, Ei in zip(A, E)])

    return X, A, Sigma


# parameters
m = 5
p = 4
n = 1000
random_state = 42
which_density = "mixture"

# generate data
X, A, Sigma = generate_data(m, p, n, random_state, which_density)
W = np.linalg.inv(A)

# Evaluate the NLL at true parameters
true_NLL = negative_log_likelihood(W, Sigma, X)
print(f"NLL at true params : {true_NLL}")

# # Evaluate the NLL at random parameters
# count = 0
# for _ in range(100):
#     B_rand = B + np.random.randn(m, p, p)
#     Sigma_rand = Sigma + np.random.uniform(low=0, high=2, size=(m, p))
#     D_rand = D + np.random.uniform(low=0, high=2, size=(m, p))
#     random_NLL = negative_log_likelihood(B_rand, Sigma_rand, D_rand, X)
    
#     if random_NLL < true_NLL:
#         count += 1

# if count == 0:
#     print("Ok!")

# %%
# Optimization of B
def jaxmin_l_bfgs_b(value_and_grad_func, x0, **kwargs):
    """Wrapper around scipy.optimize.fmin_l_bfgs_b"""
    x_flat, unravel_pytree = flatten_util.ravel_pytree(x0)

    def func_wrapper(x_flat):
        x = unravel_pytree(x_flat)
        value, grad = value_and_grad_func(x)
        return value, flatten_util.ravel_pytree(grad)[0]
    
    x, f, d = fmin_l_bfgs_b(func_wrapper, x_flat, **kwargs)
    return unravel_pytree(x), f, d


def estimate_B(X, W_Sigma_init):
    # value and grad function
    loss_partial = lambda W_Sigma: negative_log_likelihood(*W_Sigma, X)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_partial))

    # bounds 
    m, p, _ = X.shape
    bounds = [(None, None)] * m * p**2 + [(0.0, None)] * m * p

    # optimization
    W_Sigma_est, _, _ = jaxmin_l_bfgs_b(
        value_and_grad_fn, W_Sigma_init, bounds=bounds)
    
    W_est, Sigma_est = W_Sigma_est
    return W_est, Sigma_est

# %%
# Estimate B with LBFGS
# W_init = jnp.zeros((m, p, p))
W_init = jnp.array([jnp.eye(p)] * m)
Sigma_init = jnp.ones((m, p))
W_Sigma_init = (W_init, Sigma_init)
W_est, Sigma_est = estimate_B(X, W_Sigma_init)

# %%
# Evaluate the NLL at estimated parameters
est_NLL = negative_log_likelihood(W_est, Sigma_est, X)
print(f"NLL at estimated params : {est_NLL}")

# %%
from shica import shica_ml
from picard import amari_distance

W_shica, _, _ = shica_ml(X)

amari = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_est, A)])
amari_shica = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_shica, A)])

print(f"Amari distance of our method : {amari}")
print(f"Amari distance of ShICA : {amari_shica}")

# %%
