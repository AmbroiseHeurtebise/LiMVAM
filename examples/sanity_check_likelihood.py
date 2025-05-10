# %%
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
from jax import flatten_util
from scipy.optimize import fmin_l_bfgs_b


def negative_log_likelihood(B, Sigma, D, X):
    """
    Compute the negative log-likelihood of the model 
    x^i = B^i x^i + D^i s + n^i,
    where each component of s has the density 1/2 * (N(0, 1/2) + N(0, 3/2)),
    and the n^i have the density N(0, Sigma[i]).

    Args:
    ----------
    B : ndarray, shape (n_views, n_components, n_components)
        Causal effect matrices.
    
    Sigma : ndarray, shape (n_views, n_components)
        Noise variances. All entries are positive.
    
    D : ndarray, shape (n_views, n_components)
        Sources' scales. All entries are positive.
    
    X : ndarray, shape (n_views, n_components, n_samples)
        Training data.
    """
    # Compute useful quantities
    W = jnp.eye(B.shape[1]) - B                             # shape (m, p, p)
    Y = jnp.array([Wi @ Xi for Wi, Xi in zip(W, X)])        # shape (m, p, n)
    C = jnp.sum(D**2 / Sigma, axis=0)                       # shape (p,)
    S_avg = _compute_S_avg(D, Sigma, Y, C)                  # shape (m, p)
    R = Y - D[:, :, None] * S_avg                           # shape (m, p, n)
    
    # Compute the NLL
    loss = -jnp.sum(jnp.linalg.slogdet(W)[1])
    loss += 0.5 * jnp.sum(jnp.log(Sigma))
    loss += 0.5 * _weighted_norm(R, Sigma)
    loss += 0.5 * jnp.sum(jnp.log(C))
    loss -= _log_pdf(S_avg, C)
    
    return loss


def _compute_S_avg(D, Sigma, Y, C):
    weights = D / Sigma                                     # shape (m, p)
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
    shared_causal_ordering=True,
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
    
    # scales D
    D = rng.uniform(low=0.1, high=3, size=(n_views, n_components))
    
    # noise N
    Sigma = rng.uniform(low=0.1, high=3, size=(n_views, n_components))
    N = rng.normal(scale=Sigma[:, :, None], size=(n_views, n_components, n_samples))
    
    # disturbances E
    # E = np.array([Di[:, None] * S + Ni for Di, Ni in zip(D, N)])
    E = D[:, :, None] * S + N

    # causal order P
    if shared_causal_ordering:
        P = np.eye(n_components)[rng.permutation(n_components)]
    else:
        P = np.array([np.eye(n_components)[rng.permutation(n_components)] for _ in range(n_views)])
    
    # triangular matrices T
    T = rng.randn(n_views, n_components, n_components)
    for i in range(n_views):
        T[i][np.triu_indices(n_components, k=0)] = 0

    # adjacency matrices B
    if shared_causal_ordering:
        B = P.T @ T @ P
    else:
        B = np.array([Pi.T @ Ti @ Pi for Pi, Ti in zip(P, T)])

    # mixing matrices A
    A = np.linalg.inv(np.eye(n_components) - B)

    # observations X
    X = np.array([Ai @ Ei for Ai, Ei in zip(A, E)])

    return X, D, Sigma, B, T, P


# parameters
m = 5
p = 4
n = 1000
random_state = 42
which_density = "mixture"
shared_causal_ordering = True

# generate data
X, D, Sigma, B, T, P = generate_data(
    m, p, n, random_state, which_density, shared_causal_ordering)

# Evaluate the NLL at true parameters
true_NLL = negative_log_likelihood(B, Sigma, D, X)
print(f"NLL at true params : {true_NLL}")


# %%
# Optimization
def jaxmin_l_bfgs_b(value_and_grad_func, x0, **kwargs):
    """Wrapper around scipy.optimize.fmin_l_bfgs_b"""
    x_flat, unravel_pytree = flatten_util.ravel_pytree(x0)

    def func_wrapper(x_flat):
        x = unravel_pytree(x_flat)
        value, grad = value_and_grad_func(x)
        return value, flatten_util.ravel_pytree(grad)[0]
    
    x, f, d = fmin_l_bfgs_b(func_wrapper, x_flat, **kwargs)
    return unravel_pytree(x), f, d


def estimate_B(X, B_Sigma_D_init):
    # value and grad function
    loss_partial = lambda B_Sigma_D: negative_log_likelihood(*B_Sigma_D, X)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_partial))

    # bounds: D and Sigma must be positive
    m, p, _ = X.shape
    bounds = [(None, None)] * m * p**2 + [(0.0, None)] * 2 * m * p

    # optimization
    B_Sigma_D_est, _, _ = jaxmin_l_bfgs_b(
        value_and_grad_fn, B_Sigma_D_init, bounds=bounds)
    
    B_est, Sigma_est, D_est = B_Sigma_D_est
    return B_est, Sigma_est, D_est

# %%
# Estimate parameters with LBFGS
B_init = jnp.zeros((m, p, p))
Sigma_init = jnp.ones((m, p))
D_init = jnp.ones((m, p))
B_Sigma_D_init = (B_init, Sigma_init, D_init)
B_est, Sigma_est, D_est = estimate_B(X, B_Sigma_D_init)

# Evaluate the NLL at estimated parameters
est_NLL = negative_log_likelihood(B_est, Sigma_est, D_est, X)
print(f"NLL at estimated params : {est_NLL}")

# %%
