import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
from jax import lax
import numpy as np


# Residuals: shape (m, n)
def residuals(b, x, y):
    return y - (x * b[:, None])


# Empirical residual covariance: shape (m, m)
def residual_covariance(b, x, y):
    r = residuals(b, x, y)  # (m, n)
    return (r @ r.T) / r.shape[1]


# Profile log-likelihood (up to constant): log det S_e(b)
def profile_loss_b(b, x, y):
    S_e = residual_covariance(b, x, y)
    sign, logdet = jnp.linalg.slogdet(S_e)
    return logdet  # drop the sign: S_e should be pos-def anyway


# Optimizer (Adam)
@jit
def optimize_b(x, y, b_init, steps=500, lr=1e-2):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(b_init)

    @jit
    def step(i, opt_state):
        b = get_params(opt_state)
        g = grad(profile_loss_b)(b, x, y)
        return opt_update(i, g, opt_state)

    opt_state = lax.fori_loop(0, steps, step, opt_state)

    return get_params(opt_state)


def profile_log_likelihood(b, x, y):
    S_x = (x @ x.T) / x.shape[1]
    S_e = residual_covariance(b, x, y)
    return -(jnp.linalg.slogdet(S_x)[1] + jnp.linalg.slogdet(S_e)[1])


def compute_ratio_for_two_variables(x, y, steps=500, lr=1e-2):
    m, _ = x.shape

    b_init = jnp.zeros(m)
    b_hat = optimize_b(x, y, b_init, steps=steps, lr=lr)
    b_hat_reverse = optimize_b(y, x, b_init, steps=steps, lr=lr)
    
    l1 = profile_log_likelihood(b_hat, x, y)
    l2 = profile_log_likelihood(b_hat_reverse, y, x)
    
    return float(l1 - l2)


def estimate_causal_order(x, y, steps=500, lr=1e-2):
    p = x.shape[1]
    
    # Compute log-ratios for each pair of variables
    R = np.zeros((p, p))
    indices = [(i, j) for i in range(p) for j in range(p) if i != j]
    for (i, j) in indices:
        R[i, j] = compute_ratio_for_two_variables(x[:, i], y[:, j], steps=steps, lr=lr)
    
    # Penalize negative scores
    scores = np.sum(np.minimum(0, R) ** 2, axis=1)
    
    # Find causal order
    order = np.argsort(scores)
    
    return order
