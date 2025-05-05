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
    
    return float(l1 - l2), b_hat


def find_parent_variable(X, steps=500, lr=1e-2):
    m, p_current, _ = X.shape
    
    # Compute log-ratios for each pair of variables
    R = np.zeros((p_current, p_current))
    B = np.zeros((m, p_current, p_current))
    indices = [(i, j) for i in range(p_current) for j in range(p_current) if i != j]  # strictly upper triangular part
    for (i, j) in indices:
        ratio, b_hat = compute_ratio_for_two_variables(X[:, i], X[:, j], steps=steps, lr=lr)
        R[i, j] = ratio
        B[:, i, j] = b_hat
    
    # Penalize negative scores
    scores = np.sum(np.minimum(0, R) ** 2, axis=1)
    
    # Find parent variable
    parent = np.argmin(scores)
    
    # Remove the effect of the parent variable
    B_parent = B[:, parent][:, :, np.newaxis]  # shape (m, p, 1)
    X_parent = X[:, parent][:, np.newaxis, :]  # shape (m, 1, n)
    X -= B_parent * X_parent
    
    return parent, X


def estimate_causal_order(X, steps=500, lr=1e-2):
    p = X.shape[1]
    X_current = X.copy()
    
    order = []
    remaining_indices = np.arange(p)
    while len(order) < p - 1:
        parent, X_current = find_parent_variable(X_current, steps=steps, lr=lr)
        X_current = np.delete(X_current, parent, axis=1)
        order.append(remaining_indices[parent])
        remaining_indices = np.delete(remaining_indices, parent)
    order.append(remaining_indices[0])

    return order
