"""
Python implementation of the Pairwise likelihood RAtios for 
LInear Non-gaussian Estimation (PRaLiNE) algorithm.
"""

import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
from jax import lax
import numpy as np
from .utils import estimate_triangular_matrices_Ti


# Residuals: shape (m, n)
def residuals(b, x, y):
    return y - (x * b[:, None])


# Empirical residual covariance: shape (m, m)
def residual_covariance(b, x, y):
    r = residuals(b, x, y)  # (m, n)
    return (r @ r.T) / r.shape[1]


# Profile log-likelihood (up to constant): log det S_e(b)
def profile_loss_b(b, x, y, eps=1e-5):
    S_e = residual_covariance(b, x, y)
    S_e = S_e + eps * jnp.eye(S_e.shape[0])  # useful when b tends to 0
    sign, logdet = jnp.linalg.slogdet(S_e)
    return logdet  # drop the sign: S_e should be pos-def anyway


# Optimizer (Adam)
@jit
def optimize_b(x, y, b_init, steps=1000, lr=1e-2):
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


def compute_ratio_for_two_variables(x, y, steps=1000, lr=1e-2, method_for_b="LS_regression"):
    m, _ = x.shape

    if method_for_b == "MLE":
        b_init = jnp.zeros(m)
        b_hat = optimize_b(x, y, b_init, steps=steps, lr=lr)
        b_hat_reverse = optimize_b(y, x, b_init, steps=steps, lr=lr)
    elif method_for_b == "LS_regression":
        b_hat = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in range(m)])
        b_hat_reverse = b_hat
    
    l1 = profile_log_likelihood(b_hat, x, y)
    l2 = profile_log_likelihood(b_hat_reverse, y, x)
    
    return float(l1 - l2), b_hat, b_hat_reverse


def find_parent_variable(X, steps=1000, lr=1e-2, method_for_b="LS_regression"):
    m, p_current, _ = X.shape
    
    # Initialize score matrix and coefficients
    R = np.zeros((p_current, p_current))
    B = np.zeros((m, p_current, p_current))
    
    # Compute log-ratios for each pair of variables
    indices = [(i, j) for i in range(p_current) for j in range(p_current) if i < j]
    for (i, j) in indices:
        ratio, b_hat, b_hat_reverse = compute_ratio_for_two_variables(
            X[:, i], X[:, j], steps=steps, lr=lr, method_for_b=method_for_b)
        R[i, j] = ratio
        R[j, i] = -ratio
        B[:, i, j] = b_hat
        B[:, j, i] = b_hat_reverse
    
    # Penalize negative scores
    scores = np.sum(np.minimum(0, R) ** 2, axis=1)
    
    # Find parent variable
    parent = np.argmin(scores)
    
    # Remove the effect of the parent variable
    B_parent = B[:, parent][:, :, np.newaxis]  # shape (m, p, 1)
    X_parent = X[:, parent][:, np.newaxis, :]  # shape (m, 1, n)
    X -= B_parent * X_parent
    
    return parent, X


def estimate_causal_order(X, steps=1000, lr=1e-2, method_for_b="LS_regression"):
    p = X.shape[1]
    X_current = X.copy()
    
    order = []
    remaining_indices = np.arange(p)
    while len(order) < p - 1:
        parent, X_current = find_parent_variable(
            X_current, steps=steps, lr=lr, method_for_b=method_for_b)
        X_current = np.delete(X_current, parent, axis=1)
        order.append(remaining_indices[parent])
        remaining_indices = np.delete(remaining_indices, parent)
    order.append(remaining_indices[0])

    return order


def praline(X, steps=1000, lr=1e-2, method_for_b="LS_regression"):
    order = estimate_causal_order(X, steps=steps, lr=lr, method_for_b=method_for_b)
    P = np.eye(X.shape[1])[order]
    X_ordered = X[:, order]
    T = estimate_triangular_matrices_Ti(X_ordered)
    B = P.T @ T @ P
    return B, T, P
