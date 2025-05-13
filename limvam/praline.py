import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
from jax import lax
import numpy as np
from scipy.linalg import block_diag


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


def compute_ratio_for_two_variables(x, y, steps=1000, lr=1e-2):
    m, _ = x.shape

    b_init = jnp.zeros(m)
    b_hat = optimize_b(x, y, b_init, steps=steps, lr=lr)
    b_hat_reverse = optimize_b(y, x, b_init, steps=steps, lr=lr)
    
    l1 = profile_log_likelihood(b_hat, x, y)
    l2 = profile_log_likelihood(b_hat_reverse, y, x)
    
    return float(l1 - l2), b_hat


def find_parent_variable(X, steps=1000, lr=1e-2):
    m, p_current, _ = X.shape
    
    # Compute log-ratios for each pair of variables
    R = np.zeros((p_current, p_current))
    B = np.zeros((m, p_current, p_current))
    indices = [(i, j) for i in range(p_current) for j in range(p_current) if i != j]
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


def estimate_causal_order(X, steps=1000, lr=1e-2):
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


def estimate_triangular_matrices_Ti(X):
    """
    Estimate the strictly lower-triangular matrices T^i for each of m views,
    given data X of shape (m, p, n), assuming variables are ordered so that
    B^i = T^i (strictly lower-triangular).
    
    Uses one-step Feasible GLS (SUR) per row j = 1..p:
      1. OLS on each view to get residuals.
      2. Estimate cross-view noise covariance Σ_j.
      3. Run GLS to get joint estimates for T^i_{j,1:(j-1)}.
    """
    m, p, n = X.shape
    # Initialize list of T^i estimates
    Ts = [np.zeros((p, p)) for _ in range(m)]
    
    for j in range(p):
        # 1) Collect responses and parent matrices
        Ys = [X[i, j, :] for i in range(m)]                         # length‐m list of (n,)
        Xs_par = [X[i, :j, :].T if j > 0 else np.zeros((n, 0))      # each (n, j)
                  for i in range(m)]
        
        # 2) Initial OLS residuals
        residuals = []
        for i in range(m):
            Xpj, yj = Xs_par[i], Ys[i]
            if j > 0:
                beta_ols, *_ = np.linalg.lstsq(Xpj, yj, rcond=None)
                ei = yj - Xpj.dot(beta_ols)
            else:
                beta_ols = np.zeros(0)
                ei = yj
            residuals.append(ei)
        
        # 3) Estimate Σ_j (m x m)
        Σ_j = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                Σ_j[a, b] = residuals[a].dot(residuals[b]) / n
        
        # 4) Build big design and response
        X_big = block_diag(*Xs_par)        # (n*m) x (j*m)
        Y_big = np.concatenate(Ys, axis=0) # (n*m,)
        
        # 5) Compute weight W = inv(Σ_j) ⊗ I_n
        Σ_j_inv = np.linalg.inv(Σ_j)
        W = np.kron(Σ_j_inv, np.eye(n))
        
        # 6) Feasible GLS estimate
        XtW = X_big.T.dot(W)
        beta_gls = np.linalg.solve(XtW.dot(X_big), XtW.dot(Y_big))
        
        # 7) Assign back into each T^i
        for i in range(m):
            start = i * j
            end = start + j
            Ts[i][j, :j] = beta_gls[start:end]
    
    return np.array(Ts)


def praline(X, steps=1000, lr=1e-2):
    order = estimate_causal_order(X, steps=steps, lr=lr)
    P = np.eye(X.shape[1])[order]
    X_ordered = X[:, order]
    T = estimate_triangular_matrices_Ti(X_ordered)
    B = P.T @ T @ P
    return B, T, P
