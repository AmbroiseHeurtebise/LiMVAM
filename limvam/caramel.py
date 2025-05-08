import numpy as np
import jax
import jax.numpy as jnp
from jax import flatten_util
from scipy.optimize import fmin_l_bfgs_b
from functools import partial
from .utils import _estimate_causal_order


class OptimHistory:
    def __init__(self):
        self.B = []
        self.loss = []

    def __call__(self, B_now, loss_now):
        self.B.append(B_now)
        self.loss.append(loss_now)


def logpdf(x):
    y = -jnp.abs(x)
    return y - 2. * jnp.log1p(jnp.exp(y))


def mv_notears_penalty_shared_ordering(B):
    _, n_components, _ = B.shape
    B_sum = jnp.sum(jnp.array([Bi * Bi for Bi in B]), axis=0)
    return jnp.trace(jax.scipy.linalg.expm(B_sum)) - n_components


def mv_notears_penalty_multiple_orderings(B):
    _, n_components, _ = B.shape
    return jnp.sum(jnp.array([jnp.trace(jax.scipy.linalg.expm(Bi * Bi)) - n_components for Bi in B]))


def loss_function(B, X, lambda_pen=1., shared_causal_ordering=True):
    n_views, n_components, _ = X.shape
    W = jnp.eye(n_components) - B
    S = jnp.array([Wi @ Xi for Wi, Xi in zip(W, X)])
    # likelihood
    loss = -jnp.mean(logpdf(S)) * n_components * n_views
    loss -= jnp.sum(jnp.linalg.slogdet(W)[1])
    # multi-view NOTEARS penalty
    if shared_causal_ordering:
        loss += lambda_pen * mv_notears_penalty_shared_ordering(B)
    else:
        loss += lambda_pen * mv_notears_penalty_multiple_orderings(B)
    return loss


def jaxmin_l_bfgs_b(value_and_grad_func, x0, history=None, X=None, lambda_pen=None, **kwargs):
    """Wrapper around scipy.optimize.fmin_l_bfgs_b"""
    x_flat, unravel_pytree = flatten_util.ravel_pytree(x0)

    def func_wrapper(x_flat):
        x = unravel_pytree(x_flat)
        value, grad = value_and_grad_func(x)
        return value, flatten_util.ravel_pytree(grad)[0]

    def callback_func(x_flat):
        if history is not None:
            B_now = unravel_pytree(x_flat)
            loss_now = loss_function(B_now, X, lambda_pen)
            history(B_now, loss_now)
    
    x, f, d = fmin_l_bfgs_b(func_wrapper, x_flat, callback=callback_func, **kwargs)
    return unravel_pytree(x), f, d


def estimate_B(X, B_init, lambda_pen, shared_causal_ordering=True, use_callback=False):
    # value and grad function
    loss_partial = partial(
        loss_function, X=X, lambda_pen=lambda_pen, shared_causal_ordering=shared_causal_ordering)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_partial))
    # callback
    if use_callback:
        history = OptimHistory()
    else:
        history = None
    # optimization
    B_est, loss, _ = jaxmin_l_bfgs_b(
        value_and_grad_fn, B_init, history=history, X=X, lambda_pen=lambda_pen)
    return B_est, history


def caramel(
    X,
    lambda_pen=1.,
    shared_causal_ordering=True,
    use_callback=False,
):
    n_views, n_components, _ = X.shape
    
    # Estimate B with LBFGS
    B_init = jnp.zeros((n_views, n_components, n_components))
    B, history = estimate_B(
        X,
        B_init=B_init,
        lambda_pen=lambda_pen,
        shared_causal_ordering=shared_causal_ordering,
        use_callback=use_callback,
    )

    # Estimate causal order P
    if shared_causal_ordering:
        B_avg = np.mean(np.abs(B), axis=0)
        order = _estimate_causal_order(B_avg)
        P = np.eye(n_components)[order]
        T = P @ B @ P.T
    else:
        P = np.zeros((n_views, n_components, n_components))
        for i in range(n_views):
            order = _estimate_causal_order(np.abs(B[i]))
            P[i] = np.eye(n_components)[order]
        T = np.array([Pi @ Bi @ Pi.T for Pi, Bi in zip(P, B)])
    
    return B, T, P, history
