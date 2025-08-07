import numpy as np


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


def correlation_squared(a, b):
    """
    Compute squared correlation matrix between a and b.
    a, b: arrays of shape (m, n)
    Returns scalar score (Frobenius norm of the correlation matrix)
    """
    cov = a @ b.T / a.shape[1]
    std_a = a.std(axis=1, ddof=1)
    std_b = b.std(axis=1, ddof=1)
    corr = cov / np.outer(std_a, std_b)
    return np.sum(corr ** 2)


def find_direction(x, y):
    """
    Compare x → y and y → x using residual–predictor correlation.
    Lower score indicates lower correlation between residuals and predictor.
    x, y: arrays of shape (m, n)
    Returns (score_x_to_y, score_y_to_x)
    """
    r_y_on_x = compute_residuals_with_univariate_OLS(x, y)
    r_x_on_y = compute_residuals_with_univariate_OLS(y, x)
    score_x_to_y = correlation_squared(x, r_y_on_x)
    score_y_to_x = correlation_squared(y, r_x_on_y)
    return score_x_to_y, score_y_to_x


def find_parent_variable(X):
    """
    Identify the root variable (with no parents).
    X: array of shape (m, p, n) for m views, p variables, n samples
    Returns index of root variable
    """
    m, p, n = X.shape
    X_centered = X - X.mean(axis=-1, keepdims=True)

    scores = np.zeros((p, p))

    for i in range(p):
        for j in range(i + 1, p):
            score_ij, score_ji = find_direction(X_centered[:, i], X_centered[:, j])
            scores[i, j] = score_ij
            scores[j, i] = score_ji

    return np.argmin(np.sum(scores, axis=1))
