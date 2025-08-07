import numpy as np


def compute_residual(xi, xj):
    """Residual when xi is regressed on xj (scalar slope)."""
    beta = np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)
    return xi - beta * xj


def find_parent_variable(X):
    m, p, n = X.shape

    # center each time series
    X_centered = X - X.mean(axis=-1, keepdims=True)

    scores = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            # compute residuals for all m views at once
            r = np.empty((m, n))
            for k in range(m):
                r[k] = compute_residual(X_centered[k, i], X_centered[k, j])

            # correlation between residuals and x_i
            std_r = r.std(axis=1, ddof=1)
            std_x = X_centered[:, i].std(axis=1, ddof=1)
            cov = (r @ X_centered[:, i].T) / (n - 1)
            corr = cov / np.outer(std_r, std_x)

            scores[i, j] = np.sum(corr**2)  # Frobenius norm

    # choose variable minimizing column sum of scores
    return np.argmin(np.sum(scores, axis=0))
