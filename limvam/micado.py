"""
Python implementation of the Multi-view ICA-based Causal Discovery algorithm.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from multiviewica import multiviewica
from shica import shica_j, shica_ml


def micado(
    X,
    shared_causal_ordering=True,
    ica_algo="shica_ml",
    max_iter=3000,
    tol=1e-8,
    random_state=None,
    new_find_order_function=False,
    return_full=False,
):
    """
    Implementation of the Multi-view ICA-based Causal Discovery algorithm.

    Parameters
    ----------
    X : ndarray, shape (n_views, n_components, n_samples)
        Training data, where ``n_views`` is the number of views, 
        ``n_components`` is the number of components, and ``n_samples`` is
        the number of samples.

    shared_causal_ordering : bool (default=True)
        Whether we estimate a causal order common to all views, or
        one causal order per view.

    ica_algo : string, optional (default="shica_ml")
        The multiview ICA algorithm used in the first step.
        It can be either ``shica_ml``, ``shica_j``, or ``multiviewica``.
        Here are the default parameters of:
            ``shica_ml``: max_iter=3000; tol=1e-8
            ``shica_j``: max_iter=10000; tol=1e-5
            ``multiviewica``: max_iter=1000; tol=1e-3.

    max_iter : int, optional (default=3000)
        The maximum number of iterations of the multiview ICA algorithm.

    tol : float, optional (default=1e-8)
        The tolerance parameter of the multiview ICA algorithm.

    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.

    new_find_order_function : bool (default=False)
        Whether we use a new heuristic function to find P and T from B,
        or the one implemented in the original LiNGAM.

    return_full : bool (default=False)
        Whether we return all parameters or only the most important ones.

    Returns
    -------
    B : ndarray, shape (n_views, n_components, n_components)
        Causal effect matrices.

    T : ndarray, shape (n_views, n_components, n_components)
        Causal effects represented by matrices T[i] as close as possible to
        strictly lower triangular.

    P : ndarray, shape (n_components, n_components) or (n_views, n_components, n_components)
        Causal order(s) represented by a permutation matrix or multiple permutation 
        matrices, depending on ``shared_causal_ordering``.

    S_avg: ndarray, shape (n_components, n_samples)
        Source estimates. Only returned if return_full==True.

    W: ndarray, shape (n_views, n_components, n_components)
        Unmixing matrices found by the multiview ICA algorithm. 
        Only returned if return_full==True.
    """
    m, p, n = X.shape
    
    # Step 1: use a multiview ICA algorithm
    if ica_algo == "shica_ml":
        W, _, S_avg = shica_ml(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "shica_j":
        W, _, S_avg = shica_j(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "multiviewica":
        _, W, S_avg = multiviewica(
            X, max_iter=max_iter, tol=tol, random_state=random_state)
    else:
        raise ValueError(
            "ica_algo should be either 'shica_ml', 'shica_j', or 'multiviewica'")

    # Step 2: find permutation Q
    W_inv = 1 / np.sum([np.abs(Wi.T) for Wi in W], axis=0)  # shape (p, p)
    _, index = linear_sum_assignment(W_inv)
    QW = np.array([Wi[index] for Wi in W])

    # Step 3: scaling
    D = np.array([np.diag(Wi) for Wi in QW])[:, :, np.newaxis]  # shape (m, p, 1)
    DQW = QW / D

    # Step 4: causal effects
    B = np.array([np.eye(p)] * m) - DQW  # B is not lower triangular

    # Step 5: estimate causal order(s) with a simple method
    if new_find_order_function:
        find_permutation = find_order
    else:
        find_permutation = _estimate_causal_order
    if shared_causal_ordering:
        B_avg = np.mean(np.abs(B), axis=0)
        order = find_permutation(B_avg)
        P = np.eye(p)[order]
        T = P @ B @ P.T
    else:
        P = np.zeros((m, p, p))
        for i in range(m):
            order = find_permutation(np.abs(B[i]))
            P[i] = np.eye(p)[order]
        T = np.array([Pi @ Bi @ Pi.T for Pi, Bi in zip(P, B)])

    if return_full:
        return B, T, P, S_avg, W
    return B, T, P


def find_order(B):
    """
    This function finds a permutation matrix P such that P @ B @ P.T
    is as close as possible to strictly lower triangular.

    Parameters
    ----------
    B : ndarray, shape (p, p)
        Causal effect matrix, whose rows and columns will be permuted
        by a permutation P. We assume that ``B`` is such that 
        P @ B @ P.T is close to strictly lower triangular.

    Returns
    -------
    order: ndarray, shape (p,)
        ``order`` represents the permutation in P.
    """
    p = len(B)
    B_sort = np.sort(B, axis=1)[:, ::-1]
    B_argsort = np.argsort(B_sort, axis=0)
    order = []
    for i in range(p):
        col = B_argsort[:, i]
        available_id = ~np.isin(col, order)
        first_id = np.argmax(available_id)
        order.append(col[first_id])
    return order


def _search_causal_order(matrix):
    """
    Obtain a causal order from the given matrix strictly.
    This function comes from https://github.com/cdt15/lingam/blob/master/lingam/ica_lingam.py

    Parameters
    ----------
    matrix : array-like, shape (n_features, n_samples)
        Target matrix.

    Return
    ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = []

    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    while 0 < len(matrix):
        # find a row all of which elements are zero
        row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
        if len(row_index_list) == 0:
            break

        target_index = row_index_list[0]

        # append i to the end of the list
        causal_order.append(original_index[target_index])
        original_index = np.delete(original_index, target_index, axis=0)

        # remove the i-th row and the i-th column from matrix
        mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
        matrix = matrix[mask][:, mask]

    if len(causal_order) != row_num:
        causal_order = None

    return causal_order


def _estimate_causal_order(matrix):
    """
    Obtain a lower triangular from the given matrix approximately.
    This function comes from https://github.com/cdt15/lingam/blob/master/lingam/ica_lingam.py

    Parameters
    ----------
    matrix : array-like, shape (n_features, n_samples)
        Target matrix.

    Return
    ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = None

    # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    for i, j in pos_list[initial_zero_num:]:
        causal_order = _search_causal_order(matrix)
        if causal_order is not None:
            break
        else:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

    return causal_order
