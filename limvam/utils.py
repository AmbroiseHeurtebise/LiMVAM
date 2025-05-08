import numpy as np


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
