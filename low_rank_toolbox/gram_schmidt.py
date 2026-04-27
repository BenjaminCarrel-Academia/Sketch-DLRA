"""
Gram-Schmidt orthogonalization utilities.

Includes classical, modified, and randomized Gram-Schmidt (RGS).
The RGS implementation is sketch-based, following Balabanov and Grigori.
"""

import warnings
import numpy as np

__all__ = [
    "generate_rademacher_matrix",
    "classical_gram_schmidt",
    "modified_gram_schmidt",
    "randomized_gram_schmidt",
]


def generate_rademacher_matrix(k, n):
    """
    Generates a scaled Rademacher matrix for random projection.

    Args:
        k (int): The number of rows (the reduced dimension).
        n (int): The number of columns (the original dimension).

    Returns:
        np.ndarray: A k x n matrix with entries drawn from
        {-1/sqrt(k), +1/sqrt(k)}.
    """
    entries = np.random.choice([-1.0, 1.0], size=(k, n))
    return (1.0 / np.sqrt(k)) * entries


def classical_gram_schmidt(X: np.ndarray, tol=None) -> tuple[np.ndarray, np.ndarray]:
    """Classical Gram-Schmidt orthonormalization."""
    if X.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    n, m = X.shape
    if tol is None:
        tol = np.finfo(X.dtype).eps * max(n, m) * np.linalg.norm(X, 'fro')

    Q = np.zeros((n, m), dtype=X.dtype)
    R = np.zeros((m, m), dtype=X.dtype)

    for j in range(m):
        v_j = X[:, j].copy()
        for i in range(j):
            r_ij = np.dot(Q[:, i], v_j)
            R[i, j] = r_ij
            v_j -= r_ij * Q[:, i]
        r_jj = np.linalg.norm(v_j)
        if r_jj < tol:
            Q[:, j] = 0
            R[j, j] = 0
            continue
        R[j, j] = r_jj
        Q[:, j] = v_j / r_jj

    return Q, R


def modified_gram_schmidt(W: np.ndarray, tol=None) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt orthonormalization."""
    if W.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    n, m = W.shape
    if tol is None:
        tol = np.finfo(W.dtype).eps * max(n, m) * np.linalg.norm(W, 'fro')

    Q = np.zeros((n, m), dtype=W.dtype)
    R = np.zeros((m, m), dtype=W.dtype)

    for j in range(m):
        v_j = W[:, j].copy()
        for i in range(j):
            r_ij = np.dot(Q[:, i], v_j)
            R[i, j] = r_ij
            v_j -= r_ij * Q[:, i]
        r_jj = np.linalg.norm(v_j)
        if r_jj < tol:
            Q[:, j] = 0
            R[j, j] = 0
            continue
        R[j, j] = r_jj
        Q[:, j] = v_j / r_jj

    return Q, R


def randomized_gram_schmidt(X: np.ndarray, S: np.ndarray,
                             do_postprocess: bool = False, tol=None,
                             mode: str = 'full', trim: bool = False
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Randomized Gram-Schmidt (RGS) orthonormalization.

    Sketch-based variant of Gram-Schmidt, following Balabanov and Grigori.

    Parameters
    ----------
    X : (n, k) ndarray
        Input matrix whose columns are to be orthonormalized.
    S : (l, n) ndarray
        Sketch matrix.
    do_postprocess : bool, optional
        Apply a final orthogonalization on the sketched basis.
    tol : float, optional
        Tolerance for the rank-revealing test (defaults to a small
        multiple of machine epsilon).
    mode : {'full', 'rank-revealing'}, optional
        ``'full'`` (default) replaces deficient columns by random
        completion vectors; ``'rank-revealing'`` zeros them out.
    trim : bool, optional
        In ``'rank-revealing'`` mode, drop the deficient columns
        from the output.

    Returns
    -------
    Q : ndarray
        Sketch-orthonormal basis: ``S @ Q`` has orthonormal columns.
    R : ndarray
        Upper-triangular factor such that ``X = Q @ R`` (in exact
        arithmetic).
    """
    if X.ndim != 2 or S.ndim != 2:
        raise ValueError("Input matrices must be 2-dimensional")

    n, k = X.shape
    l, n_check = S.shape

    if n_check != n:
        raise ValueError(
            f"Sketching matrix shape {S.shape} incompatible with input matrix shape {X.shape}"
        )
    if mode not in ('rank-revealing', 'full'):
        raise ValueError(f"Mode must be 'rank-revealing' or 'full', got '{mode}'")
    if k == 0 or n == 0:
        return np.zeros((n, 0)), np.zeros((0, 0))

    if tol is None:
        tol = np.finfo(X.dtype).eps * max(n, k) * np.linalg.norm(X, 'fro')

    X = X.astype(float, copy=True)
    S = S.astype(float, copy=False)

    Q = np.zeros((n, k), dtype=X.dtype)
    Q_hat = np.zeros((l, k), dtype=X.dtype)
    R = np.zeros((k, k), dtype=X.dtype)

    SX = S.dot(X)
    effective_rank = 0

    for j in range(k):
        x_j = X[:, j]
        s_j = SX[:, j]

        if effective_rank > 0:
            r_j = Q_hat[:, :effective_rank].T.dot(s_j)
            q_hat_j_prime = s_j - Q_hat[:, :effective_rank].dot(r_j)
            R[:effective_rank, j] = r_j
        else:
            r_j = np.array([])
            q_hat_j_prime = s_j.copy()

        r_jj = np.linalg.norm(q_hat_j_prime)

        if r_jj > tol:
            R[effective_rank, j] = r_jj
            Q_hat[:, effective_rank] = q_hat_j_prime / r_jj
            if effective_rank > 0:
                q_j_prime = x_j - Q[:, :effective_rank].dot(r_j)
            else:
                q_j_prime = x_j.copy()
            Q[:, effective_rank] = q_j_prime / r_jj
            effective_rank += 1
        else:
            if mode == 'full':
                completion_vec = np.random.randn(n)
                s_completion = S.dot(completion_vec)
                r_completion = Q_hat[:, :effective_rank].T.dot(s_completion)
                q_hat_completion = s_completion - Q_hat[:, :effective_rank].dot(r_completion)
                q_completion_orig = completion_vec - Q[:, :effective_rank].dot(r_completion)
                completion_norm = np.linalg.norm(q_hat_completion)
                if completion_norm < tol:
                    raise ValueError("Failed to generate a suitable orthogonal vector during full mode.")
                R[effective_rank, j] = completion_norm
                Q_hat[:, effective_rank] = q_hat_completion / completion_norm
                Q[:, effective_rank] = q_completion_orig / completion_norm
                effective_rank += 1

    if mode == 'rank-revealing':
        if trim:
            Q = Q[:, :effective_rank]
            Q_hat = Q_hat[:, :effective_rank]
            R = R[:effective_rank, :]
        else:
            Q[:, effective_rank:] = 0
            Q_hat[:, effective_rank:] = 0
            R[effective_rank:, :] = 0
    elif trim:
        warnings.warn("Trimming can only be done in 'rank-revealing' mode. Ignoring trim request.")

    if do_postprocess:
        try:
            S_tilde = S.dot(Q)
            _, R1 = np.linalg.qr(S_tilde, mode='reduced')
            Q = np.linalg.lstsq(R1.T, Q.T)[0].T
            R = R1.dot(R)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Postprocessing failed ({str(e)}). Returning without postprocessing.")

    return Q, R
