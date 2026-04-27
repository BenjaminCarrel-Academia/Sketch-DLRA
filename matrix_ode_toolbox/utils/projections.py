# Tangent space projection methods - migrated from low_rank_toolbox/matrices/svd.py
# These specialized projections (DEIM, sketch) are not in the pip low-rank-toolbox library.

from __future__ import annotations

from low_rank_toolbox import LowRankMatrix, QuasiSVD, SVD
from numpy import ndarray
import numpy as np
import scipy.linalg as la

# Parameters
automatic_truncation = True
default_atol = 100 * np.finfo(float).eps


def project_onto_DEIM_tangent_space(
    X: QuasiSVD,
    Y_u: ndarray,
    Y_v: ndarray,
    Y_uv: ndarray,
    M_u: ndarray,
    M_v: ndarray,
    truncate: bool = automatic_truncation,
) -> QuasiSVD:
    """
    Oblique projection onto the tangent space at X using DEIM.

    The formula is given by:
        P_X Y = M_u Y_u - M_u Y_uv M_v^* + Y_v M_v^*

    Parameters
    ----------
    X : QuasiSVD
        The point at which to project (provides U, V)
    Y_u : ndarray
        Matrix to project interpolated along the rows
    Y_v : ndarray
        Matrix to project interpolated along the columns
    Y_uv : ndarray
        Matrix to project interpolated along the rows and columns
    M_u : ndarray
        Matrix U @ inv(U[S, :])
    M_v : ndarray
        Matrix V @ inv(V[S, :])
    truncate : bool
        Whether to truncate the output
    """
    assert isinstance(Y_u, ndarray), "Y_u must be a numpy array"
    assert isinstance(Y_v, ndarray), "Y_v must be a numpy array"

    M1 = np.column_stack([M_u, Y_v])
    M2 = np.vstack([Y_u - Y_uv.dot(M_v.T.conj()), M_v.T.conj()])
    Q1, R1 = la.qr(M1, mode='economic')
    Q2, R2 = la.qr(M2.T.conj(), mode='economic')
    if truncate:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=default_atol)
    else:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)


def oblique_projection_onto_sketch_tangent_space(
    X: QuasiSVD,
    Y,
    sketch_matrices: tuple[ndarray, ndarray],
    truncate: bool = automatic_truncation,
) -> QuasiSVD:
    """
    Oblique projection of Y onto the sketch tangent space at X.

    Parameters
    ----------
    X : QuasiSVD
        The point at which to project
    Y : ndarray or LowRankMatrix
        Matrix to project
    sketch_matrices : tuple
        (Omega1, Omega2) sketch matrices
    truncate : bool
        Whether to truncate the output
    """
    Ph = X.Uh
    Wh = X.Vh
    Omega1, Omega2 = sketch_matrices
    U1 = la.lstsq(Ph.dot(Omega1.T.conj()), Ph)[0].T.conj()
    if isinstance(Y, LowRankMatrix):
        U2 = Y.dot(Omega2.T.conj(), dense_output=True)
    else:
        U2 = Y.dot(Omega2.T.conj())
    V2h = la.lstsq(Wh.dot(Omega2.T.conj()), Wh)[0]
    if isinstance(Y, LowRankMatrix):
        V1h = Y.dot(Omega1, side='opposite', dense_output=True) - Y.dot(Omega1, side='opposite', dense_output=True).dot(Omega2.T.conj()).dot(V2h)
    else:
        V1h = Omega1.dot(Y) - Omega1.dot(Y.dot(Omega2.T.conj())).dot(V2h)
    M1 = np.column_stack([U1, U2])
    M2 = np.vstack([V1h, V2h])
    Q1, R1 = la.qr(M1, mode='economic')
    Q2, R2 = la.qr(M2.T.conj(), mode='economic')
    if truncate:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=default_atol)
    else:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)


def orthogonal_projection_onto_sketch_tangent_space(
    X: QuasiSVD,
    Y,
    truncate: bool = automatic_truncation,
) -> QuasiSVD:
    """
    Orthogonal projection of Y onto the sketch tangent space at X.

    Parameters
    ----------
    X : QuasiSVD
        The point at which to project (a SketchSVD with non-orthogonal P, W)
    Y : ndarray or LowRankMatrix
        Matrix to project
    truncate : bool
        Whether to truncate the output
    """
    P = X.U
    W = X.V
    Ph = la.lstsq(P.T.conj().dot(P), P.T.conj(), cond=None)[0]
    Wh = la.lstsq(W.T.conj().dot(W), W.T.conj(), cond=None)[0]
    if isinstance(Y, LowRankMatrix):
        YW = Y.dot(W, dense_output=True)
        PhY = Y.dot(Ph, side='opposite', dense_output=True)
    else:
        YW = Y.dot(W)
        PhY = Ph.dot(Y)
    PhYWWh = PhY.dot(W).dot(Wh)
    M1 = np.column_stack([P, YW])
    M2 = np.vstack([PhY - PhYWWh, Wh])
    Q1, R1 = la.qr(M1, mode='economic')
    Q2, R2 = la.qr(M2.T.conj(), mode='economic')
    if truncate:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=default_atol)
    else:
        return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)
