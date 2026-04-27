# SketchSVD class - migrated from low_rank_toolbox/matrices/svd.py
# This class is not part of the pip-installable low-rank-toolbox library.

from __future__ import annotations

from low_rank_toolbox import LowRankMatrix, QuasiSVD, SVD
from numpy import ndarray
import numpy as np
from low_rank_toolbox.gram_schmidt import randomized_gram_schmidt


class SketchSVD(QuasiSVD):
    """
    Sketch SVD is a low-rank matrix stored by its SVD: Y = P @ S @ W.T
    where P, W have sketch-orthogonal columns:
     (Omega1 P)^T (Omega1 P) = I
     (Omega2 W)^T (Omega2 W) = I
    and S is invertible (not necessarily diagonal).
    The dimensions are:
    - P: (m, r)
    - S: (r, q)
    - W: (n, q)
    - Omega1 : (r, m)
    - Omega2 : (q, n)

    NOTE: For memory efficiency, the sketch matrices Omega1 and Omega2 are not stored explicitly.
    Instead, the user must provide them when necessary (creation, check orthogonality, projection, etc).
    """

    _format = "SketchSVD"

    # Aliases for the matrices
    P = LowRankMatrix.create_matrix_alias(0)
    S = LowRankMatrix.create_matrix_alias(1)
    W = LowRankMatrix.create_matrix_alias(2, transpose=True, conjugate=True)
    Wh = LowRankMatrix.create_matrix_alias(2)
    Wt = LowRankMatrix.create_matrix_alias(2, conjugate=True)
    Pt = LowRankMatrix.create_matrix_alias(0, transpose=True)
    Ph = LowRankMatrix.create_matrix_alias(0, transpose=True, conjugate=True)

    def __init__(self, P: ndarray, S: ndarray, W: ndarray, **extra_data):
        """
        Create a low-rank matrix stored by its SVD: Y = P @ S @ W.T

        Parameters
        ----------
        P : ndarray
            Left sketch vectors, shape (m, r)
        S : ndarray
            Non-singular matrix, shape (r, q)
        W : ndarray
            Right sketch vectors, shape (n, q)
        """
        assert P.dtype == W.dtype, "P and W must have the same dtype"
        super().__init__(P, S, W, **extra_data)

    def is_sketch_orthogonal(self, sketch_matrices: tuple[ndarray, ndarray]) -> bool:
        """Check if the columns of P and W are sketch-orthogonal"""
        Omega1, Omega2 = sketch_matrices
        return np.allclose(self.P.T.conj().dot(Omega1.T.conj()).dot(Omega1.dot(self.P)), np.eye(self.P.shape[1])) and \
               np.allclose(self.W.T.conj().dot(Omega2.T.conj()).dot(Omega2.dot(self.W)), np.eye(self.W.shape[1]))

    @classmethod
    def from_quasiSVD(cls, mat: QuasiSVD, sketch_matrices: tuple[ndarray, ndarray]) -> SketchSVD:
        """Create a SketchSVD from a QuasiSVD"""
        Omega1, Omega2 = sketch_matrices
        assert Omega1.shape[1] == mat.U.shape[0], "Omega1 has incorrect number of rows"
        assert Omega2.shape[1] == mat.V.shape[0], "Omega2 has incorrect number of rows"
        P, R1 = randomized_gram_schmidt(mat.U, Omega1, do_postprocess=True)
        W, R2 = randomized_gram_schmidt(mat.V, Omega2, do_postprocess=True)
        S = R1.dot(mat.S).dot(R2.T.conj())
        return cls(P, S, W, **mat._extra_data)

    @classmethod
    def from_svd(cls, mat: SVD, sketch_matrices: tuple[ndarray, ndarray]) -> SketchSVD:
        """Create a SketchSVD from an SVD"""
        return cls.from_quasiSVD(mat, sketch_matrices)

    @classmethod
    def from_low_rank(cls, mat: LowRankMatrix, sketch_matrices: tuple[ndarray, ndarray]) -> SketchSVD:
        """Create a SketchSVD from a LowRankMatrix"""
        Y = SVD.from_low_rank(mat)
        return cls.from_svd(Y, sketch_matrices)

    @classmethod
    def from_matrix(cls, matrix, sketch_matrices: tuple[ndarray, ndarray], **extra_data) -> SketchSVD:
        Y = SVD.from_matrix(matrix, **extra_data)
        return cls.from_svd(Y, sketch_matrices)

    @classmethod
    def from_dense(cls, matrix, sketch_matrices: tuple[ndarray, ndarray], **extra_data):
        Y = SVD.from_dense(matrix, **extra_data)
        return cls.from_quasiSVD(Y, sketch_matrices)

    @classmethod
    def truncated_sketch_svd(cls, mat, r: int = None, rtol: float = None, atol: float = None, sketch_matrices: tuple[ndarray, ndarray] = (None, None), **extra_data) -> SketchSVD:
        """Compute a truncated Sketch SVD of rank r"""
        if sketch_matrices[0] is None or sketch_matrices[1] is None:
            raise ValueError("Sketch matrices must be provided")
        Y = SVD.truncated_svd(mat, r=r, rtol=rtol, atol=atol, **extra_data)
        return cls.from_svd(Y, sketch_matrices)
