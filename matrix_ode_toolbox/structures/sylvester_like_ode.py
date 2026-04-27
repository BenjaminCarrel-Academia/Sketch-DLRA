"""
Author: Benjamin Carrel, University of Geneva, 2023

Sylvester-like ODE structure. Subclass of MatrixOde.
"""

# %% IMPORTATIONS
from __future__ import annotations
import warnings
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Callable
import scipy.sparse.linalg as spala


_DEIM_FALLBACK_WARNED: set[int] = set()


def _warn_deim_fallback(G_fn: Callable) -> None:
    """One-shot actionable warning when G does not support rows=/cols= kwargs.

    DEIM integrators (PRK-DEIM, PERK-DEIM, BUG-DEIM) evaluate G on a small
    submatrix; without kwargs support the full n x n evaluation is formed and
    sliced, cancelling the DEIM speed-up.
    """
    key = id(G_fn)
    if key in _DEIM_FALLBACK_WARNED:
        return
    _DEIM_FALLBACK_WARNED.add(key)
    name = getattr(G_fn, "__qualname__", getattr(G_fn, "__name__", repr(G_fn)))
    warnings.warn(
        (
            f"DEIM slow path: the nonlinear field G ({name}) does not accept "
            "'rows=' / 'cols=' kwargs, so the full n x n evaluation is "
            "computed and then sliced. This cancels the DEIM speed-up, "
            "making PRK-DEIM / PERK-DEIM as slow as (or slower than) their "
            "non-DEIM variants.\n"
            "Fix: make G accept 'rows' and 'cols' kwargs and evaluate only "
            "on the requested slice. Minimal pattern:\n"
            "    def G(t, X, rows=None, cols=None):\n"
            "        Xs = X[rows, :] if rows is not None else X\n"
            "        Xs = Xs[:, cols] if cols is not None else Xs\n"
            "        if isinstance(Xs, LowRankMatrix): Xs = Xs.todense()\n"
            "        return -Xs * np.abs(Xs)   # your pointwise formula\n"
            "This warning is emitted once per G function."
        ),
        UserWarning,
        stacklevel=3,
    )


# %% CLASS SYLVESTER-LIKE
class SylvesterLikeOde(MatrixOde):
    """
    Class for Sylvester-like equations. Subclass of MatrixOde.

    Sylvester-like differential equation :
    X'(t) = A X(t) + X(t) B + G(t, X(t)).
    Initial value given by X(t_0) = X0.

    Typically, A and B are sparse matrices, and G is a non-linear function.

    The linear field is assumed to be stiff, and the non-linear field is assumed to be non-stiff. To change this, edit the stiff_field and non_stiff_field methods.
    """

    #%% ATTRIBUTES
    name = 'Sylvester-like'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    G = MatrixOde.create_parameter_alias(2)

    def __init__(self, A: ndarray | spmatrix | LowRankMatrix, B: ndarray | spmatrix | LowRankMatrix, G: Callable, **kwargs):
        """Sylvester-like differential equation: X'(t) = A X(t) + X(t) B + G(X(t))."""
        # Check inputs
        assert isinstance(A, (ndarray, spmatrix, LowRankMatrix)), "A must be a sparse matrix"
        assert isinstance(B, (ndarray, spmatrix, LowRankMatrix)), "B must be a sparse matrix"
        assert callable(G), "G must be a function"

        # INITIALIZATION
        super().__init__(A, B, G, **kwargs)

    @property
    def shape(self) -> tuple:
        return (self.A.shape[0], self.B.shape[1])

    def ode_F(self, t: float, X: ndarray | spmatrix | LowRankMatrix, rows: list = None, cols: list = None) -> ndarray | spmatrix | LowRankMatrix:
        """Return the right-hand side of the ODE."""
        if rows is not None and cols is not None:
            try:
                G_rc = self.G(t, X, rows=rows, cols=cols)
            except (TypeError, IndexError):
                _warn_deim_fallback(self.G)
                G_rc = self.G(t, X)[rows, :][:, cols]
            return self.A[rows, :].dot(X[:, cols]) + self.B[:, cols].T.dot(X[rows, :].T).T + G_rc
        elif rows is not None:
            try:
                G_r = self.G(t, X, rows=rows)
            except (TypeError, IndexError):
                _warn_deim_fallback(self.G)
                G_r = self.G(t, X)[rows, :]
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A[rows, :], side='opposite', dense_output=True) + self.B.T.dot(X[rows, :].T).T + G_r
            else:
                return self.A[rows, :].dot(X) + self.B.T.dot(X[rows, :].T).T + G_r
        elif cols is not None:
            try:
                G_c = self.G(t, X, cols=cols)
            except (TypeError, IndexError):
                _warn_deim_fallback(self.G)
                G_c = self.G(t, X)[:, cols]
            if isinstance(X, LowRankMatrix):
                return self.A.dot(X[:, cols]) + X.dot(self.B[:, cols], side='right', dense_output=True) + G_c
            else:
                return self.A.dot(X[:, cols]) + self.B[:, cols].T.dot(X.T).T + G_c
        else:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A, side='opposite') + X.dot(self.B) + self.G(t, X)
            else:
                return self.G(t, X) + self.A.dot(X) + self.B.T.dot(X.T).T

    def linear_field(self, t: float, X: ndarray | spmatrix | LowRankMatrix, rows: list = None, cols: list = None) -> ndarray | spmatrix | LowRankMatrix:
        if rows is not None and cols is not None:
            return self.A[rows, :].dot(X[:, cols]) + self.B[:, cols].T.dot(X[rows, :].T).T
        elif rows is not None:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A[rows, :], side='opposite', dense_output=True) + self.B.T.dot(X[rows, :].T).T
            else:
                return self.A[rows, :].dot(X) + self.B.T.dot(X[rows, :].T).T
        elif cols is not None:
            if isinstance(X, LowRankMatrix):
                return self.A.dot(X[:, cols]) + X.dot(self.B[:, cols], side='right', dense_output=True)
            else:
                return self.A.dot(X[:, cols]) + self.B[:, cols].T.dot(X.T).T
        else:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A, side='left') + X.dot(self.B, side='right')
            else:
                return self.A.dot(X) + self.B.T.dot(X.T).T

    def non_linear_field(self, t: float, X: ndarray | spmatrix | LowRankMatrix, **extra_args) -> ndarray | spmatrix | LowRankMatrix:
        return self.G(t, X, **extra_args)

    def stiff_field(self, t: float, X: ndarray | spmatrix | LowRankMatrix, **extra_args) -> ndarray | spmatrix | LowRankMatrix:
        return self.linear_field(t, X, **extra_args)

    def non_stiff_field(self, t: float, Y: ndarray | spmatrix | LowRankMatrix, **extra_args) -> ndarray | spmatrix | LowRankMatrix:
        return self.non_linear_field(t, Y, **extra_args)

    def solve_full_stiff_field(self, t: float, X0: ndarray | spmatrix | LowRankMatrix) -> ndarray | spmatrix | LowRankMatrix:
        "Closed form solution of the stiff field X' = A X + X B and X(0) = X0"
        if isinstance(X0, LowRankMatrix):
            return X0.expm_multiply(self.A, t, side='left').expm_multiply(self.B, t, side='right')
        else:
            X0eB = spala.expm_multiply(self.B.T, X0.T, start=0, stop=t, num=2, endpoint=True).T[-1]
            return spala.expm_multiply(self.A, X0eB, start=0, stop=t, num=2, endpoint=True)[-1]
