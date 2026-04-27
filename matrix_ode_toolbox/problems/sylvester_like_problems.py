"""
Sylvester-like test problems used in the sketch DLRA paper.

Author: Benjamin Carrel, University of Geneva, 2022
"""

import numpy as np
import scipy.linalg as la
from matrix_ode_toolbox import SylvesterLikeOde
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox.utils import laplacian_1d_dx2, centered_1d_dx2


def make_fokker_planck_2D_constant_energy(size):
    """
    Make a 2D Fokker-Planck equation with constant-energy nonlinear field.

    Conservative discretization of the nonlinear drift:
    G(t, Y) = -Dx(M1 * Y) - (M2 * Y) Dy^T.

    Parameters
    ----------
    size : int
        Size of the discretization.

    Returns
    -------
    ode : SylvesterLikeOde
        The Fokker-Planck equation in Sylvester-like ODE structure.
    X0 : ndarray
        The initial value.
    """
    sigma = 2
    f0 = lambda x, y: (np.exp(np.sin(x - y)**2) + np.sin(x + y)**2)
    mu1 = lambda x, y: (np.sin(y) - np.sin(x)) * np.cos(y) - (np.exp(np.sin(x)) + 1)
    mu2 = lambda x, y: (np.sin(x) - np.sin(y)) * np.cos(x) - (np.exp(np.sin(y)) + 1)

    nx = size
    ny = size

    xs = np.linspace(0, 2*np.pi, nx)
    ys = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    # Matrix of initial condition (row index = x, column index = y)
    X0 = f0(xs[:, None], ys[None, :])
    X0 = X0 / la.norm(X0, 'fro')

    # Matrices associated to the non linear field (same convention as X)
    M1 = mu1(X, Y)
    M2 = mu2(X, Y)

    nx = len(xs)
    dx = xs[1] - xs[0]
    ny = len(ys)
    dy = ys[1] - ys[0]
    DDx = (sigma**2/2) * laplacian_1d_dx2(nx, dx, periodic=True)
    DDy = (sigma**2/2) * laplacian_1d_dx2(ny, dy, periodic=True)

    Dx = centered_1d_dx2(nx, dx, periodic=True)
    Dy = centered_1d_dx2(ny, dy, periodic=True)

    Dy_dense = Dy.todense()
    Dy_T = np.asarray(Dy_dense.T)

    def G1(t, Y, rows: list = None, cols: list = None):
        if rows is not None and cols is not None:
            dY = - Dx[rows, :].dot(M1[:, cols] * Y[:, cols]) - (M2[rows, :] * Y[rows, :]).dot(Dy_T[:, cols])
        elif rows is not None:
            if isinstance(Y, LowRankMatrix):
                Y = Y.todense()
            dY = - Dx[rows, :].dot(M1 * Y) - (M2[rows, :] * Y[rows, :]).dot(Dy_T)
        elif cols is not None:
            if isinstance(Y, LowRankMatrix):
                Y = Y.todense()
            dY = - Dx.dot(M1[:, cols] * Y[:, cols]) - (M2 * Y).dot(Dy_T[:, cols])
        else:
            if isinstance(Y, LowRankMatrix):
                Y = Y.todense()
            dY = - Dx.dot(M1 * Y) - (M2 * Y).dot(Dy_T)
        return dY

    ode = SylvesterLikeOde(DDx, DDy, G1)
    return ode, X0


def make_allen_cahn(size: int):
    """
    Allen-Cahn equation
        X' = AX + XA^T + X - X^{*3}
        X(0) = X0
    where A is the 1D Laplacian (times epsilon) as stencil
    1/dx^2 [1 -2 1] in csc format, periodic BC.
    """
    epsilon = 0.01
    dx = 2 * np.pi / size

    xs = np.linspace(dx, 2*np.pi - dx, size)
    ys = np.linspace(dx, 2*np.pi - dx, size)

    A = epsilon * laplacian_1d_dx2(size, dx=dx, periodic=True)

    def G(t, X, rows: list = None, cols: list = None):
        if rows is not None and cols is not None:
            return X[rows, :][:, cols] - X[rows, :][:, cols]**3
        elif rows is not None:
            return X[rows, :] - X[rows, :]**3
        elif cols is not None:
            return X[:, cols] - X[:, cols]**3
        else:
            if isinstance(X, LowRankMatrix):
                return X - X.hadamard(X.hadamard(X))
            else:
                return X - X**3

    ode = SylvesterLikeOde(A, A, G)

    u = lambda x, y: (
        (np.exp(-np.tan(x)**2) + np.exp(-np.tan(y)**2))
        * np.sin(x) * np.sin(y)
        / (1 + np.exp(np.abs(1/np.sin(-x/2))) + np.exp(np.abs(1/np.sin(-y/2))))
    )
    X0 = u(xs[:, None], ys[None, ::-1])

    return ode, X0
