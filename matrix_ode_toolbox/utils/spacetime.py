"""
Author: Benjamin Carrel, University of Geneva, 2022
path: matrix_ode_toolbox/utils/spacetime.py
File for spacetime discretization
"""

#%% Imports
import numpy as np
import scipy.sparse as sparse

#%% Centered finite difference matrix O(dx^2)

## First order derivative
def centered_1d_dx2(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^2))
    """
    D = sparse.diags([-1.0, 1.0], [-1, 1], shape=(n, n), format='csc') / (2 * dx)
    if periodic:
        D = D.tolil()
        D[0, -1] = -1 / (2 * dx)
        D[-1, 0] = 1 / (2 * dx)
        D = D.tocsc()
    return D

## Second order derivative (Laplacian)
def laplacian_1d_dx2(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete Laplacian matrix in 1D (error O(dx^2))
    """
    DD = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format='csc') / (dx ** 2)
    if periodic:
        DD = DD.tolil()
        DD[0, -1] = 1 / (dx ** 2)
        DD[-1, 0] = 1 / (dx ** 2)
        DD = DD.tocsc()
    return DD

#%% Centered finite difference matrix O(dx^4)

## First order derivative
def centered_1d_dx4(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^4))
    """
    D = sparse.diags([1.0, -8.0, 8.0, -1.0], [-2, -1, 1, 2], shape=(n, n), format='csc') / (12 * dx)
    if periodic:
        D = D.tolil()
        D[0, -2] = 1 / (12 * dx)
        D[0, -1] = -8 / (12 * dx)
        D[1, -1] = 1 / (12 * dx)
        D[-1, 0] = 8 / (12 * dx)
        D[-1, 1] = -1 / (12 * dx)
        D[-2, 0] = -1 / (12 * dx)
        D = D.tocsc()
    return D

## Second order derivative (Laplacian)
def laplacian_1d_dx4(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete Laplacian matrix in 1D (error O(dx^4))
    """
    DD = sparse.diags([-1.0, 16.0, -30.0, 16.0, -1.0], [-2, -1, 0, 1, 2], shape=(n, n), format='csc') / (12 * dx ** 2)
    if periodic:
        DD = DD.tolil()
        DD[0, -2] = -1 / (12 * dx ** 2)
        DD[0, -1] = 16 / (12 * dx ** 2)
        DD[1, -1] = -1 / (12 * dx ** 2)
        DD[-1, 0] = 16 / (12 * dx ** 2)
        DD[-1, 1] = -1 / (12 * dx ** 2)
        DD[-2, 0] = -1 / (12 * dx ** 2)
        DD = DD.tocsc()
    return DD


#%% Fourier pseudospectral first-derivative matrix

def fourier_diff_1d(n: int, L: float) -> np.ndarray:
    """Fourier pseudospectral 1st-derivative matrix for a periodic
    function sampled at ``n`` equispaced points on an interval of
    length ``L``.

    For a trig-interpolant-representable input (no frequency content
    above the Nyquist mode), ``D @ f`` equals ``f'`` at the gridpoints
    to machine precision. The operator is dense (nx n), skew-symmetric,
    and Toeplitz. See Trefethen, *Spectral Methods in MATLAB*, Ch. 3.
    """
    h = 2 * np.pi / n
    I, J = np.indices((n, n))
    diff = I - J
    with np.errstate(divide='ignore', invalid='ignore'):
        if n % 2 == 0:
            D = 0.5 * (-1.0) ** diff / np.tan(diff * h / 2)
        else:
            D = 0.5 * (-1.0) ** diff / np.sin(diff * h / 2)
    np.fill_diagonal(D, 0.0)
    return D * (2 * np.pi / L)


def fourier_laplacian_1d(n: int, L: float) -> np.ndarray:
    """Fourier pseudospectral 2nd-derivative matrix for a periodic
    function sampled at ``n`` equispaced points on an interval of
    length ``L``.

    Built as ``D @ D`` where ``D = fourier_diff_1d(n, L)``. Dense,
    symmetric negative semidefinite, exact to machine precision on any
    trig-interpolant-representable input.
    """
    D = fourier_diff_1d(n, L)
    return D @ D


#%% 2D periodic Poisson solver (FFT)

def electric_field_2d(rho2d: np.ndarray, dx1: float, dx2: float):
    """Solve -Delta phi = rho on a periodic 2D grid and return E = -grad phi.

    Parameters
    ----------
    rho2d : ndarray, shape (n1, n2)
        Charge density on a periodic grid. The zero-mean component is
        enforced by zeroing the DC Fourier coefficient of phi.
    dx1, dx2 : float
        Grid spacings along axes 0 and 1.

    Returns
    -------
    E1, E2 : ndarray of shape (n1, n2), real
        Electric field components E1 = -d phi/d x1, E2 = -d phi/d x2.
    """
    n1, n2 = rho2d.shape
    k1 = 2 * np.pi * np.fft.fftfreq(n1, d=dx1)
    k2 = 2 * np.pi * np.fft.fftfreq(n2, d=dx2)
    K1, K2 = np.meshgrid(k1, k2, indexing='ij')
    Ksq = K1 ** 2 + K2 ** 2
    Ksq[0, 0] = 1.0  # avoid division by zero; DC set to zero below
    rho_hat = np.fft.fft2(rho2d)
    phi_hat = rho_hat / Ksq
    phi_hat[0, 0] = 0.0
    E1 = np.real(np.fft.ifft2(-1j * K1 * phi_hat))
    E2 = np.real(np.fft.ifft2(-1j * K2 * phi_hat))
    return E1, E2
