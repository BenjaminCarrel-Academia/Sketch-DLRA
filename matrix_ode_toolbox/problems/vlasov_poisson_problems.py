"""
Vlasov-Poisson problems used in the sketch DLRA paper.

Author: Benjamin Carrel, University of Geneva, 2024
"""

from __future__ import annotations
import numpy as np
from matrix_ode_toolbox.structures import VlasovPoissonOde
from matrix_ode_toolbox.utils import spacetime


def _build_velocity_operator(nv, dv, L, dv_order, dv_method):
    """Build the 1D velocity derivative operator as a dense ndarray.

    ``dv_method='fd'`` (default) honours ``dv_order`` and returns a
    centered FD stencil (2nd or 4th order). ``dv_method='spectral'``
    returns the Fourier pseudospectral differentiation matrix on an
    interval of length ``L``; ``dv_order`` is ignored.
    """
    if dv_method == 'spectral':
        return spacetime.fourier_diff_1d(nv, L)
    if dv_method != 'fd':
        raise ValueError(f"dv_method must be 'fd' or 'spectral', got {dv_method!r}")
    if dv_order == 2:
        return spacetime.centered_1d_dx2(nv, dv, periodic=True).todense()
    if dv_order == 4:
        return spacetime.centered_1d_dx4(nv, dv, periodic=True).todense()
    raise ValueError(f"dv_order must be 2 or 4, got {dv_order}")


def make_two_stream(nx: int = 128, nv: int = 128,
                    dx_order: int = 2, dv_order: int = 2,
                    dv_method: str = 'fd'):
    """
    Two-stream instability problem on (Omega_x, Omega_v) = ([0, 10*pi], [-6, 6]).

    Parameters
    ----------
    nx : int, optional
        Number of spatial points, by default 128.
    nv : int, optional
        Number of velocity points, by default 128.
    dx_order : int, optional
        Order of the spatial derivative (2 or 4), by default 2.
    dv_order : int, optional
        Order of the velocity derivative (2 or 4), by default 2.
    dv_method : {'fd', 'spectral'}, optional
        Velocity derivative discretisation; 'fd' (default) honours dv_order.

    Returns
    -------
    ode : VlasovPoissonOde
        Vlasov-Poisson ODE.
    A0 : ndarray
        Initial distribution function.
    """
    v0 = 2.4
    k = 0.2
    alpha = 1e-3

    xmin = 0.0
    xmax = 10 * np.pi
    dx = (xmax - xmin) / nx
    xs = np.linspace(xmin, xmax, nx)
    if dx_order == 2:
        Dx = spacetime.centered_1d_dx2(nx, dx, periodic=True).todense()
    elif dx_order == 4:
        Dx = spacetime.centered_1d_dx4(nx, dx, periodic=True).todense()

    vmin = -6.0
    vmax = 6.0
    dv = (vmax - vmin) / nv
    vs = np.linspace(vmin, vmax, nv)
    Dv = _build_velocity_operator(nv, dv, vmax - vmin, dv_order, dv_method)

    c = 1 / (2 * np.sqrt(2 * np.pi))
    X, V = np.meshgrid(xs, vs)
    A0 = c * (
        np.exp(-0.5 * (V[::-1] - v0)**2)
        + np.exp(-0.5 * (V[::-1] + v0)**2)
    ) * (1 + alpha * np.cos(k * X))
    # Make it periodic
    A0[0, :] = A0[-2, :]
    A0[-1, :] = A0[1, :]
    A0[:, 0] = A0[:, -2]
    A0[:, -1] = A0[:, 1]
    A0 = A0.T

    ode = VlasovPoissonOde(Dx, Dv, vs, dx, dv)
    return ode, A0
