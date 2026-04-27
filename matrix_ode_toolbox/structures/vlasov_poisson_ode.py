"""
Author: Benjamin Carrel, University of Geneva, 2024

Vlasov-Poisson ODE structure. Subclass of MatrixOde.
"""

#%% IMPORTATIONS
from __future__ import annotations
import numpy as np
from scipy.sparse import issparse, spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix, QuasiSVD
from .matrix_ode import MatrixOde


class VlasovPoissonOde(MatrixOde):
    """
    Class for the Vlasov-Poisson equation, subclass of MatrixOde.

    The Vlasov-Poisson equation is given by:
    A' = - D_x A V - E_t A D_v
    where:
    - A(t) = A(t, x_i, v_j)_{i,j} is the distribution function,
    - D_x is the first derivative operator in the x direction,
    - D_v is the first derivative operator in the v direction,
    - V is the velocity field,
    - E_t is the electric field.
    """

    # Attributes
    name = 'Vlasov-Poisson'
    Dx = MatrixOde.create_parameter_alias(0)
    Dv = MatrixOde.create_parameter_alias(1)
    diagV = MatrixOde.create_parameter_alias(2)
    dx = MatrixOde.create_parameter_alias(3)
    dv = MatrixOde.create_parameter_alias(4)

    # Init function
    def __init__(self, Dx: spmatrix | ndarray, Dv: spmatrix | ndarray, diagV: ndarray, dx: float, dv: float, **kwargs):
        if len(diagV.shape) == 1:
            self._diagV_vec = diagV.copy()
            diagV = np.diag(diagV)
        else:
            diagV_bis = np.diag(np.diag(diagV))
            if not np.allclose(diagV, diagV_bis):
                raise ValueError('diagV must be diagonal.')
            self._diagV_vec = np.diag(diagV)
        super().__init__(Dx, Dv, diagV, dx, dv, **kwargs)

    def electric_field(self, rho: ndarray) -> ndarray:
        # Derivative of E
        dxE = np.ones_like(rho) - rho
        # FFT and freq
        fftdxE= np.fft.fft(dxE).flatten()
        n =fftdxE.size
        freq = np.fft.fftfreq(n, d=self.dx).flatten()
        Einter = np.zeros(n, dtype=complex)
        # Integration in frequency space
        Einter[1:] = fftdxE[1:]/(1j*freq[1:]*2*np.pi) # Division by zero avoided
        # Back to physical space
        E=np.fft.ifft(Einter)
        return E.real

    # NOTE: linear_field / non_linear_field decomposition is not provided because
    # the electric field E depends nonlinearly on the full solution A (via FFT of row sums).

    # --- Conservation-quantity helpers ---
    # Advective Vlasov-Poisson conserves mass, L2 norm, and momentum at the
    # continuous level; total energy (kinetic + electric) is also conserved.
    # Physics-level validation lives in tests/test_vlasov_poisson_physics.py.

    def mass(self, A: ndarray) -> float:
        r"""Total mass :math:`\iint f\,dx\,dv`."""
        return float(self.dx * self.dv * np.sum(A))

    def momentum(self, A: ndarray) -> float:
        r"""Total momentum :math:`\iint v\,f\,dx\,dv`."""
        return float(self.dx * self.dv * np.sum(A * self._diagV_vec[None, :]))

    def kinetic_energy(self, A: ndarray) -> float:
        r"""Kinetic energy :math:`\tfrac12 \iint v^2 f\,dx\,dv`."""
        return float(
            0.5 * self.dx * self.dv
            * np.sum(A * (self._diagV_vec ** 2)[None, :])
        )

    def electric_energy(self, A: ndarray) -> float:
        r"""Electric energy :math:`\tfrac12 \int E(\rho)^2\,dx`."""
        rho = self.dv * np.sum(A, axis=1)
        E = self.electric_field(rho)
        return float(0.5 * self.dx * np.sum(E ** 2))

    def total_energy(self, A: ndarray) -> float:
        """Kinetic + electric energy."""
        return self.kinetic_energy(A) + self.electric_energy(A)

    def l2_norm_sq(self, A: ndarray) -> float:
        r""":math:`\iint f^2\,dx\,dv` (squared L^2 norm)."""
        return float(self.dx * self.dv * np.sum(A ** 2))

    def ode_F(self, t: float, A: ndarray | LowRankMatrix, rows: list = None, cols: list = None) -> ndarray | LowRankMatrix:
        if isinstance(A, LowRankMatrix):
            return self._low_rank_ode_F(t, A)
        if issparse(A):
            A = A.toarray()
        rho = self.dv * np.sum(A, axis=1)
        E = self.electric_field(rho)
        # Compute terms ensuring dense output
        term1 = self.Dx.dot(A) if not issparse(self.Dx) else self.Dx.toarray().dot(A)
        term1 = term1.dot(self.diagV)
        term2 = (E[:, None] * A).dot(self.Dv) if not issparse(self.Dv) else (E[:, None] * A).dot(self.Dv.toarray())
        dA = - term1 - term2
        if rows is not None and cols is not None:
            return dA[rows, :][:, cols]
        elif rows is not None:
            return dA[rows, :]
        elif cols is not None:
            return dA[:, cols]
        return dA

    def _low_rank_ode_F(self, t: float, A: LowRankMatrix) -> LowRankMatrix:
        """Low-rank evaluation: A' = -Dx A diagV - diag(E) A Dv. Cost: O(nr) + O(n log n)."""
        U, S, V = A.U, A.S, A.V
        n = U.shape[0]
        # Electric field from row sums: rho = dv * sum(A, axis=1) = dv * U S (V^H 1)
        ones_n = np.ones(n)
        Vh_ones = V.T.conj() @ ones_n  # r-vector
        rho = self.dv * (U @ (S @ Vh_ones))
        E = self.electric_field(rho)
        # Term 1: -Dx A diagV = -Dx U S (diagV_vec .* V)^H
        V1 = V * self._diagV_vec[:, None]  # n × r, each row scaled by diagV
        U1 = self.Dx @ U  # (sparse or dense) @ n×r
        # Term 2: -diag(E) A Dv = -(E .* U) S (Dv^T V)^H
        U2 = E[:, None] * U  # n × r, each row scaled by E
        V2 = self.Dv.T @ V if issparse(self.Dv) else self.Dv.T @ V  # n × r
        # Result: -QuasiSVD(U1, S, V1) - QuasiSVD(U2, S, V2)
        return - QuasiSVD(U1, S, V1) - QuasiSVD(U2, S, V2)
