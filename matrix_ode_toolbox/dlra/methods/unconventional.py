"""
Author: Benjamin Carrel, University of Geneva, 2022

Unconventional (BUG) integrator for the DLRA.
See Ceruti and Lubich, 2020.
"""

from low_rank_toolbox import QuasiSVD
import scipy.linalg as la
from matrix_ode_toolbox.dlra import DlraSolver
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.dlra.substeps import _make_K_rhs, _make_L_rhs, _make_S_rhs


class Unconventional(DlraSolver):
    """
    Class for the unconventional DLRA method.
    See Ceruti and Lubich, 2020.
    """

    name = 'Unconventional'

    def __init__(self,
                matrix_ode: MatrixOde,
                nb_substeps: int = 1,
                substep_kwargs: dict = None,
                **extra_args) -> None:
        """Initialize the unconventional (BUG) integrator.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve.
        nb_substeps : int
            Number of substeps per time step.
        substep_kwargs : dict, optional
            Options forwarded to the substep ODE solver.
        """
        super().__init__(matrix_ode, nb_substeps, **extra_args)
        self.substep_kwargs = substep_kwargs if substep_kwargs is not None else {'solver': 'automatic', 'nb_substeps': 1}

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Unconventional (Ceruti & Lubich 2020) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver"
        return info

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        """Perform one step of the unconventional (BUG) method.

        Parameters
        ----------
        t_subspan : tuple
            Time interval (t0, tf).
        Y0 : QuasiSVD
            Initial low-rank value in quasi-SVD form.

        Returns
        -------
        Y1 : QuasiSVD
            Solution at the final time.
        """
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'

        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.V
        problem = self.matrix_ode
        p = self._profiler

        # K-STEP
        K0 = U0.dot(S0)
        with p('K-step'):
            K1 = self._solve_substep(_make_K_rhs(problem, V0), K0.shape, 'K', t_subspan, K0)
        with p('K-step QR'):
            U1, _ = la.qr(K1, mode='economic')
        M = U1.T.conj().dot(U0)

        # L-STEP
        L0 = V0.dot(S0.T.conj())
        with p('L-step'):
            L1 = self._solve_substep(_make_L_rhs(problem, U0), L0.shape, 'L', t_subspan, L0)
        with p('L-step QR'):
            V1, _ = la.qr(L1, mode='economic')
        N = V1.T.conj().dot(V0)

        # S-STEP
        S0 = M.dot(S0.dot(N.T.conj()))
        with p('S-step'):
            S1 = self._solve_substep(_make_S_rhs(problem, U1, V1), S0.shape, 'S', t_subspan, S0)

        # SOLUTION
        return QuasiSVD(U1, S1, V1)
