"""
Orthogonal Sketch BUG method for sketch DLRA.

Uses RGS for basis orthogonalization and orthogonal projection
(P^HP)^{-1}P^H in the substep ODEs.

Author: Benjamin Carrel, Paul Scherrer Institut (PSI), Switzerland
"""

import numpy as np
import scipy.linalg as la
from numpy import ndarray
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.structures.sketch_svd import SketchSVD
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.integrate.substep_problem import SubstepProblem
from low_rank_toolbox.gram_schmidt import randomized_gram_schmidt
from matrix_ode_toolbox.dlra_sketch import SketchDlraSolver
from matrix_ode_toolbox.dlra_sketch.substeps import (
    _make_K_sketch_ortho_rhs,
    _make_L_sketch_ortho_rhs,
    _make_S_sketch_ortho_rhs,
)


class OrthoSketchBUG(SketchDlraSolver):
    """Orthogonal Sketch BUG (unconventional) integrator.

    Uses randomized Gram-Schmidt for basis extraction and orthogonal
    projection (P^HP)^{-1}P^H in the K, L, and S substep ODEs.
    """

    name = 'Orthogonal Sketch BUG'

    def __init__(self,
                 matrix_ode: MatrixOde,
                 nb_substeps: int = 1,
                 sketch_matrices: tuple[ndarray, ndarray] = (None, None),
                 substep_kwargs: dict = {'solver': 'automatic', 'nb_substeps': 1},
                 **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps, sketch_matrices, **extra_args)
        self.substep_kwargs = substep_kwargs

    @property
    def info(self) -> str:
        info = f'{self.name}\n'
        info += f'-- {self.nb_substeps} substep(s)\n'
        info += f'-- Sketch matrices shape: {self.sketch_matrices[0].shape} and {self.sketch_matrices[1].shape}\n'
        info += f'-- Projection: orthogonal (P^HP)^{{-1}}P^H\n'
        for key, value in self.substep_kwargs.items():
            info += f'-- {key}: {value}\n'
        return info

    def stepper(self, t_subspan: tuple, Y0: SketchSVD) -> SketchSVD:
        assert len(t_subspan) == 2
        assert isinstance(Y0, SketchSVD)

        Theta, Omega = self.sketch_matrices
        P0, S0, W0 = Y0.U, Y0.S, Y0.V
        problem = self.matrix_ode

        # K-STEP
        K0 = P0.dot(S0)
        K_rhs = _make_K_sketch_ortho_rhs(problem, W0)
        K1 = solve_matrix_ivp(SubstepProblem(K_rhs, K0.shape, 'K_ortho'), t_subspan, K0, **self.substep_kwargs)
        P1, _ = randomized_gram_schmidt(K1, Theta, do_postprocess=True)

        # L-STEP
        L0 = W0.dot(S0.T.conj())
        L_rhs = _make_L_sketch_ortho_rhs(problem, P0)
        L1 = solve_matrix_ivp(SubstepProblem(L_rhs, L0.shape, 'L_ortho'), t_subspan, L0, **self.substep_kwargs)
        W1, _ = randomized_gram_schmidt(L1, Omega, do_postprocess=True)

        # S-STEP (Galerkin condition + ODE)
        M = la.lstsq(P1.T.conj().dot(P1), P1.T.conj().dot(P0))[0]
        N = la.lstsq(W1.T.conj().dot(W1), W1.T.conj().dot(W0))[0]
        S0_new = M.dot(S0.dot(N.T.conj()))
        S_rhs = _make_S_sketch_ortho_rhs(problem, P1, W1)
        S1 = solve_matrix_ivp(SubstepProblem(S_rhs, S0_new.shape, 'S_ortho'), t_subspan, S0_new, **self.substep_kwargs)

        return SketchSVD(P1, S1, W1)
