"""
Orthogonal Sketch Projector Splitting method for sketch DLRA.

Uses RGS for basis orthogonalization and orthogonal projection
(P^HP)^{-1}P^H in the substep ODEs.

Supports Lie-Trotter (order 1) and Strang (order 2) splitting.

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
    _make_minus_S_sketch_ortho_rhs,
)


class OrthoSketchProjectorSplitting(SketchDlraSolver):
    """Orthogonal Sketch Projector Splitting (sKSL) integrator.

    Uses randomized Gram-Schmidt for basis extraction and orthogonal
    projection (P^HP)^{-1}P^H in the K, -S, and L substep ODEs.

    Supports order 1 (Lie-Trotter) and order 2 (Strang) splitting.
    """

    name = 'Orthogonal Sketch Projector Splitting'

    def __init__(self,
                 matrix_ode: MatrixOde,
                 nb_substeps: int = 1,
                 sketch_matrices: tuple[ndarray, ndarray] = (None, None),
                 order: int = 1,
                 substep_kwargs: dict = {'solver': 'automatic', 'nb_substeps': 1},
                 rgs_kwargs: dict = {'do_postprocess': True, 'mode': 'full'},
                 **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps, sketch_matrices, **extra_args)
        self.order = order
        self.substep_kwargs = substep_kwargs
        self.rgs_kwargs = rgs_kwargs
        if order == 1:
            self.splitting_name = 'Lie-Trotter'
        elif order == 2:
            self.splitting_name = 'Strang'
        else:
            raise ValueError("order must be 1 or 2.")

    @property
    def info(self) -> str:
        info = f'{self.name}\n'
        info += f'-- {self.nb_substeps} substep(s)\n'
        info += f'-- Sketch matrices shape: {self.sketch_matrices[0].shape} and {self.sketch_matrices[1].shape}\n'
        info += f'-- Projection: orthogonal (P^HP)^{{-1}}P^H\n'
        info += f'-- {self.splitting_name} splitting (order {self.order})\n'
        for key, value in self.substep_kwargs.items():
            info += f'-- {key}: {value}\n'
        return info

    def stepper(self, t_subspan: tuple, Y0: SketchSVD) -> SketchSVD:
        assert len(t_subspan) == 2
        assert isinstance(Y0, SketchSVD)
        if self.order == 1:
            return self.sKSL1(t_subspan, Y0)
        else:
            return self.sKSL2(t_subspan, Y0)

    def sKSL1(self, t_subspan: tuple, Y0: SketchSVD) -> SketchSVD:
        """Lie-Trotter splitting: K -> (-S) -> L."""
        Theta, Omega = self.sketch_matrices
        P0, S0, W0 = Y0.P, Y0.S, Y0.W
        problem = self.matrix_ode

        # K-STEP
        K0 = P0.dot(S0)
        K_rhs = _make_K_sketch_ortho_rhs(problem, W0)
        K1 = solve_matrix_ivp(SubstepProblem(K_rhs, K0.shape, 'K_ortho'), t_subspan, K0, **self.substep_kwargs)
        P1, _ = randomized_gram_schmidt(K1, Theta, **self.rgs_kwargs)
        S1_hat = la.lstsq(P1.T.conj().dot(P1), P1.T.conj().dot(K1))[0]

        # (-S)-STEP
        mS_rhs = _make_minus_S_sketch_ortho_rhs(problem, P1, W0)
        S0_tilde = solve_matrix_ivp(SubstepProblem(mS_rhs, S1_hat.shape, 'minus_S_ortho'), t_subspan, S1_hat, **self.substep_kwargs)

        # L-STEP
        L0 = W0.dot(S0_tilde.T.conj())
        L_rhs = _make_L_sketch_ortho_rhs(problem, P1)
        L1 = solve_matrix_ivp(SubstepProblem(L_rhs, L0.shape, 'L_ortho'), t_subspan, L0, **self.substep_kwargs)
        W1, _ = randomized_gram_schmidt(L1, Omega, **self.rgs_kwargs)
        S1 = la.lstsq(W1.T.conj().dot(W1), W1.T.conj().dot(L1))[0].T.conj()

        return SketchSVD(P1, S1, W1)

    def sKSL2(self, t_subspan: tuple, Y0: SketchSVD) -> SketchSVD:
        """Strang splitting: K/2 -> (-S)/2 -> L -> (-S)/2 -> K/2."""
        Theta, Omega = self.sketch_matrices
        P0, S0, W0 = Y0.P, Y0.S, Y0.W
        problem = self.matrix_ode
        t0, t1 = t_subspan
        t_half = t0 + (t1 - t0) / 2

        # 1. Half K-step (t0 -> t_half)
        K0 = P0.dot(S0)
        K_rhs = _make_K_sketch_ortho_rhs(problem, W0)
        K_half = solve_matrix_ivp(SubstepProblem(K_rhs, K0.shape, 'K_ortho'), (t0, t_half), K0, **self.substep_kwargs)
        P_half, _ = randomized_gram_schmidt(K_half, Theta, **self.rgs_kwargs)
        S_half_hat = la.lstsq(P_half.T.conj().dot(P_half), P_half.T.conj().dot(K_half))[0]

        # 2. Half (-S)-step (t0 -> t_half)
        mS_rhs = _make_minus_S_sketch_ortho_rhs(problem, P_half, W0)
        S_half = solve_matrix_ivp(SubstepProblem(mS_rhs, S_half_hat.shape, 'minus_S_ortho'), (t0, t_half), S_half_hat, **self.substep_kwargs)

        # 3. Full L-step (t0 -> t1)
        L0 = W0.dot(S_half.T.conj())
        L_rhs = _make_L_sketch_ortho_rhs(problem, P_half)
        L_full = solve_matrix_ivp(SubstepProblem(L_rhs, L0.shape, 'L_ortho'), (t0, t1), L0, **self.substep_kwargs)
        W1, _ = randomized_gram_schmidt(L_full, Omega, **self.rgs_kwargs)
        S1 = la.lstsq(W1.T.conj().dot(W1), W1.T.conj().dot(L_full))[0].T.conj()

        # 4. Half (-S)-step (t_half -> t1)
        mS_rhs2 = _make_minus_S_sketch_ortho_rhs(problem, P_half, W1)
        S1_half = solve_matrix_ivp(SubstepProblem(mS_rhs2, S1.shape, 'minus_S_ortho'), (t_half, t1), S1, **self.substep_kwargs)

        # 5. Half K-step (t_half -> t1)
        K1 = P_half.dot(S1_half)
        K_rhs2 = _make_K_sketch_ortho_rhs(problem, W1)
        K1_final = solve_matrix_ivp(SubstepProblem(K_rhs2, K1.shape, 'K_ortho'), (t_half, t1), K1, **self.substep_kwargs)
        P1, _ = randomized_gram_schmidt(K1_final, Theta, **self.rgs_kwargs)
        S1 = la.lstsq(P1.T.conj().dot(P1), P1.T.conj().dot(K1_final))[0]

        return SketchSVD(P1, S1, W1)
