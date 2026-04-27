"""
Shared substep RHS builders for DLRA methods.

These factory functions build closure-based right-hand sides for the
reduced K-step, L-step, and S-step ODEs used by splitting-based DLRA
integrators (projector splitting / KSL, unconventional / BUG).

Specialized branches for VlasovPoissonOde and SylvesterLikeOde
precompute reduced matrices in the closure; a generic fallback
covers any other MatrixOde subclass.

Author: Benjamin Carrel, University of Geneva, 2022
"""

import numpy as np

from matrix_ode_toolbox.structures.vlasov_poisson_ode import VlasovPoissonOde
from matrix_ode_toolbox.structures.sylvester_like_ode import SylvesterLikeOde


# ===========================================================================
# K-step: dK/dt = F(t, K V^*) V
# ===========================================================================

def _make_K_rhs(problem, V0):
    """Build K-step RHS: dK/dt = F(t, K V0*) V0."""
    if isinstance(problem, VlasovPoissonOde):
        Dx = problem.Dx
        Dvr = V0.T.conj().dot(problem.Dv.dot(V0))
        diagVr = V0.T.conj().dot(problem.diagV.dot(V0))
        sumVt = problem.dv * np.sum(V0.T.conj(), axis=1)

        def rhs(t, K):
            rho = K.dot(sumVt)
            E = problem.electric_field(rho)
            diagE = np.diag(E)
            return - Dx.dot(K.dot(diagVr)) - diagE.dot(K.dot(Dvr))
        return rhs

    if isinstance(problem, SylvesterLikeOde):
        A = problem.A
        Br = V0.T.conj().dot(problem.B.dot(V0))
        G = problem.G
        Vh = V0.T.conj()

        def rhs(t, K):
            return A.dot(K) + K.dot(Br) + G(t, K.dot(Vh)).dot(V0)
        return rhs

    # Generic fallback
    Vh = V0.T.conj()

    def rhs(t, K):
        return problem.ode_F(t, K.dot(Vh)).dot(V0)
    return rhs


# ===========================================================================
# L-step: dL/dt = F(t, U L^*)^* U
# ===========================================================================

def _make_L_rhs(problem, U0):
    """Build L-step RHS: dL/dt = F(t, U0 L*)* U0."""
    if isinstance(problem, VlasovPoissonOde):
        diagV = problem.diagV
        Dv = problem.Dv
        Dxr = U0.T.conj().dot(problem.Dx.T.conj().dot(U0))
        dv = problem.dv

        def rhs(t, L):
            rho = dv * U0.dot(np.sum(L.T.conj(), axis=1))
            E = problem.electric_field(rho)
            diagE = U0.T.conj().dot(np.diag(E).T.conj().dot(U0))
            return - diagV.T.conj().dot(L.dot(Dxr)) - Dv.T.conj().dot(L.dot(diagE))
        return rhs

    if isinstance(problem, SylvesterLikeOde):
        Ar = problem.B.T.conj()
        Br = U0.T.conj().dot(problem.A.T.conj().dot(U0))
        G = problem.G

        def rhs(t, L):
            return Ar.dot(L) + L.dot(Br) + G(t, U0.dot(L.T.conj())).T.conj().dot(U0)
        return rhs

    # Generic fallback
    def rhs(t, L):
        return problem.ode_F(t, U0.dot(L.T.conj())).T.conj().dot(U0)
    return rhs


# ===========================================================================
# S-step: dS/dt = U^* F(t, U S V^*) V
# ===========================================================================

def _make_S_rhs(problem, U0, V0):
    """Build S-step RHS: dS/dt = U0* F(t, U0 S V0*) V0."""
    if isinstance(problem, VlasovPoissonOde):
        Dxr = U0.T.conj().dot(problem.Dx.dot(U0))
        Dvr = V0.T.conj().dot(problem.Dv.dot(V0))
        diagVr = V0.T.conj().dot(problem.diagV.dot(V0))
        sumVt = problem.dv * np.sum(V0.T.conj(), axis=1)

        def rhs(t, S):
            rho = U0.dot(S.dot(sumVt))
            E = problem.electric_field(rho)
            diagE = U0.T.conj().dot(np.diag(E).dot(U0))
            return - Dxr.dot(S.dot(diagVr)) - diagE.dot(S.dot(Dvr))
        return rhs

    if isinstance(problem, SylvesterLikeOde):
        Ar = U0.T.conj().dot(problem.A.dot(U0))
        Br = V0.T.conj().dot(problem.B.dot(V0))
        G = problem.G
        Uh = U0.T.conj()
        Vh = V0.T.conj()

        def rhs(t, S):
            return Ar.dot(S) + S.dot(Br) + Uh.dot(G(t, U0.dot(S.dot(Vh))).dot(V0))
        return rhs

    # Generic fallback
    Uh = U0.T.conj()
    Vh = V0.T.conj()

    def rhs(t, S):
        USVh = np.linalg.multi_dot([U0, S, Vh])
        return np.linalg.multi_dot([Uh, problem.ode_F(t, USVh), V0])
    return rhs


# ===========================================================================
# minus_S-step: negative of S-step (used by projector splitting / KSL)
# ===========================================================================

def _make_minus_S_rhs(problem, U0, V0):
    """Build minus-S-step RHS: dS/dt = -U0* F(t, U0 S V0*) V0."""
    s_rhs = _make_S_rhs(problem, U0, V0)

    def rhs(t, S):
        return -s_rhs(t, S)
    return rhs
