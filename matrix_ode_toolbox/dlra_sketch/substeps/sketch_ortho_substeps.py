"""
Substep RHS builders for sketch DLRA — orthogonal projection on
non-orthonormal bases P, W:
    projector = (P^H P)^{-1} P^H

Used by the orthogonal sketch BUG / projector-splitting integrators.
"""

import scipy.linalg as la

from matrix_ode_toolbox.structures.sylvester_like_ode import SylvesterLikeOde


def _make_K_sketch_ortho_rhs(problem, W0):
    """Build K-sketch-ortho-step RHS: dK/dt = F(t, K W0^*) (W0^*W0)^{-1} W0^*."""
    WtW_inv_Wt = la.lstsq(W0.T.conj().dot(W0), W0.T.conj())[0]  # (r, n)
    R = WtW_inv_Wt.T.conj()                                     # (n, r)
    Wh = W0.T.conj()

    if isinstance(problem, SylvesterLikeOde):
        A = problem.A
        Br = Wh.dot(problem.B.dot(R))
        G = problem.G

        def rhs(t, K):
            return A.dot(K) + K.dot(Br) + G(t, K.dot(Wh)).dot(R)
        return rhs

    def rhs(t, K):
        return problem.ode_F(t, K.dot(Wh)).dot(R)
    return rhs


def _make_L_sketch_ortho_rhs(problem, P0):
    """Build L-sketch-ortho-step RHS: dL/dt = F(t, P0 L^*)^* (P0^*P0)^{-1} P0^*."""
    PtP_inv_Pt = la.lstsq(P0.T.conj().dot(P0), P0.T.conj())[0]  # (r, m)
    R = PtP_inv_Pt.T.conj()                                     # (m, r)

    if isinstance(problem, SylvesterLikeOde):
        Ar = problem.B.T.conj()
        Br = P0.T.conj().dot(problem.A.T.conj().dot(R))
        G = problem.G

        def rhs(t, L):
            return Ar.dot(L) + L.dot(Br) + G(t, P0.dot(L.T.conj())).T.conj().dot(R)
        return rhs

    def rhs(t, L):
        return problem.ode_F(t, P0.dot(L.T.conj())).T.conj().dot(R)
    return rhs


def _make_S_sketch_ortho_rhs(problem, P0, W0):
    """Build S-sketch-ortho-step RHS:
    dS/dt = (P^*P)^{-1} P^* F(t, P S W^*) W (W^*W)^{-1}.
    """
    PtP_inv_Pt = la.lstsq(P0.T.conj().dot(P0), P0.T.conj())[0]  # (r, m)
    WtW_inv_Wt = la.lstsq(W0.T.conj().dot(W0), W0.T.conj())[0]  # (r, n)
    R_right = WtW_inv_Wt.T.conj()                               # (n, r)
    Wh = W0.T.conj()

    if isinstance(problem, SylvesterLikeOde):
        Ar = PtP_inv_Pt.dot(problem.A.dot(P0))
        Br = Wh.dot(problem.B.dot(R_right))
        G = problem.G

        def rhs(t, S):
            return Ar.dot(S) + S.dot(Br) + PtP_inv_Pt.dot(G(t, P0.dot(S.dot(Wh))).dot(R_right))
        return rhs

    def rhs(t, S):
        return PtP_inv_Pt.dot(problem.ode_F(t, P0.dot(S.dot(Wh))).dot(R_right))
    return rhs


def _make_minus_S_sketch_ortho_rhs(problem, P0, W0):
    """Negation of S-sketch-ortho-step (used by projector splitting)."""
    s_rhs = _make_S_sketch_ortho_rhs(problem, P0, W0)

    def rhs(t, S):
        return -s_rhs(t, S)
    return rhs
