"""
Substep RHS builders for sketch DLRA — oblique sketch projection on
non-orthonormal bases P, W:
    left projector  = (Theta P)^{-1} Theta
    right projector = (Omega W)^{-1} Omega
"""

import numpy as np

from matrix_ode_toolbox.structures.sylvester_like_ode import SylvesterLikeOde


def _make_K_sketch_oblique_rhs(problem, W0, Omega):
    """Build K-sketch-oblique-step RHS:
    dK/dt = F(t, K W0^*) ((Omega W0)^{-1} Omega)^H.
    """
    OW_inv_O = np.linalg.lstsq(Omega.dot(W0), Omega, rcond=None)[0]  # (r, n)
    R = OW_inv_O.T.conj()                                            # (n, r)
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


def _make_L_sketch_oblique_rhs(problem, P0, Theta):
    """Build L-sketch-oblique-step RHS:
    dL/dt = F(t, P0 L^*)^* ((Theta P0)^{-1} Theta)^H.
    """
    TP_inv_T = np.linalg.lstsq(Theta.dot(P0), Theta, rcond=None)[0]  # (r, m)
    R = TP_inv_T.T.conj()                                            # (m, r)

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


def _make_S_sketch_oblique_rhs(problem, P0, W0, Theta, Omega):
    """Build S-sketch-oblique-step RHS:
    dS/dt = (Theta P)^{-1} Theta F(t, P S W^*) ((Omega W)^{-1} Omega)^H.
    """
    TP_inv_T = np.linalg.lstsq(Theta.dot(P0), Theta, rcond=None)[0]  # (r, m)
    OW_inv_O = np.linalg.lstsq(Omega.dot(W0), Omega, rcond=None)[0]  # (r, n)
    R_right = OW_inv_O.T.conj()                                      # (n, r)
    Wh = W0.T.conj()

    if isinstance(problem, SylvesterLikeOde):
        Ar = TP_inv_T.dot(problem.A.dot(P0))
        Br = Wh.dot(problem.B.dot(R_right))
        G = problem.G

        def rhs(t, S):
            return Ar.dot(S) + S.dot(Br) + TP_inv_T.dot(G(t, P0.dot(S.dot(Wh))).dot(R_right))
        return rhs

    def rhs(t, S):
        return TP_inv_T.dot(problem.ode_F(t, P0.dot(S.dot(Wh))).dot(R_right))
    return rhs


def _make_minus_S_sketch_oblique_rhs(problem, P0, W0, Theta, Omega):
    """Negation of S-sketch-oblique-step (used by projector splitting)."""
    s_rhs = _make_S_sketch_oblique_rhs(problem, P0, W0, Theta, Omega)

    def rhs(t, S):
        return -s_rhs(t, S)
    return rhs
