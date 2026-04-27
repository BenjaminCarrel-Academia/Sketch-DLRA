"""
Substep RHS builders for sketch DLRA — KSL form.

The sketch K-step solves: dK/dt = F(t, K * Omega * V0^*) * V0 * Omega_pinv
The sketch L-step solves: dL/dt = F(t, U * Theta^* * L^*)^* * U * Theta_pinv
"""

from matrix_ode_toolbox.structures.sylvester_like_ode import SylvesterLikeOde


def _make_K_sketch_rhs(problem, V0, Omega, Omega_pinv):
    """Build K-sketch-step RHS."""
    if isinstance(problem, SylvesterLikeOde):
        A = problem.A
        Omega_V0t = Omega.dot(V0.T)
        V0_Omega_pinv = V0.dot(Omega_pinv)
        Br = V0_Omega_pinv.T.conj().dot(problem.B.dot(V0_Omega_pinv))
        G = problem.G

        def rhs(t, K):
            return A.dot(K) + K.dot(Br) + G(t, K.dot(Omega_V0t)).dot(V0_Omega_pinv)
        return rhs

    # Generic fallback
    Omega_V0t = Omega.dot(V0.T)
    V0_Omega_pinv = V0.dot(Omega_pinv)

    def rhs(t, K):
        return problem.ode_F(t, K.dot(Omega_V0t)).dot(V0_Omega_pinv)
    return rhs


def _make_L_sketch_rhs(problem, U0, Theta, Theta_pinv):
    """Build L-sketch-step RHS."""
    if isinstance(problem, SylvesterLikeOde):
        U_Thetat = U0.dot(Theta.T)
        U_Theta_pinv = U0.dot(Theta_pinv)
        Ar = problem.B.T.conj()
        Br = U_Theta_pinv.T.conj().dot(problem.A.T.conj().dot(U_Theta_pinv))
        G = problem.G

        def rhs(t, L):
            return Ar.dot(L) + L.dot(Br) + G(t, U_Thetat.dot(L.T.conj())).T.conj().dot(U_Theta_pinv)
        return rhs

    # Generic fallback
    U_Thetat = U0.dot(Theta.T)
    U_Theta_pinv = U0.dot(Theta_pinv)

    def rhs(t, L):
        return problem.ode_F(t, U_Thetat.dot(L.T.conj())).T.conj().dot(U_Theta_pinv)
    return rhs
