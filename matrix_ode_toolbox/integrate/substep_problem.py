"""
Lightweight wrapper that makes a callable RHS look like a MatrixOde for solve_matrix_ivp.

Used by DLRA methods to wrap substep right-hand sides (K-step, L-step, S-step, etc.)
so they can be integrated with the standard solve_matrix_ivp machinery.
"""

import numpy as np


class SubstepProblem:
    """Wraps a callable RHS to look like a MatrixOde for solve_matrix_ivp.

    Parameters
    ----------
    rhs : callable
        A function (t, Y) -> dY that defines the right-hand side.
    shape : tuple
        Shape of the matrix being integrated.
    name : str
        Name for display/debugging purposes.
    """

    def __init__(self, rhs, shape, name="substep"):
        self._rhs = rhs
        self.shape = shape
        self.name = name

    def ode(self, t, Y):
        """Evaluate the right-hand side at (t, Y)."""
        return self._rhs(t, Y)

    def ode_F(self, t, Y, **kwargs):
        """Evaluate the right-hand side at (t, Y). Alias for ``ode``."""
        return self._rhs(t, Y)

    def vec_ode(self, t, y, shape):
        """Vectorized interface: reshape y, evaluate RHS, flatten result."""
        return self._rhs(t, y.reshape(shape)).flatten()

    def copy(self):
        """Return self (SubstepProblems are stateless)."""
        return self

    def __repr__(self):
        return f"SubstepProblem({self.name}, shape={self.shape})"
