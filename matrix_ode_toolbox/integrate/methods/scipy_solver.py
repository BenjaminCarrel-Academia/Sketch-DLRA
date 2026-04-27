# Author: Benjamin Carrel
#         University of Geneva, 2022

# This file contains the class MatrixScipySolver, which is a wrapper around the scipy solver.

# Imports
import numpy as np
from scipy.integrate import solve_ivp
from matrix_ode_toolbox import MatrixOde, Matrix
from matrix_ode_toolbox.integrate import MatrixOdeSolver

# Class
class ScipySolver(MatrixOdeSolver):
    """
    Scipy solver. This is a wrapper around the scipy solver solve_ivp.
    """

    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int = 1, **solve_ivp_args):
        super().__init__(matrix_ode, nb_substeps)
        solve_ivp_args = dict(solve_ivp_args)  # copy to avoid mutating caller's dict
        self.scipy_method = solve_ivp_args.pop('scipy_method', 'RK45')
        self.atol = solve_ivp_args.pop('atol', 1e-12)
        self.rtol = solve_ivp_args.pop('rtol', 1e-12)
        self.solve_ivp_args = solve_ivp_args
        self.name = f'Scipy [{self.scipy_method} - {nb_substeps} substeps]'

    @property
    def info(self) -> str:
        info = f'Scipy solver \n'
        info += f'-- {self.scipy_method} method \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- Relative tolerance: {self.rtol} \n'
        info += f'-- Absolute tolerance: {self.atol} '
        return info

    def stepper(self, t_span: tuple, X0: Matrix):
        """
        This stepper is a wrapper around the scipy solver solve_ivp.

        Parameters
        ----------
        t_span : tuple
            The time span
        X0 : Matrix
            The initial value
        """
        # Flatten the initial value
        shape = X0.shape
        x0 = X0.flatten()

        # Only request the solution at the final time to avoid storing all
        # intermediate steps (which can exceed 100 GB for stiff problems).
        t_eval = self.solve_ivp_args.get('t_eval', [t_span[1]])
        extra_args = {k: v for k, v in self.solve_ivp_args.items() if k != 't_eval'}

        # SOLVE
        sol = solve_ivp(self.matrix_ode.vec_ode,
                        t_span=t_span,
                        y0=x0,
                        method=self.scipy_method,
                        rtol=self.rtol,
                        atol=self.atol,
                        t_eval=t_eval,
                        **extra_args,
                        args=(shape,))
        # When solve_ivp fails to reach t_eval points (e.g. finite-time
        # blowup), sol.y can be a Python list instead of ndarray.
        sol_y = np.asarray(sol.y)
        if sol_y.ndim < 2 or sol_y.shape[1] == 0:
            raise RuntimeError(
                f"solve_ivp failed: {sol.message}. "
                f"No solution points were computed."
            )
        x1 = sol_y[:, -1]
        return x1.reshape(shape)
