'''
Author: Benjamin Carrel, University of Geneva, 2022

General class for implementing DLRA solvers.
'''

#%% Imports
import warnings
import numpy as np
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox import MatrixOde, Matrix
from matrix_ode_toolbox.dlra.profiling import StepTimer


#%% Common class for the DLRA solvers
class DlraSolver:
    '''
    Class for the DLRA solvers

    How to implement a new DLRA method:
    1. Create a new class that inherits from DlraSolver
    2. Create a specific init method that takes the necessary parameters
    3. Implement the method stepper
    Well done! You can now use your method in the solve_dlra function.
    '''

    name = 'Generic DLRA'

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int = 1, profile: bool = False, **extra_args):
        """Initialize the DLRA solver.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve.
        nb_substeps : int
            Number of substeps per time step.
        profile : bool
            If True, enable profiling of substep timings.
        """
        self.matrix_ode = matrix_ode
        self.nb_substeps = nb_substeps
        self.extra_data = {}
        self._profiler = StepTimer(enabled=profile)

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Generic DLRA \n'
        info += f'-- {self.nb_substeps} substep(s)'
        return info

    def __repr__(self) -> str:
        return self.info

    @property
    def timer(self):
        """Return profiling timer dict.

        Returns
        -------
        t : dict
            Dictionary mapping substep names (e.g. 'K-step', 'S-step') to
            cumulative elapsed times in seconds.
        """
        t = self._profiler.timer
        if hasattr(self, '_perk_timer'):
            t.update(self._perk_timer)
        return t

    def profiling_summary(self):
        "Return a formatted profiling summary."
        return self._profiler.summary()

    def solve(self, t_span: tuple, Y0: LowRankMatrix):
        """Solve the DLRA by calling the stepper method over all substeps.

        Parameters
        ----------
        t_span : tuple
            Time interval (t0, tf).
        Y0 : LowRankMatrix
            Initial low-rank approximation.

        Returns
        -------
        Y : LowRankMatrix
            Low-rank solution at the final time.
        """
        # Initialization
        t0, tf = t_span
        ts = np.linspace(t0, tf, self.nb_substeps + 1, endpoint=True)
        Y = Y0
        # Loop over the substeps
        for i in np.arange(self.nb_substeps):
            previous_rank = Y.rank
            Y = self.stepper(tuple(ts[i:i+2]), Y)
            # Check if the rank has changed
            if Y.rank != previous_rank:
                warnings.warn(f'Rank changed from {previous_rank} to {Y.rank} at t = {ts[i+1]}', stacklevel=2)
        return Y

    #%% Substep solver helper
    def _solve_substep(self, rhs, shape, name, t_span, X0):
        """Solve a substep ODE directly, bypassing solve_matrix_ivp dispatch.

        Parameters
        ----------
        rhs : callable
            Right-hand side function rhs(t, X) -> ndarray.
        shape : tuple
            Shape of the unknown matrix.
        name : str
            Label for the substep (e.g. 'K', 'L', 'S').
        t_span : tuple
            Time interval (t0, tf) for the substep.
        X0 : ndarray
            Initial value for the substep.

        Returns
        -------
        X1 : ndarray
            Solution at the final time of the substep.
        """
        from matrix_ode_toolbox.integrate.substep_problem import SubstepProblem
        if not hasattr(self, '_substep_solver_class'):
            self._resolve_substep_solver()
        problem = SubstepProblem(rhs, shape, name)
        solver = self._substep_solver_class(problem, **self._substep_solver_kwargs)
        return solver.solve(t_span, X0)

    def _resolve_substep_solver(self):
        """Resolve substep solver class from self.substep_kwargs (called once, cached)."""
        from matrix_ode_toolbox.integrate.methods import ScipySolver
        from matrix_ode_toolbox.integrate.solve_matrix_ivp import available_methods
        kwargs = dict(self.substep_kwargs)
        solver = kwargs.pop('solver', 'automatic')
        if isinstance(solver, str):
            if solver in ('automatic', 'scipy'):
                self._substep_solver_class = ScipySolver
            else:
                self._substep_solver_class = available_methods[solver]
        else:
            self._substep_solver_class = solver
        self._substep_solver_kwargs = kwargs

    #%% Methods to be overloaded
    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix):
        "Perform one step of the DLRA."
        raise NotImplementedError('The stepper method is not implemented. Overload it in the child class.')

