"""
Author: Benjamin Carrel, University of Geneva, 2022

This file contains useful functions for integrating matrix ODEs
"""

# Imports
from matrix_ode_toolbox import MatrixOde, Matrix
import numpy as np
import time
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .methods import *
from tqdm import tqdm
from .matrix_ode_solver import MatrixOdeSolver
from .matrix_ode_solution import MatrixOdeSolution


available_methods = {'explicit_runge_kutta': ExplicitRungeKutta,
                     'scipy': ScipySolver}

# Solve Matrix IVPs
def solve_matrix_ivp(matrix_ode: MatrixOde, 
                     t_span: tuple,
                     initial_value: Matrix,
                     solver: str | MatrixOdeSolver = 'automatic',
                     solver_kwargs: dict = None,
                     nb_substeps: int = 1,
                     t_eval: list = None,
                     dense_output: bool = False,
                     monitor: bool = False,
                     **extra_args) -> MatrixOdeSolution | Matrix:
    """
    Solve the matrix IVP with the chosen method.

    Parameters
    ----------
    matrix_ode: MatrixOde
        The matrix ODE to solve.
    t_span : tuple
        The time interval (t0, t1) where the solution is computed.
    initial_value : Matrix
        The initial value
    solver : str | MatrixOdeSolver, optional
        The solver to use, by default 'automatic'
    solver_kwargs : dict, optional
        Additional arguments for the solver, by default {}
    nb_substeps : int, optional
        The number of substeps to use, by default 1. This can be given in the solver_kwargs.
    t_eval : list, optional
        The time points where to evaluate the solution, by default None. If None, only the final value is returned.
    dense_output : bool, optional
        Whether to return a dense output, by default False. (The output can sometimes be low-rank)
    monitor : bool, optional
        Whether to monitor the computation, by default False.
    extra_args : dict
        Extra arguments

    Returns
    -------
    solution: MatrixOdeSolution | Matrix
        The solution of the matrix ODE.
    """
    # Merge solver_kwargs and extra_args
    if solver_kwargs is None:
        solver_kwargs = {}
    solver_kwargs = {**solver_kwargs, **extra_args}

    # Ensure retro-compatibility with nb_substeps
    if 'nb_substeps' not in solver_kwargs:
        solver_kwargs['nb_substeps'] = nb_substeps

    # Get the correct solver
    if isinstance(solver, MatrixOdeSolver):
        pass  # already instantiated
    elif isinstance(solver, str):
        if solver == 'automatic':
            solver = ScipySolver
        elif solver not in available_methods:
            raise ValueError(f'Unknown method {solver}.')
        else:
            solver = available_methods[solver]
        solver = solver(matrix_ode, **solver_kwargs)
    else:
        # Assume it's a solver class
        solver = solver(matrix_ode, **solver_kwargs)

    # Single output case   
    if t_eval is None:
        X1 = solver.solve(t_span, initial_value)
        if dense_output and isinstance(X1, LowRankMatrix):
            return X1.todense()
        return X1
    
    # Other cases   
    ## Process t_eval
    t_eval = np.array(t_eval)
    if t_eval[0] != t_span[0]:
        t_eval = np.concatenate([[t_span[0]], t_eval])
    if t_eval[-1] != t_span[1]:
        t_eval = np.concatenate([t_eval, [t_span[1]]])

    ## Preallocate
    n = len(t_eval)
    Xs = np.empty(n, dtype=object)

    ## Monitor
    if monitor:
        print('----------------------------------------')
        print(f'{solver.info}')
        loop = tqdm(np.arange(n-1), desc=f'Solving matrix ODE')
    else:
        loop = np.arange(n-1)
    
    ## Solve
    Xs[0] = initial_value
    computation_time = np.zeros(n-1)
    for i in loop:
        c0 = time.time()
        Xs[i+1] = solver.solve((t_eval[i], t_eval[i+1]), Xs[i])
        computation_time[i] = time.time() - c0

    ## Return
    if dense_output:
        for i in np.arange(n):
            if isinstance(Xs[i], LowRankMatrix):
                Xs[i] = Xs[i].todense()
    return MatrixOdeSolution(matrix_ode, t_eval, Xs, computation_time, **solver.extra_data)
