"""
Entry point for solving DLRA problems with the unconventional (BUG) integrator.
"""

import time
import numpy as np
from tqdm import tqdm
from typing import Tuple
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolution
from matrix_ode_toolbox.dlra import methods
from low_rank_toolbox import LowRankMatrix


available_dlra_methods = {
    'unconventional': methods.Unconventional,
    'BUG': methods.Unconventional,
}


def _validate_inputs(initial_value, t_span):
    if not isinstance(initial_value, LowRankMatrix):
        raise ValueError(f'Initial value must be a LowRankMatrix, not {type(initial_value)}.')
    if initial_value.rank is None or initial_value.rank == 0:
        raise ValueError('Initial value must have rank > 0.')
    if not isinstance(t_span, tuple) or len(t_span) != 2 or t_span[0] >= t_span[1]:
        raise ValueError(f't_span must be a tuple (t0, t1) with t0 < t1, not {t_span}.')


def solve_dlra(matrix_ode: MatrixOde,
               t_span: Tuple[float, float],
               initial_value: LowRankMatrix,
               dlra_solver='unconventional',
               dlra_kwargs: dict = None,
               t_eval: list = None,
               dense_output: bool = False,
               monitor: bool = False,
               profile: bool = False,
               **extra_args):
    """Solve the DLRA on ``t_span`` with the chosen integrator."""
    if dlra_kwargs is None:
        dlra_kwargs = {'nb_substeps': 1}

    if isinstance(dlra_solver, str):
        solver = available_dlra_methods[dlra_solver](matrix_ode, profile=profile, **dlra_kwargs)
    else:
        solver = dlra_solver(matrix_ode, profile=profile, **dlra_kwargs)

    _validate_inputs(initial_value, t_span)

    # Single-output case
    if t_eval is None:
        Y1 = solver.solve(t_span, initial_value)
        if profile:
            print(solver.profiling_summary())
        if dense_output:
            return Y1.todense()
        return Y1

    # Multi-output case
    t_eval = np.array(t_eval)
    if t_eval[0] != t_span[0]:
        t_eval = np.concatenate([[t_span[0]], t_eval])
    if t_eval[-1] != t_span[1]:
        t_eval = np.concatenate([t_eval, [t_span[1]]])

    n = len(t_eval)
    Ys = np.empty(n, dtype=object)

    if monitor:
        print('----------------------------------------')
        print(f'{solver.info}')
        loop = tqdm(np.arange(n - 1), desc='Solving DLRA')
    else:
        loop = np.arange(n - 1)

    Ys[0] = initial_value
    computation_time = np.zeros(n - 1)
    for i in loop:
        c0 = time.time()
        Ys[i + 1] = solver.solve((t_eval[i], t_eval[i + 1]), Ys[i])
        computation_time[i] = time.time() - c0

    if profile:
        print(solver.profiling_summary())
    if dense_output:
        for i in np.arange(n):
            Ys[i] = Ys[i].todense()

    return MatrixOdeSolution(matrix_ode, t_eval, Ys, computation_time, **solver.extra_data)
