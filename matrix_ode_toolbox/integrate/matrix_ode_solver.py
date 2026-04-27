"""
Author: Benjamin Carrel, University of Geneva, 2022

This file contains the class MatrixOdeSolver, which is a general class for solving matrix ODEs
"""

#%% Imports
import numpy as np
from numpy import ndarray
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox import MatrixOde, Matrix


#%% Class MatrixOdeSolver
class MatrixOdeSolver:
    """
    Matrix ODE solver class.

    How to define a new solver:
    1. Define a new class that inherits from MatrixOdeSolver.
    2. Overload the __init__ method to add the necessary arguments.
    3. Overload the stepper method to define the solver.
    4. Overload the info property to return the solver information.

    See subclasses for examples.
    """

    #%% ATTRIBUTES
    name: str = "Matrix ODE solver"

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int, **kwargs):
        """
        Initialize the matrix solver.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve
        nb_substeps : int
            The number of substeps to use in solve()
        """
        self.matrix_ode = matrix_ode
        self.nb_substeps = nb_substeps
        self.extra_data = kwargs


    @property
    def info(self) -> str:
        "Return the info string."
        info = f"{self.name} \n"
        info += f"-- {self.nb_substeps} substep(s)"
        return info

    def __repr__(self):
        return self.info

    def solve(self, t_span: tuple, X0: Matrix):
        """
        Solve the matrix ODE from t0 to t1, with initial value X0.
        Applies the stepper method nb_substeps times, and returns the final value.
        """
        # VARIABLES
        t0, t1 = t_span
        ts = np.linspace(t0, t1, self.nb_substeps + 1, endpoint=True)
        X = X0
        # LOOP
        for i in np.arange(self.nb_substeps):
            X = self.stepper(tuple(ts[i:i+2]), X)
        return X

    def solve_best_rank(
        self,
        t_span: tuple,
        X0: Matrix,
        rank: int,
        rtol: float = None,
        dense_output: bool = False,
    ) -> Matrix:
        """
        Solve the problem truncated to the given rank.

        Parameters
        ----------
        rank : int
            The rank of the low rank approximation. Truncate the solution to this rank. By default, no truncation.
        rtol : float, optional
            The relative tolerance for the truncation. By default None.
        dense_output : bool, optional
            Whether to return a dense matrix, by default False

        Returns
        -------
        Matrix
            The solution
        """
        # Check the rank of the initial value and truncate if necessary
        if isinstance(X0, LowRankMatrix):
            if X0.rank < rank:
                print(
                    f"Warning: the initial value has rank {X0.rank}, which is smaller than the desired rank {rank}."
                )
            if X0.rank > rank:
                X0 = SVD.from_low_rank(X0).truncate(rank, rtol=rtol)
        else:
            X0 = SVD.truncated_svd(X0, rank, rtol=rtol)
        # Solve
        X1 = self.solve(t_span, X0)
        # Truncate the solution if necessary
        if isinstance(X1, LowRankMatrix):
            if X1.rank > rank:
                X1 = SVD.truncated_svd(X1, rank, rtol=rtol)
        else:
            X1 = SVD.truncated_svd(X1, rank, rtol=rtol)
        if dense_output:
            X1 = X1.todense()
        return X1

    #%% Methods to be overloaded
    def stepper(self, t_span: tuple, X0: Matrix) -> Matrix:
        """
        Compute the next step of the solution.
        Overload this method to define a new solver.

        Parameters
        ----------
        t_span : tuple
            The time interval (t0, t1)
        X0 : Matrix
            The initial value at t0
        extra_args : dict
            Extra arguments

        Returns
        -------
        Matrix
            The solution at t1
        """
        raise NotImplementedError(
            "Stepper is not implemented for the generic solver. Stepper must be overloaded."
        )
