"""
Author: Benjamin Carrel, University of Geneva, 2022

General structure for matrix ODEs
"""


# Imports
from __future__ import annotations
from copy import deepcopy
from numpy import ndarray
import numpy as np
from low_rank_toolbox import LowRankMatrix, SVD, QuasiSVD
from matrix_ode_toolbox.utils.projections import (
    oblique_projection_onto_sketch_tangent_space,
    orthogonal_projection_onto_sketch_tangent_space,
)

Matrix = ndarray | LowRankMatrix

#%% Class
class MatrixOde:
    r"""General matrix ODE structure class. Contains essential methods for matrix ODEs.

    A matrix ODE is of the form :math:`\dot{X}(t) = F(t, X(t))` where :math:`X(t)` is a matrix.

    How to create a specific ODE structure:
    1. Create a new class that inherits from MatrixOdeStructure.
    2. Overload the necessary methods. See the documentation of the methods for more details. See SylvesterOdeStructure for an example.
    """

    #%% ATTRIBUTES
    name = "General"

    #%% FUNDAMENTALS
    def __init__(self, *parameters, **kwargs):
        "Initialize the problem."
        self._parameters = parameters

    def __repr__(self) -> str:
        return (f'{self.name} ODE structure with {len(self._parameters)} parameters.')

    def __call__(self, *args, **kwds):
        return self.ode(*args, **kwds)

    def copy(self):
        "Copy the problem"
        return deepcopy(self) # use deepcopy otherwise some elements might not be copied

    @staticmethod
    def create_parameter_alias(index: int) -> property:
        """Create a property that aliases ``_parameters[index]`` (used by subclasses for A, B, C, etc.)."""
        def getter(self) -> ndarray:
            return self._parameters[index]

        def setter(self, value: ndarray):
            self._parameters[index] = value

        return property(getter, setter)

    def ode(self, t: float, X: Matrix) -> Matrix:
        "Evaluate the ODE right-hand side. Delegates to ode_F."
        return self.ode_F(t, X)

    def vec_ode(self, t: float, x: np.ndarray, shape: tuple) -> np.ndarray:
        "Vectorized form of the ODE for scipy compatibility."
        X = np.reshape(x, shape)
        dX = self.ode_F(t, X)
        return dX.flatten()

    #%% VECTOR FIELDS
    ## General vector field
    def ode_F(self, t: float, X: Matrix, rows: list = None, cols: list = None, **extra_args) -> Matrix:
        """Evaluate the right-hand side F(t, X). Overload in subclasses.

        Parameters
        ----------
        t : float
            Current time.
        X : Matrix
            Current state (dense or low-rank).
        rows : list, optional
            Row indices for DEIM subsampling. Returns F(t, X)[rows, :].
        cols : list, optional
            Column indices for DEIM subsampling. Returns F(t, X)[:, cols].
            When both are given, returns F(t, X)[rows, cols].
        """
        raise NotImplementedError('Cannot compute the ODE. Overload the method "ode_F".')

    def tangent_space_ode_F(self, t: float, X: SVD, truncate: bool = False) -> SVD:
        "Project the ODE onto the tangent space of rank r matrices. The rank is given by the input matrix."
        if not isinstance(X, SVD):
            raise TypeError("X must be a SVD.")
        FX = self.ode_F(t, X)
        PFX = X.project_onto_tangent_space(FX)
        return PFX

    def sketch_tangent_space_ode_F(self, t: float, X: QuasiSVD, sketch_matrices: tuple[ndarray, ndarray], truncate: bool = False, orthogonal: bool = True) -> QuasiSVD:
        "Project the ODE onto the sketch tangent space of rank r matrices. The rank is given by the input matrix."
        if not isinstance(X, QuasiSVD):
            raise TypeError("X must be a QuasiSVD.")
        FX = self.ode_F(t, X)
        if orthogonal:
            PFX = orthogonal_projection_onto_sketch_tangent_space(X, FX, truncate=truncate)
        else:
            PFX = oblique_projection_onto_sketch_tangent_space(X, FX, sketch_matrices=sketch_matrices, truncate=truncate)
        return PFX

    ## Other vector fields
    def linear_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Linear field of the ODE. Specific to a problem. Overload this method."
        raise NotImplementedError('Cannot compute the linear field. Overload the method "linear_field".')

    def non_linear_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Non-linear field of the ODE. Specific to a problem. Overload this method."
        raise NotImplementedError('Cannot compute the non-linear field. Overload the method "non_linear_field".')

    def stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Stiff field of the ODE. Specific to a problem. Overload this method. By default, it is the linear field."
        return self.linear_field(t, Y, **extra_args)

    def non_stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Non-stiff field of the ODE. Specific to a problem. Overload this method. By default, it is the non-linear field."
        return self.non_linear_field(t, Y, **extra_args)
