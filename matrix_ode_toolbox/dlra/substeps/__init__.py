"""
Shared substep RHS builders for DLRA methods.

These builders create closure-based right-hand sides for the K, L, S
substep ODEs of the unconventional (BUG) integrator.
"""

from .kls_substeps import _make_K_rhs, _make_L_rhs, _make_S_rhs, _make_minus_S_rhs
