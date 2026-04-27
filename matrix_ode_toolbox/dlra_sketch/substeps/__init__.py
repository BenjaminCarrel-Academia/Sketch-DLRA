"""
Shared substep RHS builders for sketch DLRA methods.

Each builder creates a closure-based right-hand side for one substep ODE.

Families:
- sketch_kls: K/L substeps with pseudo-inverse Omega^+ / Theta^+ (KSL form)
- sketch_ortho: K/L/S substeps with orthogonal projection (P^TP)^{-1}P^T
- sketch_oblique: K/L/S substeps with oblique sketch projection (Theta P)^{-1} Theta
"""

from .sketch_kls_substeps import _make_K_sketch_rhs, _make_L_sketch_rhs
from .sketch_ortho_substeps import (
    _make_K_sketch_ortho_rhs, _make_L_sketch_ortho_rhs,
    _make_S_sketch_ortho_rhs, _make_minus_S_sketch_ortho_rhs,
)
from .sketch_oblique_substeps import (
    _make_K_sketch_oblique_rhs, _make_L_sketch_oblique_rhs,
    _make_S_sketch_oblique_rhs, _make_minus_S_sketch_oblique_rhs,
)
