"""
Test problems used in the sketch DLRA paper.

- ``make_allen_cahn``: 2D Allen-Cahn equation (approximately LRC).
- ``make_fokker_planck_2D_constant_energy``: 2D Fokker-Planck (approximately LRC).
- ``make_two_stream``: Vlasov-Poisson two-stream instability (non-LRC).
"""

from .sylvester_like_problems import make_allen_cahn, make_fokker_planck_2D_constant_energy
from .vlasov_poisson_problems import make_two_stream

__all__ = [
    "make_allen_cahn",
    "make_fokker_planck_2D_constant_energy",
    "make_two_stream",
]
