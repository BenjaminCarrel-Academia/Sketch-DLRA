"""
Registry of sketch DLRA integrators used in the paper.

The four methods (orthogonal/oblique x BUG/projector-splitting) are
keyed for convenience; the projector-splitting variants are also
exposed under the alias ``*_KSL``.
"""

from .orthogonal_methods import OrthoSketchBUG, OrthoSketchProjectorSplitting
from .oblique_methods import ObliqueSketchBUG, ObliqueSketchProjectorSplitting


available_sketch_dlra_methods = {
    # Orthogonal sketch DLRA: RGS + standard orthogonal projection
    'ortho_sketch_bug': OrthoSketchBUG,
    'ortho_sketch_KSL': OrthoSketchProjectorSplitting,
    'ortho_sketch_projector_splitting': OrthoSketchProjectorSplitting,
    # Oblique sketch DLRA: RGS + sketch-weighted oblique projection
    'oblique_sketch_bug': ObliqueSketchBUG,
    'oblique_sketch_KSL': ObliqueSketchProjectorSplitting,
    'oblique_sketch_projector_splitting': ObliqueSketchProjectorSplitting,
}
