# Matrix ODE toolbox

(!) There is nothing to run in this folder.

It contains the building blocks for matrix ODEs: ODE structures
(`MatrixOde`, `SylvesterLikeOde`, `VlasovPoissonOde`), problem
generators (Allen-Cahn, Fokker-Planck, Vlasov-Poisson two-stream
instability), the classical DLRA unconventional (BUG) integrator,
and the four sketch DLRA integrators of the paper
(orthogonal/oblique × BUG/projector-splitting), plus the inner
solvers (scipy adaptive RK45, fixed-step explicit RK).
