# Experiments

This folder contains the numerical experiments presented in the paper
*Sketch low-rank dynamics: orthogonal vs. oblique projections*.
Each script reproduces one figure of the paper and can be run
independently.

## Running experiments

Activate the environment (see the top-level `README.md`), then run any
of the scripts:

```bash
python experiments/exp_allen-cahn.py
```

Each script will:

- generate and solve the corresponding problem,
- create a PDF and a PNG in `experiments/figures/<problem>/`,
- print progress and accuracy metrics.

## Experiment files

### `exp_allen-cahn.py`
**Paper reference:** §6.1 (Allen-Cahn equation)

Solves the Allen-Cahn equation with `n = 64`, `r = 8`, sketch size
`ℓ = 16`, time horizon `[0, 10]`, over 20 Gaussian sketches. Plots
the median + interquartile band of the relative Frobenius error for
the four sketch DLRA variants (orthogonal/oblique × KSL/BUG) against
the unsketched BUG baseline and the best rank-r SVD truncation.

Output: `experiments/figures/allen-cahn/allen_cahn_error.{pdf,png}`.

### `exp_fokker-planck.py`
**Paper reference:** §6.2 (Fokker-Planck equation)

Solves the 2D Fokker-Planck equation with constant-energy nonlinear
drift, `n = 64`, `r = 8`, `ℓ = 16`, time horizon `[0, 1]`, over 20
Gaussian sketches. Same plotting conventions as Allen-Cahn.

Output: `experiments/figures/fokker-planck/fokker_planck_error.{pdf,png}`.

### `exp_vlasov-poisson_error.py`
**Paper reference:** §6.3 (Vlasov-Poisson equation, two-stream instability)

Solves the Vlasov-Poisson two-stream instability with
`nₓ = nᵥ = 64`, `r = 8`, `ℓ = 16`, time horizon `[0, 60]`, over 5
Gaussian sketches. Two side-by-side panels: relative Frobenius error
(left) and electric energy (right).

Output: `experiments/figures/vlasov-poisson/vlasov_poisson_error.{pdf,png}`.

### `exp_motivating.py`
**Paper reference:** §1 (motivating example)

Single-seed (seed 0) Vlasov-Poisson run with the same parameters as
the §6.3 error figure. Shows electric energy of the orthogonal vs.
oblique sketch variants in two side-by-side panels — motivating the
paper's central question.

Output: `experiments/figures/vlasov-poisson/motivating_sketch_dlra.{pdf,png}`.

### `exp_vlasov-poisson_conservation.py`
**Paper reference:** §6.3 (conservation laws)

Conservation diagnostics on the same two-stream Vlasov-Poisson
trajectory. Three panels: total energy, log-scale relative drift of
total energy, and total mass.

The script is *staged*; running with no arguments executes all four
stages sequentially:

```bash
python experiments/exp_vlasov-poisson_conservation.py
# is equivalent to
python experiments/exp_vlasov-poisson_conservation.py ref ort obl plot
```

Intermediate per-stage results are cached in `experiments/diag_*.npz`
so individual stages can be re-run without redoing the others. These
caches are gitignored.

Output: `experiments/figures/vlasov-poisson/vlasov_poisson_conservation.{pdf,png}`.

## Output

All figures live under `experiments/figures/`, organised by problem:

- `figures/allen-cahn/`
- `figures/fokker-planck/`
- `figures/vlasov-poisson/`

## Notes

- Wall-clock time per script: a few minutes for Allen-Cahn /
  Fokker-Planck, and ~30-60 minutes per Vlasov-Poisson script on a
  single modern CPU.
- Random seeds are fixed in each script for reproducibility, but
  exact bitwise outputs depend on BLAS implementations.
- The Vlasov-Poisson scripts wrap each macro step in a `try/except`
  and break on inner-integrator divergence, which is the expected
  failure mode of the oblique sketch variants on this non-LRC problem.
