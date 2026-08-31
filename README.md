# Sketch low-rank dynamics: orthogonal vs. oblique projections

This repository contains the code for reproducing the numerical
experiments of the paper *Sketch low-rank dynamics: orthogonal vs.
oblique projections*.

## Abstract

We study how sketching techniques from randomized numerical linear algebra can be incorporated into the dynamical low-rank approximation (DLRA) of large-scale matrix differential equations.

A natural approach is to sketch the Galerkin condition that defines the DLRA, which leads to an oblique tangent space projection. We show that this oblique projection approximately reproduces the standard DLRA only under restrictive conditions on the vector field, and that it fails on problems with a large perpendicular residual.

As an alternative, we propose an orthogonal sketch DLRA that evolves sketch-orthogonal bases while using standard orthogonal projections for the dynamics. This approach preserves the geometric structure of the classical DLRA and is numerically stable. The computational advantage of randomized Gram--Schmidt over Householder QR lies in fewer global synchronizations on a row-distributed basis, at a comparable flop count; when the basis is well conditioned, randomized Gram--Schmidt can be replaced by randomized Cholesky QR, which additionally shifts the basis update from BLAS-2 to BLAS-3 kernels, making it well-suited to modern accelerators.

We derive sketch versions of the projector-splitting and BUG integrators, and demonstrate the approach on the Allen--Cahn, Fokker--Planck, and Vlasov--Poisson equations.

## Authors

- Benjamin Carrel
- Laura Grigori

## Citation

(Will be updated when available on arXiv.)

## Installation

### Prerequisites

- Python 3.10 or newer
- conda (recommended) or pip
- git

### Step 1: Clone the repository

```bash
git clone https://github.com/BenjaminCarrel/Sketch-DLRA.git
cd Sketch-DLRA
```

### Step 2: Install dependencies

#### Option A: with conda (recommended)

```bash
conda env create --file environment.yml
conda activate sketch-dlra
pip install -e .
```

#### Option B: with pip only

```bash
python3.12 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### Step 3: Verify the installation

```bash
python validate_installation.py
```

The script checks Python version, required packages, custom toolboxes,
and runs one step of the orthogonal sketch BUG integrator on
Allen-Cahn.

## Reproducing the paper figures

Each script in `experiments/` produces one of the five figures of the
paper:

```bash
python experiments/exp_motivating.py                     # §1   motivating example (electric energy)        ~1 min
python experiments/exp_allen-cahn.py                     # §6.1 Allen-Cahn relative error                   ~3 min
python experiments/exp_fokker-planck.py                  # §6.2 Fokker-Planck relative error                ~2 min
python experiments/exp_vlasov-poisson_error.py           # §6.3 Vlasov-Poisson error & electric energy      ~5 min
python experiments/exp_vlasov-poisson_conservation.py    # §6.3 conservation laws                           ~30 s
```

Wall-clock estimates above are for a MacBook Pro (Apple M1, 16 GB
RAM) — the same machine used for the paper.

Outputs go to `experiments/figures/<problem>/`, named
`allen_cahn_error.{pdf,png}`, `fokker_planck_error.{pdf,png}`, etc.

See `experiments/README.md` for per-experiment details.

## Repository layout

```
.
├── README.md
├── LICENSE                 # GNU GPL v3
├── setup.py
├── environment.yml
├── validate_installation.py
├── low_rank_toolbox/       # SVD / QuasiSVD / QR containers and Gram-Schmidt
├── matrix_ode_toolbox/     # matrix ODE structures, classical and sketch DLRA integrators
└── experiments/            # five paper figures, one script per figure
    └── figures/            # output PDFs and PNGs (created on first run)
```

## License

Copyright (c) 2026 Paul Scherrer Institute (PSI). Authors: Benjamin Carrel,
Laura Grigori.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the `LICENSE` file for the full text, or
<https://www.gnu.org/licenses/>.

### Included code

`low_rank_toolbox/` and `matrix_ode_toolbox/` are copies of the authors' own
research toolboxes, included so that the experiments run standalone. They are
part of this work and are covered by the licence above. Parts of
`low_rank_toolbox/` are also published separately, under the MIT licence, at
[low-rank-toolbox](https://github.com/BenjaminCarrel-Academia/low-rank-toolbox);
that release stands on its own terms.
