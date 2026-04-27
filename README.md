# Sketch low-rank dynamics: orthogonal vs. oblique projections

This repository contains the code for reproducing the numerical
experiments of the paper *Sketch low-rank dynamics: orthogonal vs.
oblique projections*.

## Abstract

We study sketch-based dynamical low-rank approximation (DLRA) for
matrix differential equations. Two natural ways of incorporating a
random sketch into the tangent-space projection are compared: an
*orthogonal* sketch DLRA, whose projection is the standard orthogonal
projector onto the tangent space at every step, and an *oblique*
sketch DLRA, whose projection is sketch-weighted. Despite an extra
basis-orthogonalization cost, the orthogonal variant produces a
trajectory that coincides with the classical DLRA solution and
inherits its conservation behaviour, while the oblique variant solves
a perturbed ODE whose invariants depend on the random sketches and
can fail dramatically on non-LRC problems. Numerical experiments on
Allen-Cahn, Fokker-Planck, and Vlasov-Poisson illustrate the practical
implications.

## Author

- Benjamin Carrel

## Citation

```
@misc{carrel2025sketchdlra,
  author = {Benjamin Carrel},
  title  = {Sketch low-rank dynamics: orthogonal vs.\ oblique projections},
  year   = {2025},
  note   = {Preprint}
}
```

## Installation

### Prerequisites

- Python 3.10 or newer
- conda (recommended) or pip
- git

### Step 1: Clone the repository

```bash
git clone https://github.com/BenjaminCarrel/sketch-dlra.git
cd sketch-dlra
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
python experiments/exp_allen-cahn.py                     # §6.1 Allen-Cahn relative error
python experiments/exp_fokker-planck.py                  # §6.2 Fokker-Planck relative error
python experiments/exp_vlasov-poisson_error.py           # §6.3 Vlasov-Poisson error & electric energy
python experiments/exp_motivating.py                     # §1 motivating example (electric energy)
python experiments/exp_vlasov-poisson_conservation.py    # §6.3 conservation laws
```

Wall-clock budget on a single modern CPU: roughly 5-10 minutes for
Allen-Cahn and Fokker-Planck, 30-60 minutes per Vlasov-Poisson script.

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

This project is licensed under the GNU General Public License v3.0 -
see the `LICENSE` file for details.
