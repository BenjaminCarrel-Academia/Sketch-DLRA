#!/usr/bin/env python3
"""
Validation script for the sketch-dlra installation.

Checks that all required packages are installed and that the custom
toolboxes are importable, then runs a one-step Allen-Cahn solve to
exercise the full sketch-DLRA pipeline.

Usage:
    python validate_installation.py
"""

import sys
from typing import Tuple


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str) -> None:
    print(f"\n{text}")
    print("-" * 70)


def check_python_version() -> Tuple[bool, str]:
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor >= 10:
        return True, f"OK   Python {version_str}"
    return False, f"FAIL Python {version_str} (requires Python 3.10+)"


def check_package(name: str) -> Tuple[bool, str]:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        return True, f"OK   {name} ({version})"
    except ImportError:
        return False, f"FAIL {name} (not installed)"
    except Exception as e:
        return False, f"FAIL {name} ({e})"


def check_toolbox(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, f"OK   {name}"
    except ImportError as e:
        return False, f"FAIL {name} ({e})"
    except Exception as e:
        return False, f"FAIL {name} ({e})"


def test_sketch_dlra_step() -> Tuple[bool, str]:
    """Run one step of an orthogonal sketch BUG integrator on Allen-Cahn."""
    try:
        import numpy as np
        from low_rank_toolbox import SVD
        from matrix_ode_toolbox.problems import make_allen_cahn
        from matrix_ode_toolbox.structures.sketch_svd import SketchSVD
        from matrix_ode_toolbox.dlra_sketch.solve_sketch_dlra import (
            available_sketch_dlra_methods,
        )

        n, r, ell = 16, 4, 8
        ode, X0 = make_allen_cahn(n)
        Y0 = SVD.truncated_svd(X0, r)

        rng = np.random.default_rng(0)
        sketch_matrices = (
            rng.standard_normal((ell, n)),
            rng.standard_normal((ell, n)),
        )
        sY0 = SketchSVD.from_svd(Y0, sketch_matrices=sketch_matrices)

        solver = available_sketch_dlra_methods["ortho_sketch_bug"](
            ode,
            sketch_matrices=sketch_matrices,
            nb_substeps=2,
            substep_kwargs={"solver": "scipy"},
        )
        Y = solver.solve((0.0, 0.05), sY0)

        assert Y.rank == r, f"expected rank {r}, got {Y.rank}"
        assert np.isfinite(Y.todense()).all(), "non-finite entries in result"
        return True, "OK   ortho_sketch_bug one-step Allen-Cahn"
    except Exception as e:
        return False, f"FAIL sketch DLRA test ({e})"


def main():
    print_header("sketch-dlra installation validation")

    all_passed = True

    print_subheader("1. Python version")
    ok, msg = check_python_version()
    print(msg)
    all_passed &= ok

    print_subheader("2. Required packages")
    for name in ("numpy", "scipy", "matplotlib", "tqdm"):
        ok, msg = check_package(name)
        print(msg)
        all_passed &= ok

    print_subheader("3. Custom toolboxes")
    for name in ("low_rank_toolbox", "matrix_ode_toolbox"):
        ok, msg = check_toolbox(name)
        print(msg)
        all_passed &= ok

    print_subheader("4. Sketch-DLRA functionality test")
    ok, msg = test_sketch_dlra_step()
    print(msg)
    all_passed &= ok

    print_header("Summary")
    if all_passed:
        print("All checks passed. Installation is successful.")
        print("\nYou can now run the experiments:")
        print("  python experiments/exp_allen-cahn.py")
        return 0
    print("Some checks failed. Please review the errors above.")
    print("\nTroubleshooting tips:")
    print("  - Activate the conda env (conda activate sketch-dlra) or your venv.")
    print("  - Reinstall with `pip install -e .`.")
    print("  - Check README.md for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
