"""
Polished error-over-time figure for sketch DLRA on Allen-Cahn.

Runs each sketch method over many seeds and plots median + IQR band.
Ortho and oblique variants are placed in two side-by-side panels.

Author: Benjamin Carrel
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_code_root = os.path.abspath(os.path.join(_here, '..'))
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_allen_cahn
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from matrix_ode_toolbox.dlra_sketch.solve_sketch_dlra import available_sketch_dlra_methods
from matrix_ode_toolbox.structures.sketch_svd import SketchSVD


_base_fs = plt.rcParams['font.size']
plt.rcParams.update({
    'font.size': _base_fs + 3,
    'axes.titlesize': _base_fs + 5,
    'axes.labelsize': _base_fs + 4,
    'xtick.labelsize': _base_fs + 2,
    'ytick.labelsize': _base_fs + 2,
    'legend.fontsize': _base_fs + 3,
})


# Parameters
size = 64
rank = 8
ell = 2 * rank
t_span = (0.0, 10.0)
nb_steps = 60
nb_substeps = 10
n_seeds = 20
t_eval = np.linspace(t_span[0], t_span[1], nb_steps + 1)


# Problem setup
ode, X0 = make_allen_cahn(size)
Y0 = SVD.truncated_svd(X0, rank)


def rel_err(X, Xref):
    return np.linalg.norm(X - Xref, 'fro') / np.linalg.norm(Xref, 'fro')


# Reference and best rank-r baseline
print('Solving reference...')
ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
Xs_ref = [X if isinstance(X, np.ndarray) else X.todense() for X in ref_sol.Xs]

best_error = np.array([
    rel_err(SVD.truncated_svd(Xref, rank).todense(), Xref) for Xref in Xs_ref
])


# Standard (non-sketched) BUG
print('Solving standard BUG...')
bug_sol = solve_dlra(
    ode, t_span, Y0, 'unconventional',
    {'substep_kwargs': {'solver': 'scipy'}, 'nb_substeps': nb_substeps},
    t_eval=t_eval, monitor=True,
)
bug_error = np.array([rel_err(Y.todense(), Xref) for Y, Xref in zip(bug_sol.Xs, Xs_ref)])


# Sketch solves over many seeds
def run_one_seed(method_key, seed):
    rng = np.random.default_rng(seed)
    Omega1 = rng.standard_normal((ell, X0.shape[0]))
    Omega2 = rng.standard_normal((ell, X0.shape[1]))
    sketch_matrices = (Omega1, Omega2)
    sY0 = SketchSVD.from_svd(Y0, sketch_matrices=sketch_matrices)
    solver = available_sketch_dlra_methods[method_key](
        ode,
        sketch_matrices=sketch_matrices,
        nb_substeps=nb_substeps,
        substep_kwargs={'solver': 'scipy'},
    )
    errs = np.full(len(t_eval), np.nan)
    errs[0] = rel_err(sY0.todense(), Xs_ref[0])
    Y = sY0
    for i in range(len(t_eval) - 1):
        try:
            Y = solver.solve((t_eval[i], t_eval[i + 1]), Y)
        except Exception:
            break
        errs[i + 1] = rel_err(Y.todense(), Xs_ref[i + 1])
    return errs


seed_seq = np.random.SeedSequence(0).generate_state(n_seeds)

methods = [
    ('ortho_sketch_KSL', 'Ortho. sketch KSL'),
    ('ortho_sketch_bug', 'Ortho. sketch BUG'),
    ('oblique_sketch_KSL', 'Oblique sketch KSL'),
    ('oblique_sketch_bug', 'Oblique sketch BUG'),
]

results = {}
for key, label in methods:
    print(f'Solving {label} over {n_seeds} seeds...')
    runs = np.stack([run_one_seed(key, int(s)) for s in seed_seq], axis=0)
    results[key] = runs


# Aggregate (median + IQR over seeds)
def agg(runs):
    median = np.nanmedian(runs, axis=0)
    q25 = np.nanpercentile(runs, 25, axis=0)
    q75 = np.nanpercentile(runs, 75, axis=0)
    return median, q25, q75


# Plot
figures_dir = os.path.join(_here, 'figures', 'allen-cahn')
os.makedirs(figures_dir, exist_ok=True)

fig, (ax_ortho, ax_obl) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

C_BEST, C_BUG_REF, C_KSL, C_BUG_SK = 'black', 'gray', '#1f77b4', '#ff7f0e'
best_kw = dict(linestyle=':', color=C_BEST, linewidth=1.4,
               label=f'Best rank-{rank} approx.')
bug_kw  = dict(linestyle='-', color=C_BUG_REF, linewidth=1.6,
               label='BUG (unsketched)')
ksl_kw  = dict(linestyle='--', color=C_KSL, linewidth=2.0, label='Sketch KSL')
sbug_kw = dict(linestyle='-.', color=C_BUG_SK, linewidth=2.0, label='Sketch BUG')

EN, EM = '–', '—'

panels = [
    (ax_ortho, f'Allen{EN}Cahn {EM} orthogonal sketch',
        ('ortho_sketch_KSL', 'ortho_sketch_bug')),
    (ax_obl,   f'Allen{EN}Cahn {EM} oblique sketch',
        ('oblique_sketch_KSL', 'oblique_sketch_bug')),
]

for ax, title, (ksl_key, bug_key) in panels:
    ax.semilogy(t_eval, best_error, **best_kw)
    ax.semilogy(t_eval, bug_error, **bug_kw)
    for key, kw in [(ksl_key, ksl_kw), (bug_key, sbug_kw)]:
        median, q25, q75 = agg(results[key])
        ax.semilogy(t_eval, median, **kw)
        ax.fill_between(t_eval, q25, q75, color=kw['color'], alpha=0.20, linewidth=0)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.15, linewidth=0.5)

ax_ortho.set_ylabel('Relative error')

handles, labels = ax_ortho.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), ncol=4,
           frameon=True, framealpha=0.95, fontsize=_base_fs + 4)
fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

png = os.path.join(figures_dir, 'allen_cahn_error.png')
pdf = os.path.join(figures_dir, 'allen_cahn_error.pdf')
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print('Saved', png, 'and', pdf)
