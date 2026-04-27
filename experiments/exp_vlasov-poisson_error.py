"""
Polished error-over-time figure for sketch DLRA on Vlasov-Poisson (two-stream).

For the electric-energy plot see exp_motivating.py.

Author: Benjamin Carrel
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))
_code_root = os.path.abspath(os.path.join(_here, '..'))
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_two_stream
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
nx, nv = 64, 64
rank = 8
ell = 2 * rank
t_span = (0.0, 60.0)
nb_steps = 120
nb_substeps = 10
n_seeds = 5
t_eval = np.linspace(t_span[0], t_span[1], nb_steps + 1)


# Problem setup
ode, X0 = make_two_stream(nx, nv)
Y0 = SVD.truncated_svd(X0, rank)


# Reference and best rank-r baseline
print('Solving reference...')
ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
Xs_ref = [X if isinstance(X, np.ndarray) else X.todense() for X in ref_sol.Xs]


def rel_err(X, Xref):
    return np.linalg.norm(X - Xref, 'fro') / np.linalg.norm(Xref, 'fro')


def electric_energy_dense(X):
    rho = ode.dv * np.sum(X, axis=1)
    return ode.dx * np.linalg.norm(ode.electric_field(rho))


def electric_energy_lowrank(Y):
    rho = ode.dv * (Y.U @ Y.S @ np.sum(Y.V.T.conj(), axis=1))
    return ode.dx * np.linalg.norm(ode.electric_field(rho))


best_error = np.array([
    rel_err(SVD.truncated_svd(Xref, rank).todense(), Xref) for Xref in Xs_ref
])
ref_energy = np.array([electric_energy_dense(X) for X in Xs_ref])


# Standard (non-sketched) BUG
print('Solving standard BUG...')
bug_sol = solve_dlra(
    ode, t_span, Y0, 'unconventional',
    {'substep_kwargs': {'solver': 'scipy'}, 'nb_substeps': nb_substeps},
    t_eval=t_eval, monitor=True,
)
bug_error = np.array([rel_err(Y.todense(), Xref) for Y, Xref in zip(bug_sol.Xs, Xs_ref)])
bug_energy = np.array([electric_energy_lowrank(Y) for Y in bug_sol.Xs])


# Sketch solves: per-seed stepwise, fault-tolerant, tracking errors and energies.
def run_sketch_seed(method_key, seed):
    rng = np.random.default_rng(seed)
    Omega1 = rng.standard_normal((ell, nx))
    Omega2 = rng.standard_normal((ell, nv))
    sm = (Omega1, Omega2)
    sY0 = SketchSVD.from_svd(Y0, sketch_matrices=sm)
    solver = available_sketch_dlra_methods[method_key](
        ode, sketch_matrices=sm, nb_substeps=nb_substeps,
        substep_kwargs={'solver': 'scipy'},
    )
    errs = np.full(len(t_eval), np.nan)
    energy = np.full(len(t_eval), np.nan)
    errs[0] = rel_err(sY0.todense(), Xs_ref[0])
    energy[0] = electric_energy_lowrank(sY0)
    Y = sY0
    for i in range(len(t_eval) - 1):
        try:
            Y = solver.solve((t_eval[i], t_eval[i + 1]), Y)
        except Exception as exc:
            print(f'  {method_key} seed{seed} failed at t={t_eval[i+1]:.2f}: {exc}')
            break
        errs[i+1] = rel_err(Y.todense(), Xs_ref[i+1])
        energy[i+1] = electric_energy_lowrank(Y)
    return errs, energy


# Colour says family (orthogonal vs oblique), linestyle says variant (KSL vs BUG).
C_O, C_X = '#1f77b4', '#d62728'
methods = [
    ('ortho_sketch_KSL',   '--', C_O),
    ('ortho_sketch_bug',   '-.', C_O),
    ('oblique_sketch_KSL', '--', C_X),
    ('oblique_sketch_bug', '-.', C_X),
]

print(f'Running {n_seeds} sketch seeds for each of the {len(methods)} methods...')
sketch_errors = {}
sketch_energies = {}
for key, _, _ in methods:
    runs_e = np.full((n_seeds, len(t_eval)), np.nan)
    runs_E = np.full((n_seeds, len(t_eval)), np.nan)
    for s in range(n_seeds):
        runs_e[s], runs_E[s] = run_sketch_seed(key, s)
    sketch_errors[key] = runs_e
    sketch_energies[key] = runs_E


def med_iqr(runs):
    return (np.nanmedian(runs, axis=0),
            np.nanpercentile(runs, 25, axis=0),
            np.nanpercentile(runs, 75, axis=0))


# Plot: two panels (relative error, electric energy)
figures_dir = os.path.join(_here, 'figures', 'vlasov-poisson')
os.makedirs(figures_dir, exist_ok=True)

EN, EM = '–', '—'
C_BUG, C_REF = 'gray', 'black'

fig, (ax_err, ax_ene) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# (a) Relative error.
ax_err.semilogy(t_eval, best_error, ':', color='black', linewidth=1.4,
                label=f'Best rank-{rank} approx.')
ax_err.semilogy(t_eval, bug_error, '-', color=C_BUG, linewidth=1.6)
for key, ls, color in methods:
    med, q25, q75 = med_iqr(sketch_errors[key])
    nice = np.isfinite(med)
    ax_err.semilogy(t_eval[nice], med[nice], color=color, linestyle=ls, linewidth=2.0)
    ax_err.fill_between(t_eval[nice], q25[nice], q75[nice], color=color,
                        alpha=0.20, linewidth=0)
ax_err.set_xlabel('Time'); ax_err.set_ylabel('Relative error')
ax_err.set_title(f'(a) Vlasov{EN}Poisson {EM} relative error')
ax_err.grid(True, which='both', alpha=0.15, linewidth=0.5)
ax_err.set_xlim(t_eval[0], t_eval[-1])
ax_err.legend(handles=[Line2D([0], [0], linestyle=':', color='black', linewidth=1.4,
                              label=f'Best rank-{rank} approx.')],
              loc='lower right', framealpha=0.9, fontsize=_base_fs + 2)

# (b) Electric energy.
ax_ene.semilogy(t_eval, ref_energy, '-', color=C_REF, linewidth=1.6)
ax_ene.semilogy(t_eval, bug_energy, '-', color=C_BUG, linewidth=1.6)
for key, ls, color in methods:
    med, q25, q75 = med_iqr(sketch_energies[key])
    nice = np.isfinite(med)
    ax_ene.semilogy(t_eval[nice], med[nice], color=color, linestyle=ls, linewidth=2.0)
    ax_ene.fill_between(t_eval[nice], q25[nice], q75[nice], color=color,
                        alpha=0.20, linewidth=0)
ax_ene.set_xlabel('Time'); ax_ene.set_ylabel('Electric energy')
ax_ene.set_title(f'(b) Vlasov{EN}Poisson {EM} electric energy')
ax_ene.grid(True, which='both', alpha=0.15, linewidth=0.5)
ax_ene.set_xlim(t_eval[0], t_eval[-1])
ax_ene.legend(handles=[Line2D([0], [0], linestyle='-', color=C_REF,
                              linewidth=1.6, label='Reference')],
              loc='lower right', framealpha=0.9, fontsize=_base_fs + 2)

shared = [
    Line2D([0], [0], linestyle='-',  color=C_BUG, linewidth=1.6, label='BUG (unsketched)'),
    Line2D([0], [0], linestyle='--', color=C_O,   linewidth=2.0, label='Ortho. sketch (KSL)'),
    Line2D([0], [0], linestyle='-.', color=C_O,   linewidth=2.0, label='Ortho. sketch (BUG)'),
    Line2D([0], [0], linestyle='--', color=C_X,   linewidth=2.0, label='Oblique sketch (KSL)'),
    Line2D([0], [0], linestyle='-.', color=C_X,   linewidth=2.0, label='Oblique sketch (BUG)'),
]
fig.legend(handles=shared, loc='lower center',
           bbox_to_anchor=(0.5, -0.05), ncol=5,
           frameon=True, framealpha=0.95, fontsize=_base_fs + 3)
fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))

png = os.path.join(figures_dir, 'vlasov_poisson_error.png')
pdf = os.path.join(figures_dir, 'vlasov_poisson_error.pdf')
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print('Saved', png, 'and', pdf)
