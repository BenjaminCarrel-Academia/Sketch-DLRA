"""
Motivating figure: orthogonal vs oblique sketch DLRA on Vlasov-Poisson.

Two side-by-side panels of electric energy over time:
  1. Orthogonal sketch DLRA (Reference, Ortho sketch KSL, Ortho sketch BUG)
  2. Oblique sketch DLRA    (Reference, Oblique sketch KSL, Oblique sketch BUG)

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

_base_fs = plt.rcParams['font.size']
plt.rcParams.update({
    'font.size': _base_fs + 3,
    'axes.titlesize': _base_fs + 5,
    'axes.labelsize': _base_fs + 4,
    'xtick.labelsize': _base_fs + 2,
    'ytick.labelsize': _base_fs + 2,
    'legend.fontsize': _base_fs + 3,
})

from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_two_stream
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra_sketch.solve_sketch_dlra import available_sketch_dlra_methods
from matrix_ode_toolbox.structures.sketch_svd import SketchSVD


# Parameters
nx, nv = 64, 64
rank = 8
ell = 2 * rank
t_span = (0.0, 60.0)
nb_steps = 120
nb_substeps = 10
t_eval = np.linspace(t_span[0], t_span[1], nb_steps + 1)

rng = np.random.default_rng(0)


# Problem setup
ode, X0 = make_two_stream(nx, nv)
Y0 = SVD.truncated_svd(X0, rank)

Omega1 = rng.standard_normal((ell, nx))
Omega2 = rng.standard_normal((ell, nv))
sketch_matrices = (Omega1, Omega2)
sY0 = SketchSVD.from_svd(Y0, sketch_matrices=sketch_matrices)

sketch_kwargs = {
    'sketch_matrices': sketch_matrices,
    'nb_substeps': nb_substeps,
    'substep_kwargs': {'solver': 'scipy'},
}


# Reference solve
print('Solving reference...')
ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)


def run_sketch_stepwise(method_key, label):
    """Step through t_eval manually, tolerating failures."""
    print(f'Solving {label}...')
    solver_cls = available_sketch_dlra_methods[method_key]
    solver = solver_cls(ode, **sketch_kwargs)
    Ys = [None] * len(t_eval)
    Ys[0] = sY0
    for i in range(len(t_eval) - 1):
        try:
            Ys[i + 1] = solver.solve((t_eval[i], t_eval[i + 1]), Ys[i])
        except Exception as exc:
            print(f'  {label} failed at t={t_eval[i + 1]:.3f}: {exc}')
            break
    return Ys


ortho_ksl_states = run_sketch_stepwise('ortho_sketch_KSL', 'ortho sketch KSL')
ortho_bug_states = run_sketch_stepwise('ortho_sketch_bug', 'ortho sketch BUG')
oblique_ksl_states = run_sketch_stepwise('oblique_sketch_KSL', 'oblique sketch KSL')
oblique_bug_states = run_sketch_stepwise('oblique_sketch_bug', 'oblique sketch BUG')


# Electric energy
def electric_energy_dense(ode, X):
    rho = ode.dv * np.sum(X, axis=1)
    E = ode.electric_field(rho)
    return 0.5 * ode.dx * np.sum(E ** 2)


def electric_energy_lowrank(ode, Y):
    rho = ode.dv * (Y.U @ Y.S @ np.sum(Y.V.T.conj(), axis=1))
    E = ode.electric_field(rho)
    return 0.5 * ode.dx * np.sum(E ** 2)


def energy_from_states(states):
    out = np.full(len(states), np.nan)
    for j, Y in enumerate(states):
        if Y is None:
            break
        out[j] = electric_energy_lowrank(ode, Y)
    return out


def energy_from_dense_solution(solution):
    out = np.zeros(len(solution.Xs))
    for j, X in enumerate(solution.Xs):
        out[j] = electric_energy_dense(ode, X)
    return out


print('Computing electric energy traces...')
ref_energy = energy_from_dense_solution(ref_sol)
ortho_ksl_energy = energy_from_states(ortho_ksl_states)
ortho_bug_energy = energy_from_states(ortho_bug_states)
oblique_ksl_energy = energy_from_states(oblique_ksl_states)
oblique_bug_energy = energy_from_states(oblique_bug_states)


# Plot
figures_dir = os.path.join(_here, 'figures', 'vlasov-poisson')
os.makedirs(figures_dir, exist_ok=True)

ts = t_eval

C_REF, C_KSL, C_BUG = 'gray', '#1f77b4', '#ff7f0e'
ref_kw = dict(label='Reference', linestyle='-', color=C_REF, linewidth=1.4)
ksl_kw = dict(label='Sketch KSL', linestyle='--', color=C_KSL, linewidth=2.0)
bug_kw = dict(label='Sketch BUG', linestyle='-.', color=C_BUG, linewidth=2.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

EN, EM = '–', '—'

ax1.semilogy(ts, ref_energy, **ref_kw)
ax1.semilogy(ts, ortho_ksl_energy, **ksl_kw)
ax1.semilogy(ts, ortho_bug_energy, **bug_kw)
ax1.set_xlabel('Time')
ax1.set_ylabel('Electric energy')
ax1.set_title(f'Vlasov{EN}Poisson {EM} orthogonal sketch')
ax1.grid(True, which='both', alpha=0.15, linewidth=0.5)

ax2.semilogy(ts, ref_energy, **ref_kw)
ax2.semilogy(ts, oblique_ksl_energy, **ksl_kw)
ax2.semilogy(ts, oblique_bug_energy, **bug_kw)
ax2.set_xlabel('Time')
ax2.set_title(f'Vlasov{EN}Poisson {EM} oblique sketch')
ax2.grid(True, which='both', alpha=0.15, linewidth=0.5)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.05), ncol=3,
           frameon=True, framealpha=0.95, fontsize=_base_fs + 4)
fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
fig.savefig(os.path.join(figures_dir, 'motivating_sketch_dlra.png'), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(figures_dir, 'motivating_sketch_dlra.pdf'), bbox_inches='tight')
print('Figure saved to', figures_dir)
