"""
Conservation diagnostics: orthogonal vs oblique sketch DLRA on Vlasov-Poisson.

Vlasov-Poisson is Hamiltonian: E_kin(t) + E_el(t) = constant along any exact
solution. Any DLRA variant whose projection coincides with the classical
orthogonal projection inherits approximate conservation up to the classical
DLRA modelling error. The oblique sketch DLRA uses a different,
sketch-weighted projection and is solving a different ODE whose invariants
need not match Vlasov-Poisson's. This script measures exactly that.

For each method (reference full-order, ortho sketch BUG, oblique sketch BUG):
  - Evolve the trajectory.
  - At each saved step, compute E_kin(t), E_el(t), E_tot(t).
  - Also compute mass(t) := dx * dv * sum(X) for completeness.

Stages:
  ref   -- compute reference energies
  ort   -- run ortho BUG, compute energies per step
  obl   -- run oblique BUG, compute energies per step
  plot  -- assemble the three-panel figure

Run with no arguments to execute all four stages in order.

Author: Benjamin Carrel
"""

from __future__ import annotations
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_code_root = os.path.abspath(os.path.join(_here, '..'))
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_two_stream
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.structures.sketch_svd import SketchSVD
from matrix_ode_toolbox.dlra_sketch.solve_sketch_dlra import available_sketch_dlra_methods


nx, nv = 64, 64
rank = 8
ell = 2 * rank
t_span = (0.0, 60.0)
nb_steps = 120
nb_substeps = 10
t_eval = np.linspace(t_span[0], t_span[1], nb_steps + 1)
SKETCH_SEED = 0

REF_STATES = os.path.join(_here, 'diag_ref.npz')
STAGE_REF  = os.path.join(_here, 'diag_cons_ref.npz')
STAGE_ORT  = os.path.join(_here, 'diag_cons_ort.npz')
STAGE_OBL  = os.path.join(_here, 'diag_cons_obl.npz')
FIG_DIR    = os.path.join(_here, 'figures', 'vlasov-poisson')
os.makedirs(FIG_DIR, exist_ok=True)


def problem():
    ode, X0 = make_two_stream(nx, nv)
    Y0 = SVD.truncated_svd(X0, rank)
    return ode, X0, Y0


def get_reference(ode, X0):
    if os.path.exists(REF_STATES):
        data = np.load(REF_STATES)
        return [data[f'X{i}'] for i in range(len(t_eval))]
    ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
    Xs = [X if isinstance(X, np.ndarray) else X.todense() for X in ref_sol.Xs]
    np.savez(REF_STATES, **{f'X{i}': Xs[i] for i in range(len(Xs))})
    return Xs


def mass(ode, X):
    return ode.dx * ode.dv * float(np.sum(X))


def energies(ode, X):
    Ek = ode.kinetic_energy(X)
    Ee = ode.electric_energy(X)
    return Ek, Ee, Ek + Ee


def stage_ref():
    ode, X0, _ = problem()
    Xs = get_reference(ode, X0)
    Ek = np.zeros(len(t_eval))
    Ee = np.zeros(len(t_eval))
    Et = np.zeros(len(t_eval))
    M = np.zeros(len(t_eval))
    for j, X in enumerate(Xs):
        Ek[j], Ee[j], Et[j] = energies(ode, X)
        M[j] = mass(ode, X)
    np.savez(STAGE_REF, t_eval=t_eval, E_kin=Ek, E_el=Ee, E_tot=Et, mass=M)
    print(f'Saved {STAGE_REF}')


def _sketch_pair():
    rng = np.random.default_rng(SKETCH_SEED)
    return rng.standard_normal((ell, nx)), rng.standard_normal((ell, nv))


def stage_method(method_key, out_path):
    ode, X0, Y0 = problem()
    Omega1, Omega2 = _sketch_pair()
    sketch_kwargs = {
        'sketch_matrices': (Omega1, Omega2),
        'nb_substeps': nb_substeps,
        'substep_kwargs': {'solver': 'scipy'},
    }
    sY0 = SketchSVD.from_svd(Y0, sketch_matrices=(Omega1, Omega2))
    solver = available_sketch_dlra_methods[method_key](ode, **sketch_kwargs)

    Ek = np.full(len(t_eval), np.nan)
    Ee = np.full(len(t_eval), np.nan)
    Et = np.full(len(t_eval), np.nan)
    M = np.full(len(t_eval), np.nan)
    alive = np.zeros(len(t_eval), dtype=bool)

    Ek[0], Ee[0], Et[0] = energies(ode, X0)
    M[0] = mass(ode, X0)
    alive[0] = True

    Y_cur = sY0
    for i in range(len(t_eval) - 1):
        try:
            Y_next = solver.solve((t_eval[i], t_eval[i + 1]), Y_cur)
        except Exception as exc:
            print(f'  {method_key} failed at t={t_eval[i + 1]:.2f}: {exc}')
            break
        Y_cur = Y_next
        Xd = Y_next.todense()
        Ek[i + 1], Ee[i + 1], Et[i + 1] = energies(ode, Xd)
        M[i + 1] = mass(ode, Xd)
        alive[i + 1] = True

    np.savez(out_path, t_eval=t_eval, E_kin=Ek, E_el=Ee, E_tot=Et, mass=M,
             alive=alive)
    print(f'Saved {out_path}')


def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = plt.rcParams['font.size']
    plt.rcParams.update({
        'font.size': base + 3, 'axes.labelsize': base + 4,
        'xtick.labelsize': base + 2, 'ytick.labelsize': base + 2,
        'legend.fontsize': base + 3, 'axes.titlesize': base + 5,
        'lines.linewidth': 2.0, 'axes.grid': True,
        'grid.alpha': 0.2, 'grid.linewidth': 0.6,
    })

    ref = np.load(STAGE_REF)
    ort = np.load(STAGE_ORT)
    obl = np.load(STAGE_OBL)

    t = ref['t_eval']

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    def rel_drift_self(arr, alive):
        out = np.full_like(arr, np.nan)
        out[alive] = (arr[alive] - arr[0]) / abs(arr[0])
        return out

    LBL_REF = 'Reference'
    LBL_ORT = 'Ortho. sketch BUG'
    LBL_OBL = 'Oblique sketch BUG'
    C_REF, C_ORT, C_OBL = 'black', '#1f77b4', '#d62728'

    # Panel (a): total energy absolute
    ax = axes[0]
    ax.plot(t, ref['E_tot'], color=C_REF, linewidth=2.4, label=LBL_REF)
    ax.plot(t[ort['alive']], ort['E_tot'][ort['alive']], color=C_ORT,
            linewidth=2.2, label=LBL_ORT)
    ax.plot(t[obl['alive']], obl['E_tot'][obl['alive']], color=C_OBL,
            linewidth=2.2, label=LBL_OBL)
    ax.axhline(ref['E_tot'][0], color='gray', linestyle=':', linewidth=1.2,
               alpha=0.8)
    ax.text(0.98, 0.04,
            fr'$E_{{\mathrm{{tot}}}}(0) = {ref["E_tot"][0]:.2f}$',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=base + 2, color='gray',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.8))
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$E_{\mathrm{tot}}(t)$')
    ax.set_title(r'(a) Total energy $E_{\mathrm{kin}} + E_{\mathrm{el}}$')
    ax.set_xlim(t[0], t[-1])

    # Panel (b): |relative drift| of total energy, log scale
    ax = axes[1]
    abs_drift_ref = np.abs(rel_drift_self(ref['E_tot'], np.ones_like(t, dtype=bool)))
    abs_drift_ort = np.abs(rel_drift_self(ort['E_tot'], ort['alive']))
    abs_drift_obl = np.abs(rel_drift_self(obl['E_tot'], obl['alive']))
    ax.semilogy(t, abs_drift_ref, color=C_REF, linewidth=2.4, label=LBL_REF)
    ax.semilogy(t[ort['alive']], abs_drift_ort[ort['alive']],
                color=C_ORT, linewidth=2.2, label=LBL_ORT)
    ax.semilogy(t[obl['alive']], abs_drift_obl[obl['alive']],
                color=C_OBL, linewidth=2.2, label=LBL_OBL)
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$|E_{\mathrm{tot}}(t) - E_{\mathrm{tot}}(0)| / |E_{\mathrm{tot}}(0)|$')
    ax.set_title(r'(b) Relative drift of total energy')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(1e-10, 2.0)

    # Panel (c): mass
    ax = axes[2]
    ax.plot(t, ref['mass'], color=C_REF, linewidth=2.4, label=LBL_REF)
    ax.plot(t[ort['alive']], ort['mass'][ort['alive']], color=C_ORT,
            linewidth=2.2, label=LBL_ORT)
    ax.plot(t[obl['alive']], obl['mass'][obl['alive']], color=C_OBL,
            linewidth=2.2, label=LBL_OBL)
    ax.axhline(ref['mass'][0], color='gray', linestyle=':', linewidth=1.2,
               alpha=0.8)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.text(0.98, 0.96,
            fr'mass$(0) = {ref["mass"][0]:.2f}$',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=base + 2, color='gray',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.8))
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\int\!\int f\,dx\,dv$')
    ax.set_title(r'(c) Total mass')
    ax.set_xlim(t[0], t[-1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=3,
               frameon=True, framealpha=0.95, fontsize=base + 4)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    f_pdf = os.path.join(FIG_DIR, 'vlasov_poisson_conservation.pdf')
    f_png = os.path.join(FIG_DIR, 'vlasov_poisson_conservation.png')
    fig.savefig(f_pdf, bbox_inches='tight')
    fig.savefig(f_png, dpi=200, bbox_inches='tight')
    print('Saved', f_pdf, 'and', f_png)

    # Summary
    print()
    print('=' * 100)
    print('Total energy conservation summary')
    print('=' * 100)
    print(f'{"t":>6} {"E_tot_ref":>12} {"E_tot_ort":>12} {"E_tot_obl":>12} '
          f'{"drift_ref":>12} {"drift_ort":>12} {"drift_obl":>12}')
    idx = [int(np.argmin(np.abs(t - s))) for s in [0, 10, 20, 30, 40, 50, 60]]
    for i in idx:
        row = f'{t[i]:>6.1f} '
        row += f'{ref["E_tot"][i]:>12.6f} '
        row += (f'{ort["E_tot"][i]:>12.6f} ' if ort['alive'][i] else f'{"--":>12} ')
        row += (f'{obl["E_tot"][i]:>12.6f} ' if obl['alive'][i] else f'{"--":>12} ')
        row += f'{(ref["E_tot"][i] - ref["E_tot"][0]) / ref["E_tot"][0]:>12.3e} '
        row += (f'{(ort["E_tot"][i] - ort["E_tot"][0]) / ort["E_tot"][0]:>12.3e} '
                if ort['alive'][i] else f'{"--":>12} ')
        row += (f'{(obl["E_tot"][i] - obl["E_tot"][0]) / obl["E_tot"][0]:>12.3e} '
                if obl['alive'][i] else f'{"--":>12} ')
        print(row)

    print()
    print('Mass conservation:')
    for name, d in [('reference', ref), ('ortho', ort), ('oblique', obl)]:
        if 'alive' in d.files:
            a = d['alive']
            m = d['mass'][a]
        else:
            m = d['mass']
        print(f'  {name}: initial = {m[0]:.6f}, final = {m[-1]:.6f}, '
              f'rel drift = {(m[-1] - m[0]) / m[0]:.3e}')


def main():
    stages = sys.argv[1:] or ['all']
    if 'all' in stages:
        stages = ['ref', 'ort', 'obl', 'plot']
    for st in stages:
        print(f'\n### stage: {st} ###\n')
        if st == 'ref':
            stage_ref()
        elif st == 'ort':
            stage_method('ortho_sketch_bug', STAGE_ORT)
        elif st == 'obl':
            stage_method('oblique_sketch_bug', STAGE_OBL)
        elif st == 'plot':
            stage_plot()


if __name__ == '__main__':
    main()
