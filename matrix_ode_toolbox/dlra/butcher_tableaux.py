"""
Shared Butcher tableaux for explicit Runge-Kutta methods.

Used by ProjectedRungeKutta, RandomizedRungeKutta, AdaptiveProjectedMethods,
and RandomizedImplicitEuler.
"""

import numpy as np

# ORDER 1 (Forward Euler)
_a1 = float(0)
_b1 = np.ones(1)

# ORDER 2 (Heun)
_a2 = np.zeros((2, 2))
_a2[1, 0] = 1
_b2 = np.zeros(2)
_b2[0] = 1/2
_b2[1] = 1/2

# ORDER 3 (Kutta's third-order)
_a3 = np.zeros((3, 3))
_a3[1, 0] = 1/3
_a3[2, 1] = 2/3
_b3 = np.zeros(3)
_b3[0] = 1/4
_b3[2] = 3/4

# ORDER 4 (Classic RK4)
_a4 = np.zeros((4, 4))
_a4[1, 0] = 1/2
_a4[2, 1] = 1/2
_a4[3, 2] = 1
_b4 = np.zeros(4)
_b4[0] = 1/6
_b4[1] = 1/3
_b4[2] = 1/3
_b4[3] = 1/6

# ORDER 8 — Rule 6(5)9b
_a8 = np.zeros((8, 8))
_a8[1, 0] = 1/8
_a8[2, 0] = 1/18
_a8[3, 0] = 1/16
_a8[4, 0] = 1/4
_a8[5, 0] = 134/625
_a8[6, 0] = -98/1875
_a8[7, 0] = 9/50
_a8[2, 1] = 1/9
_a8[3, 2] = 3/16
_a8[4, 2] = -3/4
_a8[5, 2] = -333/625
_a8[6, 2] = 12/625
_a8[7, 2] = 21/25
_a8[4, 3] = 1
_a8[5, 3] = 476/625
_a8[6, 3] = 10736/13125
_a8[7, 3] = -2924/1925
_a8[5, 4] = 98/625
_a8[6, 4] = -1936/1875
_a8[7, 4] = 74/25
_a8[6, 5] = 22/21
_a8[7, 5] = -15/7
_a8[7, 6] = 15/22
_b8 = np.zeros(8)
_b8[0] = 11/144
_b8[3] = 256/693
_b8[5] = 125/504
_b8[6] = 125/528
_b8[7] = 5/72

_TABLES = {
    1: (_a1, _b1),
    2: (_a2, _b2),
    3: (_a3, _b3),
    4: (_a4, _b4),
    8: (_a8, _b8),
}


def get_rk_rule(order: int) -> tuple:
    """Return (a, b, c) Butcher tableau for the given order.

    Parameters
    ----------
    order : int
        Number of stages. Supported: 1, 2, 3, 4, 8.

    Returns
    -------
    a : ndarray
        RK matrix
    b : ndarray
        RK weights
    c : ndarray
        RK nodes
    """
    if order not in _TABLES:
        raise ValueError(f"Unsupported RK order {order}. Supported: {sorted(_TABLES.keys())}.")
    a, b = _TABLES[order]
    s = order
    c = np.zeros(s)
    if s > 1:
        for i in range(1, s):
            c[i] = sum(a[i, j] for j in range(i))
    return a, b, c
