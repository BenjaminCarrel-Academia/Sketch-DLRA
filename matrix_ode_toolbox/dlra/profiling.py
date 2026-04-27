"""
Lightweight profiling for DLRA solvers.

Author: Benjamin Carrel, University of Geneva, 2024
"""

import time


class StepTimer:
    """Lightweight profiler for DLRA solver phases.

    Usage:
        timer = StepTimer(enabled=True)

        # Context manager style:
        with timer('K-step'):
            K1 = solve(...)

        # Manual tic/toc:
        timer.tic('QR')
        U, S = la.qr(K1, mode='economic')
        timer.toc('QR')

        # Print results:
        print(timer.summary())
    """

    def __init__(self, enabled=False):
        """Create a new step timer.

        Parameters
        ----------
        enabled : bool
            If False, all tic/toc calls are no-ops (zero overhead).
        """
        self.enabled = enabled
        self._times = {}
        self._counts = {}
        self._tics = {}

    def __call__(self, name):
        """Return a context manager that times a named phase.

        Parameters
        ----------
        name : str
            Label for the phase (e.g. ``'K-step'``).
        """
        return _TimerContext(self, name)

    def tic(self, name):
        """Start timing a named phase.

        Parameters
        ----------
        name : str
            Phase label. Must be followed by a matching ``toc(name)``.
        """
        if self.enabled:
            self._tics[name] = time.perf_counter()

    def toc(self, name):
        """Stop timing and accumulate elapsed time for a named phase.

        Parameters
        ----------
        name : str
            Phase label previously started with ``tic(name)``.
        """
        if self.enabled:
            elapsed = time.perf_counter() - self._tics.pop(name)
            self._times[name] = self._times.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    @property
    def timer(self):
        """Dict of {phase: cumulative_time} for backward compatibility."""
        return dict(self._times)

    @property
    def counts(self):
        """Dict of {phase: call_count}."""
        return dict(self._counts)

    def summary(self):
        """Return a formatted timing breakdown table as a string.

        Returns
        -------
        table : str
            Multi-line table sorted by descending cumulative time.
        """
        if not self._times:
            return "No profiling data."
        total = sum(self._times.values())
        lines = [
            "Phase                      Time (s)   Calls   Avg (ms)      %",
            "\u2500" * 63,
        ]
        for name, t in sorted(self._times.items(), key=lambda x: -x[1]):
            c = self._counts[name]
            pct = 100 * t / total if total > 0 else 0
            lines.append(f"{name:<27s}{t:>8.4f}  {c:>6d}  {1000*t/c:>8.3f}  {pct:>5.1f}%")
        lines.append("\u2500" * 63)
        lines.append(f"{'Total':<27s}{total:>8.4f}")
        return "\n".join(lines)

    def reset(self):
        """Clear all recorded times, counts, and pending tics."""
        self._times.clear()
        self._counts.clear()
        self._tics.clear()


class _TimerContext:
    __slots__ = ('_timer', '_name')

    def __init__(self, timer, name):
        self._timer = timer
        self._name = name

    def __enter__(self):
        self._timer.tic(self._name)
        return self

    def __exit__(self, *args):
        self._timer.toc(self._name)
