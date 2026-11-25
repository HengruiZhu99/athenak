"""Continue both MOTSs through a turning point using the five-cycle ADM archive."""

from functools import lru_cache
import numpy as np
from formation_branches import NativePatch, read_binary
from trace_birth import History, trace


class DenseHistory:
    from_coefficients = History.from_coefficients

    def __init__(self, rows, mass, stride):
        self.paths = [r[2] for r in rows]
        self.times = np.array([r[0] / mass for r in rows])
        self.lo, self.hi = self.times[[0, -1]]
        self.base = NativePatch(read_binary(self.paths[-1]), stride)

        @lru_cache(maxsize=8)
        def coeff(index):
            p = NativePatch(read_binary(self.paths[index]), stride)
            if self.base.dx != p.dx or self.base.bound != p.bound:
                raise ValueError("incompatible spatial patches")
            for a, b in zip(p.splines, self.base.splines):
                if not all(
                    np.array_equal(x, y) for x, y in zip(a.get_knots(), b.get_knots())
                ):
                    raise ValueError("different geometry knots")
            return np.array([s.get_coeffs() for s in p.splines])

        self.coeff = coeff

    def geometry(self, t, method):
        times = self.times
        i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
        a, b = self.coeff(i), self.coeff(i + 1)
        dt = times[i + 1] - times[i]
        f = (t - times[i]) / dt
        c = (1 - f) * a + f * b
        if method == "hermite":
            sec = (b - a) / dt
            left = (
                sec
                if i == 0
                else (b - self.coeff(i - 1)) / (times[i + 1] - times[i - 1])
            )
            right = (
                sec
                if i + 2 >= len(times)
                else (self.coeff(i + 2) - a) / (times[i + 2] - times[i])
            )
            c += dt * f * (1 - f) * ((1 - f) * (left - sec) - f * (right - sec))
        return self.from_coefficients(c)


def dense_fold(rows, pairs, mass):
    if not pairs:
        return dict(status="unresolved", reason="no resolved marginal pair")
    # Seed both branches on a resolved leaf shortly after first detection.
    seed = pairs[min(8, len(pairs) - 1)]
    regular = [r for r in rows if r[1] % 5 == 0 and r[1] <= seed["cycle"]]
    if len(regular) < 5:
        return dict(status="unresolved", reason="insufficient early geometry")
    dt = float(np.median(np.diff([r[0] / mass for r in regular])))
    out = []
    for stride, method, order in [
        (1, "hermite", 48),
        (1, "linear", 48),
        (1, "hermite", 40),
        (2, "hermite", 48),
    ]:
        try:
            history = DenseHistory(regular, mass, stride)
            fold, points = trace(
                history,
                order,
                method,
                seed["inner"]["coeff"],
                seed["outer"]["coeff"],
                mass,
            )
            times = history.times
            idx = int(np.searchsorted(times, fold["t_M"]))
            if idx == 0 or idx == len(times):
                raise ValueError("turning point outside evolved archive")
            out.append(
                dict(
                    stride=stride,
                    method=method,
                    order=order,
                    fold=fold,
                    points=points,
                    adjacent_evolved_times_M=times[idx - 1 : idx + 1].tolist(),
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError, StopIteration) as err:
            out.append(
                dict(
                    stride=stride,
                    method=method,
                    order=order,
                    failure=str(err) or "no zero crossing",
                )
            )
    return dict(
        status="reconstructed" if "fold" in out[0] else "unresolved",
        cadence_M=dt,
        variants=out,
    )
