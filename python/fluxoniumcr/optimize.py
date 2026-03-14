from collections.abc import Callable

import numpy as np


def find_root(func, x0, x1, step_size, xtol=2e-12, maxiter=None):
    if x0 > x1:
        return None, True

    if func(x0) == 0.0:
        return x0, True

    y0 = func(x0)
    x = x0

    while x < x1:
        x = min(x + step_size, x1)
        y = func(x)

        if not np.isfinite(y) and np.isfinite(y0):
            xa = find_domain_boundary(func, x0, x, xtol=xtol, maxiter=maxiter)
            root, is_zero = find_root_bisect(func, x0, xa, xtol=xtol, maxiter=maxiter)
            if root is not None:
                return root, is_zero
            x0 = xa
            y0 = func(xa)
        elif np.isfinite(y) and not np.isfinite(y0):
            x0 = find_domain_boundary(func, x0, x, xtol=xtol, maxiter=maxiter)
            y0 = func(x0)
        elif not np.isfinite(y) and not np.isfinite(y0):
            x0 = x
            y0 = y
        elif np.sign(y) == np.sign(y0):
            x0 = x
            y0 = y
        else:
            break
    else:
        if func(x1) == 0.0:
            return x1, True
        else:
            return None, True


    root, is_zero = find_root_bisect(func, x0, x, xtol=xtol, maxiter=maxiter)

    if root is None:
        return find_root(
            func,
            x,
            x1,
            step_size=step_size,
            xtol=xtol,
            maxiter=maxiter,
        )

    return root, is_zero


def find_root_bisect(func, x0, x1, xtol=2e-12, maxiter=None):
    y0 = func(x0)
    y1 = func(x1)

    if np.sign(y0) == np.sign(y1):
        return None, True

    x = None

    iter_remaining = maxiter if maxiter is not None else -1
    while iter_remaining != 0:
        iter_remaining -= 1

        x = (x0 + x1)/2
        y = func(x)
        if not np.isfinite(y):
            xa = find_domain_boundary(func, x0, x, xtol=xtol, maxiter=maxiter)
            xb = find_domain_boundary(func, x, x1, xtol=xtol, maxiter=maxiter)
            ya = func(xa)
            yb = func(xb)
            if np.sign(ya) == np.sign(y1):
                x1 = xa
                y1 = ya
            elif np.sign(yb) == np.sign(y0):
                x0 = xb
                y0 = yb
            else:
                return x, False
                #  return None
        elif np.sign(y) == np.sign(y0):
            x0 = x
            y0 = y
        else:
            x1 = x
            y1 = y

        if x1 - x0 < xtol:
            break

    return x, True


def find_domain_boundary(func, x0, x1, xtol=2e-12, maxiter=None):
    y0 = func(x0)
    iter_remaining = maxiter if maxiter is not None else -1
    while iter_remaining != 0:
        iter_remaining -= 1
        x = (x0 + x1)/2
        y = func(x)
        if (not np.isfinite(y)) and (not np.isfinite(y0)) or np.isfinite(y) and np.isfinite(y0):
            x0 = x
            y0 = y
        else:
            x1 = x
        if abs(x1 - x0) < xtol:
            break
    if np.isnan(y0):
        return x1
    else:
        return x0


def get_monotonic_increasing_intervals(
        func: Callable[[float], float],
        dfunc: Callable[[float], float],
        x0: float,
        x1: float,
        step_size: float,
        xtol: float,
) -> tuple[list[tuple[float, float]], list[tuple[float,float]]]:
    intervals: list[tuple[float, float]] = []
    ranges: list[tuple[float, float]] = []

    xa = x0
    while True:
        xb, _ = find_root(dfunc, xa, x1, step_size=step_size, xtol=xtol)
        if xb is None:
            xb = x1

        ya = func(xa)
        yb = func(xb)
        if yb > ya:
            # Monotonic increasing interval
            intervals.append((xa, xb))
            ranges.append((ya, yb))

        if xb == x1:
            break
        else:
            xa = xb + xtol

    return intervals, ranges
