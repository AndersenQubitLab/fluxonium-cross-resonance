from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.util
from pathlib import Path
import struct
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hash_arguments(args, *, hash_obj=None) -> bytes:
    hash_obj = hash_obj or hashlib.sha1()
    for arg in args:
        if isinstance(arg, (tuple, list)):
            hash_arguments(arg, hash_obj=hash_obj)
        elif isinstance(arg, float):
            hash_obj.update(struct.pack("@f", arg))
        elif isinstance(arg, int):
            hash_obj.update(struct.pack("@l", arg))
        elif isinstance(arg, str):
            hash_obj.update(arg.encode('utf8'))
        else:
            hash_obj.update(arg.tobytes())
    return hash_obj.digest()


def load_arguments(parent_path: Path|str) -> ModuleType:
    parent_path = Path(parent_path)
    spec = importlib.util.spec_from_file_location(
        "arguments",
        parent_path/"arguments.py",
    )

    assert spec is not None
    assert spec.loader is not None


    argm = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(argm)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"arguments.py not found in '{parent_path!s}'"
        )
    return argm


finite_difference_grid = np.linspace(-4, 4, 17)
finite_difference_table = np.array([
    [1/280, 0, -4/105, 0, 1/5, 0, -4/5, 0, 0, 0, 4/5, 0, -1/5, 0, 4/105, 0, -1/280],
    [0, 0, 0, 0, 1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280, 0, 0, 0, 0],
    [-1/560, 0, 8/315, 0, -1/5, 0, 8/5, 0, -205/72, 0, 8/5, 0, -1/5, 0, 8/315, 0, -1/560],
    [0, 0, 0, 0, -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560, 0, 0, 0, 0],
    [-7/240, 0, 3/10, 0, -169/120, 0, 61/30, 0, 0, 0, -61/30, 0, 169/120, 0, -3/10, 0, 7/240],
    [0, 0, 0, 0, -7/240, 3/10, -169/120, 61/30, 0, -61/30, 169/120, -3/10, 7/240, 0, 0, 0, 0],
])
finite_difference_table[1::2] *= 2**np.arange(1, 4)[:, None]
# Divide the coefficients by n! because we want the Taylor series coefficients, not the derivatives.
finite_difference_table /= np.array([1, 1, 2, 2, 6, 6])[:, None]


def taylor_expand(
        f: Callable[[NDArray], ArrayLike],
        x0: float,
        initial_step: float = 1.0,
        maxiter=10,
        abs_tol: float = 1e-12,
        rel_tol: float = 1e-8,
        order: int = 3,
) -> NDArray:
    """Numerically calculate the Taylor expansion of `f` around `x0` using central finite differences."""
    if order < 1 or order > 3:
        raise ValueError(f"{order=} must be between 1 and 3 (inclusive).")

    step_size = initial_step
    prev_max_error = np.inf

    for _ in range(maxiter):
        x_grid = x0 + step_size * finite_difference_grid
        f_samples = np.asarray(f(x_grid))

        df_samples_both = np.einsum(
            '...i,ji',
            f_samples,
            finite_difference_table[:2*order],
        )
        df_samples_both = df_samples_both.reshape(*df_samples_both.shape[:-1], -1, 2)
        df_samples_both *= step_size**-np.arange(1, 1+order)[:, None]
        df_samples1 = df_samples_both[..., 0]
        df_samples2 = df_samples_both[..., 1]
        error = abs(df_samples2 - df_samples1)

        done = (error < abs_tol + rel_tol*abs(df_samples2)).all()
        if done or error.max() > prev_max_error: break

        prev_max_error = error.max()
        step_size /= 4

    result = np.concatenate(
        [
            f_samples[..., 8][None],  # Zeroth order term f(x)  # type: ignore
            np.moveaxis(df_samples2, -1, 0)  # type: ignore
        ],
        axis=0,
    )

    return result
