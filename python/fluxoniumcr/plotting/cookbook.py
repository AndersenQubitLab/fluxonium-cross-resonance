import matplotlib as mpl
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import numpy as np


def colorline(ax, x, y, z, **kwargs):
    assert len(x) > 1

    points = np.stack([x, y], axis=1)

    dx = np.gradient(x)
    dy = np.gradient(y)
    derivs = np.stack([dx, dy], axis=1)

    path_points: list[list] = []
    path_codes: list[list] = []
    for i in range(len(points)-1):
        this_path_points = [points[i]]
        this_path_codes = [Path.MOVETO]

        if i == 0:
            this_path_points.extend([
                points[i+1] - derivs[i+1]/2,
                points[i+1],
            ])
            this_path_codes.extend([Path.CURVE3]*2)
        elif i == len(points) - 2:
            new_path_points = [
                points[i] + derivs[i]/2,
                points[i+1],
            ]
            new_path_codes = [Path.CURVE3]*2
            this_path_points.extend(new_path_points)
            this_path_codes.extend(new_path_codes)
            path_points[-1].extend(new_path_points)
            path_codes[-1].extend(new_path_codes)
        else:
            new_path_points = [
                points[i] + derivs[i]/3,
                points[i+1] - derivs[i+1]/3,
                points[i+1],
            ]
            new_path_codes = [Path.CURVE4]*3
            this_path_points.extend(new_path_points)
            this_path_codes.extend(new_path_codes)
            path_points[-1].extend(new_path_points)
            path_codes[-1].extend(new_path_codes)

        path_points.append(this_path_points)
        path_codes.append(this_path_codes)

    paths: list[Path] = []
    for p, c in zip(path_points, path_codes):
        paths.append(Path(p, c))

    path_collection = PathCollection(
        paths,
        fc='none',
        array=z,
        **kwargs
    )

    ax.add_collection(path_collection)
    return path_collection


def plot_triangle(ax, x, y, size, direction):
    if direction == 'up':
        sign = -1
    elif direction == 'down':
        sign = 1
    else:
        raise ValueError(direction)

    p = mpl.patches.Polygon(
        np.array([
            [0, 0],
            [0.5, sign*0.866],
            [-0.5, sign*0.866],
        ])/72 * size,
        fc='black',
        transform=ax.figure.dpi_scale_trans
        + mpl.transforms.ScaledTranslation(x, y, ax.transData),
    )

    ax.add_patch(p)


def imshow_heatmap(ax, x, y, z, dx=0.0, dy=0.0, **kwargs):
    x, y, z = map(np.asarray, (x, y, z))

    for name, a, da in zip(("x", "y"), (x, y), (dx, dy)):
        if not is_equidistant(a):
            raise ValueError(f"{name} must be an array of equidistant numbers")

        if len(a) == 1 and da == 0:
            raise ValueError(
                f"d{name} must be manually specified if len({name})=1"
            )

    if dx == 0.0:
        assert len(x) > 1
        dx = x[1] - x[0]

    if dy == 0.0:
        assert len(y) > 1
        dy = y[1] - y[0]

    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('interpolation', 'none')

    return ax.imshow(
        z,
        origin='lower',
        extent=(
            x.min() - dx/2,
            x.max() + dx/2,
            y.min() - dy/2,
            y.max() + dy/2,
        ),
        **kwargs
    )


def is_equidistant(a, **kwargs) -> bool:
    """Return True if `a` is a sequence of equidistant values."""
    b = np.linspace(a[0], a[-1], len(a))
    return np.allclose(a, b, **kwargs)


def bin2d_maximum(
        x_values,
        y_values,
        z_values,
        x_edges,
        y_edges,
        minimum=False,
):
    (
        x_values,
        y_values,
        z_values,
        x_edges,
        y_edges,
    ) = map(np.asarray, (
        x_values,
        y_values,
        z_values,
        x_edges,
        y_edges,
    ))

    hm = np.full(
        (len(y_edges) + 1, len(x_edges) + 1),
        np.nan,
    )

    ii = np.digitize(
        y_values,
        y_edges,
    )
    jj = np.digitize(
        x_values,
        x_edges,
    )

    if minimum is False:
        idx = np.argsort(z_values)
    else:
        idx = np.argsort(-z_values)

    hm[ii[idx], jj[idx]] = z_values[idx]

    # Edges are beyond the bins defined by x_edges and y_edges.
    return hm[1:-1, 1:-1]
