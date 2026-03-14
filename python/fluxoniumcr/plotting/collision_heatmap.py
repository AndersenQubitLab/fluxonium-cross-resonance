import matplotlib as mpl

import numpy as np


#  From colorbrewer2.org
mycmap = mpl.colors.LinearSegmentedColormap.from_list(  # type: ignore
    'mycmap',
    [
        '#67001f',
        '#b2182b',
        '#d6604d',
        '#f4a582',
        '#fddbc7',
        '#f7f7f7',
        '#d1e5f0',
        '#92c5de',
        '#4393c3',
        '#2166ac',
        '#053061'
    ][::-1]
)


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


def make_colorbar(
        cax,
        norm,
        cmap,
        vmin,
        vmax,
        yticks,
        extendfrac=0.05,
        label="",
):
    num_total = 1000
    num_triangle = round(num_total * extendfrac)
    num_center = num_total - 2*num_triangle
    stops = np.concatenate([
        np.geomspace(norm.vmin, vmin, num_triangle, endpoint=False),
        np.geomspace(vmin, vmax, num_center, endpoint=False),
        np.geomspace(vmax, norm.vmax, num_triangle),
    ])

    gradient = cax.imshow(
        cmap(norm(stops))[:, None],
        aspect='auto',
        extent=(0, 1, 0, 1,),
        origin='lower',
        interpolation='bilinear',
    )

    path = mpl.path.Path(  # type: ignore
        [
            [0.5, 0.0],
            [1.0, extendfrac],
            [1.0, 1-extendfrac],
            [0.5, 1.0],
            [0.0, 1 - extendfrac],
            [0.0, extendfrac],
            [0.5, 0.0],
        ],
        closed=True,
    )
    patch = mpl.patches.PathPatch(  # type: ignore
        path,
        facecolor='none',
        edgecolor=mpl.rcParams['axes.edgecolor'],
        linewidth=mpl.rcParams['axes.linewidth'],
        transform=cax.transData,
        clip_on=False,
    )
    cax.add_patch(patch)
    gradient.set_clip_path(patch)
    cax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    cax.tick_params(
        which='both',
        labelleft=False,
        left=False,
        labelbottom=False,
        bottom=False,
        right=True,
        labelright=True,
    )

    yticks_minor = np.concatenate(
        [
            np.arange(1, 10) * 10**round(np.log10(y))
            for y in yticks[:-1]
        ]
    )

    yticks_pos = (
        extendfrac
        + (1 - 2*extendfrac)
            * (np.log10(yticks) - np.log10(vmin))
            / (np.log10(vmax) - np.log10(vmin))
    )

    yticks_minor_pos = (
        extendfrac
        + (1 - 2*extendfrac)
            * (np.log10(yticks_minor) - np.log10(vmin))
            / (np.log10(vmax) - np.log10(vmin))
    )

    cax.set_yticks(
        yticks_pos,
        [
            "$10^{" f"{round(np.log10(y))}" "}$"
            for y in yticks
        ]
    )
    cax.set_yticks(
        yticks_minor_pos,
        minor=True
    )

    cax.yaxis.set_label_position('right')
    cax.set_ylabel(label)
