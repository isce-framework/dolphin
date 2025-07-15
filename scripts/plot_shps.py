from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from dolphin import shp
from dolphin.io import VRTStack


def plot_shps(
    slc_stack,
    mean,
    var,
    half_window: dict[str, int],
    display_arr=None,
    shp_methods=None,
    shp_alpha=0.05,
    shp_nslc=None,
    cmap_nmap="Reds_r",
    block=False,
):
    """Interactively plot the neighborhood map over an SLC image.

    Click on a pixel to see the neighborhood map.
    """
    if shp_methods is None:
        shp_methods = ["glrt", "ks"]
    if isinstance(slc_stack, str):
        slc_stack = VRTStack(slc_stack, write_file=False)
    if display_arr is None:
        display_arr = mean
    if shp_nslc is None:
        shp_nslc = slc_stack.shape[0]
    if not isinstance(shp_alpha, list):
        shp_alpha = [shp_alpha] * len(shp_methods)

    ny, nx = half_window["y"], half_window["x"]
    halfwin_rowcol = (ny, nx)
    # window = (1 + 2 * ny, 1 + 2 * nx)

    fig, axes = plt.subplots(
        ncols=len(shp_methods), sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()
    for ax, shp_method in zip(axes, shp_methods, strict=False):
        ax.imshow(_scale_mag(mean), cmap="gray")
        ax.set_title(shp_method)

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if event.button != 1 or not event.inaxes:
            return
        if event.inaxes not in axes:
            return
        state = fig.canvas.manager.toolbar.mode
        if state != "":  # Zoom/other tool is active
            return

        # Save limits to restore after adding neighborhoods
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        row, col = int(event.ydata), int(event.xdata)
        # Somehow clicked outside image, but in axis
        if row >= display_arr.shape[0] or col >= display_arr.shape[1]:
            return

        for i, shp_method in enumerate(shp_methods):
            # Calc SHPS:
            ax = axes[i]
            rs, cs = _get_slices(row, col, ny, nx)
            amp_stack = np.abs(slc_stack[:, rs, cs]) if shp_method == "ks" else None

            cur_shps = shp.estimate_neighbors(
                halfwin_rowcol=halfwin_rowcol,
                alpha=shp_alpha[i],
                mean=mean[rs, cs],
                var=var[rs, cs],
                nslc=shp_nslc,
                amp_stack=amp_stack,
                method=shp_method,
            )
            cur_shps = cur_shps[ny, nx, :, :]
            n_img = np.ma.masked_not_equal(cur_shps, True)
            extent = _get_extent(row, col, ny, nx)

            # Remove old neighborhood images
            if len(ax.get_images()) > 1:
                # event.inaxes.get_images()[1].remove()
                ax.get_images()[-1].remove()

            ax.imshow(
                n_img,
                cmap=cmap_nmap,
                alpha=0.8,
                extent=extent,
                origin="lower",
            )
            # Remove old neighborhood patches
            for p in ax.patches:
                p.remove()
            # add a rectangle around the neighborhood
            rect = patches.Rectangle(
                (extent[0], extent[2]),
                1 + 2 * nx,
                1 + 2 * ny,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Restore original viewing bounds
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.tight_layout()

    plt.show(block=block)


def _get_extent(row, col, ny, nx):
    """Get the row/col extent of the window surrounding a pixel."""
    # Matplotlib extent is (left, right, bottom, top)
    # Also the extent for normal `imshow` is shifted by -0.5
    return col - nx - 0.5, col + nx + 1 - 0.5, row - ny - 0.5, row + ny + 1 - 0.5


def _get_slices(row, col, ny, nx):
    """Get the row/col slice of the window surrounding a pixel."""
    return slice(row - ny, row + ny + 1), slice(col - nx, col + nx + 1)


def _scale_mag(img, exponent=0.3, max_pct=99.95):
    """Scale the magnitude of complex radar image for better display."""
    out = np.abs(img) ** exponent
    max_val = np.nanpercentile(out, max_pct)
    return np.clip(out, None, max_val)


def get_cli_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively view neighborhood maps over an SLC."
    )
    parser.add_argument(
        "-n",
        "--nmap-file",
        help="Neighborhood map file",
        required=True,
    )
    parser.add_argument(
        "--slc-stack-file",
        help="SLC stack file",
    )
    parser.add_argument(
        "--slc-stack-bands",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Bands to use from SLC stack file",
    )
    parser.add_argument(
        "--slc-file",
        help="Alternative background: a single SLC filename.",
    )
    parser.add_argument(
        "--ps-file",
        default="PS/ps_pixels",
        help="PS file to overlay",
    )
    parser.add_argument(
        "--cmap-nmap",
        default="Reds_r",
        help="Colormap for neighborhood map",
    )
    parser.add_argument(
        "--shp-alpha",
        type=float,
        default=0.01,
        help="Alpha significance value for neighborhood map",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Don't block the matplotlib window (default is blocking)",
    )
    return parser.parse_args()


def run_cli():
    """Run the command line interface."""
    args = get_cli_args()
    plot_shps(
        nmap_filename=args.nmap_file,
        slc_stack_filename=args.slc_stack_file,
        slc_stack_bands=args.slc_stack_bands,
        slc_filename=args.slc_file,
        ps_filename=args.ps_file,
        cmap_nmap=args.cmap_nmap,
        block=not args.no_block,
    )


if __name__ == "__main__":
    run_cli()
