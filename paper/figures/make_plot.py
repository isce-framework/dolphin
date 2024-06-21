import sys
from pathlib import Path

import apertools.plotting
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rioxarray

p = Path(__file__).parent

filename = "velocity_compressed.tif"
image = rioxarray.open_rasterio(p / filename).sel(band=1)
image_ll = image[::4, ::4].rio.reproject("EPSG:4326")

fig, ax = apertools.plotting.plot_image_with_background(
    image_ll.where(image_ll != 0),
    cbar_label="[mm / year]",
    cmap="RdBu_r",
    vmax=10,
    vmin=-10,
    tile_zoom_level=9,
    figsize=(4),
    interpolation="none",
)
apertools.plotting.add_ticks(ax, resolution=0.5)
apertools.plotting.add_zebra_frame(ax, crs=ccrs.PlateCarree())

fig.tight_layout()
if len(sys.argv) > 1 and sys.argv[1] == "--show":
    plt.show(block=True)

outname = p / "bristol-velocity-sequential.png"
fig.savefig(outname, dpi=300, transparent=True)
