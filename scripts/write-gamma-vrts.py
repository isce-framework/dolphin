#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def create_gamma_vrts(par_files: Sequence[Path]) -> None:
    """Create VRT files that describe raw GAMMA binary files (.rslc, .slc, etc.).

    Parameters
    ----------
    par_files : Sequence[str | Path]
        List of GAMMA .par files.

    Raises
    ------
    ValueError
        If required fields (e.g. image format, dimensions) are missing.

    """
    gamma_format_to_gdal = {
        "SCOMPLEX": ("CInt16", 4),
        # TODO: I dont know GAMMA formats?
        # "FCOMPLEX": ("CFloat32", 8),
        # "FLOAT": ("Float32", 4),
    }

    key_val_re = re.compile(r"^(?P<key>[\w_]+):\s+(?P<value>.+)$")

    for par_file in par_files:
        if not par_file.exists():
            raise FileNotFoundError(par_file)

        metadata = {}
        for line in par_file.read_text().splitlines():
            if match := key_val_re.match(line.strip()):
                k, v = match["key"], match["value"]
                metadata[k] = v

        try:
            xsize = int(metadata["range_samples"])
            ysize = int(metadata["azimuth_lines"])
            fmt = metadata["image_format"].strip()
        except KeyError as e:
            raise ValueError(f"Missing required field in {par_file.name}") from e

        try:
            gdal_dtype, pixel_bytes = gamma_format_to_gdal[fmt]
        except KeyError as e:
            raise ValueError(
                f"Unsupported image_format in {par_file.name}: {fmt}"
            ) from e

        line_bytes = pixel_bytes * xsize

        vrt_path = par_file.with_suffix(".vrt")
        bin_file = par_file.with_suffix("")  # strips ".par"
        if not bin_file.exists():
            raise FileNotFoundError(bin_file)

        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write(f'<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">\n')
            f.write(
                f'  <VRTRasterBand dataType="{gdal_dtype}" band="1"'
                ' subClass="VRTRawRasterBand">\n    <SourceFilename'
                f' relativeToVRT="1">{bin_file.name}</SourceFilename>\n   '
                " <ImageOffset>0</ImageOffset>\n   "
                f" <PixelOffset>{pixel_bytes}</PixelOffset>\n   "
                f" <LineOffset>{line_bytes}</LineOffset>\n   "
                " <ByteOrder>MSB</ByteOrder>\n  </VRTRasterBand>\n</VRTDataset>\n"
            )

        print(f"[vrt] {vrt_path} -> {bin_file.name} ({gdal_dtype}, {xsize}x{ysize})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("par_files", nargs="+", type=Path)
    args = parser.parse_args()
    create_gamma_vrts(args.par_files)
