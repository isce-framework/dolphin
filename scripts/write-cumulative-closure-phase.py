#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
from opera_utils import get_dates

from dolphin import constants, io

if __name__ == "__main__":
    reader = io.RasterStackReader.from_file_list(Path(sys.argv[1]).glob("clos*tif"))

    running_sum = np.zeros(reader.shape[1:], dtype="float64")
    for idx, fin in enumerate(reader.file_list):
        running_sum += reader[idx, :, :].filled(0).squeeze().astype("float64")
        fname = f"cumulative_closure_phase_{get_dates(fin)[1].strftime('%Y%m%d')}.tif"
        fout = Path(fin).parent / fname
        io.write_arr(
            # Flip sign to match convention of displacement
            arr=running_sum.astype("float32")
            * constants.SENTINEL_1_WAVELENGTH
            / -4
            / np.pi,
            output_name=fout,
            like_filename=fin,
            options=io.EXTRA_COMPRESSED_TIFF_OPTIONS,
        )
