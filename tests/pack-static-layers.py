import subprocess

import h5py
import numpy as np

from dolphin import io

if __name__ == "__main__":
    for fn in [
        "OPERA_L2_CSLC-S1-STATIC_T087-185683-IW2_20140403_S1A_v1.0.h5",
        "OPERA_L2_CSLC-S1-STATIC_T087-185684-IW2_20140403_S1A_v1.0.h5",
    ]:
        with h5py.File(fn, "a") as hf:
            shape = hf["data/layover_shadow_mask"].shape
            hf["data/layover_shadow_mask"][()] = np.zeros(shape, dtype="float32")
            hf["data/local_incidence_angle"][()] = np.zeros(shape, dtype="float32")
            le = hf["data/los_east"][()]
            ln = hf["data/los_north"][()]
            io.round_mantissa(le, keep_bits=6)
            io.round_mantissa(ln, keep_bits=6)
            hf["data/los_north"][()] = ln
            hf["data/los_east"][()] = le

        subprocess.run(f"h5repack {fn} {fn}.repack.h5", shell=True, check=True)
