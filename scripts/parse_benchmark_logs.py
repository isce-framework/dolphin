import re
from pathlib import Path

import pandas as pd


def get_df(path):
    """Create a dataframe from a directory of log files."""
    rows = []
    for f in Path(path).glob("*.log"):
        result = _get_memory(f)
        attr = _parse_filename(f)
        result.update(attr)
        rows.append(result)
    return pd.DataFrame(rows)


def _get_memory(logfile):
    out = {"file": str(Path(logfile).resolve())}
    mempat = r"Maximum memory usage: (\d\.\d{2}) GB"
    timepat = (
        r"Total elapsed time for dolphin.workflows.s1_disp.run : (\d*\.\d{2}) minutes"
        r" \((\d*\.\d{2}) seconds\)"
    )
    for line in open(logfile).readlines():
        m = re.search(mempat, line)
        if m:
            out["memory"] = float(m.groups()[0])
            continue
        m = re.search(timepat, line)
        if m:
            out["runtime"] = float(m.groups()[1])
    return out


def _parse_filename(logfile):
    # dolphin_config_cpu_block1GB_strides2_tpw8_nslc27_nworkers4.log
    # dolphin_config_gpu_block1GB_strides2_tpw32_nslc15.log
    # Create the regular expression pattern with named groups
    pat = (
        r"dolphin_config_"
        r"(?P<device>\w+)_"
        r"block(?P<block>\d+)GB_"
        r"strides(?P<strides>\d+)_"
        r"tpw(?P<tpw>\d+)_"
        r"nslc(?P<nslc>\d+)"
        # nworkers is optional
        r"(_nworkers(?P<nworkers>\d+))?"
    )
    m = re.match(pat, Path(logfile).stem)
    if not m:
        raise ValueError(f"Could not parse filename: {logfile}")
    out_dict = m.groupdict()
    # convert numbers to ints
    for k, v in out_dict.items():
        if v is not None and k != "device":
            out_dict[k] = int(v)

    return out_dict
