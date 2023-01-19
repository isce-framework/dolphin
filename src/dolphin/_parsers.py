import datetime
import re
from collections import namedtuple
from pathlib import Path

from ._types import Filename

# Example filename for OPERA CSLC:
# t087_185678_iw2_20180210_VV.h5
DATETIME_FORMAT = "%Y%m%d"
OPERA_CSLC_RE = re.compile(
    r"""t(?P<track>\d{3})      # track number, 3 digits
        _(?P<burst_id>\d{6})   # ESA burst id, 6 digits
        _(?P<subswath>iw[1-3]) # subswath iw1, iw2, iw3
        _(?P<datetime>\d{8})       # datetime, YYYYMMDD
        _(?P<pol>[HV]{2})      # polarization, HH, HV, VV, VH
        (\.[h5|nc])?           # extension of .h5 or .nc
    """,
    re.VERBOSE,
)

__all__ = ["parse_opera_cslc", "BurstSlc"]

BurstSlc = namedtuple(
    "BurstSlc", ["track", "burst_id", "subswath", "datetime", "pol", "filename"]
)


def parse_opera_cslc(filename: Filename) -> BurstSlc:
    """Parse OPERA CSLC filename.

    Parameters
    ----------
    filename : str
        Filename to parse.

    Returns
    -------
    dict
        BurstSlc parsed from `filename`.

    Examples
    --------
    >>> _parse_opera_cslc('t087_185678_iw2_20180210_VV.h5')
    BurstSlc(track=87, burst_id=185678, subswath='iw2',\
 datetime=datetime.datetime(2018, 2, 10, 0, 0), pol='VV',\
 filename=PosixPath('t087_185678_iw2_20180210_VV.h5'))
    """
    p = Path(filename)
    m = OPERA_CSLC_RE.match(p.stem)
    if m is None:
        raise ValueError("Could not parse filename: %s" % filename)
    d = m.groupdict()
    d["track"] = int(d["track"])
    d["burst_id"] = int(d["burst_id"])
    d["datetime"] = datetime.datetime.strptime(d["datetime"], DATETIME_FORMAT)
    d["filename"] = p

    return BurstSlc(**d)
