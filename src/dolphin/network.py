import itertools
from typing import List, Optional, Tuple

from dolphin.log import get_log
from dolphin.utils import parse_slc_strings

logger = get_log()


def make_ifg_list(
    slc_list: List[str],
    max_bandwidth: Optional[int] = None,
    max_temporal_baseline: Optional[float] = None,
    reference_idx: Optional[int] = None,
    final_only: bool = False,
) -> List[Tuple]:
    """Form a list of interferogram names from a list of dates."""
    slc_list = sorted(slc_list)
    if final_only:
        # Just form the final nearest-neighbor ifg
        return [tuple(slc_list[-2:])]
    elif max_bandwidth is not None:
        return limit_by_bandwidth(slc_list, max_bandwidth)
    elif max_temporal_baseline is not None:
        return limit_by_temporal_baseline(slc_list, max_temporal_baseline)
    elif reference_idx is not None:
        return single_reference_network(slc_list, reference_idx)
    else:
        raise ValueError("No valid ifg list generation method specified")


def single_reference_network(date_list: List[str], reference_idx=0) -> List[Tuple]:
    """Form a list of single-reference interferograms."""
    if len(date_list) < 2:
        raise ValueError("Need at least two dates to make an interferogram list")
    ref = date_list[reference_idx]
    ifgs = [tuple(sorted([ref, date])) for date in date_list if date != ref]
    return ifgs


def limit_by_bandwidth(slc_date_list: List[str], max_bandwidth: int):
    """Form a list of the "nearest-`max_bandwidth`" ifgs.

    Parameters
    ----------
    slc_date_list : list
        List of dates of SLCs
    max_bandwidth : int
        Largest allowed span of ifgs, by index distance, to include.
        max_bandwidth=1 will only include nearest-neighbor ifgs.

    Returns
    -------
    list
        Pairs of (date1, date2) ifgs
    """
    slc_to_idx = {s: idx for idx, s in enumerate(slc_date_list)}
    return [
        (a, b)
        for (a, b) in _all_pairs(slc_date_list)
        if slc_to_idx[b] - slc_to_idx[a] <= max_bandwidth
    ]


def limit_by_temporal_baseline(
    slc_date_list: List[str], max_baseline: Optional[float] = None
):
    """Form a list of the ifgs limited to a maximum temporal baseline.

    Parameters
    ----------
    slc_date_list : list
        List of dates of SLCs
    max_baseline : float, optional
        Largest allowed span of ifgs, by index distance, to include.
        max_bandwidth=1 will only include nearest-neighbor ifgs.

    Returns
    -------
    list
        Pairs of (date1, date2) ifgs
    """
    ifg_strs = _all_pairs(slc_date_list)
    ifg_dates = _all_pairs(parse_slc_strings(slc_date_list))
    baselines = [_temp_baseline(ifg) for ifg in ifg_dates]
    return [ifg for ifg, b in zip(ifg_strs, baselines) if b <= max_baseline]


def _all_pairs(slclist):
    """Create the list of all possible ifg pairs from slclist."""
    return list(itertools.combinations(slclist, 2))


def _temp_baseline(ifg_pair):
    return (ifg_pair[1] - ifg_pair[0]).days
