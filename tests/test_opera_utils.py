import random
from itertools import chain
from pathlib import Path

import pytest

from dolphin.opera_utils import group_by_burst


def test_group_by_burst():
    expected = {
        "t087_185678_iw2": [
            Path("t087_185678_iw2/20180210/t087_185678_iw2_20180210_VV.h5"),
            Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318_VV.h5"),
            Path("t087_185678_iw2/20180423/t087_185678_iw2_20180423_VV.h5"),
        ],
        "t087_185678_iw3": [
            Path("t087_185678_iw3/20180210/t087_185678_iw3_20180210_VV.h5"),
            Path("t087_185678_iw3/20180318/t087_185678_iw3_20180318_VV.h5"),
            Path("t087_185678_iw3/20180517/t087_185678_iw3_20180517_VV.h5"),
        ],
        "t087_185679_iw1": [
            Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            Path("t087_185679_iw1/20180318/t087_185679_iw1_20180318_VV.h5"),
        ],
    }
    in_files = list(chain.from_iterable(expected.values()))

    assert group_by_burst(in_files) == expected

    # Any order should work
    random.shuffle(in_files)
    # but the order of the lists of each key may be different
    for burst, file_list in group_by_burst(in_files).items():
        assert sorted(file_list) == sorted(expected[burst])


def test_group_by_burst_product_version():
    # Should also match this:
    # OPERA_L2_CSLC-S1_T078-165495-IW3_20190906T232711Z_20230101T100506Z_S1A_VV_v1.0.h5
    base = "OPERA_L2_CSLC-S1_"
    ending = "20230101T100506Z_S1A_VV_v1.0.h5"
    expected = {
        "t087_185678_iw2": [
            Path(f"{base}_T087-185678-IW2_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW2_20180318T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW2_20180423T232711Z_{ending}"),
        ],
        "t087_185678_iw3": [
            Path(f"{base}_T087-185678-IW3_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW3_20180318T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW3_20180517T232711Z_{ending}"),
        ],
        "t087_185679_iw1": [
            Path(f"{base}_T087-185679-IW1_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185679-IW1_20180318T232711Z_{ending}"),
        ],
    }
    in_files = list(chain.from_iterable(expected.values()))

    assert group_by_burst(in_files) == expected


def test_group_by_burst_non_opera():
    with pytest.raises(ValueError, match="Could not parse burst id"):
        group_by_burst(["20200101.slc", "20200202.slc"])
        # A combination should still error
        group_by_burst(
            [
                "20200101.slc",
                Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            ]
        )
