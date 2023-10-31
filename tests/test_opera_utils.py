from __future__ import annotations

import random
import zipfile
from itertools import chain
from pathlib import Path

import pytest

from dolphin.opera_utils import (
    BurstSubsetOption,
    get_burst_id,
    get_missing_data_options,
    group_by_burst,
)


def test_get_burst_id():
    assert (
        get_burst_id("t087_185678_iw2/20180210/t087_185678_iw2_20180210.h5")
        == "t087_185678_iw2"
    )
    # Check the official naming convention
    fn = "OPERA_L2_CSLC-S1_T087-185678-IW2_20180210T232711Z_20230101T100506Z_S1A_VV_v1.0.h5"  # noqa
    assert get_burst_id(fn) == "t087_185678_iw2"


def test_group_by_burst():
    expected = {
        "t087_185678_iw2": [
            Path("t087_185678_iw2/20180210/t087_185678_iw2_20180210.h5"),
            Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318.h5"),
            Path("t087_185678_iw2/20180423/t087_185678_iw2_20180423.h5"),
        ],
        "t087_185678_iw3": [
            Path("t087_185678_iw3/20180210/t087_185678_iw3_20180210.h5"),
            Path("t087_185678_iw3/20180318/t087_185678_iw3_20180318.h5"),
            Path("t087_185678_iw3/20180517/t087_185678_iw3_20180517.h5"),
        ],
        "t087_185679_iw1": [
            Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210.h5"),
            Path("t087_185679_iw1/20180318/t087_185679_iw1_20180318.h5"),
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


@pytest.fixture
def idaho_slc_list() -> list[str]:
    p = Path(__file__).parent / "data" / "idaho_slc_file_list.txt.zip"

    # unzip the file and return the list of strings
    with zipfile.ZipFile(p) as z:
        with z.open(z.namelist()[0]) as f:
            return f.read().decode().splitlines()


def test_get_missing_data_options(idaho_slc_list):
    burst_subset_options = get_missing_data_options(idaho_slc_list)

    full_burst_id_list = [
        "t071_151161_iw1",
        "t071_151161_iw2",
        "t071_151161_iw3",
        "t071_151162_iw1",
        "t071_151162_iw2",
        "t071_151162_iw3",
        "t071_151163_iw1",
        "t071_151163_iw2",
        "t071_151163_iw3",
        "t071_151164_iw1",
        "t071_151164_iw2",
        "t071_151164_iw3",
        "t071_151165_iw1",
        "t071_151165_iw2",
        "t071_151165_iw3",
        "t071_151166_iw1",
        "t071_151166_iw2",
        "t071_151166_iw3",
        "t071_151167_iw1",
        "t071_151167_iw2",
        "t071_151167_iw3",
        "t071_151168_iw1",
        "t071_151168_iw2",
        "t071_151168_iw3",
        "t071_151169_iw1",
        "t071_151169_iw2",
        "t071_151169_iw3",
    ]
    # The correct options should be
    expected_1 = full_burst_id_list[3:]
    expected_2 = full_burst_id_list[-3:]
    expected_3 = full_burst_id_list

    assert isinstance(burst_subset_options[0], BurstSubsetOption)

    expected_id_lists = [expected_1, expected_2, expected_3]
    expected_num_dates = [173, 245, 11]
    expected_total_num_bursts = [4152, 735, 297]
    for i, option in enumerate(burst_subset_options):
        assert option.burst_id_list == expected_id_lists[i]
        assert option.num_burst_ids == len(expected_id_lists[i])
        assert option.num_dates == expected_num_dates[i]
        assert option.total_num_bursts == expected_total_num_bursts[i]
