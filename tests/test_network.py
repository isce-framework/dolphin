import pytest

from dolphin import network


@pytest.fixture
def slc_list():
    return ["20220101", "20220201", "20220301", "20220401"]


def test_single_reference_network(slc_list):
    ifg_list = network.single_reference_network(slc_list, reference_idx=0)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
    ]
    ifg_list = network.single_reference_network(slc_list, reference_idx=1)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
    ]


def test_limit_by_bandwidth(slc_list):
    ifg_list = network.limit_by_bandwidth(slc_list, max_bandwidth=1)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220301", "20220401"),
    ]
    ifg_list = network.limit_by_bandwidth(slc_list, max_bandwidth=2)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]
    ifg_list = network.limit_by_bandwidth(slc_list, max_bandwidth=3)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]


def test_limit_by_temporal_baseline(slc_list):
    ifg_list = network.limit_by_temporal_baseline(slc_list, max_baseline=1)
    assert ifg_list == []

    ifg_list = network.limit_by_temporal_baseline(slc_list, max_baseline=31)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220301", "20220401"),
    ]
    ifg_list = network.limit_by_temporal_baseline(slc_list, max_baseline=61)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]
    ifg_list = network.limit_by_temporal_baseline(slc_list, max_baseline=500)
    assert ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]
