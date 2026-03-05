from pathlib import Path

import pytest

from dolphin import baseline

isce3 = pytest.importorskip("isce3")


@pytest.fixture
def cslc_files():
    p = Path(__file__).parent / "data/baseline_cslcs"
    return sorted(p.glob("*.h5"))


def test_baseline_compute(cslc_files):
    ref_file, sec_file = cslc_files
    lon, lat, baseline_grid = baseline.compute_baselines(
        h5file_ref=ref_file, h5file_sec=sec_file
    )
    assert lon.shape == lat.shape == baseline_grid.shape == (46, 199)
    pytest.approx([baseline_grid.min(), 134.98131295549837])
    pytest.approx([baseline_grid.max(), 141.95610459910463])
