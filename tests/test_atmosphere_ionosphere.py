"""Test IONEX functionalities: file reading and interpolation."""

from pathlib import Path

import numpy as np

from dolphin.atmosphere import ionosphere

TEST_TEC_FILE = Path(__file__).parent / "data/ionosphere_files/jplg3190.15i"


def test_read_ionex():
    """Test the reader for IONEX data."""

    time_ind = 1
    x0, x1, y0, y1 = 3, 9, 28, 33

    # Create TEC data on a region of interest (AOI)
    tec_aoi = np.array(
        [
            [71.8, 64.2, 55.9, 47.1, 38.6, 31.2],
            [80.2, 73.9, 66.6, 58.2, 49.5, 41.1],
            [83.2, 79.6, 74.6, 68.0, 60.1, 51.6],
            [79.6, 79.5, 78.1, 74.5, 68.5, 60.9],
            [71.9, 74.5, 76.5, 76.2, 73.1, 67.3],
        ],
    )

    # Read IONEX tec_maps data - ignore mins, lats, and lons
    times, lats, lons, tec_maps = ionosphere.read_ionex(TEST_TEC_FILE)[:4]
    assert np.allclose(tec_maps[time_ind, y0:y1, x0:x1], tec_aoi)
    # Check the shape of the data

    assert times.shape == (13,)
    assert lats.shape == (71,)
    assert lons.shape == (73,)
    assert tec_maps.shape == (13, 71, 73)

    assert np.allclose(times[:4], [0, 120, 240, 360])
    assert np.allclose(times[-4:], [1080, 1200, 1320, 1440])

    assert np.allclose(lats[:4], [87.5, 85, 82.5, 80])

    assert np.allclose(lons[:4], [-180, -175, -170, -165])


def test_get_ionex_value():
    """Test IONEX TEC data interpolation."""

    # Lat/Lon coordinates over Chile
    lat, lon = -21.3, -67.4

    # 23:07 UTC time
    utc_sec = 23 * 3600 + 7 * 60

    value = 64.96605174

    # Perform comparison
    tec_val = ionosphere.get_ionex_value(
        TEST_TEC_FILE,
        utc_sec,
        lat,
        lon,
    )
    assert np.allclose(tec_val, value, atol=1e-05, rtol=1e-05)


def test_get_ionex_value_arrays():
    """Test IONEX TEC data interpolation."""

    # Lat/Lon coordinates over Chile
    lat = [-21.3] * 3
    lon = [-67.4] * 3

    utc_sec = 23 * 3600 + 7 * 60

    value = 64.96605174

    # Perform comparison
    tec_val = ionosphere.get_ionex_value(
        TEST_TEC_FILE,
        utc_sec,
        lat,
        lon,
    )
    assert np.allclose(tec_val, value, atol=1e-05, rtol=1e-05)
