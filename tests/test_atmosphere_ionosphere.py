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
    #     (array([   0.,  120.,  240.,  360.,  480.,  600.,  720.,  840.,  960.,
    #         1080., 1200., 1320., 1440.]),
    #  array([ 87.5,  85. ,  82.5,  80. ,  77.5,  75. ,  72.5,  70. ,  67.5,
    #          65. ,  62.5,  60. ,  57.5,  55. ,  52.5,  50. ,  47.5,  45. ,
    #          42.5,  40. ,  37.5,  35. ,  32.5,  30. ,  27.5,  25. ,  22.5,
    #          20. ,  17.5,  15. ,  12.5,  10. ,   7.5,   5. ,   2.5,   0. ,
    #          -2.5,  -5. ,  -7.5, -10. , -12.5, -15. , -17.5, -20. , -22.5,
    #         -25. , -27.5, -30. , -32.5, -35. , -37.5, -40. , -42.5, -45. ,
    #         -47.5, -50. , -52.5, -55. , -57.5, -60. , -62.5, -65. , -67.5,
    #         -70. , -72.5, -75. , -77.5, -80. , -82.5, -85. , -87.5]),
    #  array([-180., -175., -170., -165., -160., -155., -150., -145., -140.,
    #         -135., -130., -125., -120., -115., -110., -105., -100.,  -95.,
    #          -90.,  -85.,  -80.,  -75.,  -70.,  -65.,  -60.,  -55.,  -50.,
    #          -45.,  -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,
    #            0.,    5.,   10.,   15.,   20.,   25.,   30.,   35.,   40.,
    #           45.,   50.,   55.,   60.,   65.,   70.,   75.,   80.,   85.,
    #           90.,   95.,  100.,  105.,  110.,  115.,  120.,  125.,  130.,
    #          135.,  140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.,
    #          180.]),

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
