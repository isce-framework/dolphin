from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

from dolphin._readers import (
    BinaryFile,
    VRTStack,
    _parse_vrt_file,
)
from dolphin.utils import _get_path_from_gdal_str

# Note: uses the fixtures from conftest.py


@pytest.fixture(scope="module")
def binary_file_list(tmp_path_factory, slc_stack):
    """Flat binary files in the ENVI format."""
    import rasterio as rio
    from rasterio.errors import NotGeoreferencedWarning

    shape = slc_stack[0].shape
    dtype = slc_stack.dtype
    tmp_path = tmp_path_factory.mktemp("data")

    # Create a stack of binary files
    files = []
    for i, slc in enumerate(slc_stack):
        f = tmp_path / f"test_{i}.bin"
        # Ignore warning
        with pytest.warns(NotGeoreferencedWarning):
            with rio.open(
                f,
                "w",
                driver="ENVI",
                width=shape[1],
                height=shape[0],
                count=1,
                dtype=dtype,
            ) as dst:
                dst.write(slc, 1)
        files.append(f)

    return files


@pytest.fixture
def binary_file(slc_stack, binary_file_list):
    f = BinaryFile(binary_file_list[0], shape=slc_stack[0].shape, dtype=slc_stack.dtype)
    assert f.shape == slc_stack[0].shape
    assert f.dtype == slc_stack[0].dtype
    return f


def test_binary_file_read(binary_file, slc_stack):
    npt.assert_array_almost_equal(binary_file[()], slc_stack[0])
    # Check the reading of a subset
    npt.assert_array_almost_equal(binary_file[0:10, 0:10], slc_stack[0][0:10, 0:10])


# #### VRT Tests ####


@pytest.fixture
def vrt_stack(tmp_path, slc_stack, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    assert len(s) == len(slc_stack) == len(slc_file_list)
    return s


@pytest.fixture
def vrt_stack_nc(tmp_path, slc_stack, slc_file_list_nc):
    vrt_file = tmp_path / "test_nc.vrt"
    s = VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture
def vrt_stack_nc_subdataset(tmp_path, slc_stack, slc_file_list_nc_with_sds):
    vrt_file = tmp_path / "test_nc.vrt"
    files_only = [_get_path_from_gdal_str(f) for f in slc_file_list_nc_with_sds]
    s = VRTStack(files_only, outfile=vrt_file, subdataset="data/VV")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture
def vrt_stack_nc_wgs84(tmp_path, slc_stack, slc_file_list_nc_wgs84):
    # Check an alternative projection system
    vrt_file = tmp_path / "test_nc_wgs84.vrt"
    s = VRTStack(slc_file_list_nc_wgs84, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    return s


def test_create(vrt_stack, vrt_stack_nc):
    for v in [vrt_stack, vrt_stack_nc]:
        vrt_file = vrt_stack.outfile
        assert vrt_file.exists()
        assert vrt_file.stat().st_size > 0

        # Check that the VRT is valid
        ds = gdal.Open(str(vrt_file))
        assert ds is not None
        ds = None


def test_create_over_existing(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    VRTStack(slc_file_list, outfile=vrt_file)
    VRTStack(slc_file_list, outfile=vrt_file, fail_on_overwrite=False)
    with pytest.raises(FileExistsError):
        VRTStack(slc_file_list, outfile=vrt_file, fail_on_overwrite=True)


def test_from_vrt_file(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)
    s2 = VRTStack.from_vrt_file(vrt_file)
    assert s == s2


def test_read_stack(vrt_stack, slc_stack):
    ds = gdal.Open(str(vrt_stack.outfile))
    loaded = ds.ReadAsArray()
    npt.assert_array_almost_equal(loaded, slc_stack)
    npt.assert_array_almost_equal(vrt_stack.read_stack(), slc_stack)


def test_read_stack_nc(vrt_stack_nc, slc_stack):
    ds = gdal.Open(str(vrt_stack_nc.outfile))
    loaded = ds.ReadAsArray()
    npt.assert_array_almost_equal(loaded, slc_stack)
    npt.assert_array_almost_equal(vrt_stack_nc.read_stack(), slc_stack)


def test_sort_order(tmp_path, slc_file_list):
    random_order = [Path(f) for f in np.random.permutation(slc_file_list)]
    # Make sure the files are sorted by date
    vrt_stack = VRTStack(random_order, outfile=tmp_path / "test.vrt")
    assert vrt_stack.file_list == [Path(f) for f in slc_file_list]

    vrt_stack2 = VRTStack(
        random_order, sort_files=False, outfile=tmp_path / "test2.vrt"
    )
    assert vrt_stack2.file_list == random_order


def test_dates(vrt_stack, vrt_stack_nc_subdataset):
    dates = vrt_stack.dates
    assert len(dates) == len(vrt_stack)
    d0 = 20220101
    for d in dates:
        assert d[0].strftime("%Y%m%d") == str(d0)
        d0 += 1

    d0 = 20220101
    for d in vrt_stack_nc_subdataset.dates:
        assert d[0].strftime("%Y%m%d") == str(d0)
        d0 += 1


def test_bad_sizes(slc_file_list, raster_10_by_20):
    from dolphin.io import get_raster_xysize

    # Make sure the files are the same size
    assert get_raster_xysize(slc_file_list[0]) == get_raster_xysize(slc_file_list[1])
    assert get_raster_xysize(slc_file_list[0]) != get_raster_xysize(raster_10_by_20)
    with pytest.raises(ValueError):
        VRTStack(slc_file_list + [raster_10_by_20], outfile="other.vrt")


def test_iter_blocks(vrt_stack):
    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(5, 5))))
    # (5, 10) total shape, breaks into 5x5 blocks
    assert len(blocks) == 2
    for b in blocks:
        assert b.shape == (len(vrt_stack), 5, 5)

    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(1, 2))))
    assert len(blocks) == 25
    for b in blocks:
        assert b.shape == (len(vrt_stack), 1, 2)


def test_tiled_iter_blocks(tmp_path, tiled_file_list):
    outfile = tmp_path / "stack.vrt"
    vrt_stack = VRTStack(tiled_file_list, outfile=outfile)
    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(32, 32))))
    # (100, 200) total shape, breaks into 32x32 blocks
    assert len(blocks) == len(slices) == 28
    for i, b in enumerate(blocks, start=1):
        # Account for the smaller block sizes at the ends
        if i % 7 == 0:
            # last col
            if i > 21:  # Last row
                assert b.shape == (len(vrt_stack), 4, 8)
            else:
                assert b.shape == (len(vrt_stack), 32, 8)

    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(50, 100))))
    assert len(blocks) == len(slices) == 4


@pytest.fixture
def test_vrt():
    return """<VRTDataset rasterXSize="128" rasterYSize="128">
  <SRS dataAxisToSRSAxisMapping="1,2">PROJCS["WGS 84 / UTM zone 15N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-93],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32615"]]</SRS>
  <GeoTransform> -6.4500000000000000e+01,  1.0000000000000000e+00,  0.0000000000000000e+00,  6.6500000000000000e+01,  0.0000000000000000e+00, -1.0000000000000000e+00</GeoTransform>
  <VRTRasterBand dataType="CFloat32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">compressed_20220101_20220104.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="128" ySize="128" />
      <DstRect xOff="0" yOff="0" xSize="128" ySize="128" />
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="CFloat32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">NETCDF:"t087_185684_iw2_20220102.h5":"//data/VV"</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="128" ySize="128" />
      <DstRect xOff="0" yOff="0" xSize="128" ySize="128" />
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="CFloat32" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">NETCDF:"t087_185684_iw2_20220103.h5":"//data/VV"</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="128" ySize="128" />
      <DstRect xOff="0" yOff="0" xSize="128" ySize="128" />
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="CFloat32" band="3">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">NETCDF:"t087_185684_iw2_20220104.h5":"//data/VV"</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="128" ySize="128" />
      <DstRect xOff="0" yOff="0" xSize="128" ySize="128" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""


def test_parse_vrt(tmp_path, test_vrt):
    with open(tmp_path / "t.vrt", "w") as f:
        f.write(test_vrt)

    filepaths, sds = _parse_vrt_file(tmp_path / "t.vrt")
    assert filepaths == [
        "compressed_20220101_20220104.tif",
        "t087_185684_iw2_20220102.h5",
        "t087_185684_iw2_20220103.h5",
        "t087_185684_iw2_20220104.h5",
    ]
    assert sds == "data/VV"
