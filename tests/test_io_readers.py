from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import rasterio as rio
from osgeo import gdal
from rasterio.errors import NotGeoreferencedWarning

from dolphin.io._readers import (
    BinaryReader,
    BinaryStackReader,
    EagerLoader,
    HDF5Reader,
    HDF5StackReader,
    RasterReader,
    RasterStackReader,
    VRTStack,
    _parse_vrt_file,
)
from dolphin.utils import _get_path_from_gdal_str

# Note: uses the fixtures from conftest.py

# Get combinations of slices
# TODO: if we want to test `1`, and have it work the same as
# numpy slicing which drops the dimension, we'll need to change this
slices_to_test = [slice(None), slice(1), slice(0, 10, 2)]


@pytest.fixture(scope="module")
def binary_file_list(tmp_path_factory, slc_stack):
    """Flat binary files in the ENVI format."""

    shape = slc_stack[0].shape
    dtype = slc_stack.dtype
    tmp_path = tmp_path_factory.mktemp("data")

    # Create a stack of binary files
    files = []
    for i, slc in enumerate(slc_stack):
        f = tmp_path / f"test_{i}.bin"
        # Ignore warning
        with (
            pytest.warns(NotGeoreferencedWarning),
            rio.open(
                f,
                "w",
                driver="ENVI",
                width=shape[1],
                height=shape[0],
                count=1,
                dtype=dtype,
            ) as dst,
        ):
            dst.write(slc, 1)
        files.append(f)

    return files


@pytest.fixture()
def binary_reader(slc_stack, binary_file_list):
    f = BinaryReader(
        binary_file_list[0], shape=slc_stack[0].shape, dtype=slc_stack.dtype
    )
    assert f.shape == slc_stack[0].shape
    assert f.dtype == slc_stack[0].dtype
    return f


class TestBinary:
    def test_binary_file_read(self, binary_reader, slc_stack):
        npt.assert_array_almost_equal(binary_reader[()], slc_stack[0])
        # Check the reading of a subset
        npt.assert_array_almost_equal(
            binary_reader[0:10, 0:10], slc_stack[0][0:10, 0:10]
        )

    @pytest.fixture(scope="module")
    def binary_stack(self, slc_stack, binary_file_list):
        s = BinaryStackReader.from_file_list(
            binary_file_list, shape_2d=slc_stack[0].shape, dtype=slc_stack.dtype
        )
        assert s.shape == slc_stack.shape
        assert len(s) == len(slc_stack) == len(binary_file_list)
        assert s.ndim == 3
        assert s.dtype == slc_stack.dtype
        return s

    @pytest.mark.parametrize("dslice", slices_to_test)
    @pytest.mark.parametrize("rslice", slices_to_test)
    @pytest.mark.parametrize("cslice", slices_to_test)
    def test_binary_stack_read_slices(
        self, binary_stack, slc_stack, dslice, rslice, cslice
    ):
        s = binary_stack[dslice, rslice, cslice]
        expected = slc_stack[dslice, rslice, cslice]
        assert s.shape == expected.shape
        npt.assert_array_almost_equal(s, expected)


# #### HDF5 Tests ####


@pytest.fixture(scope="module")
def hdf5_file_list(tmp_path_factory, slc_stack):
    """Flat binary files in the ENVI format."""
    import h5py

    tmp_path = tmp_path_factory.mktemp("data")

    # Create a stack of binary files
    files = []
    for i, slc in enumerate(slc_stack):
        f = tmp_path / f"test_{i}.h5"
        with h5py.File(f, "w") as dst:
            dst.create_dataset("data", data=slc)
        files.append(f)

    return files


@pytest.fixture()
def hdf5_reader(hdf5_file_list, slc_stack):
    r = HDF5Reader(hdf5_file_list[0], dset_name="data", keep_open=True)
    assert r.shape == slc_stack[0].shape
    assert r.dtype == slc_stack[0].dtype
    return r


class TestHDF5:
    def test_hdf5_reader_read(self, hdf5_reader, slc_stack):
        npt.assert_array_almost_equal(hdf5_reader[()], slc_stack[0])
        # Check the reading of a subset
        npt.assert_array_almost_equal(hdf5_reader[0:10, 0:10], slc_stack[0][0:10, 0:10])

    @pytest.mark.parametrize("keep_open", [True, False])
    def hdf5_stack(self, hdf5_file_list, slc_stack, keep_open):
        s = HDF5StackReader.from_file_list(
            hdf5_file_list, dset_names="data", keep_open=keep_open
        )
        assert s.shape == slc_stack.shape
        assert len(s) == len(slc_stack) == len(hdf5_file_list)
        assert s.ndim == 3
        assert s.dtype == slc_stack.dtype
        return s

    @pytest.mark.parametrize("dslice", slices_to_test)
    @pytest.mark.parametrize("rslice", slices_to_test)
    @pytest.mark.parametrize("cslice", slices_to_test)
    @pytest.mark.parametrize("keep_open", [True, False])
    def test_hdf5_stack_read_slices(
        self, hdf5_file_list, slc_stack, keep_open, dslice, rslice, cslice
    ):
        reader = HDF5StackReader.from_file_list(
            hdf5_file_list, dset_names="data", keep_open=keep_open
        )
        s = reader[dslice, rslice, cslice]
        expected = slc_stack[dslice, rslice, cslice]
        assert s.shape == expected.shape
        npt.assert_array_almost_equal(s, expected)


# #### RasterReader Tests ####
@pytest.fixture()
def raster_reader(slc_file_list, slc_stack):
    # ignore georeferencing warnings
    with pytest.warns(NotGeoreferencedWarning):
        r = RasterReader.from_file(slc_file_list[0])
    assert r.shape == slc_stack[0].shape
    assert r.dtype == slc_stack[0].dtype
    assert r.ndim == 2
    assert r.dtype == np.complex64
    return r


@pytest.fixture(scope="module")
def image_513(tmp_path_factory):
    from dolphin import io

    tmp_path = tmp_path_factory.mktemp("data")
    shape = (65, 65)
    d = tmp_path / "shape513"
    d.mkdir()

    name = d / "image.tif"

    data = np.random.rand(*shape)
    io.write_arr(arr=data, output_name=name)
    return name


def test_single_pixel_leftover_squeeze(image_513):
    # ignore georeferencing warnings
    with pytest.warns(NotGeoreferencedWarning):
        r = RasterReader.from_file(image_513, keepdims=True)
        r2 = RasterReader.from_file(image_513, keepdims=False)

    d = r[slice(64, 65), slice(0, 25)]
    assert d.shape == (1, 25)
    assert d.ndim == 2
    assert r[slice(64, 65), slice(64, 65)].ndim == 2

    d = r2[slice(64, 65), slice(0, 25)]
    assert d.ndim == 1
    assert d.shape == (25,)
    assert r2[slice(64, 65), slice(64, 65)].ndim == 0


class TestRasterStack:
    @pytest.mark.parametrize("keep_open", [True, False])
    def test_raster_stack_reader(
        self,
        slc_file_list,
        slc_stack,
        keep_open,
    ):
        with pytest.warns(NotGeoreferencedWarning):
            reader = RasterStackReader.from_file_list(
                slc_file_list, keep_open=keep_open
            )
            assert reader.ndim == 3
            assert reader.shape == slc_stack.shape
            assert reader.dtype == slc_stack.dtype
            assert len(reader) == len(slc_stack) == len(slc_file_list)

    @pytest.mark.parametrize("dslice", slices_to_test)
    @pytest.mark.parametrize("rslice", slices_to_test)
    @pytest.mark.parametrize("cslice", slices_to_test)
    @pytest.mark.parametrize("keep_open", [True, False])
    def test_raster_stack_read_slices(
        self, slc_file_list, slc_stack, keep_open, dslice, rslice, cslice
    ):
        with pytest.warns(NotGeoreferencedWarning):
            reader = RasterStackReader.from_file_list(
                slc_file_list, keep_open=keep_open
            )
            s = reader[dslice, rslice, cslice]
        expected = slc_stack[dslice, rslice, cslice]
        assert s.shape == expected.shape
        npt.assert_array_almost_equal(s, expected)

    def test_single_pixel_leftover_squeeze(self, image_513):
        with pytest.warns(NotGeoreferencedWarning):
            reader = RasterStackReader.from_file_list(
                [image_513, image_513], keepdims=True
            )
            reader_sq = RasterStackReader.from_file_list(
                [image_513, image_513], keepdims=False
            )
        d = reader[:, slice(64, 65), slice(64, 65)]
        assert d.shape == (2, 1, 1)

        d2 = reader_sq[:, slice(64, 65), slice(64, 65)]
        assert d2.shape == (2,)


@pytest.mark.parametrize("rows", slices_to_test)
@pytest.mark.parametrize("cols", slices_to_test)
def test_ellipsis_reads(binary_reader, hdf5_reader, raster_reader, rows, cols):
    # Test that the ellipsis works
    npt.assert_array_equal(binary_reader[...], binary_reader[()])
    npt.assert_array_equal(hdf5_reader[...], hdf5_reader[()])
    npt.assert_array_equal(raster_reader[...], raster_reader[()])
    # Test we can still do rows/cols with a leading ellipsis
    npt.assert_array_equal(binary_reader[..., rows, cols], binary_reader[rows, cols])
    npt.assert_array_equal(hdf5_reader[..., rows, cols], hdf5_reader[rows, cols])
    npt.assert_array_equal(raster_reader[..., rows, cols], raster_reader[rows, cols])


# #### VRT Tests ####


@pytest.fixture()
def vrt_stack(tmp_path, slc_stack, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    assert len(s) == len(slc_stack) == len(slc_file_list)
    return s


@pytest.fixture()
def vrt_stack_nc(tmp_path, slc_stack, slc_file_list_nc):
    vrt_file = tmp_path / "test_nc.vrt"
    s = VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture()
def vrt_stack_nc_subdataset(tmp_path, slc_stack, slc_file_list_nc_with_sds):
    vrt_file = tmp_path / "test_nc.vrt"
    files_only = [_get_path_from_gdal_str(f) for f in slc_file_list_nc_with_sds]
    s = VRTStack(files_only, outfile=vrt_file, subdataset="data/VV")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture()
def vrt_stack_nc_wgs84(tmp_path, slc_stack, slc_file_list_nc_wgs84):
    # Check an alternative projection system
    vrt_file = tmp_path / "test_nc_wgs84.vrt"
    s = VRTStack(slc_file_list_nc_wgs84, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    return s


def test_create(vrt_stack, vrt_stack_nc):
    for _v in [vrt_stack, vrt_stack_nc]:
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

    data = vrt_stack.read_stack()
    npt.assert_array_almost_equal(data, slc_stack)
    assert data.ndim == vrt_stack.ndim
    assert data.shape == vrt_stack.shape


def test_read_stack_nc(vrt_stack_nc, slc_stack):
    ds = gdal.Open(str(vrt_stack_nc.outfile))
    loaded = ds.ReadAsArray()
    npt.assert_array_almost_equal(loaded, slc_stack)
    npt.assert_array_almost_equal(vrt_stack_nc.read_stack(), slc_stack)


def test_read_stack_1_file(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test_1file.vrt"
    v = VRTStack(slc_file_list[:1], outfile=vrt_file)
    data = v.read_stack()
    assert data.ndim == v.ndim
    assert data.shape == v.shape

    data = v[:, :, :]
    assert data.ndim == v.ndim
    assert data.shape == v.shape


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
        VRTStack([*slc_file_list, raster_10_by_20], outfile="other.vrt")


def test_iter_blocks(vrt_stack):
    loader = EagerLoader(reader=vrt_stack, block_shape=(5, 5))
    blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
    # (5, 10) total shape, breaks into 5x5 blocks
    assert len(blocks) == 2
    for b in blocks:
        assert b.shape == (len(vrt_stack), 5, 5)

    loader = EagerLoader(reader=vrt_stack, block_shape=(5, 2))
    blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
    assert len(blocks) == 5
    for b in blocks:
        assert b.shape == (len(vrt_stack), 5, 2)


def test_tiled_iter_blocks(tmp_path, tiled_file_list):
    outfile = tmp_path / "stack.vrt"
    vrt_stack = VRTStack(tiled_file_list, outfile=outfile)
    loader = EagerLoader(reader=vrt_stack, block_shape=(32, 32))
    blocks, slices = zip(*list(loader.iter_blocks()), strict=False)
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

    loader = EagerLoader(reader=vrt_stack, block_shape=(50, 100))
    blocks, slices = zip(*list(loader.iter_blocks()), strict=False)
    assert len(blocks) == len(slices) == 4


@pytest.fixture()
def test_vrt():
    return """<VRTDataset rasterXSize="128" rasterYSize="128">
  <SRS dataAxisToSRSAxisMapping="1,2">PROJCS["WGS 84 / UTM zone 15N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-93],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32615"]]</SRS>
  <GeoTransform> -6.4500000000000000e+01,  1.0000000000000000e+00,  0.0000000000000000e+00,  6.6500000000000000e+01,  0.0000000000000000e+00, -1.0000000000000000e+00</GeoTransform>
  <VRTRasterBand dataType="CFloat32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">compressed_20220101_20220101_20220104.tif</SourceFilename>
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
"""  # noqa: E501


def test_parse_vrt(tmp_path, test_vrt):
    with open(tmp_path / "t.vrt", "w") as f:
        f.write(test_vrt)

    filepaths, sds = _parse_vrt_file(tmp_path / "t.vrt")
    assert filepaths == [
        "compressed_20220101_20220101_20220104.tif",
        "t087_185684_iw2_20220102.h5",
        "t087_185684_iw2_20220103.h5",
        "t087_185684_iw2_20220104.h5",
    ]
    assert sds == "data/VV"


class TestEagerLoader:
    def test_iter_blocks(self, tiled_raster_100_by_200):
        # Try the whole raster
        bs = (100, 200)
        loader = EagerLoader(
            reader=RasterReader.from_file(tiled_raster_100_by_200), block_shape=bs
        )
        # `list` should try to load all at once`
        block_slice_tuples = list(loader.iter_blocks())
        assert not loader._thread.is_alive()
        assert len(block_slice_tuples) == 1
        blocks, slices = zip(*list(block_slice_tuples), strict=False)
        assert blocks[0].shape == (100, 200)
        rows, cols = slices[0]
        assert rows == slice(0, 100)
        assert cols == slice(0, 200)

        # now one block at a time
        bs = (32, 32)
        reader = RasterReader.from_file(tiled_raster_100_by_200)
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, slices = zip(*list(loader.iter_blocks()), strict=False)

        row_blocks = 100 // 32 + 1
        col_blocks = 200 // 32 + 1
        expected_num_blocks = row_blocks * col_blocks
        assert len(blocks) == expected_num_blocks
        assert blocks[0].shape == (32, 32)
        # at the ends, the block_slice_tuples are smaller
        assert blocks[6].shape == (32, 8)
        assert blocks[-1].shape == (4, 8)

    def test_iter_blocks_rowcols(self, tiled_raster_100_by_200):
        # Block size that is a multiple of the raster size
        reader = RasterReader.from_file(tiled_raster_100_by_200)
        loader = EagerLoader(reader=reader, block_shape=(10, 20))
        blocks, slices = zip(*list(loader.iter_blocks()), strict=False)

        assert blocks[0].shape == (10, 20)
        for rs, cs in slices:
            assert rs.stop - rs.start == 10
            assert cs.stop - cs.start == 20
        loader.notify_finished()

        # Non-multiple block size
        reader = RasterReader.from_file(tiled_raster_100_by_200)
        loader = EagerLoader(reader=reader, block_shape=(32, 32))
        blocks, slices = zip(*list(loader.iter_blocks()), strict=False)
        assert blocks[0].shape == (32, 32)
        for b, (rs, cs) in zip(blocks, slices, strict=False):
            assert b.shape == (rs.stop - rs.start, cs.stop - cs.start)
        loader.notify_finished()

    def test_iter_nodata(
        self,
        raster_with_nan,
        raster_with_nan_block,
        raster_with_zero_block,
        tiled_raster_100_by_200,
    ):
        # load one block at a time
        bs = (100, 200)
        reader = RasterReader.from_file(tiled_raster_100_by_200)
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
        loader.notify_finished()

        bs = (32, 32)
        row_blocks = 100 // 32 + 1
        col_blocks = 200 // 32 + 1
        expected_num_blocks = row_blocks * col_blocks
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
        assert len(blocks) == expected_num_blocks
        assert blocks[0].shape == bs

        # One nan should be fine, will get loaded
        reader = RasterReader.from_file(raster_with_nan)
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
        loader.notify_finished()
        assert len(blocks) == expected_num_blocks

        # Now check entire block for a skipped block
        reader = RasterReader.from_file(raster_with_nan_block)
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
        loader.notify_finished()
        assert len(blocks) == expected_num_blocks - 1

        # Now check entire block for a skipped block
        reader = RasterReader.from_file(raster_with_zero_block)
        loader = EagerLoader(reader=reader, block_shape=bs)
        blocks, _slices = zip(*list(loader.iter_blocks()), strict=False)
        loader.notify_finished()
        assert len(blocks) == expected_num_blocks - 1
