from __future__ import annotations

import mmap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generator,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import h5py
import numpy as np
import rasterio as rio
from numpy.typing import ArrayLike
from osgeo import gdal

from dolphin import io, utils
from dolphin._background import _DEFAULT_TIMEOUT, BackgroundReader
from dolphin._blocks import iter_blocks
from dolphin._dates import get_dates, sort_files_by_date
from dolphin._types import Filename
from dolphin.stack import logger
from dolphin.utils import progress

__all__ = [
    "DatasetReader",
    "BinaryReader",
    "StackReader",
    "BinaryStackReader",
    "VRTStack",
]

if TYPE_CHECKING:
    from builtins import ellipsis

Index = ellipsis | slice | int


@runtime_checkable
class DatasetReader(Protocol):
    """An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be read by functions which iterate in blocks over the input data.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing.

    Note that this protol allows objects to be passed to `dask.array.from_array`
    which needs `.shape`, `.ndim`, `.dtype` and support numpy-style slicing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""  # noqa: D403

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""  # noqa: D403

    ndim: int
    """int : Number of array dimensions."""  # noqa: D403

    masked: bool = False
    """bool : If True, return a masked array with the nodata values masked out."""

    def __getitem__(self, key: tuple[Index, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


@runtime_checkable
class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""  # noqa: D403

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    @property
    def __len__(self) -> int:
        """int : Number of images in the stack."""
        return self.shape[0]


def _mask_array(arr: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """Mask an array based on a nodata value."""
    if np.isnan(nodata_value):
        return np.ma.masked_invalid(arr)
    return np.ma.masked_equal(arr, nodata_value)


@dataclass
class BinaryReader(DatasetReader):
    """A flat binary file for storing array data.

    See Also
    --------
    HDF5Dataset
    RasterReader

    Notes
    -----
    This class does not store an open file object. Instead, the file is opened on-demand
    for reading or writing and closed immediately after each read/write operation. This
    allows multiple spawned processes to write to the file in coordination (as long as a
    suitable mutex is used to guard file access.)
    """

    filepath: Path
    """pathlib.Path : The file path."""  # noqa: D403

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""  # noqa: D403

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""  # noqa: D403

    nodata_value: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels."""

    def __post_init__(self):
        self.filepath = Path(self.filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        self.dtype = np.dtype(self.dtype)

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""  # noqa: D403
        return len(self.shape)

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        with self.filepath.open("rb") as f:
            # Memory-map the entire file.
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # In order to safely close the memory-map, there can't be any dangling
                # references to it, so we return a copy (not a view) of the requested
                # data and decref the array object.
                arr = np.frombuffer(mm, dtype=self.dtype).reshape(self.shape)
                data = arr[key].copy()
                del arr
        return _mask_array(data, self.nodata_value) if self.masked else data

    def __array__(self) -> np.ndarray:
        return self[:,]

    @classmethod
    def from_gdal(
        cls, filename: Filename, band: int = 1, nodata_value: Optional[float] = None
    ) -> BinaryReader:
        """Create a BinaryReader from a GDAL-readable file.

        Parameters
        ----------
        filename : Filename
            Path to the file to read.
        band : int, optional
            Band to read from the file, by default 1
        nodata_value : float, optional
            Value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        BinaryReader
            The BinaryReader object.
        """
        with rio.open(filename) as src:
            dtype = src.dtypes[band - 1]
            shape = src.shape
            nodata = src.nodatavals[band - 1]
        return cls(
            Path(filename),
            shape=shape,
            dtype=dtype,
            nodata_value=nodata_value or nodata,
        )


@dataclass
class HDF5Reader(DatasetReader):
    """A Dataset in an HDF5 file.

    Attributes
    ----------
    filepath : pathlib.Path | str
        Location of HDF5 file.
    dset_name : str
        Path to the dataset within the file.
    chunks : tuple[int, ...], optional
        Chunk shape of the dataset, or None if file is unchunked.
    keep_open : bool, optional (default False)
        If True, keep the HDF5 file handle open for faster reading.


    See Also
    --------
    BinaryReader
    RasterReader

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5Reader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .
    """

    filepath: Path
    """pathlib.Path : The file path."""

    dset_name: str
    """str : The path to the dataset within the file."""

    nodata_value: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels.

    If None, looks for `_FillValue` or `missing_value` attributes on the dataset.
    """

    keep_open: bool = False
    """bool : If True, keep the HDF5 file handle open for faster reading."""

    def __post_init__(self):
        filepath = Path(self.filepath)

        hf = h5py.File(filepath, "r")
        dset = hf[self.dset_name]
        self.shape = dset.shape
        self.dtype = dset.dtype
        self.chunks = dset.chunks
        if self.nodata_value is None:
            self.nodata_value = dset.attrs.get("_FillValue", None)
            if self.nodata_value is None:
                self.nodata_value = dset.attrs.get("missing_value", None)
        if self.keep_open:
            self._hf = hf
            self._dset = dset
        else:
            hf.close()

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""
        return len(self.shape)

    def __array__(self) -> np.ndarray:
        return self[:,]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        if self.keep_open:
            data = self._dset[key]
        else:
            with h5py.File(self.filepath, "r") as f:
                data = f[self.dset_name][key]
        return _mask_array(data, self.nodata_value) if self.masked else data


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)


@dataclass
class RasterReader(DatasetReader):
    """A single raster band of a GDAL-compatible dataset.

    See Also
    --------
    BinaryReader
    HDF5

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `RasterReader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .
    """

    filepath: Filename
    """Filename : The file path."""

    band: int
    """int : Band index (1-based)."""

    driver: str
    """str : Raster format driver name."""

    crs: rio.crs.CRS
    """rio.crs.CRS : The dataset's coordinate reference system."""

    transform: rio.transform.Affine
    """
    rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

    This transform maps pixel row/column coordinates to coordinates in the dataset's
    coordinate reference system.
    """

    shape: tuple[int, int]
    dtype: np.dtype

    nodata_value: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels."""

    keep_open: bool = False
    """bool : If True, keep the rasterio file handle open for faster reading."""

    @classmethod
    def from_file(
        cls,
        filepath: Filename,
        band: int = 1,
        nodata_value: Optional[float] = None,
        keep_open: bool = False,
        **options,
    ) -> RasterReader:
        with rio.open(filepath, "r", **options) as src:
            shape = (src.height, src.width)
            dtype = np.dtype(src.dtypes[band - 1])
            driver = src.driver
            crs = src.crs
            nodata_value = nodata_value or src.nodatavals[band - 1]
            transform = src.transform

            return cls(
                filepath=filepath,
                band=band,
                driver=driver,
                crs=crs,
                transform=transform,
                shape=shape,
                dtype=dtype,
                nodata_value=nodata_value,
                keep_open=keep_open,
            )

    def __post_init__(self):
        if self.keep_open:
            self._src = rio.open(self.filepath, "r")

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""
        return 2

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        import rasterio.windows

        if not isinstance(key, tuple):
            raise ValueError("Index must be a tuple of slices or integers.")

        r_slice, c_slice = _ensure_slices(*key[-2:])
        window = rasterio.windows.Window.from_slices(
            r_slice,
            c_slice,
            height=self.shape[0],
            width=self.shape[1],
        )
        if self.keep_open:
            out = self._src.read(self.band, window=window)

        with rio.open(self.filepath) as src:
            out = src.read(self.band, window=window)
        out_masked = _mask_array(out, self.nodata_value) if self.masked else out
        # Note that Rasterio doesn't use the `step` of a slice, so we need to
        # manually slice the output array.
        r_step, c_step = r_slice.step or 1, c_slice.step or 1
        return out_masked[::r_step, ::c_step].squeeze()


def _read_3d(
    key: tuple[Index, ...], readers: Sequence[DatasetReader], num_threads: int = 1
):
    # Check that it's a tuple of slices
    if not isinstance(key, tuple):
        raise ValueError("Index must be a tuple of slices.")
    if len(key) not in (1, 3):
        raise ValueError("Index must be a tuple of 1 or 3 slices.")
    # If only the band is passed (e.g. stack[0]), convert to (0, :, :)
    if len(key) == 1:
        key = (key[0], slice(None), slice(None))
    # unpack the slices
    bands, rows, cols = key
    # convert the rows/cols to slices
    r_slice, c_slice = _ensure_slices(rows, cols)

    if isinstance(bands, slice):
        # convert the bands to -1-indexed list
        total_num_bands = len(readers)
        band_idxs = list(range(*bands.indices(total_num_bands)))
    elif isinstance(bands, int):
        band_idxs = [bands]
    else:
        raise ValueError("Band index must be an integer or slice.")

    # Get only the bands we need
    if num_threads == 1:
        out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(lambda i: readers[i][r_slice, c_slice], band_idxs)
        out = np.stack(list(results), axis=0)
    return np.squeeze(out)


@dataclass
class BaseStackReader(StackReader):
    """Base class for stack readers."""

    file_list: Sequence[Filename]
    readers: Sequence[DatasetReader]
    num_threads: int = 1
    nodata_value: Optional[float] = None

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return _read_3d(key, self.readers, num_threads=self.num_threads)

    @property
    def shape_2d(self):
        return self.readers[0].shape

    @property
    def shape(self):
        return (len(self.file_list), *self.shape_2d)

    @property
    def dtype(self):
        return self.readers[0].dtype

    def iter_blocks(
        self,
        overlaps: tuple[int, int] = (0, 0),
        block_shape: tuple[int, int] = (512, 512),
        skip_empty: bool = True,
        nodata_mask: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> Generator[tuple[np.ndarray, tuple[slice, slice]], None, None]:
        """Iterate over blocks of the stack.

        Loads all images for one window at a time into memory, while eagerly
        fetching the next block in the background.

        Parameters
        ----------
        overlaps : tuple[int, int], optional
            Pixels to overlap each block by (rows, cols)
            By default (0, 0)
        block_shape : tuple[int, int], optional
            2D shape of blocks to load at a time.
            Loads all dates/bands at a time with this shape.
        skip_empty : bool, optional (default True)
            Skip blocks that are entirely empty (all NaNs)
        nodata_mask : bool, optional
            Optional mask indicating nodata values. If provided, will skip
            blocks that are entirely nodata.
            1s are the nodata values, 0s are valid data.
        show_progress : bool, default=True
            If true, displays a `rich` ProgressBar.

        Yields
        ------
        tuple[np.ndarray, tuple[slice, slice]]
            Iterator of (data, (slice(row_start, row_stop), slice(col_start, col_stop))

        """
        loader = EagerLoader(
            self,
            block_shape=block_shape,
            overlaps=overlaps,
            nodata_mask=nodata_mask,
            nodata_value=self.nodata_value,
            skip_empty=skip_empty,
            show_progress=show_progress,
        )
        yield from loader.iter_blocks()


@dataclass
class BinaryStackReader(BaseStackReader):
    @classmethod
    def from_file_list(
        cls, file_list: Sequence[Filename], shape_2d: tuple[int, int], dtype: np.dtype
    ) -> BinaryStackReader:
        """Create a BinaryStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        shape_2d : tuple[int, int]
            Shape of each file.
        dtype : np.dtype
            Data type of each file.

        Returns
        -------
        BinaryStackReader
            The BinaryStackReader object.
        """
        readers = [
            BinaryReader(Path(f), shape=shape_2d, dtype=dtype) for f in file_list
        ]
        return cls(file_list=file_list, readers=readers, num_threads=1)

    @classmethod
    def from_gdal(
        cls,
        file_list: Sequence[Filename],
        band: int = 1,
        num_threads: int = 1,
        nodata_value: Optional[float] = None,
    ) -> BinaryStackReader:
        """Create a BinaryStackReader from a list of GDAL-readable files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        band : int, optional
            Band to read from the file, by default 1
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata_value : float, optional
            Manually set value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        BinaryStackReader
            The BinaryStackReader object.
        """
        readers = []
        dtypes = set()
        shapes = set()
        for f in file_list:
            with rio.open(f) as src:
                dtypes.add(src.dtypes[band - 1])
                shapes.add(src.shape)
            if len(dtypes) > 1:
                raise ValueError("All files must have the same data type.")
            if len(shapes) > 1:
                raise ValueError("All files must have the same shape.")
            readers.append(BinaryReader.from_gdal(f, band=band))
        return cls(
            file_list=file_list,
            readers=readers,
            num_threads=num_threads,
            nodata_value=nodata_value,
        )


@dataclass
class HDF5StackReader(BaseStackReader):
    """A stack of datasets in an HDF5 file.

    See Also
    --------
    BinaryStackReader
    StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5StackReader` to multiple spawned processes, it is recommended
    to set `keep_open=False`.
    """

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[Filename],
        dset_names: str | Sequence[str],
        keep_open: bool = False,
        num_threads: int = 1,
        nodata_value: Optional[float] = None,
    ) -> HDF5StackReader:
        """Create a HDF5StackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        dset_names : str | Sequence[str]
            Name of the dataset to read from each file.
            If a single string, will be used for all files.
        keep_open : bool, optional (default False)
            If True, keep the HDF5 file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata_value : float, optional
            Manually set value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        HDF5StackReader
            The HDF5StackReader object.
        """
        if isinstance(dset_names, str):
            dset_names = [dset_names] * len(file_list)

        readers = [
            HDF5Reader(
                Path(f), dset_name=dn, keep_open=keep_open, nodata_value=nodata_value
            )
            for (f, dn) in zip(file_list, dset_names)
        ]
        # Check if nodata values were found in the files
        nds = set([r.nodata_value for r in readers])
        if len(nds) == 1:
            nodata_value = nds.pop()

        return cls(
            file_list, readers, num_threads=num_threads, nodata_value=nodata_value
        )


@dataclass
class RasterStackReader(BaseStackReader):
    """A stack of datasets for any GDAL-readable rasters.

    See Also
    --------
    BinaryStackReader
    HDF5StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    """

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[Filename],
        bands: int | Sequence[int] = 1,
        keep_open: bool = False,
        num_threads: int = 1,
        nodata_value: Optional[float] = None,
    ) -> RasterStackReader:
        """Create a RasterStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        bands : int | Sequence[int]
            Band to read from each file.
            If a single int, will be used for all files.
            Default = 1.
        keep_open : bool, optional (default False)
            If True, keep the rasterio file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata_value : float, optional
            Manually set value to use for nodata pixels, by default None

        Returns
        -------
        RasterStackReader
            The RasterStackReader object.
        """
        if isinstance(bands, int):
            bands = [bands] * len(file_list)

        readers = [
            RasterReader.from_file(f, band=b, keep_open=keep_open)
            for (f, b) in zip(file_list, bands)
        ]
        # Check if nodata values were found in the files
        nds = set([r.nodata_value for r in readers])
        if len(nds) == 1:
            nodata_value = nds.pop()
        return cls(
            file_list, readers, num_threads=num_threads, nodata_value=nodata_value
        )


class VRTStack(StackReader):
    """Class for creating a virtual stack of raster files.

    Attributes
    ----------
    file_list : list[Filename]
        Paths or GDAL-compatible strings (NETCDF:...) for paths to files.
    outfile : pathlib.Path, optional (default = Path("slc_stack.vrt"))
        Name of output file to write
    dates : list[list[DateOrDatetime]]
        list, where each entry is all dates matched from the corresponding file
        in `file_list`. This is used to sort the files by date.
        Each entry is a list because some files (compressed SLCs) may have
        multiple dates in the filename.
    use_abs_path : bool, optional (default = True)
        Write the filepaths of the SLCs in the VRT as "relative=0"
    subdataset : str, optional
        Subdataset to use from the files in `file_list`, if using NetCDF files.
    sort_files : bool, optional (default = True)
        Sort the files in `file_list`. Assumes that the naming convention
        will sort the files in increasing time order.
    nodata_mask_file : pathlib.Path, optional
        Path to file containing a mask of pixels containing with nodata
        in every images. Used for skipping the loading of these pixels.
    file_date_fmt : str, optional (default = "%Y%m%d")
        Format string for parsing the dates from the filenames.
        Passed to [dolphin._dates.get_dates][].
    """

    def __init__(
        self,
        file_list: Sequence[Filename],
        outfile: Filename = "slc_stack.vrt",
        use_abs_path: bool = True,
        subdataset: Optional[str] = None,
        sort_files: bool = True,
        file_date_fmt: str = "%Y%m%d",
        write_file: bool = True,
        fail_on_overwrite: bool = False,
        skip_size_check: bool = False,
        num_threads: int = 1,
    ):
        if Path(outfile).exists() and write_file:
            if fail_on_overwrite:
                raise FileExistsError(
                    f"Output file {outfile} already exists. "
                    "Please delete or specify a different output file. "
                    "To create from an existing VRT, use the `from_vrt_file` method."
                )
            else:
                logger.info(f"Overwriting {outfile}")

        # files: list[Filename] = [Path(f) for f in file_list]
        self._use_abs_path = use_abs_path
        if use_abs_path:
            files = [utils._resolve_gdal_path(p) for p in file_list]
        else:
            files = list(file_list)
        # Extract the date/datetimes from the filenames
        dates = [get_dates(f, fmt=file_date_fmt) for f in file_list]
        if sort_files:
            files, dates = sort_files_by_date(  # type: ignore
                files, file_date_fmt=file_date_fmt
            )

        # Save the attributes
        self.file_list = files
        self.dates = dates
        self.num_threads = num_threads

        self.outfile = Path(outfile).resolve()
        # Assumes that all files use the same subdataset (if NetCDF)
        self.subdataset = subdataset

        if not skip_size_check:
            io._assert_images_same_size(self._gdal_file_strings)

        # Use the first file in the stack to get size, transform info
        ds = gdal.Open(fspath(self._gdal_file_strings[0]))
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        # Should be CFloat32
        self.gdal_dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        # Save these for setting at the end
        self.gt = ds.GetGeoTransform()
        self.proj = ds.GetProjection()
        self.srs = ds.GetSpatialRef()
        ds = None
        # Save the subset info

        self.xoff, self.yoff = 0, 0
        self.xsize_sub, self.ysize_sub = self.xsize, self.ysize

        if write_file:
            self._write()

    def _write(self):
        """Write out the VRT file pointing to the stack of SLCs, erroring if exists."""
        with open(self.outfile, "w") as fid:
            fid.write(
                f'<VRTDataset rasterXSize="{self.xsize_sub}"'
                f' rasterYSize="{self.ysize_sub}">\n'
            )

            for idx, filename in enumerate(self._gdal_file_strings, start=1):
                chunk_size = io.get_raster_chunk_size(filename)
                # chunks in a vrt have a min of 16, max of 2**14=16384
                # https://github.com/OSGeo/gdal/blob/2530defa1e0052827bc98696e7806037a6fec86e/frmts/vrt/vrtrasterband.cpp#L339
                if any([b < 16 for b in chunk_size]) or any(
                    [b > 16384 for b in chunk_size]
                ):
                    chunk_str = ""
                else:
                    chunk_str = (
                        f'blockXSize="{chunk_size[0]}" blockYSize="{chunk_size[1]}"'
                    )
                outstr = f"""  <VRTRasterBand dataType="{self.gdal_dtype}" band="{idx}" {chunk_str}>
    <SimpleSource>
      <SourceFilename>{filename}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="{self.xoff}" yOff="{self.yoff}" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
      <DstRect xOff="0" yOff="0" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
    </SimpleSource>
  </VRTRasterBand>\n"""  # noqa: E501
                fid.write(outstr)

            fid.write("</VRTDataset>")

        # Set the georeferencing metadata
        ds = gdal.Open(fspath(self.outfile), gdal.GA_Update)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        ds.SetSpatialRef(self.srs)
        ds = None

    @property
    def _gdal_file_strings(self):
        """Get the GDAL-compatible paths to write to the VRT.

        If we're not using .h5 or .nc, this will just be the file_list as is.
        """
        return [io.format_nc_filename(f, self.subdataset) for f in self.file_list]

    def read_stack(
        self,
        band: Optional[int] = None,
        subsample_factor: int = 1,
        rows: Optional[slice] = None,
        cols: Optional[slice] = None,
        masked: bool = False,
    ):
        """Read in the SLC stack."""
        return io.load_gdal(
            self.outfile,
            band=band,
            subsample_factor=subsample_factor,
            rows=rows,
            cols=cols,
            masked=masked,
        )

    def __fspath__(self):
        # Allows os.fspath() to work on the object, enabling rasterio.open()
        return fspath(self.outfile)

    @classmethod
    def from_vrt_file(cls, vrt_file, new_outfile=None, **kwargs):
        """Create a new VRTStack using an existing VRT file."""
        file_list, subdataset = _parse_vrt_file(vrt_file)
        if new_outfile is None:
            # Point to the same, if none provided
            new_outfile = vrt_file

        return cls(
            file_list,
            outfile=new_outfile,
            subdataset=subdataset,
            write_file=False,
            **kwargs,
        )

    def iter_blocks(
        self,
        overlaps: tuple[int, int] = (0, 0),
        block_shape: tuple[int, int] = (512, 512),
        skip_empty: bool = True,
        nodata_mask: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> Generator[tuple[np.ndarray, tuple[slice, slice]], None, None]:
        """Iterate over blocks of the stack.

        Loads all images for one window at a time into memory, while eagerly
        fetching the next block in the background.

        Parameters
        ----------
        overlaps : tuple[int, int], optional
            Pixels to overlap each block by (rows, cols)
            By default (0, 0)
        block_shape : tuple[int, int], optional
            2D shape of blocks to load at a time.
            Loads all dates/bands at a time with this shape.
        skip_empty : bool, optional (default True)
            Skip blocks that are entirely empty (all NaNs)
        nodata_mask : bool, optional
            Optional mask indicating nodata values. If provided, will skip
            blocks that are entirely nodata.
            1s are the nodata values, 0s are valid data.
        show_progress : bool, default=True
            If true, displays a `rich` ProgressBar.

        Yields
        ------
        tuple[np.ndarray, tuple[slice, slice]]
            Iterator of (data, (slice(row_start, row_stop), slice(col_start, col_stop))

        """
        loader = EagerLoader(
            self,
            block_shape=block_shape,
            overlaps=overlaps,
            nodata_mask=nodata_mask,
            skip_empty=skip_empty,
            show_progress=show_progress,
        )
        yield from loader.iter_blocks()

    @property
    def shape(self):
        """Get the 3D shape of the stack."""
        xsize, ysize = io.get_raster_xysize(self._gdal_file_strings[0])
        return (len(self.file_list), ysize, xsize)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        outname = fspath(self.outfile) if self.outfile else "(not written)"
        return f"VRTStack({len(self.file_list)} bands, outfile={outname})"

    def __eq__(self, other):
        if not isinstance(other, VRTStack):
            return False
        return (
            self._gdal_file_strings == other._gdal_file_strings
            and self.outfile == other.outfile
        )

    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            return self.read_stack(band=index + 1)

        # TODO: raise an error if they try to skip like [::2, ::2]
        # or pass it to read_stack... but I dont think I need to support it.
        n, rows, cols = index
        if isinstance(rows, int):
            rows = slice(rows, rows + 1)
        if isinstance(cols, int):
            cols = slice(cols, cols + 1)
        if isinstance(n, int):
            if n < 0:
                n = len(self) + n
            return self.read_stack(band=n + 1, rows=rows, cols=cols)

        bands = list(range(1, 1 + len(self)))[n]
        if len(bands) == len(self):
            # This will use gdal's ds.ReadAsRaster, no iteration needed
            data = self.read_stack(band=None, rows=rows, cols=cols)
        else:
            # Get only the bands we need
            if self.num_threads == 1:
                # out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
                data = np.stack(
                    [self.read_stack(band=i, rows=rows, cols=cols) for i in bands],
                    axis=0,
                )
            else:
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    results = executor.map(
                        lambda i: self.read_stack(band=i, rows=rows, cols=cols), bands
                    )
                data = np.stack(list(results), axis=0)

        return data.squeeze()

    @property
    def dtype(self):
        return io.get_raster_dtype(self._gdal_file_strings[0])


def _parse_vrt_file(vrt_file):
    """Extract the filenames, and possible subdatasets, from a .vrt file.

    Assumes, if using HDFS/NetCDF files, that the subdataset is the same.

    Note that we are parsing the XML, not using `GetFilelist`, because the
    function does not seem to work when using HDF5 files. E.g.

        <SourceFilename ="1">NETCDF:20220111.nc:SLC/VV</SourceFilename>

    This would not get added to the result of `GetFilelist`

    Parameters
    ----------
    vrt_file : Filename
        Path to the VRT file to read.

    Returns
    -------
    filepaths
        List of filepaths to the SLCs
    sds
        Subdataset name, if using NetCDF/HDF5 files
    """
    file_strings = []
    with open(vrt_file) as f:
        for line in f:
            if "<SourceFilename" not in line:
                continue
            # Get the middle part of < >filename</ >
            fn = line.split(">")[1].strip().split("<")[0]
            file_strings.append(fn)

    sds = ""
    filepaths = []
    for name in file_strings:
        if name.upper().startswith("HDF5:") or name.upper().startswith("NETCDF:"):
            prefix, filepath, subdataset = name.split(":")
            # Clean up subdataset
            sds = subdataset.replace('"', "").replace("'", "").lstrip("/")
            # Remove quoting if it was present
            filepaths.append(filepath.replace('"', "").replace("'", ""))
        else:
            filepaths.append(name)

    return filepaths, sds


class EagerLoader(BackgroundReader):
    """Class to pre-fetch data chunks in a background thread."""

    def __init__(
        self,
        reader: DatasetReader,
        block_shape: tuple[int, int],
        overlaps: tuple[int, int] = (0, 0),
        skip_empty: bool = True,
        nodata_value: Optional[float] = None,
        nodata_mask: Optional[ArrayLike] = None,
        queue_size: int = 1,
        timeout: float = _DEFAULT_TIMEOUT,
        show_progress: bool = True,
    ):
        super().__init__(nq=queue_size, timeout=timeout, name="EagerLoader")
        self.reader = reader
        # Set up the generator of ((row_start, row_end), (col_start, col_end))
        # convert the slice generator to a list so we have the size
        nrows, ncols = self.reader.shape[-2:]
        self.slices = list(
            iter_blocks(
                arr_shape=(nrows, ncols),
                block_shape=block_shape,
                overlaps=overlaps,
            )
        )
        self._queue_size = queue_size
        self._skip_empty = skip_empty
        self._nodata_mask = nodata_mask
        self._block_shape = block_shape
        self._nodata = nodata_value
        self._show_progress = show_progress
        if self._nodata is None:
            self._nodata = np.nan

    def read(self, rows: slice, cols: slice) -> tuple[np.ndarray, tuple[slice, slice]]:
        logger.debug(f"EagerLoader reading {rows}, {cols}")
        cur_block = self.reader[..., rows, cols]
        return cur_block, (rows, cols)

    def iter_blocks(
        self,
    ) -> Generator[tuple[np.ndarray, tuple[slice, slice]], None, None]:
        # Queue up all slices to the work queue
        queued_slices = []
        for rows, cols in self.slices:
            # Skip queueing a read if all nodata
            if self._skip_empty and self._nodata_mask is not None:
                logger.debug("Checking nodata mask")
                if self._nodata_mask[rows, cols].all():
                    logger.debug("Skipping!")
                    continue
            self.queue_read(rows, cols)
            queued_slices.append((rows, cols))

        s_iter = range(len(queued_slices))
        desc = f"Processing {self._block_shape} sized blocks..."
        with progress(dummy=not self._show_progress) as p:
            for _ in p.track(s_iter, description=desc):
                cur_block, (rows, cols) = self.get_data()
                logger.debug(f"got data for {rows, cols}: {cur_block.shape}")

                # Otherwise look at the actual block we loaded
                if np.isnan(self._nodata):
                    block_nodata = np.isnan(cur_block)
                else:
                    block_nodata = cur_block == self._nodata
                if np.all(block_nodata):
                    logger.debug("Skipping block since it was all nodata")
                    continue
                yield cur_block, (rows, cols)

        self.notify_finished()
