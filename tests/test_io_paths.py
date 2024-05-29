import pytest

from dolphin import io

URL = "s3://testzarr-insar-plotting/unwrapped/20231009_20231021.unw.conncomp.tif"


class TestS3Path:
    @pytest.fixture
    def s3path(self):
        return io.S3Path(URL, unsigned=True)

    @pytest.mark.vcr
    def test_exists(self, s3path):
        assert s3path.exists()

    def test_s3path_parent(self, s3path):
        assert s3path.parent == io.S3Path("s3://testzarr-insar-plotting/unwrapped/")

    def test_s3path_suffix(self, s3path):
        assert s3path.suffix == ".tif"

    def test_from_bucket_key(self):
        bucket = "testzarr-insar-plotting"
        key = "unwrapped/20231009_20231021.unw.conncomp.tif"
        s3path = io.S3Path.from_bucket_key(bucket, key)
        assert str(s3path) == str(io.S3Path(URL))

    @pytest.mark.vcr
    def test_rasterio_open(self, s3path):
        import rasterio as rio

        with rio.Env(AWS_NO_SIGN_REQUEST="YES"):
            with rio.open(URL) as src:
                driver, shape = src.driver, src.shape

            with rio.open(s3path) as src:
                assert src.driver == driver
                assert src.shape == shape

    def test_to_gdal_string(self, s3path):
        assert s3path.to_gdal() == f"/vsis3/{URL[5:]}"

    @pytest.mark.vcr
    def test_io_read_gdal(self, s3path):
        from osgeo import gdal

        with gdal.config_option("AWS_NO_SIGN_REQUEST", "YES"):
            assert io.get_raster_driver(s3path.to_gdal()) == "GTiff"
