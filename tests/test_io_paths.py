import tempfile

import boto3
import numpy as np
import pytest
from moto import mock_aws

from dolphin import io

BUCKET_NAME = "fake-bucket"
KEY = "unwrapped/20231009_20231021.unw.tif"
URL = f"s3://{BUCKET_NAME}/{KEY}"

pytestmark = pytest.mark.filterwarnings(
    # Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    # Botocore: DeprecationWarning: datetime.datetime.utcnow()
    "ignore:.*datetime.*:DeprecationWarning:botocore",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture(scope="module")
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module", autouse=True)
def moto_server_handler(monkeymodule):
    from moto.server import ThreadedMotoServer

    arr = np.random.random((10, 10)).astype(np.float32)
    # Save to a tiff
    with tempfile.NamedTemporaryFile(suffix=".tif") as f:
        io.write_arr(arr=arr, output_name=f.name)
        with open(f.name, "rb") as fout:
            arr_bytes = fout.read()

    local_server = ThreadedMotoServer(ip_address="127.0.0.1", port=5000)
    local_server.start()

    monkeymodule.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeymodule.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeymodule.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeymodule.setenv("AWS_SESSION_TOKEN", "testing")
    monkeymodule.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeymodule.setenv("AWS_ENDPOINT_URL", "http://localhost:5000")

    # Post the dummy file to the local server
    with mock_aws():
        s3 = boto3.resource(
            service_name="s3",
            region_name="us-east-1",
            endpoint_url="http://localhost:5000",
        )

        b = s3.create_bucket(Bucket=BUCKET_NAME)
        b.Acl().put(ACL="public-read")
        o = s3.Object(BUCKET_NAME, KEY)
        o.put(Body=arr_bytes)

        yield  # Run any test logic

    local_server.stop()


class TestS3Path:
    @pytest.fixture
    def s3path(self):
        return io.S3Path(URL, unsigned=True)

    def test_s3path_parent(self, s3path):
        assert s3path.parent == io.S3Path("s3://fake-bucket/unwrapped/")

    def test_s3path_suffix(self, s3path):
        assert s3path.suffix == ".tif"

    def test_s3path_resolve(self, s3path):
        assert s3path.resolve() == s3path

    def test_exists(self, s3path):
        with mock_aws():
            assert s3path.exists()

    def test_from_bucket_key(self):
        bucket = BUCKET_NAME
        key = "unwrapped/20231009_20231021.unw.tif"
        s3path = io.S3Path.from_bucket_key(bucket, key)
        assert str(s3path) == str(io.S3Path(URL))

    def test_rasterio_open(self, s3path):
        import rasterio as rio

        with rio.Env(
            aws_unsigned=True,
            AWS_HTTPS="NO",
            AWS_S3_ENDPOINT="localhost:5000",
        ):
            with rio.open(URL) as src:
                driver, shape = src.driver, src.shape

            with rio.open(s3path) as src:
                assert src.driver == driver
                assert src.shape == shape

    def test_to_gdal_string(self, s3path):
        assert s3path.to_gdal() == f"/vsis3/{URL[5:]}"

    def test_io_read_gdal(self, s3path):
        from osgeo import gdal

        with gdal.config_options(
            {
                "AWS_NO_SIGN_REQUEST": "YES",
                "AWS_S3_ENDPOINT": "localhost:5000",
                "AWS_HTTPS": "NO",
            }
        ):
            assert io.get_raster_driver(s3path.to_gdal()) == "GTiff"
