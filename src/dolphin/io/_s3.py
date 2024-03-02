import boto3


def list_bucket_boto3(
    bucket: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    full_bucket_glob: str | None = None,
    aws_profile: str | None = None,
) -> list[str]:
    """List items in a bucket using boto3.

    Parameters
    ----------
    bucket : str, optional
        Name of the bucket.
    prefix : str, optional
        Prefix to filter by.
    suffix : str, optional
        Suffix to filter by.
    full_bucket_glob : str, optional
        Full glob to filter by.
    aws_profile : str, optional
        AWS profile to use.

    Returns
    -------
    List[str]
        List of items in the bucket.

    """
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3 = session.client("s3")
    out: list[str] = []

    # Determine the prefix for listing objects
    if full_bucket_glob:
        # If full_bucket_glob is provided, extract bucket and prefix from it
        if full_bucket_glob.startswith("s3://"):
            full_bucket_glob = full_bucket_glob[5:]  # Remove 's3://'
        bucket, *glob_prefix = full_bucket_glob.split("/", 1)
        prefix = glob_prefix[0] if glob_prefix else ""

    # Ensure bucket is specified
    if not bucket:
        raise ValueError("Bucket name must be specified")

    paginator = s3.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket}
    if prefix:
        operation_parameters["Prefix"] = prefix

    page_iterator = paginator.paginate(**operation_parameters)

    for page in page_iterator:
        if "Contents" in page:
            for item in page["Contents"]:
                key = item["Key"]
                if suffix:
                    if key.endswith(suffix):
                        out.append(key)
                else:
                    out.append(key)
    return out
