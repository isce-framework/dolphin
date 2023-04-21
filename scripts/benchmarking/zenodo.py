from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pooch
import requests

from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)

RECORD_ID = "1171149"


def make_pooch(
    record: Union[int, str], sandbox: bool = True, path: Optional[Filename] = None
):
    """Create a Pooch instance for a Zenodo record.

    Parameters
    ----------
    record : Union[int, str]
        The Zenodo record ID.
    sandbox : bool, optional
        Whether to use the sandbox (default) or the real Zenodo.
    path : Optional[Filename], optional
        The path to the cache folder. If None (default), use the default
        cache folder for the operating system.

    Returns
    -------
    pooch.Pooch
        The Pooch instance.
    """
    base_url, registry = get_zenodo_links(record, sandbox)
    dog = pooch.create(
        # Use the default cache folder for the operating system
        path=path or pooch.os_cache("dolphin"),
        base_url=base_url,
        # The registry specifies the files that can be fetched
        registry=registry,
    )
    return dog


def get_zenodo_links(
    record: Union[int, str], sandbox=True
) -> tuple[str, dict[str, str]]:
    """Get the urls and MD5 checksums for the files in a Zenodo record."""
    # Get the record metadata
    if sandbox:
        url = f"https://sandbox.zenodo.org/api/records/{record}"
    else:
        url = f"https://zenodo.org/api/records/{record}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    # Get the files
    files = data["files"]
    # print the total size of the files
    total_size = sum(f["size"] for f in files)
    logger.info(f"Total size of {len(files)} files: {total_size / 1e6:.1f} MB")

    # Extract the urls and checksums
    urls = [f["links"]["self"] for f in files]
    # Get the base url
    base_url = "/".join(urls[0].split("/")[:-1]) + "/"

    filenames = [Path(fn).name for fn in urls]
    checksums = [f["checksum"] for f in files]

    # Return a dict compatible with the Pooch registry
    return base_url, dict(zip(filenames, checksums))


POOCH = make_pooch(RECORD_ID)


def get_all_files(
    record: Union[int, str], sandbox: bool = True, path: Optional[Filename] = None
):
    """Download all the files in a Zenodo record."""
    dog = make_pooch(record, sandbox, path)
    # Fetch all the files
    for fn in dog.registry_files:
        logger.info("Fetching " + fn)
        dog.fetch(fn)
    return dog
