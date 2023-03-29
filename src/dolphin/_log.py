"""Exports a get_log function which sets up easy logging.

Uses the standard python logging utilities, just provides
nice formatting out of the box across multiple files.

Usage:

    from ._log import get_log
    logger = get_log(__name__)

    logger.info("Something happened")
    logger.warning("Something concerning happened")
    logger.error("Something bad happened")
    logger.critical("Something just awful happened")
    logger.debug("Extra printing we often don't need to see.")
    # Custom output for this module:
    logger.success("Something great happened: highlight this success")
"""
import logging
import time
from collections.abc import Callable
from functools import wraps
from logging import Formatter
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from dolphin._types import Filename

__all__ = ["get_log", "log_runtime"]


def get_log(
    name: str = "dolphin._log", debug: bool = False, filename: Optional[Filename] = None
) -> logging.Logger:
    """Create a nice log format for use across multiple files.

    Default logging level is INFO

    Parameters
    ----------
    debug : bool, optional
        If true, sets logging level to DEBUG (Default value = False)
    name : str, optional
        The name the logger will use when printing statements
        (Default value = "dolphin._log")
    filename : str, optional
        If provided, will log to this file in addition to stderr.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        setup_logging(debug=debug)
    if debug:
        logger.setLevel(logging.DEBUG)

    # In addition to stderr, log to a file if requested
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def setup_logging(debug: bool = False) -> None:
    """Make the logging output pretty and colored with times.

    Parameters
    ----------
    debug : bool (Default value = False)
        If true, sets logging level to DEBUG

    """
    # Set for all dolphin modules
    logger = logging.getLogger("dolphin")
    h = RichHandler(rich_tracebacks=True, log_time_format="[%Y-%m-%d %H:%M:%S]")
    logger.addHandler(h)
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)


def log_runtime(f: Callable) -> Callable:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = get_log(__name__)

    @wraps(f)
    def wrapper(*args, **kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0
        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__} : "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )

        logger.info(time_string)

        return result

    return wrapper
