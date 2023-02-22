"""Exports a get_log function which sets up easy logging.

Uses the standard python logging utilities, just provides
nice formatting out of the box across multiple files.

Usage:

    from ._log import get_log
    logger = get_log()

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
from typing import Optional

__all__ = ["get_log", "log_runtime"]


def get_log(
    name: str = "dolphin._log", debug: bool = False, filename: Optional[str] = None
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
    return format_log(logger, debug=debug, filename=filename)


def format_log(
    logger: logging.Logger, debug: bool = False, filename: Optional[str] = None
) -> logging.Logger:
    """Make the logging output pretty and colored with times.

    Parameters
    ----------
    logger : logging.Logger
        The logger to format
    debug : bool (Default value = False)
        If true, sets logging level to DEBUG
    filename : str, optional
        If provided, will log to this file in addition to stderr.

    Returns
    -------
    logging.Logger
    """
    log_level = logging.DEBUG if debug else logging.INFO
    format_ = "[%(asctime)s] [%(levelname)s %(filename)s] %(message)s"
    formatter = Formatter(format_, datefmt="%m/%d %H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(log_level)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(debug)

    return logger


def log_runtime(f: Callable) -> Callable:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = get_log()

    @wraps(f)
    def wrapper(*args, **kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_time = t2 - t1
        time_string = "Total elapsed time for {} : {} minutes ({} seconds)".format(
            f.__name__,
            "{0:.2f}".format(elapsed_time / 60.0),
            "{0:.2f}".format(elapsed_time),
        )

        logger.info(time_string)
        return result

    return wrapper
