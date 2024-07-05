import datetime as dt
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from logging import Formatter

from dolphin._types import P, T

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}
__all__ = ["log_runtime"]


def setup_logging(debug: bool = False, root_name: str = "dolphin") -> None:
    """Make the logging output pretty and colored with times.

    Parameters
    ----------
    debug : bool (default = False)
        If true, sets logging level to DEBUG
    root_name : str (default = "dolphin")
        Name of the base logger to configure.
        All sub-loggers (e.g. modules with <root_name>.<module_name>) will
        also get access to the handler.

    """
    # Set for all dolphin modules
    logger = logging.getLogger(root_name)
    h = logging.StreamHandler()
    formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    h.setFormatter(formatter)
    logger.addHandler(h)
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    # Temp work around for tqdm on py312

    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        os.environ["TQDM_DISABLE"] = "1"


def log_runtime(f: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = logging.getLogger(__name__)

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
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


class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: (
                msg_val
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message
