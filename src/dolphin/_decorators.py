import functools
import inspect
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)

__all__ = [
    "atomic_output",
]


def atomic_output(
    function: Optional[Callable] = None,
    output_arg: str = "output_file",
    is_dir: bool = False,
    scratch_dir: Optional[Filename] = None,
) -> Callable:
    """Use a temporary file/directory for the `output_arg` until the function finishes.

    Decorator is used on a function which writes to an output file/directory in blocks.
    If the function were interrupted, the file/directory would be partiall complete.

    This decorator replaces the final output name with a temp file/dir, and then
    renames the temp file/dir to the final name after the function finishes.

    Note that when `is_dir=True`, `output_arg` can be a directory (if multiple files
    are being written to). In this case, the entire directory is temporary, and
    renamed after the function finishes.

    Parameters
    ----------
    function : Optional[Callable]
        Used if the decorator is called without any arguments (i.e. as
        `@atomic_output` instead of `@atomic_output(output_arg=...)`)
    output_arg : str, optional
        The name of the argument to replace, by default 'output_file'
    is_dir : bool, default = False
        If True, the output argument is a directory, not a file
    scratch_dir : Optional[Filename]
        The directory to use for the temporary file, by default None
        If None, uses the same directory as the final requested output.

    Returns
    -------
    Callable
        The decorated function
    """

    def actual_decorator(func: Callable) -> Callable:
        # Want to be able to use this decorator with or without arguments:
        # https://stackoverflow.com/a/19017908/4174466
        # Code adapted from the `@login_required` decorator in Django:
        # https://github.com/django/django/blob/d254a54e7f65e83d8971bd817031bc6af32a7a46/django/contrib/auth/decorators.py#L43  # noqa
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract the output file path
            if output_arg in kwargs:
                final_out_name = kwargs[output_arg]
            else:
                sig = inspect.signature(func)
                if output_arg in sig.parameters:
                    final_out_name = sig.parameters[output_arg].default
                else:
                    raise ValueError(
                        f"Argument {output_arg} not found in function {func.__name__}"
                    )

            final_path = Path(final_out_name)
            if scratch_dir is None:
                tmp_dir = final_path.parent
            else:
                tmp_dir = None

            prefix = final_path.name
            if is_dir:
                # Create a temporary directory
                temp_path = tempfile.mkdtemp(dir=tmp_dir, prefix=prefix)
            else:
                # Create a temporary file
                _, temp_path = tempfile.mkstemp(dir=tmp_dir, prefix=prefix)
            logger.debug("Writing to temp file %s instead of %s", temp_path, final_path)

            try:
                # Replace the output file path with the temp file
                kwargs[output_arg] = temp_path
                # Execute the original function
                result = func(*args, **kwargs)
                # Move the temp file to the final location
                shutil.move(temp_path, final_path)

                return result
            finally:
                logger.debug("Cleaning up temp file %s", temp_path)
                if is_dir:
                    shutil.rmtree(temp_path, ignore_errors=True)
                else:
                    Path(temp_path).unlink(missing_ok=True)

        return wrapper

    if function is not None:
        # Decorator used without arguments
        return actual_decorator(function)
    # Decorator used with arguments
    return actual_decorator
