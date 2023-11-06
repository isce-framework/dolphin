import functools
import inspect
import shutil
import tempfile
from inspect import Parameter
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
                # Track where we will slot in the new tempfile name (kwargs here)
                replace_tuple = (kwargs, output_arg)
            else:
                # Check that it was provided as positional, in `args`
                sig = inspect.signature(func)
                for idx, param in enumerate(sig.parameters.values()):
                    if output_arg == param.name:
                        try:
                            # If the gave it in the args, use that
                            final_out_name = args[idx]
                            # Track where we will slot in the new tempfile name (args here)
                            # Need to make `args` into a list to we can mutate
                            replace_tuple = (list(args), idx)
                        except IndexError:
                            # Otherwise, nothing was given, so use the default
                            final_out_name = param.default
                            if param.kind == Parameter.POSITIONAL_ONLY:
                                # Insert as a positional arg if it needs to be
                                replace_tuple = (list(args), idx)
                            else:
                                replace_tuple = (kwargs, output_arg)
                        break
                else:
                    raise ValueError(
                        f"Argument {output_arg} not found in function {func.__name__}"
                    )

            final_path = Path(final_out_name)
            if scratch_dir is None:
                tmp_dir = final_path.parent
            else:
                tmp_dir = None

            # Make the tempfile start the same as the desired output
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
                # It would be like this if we only allows keyword:
                # kwargs[output_arg] = temp_path
                replace_tuple[0][replace_tuple[1]] = temp_path
                # Execute the original function
                result = func(*args, **kwargs)
                # Move the temp file to the final location
                shutil.move(temp_path, final_path)

                return result
            finally:
                logger.debug("Cleaning up temp file %s", temp_path)
                # Different cleanup is needed
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
