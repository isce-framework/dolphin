from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Union

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.8 and earlier, os.PathLike is not subscriptable, results in a runtime error
if TYPE_CHECKING:
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

Filename = Union[str, Path, PathLikeStr]
