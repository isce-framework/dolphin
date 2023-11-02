from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, TypeVar, Union

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in a runtime error
# https://stackoverflow.com/questions/71077499/typeerror-abcmeta-object-is-not-subscriptable
if TYPE_CHECKING:
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

Filename = Union[str, PathLikeStr]
FilenameT = TypeVar("FilenameT", str, PathLikeStr, Path)

# left, bottom, right, top
Bbox = Tuple[float, float, float, float]
