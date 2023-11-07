from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Tuple, TypeVar, Union

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in a runtime error
# https://stackoverflow.com/questions/71077499/typeerror-abcmeta-object-is-not-subscriptable
if TYPE_CHECKING:
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike


PathOrStr = Union[str, PathLikeStr]
Filename = PathOrStr  # May add a deprecation notice for `Filename`
PathLikeT = TypeVar("PathLikeT", bound=PathLikeStr)

# left, bottom, right, top
Bbox = Tuple[float, float, float, float]
