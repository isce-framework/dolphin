from os import PathLike
from typing import Union

# In Python 3.8 and earlier, os.PathLike is not subscriptable, results in a runtime error
PathLikeStr = PathLike[str]
Filename = Union[str, PathLikeStr]
