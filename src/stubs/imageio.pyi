
import numpy.typing as npt
from typing import Union
ImSaveFormatParamType = Union[str, None]


# Function bodies cannot be completely removed. By convention,
# we replace them with `...` instead of the `pass` statement.
def imread(file_path: str) -> npt.NDArray: ...

# We can do the same with default arguments.
def imsave(file_path: str, image: npt.NDArray, save_format: ImSaveFormatParamType) -> None: ...
