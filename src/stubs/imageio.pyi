
import numpy.typing as npt
from typing import Union
ImSaveFormatParamType = Union[str, None]


def imread(file_path: str) -> npt.NDArray: ...


def imsave(file_path: str, image: npt.NDArray, format: ImSaveFormatParamType) -> None: ...
