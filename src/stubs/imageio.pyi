from typing import Union

import numpy.typing as npt

ImSaveFormatParamType = Union[str, None]

def imread(file_path: str) -> npt.NDArray: ...
def imsave(file_path: str, image: npt.NDArray, format: ImSaveFormatParamType) -> None: ...
