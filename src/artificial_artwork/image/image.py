import attr
from numpy.typing import NDArray


@attr.s
class Image:
    """An image loaded into memory, represented as a multidimension mathematical matrix/array.

    The 'file_path' attribute indicates a file in the disk that corresponds to the matrix

    Args:
        file_path (str): the file in disk that the image was loaded from
        matrix (NDArray): the loaded image as mathmatical array/matrix
    """

    file_path: str = attr.ib()
    matrix: NDArray = attr.ib(default=None)
