import attr
from numpy.typing import NDArray


@attr.s
class Image:
    """An image loaded into memory, represented as a multidimension mathematical matrix/array.
    
    The 'file_path' attribute indicates a file in the disk that corresponds to the matrix

    Args:
        file_path (str): 
        matrix (NDArray): 
    """
    file_path: str = attr.ib()
    matrix: NDArray = attr.ib(default=None)
