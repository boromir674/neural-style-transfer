import imageio
from numpy.typing import NDArray

from .disk_interface import DiskInterface


class Disk(DiskInterface):
    """Save or load images to and from the disk."""
    
    @staticmethod
    def save_image(image: NDArray, file_path: str, format=None) -> None:
        """Save a numpy ndarray into a file on the disk.

        Args:
            image (NDArray): the image to save into a file
            file_path (str): the path (on the disk) of the file
        """
        imageio.imsave(file_path, image, format=format)

    @staticmethod
    def load_image(file_path: str) -> NDArray:
        """Load an image as numpy ndarray from a file on the disk.

        Args:
            file_path (str): the path (on the disk) of the file

        Returns:
            NDArray: the image as numpy ndarray
        """
        return imageio.imread(file_path)
