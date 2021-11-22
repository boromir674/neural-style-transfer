import os
from typing import Callable
from attr import define
import numpy as np
import numpy.typing as npt

from .utils.notification import Observer


@define
class StylingObserver(Observer):
    save_on_disk_callback: Callable[[str, npt.NDArray], None]
    convert_to_unit8: Callable[[npt.NDArray], npt.NDArray]
    """Store a snapshot of the image under construction.

    Args:
        Observer ([type]): [description]
    """
    def update(self, *args, **kwargs):
        output_dir = args[0].state.output_path
        content_image_path = args[0].state.content_image_path
        style_image_path = args[0].state.style_image_path
        iterations_completed = args[0].state.metrics['iterations']
        matrix = args[0].state.matrix

        # Future work: Impelement handling of the "request to persist" with a
        # chain of responsibility design pattern. It suits this case  since we
        # do not know how many checks and/or image transformation will be
        # required before saving on disk

        output_file_path = os.path.join(
            output_dir,
            f'{os.path.basename(content_image_path)}+{os.path.basename(style_image_path)}-{iterations_completed}.png'
        )
        # if we have shape of form (1, Width, Height, Number_of_Color_Channels)
        if matrix.ndim == 4 and matrix.shape[0] == 1:
            # reshape to (Width, Height, Number_of_Color_Channels)
            matrix = np.reshape(matrix, tuple(matrix.shape[1:]))

        if str(matrix.dtype) != 'uint8':
            matrix = self.convert_to_unit8(matrix)
        self.save_on_disk_callback(matrix, output_file_path, save_format='png')
