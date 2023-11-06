import os
from typing import Callable

import numpy as np
import numpy.typing as npt
from attr import Factory, define, field
from software_patterns import Observer


def build_default_get_file_name(
    max_iterations,
):
    max_iterations_str_len = len(str(max_iterations))

    def get_iteration_string(iteration):
        return str(iteration).zfill(max_iterations_str_len)

    def _get_file_name(content_image_path, style_image_path, iterations_completed):
        return f"{os.path.basename(content_image_path)}+{os.path.basename(style_image_path)}-{get_iteration_string(iterations_completed)}.png"

    return _get_file_name


@define
class StylingObserver(Observer):
    """Store a snapshot of the image under construction.

    Args:
        save_on_disk_callback (Callable[[str, npt.NDArray], None]): Callback
        convert_to_unit8 (Callable[[npt.NDArray], npt.NDArray]): Callback
    """

    save_on_disk_callback: Callable[[str, npt.NDArray], None]
    convert_to_unit8: Callable[[npt.NDArray], npt.NDArray]
    max_iterations: int
    get_file_name: Callable[[], str] = field(
        default=Factory(
            lambda self: build_default_get_file_name(self.max_iterations), takes_self=True
        )
    )

    def update(self, *args, **kwargs):
        output_dir = args[0].state.output_path
        content_image_path = args[0].state.content_image_path
        style_image_path = args[0].state.style_image_path
        iterations_completed = args[0].state.metrics["iterations"]
        matrix = args[0].state.matrix

        # Future work: Impelement handling of the "request to persist" with a
        # chain of responsibility design pattern. It suits this case  since we
        # do not know how many checks and/or image transformation will be
        # required before saving on disk

        output_file_path = os.path.join(
            output_dir,
            self.get_file_name(content_image_path, style_image_path, iterations_completed),
        )
        # if we have shape of form (1, Width, Height, Number_of_Color_Channels)
        if matrix.ndim == 4 and matrix.shape[0] == 1:
            # reshape to (Width, Height, Number_of_Color_Channels)
            matrix = np.reshape(matrix, tuple(matrix.shape[1:]))

        if str(matrix.dtype) != "uint8":
            matrix = self.convert_to_unit8(matrix)
        if np.nanmin(matrix) < 0:
            raise ImageDataValueError("Generated Image has pixel(s) with negative values.")
        if np.nanmax(matrix) >= np.power(2.0, 8):
            raise ImageDataValueError("Generated Image has pixel(s) with value >= 255.")
        self.save_on_disk_callback(matrix, output_file_path, save_format="png")


class ImageDataValueError(Exception):
    pass
