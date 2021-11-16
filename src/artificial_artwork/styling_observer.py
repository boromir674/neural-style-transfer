from attr import define 
from .utils.notification import Observer
from typing import Callable
import numpy.typing as npt
import numpy as np
import os


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

        # Impelement handling of the request to persist with a chain of responsibility design pattern
        # it suit since we do not knw how many checks and/or image transformation will be required before
        # saving on disk

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
        self.save_on_disk_callback(matrix, output_file_path, format='png')

    # bit_2_data_type = {8: np.uint8}

    # def _convert_to_uint8(self, im):
    #     bitdepth = 8
    #     out_type = type(self).bit_2_data_type[bitdepth]
    #     mi = np.nanmin(im)
    #     ma = np.nanmax(im)
    #     if not np.isfinite(mi):
    #         raise ValueError("Minimum image value is not finite")
    #     if not np.isfinite(ma):
    #         raise ValueError("Maximum image value is not finite")
    #     if ma == mi:
    #         return im.astype(out_type)

    #     # Make float copy before we scale
    #     im = im.astype("float64")
    #     # Scale the values between 0 and 1 then multiply by the max value
    #     im = (im - mi) / (ma - mi) * (np.power(2.0, bitdepth) - 1) + 0.499999999
    #     assert np.nanmin(im) >= 0
    #     assert np.nanmax(im) < np.power(2.0, bitdepth)
    #     im = im.astype(out_type)
    #     return im
