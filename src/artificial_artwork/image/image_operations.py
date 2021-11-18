import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def reshape_image(image: NDArray, shape: Tuple[int, ...]) -> NDArray:
    return np.reshape(image, shape)


def subtract(image: NDArray, array: NDArray) -> NDArray:
    """Normalize the input image.

    Args:
        image (NDArray): [description]

    Raises:
        ShapeMissmatchError: in case of ValueError due to numpy broadcasting failing

    Returns:
        NDArray: [description]
    """ 
    try:
        return image - array
    except ValueError as numpy_broadcast_error: 
        raise ShapeMissmatchError(
            f'Expected arrays with matching shapes.') from numpy_broadcast_error


class ShapeMissmatchError(Exception): pass


def noisy(image: NDArray, ratio: float) -> NDArray:
    """Generates a noisy image by adding random noise to the content_image"""
    if ratio < 0 or 1 < ratio:
        raise InvalidRatioError('Expected a ratio value x such that 0 <= x <= 1') 

    prod_shape = image.shape
    # assert prod_shape == (1, self.config.image_height, self.config.image_width, self.config.color_channels)

    noise_image = np.random.uniform(-20, 20, prod_shape).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    return noise_image * ratio + image * (1 - ratio)


class InvalidRatioError(Exception): pass


class ImageDTypeConverter:

    bit_2_data_type = {8: np.uint8}

    def __call__(self, image: NDArray):
        return self._convert_to_uint8(image)

    def _convert_to_uint8(self, im):
        bitdepth = 8
        out_type = type(self).bit_2_data_type[bitdepth]
        mi = np.nanmin(im)
        ma = np.nanmax(im)
        if not np.isfinite(mi):
            raise ValueError("Minimum image value is not finite")
        if not np.isfinite(ma):
            raise ValueError("Maximum image value is not finite")
        if ma == mi:
            return im.astype(out_type)

        # Make float copy before we scale
        im = im.astype("float64")
        # Scale the values between 0 and 1 then multiply by the max value
        im = (im - mi) / (ma - mi) * (np.power(2.0, bitdepth) - 1) + 0.499999999
        assert np.nanmin(im) >= 0
        assert np.nanmax(im) < np.power(2.0, bitdepth)
        im = im.astype(out_type)
        return im


convert_to_uint8 = ImageDTypeConverter()
