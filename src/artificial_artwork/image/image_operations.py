import typing as t
from typing import Tuple

import numpy as np
from attr import Factory, define, field
from numpy.typing import NDArray

__all__ = ["reshape_image", "subtract", "noisy", "convert_to_uint8"]

MinPixelValue = int
MaxPixelValue = int
ArrayShape = t.Tuple[int, ...]
RandomArrayGenerator = t.Callable[[MinPixelValue, MaxPixelValue, ArrayShape], NDArray]


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
            "Expected arrays with matching shapes."
        ) from numpy_broadcast_error


class ShapeMissmatchError(Exception):
    pass


# Add random noise to an image, given a ratio percentage, and optional seed
@define(slots=True, kw_only=True)
class ImageNoiseAdder:
    """Add random noise to an image, given a ratio and optional seed number.

    If seed is passed other than None, then the instance is configued with a
    seeded random number generator (RNG). Otherwise, the instance is configured
    with a non-seeded RNG.
    """

    seed: t.Union[int, None] = field(default=None)
    min_pixel_value: int = field(default=-20)
    max_pixel_value: int = field(default=20)

    # Private Attributes
    _default_rng: RandomArrayGenerator = field(
        init=False,
        default=Factory(
            lambda self: self._create_rng(
                self.seed,
            ),
            takes_self=True,
        ),
    )

    def __call__(
        self, image: NDArray, ratio: float, seed: t.Union[int, None] = None
    ) -> NDArray:
        """Generates a noisy image by adding random noise to the content_image

        If instance has been configured without Seed, and you wish to continue
        using the same RNG, then pass no seed.

        If instance has been configured with Seed, and you wish to continue
        using the same RNG, then pass no seed.

        If instance has been configured with Seed, and you wish to use a new
        RNG, then pass a new seed.
        """
        if ratio < 0 or 1 < ratio:
            raise InvalidRatioError("Expected a ratio value x such that 0 <= x <= 1")

        if seed is not None:  # user want to re-configure new RNG Stochastic Process
            # with a provided seed
            self._default_rng = self._create_rng(seed)
        # if seed is None:  # user wants to continue sampling from the configured RNG
        # we continue using the 'default Stochastic Process' for sampling
        random_noise_pixel_array = self._default_rng(  # numpy float32 array
            self.min_pixel_value,
            self.max_pixel_value,
            image.shape,
        )
        # Set the input_image to be a weighted average of the content_image and a noise_image
        return random_noise_pixel_array * ratio + image * (1 - ratio)

    # Private methods
    @staticmethod
    def _create_rng(seed: t.Union[int, None]) -> RandomArrayGenerator:
        """Handle request to re-configure RNG potentially seeded with provided seed."""
        _rng = np.random.default_rng(seed=seed)
        rng: RandomArrayGenerator = _rng.uniform

        def _rng_float32(
            minimum_pixel_value: int,
            maximum_pixel_value: int,
            pixel_array_shape: t.Tuple[int, ...],
        ) -> NDArray:
            return rng(
                minimum_pixel_value,
                maximum_pixel_value,
                pixel_array_shape,
            ).astype("float32")

        return _rng_float32


def noisy(
    image: NDArray,
    ratio: float,
    seed: int = None,
) -> NDArray:
    """Generates a noisy image by adding random noise to the content_image"""
    if ratio < 0 or 1 < ratio:
        raise InvalidRatioError("Expected a ratio value x such that 0 <= x <= 1")

    noise_image = np.random.uniform(-20, 20, image.shape).astype("float32")

    # Set the input_image to be a weighted average of the content_image and a noise_image
    return noise_image * ratio + image * (1 - ratio)


class InvalidRatioError(Exception):
    pass


class ImageDTypeConverter:
    bit_2_data_type = {8: np.uint8}

    def __call__(self, image: NDArray):
        return self._convert_to_uint8(image)

    def _convert_to_uint8(self, image):
        bitdepth = 8
        out_type = type(self).bit_2_data_type[bitdepth]
        min_pixel_value = np.nanmin(image)
        max_pixel_value = np.nanmax(image)
        if not np.isfinite(min_pixel_value):
            raise ValueError("Minimum image value is not finite")
        if not np.isfinite(max_pixel_value):
            raise ValueError("Maximum image value is not finite")
        if max_pixel_value == min_pixel_value:
            return image.astype(out_type)

        # Make float copy before we scale
        im = image.astype("float64")
        # Scale the values between 0 and 1 then multiply by the max value
        im = (im - min_pixel_value) / (max_pixel_value - min_pixel_value) * (
            np.power(2.0, bitdepth) - 1
        ) + 0.499999999
        return im.astype(out_type)


convert_to_uint8 = ImageDTypeConverter()
