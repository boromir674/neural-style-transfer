import json
import logging
import sys

import numpy as np

from .nst_image import ImageManager

logger = logging.getLogger(__name__)


# Helper Function
def load_pretrained_model_functions():
    """Load Pretrained Model Interface Implementations."""
    # future work: discover dynamically the modules inside the pre_trained_model
    # package
    from .pre_trained_models import vgg

    return vgg


# Helper Function
def read_images(content, style, auto_resize_style=False):

    # todo dynamically find means
    # TODOD !!! find dynamically the means
    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # means

    image_manager = ImageManager.default(means)
    # probably load each image in separate thread and then join
    image_manager.load_from_disk(content, "content")
    logger.info(
        "Loaded CONTENT image tensor [1 x Height x Width x Channels] %s",
        image_manager.content_image.matrix.shape,
    )
    logger.debug(
        "CONTENT image Width x Height: %s x %s",
        image_manager.content_image.matrix.shape[2],
        image_manager.content_image.matrix.shape[1],
    )

    if auto_resize_style:
        from PIL import Image

        image = Image.open(str(style))
        logger.info("Read STYLE image.size %s", image.size)

        # image = Image.fromarray(image_manager.style_image.matrix)
        # get width, height from content_image
        content_image_dimensions = (  # width, height
            image_manager.content_image.matrix.shape[2],
            image_manager.content_image.matrix.shape[1],
        )
        if image.size != content_image_dimensions:
            print(f"[INFO] Resizing STYLE image to fit the CONTENT image")
            logger.info(
                "Resizing STYLE image to match the CONTENT image: %s",
                json.dumps(
                    {
                        "content_image_dimensions": tuple(content_image_dimensions),
                        "style_image_dimensions": tuple(image.size),
                    },
                    sort_keys=True,
                    indent=4,
                ),
            )

            # Resize the PIL image to match the content image
            image = image.resize(content_image_dimensions)
            image_array = np.array(image)
            print(f"SANITY resized dims: {image_array.shape}")
            image_manager.load_from_disk(style, "style", array=image_array)

    if not image_manager.style_image:
        image_manager.load_from_disk(style, "style")

    logger.info(
        "Loaded STYLE image tensor [1 x Height x Width x Channels] %s",
        image_manager.style_image.matrix.shape,
    )

    if not image_manager.images_compatible:
        print(
            "Given CONTENT image '{content_image}' has 'height' x 'width' x "
            f"'color_channels': {image_manager.content_image.matrix.shape}"
        )
        print(
            "Given STYLE image '{style_image}' has 'height' x 'width' x "
            f"'color_channels': {image_manager.style_image.matrix.shape}"
        )
        print("Expected to receive images (matrices) of identical shape")
        print("Exiting..")
        sys.exit(1)

    return image_manager.content_image, image_manager.style_image
