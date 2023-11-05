import sys

import numpy as np

from .nst_image import ImageManager


# Helper Function
def load_pretrained_model_functions():
    """Load Pretrained Model Interface Implementations."""
    # future work: discover dynamically the modules inside the pre_trained_model
    # package
    from .pre_trained_models import vgg

    return vgg


# Helper Function
def read_images(content, style):
    # todo dynamically find means
    # TODOD !!! find dynamically the means
    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # means

    image_manager = ImageManager.default(means)

    # probably load each image in separate thread and then join
    image_manager.load_from_disk(content, "content")
    image_manager.load_from_disk(style, "style")

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
