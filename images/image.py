
import numpy as np
import scipy.misc
import attr


class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


@attr.s
class ArtImage:
    path = attr.ib(init=True)
    matrix = attr.ib(init=True)
    saved = attr.ib(init=True)


    @classmethod
    def from_file(cls, file_path):
        return ArtImage(file_path,
                        reshape_and_normalize_image(scipy.misc.imread(file_path)),
                        True)

    @classmethod
    def from_matrix(cls, image):
        return ArtImage('', image, False)

    @classmethod
    def noisy(cls, image, noise_ratio=CONFIG.NOISE_RATIO):
        """Generates a noisy image by adding random noise to the content_image"""
        noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')

        # Set the input_image to be a weighted average of the content_image and a noise_image
        return noise_image * noise_ratio + image * (1 - noise_ratio)

    def save(self, path):
        self.save_image(path, self.matrix)

    @staticmethod
    def save_image(path, image):
        # Un-normalize the image so that it looks good
        image = image + CONFIG.MEANS

        # Clip and Save the image
        image = np.clip(image[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, image)


def reshape_and_normalize_image(image):
    """Reshape and normalize the input image (content or style)"""
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    # Substract the mean to match the expected input of VGG16
    return image - CONFIG.MEANS
