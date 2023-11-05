import pytest


# TEST that when adding random noise to the same image, twice in a row, the
# result is different
def test_adding_noise_results_in_random_image(
    test_suite,
):
    import typing as t
    from pathlib import Path

    from numpy.typing import NDArray

    # CONSTANTS #
    expected_content_image_shape = (
        225,  # Height
        300,  # Width
        3,
    )
    # GIVEN an "original" Content Image
    import imageio

    canoe_content_image: Path = Path(test_suite) / "data" / "canoe_water_w300-h225.jpg"

    content_image: NDArray = imageio.imread(canoe_content_image)
    assert content_image.shape == expected_content_image_shape

    # GIVEN an "Add Random Noise Operation"
    from artificial_artwork.image import noisy

    NOISY_RATIO: float = 0.6
    apply_noise: t.Callable[[NDArray], NDArray] = lambda matrix_array: noisy(
        matrix_array, NOISY_RATIO
    )

    # WHEN we add random noise to the "original" Content Image, twice in a row
    # noisy_content_image_matrix = self.apply_noise(self.nst_algorithm.parameters.content_image.matrix)
    first_noisy_content_image_matrix = apply_noise(content_image)
    second_noisy_content_image_matrix = apply_noise(content_image)

    # THEN 2 new Images are produced, and they are different
    assert first_noisy_content_image_matrix.shape == expected_content_image_shape
    assert second_noisy_content_image_matrix.shape == expected_content_image_shape
    assert not (first_noisy_content_image_matrix == second_noisy_content_image_matrix).all()


def test_numpy_random_generator_stochastic_process_is_reproducable():
    import numpy as np

    # Expected 1st Sample from RNG with seed 1234
    expected_array_1 = np.array(
        [
            [
                [19.067991, -4.7921705, 16.92985],
                [-9.532303, -7.236118, -15.276351],
                [-10.329349, -7.2586427, 18.56317],
            ],
            [
                [-9.454008, -2.359755, 4.3948326],
                [14.544852, 14.550307, 6.9952526],
                [6.3949738, 9.430308, -11.089853],
            ],
        ],
        dtype="float32",
    )
    # Expected 2nd Sample from RNG with seed 1234
    expected_array_2 = np.array(
        [
            [
                [-13.1173525, 14.816599, -17.594454],
                [7.3475566, 6.8495207, 4.440719],
                [-17.594507, 19.110771, -2.4419348],
            ],
            [
                [1.3038008, -19.874708, -9.949316],
                [14.339618, -2.988066, 9.432759],
                [16.88173, -13.861033, 19.690369],
            ],
        ],
        dtype="float32",
    )

    expected_image_shape = (
        2,  # Height
        3,  # Width
        3,  # color channels
    )
    seed: int = 1234
    rng1 = np.random.default_rng(seed=seed)
    # rng = np.random.default_rng()
    random_noise_image_11 = rng1.uniform(-20, 20, expected_image_shape).astype("float32")

    assert expected_array_1.shape == expected_image_shape
    assert random_noise_image_11.shape == expected_image_shape
    assert (random_noise_image_11 == expected_array_1).all()

    # Verify second RNG 1st sample is DIFFERENT than first RNG 1st sample
    rng2 = np.random.default_rng(seed=seed + 1)
    random_noise_image_21 = rng2.uniform(-20, 20, expected_image_shape).astype("float32")
    assert random_noise_image_21.shape == expected_image_shape
    assert not (random_noise_image_21 == expected_array_1).all()

    assert not (random_noise_image_21 == random_noise_image_11).all()

    # Take second sample and verify it is the expected 2nd Sample from RNG 1234
    random_noise_image_12 = rng1.uniform(-20, 20, expected_image_shape).astype("float32")
    assert random_noise_image_12.shape == expected_image_shape
    assert not (random_noise_image_12 == expected_array_1).all()
    assert (random_noise_image_12 == expected_array_2).all()


def test_verify_if_production_uses_the_same_seed_on_restart():
    # GIVEN a "seed" value
    seed: int = 1234

    # WHEN we add random noise to the "original" Content Image, twice in a row
    # noisy_content_image_matrix = self.apply_noise(self.nst_algorithm.parameters.content_image.matrix)


@pytest.fixture
def prod_read_image_from_disk():
    import typing as t

    import numpy as np
    from numpy.typing import NDArray

    from artificial_artwork.disk_operations import Disk
    from artificial_artwork.image.image_factory import ImageFactory
    from artificial_artwork.image.image_operations import reshape_image, subtract

    def _prod_read_image_from_disk(image_path: str) -> NDArray:
        means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # means
        image_fct = ImageFactory(
            Disk.load_image,
        )
        prod_preprocessing_pipeline = [
            lambda x: reshape_image(x, ((1,) + x.shape)),
            lambda x: subtract(x, means),
        ]
        img_wrapper = image_fct.from_disk(image_path, prod_preprocessing_pipeline)
        # file_path: str
        # matrix: NDArray
        return img_wrapper.matrix

    return _prod_read_image_from_disk


# TEST that the first operation to add random noise to an image results to the
# same prerecorded generated-with-noise image produced with the same seed
def test_adding_noise_with_the_same_seed_results_in_same_image(
    test_suite,
    prod_read_image_from_disk,
):
    import typing as t
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    rng = np.random.default_rng(seed=42)

    # CONSTANTS #
    expected_content_image_shape = (
        225,  # Height
        300,  # Width
        3,
    )
    # GIVEN the Noisy Content Image of Running the Production Demo on process A
    noisy_canoe_content_image_v1: Path = (
        Path(test_suite) / "data" / "demo-image-noisy-iter-1-1234_v1.png"
    )
    # noisy_content_image_v1: NDArray = imageio.imread(noisy_canoe_content_image_v1)
    noisy_content_image_v1: NDArray = prod_read_image_from_disk(noisy_canoe_content_image_v1)
    # GIVEN the Noisy Content Image of Running the Production Demo on process B
    noisy_canoe_content_image_v2: Path = (
        Path(test_suite) / "data" / "demo-image-noisy-iter-1-1234_v2.png"
    )
    # noisy_content_image_v2: NDArray = imageio.imread(noisy_canoe_content_image_v2)
    noisy_content_image_v2: NDArray = prod_read_image_from_disk(noisy_canoe_content_image_v2)

    # THEN since the Production RNG uses initializes a Stochastic Process with default Seed 1234
    # THEN the Noisy Content Image of Running the Production Demo on process A
    # is the same as the Noisy Content Image of Running the Production Demo on process B
    assert (noisy_content_image_v1 == noisy_content_image_v2).all()

    # GIVEN a "seed" value
    PROD_DEFAULT_SEED = 1234

    # GIVEN an "Add Random Noise Operation"
    from artificial_artwork.image.image_operations import ImageNoiseAdder

    noise_adder = ImageNoiseAdder(seed=PROD_DEFAULT_SEED)
    PROD_DEFAULT_NOISY_RATIO: float = 0.6
    apply_noise = lambda x: noise_adder(x, PROD_DEFAULT_NOISY_RATIO)

    # GIVEN an "original" Content Image
    canoe_content_image: Path = Path(test_suite) / "data" / "canoe_water_w300-h225.jpg"
    content_image: NDArray = prod_read_image_from_disk(canoe_content_image)

    assert content_image.shape == (1,) + expected_content_image_shape

    # WHEN we add random noise to the "original" Content Image, twice in a row
    # noisy_content_image_matrix = self.apply_noise(self.nst_algorithm.parameters.content_image.matrix)
    first_noisy_content_image_matrix = apply_noise(content_image)
    second_noisy_content_image_matrix = apply_noise(content_image)

    # assert first_noisy_content_image_matrix.shape == expected_content_image_shape
    # assert second_noisy_content_image_matrix.shape == expected_content_image_shape

    # THEN 2 new Images are produced, and they are different
    assert not (first_noisy_content_image_matrix == second_noisy_content_image_matrix).all()

    # THEN the first noisy image is the same as the prerecorded generated-with-noise image
    # produced with the same seed
    from artificial_artwork.image.image_operations import convert_to_uint8

    # if we have shape of form (1, Width, Height, Number_of_Color_Channels)
    if (
        first_noisy_content_image_matrix.ndim == 4
        and first_noisy_content_image_matrix.shape[0] == 1
    ):
        import numpy as np

        # reshape to (Width, Height, Number_of_Color_Channels)
        first_noisy_content_image_matrix = np.reshape(
            first_noisy_content_image_matrix, tuple(first_noisy_content_image_matrix.shape[1:])
        )
    if str(first_noisy_content_image_matrix.dtype) != "uint8":
        first_noisy_content_image_matrix = convert_to_uint8(first_noisy_content_image_matrix)

    from artificial_artwork.disk_operations import Disk

    ppath = "/tmp/nst-unit-test-image.png"
    Disk.save_image(first_noisy_content_image_matrix, ppath, save_format="png")
    first_noisy_content_image_matrix: NDArray = prod_read_image_from_disk(ppath)

    assert first_noisy_content_image_matrix.shape == noisy_content_image_v1.shape
    assert (first_noisy_content_image_matrix == noisy_content_image_v1).all()


def test_prod_noise():
    import numpy as np

    from artificial_artwork.image.image_operations import ImageNoiseAdder

    # Expected 1st Sample from RNG with seed 1234
    expected_array_1 = np.array(
        [
            [
                [19.067991, -4.7921705, 16.92985],
                [-9.532303, -7.236118, -15.276351],
                [-10.329349, -7.2586427, 18.56317],
            ],
            [
                [-9.454008, -2.359755, 4.3948326],
                [14.544852, 14.550307, 6.9952526],
                [6.3949738, 9.430308, -11.089853],
            ],
        ],
        dtype="float32",
    )
    # Expected 2nd Sample from RNG with seed 1234
    expected_array_2 = np.array(
        [
            [
                [-13.1173525, 14.816599, -17.594454],
                [7.3475566, 6.8495207, 4.440719],
                [-17.594507, 19.110771, -2.4419348],
            ],
            [
                [1.3038008, -19.874708, -9.949316],
                [14.339618, -2.988066, 9.432759],
                [16.88173, -13.861033, 19.690369],
            ],
        ],
        dtype="float32",
    )

    expected_image_shape = (
        2,  # Height
        3,  # Width
        3,  # color channels
    )
    seed: int = 1234

    rng = ImageNoiseAdder(seed=seed)._default_rng

    random_noise_image_11 = rng(-20, 20, expected_image_shape)

    assert expected_array_1.shape == expected_image_shape
    assert random_noise_image_11.shape == expected_image_shape
    assert (random_noise_image_11 == expected_array_1).all()

    # Take second sample and verify it is the expected 2nd Sample from RNG 1234
    random_noise_image_12 = rng(-20, 20, expected_image_shape)

    assert random_noise_image_12.shape == expected_image_shape
    assert not (random_noise_image_12 == expected_array_1).all()
    assert (random_noise_image_12 == expected_array_2).all()


def test_changing_seeds_prod_noise():
    import numpy as np

    from artificial_artwork.image.image_operations import ImageNoiseAdder

    # Expected 1st Sample from RNG with seed 1234
    expected_array_1 = np.array(
        [
            [
                [19.067991, -4.7921705, 16.92985],
                [-9.532303, -7.236118, -15.276351],
                [-10.329349, -7.2586427, 18.56317],
            ],
            [
                [-9.454008, -2.359755, 4.3948326],
                [14.544852, 14.550307, 6.9952526],
                [6.3949738, 9.430308, -11.089853],
            ],
        ],
        dtype="float32",
    )
    # Expected 2nd Sample from RNG with seed 1234
    expected_array_2 = np.array(
        [
            [
                [-13.1173525, 14.816599, -17.594454],
                [7.3475566, 6.8495207, 4.440719],
                [-17.594507, 19.110771, -2.4419348],
            ],
            [
                [1.3038008, -19.874708, -9.949316],
                [14.339618, -2.988066, 9.432759],
                [16.88173, -13.861033, 19.690369],
            ],
        ],
        dtype="float32",
    )

    expected_image_shape = (
        2,  # Height
        3,  # Width
        3,  # color channels
    )
    seed: int = 1234

    rng = ImageNoiseAdder(seed=seed)._default_rng

    random_noise_image_11 = rng(-20, 20, expected_image_shape)

    assert expected_array_1.shape == expected_image_shape
    assert random_noise_image_11.shape == expected_image_shape
    assert (random_noise_image_11 == expected_array_1).all()

    # use infra to configure RNG with the seed
    # rng2 = ImageNoiseAdder._create_rng(seed=seed+1)

    # use infra to configure RNG with Previous Seed
    rng1 = ImageNoiseAdder._create_rng(seed=seed)

    # Take second sample from RNG 1
    random_noise_image_12 = rng1(-20, 20, expected_image_shape)

    assert random_noise_image_12.shape == expected_image_shape
    assert not (random_noise_image_12 == expected_array_2).all()

    random_noise_image_13 = rng(-20, 20, expected_image_shape)
    assert random_noise_image_13.shape == expected_image_shape
    assert (random_noise_image_13 == expected_array_2).all()
