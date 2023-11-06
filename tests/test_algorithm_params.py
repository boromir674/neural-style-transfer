import pytest


@pytest.fixture
def algorithm_parameters():
    from artificial_artwork.algorithm import AlogirthmParameters

    return AlogirthmParameters(
        "content_image",
        "style_image",
        "termination_condition",
        "output_path",
    )


def test_algorithm_parameters(algorithm_parameters):
    assert hasattr(algorithm_parameters, "content_image")
    assert hasattr(algorithm_parameters, "style_image")
    assert hasattr(algorithm_parameters, "termination_condition")
    assert hasattr(algorithm_parameters, "output_path")
