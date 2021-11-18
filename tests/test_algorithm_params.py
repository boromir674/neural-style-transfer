import pytest

@pytest.fixture
def algorithm_parameters():
    from artificial_artwork.algorithm import AlogirthmParameters
    return AlogirthmParameters(
        'content_image',
        'style_image',
        [
            ('layer-1', 0.5),
            ('layer-2', 0.5),
        ],
        'termination_condition',
        'output_path',
    )


def test_algorithm_parameters(algorithm_parameters):
    assert hasattr(algorithm_parameters, 'content_image')
    assert hasattr(algorithm_parameters, 'style_image')
    assert hasattr(algorithm_parameters, 'style_layers')
    assert hasattr(algorithm_parameters, 'termination_condition')
    assert hasattr(algorithm_parameters, 'output_path')
