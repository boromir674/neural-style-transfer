import pytest


@pytest.fixture
def valid_style_layers():
    def _valid_style_layers_selection(nb_layers):
        return [(f'conv{index}_1', 1.0 / nb_layers) for index in range(1, nb_layers+1)]
    return _valid_style_layers_selection


@pytest.fixture
def valid_nst_layers_list(valid_style_layers):
    from artificial_artwork.style_layer_selector import NSTStyleLayer
    return [NSTStyleLayer(*layer) for layer in valid_style_layers(5)]


@pytest.fixture
def invalid_nst_layers_list():
    from artificial_artwork.style_layer_selector import NSTStyleLayer
    return [NSTStyleLayer(*layer) for layer in [
        ('conv1_1', 0.2),
        ('conv2_1', 0.5),
        ('conv3_1', 0.5),]]


@pytest.fixture
def layers_selection(valid_style_layers):
    from artificial_artwork.style_layer_selector import NSTLayersSelection
    return NSTLayersSelection.from_tuples(valid_style_layers(5))



def test_layers_selection(layers_selection, valid_nst_layers_list, invalid_nst_layers_list):
    layers_selection.layers = valid_nst_layers_list
    assert layers_selection.layers == valid_nst_layers_list

    with pytest.raises(ValueError):
        layers_selection.layers = invalid_nst_layers_list

    assert layers_selection[1] == valid_nst_layers_list[1]
    assert dict(layers_selection) == {layer.id: layer for layer in valid_nst_layers_list}


@pytest.fixture(params=[
    [
        ('conv1_1', 0.2),
        ('conv2_1', 0.5),
        ('conv3_1', 0.5),
    ],
    [
        ('conv1_1', 0.2),
        ('conv1_1', 0.2),
        ('conv2_1', 0.6),
    ]
])
def invalid_style_layers_list(request):
    return request.param


def test_invalid_construction(invalid_style_layers_list):
    from artificial_artwork.style_layer_selector import NSTLayersSelection
    with pytest.raises(ValueError):
        _ = NSTLayersSelection.from_tuples(invalid_style_layers_list)


def test_invalid_layer_coefficient():
    from artificial_artwork.style_layer_selector import NSTStyleLayer
    with pytest.raises(ValueError):
        _ = NSTStyleLayer('layer-id', 1.1)
