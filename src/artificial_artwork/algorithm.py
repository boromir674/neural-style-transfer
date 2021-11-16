import attr

from .style_layer_selector import NSTLayersSelection


@attr.s
class NSTAlgorithm:
    parameters = attr.ib()


@attr.s
class AlogirthmParameters:
    # TODO remove content and style images and output_path
    # retain only algorithm parameters (variables governing how the algo will behave)
    # from the algo input (runtime objects that are the INPUT to the algo)
    content_image = attr.ib()
    style_image = attr.ib()
    style_layers = attr.ib(converter=NSTLayersSelection.from_tuples)
    termination_condition = attr.ib()
    output_path = attr.ib()
