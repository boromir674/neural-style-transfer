import attr


@attr.s
class ModelParameters:
    # params = attr.ib(default=attr.Factory(load_default_model_parameters))
    params = attr.ib()


@attr.s
class NSTModelDesign:
    network_layers = attr.ib()
    # parameters_loader = attr.ib(default=None, converter=lambda x: ModelParameters(*list(filter(None, [x]))))
    parameters_loader = attr.ib()
