import attr


@attr.s
class ModelParameters:
    params = attr.ib()


@attr.s
class NSTModelDesign:
    network_layers = attr.ib()
    parameters_loader = attr.ib()
