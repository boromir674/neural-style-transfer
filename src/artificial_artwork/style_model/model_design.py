import attr


@attr.s
class ModelParameters:
    params = attr.ib()


@attr.s
class NSTModelDesign:
    network_layers = attr.ib()
    parameters_loader = attr.ib()


@attr.s
class ModelDesign:
    pretrained_model = attr.ib()  # model handler instance
    # model_routines: PretrainedModelRoutines = attr.ib()
    # environment_variable: str = attr.ib()
    network_design = attr.ib()  
    # network_layers = attr.ib()
    # style_layers = attr.ib()
    # output_layer = attr.ib()

    # @classmethod
    # def from_layers(cls, layers, network_design):
    #     model_handler = ModelHandlerFacility.create('vgg')
    #     model_handler.reporter = layers
    #     return ModelDesign(
    #         model_handler,
    #         network_design
    #     )