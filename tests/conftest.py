import os
import typing as t

import pytest


@pytest.fixture
def test_suite():
    """Path of the test suite directory."""
    import os

    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def test_image(test_suite):
    import os

    def get_image_file_path(file_name):
        return os.path.join(test_suite, "data", file_name)

    return get_image_file_path


@pytest.fixture
def disk():
    from artificial_artwork.disk_operations import Disk

    return Disk


@pytest.fixture
def session():
    """Tensorflow v1 Session, with seed defined at runtime.

    >>> import tensorflow as tf
    >>> with session(2) as test:
    ...  a_C = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    ...  a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    ...  J_content = compute_cost(a_C, a_G)
    ...  assert abs(J_content.eval() - 7.0738883) < 1e-5

    Returns:
        (MySession): A tensorflow session with a set random seed
    """
    import tensorflow as tf

    class MySession:
        def __init__(self, seed):
            tf.compat.v1.reset_default_graph()
            self.tf_session = tf.compat.v1.Session()
            self.seed = seed

        def __enter__(self):
            entering_output = self.tf_session.__enter__()
            tf.compat.v1.set_random_seed(self.seed)
            return entering_output

        def __exit__(self, type, value, traceback):
            # Exception handling here
            self.tf_session.__exit__(type, value, traceback)

    return MySession


@pytest.fixture
def image_factory():
    """Production Image Factory.

    Exposes the 'from_disk(file_path, preprocess=True)'.

    Returns:
        ImageFactory: an instance of the ImageFactory class
    """
    from artificial_artwork.disk_operations import Disk
    from artificial_artwork.image.image_factory import ImageFactory

    return ImageFactory(Disk.load_image)


@pytest.fixture
def termination_condition_module():
    from artificial_artwork.termination_condition.termination_condition import (
        Convergence,
        MaxIterations,
        TerminationConditionFacility,
        TerminationConditionInterface,
        TimeLimit,
    )

    # all tests require that the Facility already contains some implementations of TerminationCondition
    assert TerminationConditionFacility.class_registry.subclasses == {
        "max-iterations": MaxIterations,
        "time-limit": TimeLimit,
        "convergence": Convergence,
    }
    return type(
        "M",
        (),
        {
            "facility": TerminationConditionFacility,
            "interface": TerminationConditionInterface,
        },
    )


@pytest.fixture
def termination_condition(termination_condition_module):
    def create_termination_condition(
        term_cond_type: str, *args, **kwargs
    ) -> termination_condition_module.interface:
        return termination_condition_module.facility.create(term_cond_type, *args, **kwargs)

    return create_termination_condition


@pytest.fixture
def subscribe():
    def _subscribe(broadcaster, listeners):
        broadcaster.add(*listeners)

    return _subscribe


@pytest.fixture
def broadcaster_class():
    class TestSubject:
        def __init__(self, subject, done_callback):
            self.subject = subject
            self.done = done_callback

        def iterate(self):
            i = 0
            while not self.done():
                # do something in the current iteration
                print("Iteration with index", i)

                # notify when we have completed i+1 iterations
                self.subject.state = type(
                    "Subject",
                    (),
                    {
                        "metrics": {"iterations": i + 1},  # we have completed i+1 iterations
                    },
                )
                self.subject.notify()
                i += 1
            return i

    return TestSubject


@pytest.fixture
def toy_network_design():
    # layers we pick to use for our Neural Network
    network_layers = ("conv1_1",)  # Toy Network has 1 Layer
    weight = 1.0 / len(network_layers)  # equally weight all Style Layers
    # for the Toy Network Design, select all Network Layers to be Style Layers
    style_layers = [(layer_id, weight) for layer_id in network_layers]
    return type(
        "ModelDesign",
        (),
        {
            "network_layers": ("conv1_1",),
            "style_layers": style_layers,
            "output_layer": "conv1_1",
        },
    )


@pytest.fixture
def image_manager_class():
    from artificial_artwork.nst_image import ImageManager

    return ImageManager


## Supported pretrained models and their expected layers


@pytest.fixture
def vgg_layers():
    """Production vgg image model Complete network's layer Architecture."""
    VGG_LAYERS = (
        (0, "conv1_1"),  # (3, 3, 3, 64)
        (1, "relu1_1"),
        (2, "conv1_2"),  # (3, 3, 64, 64)
        (3, "relu1_2"),
        (4, "pool1"),  # maxpool
        (5, "conv2_1"),  # (3, 3, 64, 128)
        (6, "relu2_1"),
        (7, "conv2_2"),  # (3, 3, 128, 128)
        (8, "relu2_2"),
        (9, "pool2"),
        (10, "conv3_1"),  # (3, 3, 128, 256)
        (11, "relu3_1"),
        (12, "conv3_2"),  # (3, 3, 256, 256)
        (13, "relu3_2"),
        (14, "conv3_3"),  # (3, 3, 256, 256)
        (15, "relu3_3"),
        (16, "conv3_4"),  # (3, 3, 256, 256)
        (17, "relu3_4"),
        (18, "pool3"),
        (19, "conv4_1"),  # (3, 3, 256, 512)
        (20, "relu4_1"),
        (21, "conv4_2"),  # (3, 3, 512, 512)
        (22, "relu4_2"),
        (23, "conv4_3"),  # (3, 3, 512, 512)
        (24, "relu4_3"),
        (25, "conv4_4"),  # (3, 3, 512, 512)
        (26, "relu4_4"),
        (27, "pool4"),
        (28, "conv5_1"),  # (3, 3, 512, 512)
        (29, "relu5_1"),
        (30, "conv5_2"),  # (3, 3, 512, 512)
        (31, "relu5_2"),
        (32, "conv5_3"),  # (3, 3, 512, 512)
        (33, "relu5_3"),
        (34, "conv5_4"),  # (3, 3, 512, 512)
        (35, "relu5_4"),
        (36, "pool5"),
        (37, "fc6"),  # fullyconnected (7, 7, 512, 4096)
        (38, "relu6"),
        (39, "fc7"),  # fullyconnected (1, 1, 4096, 4096)
        (40, "relu7"),
        (41, "fc8"),  # fullyconnected (1, 1, 4096, 1000)
        (42, "prob"),  # softmax
    )

    return tuple((layer_id for _, layer_id in VGG_LAYERS))


PRODUCTION_IMAGE_MODEL = os.environ.get("AA_VGG_19", "PRETRAINED_MODEL_NOT_FOUND")


@pytest.fixture
def pre_trained_models_1(vgg_layers, toy_model_data, toy_network_design):
    import typing as t

    from numpy.typing import NDArray

    from artificial_artwork.pretrained_model import ModelHandlerFacility
    from artificial_artwork.production_networks import NetworkDesign

    toy_layers_loader: t.Callable[..., NDArray] = toy_model_data[0]
    pretrained_toy_model_layers: t.List[str] = toy_model_data[1]

    # help implement the ModelHandler Interface for the toy handler
    # by using parts of production code, that we don't need to mock
    from artificial_artwork.pre_trained_models.vgg import (
        VggModelHandler,
        VggModelRoutines,
    )

    class ToyModelRoutines(VggModelRoutines):
        # override only critical operations integrating with Prod Pretrained Stored Layers/Weights
        def load_layers(self, file_path: str):
            return toy_layers_loader(file_path)

    toy_model_routines = ToyModelRoutines()

    @ModelHandlerFacility.factory.register_as_subclass("toy")
    class ToyModelHandler(VggModelHandler):
        def _load_model_layers(self):
            return toy_model_routines.load_layers("")["layers"][0]

        @property
        def model_routines(self):
            return toy_model_routines

    return {
        # 'vgg': type('NSTModel', (), {
        #     'pretrained_model': type('PTM', (), {
        #         'expected_layers': vgg_layers,
        #         'id': 'vgg',
        #         'handler': ModelHandlerFacility.create('vgg'),
        #     }),
        #     # Production Style Layers and Output (Content) Layer picked from vgg
        #     'network_design': NetworkDesign.from_default_vgg()
        # }),
        "toy": type(
            "NSTModel",
            (),
            {
                "pretrained_model": type(
                    "PTM",
                    (),
                    {
                        "expected_layers": pretrained_toy_model_layers,  # t.List[str]
                        "id": "toy",
                        "handler": ModelHandlerFacility.create("toy"),
                    },
                ),
                "network_design": NetworkDesign(
                    toy_network_design.network_layers,
                    toy_network_design.style_layers,
                    toy_network_design.output_layer,
                ),
            },
        ),
    }


@pytest.fixture
def model(pre_trained_models_1):
    import os

    print(f"\n -- PROD IM MODEL: {PRODUCTION_IMAGE_MODEL}")
    print(f"Selected Prod?: {os.path.isfile(PRODUCTION_IMAGE_MODEL)}")

    return pre_trained_models_1["toy"]
    # return {
    #     True: pre_trained_models_1['vgg'],
    #     False: pre_trained_models_1['toy'],
    # }[os.path.isfile(PRODUCTION_IMAGE_MODEL)]


# CONSTANT DATA Representing Layers Information (ie weight values) of Toy Network
@pytest.fixture
def toy_model_data():
    from functools import reduce

    import numpy as np

    # This data format emulates the format the production pretrained VGG layer
    # IDs are stored in
    model_layers = (
        "conv1_1",
        "relu1",
        "maxpool1",
    )
    convo_w_weights_shape = (3, 3, 3, 4)

    def load_layers(*args):
        """Load Layers of 3-layered Toy Neural Net, emulating prod VGG format.

        It emulates what the production implementation (scipy.io.loadmat) does,
        by returning an object following the same interface as the one returned
        by scipy.io.loadmat, when called on the file storing the production
        pretrained VGG model.
        """
        # here we use pytest to emit some text, leveraging pytest, so that the test code using this fixture
        # can somehow verify that the text appeared in the expected place (ie console, log or sth)
        print("VGG Mat Weights Mock Loader Called")

        return {
            "layers": [
                [
                    # 1st Layer: conv1_1
                    [
                        [
                            [
                                [model_layers[0]],
                                "unused",
                                [
                                    [
                                        # 'A' Matrix weights tensor with shape (3, 3, 3, 4) (total nb of values = 3*3*3*4 = 108)
                                        # for this toy Conv Layer we set the tensor values to be 1, 2, 3, ... 3 * 3 * 3 * 4 + 1 = 109
                                        np.reshape(
                                            np.array(
                                                [
                                                    i
                                                    for i in range(
                                                        1,
                                                        reduce(
                                                            lambda i, j: i * j,
                                                            convo_w_weights_shape,
                                                        )
                                                        + 1,
                                                    )
                                                ],
                                                dtype=np.float32,
                                            ),
                                            convo_w_weights_shape,
                                        ),
                                        # 'b' bias vector, which here is an array of shape (1,)
                                        # for this toy Conv Layer we set the bias value to be 5
                                        np.array([5], dtype=np.float32),
                                    ]
                                ],
                            ]
                        ]
                    ],
                    # 2nd Layer: relu1
                    [
                        [[[model_layers[1]], "unused", [["W", "b"]]]]
                    ],  # these layer weights are not expected to be used, because the layer is not a Conv layer
                    # 3rd Layer: maxpool1
                    [
                        [[[model_layers[2]], "unused", [["W", "b"]]]]
                    ],  # these layer weights are not expected to be used, because the layer is not a Conv layer
                ]
            ]
        }

    return load_layers, model_layers


# MONKEYPATH PROD NST ALGO at RUNTIME with Algo using Toy Network
@pytest.fixture
def toy_nst_algorithm(toy_model_data, toy_network_design, monkeypatch):
    from numpy.typing import NDArray

    toy_layers_loader: t.Callable[..., NDArray] = toy_model_data[0]
    # pretrained_toy_model_layer_ids: t.List[str] = toy_model_data[1]

    def _monkeypatch():
        return_toy_layers, _ = toy_model_data
        import scipy.io

        # equip Handler Facility Facory with the 'vgg' implementation
        from artificial_artwork.pre_trained_models import vgg
        from artificial_artwork.pretrained_model import ModelHandlerFacility
        from artificial_artwork.production_networks import NetworkDesign

        # if prod VGG Handler tries to load VGG Prod Weights, return Toy Weights instead
        # 1st we patch the scipy.io.loadmat, which is used by the production VGG Handler
        monkeypatch.setattr(scipy.io, "loadmat", return_toy_layers)  # Patch/replace-with-mock

        from artificial_artwork.pre_trained_models.vgg import (
            VggModelHandler,
            VggModelRoutines,
        )

        class ToyModelRoutines(VggModelRoutines):
            # override only critical operations integrating with Prod Pretrained Stored Layers/Weights
            def load_layers(self, file_path: str):
                return toy_layers_loader(file_path)

        toy_model_routines = ToyModelRoutines()

        class ToyModelHandler(VggModelHandler):
            def _load_model_layers(self):
                return toy_model_routines.load_layers("")["layers"][0]

            @property
            def model_routines(self):
                return toy_model_routines

        monkeypatch.setattr(vgg, "VggModelHandler", ToyModelHandler)  # Patch/replace-with-mock

        # 2nd we patch the AA_VGG_19 env var which the code strictly requires to find
        import os

        os.environ["AA_VGG_19"] = "unit-tests-toy-value"  # Patch/replace-with-mock

        # Prod Code uses the 'default' factory (classmetod) method of class
        # NetworkDesign, in order to instantiate a NetworkDesign object
        # according to the 'Original' NST Algorithm (which layers to pick for
        # creating ReLUs from their pretrained Conv A, b weights, or which is the Output Layer)

        # Monkey patching objects used in the 'default' factory method
        monkeypatch.setattr(
            NetworkDesign,
            "from_default_vgg",
            lambda: NetworkDesign(
                toy_network_design.network_layers,  # full list of layer IDs available in Pretrained Model
                toy_network_design.style_layers,  # list of tuples with layer IDs and coefficients governing their proportional contribution to the Style Cost/Loss formula
                toy_network_design.output_layer,  # layer ID to be used for Content Loss (ie last layer of Pretrained Model/Network)
            ),
        )
        # for convenience, construct here a ModelHanlder instance, equiped with
        # handling all operations (of ModelHandlerInterface) with mocked Toy operations
        # when needed and provide it to test code
        # TODO remove the need for that
        toy_model_handler = ModelHandlerFacility.create(
            "vgg"
        )  # handler instances are stateless, and lightweight
        return toy_model_handler

    return _monkeypatch
