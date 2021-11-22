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
        return os.path.join(test_suite, 'data', file_name)
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
    class MySession():
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


# @pytest.fixture
# def default_image_processing_config():
#     from artificial_artwork.image import ImageProcessingConfig
#     return ImageProcessingConfig.from_image_dimensions()


@pytest.fixture
def image_factory():
    """Production Image Factory.
    
    Exposes the 'from_disk(file_path, preprocess=True)'.

    Returns:
        ImageFactory: an instance of the ImageFactory class
    """
    from artificial_artwork.image.image_factory import ImageFactory
    from artificial_artwork.disk_operations import Disk
    return ImageFactory(Disk.load_image)


@pytest.fixture
def termination_condition_module():
    from artificial_artwork.termination_condition.termination_condition import TerminationConditionFacility, \
        TerminationConditionInterface, MaxIterations, TimeLimit, Convergence

    # all tests require that the Facility already contains some implementations of TerminationCondition
    assert TerminationConditionFacility.class_registry.subclasses == {
        'max-iterations': MaxIterations,
        'time-limit': TimeLimit,
        'convergence': Convergence,
    }
    return type('M', (), {
        'facility': TerminationConditionFacility,
        'interface': TerminationConditionInterface,
    })


@pytest.fixture
def termination_condition(termination_condition_module):
    def create_termination_condition(term_cond_type: str, *args, **kwargs) -> termination_condition_module.interface:
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
                print('Iteration with index', i)

                # notify when we have completed i+1 iterations
                self.subject.state = type('Subject', (), {
                    'metrics': {'iterations': i + 1},  # we have completed i+1 iterations
                }) 
                self.subject.notify()
                i += 1
            return i

    return TestSubject



@pytest.fixture
def toy_pre_trained_model():
    model_layers = (
        # (0, 'conv'),
        # (1, 'relu'),
        # (2, 'maxpool'),
        'conv1_1',
        'relu1',
        'maxpool1',
    )
    # layers we pick to use for our Neural Network
    network_layers = ('conv1_1',)

    # layers we pick that capture the 'style' features of an image
    # we place equal weights on each layer when computing the weighted 'style cost'
    weight = 1.0 / len(network_layers)
    style_layers = [(layer_id, weight) for layer_id in network_layers]
    import numpy as np
    from functools import reduce
    convo_w_weights_shape = (3, 3, 3, 4)
    def load_parameters():
        return {
            'layers': [[
                [[[['conv1_1'], 'unused', [[
                    np.reshape(np.array([i for i in range(1, reduce(lambda i,j: i*j, convo_w_weights_shape)+1)], dtype=np.float32), convo_w_weights_shape),
                    np.array([5], dtype=np.float32)
                ]]]]],
                [[[['relu1'], 'unused', [['W', 'b']]]]],
                [[[['maxpool1'], 'unused', [['W', 'b']]]]],
            ]]
        }
    return {
        'parameters_loader': load_parameters,
        'model_layers': model_layers,
        'network_layers': network_layers,
        'style_layers': style_layers,
    }


@pytest.fixture
def image_manager_class():
    from artificial_artwork.nst_image import ImageManager
    return ImageManager


@pytest.fixture
def pre_trained_models(toy_pre_trained_model):
    
    from artificial_artwork.pretrained_model.model_loader import load_default_model_parameters
    from artificial_artwork.pretrained_model.vgg_architecture import LAYERS as vgg_layers
    from artificial_artwork.pretrained_model.image_model import LAYERS as picked_layers
    from artificial_artwork.cli import STYLE_LAYERS

    return {
        # Production vgg pretrained model
        'vgg': type('PretrainedModel', (), {
            'parameters_loader': load_default_model_parameters,
            'vgg_layers': vgg_layers,
            'network_layers': picked_layers,
            'style_layers': STYLE_LAYERS,
            'output_layer': 'conv4_2'
        }),
        # Toy simulated pretrained model for (mock) testing
        'toy': type('PretrainedModel', (), {
            'parameters_loader': toy_pre_trained_model['parameters_loader'],
            'vgg_layers': toy_pre_trained_model['model_layers'],
            'network_layers': toy_pre_trained_model['network_layers'],
            'style_layers': toy_pre_trained_model['style_layers'],
            'output_layer': 'conv1_1'
        }),
    }

import os
PRODUCTION_IMAGE_MODEL = os.environ.get('AA_VGG_19', 'PRETRAINED_MODEL_NOT_FOUND')


@pytest.fixture
def pre_trained_model(pre_trained_models):
    import os
    return {
        True: pre_trained_models['vgg'],
        False: pre_trained_models['toy'],
    }[os.path.isfile(PRODUCTION_IMAGE_MODEL)]
