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
    from neural_style_transfer.disk_operations import Disk
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
