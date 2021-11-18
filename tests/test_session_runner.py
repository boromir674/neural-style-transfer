import pytest


@pytest.fixture
def prod_session_runner():
    """Production Tensorflow Session Runner.
    
    Uses an Interactive Session.
    """
    from artificial_artwork.tf_session_runner import TensorflowSessionRunner
    return TensorflowSessionRunner.with_default_graph_reset()


@pytest.fixture
def test_content_image(image_factory, test_image):
    return image_factory.from_disk(test_image('canoe_water.jpg'))


def test_session_runner_behaviour(prod_session_runner, test_content_image):
    assert prod_session_runner.args_history == []
    # prod_session_runner.run()
