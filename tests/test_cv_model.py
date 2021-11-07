import os
import pytest


my_dir = os.path.dirname(os.path.realpath(__file__))

IMAGE_MODEL_FILE_NAME = 'imagenet-vgg-verydeep-19.mat'


@pytest.fixture
def production_image_model():
    return os.path.join(my_dir, '..', IMAGE_MODEL_FILE_NAME)


@pytest.fixture
def load_model(default_image_processing_config):
    from neural_style_transfer.model_loader import load_vgg_model
    return lambda model_path: load_vgg_model(model_path, default_image_processing_config)


@pytest.mark.xfail(not os.path.isfile(os.path.join(my_dir, '..', IMAGE_MODEL_FILE_NAME)),
    reason="No file found to load the pretrained image (cv) model.")
def test_pretrained_model(load_model, production_image_model):
    _ = load_model(production_image_model)
