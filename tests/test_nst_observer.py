import pytest


@pytest.fixture
def nst_observer(disk):
    """An instance object of StylingObserver class."""
    from artificial_artwork.styling_observer import StylingObserver
    from artificial_artwork.image.image_operations import convert_to_uint8
    return StylingObserver(disk.save_image, convert_to_uint8)



@pytest.fixture
def subject_container(tmpdir):
    import numpy as np
    class TestSubject:
        def __init__(self, subject):
            self.subject = subject
        
        def build_state(self, runtime_data):
            return {
                'output_path': tmpdir,
                'content_image_path': 'c',
                'style_image_path': 's',
                'metrics': {
                    'iterations': runtime_data
                },
                'matrix': np.random.randint(0, high=255, size=(30, 40), dtype=np.uint8)
            }

        def notify(self, runtime_data):
            self.subject.state = type('Subject', (), self.build_state(runtime_data))
            self.subject.notify()

    return TestSubject


@pytest.fixture
def styling_observer_data(subject_container, nst_observer):
    from artificial_artwork.utils.notification import Subject
    return type('TestData', (), {
        'broadcaster': subject_container(Subject()),
        'observer': nst_observer,
    })


def test_styling_observer(styling_observer_data, subscribe, tmpdir):
    d = styling_observer_data
    # Subscribe observer/listener to subject/broadcaster at runtime
    subscribe(
        d.broadcaster.subject,
        [d.observer]
    )

    import os
    
    d.broadcaster.notify(1)
    assert os.path.isfile(os.path.join(tmpdir, 'c+s-1.png'))
    d.broadcaster.notify(2)
    assert os.path.isfile(os.path.join(tmpdir, 'c+s-2.png'))
