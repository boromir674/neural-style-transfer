import pytest

@pytest.fixture
def termination_condition_adapter(termination_condition):
    from artificial_artwork.termination_condition_adapter import TerminationConditionAdapterFactory
    termination_condition_instance = termination_condition('max-iterations', 4)
    return TerminationConditionAdapterFactory.create('max-iterations', termination_condition_instance)


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
def test_objects(broadcaster_class, termination_condition_adapter):
    from artificial_artwork.utils.notification import Subject
    return type('O', (), {
        'broadcaster': broadcaster_class(Subject(), lambda: termination_condition_adapter.satisfied),
        'adapter': termination_condition_adapter
    })


def test_adapter(test_objects):
    # subscribe adapter to broadcaster
    test_objects.broadcaster.subject.attach(test_objects.adapter)

    # select a loop function (from test implementations)

    # iterate
    iterations = test_objects.broadcaster.iterate()

    # assert iterations completed are the expected nb of oiterations
    assert iterations == 4
