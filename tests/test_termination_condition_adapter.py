import pytest

@pytest.fixture
def termination_condition_adapter(termination_condition):
    from artificial_artwork.termination_condition_adapter import TerminationConditionAdapterFactory
    termination_condition_instance = termination_condition('max-iterations', 4)
    return TerminationConditionAdapterFactory.create('max-iterations', termination_condition_instance)



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
