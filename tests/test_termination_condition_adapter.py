import pytest


@pytest.fixture
def termination_condition_adapter(termination_condition):
    from artificial_artwork.termination_condition_adapter_factory import (
        TerminationConditionAdapterFactory,
    )

    return TerminationConditionAdapterFactory.create("max-iterations", 4)


@pytest.fixture
def test_objects(broadcaster_class, termination_condition_adapter):
    from software_patterns import Subject

    return type(
        "O",
        (),
        {
            "broadcaster": broadcaster_class(
                Subject(), lambda: termination_condition_adapter.satisfied
            ),
            "adapter": termination_condition_adapter,
        },
    )


def test_adapter(test_objects):
    # subscribe adapter to broadcaster
    test_objects.broadcaster.subject.attach(test_objects.adapter)

    # iterate
    iterations = test_objects.broadcaster.iterate()

    # assert iterations completed are the expected nb of iterations
    assert iterations == 4
