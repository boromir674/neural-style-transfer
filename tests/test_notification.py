import pytest


@pytest.fixture
def subject():
    from artificial_artwork.utils.notification import Subject
    return Subject


@pytest.fixture
def observer():
    from artificial_artwork.utils.notification import Observer
    return Observer


def test_observers_sanity_test1(subject):
    subject1 = subject([])
    subject2 = subject([])
    assert hasattr(subject1, '_observers')
    assert hasattr(subject2, '_observers')
    assert id(subject1._observers) != id(subject2._observers)


def test_observer_as_constructor(observer):
    with pytest.raises(TypeError) as instantiation_from_interface_error:
        observer_instance = observer()
    assert "Can't instantiate abstract class Observer with abstract methods update" in str(instantiation_from_interface_error.value)


def test_scenario(subject, observer):
# The client code.

    print("------ Scenario 1 ------\n")
    class ObserverA(observer):
        def update(self, a_subject) -> None:
            print("ObserverA: Reacted to the event")

    s1 = subject([])
    o1 = ObserverA()
    s1.attach(o1)

    # business logic
    s1.state = 0
    s1.notify()

    print("------ Scenario 2 ------\n")
    # example 2
    class Businessubject(subject):

        def some_business_logic(self) -> None:
            """
            Usually, the subscription logic is only a fraction of what a Subject can
            really do. Subjects commonly hold some important business logic, that
            triggers a notification method whenever something important is about to
            happen (or after it).
            """
            print("\nSubject: I'm doing something important.")
            from random import randrange
            self._state = randrange(0, 10)
            print(f"Subject: My state has just changed to: {self._state}")
            self.notify()

    class ObserverB(observer):
        def update(self, a_subject) -> None:
            if a_subject.state == 0 or a_subject.state >= 2:
                print("ObserverB: Reacted to the event")

    s2 = Businessubject([])
    assert id(s1) != id(s2)
    assert id(s1._observers) != id(s2._observers)
    o1, o2 = ObserverA(), ObserverB()
    # s2.attach(o1)
    # s2.attach(o2)
    s2.add(o1, o2)
    # business logic
    print(s2._observers)
    s2.some_business_logic()
    s2.some_business_logic()

    s2.detach(o1)
    s2.some_business_logic()
