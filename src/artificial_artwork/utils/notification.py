"""Typical subject/observers pattern implementation. You can see this pattern
mentioned also as event/notification or broadcast/listeners.

Provides the Observer class, serving as the interface that needs to be implemented by concrete classes; the update
method needs to be overrode. Concrete Observers react to the notifications/updates issued by the Subject they had been
attached to.

Provides the Subject class, serving with mechanisms to subscribe/unsubscribe (attach/detach) observers and also with a
method to "notify" all subscribers about events.
"""

from abc import ABC, abstractmethod
from typing import List

__all__ = ['Subject', 'Observer']


class ObserverInterface(ABC):
    """The Observer interface declares the update method, used by subjects.

    Enables objects to act as "event" listeners; react to "notifications"
    by executing specific handling logic.
    """
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Receive an update (from a subject); handle an event notification."""
        raise NotImplementedError


class SubjectInterface(ABC):
    """The Subject interface declares a set of methods for managing subscribers.

    Enables objects to act as "subjects of observations"; notify the subscribed observers/listeners.
    """

    @abstractmethod
    def attach(self, observer: ObserverInterface) -> None:
        """Attach an observer to the subject; subscribe the observer."""
        raise NotImplementedError

    @abstractmethod
    def detach(self, observer: ObserverInterface) -> None:
        """Detach an observer from the subject; unsubscribe the observer."""
        raise NotImplementedError

    @abstractmethod
    def notify(self) -> None:
        """Notify all observers about an event."""
        raise NotImplementedError


class Observer(ObserverInterface, ABC):
    pass


class Subject(SubjectInterface):
    """The Subject owns some important state and can notify observers.

    Both the _state and _observers attributes have a simple implementation,
    but can be overrode to accommodate for more complex scenarios.

    The observers/subscribers are implemented as a python list.
    In more complex scenarios, the list of subscribers can
    be stored more comprehensively (categorized by event type, etc.).


    The subscription management methods provided are 'attach' and 'detach'
    to add or remove a subscriber respectively
    """
    def __init__(self, *args, **kwargs):
        self._observers: List[ObserverInterface] = []
        self._state = None

    def add(self, *observers):
        """Subscribe multiple observers at once."""
        self._observers.extend(list(observers))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def attach(self, observer: ObserverInterface) -> None:
        self._observers.append(observer)

    def detach(self, observer: ObserverInterface) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        """Trigger an update in each subscriber/observer."""
        for observer in self._observers:
            observer.update(self)
