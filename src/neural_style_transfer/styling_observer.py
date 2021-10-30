from attr import define 
from .utils.notification import Observer
from typing import Callable
import numpy.typing as npt


@define
class StylingObserver(Observer):
    save_on_disk_callback: Callable[[str, npt.NDArray], None]
    """Store a snapshot of the image under construction.

    Args:
        Observer ([type]): [description]
    """
    def update(self, *args, **kwargs):
        self.save_on_disk_callback(*args, **kwargs)
