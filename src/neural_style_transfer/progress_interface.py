from abc import ABC


class ProgressInterface(ABC):
    """The progress made by an iterative algorithm.

    Args:
        ABC ([type]): [description]
    """
    iterations: int
    duration: float
    cost_improvement: float
