from abc import ABC
import attr


class AlgorithmInterface(ABC):
    """An algorithm is a series of execution steps, aiming to solve a problem.

    This interface provides a 'run' method that sequentially runs the
    algorithm's steps.
    """
    def run(self, *args, **kwargs) -> None:
        """Run the algorithm."""
        raise NotImplementedError


class IterativeAlgorithm(AlgorithmInterface):

    def run(self, *args, **kwargs) -> None:
        pass


class LearningAlgorithm(IterativeAlgorithm):

    def run(self, *args, **kwargs) -> None:
        pass

    def compute_cost(self, *args, **kwargs) -> float:
        raise NotImplementedError




@attr.s
class NSTAlgorithm(IterativeAlgorithm):
    



@attr.s
class AlogirthmParameters:
    termination_condition = attr.ib()
    


class TerminationCondition:

    def satisfied(self, progress) -> bool:
        raise NotImplementedError


class MaxIterations(TerminationCondition):
    max_iterations: int = attr.ib()

    def satisfied(self, progress) -> bool:
        return self.max_iterations <= progress.iterations


class Convergence(TerminationCondition):
    min_improvement: float = attr.ib()

    def satisfied(self, progress) -> bool:
        return progress.last_improvement < self.min_improvement


class TimeLimit(TerminationCondition):
    time_limit: float = attr.ib()

    def satisfied(self, progress) -> bool:
        return self.time_limit < progress.time



@attr.s
class Progress:
    data = attr.ib()
