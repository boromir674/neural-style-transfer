import attr


@attr.s
class NSTAlgorithmProgress:
    tracked_metrics: dict = attr.ib(converter=lambda x: {k: [v] for k, v in dict(x).items()})
    _callbacks = attr.ib(default=attr.Factory(lambda self: {
        True: self._append,
        False: self._set,
    }, takes_self=True))

    def update(self, *args, **kwargs):
        metrics = args[0].state.metrics
        for metric_key, value in metrics.items():
            self._callbacks[metric_key in self.tracked_metrics](metric_key, value)

    def _set(self, key, value):
        self.tracked_metrics[key] = [value]
    
    def _append(self, key, value):
        self.tracked_metrics[key].append(value)

    # Properties

    @property
    def iterations(self):
        """Iterations completed."""
        return self.tracked_metrics.get('iterations', [None])[-1]

    @property
    def duration(self):
        """Time in seconds the iterative algorithm has been running."""
        return self.tracked_metrics.get('duration', [None])[-1]

    @property
    def cost_improvement(self):
        """Difference of loss function between the last 2 measurements.
        
        Positive value indicates that the loss went down and that the learning
        process moved towards the (local) minimum (in terms of minimizing the
        loss/cost function).
        
        So roughly, positive values indicate improvement [moving towards (local)
        minimum] and negative indicate moving away from minimum.

        Moving refers to the learning parameters.
        """
        if 1 < len(self.tracked_metrics.get('cost', [])):
            return self.tracked_metrics['cost'][-2] - self.tracked_metrics['cost'][-1]
