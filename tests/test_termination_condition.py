import pytest


@pytest.fixture
def iterative_loop():
    def _max_iterations_loop(termination_condition):
        i = 0
        while not termination_condition.satisfied(i):
            print(f'_max_iterations_loop {i}')
            i += 1
        return i

    import time
    def _max_time_limit_loop(termination_condition):
        starting_time = time.time()
        i = 0
        duration = 0
        while not termination_condition.satisfied(duration):
            print(f'_max_time_limit_loop {i, duration}')
            duration += time.time() - starting_time
            i += 1
        return i

    def _convergence_loop(termination_condition):
        i = 0
        cost_improvement = 100
        while not termination_condition.satisfied(cost_improvement):
            print(f'_convergence_loop {i}')
            i += 1
            cost_improvement *= 0.1
        return i

    condition_type_2_loop = {
        'max-iterations': _max_iterations_loop,
        'time-limit': _max_time_limit_loop,
        'convergence': _convergence_loop,
    }
    return condition_type_2_loop


@pytest.fixture
def monkeypatch_time(monkeypatch):
    def patch(base_time: float, step: float):
        def __call__(self):
            self.i += 1
            return self.starting_timestamp + self.i * step
        mock_time_callable = type('TimeMock', (), {
            '__call__': __call__,
            'i': -1,
            'starting_timestamp': base_time,
        })()

        import time
        monkeypatch.setattr(time, 'time', mock_time_callable)
    return patch


@pytest.fixture(params=[
    ['max-iterations', [0], 0],
    ['max-iterations', [2], 2],
    ['time-limit', [1], 1],
    ['time-limit', [2], 2],
    ['convergence', [0.01], 5],
])
def termination_condition_data(request, termination_condition, iterative_loop, monkeypatch_time):
    monkeypatch_time(100, 1)
    return type('D', (), {
        'termination_condition': termination_condition(request.param[0], *request.param[1]),
        'loop': iterative_loop[request.param[0]],
        'expected_completed_iterations': request.param[2],
    })

def test_term(termination_condition_data):
    iterations = termination_condition_data.loop(termination_condition_data.termination_condition)
    assert iterations == termination_condition_data.expected_completed_iterations
