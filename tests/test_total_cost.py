import pytest

@pytest.fixture
def total_cost():
    from artificial_artwork.cost_computer import NSTCostComputer
    return NSTCostComputer.compute


@pytest.mark.parametrize('a, x, b, y', [
    (1, 2, 3, 4),
    (0, 1, 2, 3),
])
def test_total_cost_computation(a, x, b, y, total_cost):
    assert total_cost(x, y, a, b) == a * x + b * y


@pytest.mark.parametrize('seed, a, b', [
    (3, 10, 40),
])
def test_random_total_cost_computation(seed, a, b, total_cost, session):
    import numpy as np
    with session(seed) as test:
        np.random.seed(seed)
        J_content = abs(np.random.randn())
        J_style = abs(np.random.randn())
        J = total_cost(J_content, J_style, alpha=a, beta=b)
        assert abs(J - 35.34667875478276) < 1e-6


@pytest.mark.parametrize('seed', [
    (3,),
])
def test_default_total_cost_computation(seed, total_cost, session):
    import numpy as np

    with session(seed) as test:
        np.random.seed(seed)
        J_content = abs(np.random.randn())
        J_style = abs(np.random.randn())
        J1 = total_cost(J_content, J_style)
        J2 = total_cost(J_content, J_style, alpha=10, beta=40)
        assert J1 == J2
        assert abs(J1 - 65.36223497201107) < 1e-6
