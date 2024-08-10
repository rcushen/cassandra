import pytest
import numpy as np

from cassandra.core import Factor

@pytest.fixture
def simple_factor():
    return Factor(['A', 'B'], np.array([[0.1, 0.2], [0.3, 0.4]]))

@pytest.fixture
def complex_factor():
    return Factor(['A', 'B', 'C'], np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]))

# __init__
## validation
def test_init_invalid_variables():
    with pytest.raises(ValueError):
        Factor('factor', np.array([[0.1, 0.2], [0.3, 0.4]]))

def test_init_invalid_variable_types():
    with pytest.raises(ValueError):
        Factor([1, 2], np.array([[0.1, 0.2], [0.3, 0.4]]))

def test_init_invalid_probabilities():
    with pytest.raises(ValueError):
        Factor(['A', 'B'], 'probabilities')

def test_init_invalid_probability_types():
    with pytest.raises(ValueError):
        Factor(['A', 'B'], np.array([[1, 2], [3, 4]]))

def test_init_variables_nonunique():
    with pytest.raises(ValueError):
        Factor(['A', 'A'], np.array([[0.1, 0.2], [0.3, 0.4]]))

def test_init_probabilities_wrong_shape():
    with pytest.raises(ValueError):
        Factor(['A', 'B'], np.array([0.1, 0.9]))

# evaluate
## validation
def test_evaluate_invalid_assignment_type(simple_factor):
    with pytest.raises(ValueError, match="The assignments must be a dictionary"):
        simple_factor.evaluate(['A', 'B'])

def test_evaluate_invalid_variable_type(simple_factor):
    with pytest.raises(ValueError, match="The keys of the assignments must be strings"):
        simple_factor.evaluate({1: 0, 'B': 1})

def test_evaluate_invalid_value_type(simple_factor):
    with pytest.raises(ValueError, match="The values of the assignments must be integers"):
        simple_factor.evaluate({'A': 0.5, 'B': 1})

def test_evaluate_missing_variable(simple_factor):
    with pytest.raises(ValueError, match="The assignments are invalid"):
        simple_factor.evaluate({'A': 0})

def test_evaluate_extra_variable(simple_factor):
    with pytest.raises(ValueError, match="The assignments are invalid"):
        simple_factor.evaluate({'A': 0, 'B': 1, 'C': 2})

def test_evaluate_out_of_bounds_assignment(simple_factor):
    with pytest.raises(ValueError, match="The value 2 is invalid for variable A"):
        simple_factor.evaluate({'A': 2, 'B': 0})

def test_evaluate_negative_assignment(simple_factor):
    with pytest.raises(ValueError, match="The value -1 is invalid for variable B"):
        simple_factor.evaluate({'A': 0, 'B': -1})

## correctness
def test_evaluate_simple():
    factor = Factor(['A', 'B'], np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert factor.evaluate({'A': 0, 'B': 0}) == 0.1

# multiply
## validation
def test_multiply_invalid_factor_type(simple_factor):
    with pytest.raises(ValueError, match="The other factor must be a Factor"):
        simple_factor.multiply('factor')

## correctness
def test_multiply_simple(simple_factor):
    result = simple_factor.multiply(simple_factor)
    assert result.scope == ['A', 'B']
    expected_results = np.array([[0.01, 0.04], [0.09, 0.16]])
    np.testing.assert_array_almost_equal(result.values, expected_results)

def test_multiply_complex(simple_factor, complex_factor):
    result = simple_factor.multiply(complex_factor)
    assert result.scope == ['A', 'B', 'C']
    print(result.values)
    expected_results = np.array([[[0.01, 0.02], [0.06, 0.08]], [[0.15, 0.18], [0.28, 0.32]]])
    np.testing.assert_array_almost_equal(result.values, expected_results)

# sum_out
## validation
def test_sum_out_invalid_variable(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.sum_out('C')

def test_sum_out_trivial():
    factor = Factor(['A'], np.array([0.1, 0.9]))
    with pytest.raises(ValueError):
        factor.sum_out('A')

## correctness
def test_sum_out_simple(simple_factor):
    result = simple_factor.sum_out('A')
    assert result.scope == ['B']
    np.testing.assert_array_almost_equal(result.values, np.array([0.4, 0.6]))

def test_sum_out_complex(complex_factor):
    result = complex_factor.sum_out('B')
    assert result.scope == ['A', 'C']
    expected_results = np.array([[0.4, 0.6], [1.2, 1.4]])
    np.testing.assert_array_almost_equal(result.values, expected_results)

# normalise
## correctness
def test_normalise_simple(simple_factor):
    simple_factor.normalise()
    np.testing.assert_array_almost_equal(simple_factor.values, np.array([[0.1, 0.2], [0.3, 0.4]]))

