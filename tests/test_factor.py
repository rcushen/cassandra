import pytest
import numpy as np

from cassandra.core import Factor

# __init__
## validation
def test__init__invalid_scope():
    with pytest.raises(ValueError):
        Factor("factor", np.array([[0.1, 0.2], [0.3, 0.4]]))

def test__init__invalid_scope_types():
    with pytest.raises(ValueError):
        Factor([1, 2], np.array([[0.1, 0.2], [0.3, 0.4]]))

def test__init__invalid_values():
    with pytest.raises(ValueError):
        Factor(["A", "B"], "probabilities")

def test__init__invalid_values_shape():
    with pytest.raises(ValueError):
        Factor(["A", "B"], np.array([[1, 2], [3, 4]]))

def test__init__nonunique_scope():
    with pytest.raises(ValueError):
        Factor(["A", "A"], np.array([[0.1, 0.2], [0.3, 0.4]]))

def test__init__incompatible_dimensions():
    with pytest.raises(ValueError):
        Factor(["A", "B"], np.array([0.1, 0.9]))


# evaluate
## validation
def test__evaluate__invalid_assignment(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate(["A", "B"])

def test__evaluate__invalid_assignment_variable_type(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({1: 0, "B": 1})

def test__evaluate__invalid_assignment_value_type(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({"A": 0.5, "B": 1})

def test__evaluate__missing_variable(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({"A": 0})

def test__evaluate__extra_variable(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({"A": 0, "B": 1, "C": 2})

def test__evaluate__out_of_bounds_assignment(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({"A": 2, "B": 0})

def test__evaluate__negative_assignment(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.evaluate({"A": 0, "B": -1})

## correctness
def test__evaluate__simple():
    factor = Factor(["A", "B"], np.array([[0.1, 0.2], [0.3, 0.4]]))

    expected_result = 0.1
    assert factor.evaluate({"A": 0, "B": 0}) == expected_result

# multiply
## validation
def test__multiply__invalid_factor_type(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.multiply("factor")

## correctness
def test__multiply__simple(simple_factor):
    result = simple_factor.multiply(simple_factor)
    expected_result = Factor(["A", "B"], np.array([[0.01, 0.04], [0.09, 0.16]]))

    assert result == expected_result

def test__multiply__complex(simple_factor, complex_factor):
    result = simple_factor.multiply(complex_factor)
    expected_result = Factor(["A", "B", "C"], np.array(
        [[[0.01, 0.02], [0.06, 0.08]], [[0.15, 0.18], [0.28, 0.32]]]
    ))

    assert result == expected_result

    result = complex_factor.multiply(simple_factor)
    expected_result = Factor(["A", "B", "C"], np.array(
        [[[0.01, 0.02], [0.06, 0.08]], [[0.15, 0.18], [0.28, 0.32]]]
    ))

    assert result == expected_result


# sum_out
## validation
def test__sum_out__invalid_variable(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.sum_out("C")

def test__sum_out__trivial():
    factor = Factor(["A"], np.array([0.1, 0.9]))
    with pytest.raises(ValueError):
        factor.sum_out("A")

## correctness
def test__sum_out__simple(simple_factor):
    result = simple_factor.sum_out("A")
    expected_result = Factor(["B"], np.array([0.4, 0.6]))

    assert result == expected_result

def test__sum_out__complex(complex_factor):
    result = complex_factor.sum_out("B")
    expected_result = Factor(["A", "C"], np.array([[0.4, 0.6], [1.2, 1.4]]))

    assert result == expected_result


# normalise
## validation
def test__normalise__invalid_n_dimensions_type(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.normalise("n_dimensions")

def test__normalise__invalid_n_dimensions(simple_factor):
    with pytest.raises(ValueError):
        simple_factor.normalise(5)

## correctness
def test__normalise__simple(simple_factor):
    simple_factor.normalise()
    row1 = np.divide(np.array([0.1, 0.2]), 0.3)
    row2 = np.divide(np.array([0.3, 0.4]), 0.7)
    expected_result = Factor(["A", "B"], np.array([row1, row2]))

    assert simple_factor == expected_result

def test__normalise__complex(complex_factor):
    complex_factor.normalise(n_dimensions=2)
    row1 = np.divide(np.array([0.1, 0.2]), 1.0)
    row2 = np.divide(np.array([0.3, 0.4]), 1.0)
    row3 = np.divide(np.array([0.5, 0.6]), 2.6)
    row4 = np.divide(np.array([0.7, 0.8]), 2.6)
    expected_result = Factor(["A", "B", "C"], np.array([[row1, row2], [row3, row4]]))

    assert complex_factor == expected_result

