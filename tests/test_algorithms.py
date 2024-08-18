import pytest
import numpy as np

from cassandra.core import Factor, sum_product_eliminate


# sum_product_eliminate
## validation
def test__sum_product_eliminate__invalid_factors():
    with pytest.raises(ValueError):
        sum_product_eliminate(["factor", "factor"], "A")


## correctness
def test__sum_product_eliminate__case_1_1(simple_factor):
    factors = {simple_factor}
    result = sum_product_eliminate(factors, "A")

    resulting_single_factor = list(result)[0]

    assert len(result) == 1
    assert resulting_single_factor.scope == ["B"]

    expected_values = np.array([0.4, 0.6])
    np.testing.assert_array_almost_equal(
        resulting_single_factor.values, expected_values
    )


def test__sum_product_eliminate__case_1_2(simple_factor, complex_factor):
    factors = {simple_factor, complex_factor}
    result = sum_product_eliminate(factors, "A")

    resulting_single_factor = list(result)[0]

    assert len(result) == 1
    assert resulting_single_factor.scope == ["B", "C"]

    expected_values = np.array([[0.16, 0.2], [0.34, 0.4]])
    np.testing.assert_array_almost_equal(
        resulting_single_factor.values, expected_values
    )


def test__sum_product_eliminate__case_2_1(simple_nodes):
    factors = {node.to_factor() for node in simple_nodes}
    result = sum_product_eliminate(factors, "C")

    node_a, node_b, _ = simple_nodes
    expected_result = {
        node_a.to_factor(),
        node_b.to_factor(),
        Factor(["A", "B"], np.array([[1, 1], [1, 1]], dtype=np.float64)),
    }
    assert result == expected_result


def test__sum_product_eliminate__case_2_1(simple_nodes):
    factors = {node.to_factor() for node in simple_nodes}
    result = sum_product_eliminate(factors, "B")

    node_a, _, _ = simple_nodes
    expected_result = {
        node_a.to_factor(),
        Factor(["A", "C"], np.array([[0.78, 0.22], [0.14, 0.86]], dtype=np.float64)),
    }
    assert result == expected_result


def test__sum_product_eliminate__case_2_3(simple_nodes):
    factors = {node.to_factor() for node in simple_nodes}
    result = sum_product_eliminate(factors, "A")

    expected_result = {Factor(["B", "C"], np.array([[0.402, 0.098], [0.122, 0.378]]))}

    assert result == expected_result
