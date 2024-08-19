import pytest
import numpy as np

from cassandra.core import Node


# __init__
## validation
def test__init__invalid_cpd(simple_nodes):
    with pytest.raises(ValueError):
        Node("D", [], [0.5, 0.5])

def test__init__invalid_variable_name():
    with pytest.raises(ValueError):
        Node(123, [], np.array([0.5, 0.5]))

def test__init__invalid_parent_nodes():
    with pytest.raises(ValueError):
        Node("D", "not a list", np.array([0.5, 0.5]))

def test__init__inconsistent_cpd(simple_nodes):
    node_a, _, _ = simple_nodes
    with pytest.raises(ValueError):
        Node("D", [node_a], np.array([0.5, 0.5]))

def test__init__invalid_probability_distribution():
    with pytest.raises(ValueError):
        Node("D", [], np.array([0.6, 0.5]))

def test__init__long_variable_name():
    with pytest.raises(ValueError):
        Node("A" * 21, [], np.array([0.5, 0.5]))


# __repr__
## correctness
def test__repr__simple(simple_nodes):
    node_a, _, _ = simple_nodes
    repr_str = node_a.__repr__()
    assert repr_str == "Node('A', parents=[], states=2)"


# get_cardinality
## correctness
def test__get_cardinality__simple(simple_nodes):
    node_a, node_b, node_c = simple_nodes
    assert node_a.get_cardinality() == 2
    assert node_b.get_cardinality() == 2
    assert node_c.get_cardinality() == 2


# get_conditional_distribution
## validation
def test__get_conditional_distribution__invalid_input(simple_nodes):
    _, node_b, _ = simple_nodes
    with pytest.raises(ValueError):
        node_b.get_conditional_distribution({"A": "not an int"})

def test__get_conditional_distribution__missing_parent(simple_nodes):
    _, _, node_c = simple_nodes
    with pytest.raises(ValueError):
        node_c.get_conditional_distribution({"A": 0})

def test__get_conditional_distribution__invalid_state(simple_nodes):
    _, node_b, _ = simple_nodes
    with pytest.raises(ValueError):
        node_b.get_conditional_distribution({"A": 2})

def test__get_conditional_distribution__extra_parent(simple_nodes):
    _, _, node_c = simple_nodes
    with pytest.raises(ValueError):
        node_c.get_conditional_distribution({"A": 1, "B": 0, "D": 0})

## correctness
def test__get_conditional_distribution__no_parents(simple_nodes):
    node_a, _, _ = simple_nodes
    np.testing.assert_array_equal(
        node_a.get_conditional_distribution({}), np.array([0.6, 0.4])
    )

def test__get_conditional_distribution__with_parents(simple_nodes):
    _, node_b, node_c = simple_nodes
    np.testing.assert_array_equal(
        node_b.get_conditional_distribution({"A": 0}), np.array([0.7, 0.3])
    )
    np.testing.assert_array_equal(
        node_c.get_conditional_distribution({"A": 1, "B": 0}), np.array([0.3, 0.7])
    )


# compute_conditional_probability
## validation
def test__compute_conditional_probability__invalid_input(simple_nodes):
    node_a, _, _ = simple_nodes
    with pytest.raises(ValueError):
        node_a.compute_conditional_probability("not an int", {})

def test__compute_conditional_probability__invalid_state(simple_nodes):
    node_a, _, _ = simple_nodes
    with pytest.raises(ValueError):
        node_a.compute_conditional_probability(2, {})

def test__compute_conditional_probability__extra_parent(simple_nodes):
    _, node_b, _ = simple_nodes
    with pytest.raises(ValueError):
        node_b.compute_conditional_probability(1, {"A": 0, "C": 1})

def test__compute_conditional_probability__missing_parent(simple_nodes):
    _, node_b, _ = simple_nodes
    with pytest.raises(ValueError):
        node_b.compute_conditional_probability(1, {})

## correctness
def test__compute_conditional_probability__simple(simple_nodes):
    node_a, node_b, node_c = simple_nodes

    assert node_a.compute_conditional_probability(0, {}) == pytest.approx(0.6)
    assert node_a.compute_conditional_probability(1, {}) == pytest.approx(0.4)

    assert node_b.compute_conditional_probability(0, {"A": 0}) == pytest.approx(0.7)
    assert node_b.compute_conditional_probability(1, {"A": 0}) == pytest.approx(0.3)
    assert node_b.compute_conditional_probability(0, {"A": 1}) == pytest.approx(0.2)
    assert node_b.compute_conditional_probability(1, {"A": 1}) == pytest.approx(0.8)

    assert node_c.compute_conditional_probability(0, {"A": 0, "B": 0}) == pytest.approx(
        0.9
    )
    assert node_c.compute_conditional_probability(1, {"A": 0, "B": 0}) == pytest.approx(
        0.1
    )
    assert node_c.compute_conditional_probability(0, {"A": 0, "B": 1}) == pytest.approx(
        0.5
    )
    assert node_c.compute_conditional_probability(1, {"A": 0, "B": 1}) == pytest.approx(
        0.5
    )
    assert node_c.compute_conditional_probability(0, {"A": 1, "B": 0}) == pytest.approx(
        0.3
    )
    assert node_c.compute_conditional_probability(1, {"A": 1, "B": 0}) == pytest.approx(
        0.7
    )
    assert node_c.compute_conditional_probability(0, {"A": 1, "B": 1}) == pytest.approx(
        0.1
    )
    assert node_c.compute_conditional_probability(1, {"A": 1, "B": 1}) == pytest.approx(
        0.9
    )
