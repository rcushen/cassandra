import pytest
import numpy as np

from cassandra.core import Node, Network


# __init__
def test_init_invalid_nodes():
    with pytest.raises(ValueError):
        Network("network")


def test_init_invalid_node_types():
    with pytest.raises(ValueError):
        Network([1, 2, 3])


def test_init_nodes_unique(simple_nodes):
    with pytest.raises(ValueError):
        Network(list(simple_nodes) + list(simple_nodes))


def test_init_closed_network(simple_nodes):
    _, node_b, node_c = simple_nodes
    with pytest.raises(ValueError):
        Network([node_b, node_c])


# get_cardinality
def test_get_cardinality(simple_nodes):
    network = Network(list(simple_nodes))
    assert network.get_cardinality() == 3


# get_variable_names
def test_get_variable_names(simple_nodes):
    network = Network(list(simple_nodes))
    assert network.get_variable_names() == set(["A", "B", "C"])


# joint_probability
## validation
def test_joint_probability_bad_input(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(TypeError):
        network.evaluate_joint_probability(["A", "B", "C"])


def test_joint_probability_missing_inputs(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.evaluate_joint_probability({"A": 0, "B": 1})


def test_joint_probability_invalid_values(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.evaluate_joint_probability({"A": 0, "B": 1, "C": 2})


## correctness
def test_joint_probability_simple(simple_nodes):
    network = Network(list(simple_nodes))
    # Assignment {A=0, B=0, C=0}:
    # P(A=0) = 0.6
    # P(B=0|A=0) = 0.7
    # P(C=0|A=0, B=0) = 0.9
    # => P(A=0, B=0, C=0) = 0.6 * 0.7 * 0.9 = 0.378
    assert np.isclose(
        network.evaluate_joint_probability({"A": 0, "B": 0, "C": 0}), 0.378
    )
    # Assignment {A=1, B=0, C=1}:
    # P(A=1) = 0.4
    # P(B=0|A=1) = 0.2
    # P(C=1|A=1, B=0) = 0.7
    # => P(A=0, B=0, C=0) = 0.4 * 0.2 * 0.7 = 0.056
    assert np.isclose(
        network.evaluate_joint_probability({"A": 1, "B": 0, "C": 1}), 0.056
    )


def test_joint_probability_complex(complex_nodes):
    network = Network(list(complex_nodes))
    # Assignment {A=0, B=0, C=0, D=1}:
    # P(A=0) = 0.6
    # P(B=0|A=0) = 0.7
    # P(C=0|A=0, B=0) = 0.9
    # P(D=1|B=0, C=0) = 0.2
    # => P(A=0, B=0, C=0, D=1) = 0.6 * 0.7 * 0.9 * 0.2 = 0.0756
    assert np.isclose(
        network.evaluate_joint_probability({"A": 0, "B": 0, "C": 0, "D": 1}), 0.0756
    )
    network = Network(list(complex_nodes))
    # Assignment {A=1, B=0, C=1, D=0}:
    # P(A=1) = 0.4
    # P(B=0|A=1) = 0.2
    # P(C=1|A=1, B=0) = 0.7
    # P(D=0|B=0, C=1) = 0.4
    # => P(A=0, B=0, C=0, D=1) = 0.4 * 0.2 * 0.7 * 0.4 = 0.0224
    assert np.isclose(
        network.evaluate_joint_probability({"A": 1, "B": 0, "C": 1, "D": 0}), 0.0224
    )


# query
## validation
def test_query_incorrect_input_types(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(TypeError):
        network.query("A", "B")
    with pytest.raises(TypeError):
        network.query("A", {"B": 1})
    with pytest.raises(TypeError):
        network.query({"A": 2}, "B")
    with pytest.raises(TypeError):
        network.query({"A": "A"}, {"B": 1})
    with pytest.raises(TypeError):
        network.query({"A": 1}, {"B": "B"})


def test_query_no_query_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({}, {"A": 0})


def test_query_overlapping_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({"A": 1}, {"A": 0})


def test_query_invalid_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({"A": 1}, {"X": 1})
    with pytest.raises(ValueError):
        network.query({"A": 1, "X": 2}, {"B": 1})


## correctness
# def test_sequential_network_1(sequential_network):
#     prob = sequential_network.query({"D": 1})
#     assert np.isclose(prob, 0.5)

# def test_sequential_network_2(sequential_network):
#     prob = sequential_network.query({"D": 1}, {"A": 0})
#     assert np.isclose(prob, 0.2)
