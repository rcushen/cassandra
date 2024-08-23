import pytest
import numpy as np

from cassandra.core import Node, Network


# __init__
## validation
def test__init__invalid_nodes():
    with pytest.raises(ValueError):
        Network("network")

def test__init__invalid_node_types():
    with pytest.raises(ValueError):
        Network([1, 2, 3])

def test__init__nodes_unique(simple_nodes):
    with pytest.raises(ValueError):
        Network(list(simple_nodes) + list(simple_nodes))

def test__init__closed_network(simple_nodes):
    _, node_b, node_c = simple_nodes
    with pytest.raises(ValueError):
        Network([node_b, node_c])


# get_cardinality
## correctness
def test__get_cardinality__simple(simple_nodes):
    network = Network(list(simple_nodes))
    assert network.get_cardinality() == 3


# get_variable_names
## correctness
def test__get_variable_names__simple(simple_nodes, complex_nodes):
    network = Network(list(simple_nodes))
    assert network.get_variable_names() == set(["A", "B", "C"])
    network = Network(list(complex_nodes))
    assert network.get_variable_names() == set(["A", "B", "C", "D"])


# joint_probability
## validation
def test__joint_probability__bad_input(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(TypeError):
        network.evaluate_joint_probability(["A", "B", "C"])

def test__joint_probability__missing_inputs(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.evaluate_joint_probability({"A": 0, "B": 1})

def test__joint_probability__invalid_values(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.evaluate_joint_probability({"A": 0, "B": 1, "C": 2})

## correctness
def test__joint_probability__simple(simple_nodes):
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

def test__joint_probability__complex(complex_nodes):
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
def test__query__incorrect_input_types(simple_nodes):
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

def test__query__no_query_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({}, {"A": 0})

def test__query__overlapping_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({"A": 1}, {"A": 0})

def test__query__invalid_vars(simple_nodes):
    network = Network(list(simple_nodes))
    with pytest.raises(ValueError):
        network.query({"A": 1}, {"X": 1})
    with pytest.raises(ValueError):
        network.query({"A": 1, "X": 2}, {"B": 1})

## correctness
def test__query__simple_network_1(simple_network):
    prob = simple_network.query({"A": 1})
    assert np.isclose(prob, 0.4)

    prob = simple_network.query({"B": 0})
    assert np.isclose(prob, 0.5)

    prob = simple_network.query({"C": 0})
    assert np.isclose(prob, 0.524)

def test__query__simple_network_2(simple_network):
    prob = simple_network.query({"B": 1}, {"A": 0})
    assert np.isclose(prob, 0.3)

    prob = simple_network.query({"A": 1}, {"B": 0})
    assert np.isclose(prob, 0.16)

    prob = simple_network.query({"B": 0}, {"C": 1})
    assert np.isclose(prob, 0.20588235)

def test__query_simple_network_3(simple_network):
    prob = simple_network.query({"C": 1}, {"A": 0, "B": 1})
    assert np.isclose(prob, 0.5)

    prob = simple_network.query({"B": 1}, {"A": 0, "C": 1})
    assert np.isclose(prob, 0.681818)

# def test_sequential_network_1(sequential_network):
#     prob = sequential_network.query({"D": 1})
#     print(f"prob: {prob}")
#     assert np.isclose(prob, 0.5)

# def test_sequential_network_2(sequential_network):
#     prob = sequential_network.query({"D": 1}, {"A": 0})
#     assert np.isclose(prob, 0.2)

