import pytest
import numpy as np

from cassandra.core import Node, Network

@pytest.fixture
def simple_nodes():
    node_a = Node("A", [], np.array([0.6, 0.4]))
    node_b = Node("B", [node_a], np.array([[0.7, 0.3], [0.2, 0.8]]))
    node_c = Node("C", [node_a, node_b], np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.3, 0.7], [0.1, 0.9]]]))
    return node_a, node_b, node_c

# __init__
def test_init_invalid_nodes():
    with pytest.raises(ValueError):
        Network('network')

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
    assert network.get_variable_names() == set(['A', 'B', 'C'])

