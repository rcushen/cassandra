import pytest
import numpy as np

from cassandra.core import Node, Network, Factor

# Factors
@pytest.fixture
def simple_factor():
    return Factor(["A", "B"], np.array([[0.1, 0.2], [0.3, 0.4]]))

@pytest.fixture
def complex_factor():
    return Factor(
        ["A", "B", "C"], np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    )

# Nodes
@pytest.fixture
def simple_nodes():
    """
    A simple collection of three nodes, with limited dependencies
    """
    # P(A=0) = 0.6
    # P(A=1) = 0.4
    node_a = Node("A", [], np.array([0.6, 0.4]))

    # P(B=0|A=0) = 0.7
    # P(B=1|A=0) = 0.3
    # P(B=0|A=1) = 0.2
    # P(B=1|A=1) = 0.8
    node_b = Node("B", [node_a], np.array([[0.7, 0.3], [0.2, 0.8]]))

    # P(C=0|A=0, B=0) = 0.9
    # P(C=1|A=0, B=0) = 0.1
    # P(C=0|A=0, B=1) = 0.5
    # P(C=1|A=0, B=1) = 0.5
    # P(C=0|A=1, B=0) = 0.3
    # P(C=1|A=1, B=0) = 0.7
    # P(C=0|A=1, B=1) = 0.1
    # P(C=1|A=1, B=1) = 0.9
    node_c = Node(
        "C",
        [node_a, node_b],
        np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.3, 0.7], [0.1, 0.9]]]),
    )

    return node_a, node_b, node_c

@pytest.fixture
def complex_nodes():
    """
    A more complex collection of four nodes with greater dependencies.
    """
    # P(A=0) = 0.6
    # P(A=1) = 0.4
    node_a = Node("A", [], np.array([0.6, 0.4]))

    # P(B=0|A=0) = 0.7
    # P(B=1|A=0) = 0.3
    # P(B=0|A=1) = 0.2
    # P(B=1|A=1) = 0.8
    node_b = Node("B", [node_a], np.array([[0.7, 0.3], [0.2, 0.8]]))

    # P(C=0|A=0, B=0) = 0.9
    # P(C=1|A=0, B=0) = 0.1
    # P(C=0|A=0, B=1) = 0.5
    # P(C=1|A=0, B=1) = 0.5
    # P(C=0|A=1, B=0) = 0.3
    # P(C=1|A=1, B=0) = 0.7
    # P(C=0|A=1, B=1) = 0.1
    # P(C=1|A=1, B=1) = 0.9
    node_c = Node(
        "C",
        [node_a, node_b],
        np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.3, 0.7], [0.1, 0.9]]]),
    )

    # P(D=0|B=0, C=0) = 0.8
    # P(D=1|B=0, C=0) = 0.2
    # P(D=0|B=0, C=1) = 0.4
    # P(D=1|B=0, C=1) = 0.6
    # P(D=0|B=1, C=0) = 0.1
    # P(D=1|B=1, C=0) = 0.9
    # P(D=0|B=1, C=1) = 0.3
    # P(D=1|B=1, C=1) = 0.7
    node_d = Node(
        "D",
        [node_b, node_c],
        np.array([[[0.8, 0.2], [0.4, 0.6]], [[0.1, 0.9], [0.3, 0.7]]]),
    )

    return node_a, node_b, node_c, node_d

# Networks
@pytest.fixture
def simple_network(simple_nodes):
    """
    A network constructed from the set of simple nodes.
    """
    node_a, node_b, node_c = simple_nodes
    simple_network = Network([node_a, node_b, node_c])
    return simple_network

@pytest.fixture
def complex_network(complex_nodes):
    """
    A network constructed from the set of complex nodes.
    """
    node_a, node_b, node_c, node_d = complex_nodes
    complex_network = Network([node_a, node_b, node_c, node_d])
    return complex_network

@pytest.fixture
def sequential_network():
    """
    A four-node network with sequential dependencies.
    """
    # P(A=0) = 0.6
    # P(A=1) = 0.4
    node_a = Node("A", [], np.array([0.6, 0.4]))

    # P(B=0|A=0) = 0.7
    # P(B=1|A=0) = 0.3
    # P(B=0|A=1) = 0.2
    # P(B=1|A=1) = 0.8
    node_b = Node("B", [node_a], np.array([[0.7, 0.3], [0.2, 0.8]]))

    # P(C=0|B=0) = 0.5
    # P(C=1|B=0) = 0.5
    # P(C=0|B=1) = 0.4
    # P(C=1|B=1) = 0.6
    node_c = Node("C", [node_b], np.array([[0.5, 0.5], [0.4, 0.6]]))

    # P(D=0|C=0) = 0.1
    # P(D=1|C=0) = 0.9
    # P(D=0|C=1) = 0.8
    # P(D=1|C=1) = 0.2
    node_d = Node("D", [node_c], np.array([[0.1, 0.9], [0.8, 0.2]]))

    sequential_network = Network([node_a, node_b, node_c, node_d])
    return sequential_network
