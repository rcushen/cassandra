import pytest
import numpy as np

from cassandra.core import Node

@pytest.fixture
def simple_nodes():
    # P(A=0) = 0.6
    # P(A=1) = 0.4
    node_a = Node("A", [], np.array([0.6, 0.4]))

    # P(B=0|A=0) = 0.7
    # P(B=1|A=0) = 0.3
    # P(B=0|A=1) = 0.2
    # P(B=1|A=1) = 0.8
    node_b = Node("B", [node_a], np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ]))

    # P(C=0|A=0, B=0) = 0.9
    # P(C=1|A=0, B=0) = 0.1
    # P(C=0|A=0, B=1) = 0.5
    # P(C=1|A=0, B=1) = 0.5
    # P(C=0|A=1, B=0) = 0.3
    # P(C=1|A=1, B=0) = 0.7
    # P(C=0|A=1, B=1) = 0.1
    # P(C=1|A=1, B=1) = 0.9
    node_c = Node("C", [node_a, node_b], np.array([
        [
            [0.9, 0.1],
            [0.5, 0.5]
        ],
        [
            [0.3, 0.7],
            [0.1, 0.9]
        ]
    ]))

    return node_a, node_b, node_c

@pytest.fixture
def complex_nodes():
    # P(A=1) = 0.4
    # P(B=0|A=1) = 0.2
    # P(C=0|A=1, B=0) = 0.3
    # P(D=1|B=0, C=0) = 0.9
    # P(A=1, B=0, C=0, D=1) = 0.4 * 0.2 * 0.3 * 0.9 = 0.0216
    node_a = Node("A", [], np.array([0.6, 0.4]))
    node_b = Node("B", [node_a], np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ]))
    node_c = Node("C", [node_a, node_b], np.array([
        [
            [0.9, 0.1],
            [0.5, 0.5]
        ],
        [
            [0.3, 0.7],
            [0.1, 0.9]
        ]
    ]))
    node_d = Node("D", [node_b, node_c], np.array([
        [
            [0.8, 0.2],
            [0.4, 0.6]
        ],
        [
            [0.1, 0.9],
            [0.3, 0.7]
        ]
    ]))
    return node_a, node_b, node_c, node_d
