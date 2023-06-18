import pytest

from cassandra.network import Node, NodeType, Factor, BayesianNetwork
from scipy.stats import norm

# Helper functions
def assert_approx_equal(a, b, epsilon=1e-6):
    assert abs(a - b) < epsilon

# Section 1: Node class tests
# Test Node creation with no parent variables (i.e. a root node)
def test_root_node_creation():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )

    assert node.type == NodeType.ROOT
    assert node.variable_name == 'A'
    assert node.domain == (0, 1)
    assert node.parent_variable_names == []

# Test Node creation with parent variables (i.e. a child node)
def test_child_node_creation():
    def equation_func(A: float) -> float:
        return 2 * A

    node = Node('B', (0, 2), ['A'], equation=equation_func)

    assert node.type == NodeType.CHILD
    assert node.variable_name == 'B'
    assert node.domain == (0, 2)
    assert node.parent_variable_names == ['A']
    assert node.equation({'A':1}) == 2

# Test errors when creating improperly specified nodes
def test_error_root_node_creation():
    with pytest.raises(ValueError, match="Root nodes must have a marginal probability distribution and parameters."):
        Node('A', (0, 1), [])

    with pytest.raises(ValueError, match="Root nodes must have a marginal probability distribution and parameters."):
        Node('A', (0, 1), [], marginal_pdf=norm.pdf)

def test_error_child_node_creation():
    with pytest.raises(ValueError, match="Child nodes must have an equation."):
        Node('B', (0, 2), ['A'])

# Test marginal_pdf method
def test_marginal_pdf():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )

    assert_approx_equal(
        node.marginal_pdf(0.5),
        norm.pdf(0.5, loc=0, scale=1)
    )

# Test error when calling marginal_pdf for child node
def test_error_marginal_pdf_child_node():
    def equation_func(A: float) -> float:
        return 2 * A

    node = Node('B', (0, 2), ['A'], equation=equation_func)

    with pytest.raises(ValueError, match="Child nodes do not have a predefined marginal probability distribution."):
        node.marginal_pdf(0.5)

# Test conditional_pdf method
def test_conditional_pdf():
    def equation_func(A: float) -> float:
        return 2 * A

    node = Node(
        'B',
        (0, 2),
        ['A'],
        equation=equation_func,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )

    assert_approx_equal(
        node.conditional_pdf(2, {'A': 1}),
        norm.pdf(2, loc=2, scale=1)
    )

# Test error when calling conditional_pdf for root node
def test_error_conditional_pdf_root_node():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )

    with pytest.raises(ValueError, match="Root nodes do not have a conditional probability distribution."):
        node.conditional_pdf(0.5, {'A': 1})

# Section 2: Factor class tests
# Test creating Factor from a root node
def test_factor_creation_root_node():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    factor = Factor(node)

    assert factor.scope == ['A']
    assert_approx_equal(
        factor.pdf({'A': 0.5}),
        norm.pdf(0.5, loc=0, scale=1)
    )

# Test creating Factor from a child node
def test_factor_creation_child_node():
    def equation_func(A: float) -> float:
        return 2 * A

    node = Node('B', (0, 8), ['A'], equation=equation_func)
    factor = Factor(node)

    assert factor.scope == sorted(['B', 'A'])
    assert_approx_equal(
        factor.pdf({'B': 1.5, 'A': 2}),
        norm.pdf(1.5, loc=4, scale=1)
    )

# Test multiplying two factors
def test_factor_multiplication():
    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    factorA = Factor(nodeA)

    def equation_func(A: float) -> float:
        return 2 * A

    nodeB = Node(
        'B',
        (0, 2),
        ['A'],
        equation=equation_func
    )
    factorB = Factor(nodeB)

    combined_factor = factorA * factorB

    assert combined_factor.scope == sorted(['A', 'B'])
    assert_approx_equal(
        combined_factor.pdf({'A': 0.5, 'B': 1.5}),
        norm.pdf(0.5, loc=0, scale=1) * norm.pdf(1.5, loc=1, scale=1)
    )

# Test error when multiplying with non-Factor object
def test_factor_multiplication_with_non_factor():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    factor = Factor(node)

    with pytest.raises(TypeError, match="Both multiplication operands must be instances of the Factor class."):
        factor * "not a factor"

# Test error when pdf method receives an incomplete dictionary
def test_factor_pdf_invalid_input():
    node = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    factor = Factor(node)

    with pytest.raises(KeyError, match="The values of all variables in the scope must be provided."):
        factor.pdf({'B': 0.5})

# Section 3: BayesianNetwork class tests
# Test BayesianNetwork creation
def test_bayesian_network_creation():
    def equation_func(A: float) -> float:
        return 2 * A

    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    nodeB = Node(
        'B',
        (0, 2),
        ['A'],
        equation=equation_func
    )
    nodeC = Node(
        'C',
        (0, 1),
        ['B'],
        equation=lambda B: 3 * B,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC])

    assert set(network.get_nodes()) == {'A', 'B', 'C'}
    assert set(network.get_edges()) == {('A', 'B'), ('B', 'C')}
    assert network.get_node('A') == nodeA
    assert network.get_node('B') == nodeB
    assert network.get_node('C') == nodeC

# Test error when creating BayesianNetwork with invalid node
def test_error_bayesian_network_creation_invalid_node():
    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    nodeB = Node(
        'B',
        (0, 2),
        ['C'],
        equation=lambda A: 2 * A
    )

    with pytest.raises(ValueError, match="C is not in the Bayesian network."):
        network = BayesianNetwork([nodeA, nodeB])

# Test joint_pdf method for BayesianNetwork
def test_bayesian_network_joint_pdf():
    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    nodeB = Node(
        'B',
        (0, 2),
        ['A'],
        equation=lambda A: 2 * A
    )
    nodeC = Node(
        'C',
        (0, 6),
        ['B'],
        equation=lambda B: 3 * B,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC])

    values = {'A': 0.5, 'B': 1.5, 'C': 4.5}
    expected_probability = (
        nodeA.marginal_pdf(values['A'])
        * nodeB.conditional_pdf(values['B'], {'A': values['A']})
        * nodeC.conditional_pdf(values['C'], {'B': values['B']})
    )

    assert_approx_equal(network.joint_pdf(values), expected_probability)

# Test marginalise_factor method for BayesianNetwork
def test_marginalise_factor():
    parent_node = Node(
        'parent',
        (0, 1),
        [],
        marginal_pdf=lambda x: 1.0,
        parameters={}
    )
    child_node = Node(
        'child',
        domain=(-20, 21),
        parent_variable_names=['parent'],
        equation=lambda parent: parent,
    )

    bn = BayesianNetwork([parent_node, child_node])
    child_factor = Factor(child_node)
    marginalised_factor = bn._marginalise_factor(child_factor, 'parent')

    values = {'child': 0.5}
    expected_marginal_prob = (
        norm.cdf(1, loc=0.5, scale=1) - norm.cdf(0, loc=0.5, scale=1)
    ) # this is just basic math!
    computed_marginal_prob = marginalised_factor.pdf(values)
    print(expected_marginal_prob)
    print(computed_marginal_prob)

    assert_approx_equal(
        expected_marginal_prob,
        computed_marginal_prob,
        0.001
    )

# Test inference for a simple network
def test_inference_simple_network():
    def equation_func(A: float) -> float:
        return 2 * A

    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    nodeB = Node('B', (0, 2), ['A'], equation=equation_func)
    nodeC = Node(
        'C',
        (0, 1),
        ['B'],
        equation=lambda B: 3 * B,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC])

    query_variable = 'A'
    query_range = (0, 1)
    evidence = {'C': 0.5}

    inferred_probability = network.infer(query_variable, query_range, evidence)
    expected_probability = (
        nodeA.marginal_pdf(query_range[0]) * nodeC.conditional_pdf(evidence['C'], {'B': equation_func(query_range[0])})
        + nodeA.marginal_pdf(query_range[1]) * nodeC.conditional_pdf(evidence['C'], {'B': equation_func(query_range[1])})
    )

    assert_inferred_probability(inferred_probability, expected_probability)

# Test inference for a more complex network
def test_inference_complex_network():
    nodeA = Node(
        'A',
        (0, 1),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 0, 'scale': 1}
    )
    nodeB = Node(
        'B',
        (0, 5),
        [],
        marginal_pdf=norm.pdf,
        parameters={'loc': 3, 'scale': 1}
    )
    nodeC = Node(
        'C',
        (-10, 10),
        ['A', 'B'],
        equation=lambda A, B: 2 * A + 0.5 * B,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )
    nodeD = Node(
        'D',
        (-20, 20),
        ['A', 'C'],
        equation=lambda A, C: -2 * A - C,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )
    nodeE = Node(
        'E',
        (-5, 5),
        ['B', 'C'],
        equation=lambda B, C: 0.5 * B + C,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )
    nodeF = Node(
        'F',
        (-3, 3),
        ['D', 'E'],
        equation=lambda D, E: D - E,
        parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1}
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC, nodeD, nodeE, nodeF])

    query_variable = 'F'
    query_range = (-3, 3)
    evidence = {'A': 0.5, 'B': 4}

    inferred_probability = network.infer(query_variable, query_range, evidence)
    expected_probability = network.joint_pdf({**evidence, query_variable: query_range[0]}) + network.joint_pdf({**evidence, query_variable: query_range[1]})

    assert_inferred_probability(inferred_probability, expected_probability)