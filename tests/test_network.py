import pytest

from cassandra.network import Node, NodeType

from scipy.stats import norm

# Test Node creation with no parent variables (root node)
def test_root_node_creation():
    node = Node('A', (0, 1), [], marginal_pdf=norm.pdf, parameters={'loc': 0, 'scale': 1})

    assert node.type == NodeType.ROOT
    assert node.variable_name == 'A'
    assert node.domain == (0, 1)
    assert node.parent_variable_names == []

# Test Node creation with parent variables (child node)
def test_child_node_creation():
    def equation_func(parentA: float) -> float:
        return 2 * parentA

    node = Node('B', (0, 2), ['parentA'], equation=equation_func)

    assert node.type == NodeType.CHILD
    assert node.variable_name == 'B'
    assert node.domain == (0, 2)
    assert node.parent_variable_names == ['parentA']
    assert node.equation(parentA=1) == 2

# Test errors when creating improperly specified nodes
def test_error_root_node_creation():
    with pytest.raises(ValueError, match="Root nodes must have a marginal probability distribution and parameters."):
        Node('A', (0, 1), [])

    with pytest.raises(ValueError, match="Root nodes must have a marginal probability distribution and parameters."):
        Node('A', (0, 1), [], marginal_pdf=norm.pdf)

def test_error_child_node_creation():
    with pytest.raises(ValueError, match="Child nodes must have an equation."):
        Node('B', (0, 2), ['parentA'])

# Test marginal_pdf method
def test_marginal_pdf():
    node = Node('A', (0, 1), [], marginal_pdf=norm.pdf, parameters={'loc': 0, 'scale': 1})
    result = node.marginal_pdf(0.5)

    assert result == norm.pdf(0.5, loc=0, scale=1)

# Test error when calling marginal_pdf for child node
def test_error_marginal_pdf_child_node():
    def equation_func(parentA: float) -> float:
        return 2 * parentA

    node = Node('B', (0, 2), ['parentA'], equation=equation_func)

    with pytest.raises(ValueError, match="Child nodes do not have a predefined marginal probability distribution."):
        node.marginal_pdf(0.5)

# Test conditional_pdf method
def test_conditional_pdf():
    def equation_func(parentA: float) -> float:
        return 2 * parentA

    node = Node('B', (0, 2), ['parentA'], equation=equation_func,
                parameters={'locs': {'intercept': 0, 'slope': 1}, 'scale': 1})
    result = node.conditional_pdf(2, {'parentA': 1})

    assert result == norm.pdf(2, loc=2, scale=1)

# Test error when calling conditional_pdf for root node
def test_error_conditional_pdf_root_node():
    node = Node('A', (0, 1), [], marginal_pdf=norm.pdf, parameters={'loc': 0, 'scale': 1})

    with pytest.raises(ValueError, match="Root nodes do not have a conditional probability distribution."):
        node.conditional_pdf(0.5, {'parentA': 1})