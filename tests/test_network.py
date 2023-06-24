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
        "A",
        (0, 1),
        [],
        ["gear", "is_on"],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )

    assert node.type == NodeType.ROOT
    assert node.variable_name == "A"
    assert node.domain == (0, 1)
    assert node.parent_variable_names == []
    assert node.system_parameter_names == ["gear", "is_on"]
    assert node.distribution_parameters == {"loc": 0, "scale": 1}


# Test Node creation with parent variables (i.e. a child node)
def test_child_node_creation():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    node = Node("B", (0, 2), ["A"], ["state"], equation=equation_func)

    assert node.type == NodeType.CHILD
    assert node.variable_name == "B"
    assert node.domain == (0, 2)
    assert node.parent_variable_names == ["A"]
    assert node.system_parameter_names == ["state"]
    assert node.equation({"A": 1}, {"state": 1}) == 2


# Test errors when creating improperly specified nodes
def test_error_root_node_creation():
    with pytest.raises(
        ValueError,
        match="Root nodes must have a marginal probability distribution and distribution parameters.",
    ):
        Node("A", (0, 1), [], [])

    with pytest.raises(
        ValueError,
        match="Root nodes must have a marginal probability distribution and distribution parameters.",
    ):
        Node("A", (0, 1), [], [], marginal_pdf=norm.pdf)


def test_error_child_node_creation():
    with pytest.raises(ValueError, match="Child nodes must have an equation."):
        Node("B", (0, 2), ["A"], ["state"])


# Test marginal_pdf method
def test_marginal_pdf():
    node = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )

    assert_approx_equal(node.marginal_pdf(0.5), norm.pdf(0.5, loc=0, scale=1))


# Test error when calling marginal_pdf for child node
def test_error_marginal_pdf_child_node():
    def equation_func(A: float) -> float:
        return 2 * A

    node = Node("B", (0, 2), ["A"], ["state"], equation=equation_func)

    with pytest.raises(
        ValueError,
        match="Child nodes do not have a predefined marginal probability distribution.",
    ):
        node.marginal_pdf(0.5)


# Test conditional_pdf method
def test_conditional_pdf():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    node = Node(
        "B",
        (0, 2),
        ["A"],
        ["state"],
        equation=equation_func,
        distribution_parameters={"locs": {"intercept": 0, "slope": 1}, "scale": 1},
    )

    assert_approx_equal(
        node.conditional_pdf(2, {"A": 1}, {"state": 1}), norm.pdf(2, loc=2, scale=1)
    )


# Test error when calling conditional_pdf for root node
def test_error_conditional_pdf_root_node():
    node = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )

    with pytest.raises(
        ValueError,
        match="Root nodes do not have a conditional probability distribution.",
    ):
        node.conditional_pdf(0.5, {}, {})


# Section 2: Factor class tests
# Test creating Factor from a root node
def test_factor_creation_root_node():
    node = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    factor = Factor(node)

    assert factor.scope == ["A"]
    assert_approx_equal(factor.pdf({"A": 0.5}, {}), norm.pdf(0.5, loc=0, scale=1))


# Test creating Factor from a child node
def test_factor_creation_child_node():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    node = Node("B", (0, 8), ["A"], ["state"], equation=equation_func)
    factor = Factor(node)

    assert factor.scope == sorted(["B", "A"])
    assert_approx_equal(
        factor.pdf({"B": 1.5, "A": 2}, {"state": 1}), norm.pdf(1.5, loc=4, scale=1)
    )


# Test multiplying two factors
def test_factor_multiplication():
    nodeA = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    factorA = Factor(nodeA)

    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    nodeB = Node("B", (0, 2), ["A"], ["state"], equation=equation_func)
    factorB = Factor(nodeB)

    combined_factor = factorA * factorB

    assert combined_factor.scope == sorted(["A", "B"])
    assert_approx_equal(
        combined_factor.pdf({"A": 0.5, "B": 1.5}, {"state": 1}),
        norm.pdf(0.5, loc=0, scale=1) * norm.pdf(1.5, loc=1, scale=1),
    )


# Test error when multiplying with non-Factor object
def test_factor_multiplication_with_non_factor():
    node = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    factor = Factor(node)

    with pytest.raises(
        TypeError,
        match="Both multiplication operands must be instances of the Factor class.",
    ):
        factor * "not a factor"


# Test error when pdf method receives an incomplete dictionary
def test_factor_pdf_invalid_input():
    node = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    factor = Factor(node)

    with pytest.raises(
        KeyError, match="The values of all variables in the scope must be provided."
    ):
        factor.pdf({"B": 0.5}, {})


# Section 3: BayesianNetwork class tests
# Test BayesianNetwork creation
def test_bayesian_network_creation():
    def equation_func_B(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    def equation_func_C(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 3 * variable_values["B"] * parameter_values["state"]

    nodeA = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    nodeB = Node(
        "B",
        (0, 2),
        ["A"],
        ["state"],
        equation=equation_func_B,
    )
    nodeC = Node(
        "C",
        (0, 6),
        ["B"],
        ["state"],
        equation=equation_func_C,
        distribution_parameters={"locs": {"intercept": 0, "slope": 1}, "scale": 1},
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC])

    assert set(network.get_nodes()) == {"A", "B", "C"}
    assert set(network.get_edges()) == {("A", "B"), ("B", "C")}
    assert network.get_node("A") == nodeA
    assert network.get_node("B") == nodeB
    assert network.get_node("C") == nodeC


# Test error when creating BayesianNetwork with invalid node
def test_error_bayesian_network_creation_invalid_node():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    nodeA = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    nodeB = Node("B", (0, 2), ["C"], ["state"], equation=equation_func)

    with pytest.raises(ValueError, match="C is not in the Bayesian network."):
        network = BayesianNetwork([nodeA, nodeB])


# Test joint_pdf method for BayesianNetwork
def test_bayesian_network_joint_pdf():
    def equation_func_B(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 2 * variable_values["A"] * parameter_values["state"]

    def equation_func_C(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return 3 * variable_values["B"] * parameter_values["state"]

    nodeA = Node(
        "A",
        (0, 1),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    nodeB = Node(
        "B",
        (0, 2),
        ["A"],
        ["state"],
        equation=equation_func_B,
    )
    nodeC = Node(
        "C",
        (0, 6),
        ["B"],
        ["state"],
        equation=equation_func_C,
        distribution_parameters={"locs": {"intercept": 0, "slope": 1}, "scale": 1},
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC])

    variable_values = {"A": 0.5, "B": 1.5, "C": 4.5}
    system_parameter_values = {"state": 1.0}

    calculated_probability = network.joint_pdf(variable_values, system_parameter_values)
    true_probability = (
        nodeA.marginal_pdf(variable_values["A"], system_parameter_values)
        * nodeB.conditional_pdf(
            variable_values["B"], {"A": variable_values["A"]}, system_parameter_values
        )
        * nodeC.conditional_pdf(
            variable_values["C"], {"B": variable_values["B"]}, system_parameter_values
        )
    )

    assert_approx_equal(calculated_probability, true_probability)


# Test marginalise_factor method for BayesianNetwork
def test_marginalise_factor():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return variable_values["A"] * parameter_values["state"]

    nodeA = Node(
        "A", (0, 1), [], [], marginal_pdf=lambda x: 1.0, distribution_parameters={}
    )
    nodeB = Node(
        "B",
        (-20, 21),
        ["A"],
        ["state"],
        equation=equation_func,
    )

    bn = BayesianNetwork([nodeA, nodeB])
    factorB = Factor(nodeB)

    marginalised_factor = bn._marginalise_factor(factorB, "A")

    variable_values = {"B": 0.5}
    system_parameter_values = {"state": 1.0}

    calculated_probability = marginalised_factor.pdf(
        variable_values, system_parameter_values
    )
    true_probability = norm.cdf(1, loc=0.5, scale=1) - norm.cdf(0, loc=0.5, scale=1)

    assert_approx_equal(calculated_probability, true_probability)


# Test inference for a simple network
def test_inference_simple_network():
    def equation_func(
        variable_values: dict[str, float], parameter_values: dict[str, float]
    ) -> float:
        return variable_values["A"] * parameter_values["state"]

    nodeA = Node(
        "A", (0, 1), [], [], marginal_pdf=lambda x: 1.0, distribution_parameters={}
    )
    nodeB = Node(
        "B",
        (-20, 21),
        ["A"],
        ["state"],
        equation=equation_func,
    )

    network = BayesianNetwork([nodeA, nodeB])

    query_variable_name = "B"
    query_range = (0, 1)
    evidence = {"A": 0.5}
    system_parameter_values = {"state": 1.0}

    calculated_probability = network.infer(
        query_variable_name, query_range, evidence, system_parameter_values
    )
    true_probability = norm.cdf(1, loc=0.5, scale=1) - norm.cdf(0, loc=0.5, scale=1)

    assert_approx_equal(calculated_probability, true_probability)


# Test inference for a more complex network
def test_inference_complex_network():
    # assume throughout that 5 sd is enough to capture the distribution
    nodeA = Node(
        "A",
        (-5, 5),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    nodeB = Node(
        "B",
        (-2, 8),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 3, "scale": 1},
    )
    nodeC = Node(
        "C",
        (-16, 19),
        ["A", "B"],
        ["state"],
        equation=lambda v, p: p["state"] * (2 * v["A"] + 0.5 * v["B"]),
    )
    nodeD = Node(
        "D",
        (-30, 30),
        ["A", "C"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["A"] - v["C"]),
    )
    nodeE = Node(
        "E",
        (-50, 50),
        ["B", "C"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["B"] + v["C"]),
    )
    nodeF = Node(
        "F",
        (-200, 200),
        ["D", "E"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["D"] + v["E"]),
    )

    network = BayesianNetwork([nodeA, nodeB, nodeC, nodeD, nodeE, nodeF])

    query_variable_name = "F"
    query_range = (-3, 3)
    evidence = {"A": 0.5, "B": 4}
    system_parameter_values = {"state": 1.0}

    calculated_probability = network.infer(
        query_variable_name, query_range, evidence, system_parameter_values
    )
    true_probability = 0.000001

    assert_approx_equal(calculated_probability, true_probability)
