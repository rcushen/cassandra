from .node import Node
from .factor import Factor

from typing import List


class Network:
    """
    A Bayesian network.

    Composed of nodes, a Bayesian network is a directed acyclic graph that
    represents a set of variables and their conditional dependencies.

    Attributes:
    - nodes (Dict[str, Node]): a dictionary of nodes in the network, where
        the keys are the variable names of the nodes.

    Methods:
    - get_cardinality: returns the number of nodes in the network
    - get_variable_names: returns a set of the variable names of the nodes in
        the network
    - evaluate_joint_probability: evaluates the joint probability of the
        network, given a full suite of evidence variables

    """

    def __init__(self, nodes: List) -> None:
        """
        Initializes a Bayesian network, composed of nodes.

        The network as constructed must be closed; that is, each node must have
        its parent nodes present in the network. The network must also be a
        directed acyclic graph.

        Args:
        - nodes (List[Node]): a list of nodes in the Bayesian network

        Raises:
        - ValueError: if the nodes are not a list
        - ValueError: if the nodes are not instances of the Node class
        - ValueError: if the nodes are not unique
        - ValueError: if the network is not closed
        - ValueError: if the network is not a directed acyclic graph

        Returns: None

        Attributes:
        - nodes (Dict[str, Node]): a dictionary of nodes in the network, where
            the keys are the variable names of the nodes.
        """

        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(nodes, list):
            raise ValueError("The nodes must be a list")
        if not all(isinstance(node, Node) for node in nodes):
            raise ValueError("The nodes must be instances of the Node class")
        # 1. Check that the nodes are unique
        if len(set(nodes)) != len(nodes):
            raise ValueError("The nodes must be unique")
        # 2. Check that the network is closed
        for node in nodes:
            for parent_node in node.parent_nodes:
                if parent_node not in nodes:
                    raise ValueError("The network is not closed")
        # 3. Check that the network is a directed acyclic graph
        # TODO: Implement this check

        self.nodes = {node.variable_name: node for node in nodes}

    def get_cardinality(self) -> int:
        """
        Returns the number of nodes in the network.

        Args: None

        Raises: None

        Returns: an integer representing the number of nodes in the network
        """
        return len(self.nodes)

    def get_variable_names(self) -> set[str]:
        """
        Returns the variable names of the nodes in the network.

        Args: None

        Raises: None

        Returns: a set of strings representing the variable names of the nodes
        """
        return set([node_name for node_name in self.nodes.keys()])

    def get_factors(self) -> list[Factor]:
        """
        Returns a set of factor representations for the network.

        Args: None

        Raises: None

        Returns: a list of factors for the network
        """
        return set([node.to_factor() for node in self.nodes.values()])

    def evaluate_joint_probability(self, e: dict[str, int]) -> float:
        """
        Evaluates the joint probability of the network, given a full suite of
        evidence variables.

        Args:
        - e (Dict[str, int]): a dictionary of all variables and their values

        Raises:
        - ValueError: if the evidence variables are not a dictionary
        - ValueError: if the evidence variables are not complete

        Returns: a float representing the joint probability of the network
        """

        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(e, dict):
            raise TypeError("The evidence variables must be a dictionary")
        # 1. Check that the evidence variables are complete
        if set(e.keys()) != self.get_variable_names():
            raise ValueError("The evidence variables must be complete")
        # 2. Check that the evidence observations are valid
        for variable_name, value in e.items():
            if value < 0 or value >= self.nodes[variable_name].get_cardinality():
                raise ValueError("The evidence observations are invalid")

        # Compute the joint probability
        prob = 1
        for variable_name, node in self.nodes.items():
            parent_variable_names = [
                parent_node.variable_name for parent_node in node.parent_nodes
            ]
            conditional_probability = node.compute_conditional_probability(
                e[variable_name],
                {
                    parent_variable_name: e[parent_variable_name]
                    for parent_variable_name in parent_variable_names
                },
            )
            prob *= conditional_probability

        return prob

    def query(self, Y, e={}):
        """
        Evaluates a conditional probability query on the network, using the
        variable elimination algorithm.

        Args:
        - Y (Dict[str, int]): a dictionary of query variables and their values
        - e (Dict[str, int]): a dictionary of evidence variables and their
            values

        Raises: None

        Returns: a dictionary where the keys are the variable names of the query
        and the values are the probabilities of the query variables.
        """

        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(Y, dict):
            raise TypeError("The query variables must be a dictionary")
        if not isinstance(e, dict):
            raise TypeError("The evidence variables must be a dictionary")
        if not all(isinstance(value, int) for value in Y.values()):
            raise TypeError("The query variables must be integers")
        if not all(isinstance(value, int) for value in e.values()):
            raise TypeError("The evidence variables must be integers")
        # 1. Check that there is at least one query variable
        if len(Y) == 0:
            raise ValueError("There must be at least one query variable")
        # 2. Check that there is no overlap between query and evidence variables
        if set(Y.keys()).intersection(e.keys()):
            raise ValueError("The query and evidence variables must be disjoint")
        # 3. Check that the union of the query and evidence variables is a subset
        # of the network's variable names
        if not set(Y.keys()).union(e.keys()).issubset(self.get_variable_names()):
            raise ValueError(
                "The query and evidence variables must be a subset of the network"
            )

        # Identify the elimination variables
        elimination_vars = self.get_variable_names().difference(
            set(Y.keys()).union(set(e.keys()))
        )

        # Select an elimination ordering
        elimination_ordering = list(elimination_vars)

        # Eliminate variables in the elimination ordering
        factors = self.get_factors()
        for variable_name in elimination_ordering:
            factors = sum_product_eliminate(factors, variable_name)

        prob = 0
        return prob
