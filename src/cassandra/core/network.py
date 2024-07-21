from .node import Node

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
    - joint_probability: evaluates the joint probability of the network, given a
        full suite of evidence variables

    """
    def __init__(
            self,
            nodes: List
        ) -> None:
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

        self.nodes = { node.variable_name: node for node in nodes }

    def get_cardinality(self):
        """
        Returns the number of nodes in the network.

        Args: None

        Raises: None

        Returns: an integer representing the number of nodes in the network
        """
        return len(self.nodes)

    def get_variable_names(self):
        """
        Returns the variable names of the nodes in the network.

        Args: None

        Raises: None

        Returns: a set of strings representing the variable names of the nodes

        """
        return set([node for node in self.nodes.keys()])

    def joint_probability(self, e: dict[str, int]) -> float:
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
            raise ValueError("The evidence variables must be a dictionary")
        # 1. Check that the evidence variables are complete
        if set(e.keys()) != self.get_variable_names():
            raise ValueError("The evidence variables must be complete")
        # 2. Check that the evidence observations are valid
        for variable_name, value in e.items():
            if value < 0 or value >= self.nodes[variable_name].get_cardinality():
                raise ValueError("The evidence observations are invalid")

        # Compute the joint probability
        prob = 1
        nodes = self.nodes
        for variable_name, node in nodes.items():
            parent_variable_names = [parent_node.variable_name for parent_node in node.parent_nodes]
            conditional_probability = node.compute_conditional_probability(
                e[variable_name],
                { parent_variable_name: e[parent_variable_name] for parent_variable_name in parent_variable_names }
            )
            prob *= conditional_probability

        return prob

    def query(self, Y, e):
        """
        Evaluates a conditional probability query on the network, using the
        variable elimination algorithm.

        Args:
        - Y (Set[str]): a set of variable names to query
        - e (Dict[str, int]): a dictionary of evidence variables and their values

        Raises: None

        Returns: a dictionary where the keys are the variable names of the query
        and the values are the probabilities of the query variables.
        """
        prob = 0
        return prob
