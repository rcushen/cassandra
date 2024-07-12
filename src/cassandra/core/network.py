from .node import Node

from typing import List

class Network:
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
        # (to be implemented)

        self.nodes = nodes


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
        return set([node.variable_name for node in self.nodes])

    def query(self, Y, e):
        """
        TO BE IMPLEMENTED
        """
        prob = 0
        return prob
