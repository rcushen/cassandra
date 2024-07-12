import numpy as np

from functools import reduce
from typing import List

class Node:
    def __init__(
            self,
            variable_name: str,
            parent_nodes: List,
            cpd: np.ndarray
            ) -> None:
        """
        Initializes a node for a Bayesian network.

        A node is composed of a variable name, a list of parent nodes, and a
        conditional probability distribution (CPD) that represents the
        probability distribution of the node given the states of the parent
        nodes.

        Args:
        - variable_name (str): the name of the variable
        - parent_nodes (List[Node]): a list of parent nodes
        - cpd (np.Array): a (N+1)-dimensional numpy array representing the
            conditional probability distribution associated with the node, where
            each dimension N corresponds to a parent node and the last dimension
            corresponds to the variable itself.

        Raises:
        - ValueError: if the CPD is not a numpy array
        - ValueError: if the variable name is not a string or is too long
        - ValueError: if the parent nodes are not a list
        - ValueError: if the shape of the CPD is inconsistent with the parent nodes
        - ValueError: if the CPD does not represent a valid probability distribution

        Returns: None
        """

        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(cpd, np.ndarray):
            raise ValueError("The CPD must be a numpy array")
        if not isinstance(variable_name, str):
            raise ValueError("The variable name must be a string")
        if len(variable_name) == 0 or len(variable_name) > 20:
            raise ValueError("The variable name must be a reasonably short string")
        if not isinstance(parent_nodes, list):
            raise ValueError("The parent nodes must be a list")

        # 1. Check that the CPD is consistent with the parent nodes
        if len(parent_nodes) > 0:
            parent_node_cardinalities = tuple(node.get_cardinality() for node in parent_nodes)
            current_node_cardinality = cpd.shape[-1]
            expected_shape = parent_node_cardinalities + (current_node_cardinality,)
            if cpd.shape != expected_shape:
                raise ValueError(f"The shape of the CPD is {cpd.shape}, but the expected shape is {expected_shape}")
        else:
            if cpd.ndim != 1:
                raise ValueError("The dimension of the CPD should be 1 if there are no parent nodes")

        # 2. Check that each row in the CPD represents a valid probability distribution,
        #    i.e. the sum of the elements in each row should be equal to 1
        if len(parent_nodes) > 0:
            if not np.all(np.isclose(np.sum(cpd, axis=-1), 1)):
                raise ValueError("The CPD does not represent a valid probability distribution")
        else:
            if not np.isclose(np.sum(cpd), 1):
                raise ValueError("The CPD does not represent a valid probability distribution")

        self.variable_name = variable_name
        self.parent_nodes = parent_nodes
        self.cpd = cpd

    def __repr__(self):
        return """
        Variable name: {}
        Parent nodes: {}
        CPD: {}
        """.format(self.variable_name, self.parent_nodes, self.cpd)

    def get_cardinality(self) -> int:
        """
        Returns the number of possible states of the variable, which is equal to
        the cardinality of the last dimension of the CPD.

        Args: None

        Raises: None

        Returns: an integer representing the number of possible states of the variable
        """
        return self.cpd.shape[-1]

    def get_parent_nodes(self) -> List:
        """
        Returns the parent nodes of the current node.

        Args: None

        Raises: None

        Returns: a list of parent nodes
        """
        return self.parent_nodes

    def get_conditional_distribution(self, parent_variable_assignment: dict[str, int]) -> np.ndarray:
        """
        Returns the conditional distribution of the node, given an assignment
        of observed states of all the parent variables.

        Args:
        - parent_variable_assignment (dict): a dictionary where the keys are the
            names of the parent nodes and the values are the indices of the
            observed states of these parent nodes.

        Raises:
        - ValueError: if the parent nodes are not provided
        - ValueError: if the parent nodes are not valid
        - ValueError: if the parent nodes are not integers

        Returns: a (cardinality,) numpy array representing the conditional
        distribution of the variable, given the parent observations.
        """
        # Validate inputs
        # 0. Check all inputs are integers
        if not all(map(lambda x: isinstance(x, int), parent_variable_assignment.values())):
            raise ValueError("All inputs should be integers")

        # 1. Check that all parent nodes are provided
        if set(parent_variable_assignment.keys()) != set(map(lambda x: x.variable_name, self.parent_nodes)):
            raise ValueError("Not all parent nodes are provided")

        # 2. Check that the provided states are valid, i.e. the integers are
        #    within the range of the number of states of the parent nodes
        parent_node_cardinalities = {node.variable_name: node.get_cardinality() for node in self.parent_nodes}
        for item in parent_variable_assignment.items():
            variable_name, provided_index = item
            if provided_index < 0 or provided_index >= parent_node_cardinalities[variable_name]:
                raise ValueError(f"The state {provided_index} of parent node {variable_name} is invalid; it exceeds the number of states of the parent node")

        # Construct the correct order of the indices
        indices = tuple(map(lambda node: parent_variable_assignment[node.variable_name], self.parent_nodes))

        # Get the conditional distribution, i.e. index into the CPD
        return self.cpd[indices]

    def compute_conditional_probability(self, variable_assignment: int, parent_variable_assignments: dict[str, int]) -> float:
        """
        Computes the conditional probability of the variable given an
        assignment of parent variables and a particular state of the variable.

        Args:
        - variable_assignment (int): an integer representing the state of the variable
        - parent_variable_assignments: a set of keyword arguments, where the keys are the names of
            the parent nodes and the values are the indices of the observed
            states of these parent nodes.

        Raises:
        - ValueError: if the variable assignment is not a valid state

        Returns: a float representing the conditional probability of the state,
        given the observed states of the parent nodes.
        """
        # Validate inputs
        # 0. Check that the variable assignment is an integer
        if not isinstance(variable_assignment, int):
            raise ValueError("The variable assignment should be an integer")
        # 1. Check that the variable assignment is a valid state
        if variable_assignment < 0 or variable_assignment >= self.get_cardinality():
            raise ValueError(f"The variable assignment {variable_assignment} is invalid; it exceeds the number of states of the variable")

        return self.get_conditional_distribution(parent_variable_assignments)[variable_assignment]

