import numpy as np

from typing import List

class Factor:
    """
    The general class for a factor, which is represented as an an array of
    (unnormalised) pseudo-probabilities corresponding to an ordered list of
    variables, referred to as the factor scope.

    Attributes:
    - scope (List[str]): an ordered list of variable names
    - values (np.ndarray): an array of pseduo-probabilities, where the shape
        is determined by the cardinality of the variables

    Methods:
    - evaluate: evaluates the factor given a set of assignments to the variables
    - multiply: multiplies the factor with another factor and returns a new factor
    - sum_out: sums out a variable from the factor and returns a new factor
    - normalise: normalises the factor
    """
    def __init__(self, scope: List[str], values: np.ndarray) -> None:
        """
        Initializes a factor.

        Args:
        - scope (List[str]): a list of variable names
        - values (np.ndarray): an array of pseudo-probabilities, where the shape
            is determined by the cardinality of the scope

        Raises:
        - ValueError: if the scope are not a list
        - ValueError: if the values are not a numpy array
        - ValueError: if the shape of the values is not consistent with the scope

        Returns: None
        """
        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(scope, list):
            raise ValueError("The scope must be a list")
        if not all(isinstance(variable, str) for variable in scope):
            raise ValueError("The scope must be strings")
        if not isinstance(values, np.ndarray):
            raise ValueError("The values must be a numpy array")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("The values must be floats")

        # 1. Check that the scope is composed of unique variables
        if len(set(scope)) != len(scope):
            raise ValueError("The scope must be unique")

        # 2. Check that the shape of the values is consistent with the scope
        n_scope = len(scope)
        if len(values.shape) != n_scope:
            raise ValueError(f"There are {n_scope} scope, but the values array has {len(values.shape)} dimensions")

        self.scope = scope
        self.values = values

    def evaluate(self, assignments: dict[str, int]) -> float:
        """
        Evaluates the factor given a set of assignments to the variables.

        Args:
        - assignments (dict[str, int]): a dictionary of variable assignments

        Raises:
        - ValueError: if the assignments are not a dictionary of strings to integers
        - ValueError: if the assignments are not exactly the variables in the scope
        - ValueError: if the assignments are not within the cardinality of the variables

        Returns: the value of the factor given the assignments
        """
        # Check validity of inputs
        # 0. Check variable types
        if not isinstance(assignments, dict):
            raise ValueError("The assignments must be a dictionary")
        if not all(isinstance(var, str) for var in assignments.keys()):
            raise ValueError("The keys of the assignments must be strings")
        if not all(isinstance(val, int) for val in assignments.values()):
            raise ValueError("The values of the assignments must be integers")
        # 1. Check that the assignments are exactly the variables in the scope
        if not set(self.scope) == set(assignments.keys()):
            raise ValueError("The assignments are invalid")
        # 2. Check that the assignments are within the cardinality of the variables
        for variable_name, value in assignments.items():
            if value < 0 or value >= self.values.shape[self.scope.index(variable_name)]:
                raise ValueError(f"The value {value} is invalid for variable {variable_name}")

        # Find the indices of the assignments
        indices = [assignments[var] for var in self.scope]

        # Index into the values array
        return self.values[tuple(indices)]

    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Multiplies two factors together and returns a new, composite factor,
        with a scope that is the union of the scopes of the two factors.

        Args:
        - other (Factor): the other factor to multiply

        Raises:
        None

        Returns: a new factor that is the product of the two factors
        """
        # Check validity of inputs
        # 0. Check that the other factor is a Factor
        if not isinstance(other, Factor):
            raise ValueError("The other factor must be a Factor")

        # Combine the scopes
        new_scope = self.scope + [var for var in other.scope if var not in self.scope]

        # Create new axes for broadcasting
        self_extended_shape = [self.values.shape[self.scope.index(var)] if var in self.scope else 1 for var in new_scope]
        other_extended_shape = [other.values.shape[other.scope.index(var)] if var in other.scope else 1 for var in new_scope]

        # Reshape the values arrays for broadcasting
        self_values_reshaped = self.values.reshape(self_extended_shape)
        other_values_reshaped = other.values.reshape(other_extended_shape)

        # Multiply the reshaped arrays
        new_values = self_values_reshaped * other_values_reshaped

        return Factor(new_scope, new_values)

    def __mul__(self, other: 'Factor') -> 'Factor':
        """
        Overloads the multiplication operator to multiply two factors together.
        """
        return self.multiply(other)

    # def sum_out(factor: Factor, variable: str) -> Factor:
    #     """
    #     Sums out a variable from a factor.

    #     Args:
    #     - factor (Factor): the factor to sum out
    #     - variable (str): the variable to sum out

    #     Raises:
    #     - ValueError: if the variable is not in the factor
    #     - ValueError: if the variable is the only variable in the factor

    #     Returns: a new factor with the variable summed out
    #     """
    #     # Check validity of inputs
    #     # 0. Check that the variable is in the factor
    #     if variable not in factor.scope:
    #         raise ValueError("The variable to sum out is not in the factor")

    #     # Find the index of the variable to sum out
    #     variable_index = factor.scope.index(variable)

    #     # Sum out the variable
    #     new_probabilities = np.sum(factor.probabilities, axis=variable_index)

    #     # Remove the variable from the list of scope
    #     new_scope = [var for var in factor.scope if var != variable]

    #     return Factor(new_scope, new_probabilities)
