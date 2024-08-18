import numpy as np

from typing import List


class Factor:
    """
    The general class for a factor, which is represented as an an N-dimensional
    array of (unnormalised) pseudo-probabilities, with each dimension
    corresponding to a variable in the factor scope.

    Attributes:
    - scope (List[str]): an ordered list of variable names
    - values (np.ndarray): an array of pseduo-probabilities, where the shape
        is determined by the cardinality of the variables, and the
        dimensionality aligns with the factor scope.

    Methods:
    - evaluate: evaluates the factor given a set of assignments to the variables
    - multiply: multiplies the factor with another factor and returns a new
        factor
    - sum_out: sums out a variable from the factor and returns a new factor
    - normalise: normalises the factor
    """

    def __init__(self, scope: List[str], values: np.ndarray) -> None:
        """
        Initializes a factor.

        Args:
        - scope (List[str]): a list of variable names
        - values (np.ndarray): an array of pseudo-probabilities

        Raises:
        - ValueError: if the scope is not a list of unique strings
        - ValueError: if the values are not a numpy array of floats
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
        n_variables = len(scope)
        if len(values.shape) != n_variables:
            raise ValueError(
                f"There are {n_variables} variables in the scope, but the values array has {len(values.shape)} dimensions"
            )

        self.scope = scope
        self.values = values

    def __eq__(self, other: "Factor") -> bool:
        """
        Overloads the equality operator to compare two factors.
        """
        return self.scope == other.scope and np.allclose(self.values, other.values)

    def __hash__(self) -> int:
        """
        Overloads the hash function to hash the factor.
        """
        scope = tuple(self.scope)
        values_rounded = np.round(self.values, decimals=10)
        return hash((scope, values_rounded.tobytes()))

    def __repr__(self) -> str:
        """
        Overloads the string representation of the factor.
        """
        return f"Factor({self.scope}, {self.values.shape})"

    def __mul__(self, other: "Factor") -> "Factor":
        """
        Overloads the multiplication operator to multiply two factors together.
        """
        return self.multiply(other)

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
                raise ValueError(
                    f"The value {value} is invalid for variable {variable_name}"
                )

        # Find the indices of the assignments
        indices = [assignments[var] for var in self.scope]

        # Index into the values array
        return self.values[tuple(indices)]

    def multiply(self, other: "Factor") -> "Factor":
        """
        Multiplies two factors together and returns a new, composite factor,
        with a scope that is the union of the scopes of the two factors.

        Args:
        - other (Factor): the other factor to multiply

        Raises:
        - ValueError: if the other factor is not a Factor

        Returns: a new factor that is the product of the two factors
        """
        # Check validity of inputs
        # 0. Check that the other factor is a Factor
        if not isinstance(other, Factor):
            raise ValueError("The other factor must be a Factor")

        # Combine the scopes
        new_scope = self.scope + [var for var in other.scope if var not in self.scope]

        # Create new axes for broadcasting
        self_extended_shape = [
            self.values.shape[self.scope.index(var)] if var in self.scope else 1
            for var in new_scope
        ]
        other_extended_shape = [
            other.values.shape[other.scope.index(var)] if var in other.scope else 1
            for var in new_scope
        ]

        # Reshape the values arrays for broadcasting
        self_values_reshaped = self.values.reshape(self_extended_shape)
        other_values_reshaped = other.values.reshape(other_extended_shape)

        # Multiply the reshaped arrays
        new_values = self_values_reshaped * other_values_reshaped

        return Factor(new_scope, new_values)

    def sum_out(self, variable: str) -> "Factor":
        """
        Sums out a variable from a factor.

        Args:
        - variable (str): the variable to sum out

        Raises:
        - ValueError: if the variable is not in the factor
        - ValueError: if the variable is the only variable in the factor

        Returns: a new factor with the variable summed out
        """
        # Check validity of inputs
        # 0. Check that the variable is in the factor
        if variable not in self.scope:
            raise ValueError("The variable to sum out is not in the factor")

        # 1. Check that the variable is not the only variable in the factor
        if len(self.scope) == 1:
            raise ValueError("The variable is the only variable in the factor")

        # Find the index of the variable to sum out
        variable_index = self.scope.index(variable)

        # Sum out the variable
        new_values = np.sum(self.values, axis=variable_index)

        # Remove the variable from the list of scope
        new_scope = [var for var in self.scope if var != variable]

        return Factor(new_scope, new_values)

    def normalise(self, n_dimensions=1) -> None:
        """
        Normalises the factor by dividing by the sum of (some of) the values of
        the factor.

        Args:
        None

        Raises:
        - ValueError: if the sum of the values is zero

        Returns: a new factor that is normalised
        """
        # Check validity of inputs
        # 0. Check that the n_dimensions is an integer
        if not isinstance(n_dimensions, int):
            raise ValueError("The number of dimensions to normalise over must be an integer")

        # 1. Check that the n_dimensions conforms to the dimensions of the values
        if n_dimensions <= 0 or n_dimensions > self.values.ndim:
            raise ValueError("n_dimensions must be between 1 and the number of dimensions in the array")

        # Calculate the sum along the last n_dimensions dimensions
        sum_axes = tuple(range(self.values.ndim - n_dimensions, self.values.ndim))
        sums = np.sum(self.values, axis=sum_axes, keepdims=True)

        # Avoid division by zero
        sums = np.where(sums == 0, 1, sums)

        # Normalize the array
        normalised_values = self.values / sums

        # Update the values
        self.values = normalised_values

        return None


