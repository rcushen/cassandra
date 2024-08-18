from .factor import Factor

from functools import reduce

def sum_product_eliminate(factors: set[Factor], variable_name: str) -> set[Factor]:
    """
    Eliminates a variable from a set of factors using the sum-product algorithm.

    Args:
    - factors (Set[Factor]): a set of factors
    - variable_name (str): the name of the variable to eliminate

    Raises: None

    Returns: a set of factors after eliminating the variable
    """

    # Validate inputs
    # 0. Check variable types
    if not isinstance(factors, set):
        raise ValueError("The factors must be a set")
    if not all(isinstance(factor, Factor) for factor in factors):
        raise ValueError("The factors must be instances of the Factor class")
    if not isinstance(variable_name, str):
        raise ValueError("The variable name must be a string")
    # 1. Check that the variable is in at least one of the factors
    if not any(variable_name in factor.scope for factor in factors):
        raise ValueError("The variable is not in any of the factors")

    # Separate the relevant and irrelevant factors
    relevant_factors = [factor for factor in list(factors) if variable_name in factor.scope]
    irrelevant_factors = [factor for factor in list(factors) if variable_name not in factor.scope]

    # Multiply all the relevant factors together
    phi = reduce(lambda x, y: x.multiply(y), relevant_factors)

    # Sum out the variable
    tau = phi.sum_out(variable_name)

    return set(irrelevant_factors + [tau])
