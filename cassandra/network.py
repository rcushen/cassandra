from functools import reduce
from enum import Enum, auto

from scipy.integrate import quad
from scipy.stats import norm


class NodeType(Enum):
    ROOT = auto()
    CHILD = auto()


class Node:
    """
    A high-level class representing a node in a Bayesian Network.

    A node can be either a root node, for which a marginal probability density
    function must be defined, or a child node, for which a physical equation
    must provided which determines the value of the node variable, written as a
    function of its parents and the external system parameters.

    The conditional probability density function of the node is then defined as
    a Gaussian distribution with mean equal to the value of the physical
    equation and a fixed standard deviation.

    Attributes
    ----------
    variable_name : str
        The name of the variable identified with the node.
    domain : tuple[float, float]
        The domain of the variable, defined as an interval.
    type : str
        The type of the node, either 'root' or 'child'.
    parent_variable_names: list[str]
        The names of the parent variables of the node.
    system_parameter_names : list[str]
        The names of the external system parameters which enter into the node's
        equation or marginal probability density function.
    distribution_parameters : dict
        If the node is a child node, a dictionary of 'locs' and 'scale'
        parameters to associate with the conditional Gaussian probability
        density function; if the node is a root node, an arbitrary list of
        parameters to characterise the marginal density function (e.g. the
        alpha and beta parameters of a Beta distribution).

    Methods
    -------
    equation: callable
        If the node is a child node, the physical equation which determines
        the value of the node variable, written as a function of its parents and
        the external system parameters.
    marginal_pdf : callable
        If the node is a root node, the marginal probability density function
        that defines the node.
    conditional_pdf : callable
        If the node is a child node, the conditional probability density
        function that defines the node.
    """

    def __init__(
        self,
        variable_name: str,
        domain: tuple[float, float],
        parent_variable_names: list[str],
        system_parameter_names: list[str],
        equation: callable = None,
        marginal_pdf: callable = None,
        distribution_parameters: dict = None,
    ) -> None:
        """
        Creates a Node, identified with the variable `variable_name`.

        This can be either a root node, for which a marginal probability density
        function must be defined, or a child node, for which a conditional
        probability density function must be defined.

        Parameters
        ----------
        variable_name : str
            The name of the variable identified with the node.
        domain : tuple[float, float]
            The domain of the variable, defined as an interval.
        parent_variable_names: list[str]
            The names of the parent variables.
        system_parameter_names : list[str]
            The names of the external system parameters which enter into the
            node's equation or marginal probability density function.
        equation: callable
            If the node is a child node, the physical equation which determines
            the value of the node variable, written as a function of its parents
            and the external system parameters. The function signature must be
            equal to `parent_variable_values: dict[str, float],
            system_parameter_values: dict[str, float]`, and it must return a
            float within the domain of the of the variable.
        marginal_pdf : callable
            If the node is a root node, the marginal probability density
            function that defines the node.
        distribution_parameters : dict
            If the node is a child node, a dictionary of 'locs' and 'scale'
            parameters to associate with the conditional Gaussian probability
            density function; if the node is a root node, an arbitrary list of
            parameters to characterise the marginal density function (e.g. the
            alpha and beta parameters of a Beta distribution).

        Raises
        ------
        ValueError
            If the Node is not correctly specified.
        """
        self.variable_name = variable_name
        self.domain = domain
        self.parent_variable_names = parent_variable_names
        self.system_parameter_names = system_parameter_names
        if len(parent_variable_names) == 0:
            # If there are no parent variables, the node is a root node
            self.type = NodeType.ROOT
            if marginal_pdf is None or distribution_parameters is None:
                raise ValueError(
                    "Root nodes must have a marginal probability distribution and distribution parameters."
                )
            self._marginal_pdf = marginal_pdf
            self.distribution_parameters = distribution_parameters
        else:
            # Else the node is a child node
            self.type = NodeType.CHILD
            if equation is None:
                raise ValueError("Child nodes must have an equation.")
            # If no parameters have been specified for the conditional
            # Gaussian, set them to default values
            self._equation = equation
            if distribution_parameters is None:
                self.distribution_parameters = {
                    "locs": {"intercept": 0, "slope": 1},
                    "scale": 1,
                }
            else:
                self.distribution_parameters = distribution_parameters

    def __repr__(self):
        """
        Returns a string representation of the Node.

        Returns
        -------
        str
            A string representation of the Node.
        """
        if self.type == NodeType.CHILD:
            return f"Node({self.variable_name}, {self.parent_variable_names}, {self.system_parameter_names}, {self.equation}, {self.distribution_parameters})"
        else:
            return f"Node({self.variable_name}, {self.system_parameter_names}, {self.marginal_pdf}, {self.distribution_parameters})"

    def equation(
        self,
        parent_variable_values: dict[str, float],
        system_parameter_values: dict[str, float],
    ) -> float:
        """
        Returns the value of the physical equation that determines the value of
        the node variable, written as a function of its parents and the external
        system parameters.

        Parameters
        ----------
        parent_variable_values : dict[str, float]
            The values of the parents of the node.
        system_parameter_values : dict[str, float]
            The values of the external system parameters.

        Returns
        -------
        float
            The value of the node variable.
        """
        return self._equation(parent_variable_values, system_parameter_values)

    def conditional_pdf(
        self,
        variable_value: float,
        parent_variable_values: dict[str, float],
        system_parameter_values: dict[str, float],
    ) -> float:
        """
        Returns the conditional probability of `variable_name` given the parent
        variable values, and system parameter values, where the node is a child
        node.

        By construction, this probability density is taken as a Gaussian, where
        the conditional mean is set as the value of `equation`, scaled and with
        an intercept, with a fixed variance.

        Parameters
        ----------
        variable_value : float
            The value of `variable_name` for which to compute the conditional
            probability.
        parent_variable_values : dict[str, float]
            The values of the parents of the node.
        system_parameter_values : dict[str, float]
            The values of the external system parameters.

        Returns
        -------
        float
            The conditional probability of `variable_name` given the parent
            variable values, and system parameter values.

        Raises
        ------
        ValueError
            If the node is a root node, as conditional probabilities are not
            automatically available for root nodes.
        """
        if self.type == NodeType.ROOT:
            raise ValueError(
                "Root nodes do not have a conditional probability distribution."
            )

        if variable_value < self.domain[0] or variable_value > self.domain[1]:
            return 0

        theoretical_value = self.equation(
            parent_variable_values, system_parameter_values
        )
        adjusted_theoretical_value = (
            self.distribution_parameters["locs"]["intercept"]
            + self.distribution_parameters["locs"]["slope"] * theoretical_value
        )
        return norm.pdf(
            variable_value,
            loc=adjusted_theoretical_value,
            scale=self.distribution_parameters["scale"],
        )

    def marginal_pdf(
        self, variable_value: float, system_parameter_values: dict[str, float] = None
    ) -> float:
        """
        Returns the marginal probability of `variable_name`, where the node is a
        root node.

        This probability density is set by the user on Node instantiation, and
        can take any form.

        Parameters
        ----------
        variable_value : float
            The value to calculate the marginal probability of, which must be
            within the domain of the variable.
        system_parameter_values : dict[str, float]
            The values of the external system parameters.

        Returns
        -------
        float
            The marginal probability of `variable_name`.

        Raises
        ------
        ValueError
            If the node is a child node, as marginal probabilies are not
            automatically available for child nodes.
        """
        if self.type == NodeType.CHILD:
            raise ValueError(
                "Child nodes do not have a predefined marginal probability distribution."
            )

        if variable_value < self.domain[0] or variable_value > self.domain[1]:
            return 0

        return self._marginal_pdf(variable_value, **self.distribution_parameters)


class Factor:
    """
    A generalised factor in a Bayesian network.

    Has a 'scope', which is just a set of variables which enter into the
    factor. Is constructed from a Node object.

    Attributes
    ----------
    scope : list[str]
        The variables which enter into the factor.

    Methods
    -------
    pdf : callable
        Returns the probability density function of the factor, given a set of
        values for the variables in the scope.
    """

    def __init__(self, node: Node) -> None:
        """
        Constructs a Factor object from a Node object.

        Parameters
        ----------
        node : Node
            The node from which to construct the factor.
        """
        self.scope = sorted([node.variable_name] + node.parent_variable_names)

        if node.type == NodeType.ROOT:

            def pdf(
                variable_values: dict[str, float],
                system_parameter_values: dict[str, float],
            ) -> float:
                variable_value = variable_values[node.variable_name]
                return node.marginal_pdf(variable_value, system_parameter_values)

        else:

            def pdf(
                variable_values: dict[str, float],
                system_parameter_values: dict[str, float],
            ) -> float:
                variable_value = variable_values[node.variable_name]
                parent_variable_values = {
                    parent: variable_values[parent]
                    for parent in node.parent_variable_names
                }
                return node.conditional_pdf(
                    variable_value, parent_variable_values, system_parameter_values
                )

        self._pdf = pdf

    def __repr__(self) -> str:
        return f"Factor({self.scope})"

    def __mul__(self, other):
        """
        Returns the product of two factors.

        Any two factors can be multiplied together; they do not need to have
        the same scope.

        Parameters
        ----------
        other : Factor
            The factor to multiply by.

        Returns
        -------
        Factor
            The product of the two factors.
        """
        if not isinstance(other, Factor):
            raise TypeError(
                "Both multiplication operands must be instances of the Factor class."
            )

        # Sort the scopes alphabetically, so that the combined scope is
        # deterministic.
        combined_scope = sorted(list(set(self.scope).union(set(other.scope))))

        def new_pdf(
            variable_values: dict[str, float], system_parameter_values: dict[str, float]
        ) -> float:
            factor1_variable_values = {
                key: value
                for key, value in variable_values.items()
                if key in self.scope
            }
            factor2_variable_values = {
                key: value
                for key, value in variable_values.items()
                if key in other.scope
            }
            return self.pdf(
                factor1_variable_values, system_parameter_values
            ) * other.pdf(factor2_variable_values, system_parameter_values)

        new_factor = Factor.__new__(Factor)
        new_factor.scope = combined_scope
        new_factor._pdf = new_pdf
        return new_factor

    def pdf(
        self,
        variable_values: dict[str, float],
        system_parameter_values: dict[str, float],
    ) -> float:
        """
        Returns the probability density function of the factor, given a set of
        values for the variables in the scope.

        Parameters
        ----------
        variable_values : dict[str, float]
            The values of the variables in the scope.

        Returns
        -------
        float
            The value of the probability density function.

        Raises
        ------
        KeyError
            If the values of the variables in the scope are not provided.
        """
        if not set(self.scope).issubset(set(variable_values.keys())):
            raise KeyError("The values of all variables in the scope must be provided.")

        return self._pdf(variable_values, system_parameter_values)


class BayesianNetwork:
    """
    A high-level class representing a Bayesian network.

    Composed of a set of nodes, which are connected by edges that are defined
    implicitly by the parent variables of each node.

    Parametrised by the external system parameters that are implicitly
    referenced in each node.

    Attributes
    ----------
    nodes : dict[str, Node]
        The nodes in the Bayesian network, indexed by their variable name.
    edges : set[tuple[str, str]]
        The edges in the Bayesian network, as a set of tuples of the form
        (parent, child).
    system_parameter_names : set[str]
        The external system parameters that are implicitly referenced in the
        Bayesian network.

    Methods
    -------
    get_nodes : callable
        Returns a list of the nodes in the Bayesian network.
    get_edges : callable
        Returns a list of the edges in the Bayesian network.
    get_node : callable
        Returns the node with the given variable name.
    """

    def __init__(self, nodes: list[Node]) -> None:
        """
        Creates a Bayesian network, given a predefined set of nodes.

        Parameters
        ----------
        nodes : list[Node]
            The nodes in the Bayesian network.

        Raises
        ------
        ValueError
            If the Bayesian network is not a DAG.
        """
        self.nodes = {node.variable_name: node for node in nodes}

        # Walk through the nodes and pull out the edge relations
        self.edges = set()
        system_parameter_names = []
        for node in self.nodes.values():
            # Add the node's parameters to the network parameters
            system_parameter_names += node.system_parameter_names
            if node.type == "root":
                # If the node is a root node, it has no parents so there are no
                # edges to add. Skip it!
                continue
            else:
                # If the node is a child node, add an edge from each of its parents
                # to the node, after first checking if the parent has been added
                # to the network.
                for parent_variable_name in node.parent_variable_names:
                    if parent_variable_name not in self.nodes.keys():
                        raise ValueError(
                            f"{parent_variable_name} is not in the Bayesian network."
                        )
                    self.edges.add((parent_variable_name, node.variable_name))

        self.system_parameter_names = set(system_parameter_names)

    def __repr__(self):
        return f"BayesianNetwork({self.nodes}, {self.edges})"

    def get_nodes(self) -> list[str]:
        """
        Returns the nodes in the Bayesian network.

        Returns
        -------
        dict[str, Node]
            A dictionary of the nodes in the Bayesian network.
        """
        return self.nodes

    def get_edges(self) -> list[tuple[str, str]]:
        """
        Returns the edges in the Bayesian network.

        Returns
        -------
        list[tuple[str, str]]
            The edges in the Bayesian network.
        """
        return self.edges

    def get_node(self, variable_name: str) -> Node:
        """
        Returns a node from the network, if it exists.

        Parameters
        ----------
        variable_name : str
            The name of the node to select.

        Returns
        -------
        Node
            The selected node object.

        Raises
        ------
        ValueError
            If the Node does not exist in the network.
        """
        if variable_name not in self.nodes.keys():
            raise ValueError(f"{variable_name} is not in the Bayesian network.")
        else:
            return self.nodes[variable_name]

    def joint_pdf(
        self,
        variable_values: dict[str, float],
        system_parameter_values: dict[str, float],
    ) -> float:
        """
        Returns the joint probability of the network, given the values of the
        variables and system parameters.

        Parameters
        ----------
        variable_values : dict[str, float]
            The values of the variables in the network.
        system_parameter_values : dict[str, float]
            The values of the parameters that characterise the network.

        Returns
        -------
        float
            The joint probability of the network.

        Raises
        ------
        TypeError
            If not all variables in the network are present in the variable
            values dictionary.
        TypeError
            If not all system parameters in the network are present in the
            system parameters dictionary.
        ValueError
            If the values of the variables are not within the domain of the
            variable.
        """
        # Check that all variables and parameters have been provided
        if not set(self.nodes.keys()).issubset(set(variable_values.keys())):
            raise TypeError(
                "Not all variables in the network are present in the variable values dictionary."
            )
        if not self.system_parameter_names.issubset(
            set(system_parameter_values.keys())
        ):
            raise TypeError(
                "Not all system parameters in the network are present in the system parameters dictionary."
            )

        # Walk through the nodes and calculate the joint probability
        joint_probability = 1
        for node in self.nodes.values():
            # Check that the value of the variable falls within the domain of
            # the variable, obtained from the node.
            variable_value = variable_values[node.variable_name]
            if variable_value < node.domain[0] or variable_value > node.domain[1]:
                raise ValueError(
                    f"{variable_value} is not within the domain of {node.variable_name}."
                )
            # If the node is a root node, use the marginal PDF...
            if node.type == NodeType.ROOT:
                joint_probability *= node.marginal_pdf(
                    variable_value, system_parameter_values
                )
            # ...else use the conditional PDF.
            else:
                parent_variable_values = {
                    parent_variable_name: variable_values[parent_variable_name]
                    for parent_variable_name in node.parent_variable_names
                }
                joint_probability *= node.conditional_pdf(
                    variable_value, parent_variable_values, system_parameter_values
                )
        return joint_probability

    def _get_joint_factorisation(self) -> list[dict]:
        """
        Returns a representation of the joint probability distribution of the
        network as a list of Factors.

        The ordering of the list has no significance.

        Returns
        -------
        list[Factor]
            A list of Factors, which together represent the joint probability
            distribution of the network.
        """

        factorisation = [Factor(node) for node in self.nodes.values()]
        return factorisation

    def _compute_factor_product(self, factors: list[Factor]) -> Factor:
        """
        Returns the product of a list of Factors.

        Parameters
        ----------
        factors : list[Factor]
            A list of Factors to multiply together.

        Returns
        -------
        Factor
            The product of the Factors.
        """
        return reduce(lambda x, y: x * y, factors)

    def _marginalise_factor(
        self,
        factor: Factor,
        elimination_variable_name: str,
    ) -> Factor:
        """
        Marginalises a factor over the given elimination variable, returning a
        new reduced factor.

        Parameters
        ----------
        factor : Factor
            The factor to marginalise.
        elimination_variable_name : str
            The name of the variable to eliminate.

        Returns
        -------
        Factor
            The marginalised factor.

        Raises
        ------
        ValueError
            If the elimination variable is not in the scope of the factor.
        """
        # Check that the elimination variable lies within the scope of the
        # factor
        if elimination_variable_name not in factor.scope:
            raise ValueError(
                f"{elimination_variable_name} is not in the scope of the factor."
            )

        # Define the scope for the new factor, omitting the elimination variable
        new_scope = [var for var in factor.scope if var != elimination_variable_name]

        # Get the node and domain of the elimination variable
        elimination_variable_node = self.nodes[elimination_variable_name]
        elimination_variable_domain = elimination_variable_node.domain

        # Define the PDF for the new factor
        def new_pdf(
            variable_values: dict[str, float], system_parameter_values: dict[str, float]
        ) -> float:
            def integrand(elimination_variable_value: float) -> float:
                variable_values_with_elimination_variable = {
                    **variable_values,
                    elimination_variable_name: elimination_variable_value,
                }
                return factor.pdf(
                    variable_values_with_elimination_variable, system_parameter_values
                )

            integral_value, _ = quad(integrand, *elimination_variable_domain)
            return integral_value

        # Create the new, marginalised factor
        new_factor = Factor.__new__(Factor)
        new_factor.scope = new_scope
        new_factor._pdf = new_pdf

        return new_factor

    def infer(
        self,
        query_variable_name: str,
        range: tuple[float, float],
        evidence: dict[str, float],
        system_parameter_values: dict[str, float],
    ) -> float:
        """
        Returns the probability that the query variable lies in the given range,
        given a set of evidence.

        Applies the variable elimination algorithm described in Koller and
        Friedman (2009), Algorithm 9.2.

        Parameters
        ----------
        query_variable_name : str
            The name of the query variable for which to compute probability.
        range : tuple[float, float]
            The range of the query variable over which to compute probability.
            Equivalent to the lower and upper bounds of the integral.
        evidence : dict[str, float]
            A dictionary of the evidence, where the keys and values are the
            names and values of these variables.
        system_parameter_values : dict[str, float]
            A dictionary of the external system parameters.

        Returns
        -------
        float
            The probability of the query variable, integrated over the given
            range, given the evidence.

        Raises
        ------
        ValueError
            If the query variable is not in the Bayesian network.
        """
        # Check that the query and evidence are valid
        for variable_name in [query_variable_name] + list(evidence.keys()):
            if variable_name not in self.nodes.keys():
                raise ValueError(f"{variable_name} is not in the Bayesian network.")

        # Define the elimination variables as the set difference between the
        # variables in the network, the query variable and the evidence.
        elimination_variables = (
            set(self.nodes.keys()) - {query_variable_name} - set(evidence.keys())
        )
        # Set an (arbitrary) ordering for the elimination variables
        elimination_variables = list(elimination_variables)

        # Get a representation of the joint probability distribution as a list
        # of factors. We will eliminate the elimination variables from this
        # factorisation.
        factors = self._get_joint_factorisation()
        print(f"Factorisation: {factors}")

        # Walk through each elimination variable and eliminate it from the
        # factorisation, updating the factorisation at each step.
        for n, elimination_variable_name in enumerate(elimination_variables):
            print(
                f"Eliminating variable {elimination_variable_name} ({n+1}/{len(elimination_variables)})"
            )
            # For each elimination variable, we must categorise the factors into
            # two disjoint sets: those that depend on the elimination variable
            # (i.e. factors for which the elimination variable lies in the
            # scope of the factor) and those that do not.
            relevant_factors, irrelevant_factors = [], []
            for factor in factors:
                print(f"\tfactor {factor.scope}...", end="")
                if elimination_variable_name in factor.scope:
                    relevant_factors.append(factor)
                    print(f"\t\trelevant")
                else:
                    irrelevant_factors.append(factor)
                    print(f"\t\tirrelevant")
            # We then multiply all the relevant factors together to obtain a
            # new composite factor $\psi$.
            psi = self._compute_factor_product(relevant_factors)
            # We then marginalise the combined factor $\psi$ over the
            # elimination variable, to obtain a new factor $\tau$.
            tau = self._marginalise_factor(psi, elimination_variable_name)
            # We then replace the relevant factors in the factorisation with
            # the new factor $\tau$.
            factors = irrelevant_factors + [tau]

        # Finally, we multiply all the remaining factors together to obtain the
        # final result, denoted as $\phi$.
        phi = self._compute_factor_product(factors)
        print(f"Final factor: {phi}")

        # We then need to compute the normalisation constant, $\alpha$, which
        # is the integral of $\phi$ over the entire domain of the query
        # variable.
        query_variable_domain = self.nodes[query_variable_name].domain

        def integrand(x: float) -> float:
            return phi.pdf(
                {query_variable_name: x, **evidence}, system_parameter_values
            )

        alpha, _ = quad(integrand, *query_variable_domain)

        # The final conditional pdf is then simply the normalisation of $\phi$
        # by $\alpha$, evaluated using the evidence.
        def final_conditional_pdf(x: float) -> float:
            return (
                phi.pdf({query_variable_name: x, **evidence}, system_parameter_values)
                / alpha
            )

        # We then integrate the conditional pdf over the range of the query
        # variable to obtain the final result.
        prob, _ = quad(final_conditional_pdf, *range)
        return prob
