import itertools
from scipy.integrate import quad
from scipy.stats import norm

class Node:
    def __init__(
            self,
            variable_name: str,
            domain: tuple[float, float],
            parent_variable_names: list[str],
            equation: callable = None,
            marginal_pdf: callable = None,
            parameters: dict = None
        ) -> None:
        '''
        Creates a Node for a Bayesian Network, identified with the variable
        `variable_name`.

        This can be either a root node, for which a marginal probability density
        function must be defined, or a child node, for which a conditional
        probability density function must be defined.

        Is uniquely identified by variable name.

        Parameters
        ----------
        variable_name : str
            The name of the variable identified with the node.
        domain : tuple[float, float]
            The domain of the variable, defined as an interval.
        parent_variable_names: list[str]
            The names of the parent variables.
        equation: callable
            If the node is a child node, the physical equation which determines
            the value of the node variable, written as a function of its
            parents. The function signature must be equal to
            `*parent_variable_names`, and it must return a float within the
            domain of the variable.
        marginal_pdf : callable
            If the node is a root node, the marginal probability density
            function that defines the node.
        parameters : dict
            If the node is a child node, a dictionary of 'locs' and 'scale'
            parameters to associate with the conditional Gaussian probability
            density function; if the node is a root node, an arbitrary list of
            parameters to characterise the marginal density function.

        Raises
        ------
        ValueError
            If the Node is not correctly specified.
        '''
        self.variable_name = variable_name
        self.domain = domain
        self.parent_variable_names = parent_variable_names
        if len(parent_variable_names) == 0:
            # If there are no parent variables, the node is a root node
            self.node_type = 'root'
            if marginal_pdf is None or parameters is None:
                raise ValueError('Root nodes must have a marginal probability distribution and parameters.')
            self.marginal_pdf_alias = marginal_pdf
            self.parameters = parameters
        else:
            # Else the node is a child node
            self.node_type = 'child'
            if equation is None:
                raise ValueError('Child nodes must have an equation.')
            self.equation = equation
            if parameters is None:
                self.parameters = {
                    'locs': {
                        'intercept': 0,
                        'slope': 1
                    },
                    'scale': 1
                }
            else:
                self.parameters = parameters

    def __repr__(self):
        if self.node_type == 'child':
            return f'Node({self.variable_name}, {self.parent_variable_names}, {self.equation}, {self.parameters})'
        else:
            return f'Node({self.variable_name}, {self.parameters})'

    def conditional_pdf(
            self,
            x: float,
            parent_values: dict[str, float]
        ) -> float:
        '''
        Returns the conditional probability of `variable_name` given the parent
        values, where the node is a child node.

        This probability density is taken as a Gaussian, where the conditional
        mean is set as the value of `equation`, scaled and with an intercept,
        with a fixed variance.

        Parameters
        ----------
        x : float
            The value of `variable_name` for which to compute the conditional
            probability.
        parent_values : dict[str, float]
            The values of the parents of the node.

        Returns
        -------
        float
            The conditional probability of x given the parent values.

        Raises
        ------
        ValueError
            If the node is a root node, as conditional probabilities are not
            automatically available for root nodes.
        '''
        if self.node_type == 'root':
            raise ValueError('Root nodes do not have a conditional probability distribution.')

        theoretical_value = self.equation(**parent_values)
        adjusted_theoretical_value = (
            self.parameters['locs']['intercept'] +
            self.parameters['locs']['slope'] * theoretical_value
        )
        return norm.pdf(x,
            loc=adjusted_theoretical_value,
            scale=self.parameters['scale']
        )

    def marginal_pdf(self, x: float) -> float:
        '''
        Returns the marginal probability of `variable_name`, where the node is a
        root node.

        This probability density is set by the user on Node instantiation, and
        can take any form.

        Parameters
        ----------
        x : float
            The value to calculate the marginal probability of.

        Returns
        -------
        float
            The marginal probability of x.

        Raises
        ------
        ValueError
            If the node is a child node, as marginal probabilies are not
            automatically available for child nodes.
        '''
        if self.node_type == 'child':
            raise ValueError('Child nodes do not have a predefined marginal probability distribution.')

        return self.marginal_pdf_alias(x, **self.parameters)

class BayesianNetwork:
    def __init__(self, nodes: list[Node]) -> None:
        '''
        Creates a Bayesian network, given a predefined set of nodes.

        Parameters
        ----------
        nodes : list[Node]
            The nodes in the Bayesian network.

        Raises
        ------
        ValueError
            If the Bayesian network is not a DAG.
        '''
        self.nodes = {node.variable_name: node for node in nodes}

        # Walk through the nodes and pull out the edge relations
        self.edges = set()
        for node in self.nodes.values():
            if node.node_type == 'root':
                # If the node is a root node, it has no parents so there are no
                # edges to add. Skip it!
                continue
            else:
                # If the node is a child node, add an edge from each of its parents
                # to the node, after first checking if the parent has been added
                # to the network.
                for parent_variable_name in node.parent_variable_names:
                    if parent_variable_name not in self.nodes.keys():
                        raise ValueError(f'{parent_variable_name} is not in the Bayesian network.')
                    self.edges.add((parent_variable_name, node.variable_name))

    def __repr__(self):
        return f'BayesianNetwork({self.nodes}, {self.edges})'

    def get_nodes(self) -> list[str]:
        '''
        Returns the nodes in the Bayesian network.

        Returns
        -------
        dict[str, Node]
            A dictionary of the nodes in the Bayesian network.
        '''
        return self.nodes

    def get_edges(self) -> list[tuple[str, str]]:
        '''
        Returns the edges in the Bayesian network.

        Returns
        -------
        list[tuple[str, str]]
            The edges in the Bayesian network.
        '''
        return self.edges

    def get_node(self, variable_name: str) -> Node:
        '''
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
        '''
        if variable_name not in self.nodes.keys():
            raise ValueError(f'{variable_name} is not in the Bayesian network.')
        else:
            return self.nodes[variable_name]

    def _recursive_elimination(self, factors, evidence):
        """
        Recursively performs variable elimination on a list of factors until the list is empty.
        Returns the final factor obtained after eliminating all the variables using the given evidence.

        Parameters
        ----------
        factors : list[callable]
            A list of probability distribution functions (pdfs) to eliminate variables from.
        evidence : dict[str, float]
            A dictionary of the fixed variable values, where keys are variable names and values are their values.

        Returns
        -------
        callable
            The final factor obtained after eliminating all variables using the given evidence.
        """
        if len(factors) == 0:
            # If the factors list is empty, return a trivial function that returns 1.0.
            return lambda **kwargs: 1.0

        # Pick the first factor from the list and eliminate its variable from it using integration,
        # taking into account the evidence.
        factor = factors.pop(0)
        variable_name = self._get_elimination_variable(factor)

        if variable_name in evidence:
            # If the variable being eliminated is in the evidence, we don't need to integrate.
            # Instead, we directly substitute the observed value from the evidence dictionary.
            marginalized_factor = lambda **kwargs: factor(**kwargs, **{variable_name: evidence[variable_name]})
        else:
            def marginalized_factor(**kwargs):
                def _integrand(value, **kwargs):
                    kwargs.update({variable_name: value})
                    return factor(**kwargs)

                # Integrate out the variable from the factor by using quadrature integration.
                integration_domain = self.nodes[variable_name].domain
                return quad(_integrand, *integration_domain, args=(**kwargs,))[0]

        # Eliminate the remaining variables in the factors list.
        remaining_factor = self._recursive_elimination(factors, evidence)

        # Combine the marginalized_factor and remaining_factor for the eliminated variables.
        return lambda **kwargs: marginalized_factor(**kwargs) * remaining_factor(**kwargs)

    def _get_elimination_variable(self, factor):
        """
        A helper function to extract the name of the variable to be eliminated for a given factor.

        Parameters
        ----------
        factor : callable
            A probability distribution function (pdf) from which the variable should be extracted.

        Returns
        -------
        str
            The name of the variable to be eliminated.
        """
        for variable_name, node in self.nodes.items():
            if variable_name not in node.parent_variable_names and variable_name != node.variable_name:
                continue
            if variable_name in factor.__code__.co_varnames:
                return variable_name
        return None

    def infer(
            self,
            query_variable: str,
            range: tuple[float, float],
            evidence: dict[str, float]
        ) -> float:
        """
        Returns the probability that the query variable is in the given range,
        given a set of evidence.

        Parameters
        ----------
        query_variable : str
            The name of the variable to calculate the probability of.
        range : tuple[float, float]
            The range of the query variable to calculate the probability of.
        evidence : dict[str, float]
            A dictionary of the evidence, where the keys are the names of the
            variables and the values the values of these variables.

        Returns
        -------
        float
            The probability of the query given the evidence.

        Raises
        ------
        ValueError
            If the query is not in the Bayesian network.
        """
        if query_variable not in self.nodes.keys():
            raise ValueError(f'{query_variable} is not in the Bayesian network.')
        for evidence_variable in evidence.keys():
            if evidence_variable not in self.nodes.keys():
                raise ValueError(f'{evidence_variable} is not in the Bayesian network.')

        # Perform variable elimination on the factors in the network.
        factors = [
            node.marginal_pdf if node.node_type == 'root'
            else node.conditional_pdf
            for node
            in self.nodes.values()
        ]
        final_factor = self._recursive_elimination(factors, evidence)

        # Integrate the final factor over the given range of the query variable.
        def _integrand(value):
            return final_factor(**{query_variable: value})

        integration_domain = range
        result, error = quad(_integrand, *integration_domain)
        return result
