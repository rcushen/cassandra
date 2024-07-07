
class BayesianNetwork:
    """
    A class to represent a Bayesian network, which is a collection of nodes that
    together form a directed graph.
    """
    def __init__(self, nodes):
        self.nodes = nodes

    def query(self, Y, e):
        prob = 0
        return prob

    def get_cardinality(self):
        return len(self.nodes)

    def get_variable_names(self):
        return set([node.variable_name for node in self.nodes])
