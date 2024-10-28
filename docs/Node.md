# Node

A node is the constituent unit of a Bayesian network. It represents a random variable, and has a domain, a name, and a set of parent nodes. The parent nodes are the nodes that directly influence the node's value. The node also has a conditional probability distribution, which represents the conditional probability distribution of the node given its parent nodes.

## Operations

### Cardinality

The cardinality of a node is the number of possible values that the node can take. This is equivalent to the length of the domain array.

### Conditional Probability Distribution

If all the values of the parent nodes of a node are known, the node can be evaluated to produce a conditional probability distribution. This is done by looking up the values in the conditional probability distribution table, using the values of the parent nodes as indices.

### Conditional Probability

If we know all relevant values, we can evaluate the node to produce a conditional probability.

### Factors

A node can be converted to a factor, which represents the conditional probability distribution of the node given its parent nodes. This is useful for performing inference in Bayesian networks.

## Notes

* Because nodes must reference their parent nodes upon instantiation, all nodes must be created in a topological order. This was an implementation decision, made because it allows nodes to exist without reference to a network.
