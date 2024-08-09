# Factor

A factor is a generalised representation of a conditional probability distribution from a Bayesian network. It is a function that maps an assignment of values, defined over the 'scope' of the factor, to some real number. In the discrete case, this means just indexing into a big lookup table of values.

Factors can be multiplied to produce composite factors, and can be 'summed out' to remove a variable from the factor's scope. Both of these operations are highly useful in the conttext of inference in Bayesian networks.

Sometimes, a factor may correspond directly to the conditional probability or marginal probability distributions of a node in a Bayesian Network. In this case, the values array represents a conditional probability distribution, in which the last dimensions corresponds to the node's variable domain, and the other dimensions correspond to the parent nodes' variable domains. And crucially, values in the last dimension sum to 1.

But in other cases, factors may not correspond to any probability distribution. For example...

## Operations

### Evaluation

The most basic operation is to evaluate a factor at a given assignment of values. This is done by indexing into the values array with the given assignment.

### Multiplication

The product operation for factors is commutative and associative. It is equivalent to the Hadamard product of the values arrays of the two factors.

Consider the product of two marginal distribution factors, \tau(a) and \tau(b), corresponding to P(a) and P(b). The product of these two factors, \tau(a) \times \tau(b), corresponds to the joint distribution P(a,b), and is obtained as the outer product of the two factors' values arrays. Obviously, this will not be a normalised distribution, as the values in the last dimension do not sum to 1.
