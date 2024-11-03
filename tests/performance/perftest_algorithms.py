import pytest
import numpy as np
import random
import pprint
import time
import psutil
import os
import networkx as nx

import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List, Tuple, Dict, Set

from cassandra.core import Node, Network

# Key functions
def generate_random_graph(n_nodes: int, connection_prob: float) -> List[List[any]]:
    """
    Generates a random directed acyclic graph with n nodes.

    The result is represented as a list of pairs [node, [parent1, parent2, ...]]
    which have been topologically sorted.

    Args:
    - n_nodes: an integer representing the number of nodes in the graph
    - connection_prob: a float representing the probability of a connection between two nodes

    Raises:
    - None

    Returns:
    - a list of pairs, where each pair represnts a node and its parents
    """
    # Create the graph, represented using edges
    graph_nodes: List[int] = [n for n in range(n_nodes)]
    graph_edges: List[Tuple[int, int]] = []
    for n1 in graph_nodes:
        for n2 in graph_nodes:
            nodes_match = n1 == n2
            passes_threshold = random.random() < connection_prob
            edge_induced_cycle = induces_cycle(graph_edges, (n1, n2))
            if not nodes_match and passes_threshold and not edge_induced_cycle:
                graph_edges.append((n1, n2))

    # Convert the graph to a node representation
    node_representation = []
    for parent, child in graph_edges:
        # If we have never seen the parent, we need to add it to the node
        # representation
        if parent not in [c for c, _ in node_representation]:
            node_representation.append([parent, []])

        # If we have already seen the child, we just need to add another
        # parent to the node representation
        if child in [c for c, _ in node_representation]:
            for c, parents in node_representation:
                if c == child:
                    parents.append(parent)
                    break
        # Else we need to create the child, with a connection to the parent
        else:
            node_representation.append([child, [parent]])

    # Sort the node representation topologically
    sorted_node_representation = topological_sort(node_representation, rev=True)

    return sorted_node_representation

def construct_network_from_graph(graph: List[List[any]], n_values: int) -> Network:
    """
    Constructs a network from an abstract graph object.
    """
    nodes = []
    for node, parents in graph:
        if len(parents) == 0:
            cpd = create_cpd(n_values, 0)
            nodes.append(Node(f"V{node}", [], cpd))
        else:
            cpd = create_cpd(n_values, len(parents))

            existing_parent_nodes = []
            for parent in parents:
                for n in nodes:
                    if n.variable_name == f"V{parent}":
                        existing_parent_nodes.append(n)
                        break

            nodes.append(Node(f"V{node}", [n for n in existing_parent_nodes], cpd))

    network = Network(nodes)

    return network

# Helper functions
def induces_cycle(existing_graph_edges: List[Tuple[int, int]], new_edge: Tuple[int, int]) -> bool:
    """
    Checks if adding a new edge to a graph induces a cycle, thus breaking the
    acyclic invariant.

    If the graph is already cyclic, then adding a new edge will always induce a cycle,
    so the function will return True.

    Args:
    - existing_graph_edges: a list of tuples representing the directed edges of a graph
    - new_edge: a tuple representing a new directed edge to be added to the graph

    Returns:
    - a boolean indicating whether adding the new edge induces a cycle
    """
    # Build the graph as an adjacency list
    graph: Dict[int, Set[int]] = {}
    for src, dest in existing_graph_edges + [new_edge]:
        if src not in graph:
            graph[src] = set()
        graph[src].add(dest)
        if dest not in graph:
            graph[dest] = set()

    # Perform DFS to detect cycles
    def dfs(node: int, visited: Set[int], recursion_stack: Set[int]) -> bool:
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor, visited, recursion_stack):
                    return True
            elif neighbor in recursion_stack:
                return True

        recursion_stack.remove(node)
        return False

    # Check for cycles starting from each node
    visited: Set[int] = set()
    for node in graph:
        if node not in visited:
            if dfs(node, visited, set()):
                return True

    return False

def topological_sort(graph: List[List[any]], rev=False) -> List[List[any]]:
    """
    Sorts a graph topologically.
    Args:
    - graph: a list of pairs [node, [parent1, parent2, ...]]
    Returns:
    - A list of nodes in topological order
    """
    # Create a dictionary to store the graph
    graph_dict = {node: set(parents) for node, parents in graph}

    # Create a set of all nodes
    all_nodes = set(graph_dict.keys()).union(*graph_dict.values())

    # Create a dictionary to store in-degrees
    in_degree = defaultdict(int)
    for parents in graph_dict.values():
        for parent in parents:
            in_degree[parent] += 1

    # Initialize queue with nodes that have no incoming edges
    queue = [node for node in all_nodes if in_degree[node] == 0]

    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree for all neighbors
        for neighbor in graph_dict.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if there's a cycle
    if len(result) != len(all_nodes):
        raise ValueError("Graph contains a cycle")

    sorted_graph = [(node, list(graph_dict[node])) for node in result]

    if rev:
        return sorted_graph[::-1]
    else:
        return sorted_graph

def create_cpd(n_values: int, n_parents: int, distribution: str = "uniform") -> np.ndarray:
    """
    Creates a discrete conditional probability distribution for a node, given
    a number of possible states and a number of parent nodes.

    The CPD is structured such that the last dimension always sums to 1, since
    this is the convention used in the rest of the codebase.

    Args:
    - n_values: an integer representing the number of possible states of the node
    - n_parents: an integer representing the number of parent nodes
    - distribution: a string representing the distribution to use

    Returns:
    - a numpy array representing the conditional probability distribution
    """

    if distribution == "uniform":
        cpd = np.random.rand(*([n_values] + [n_values] * n_parents))
        cpd /= np.sum(cpd, axis=-1, keepdims=True)
        return cpd
    else:
        raise ValueError("Invalid distribution type")

# Tests
def test__network_scaling():
    """
    Tests the performance of the inference algorithms on random networks of
    increasing size.
    """
    random.seed(0)
    os.makedirs('./tests/performance/outputs', exist_ok=True)

    # Set configuration parameters
    connection_prob = 0.5
    n_values = 10
    # Set test parameters
    min_graph_size = 5
    max_graph_size = 9
    n_trials = 10

    # Iterate over graph sizes
    query_times = []
    for n_nodes in range(min_graph_size, max_graph_size + 1):
        print(f"Testing graph size {n_nodes}")

        query_times = []
        for trial in range(n_trials):
            trial_times = []

            # Generate a random graph
            random_graph = generate_random_graph(n_nodes, connection_prob)

            # Construct a network from this graph
            network = construct_network_from_graph(random_graph, n_values)

            # Compute the marginal probability of each node in the graph
            for variable_name in network.nodes.keys():
                start_time = time.time()
                marginal = network.query({variable_name: 0})
                end_time = time.time()
                query_time = end_time - start_time

                trial_times.append(query_time)

            mean_trial_time = np.mean(trial_times)

            # Save a visualisation of the random network
            plt.figure(figsize=(10, 8))

            G = nx.DiGraph()

            all_nodes = set()
            for node, parents in random_graph:
                all_nodes.add(node)
                all_nodes.update(parents)
            G.add_nodes_from(all_nodes)

            for node, parents in random_graph:
                for parent in parents:
                    G.add_edge(parent, node)

            pos = nx.kamada_kawai_layout(G)

            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

            plt.title(f"{n_nodes} Nodes - Trial #{trial + 1}\n"
                     f"Mean query time: {mean_trial_time:.6f} seconds",
                     pad=20)

            plt.savefig(f'./tests/performance/outputs/dag_{n_nodes}_nodes_trial_{trial + 1}.png',
                       bbox_inches='tight',
                       dpi=300)
            plt.close()


    assert True

test__network_scaling()
