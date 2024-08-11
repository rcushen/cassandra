import time

from cassandra.network import Node, BayesianNetwork
from scipy.stats import norm


def test_scaling():
    nodeA = Node(
        "A",
        (-5, 5),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 0, "scale": 1},
    )
    nodeB = Node(
        "B",
        (-2, 8),
        [],
        [],
        marginal_pdf=norm.pdf,
        distribution_parameters={"loc": 3, "scale": 1},
    )
    nodeC = Node(
        "C",
        (-16, 19),
        ["A", "B"],
        ["state"],
        equation=lambda v, p: p["state"] * (2 * v["A"] + 0.5 * v["B"]),
    )
    nodeD = Node(
        "D",
        (-30, 30),
        ["A", "C"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["A"] - v["C"]),
    )
    nodeE = Node(
        "E",
        (-50, 50),
        ["B", "C"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["B"] + v["C"]),
    )
    nodeF = Node(
        "F",
        (-200, 200),
        ["D", "E"],
        ["state"],
        equation=lambda v, p: p["state"] * (v["D"] + v["E"]),
    )

    default_system_parameters = {"state": 1.0}

    network = BayesianNetwork([nodeA])
    start_time = time.time()
    inference1 = network.infer("A", (-1, 1), {}, default_system_parameters)
    end_time = time.time()
    print(f"Time for inference 1: {end_time - start_time}")

    network = BayesianNetwork([nodeA, nodeB, nodeC])
    start_time = time.time()
    inference2 = network.infer("C", (5, 9), {"A": 0, "B": 3}, default_system_parameters)
    end_time = time.time()
    print(f"Time for inference 2: {end_time - start_time}")

    network = BayesianNetwork([nodeA, nodeB, nodeC, nodeD])
    start_time = time.time()
    inference3 = network.infer(
        "D", (-10, 10), {"A": 0, "B": 4}, default_system_parameters
    )
    end_time = time.time()
    print(f"Time for inference 3: {end_time - start_time}")

    network = BayesianNetwork([nodeA, nodeB, nodeC, nodeD, nodeE])
    start_time = time.time()
    inference4 = network.infer(
        "E", (-10, 10), {"A": 0, "B": 4}, default_system_parameters
    )
    end_time = time.time()
    print(f"Time for inference 4: {end_time - start_time}")


test_scaling()
