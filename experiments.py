from cassandra.network import Node, BayesianNetwork

from scipy.stats import norm


def torque_equation(variables: dict[str, float], parameters: dict) -> float:
    wind_speed, temperature = variables["wind_speed"], variables["temperature"]
    torque_factor = parameters["torque_factor"]
    return torque_factor * (wind_speed**2) / (temperature)


def power_equation(variables: dict[str, float], parameters: dict) -> float:
    torque = variables["torque"]
    power_factor = parameters["power_factor"]
    return power_factor * torque


temperature_node = Node(
    variable_name="temperature",
    domain=(-100, 100),
    parent_variable_names=[],
    marginal_pdf=norm.pdf,
    parameters={"loc": 25, "scale": 10},
)
wind_speed_node = Node(
    variable_name="wind_speed",
    domain=(0, 100),
    parent_variable_names=[],
    marginal_pdf=norm.pdf,
    parameters={"loc": 25, "scale": 5},
)

torque_node = Node(
    variable_name="torque",
    domain=(-100, 100),
    parent_variable_names=["temperature", "wind_speed"],
    equation=torque_equation,
)
power_node = Node(
    variable_name="power",
    domain=(0, 100),
    parent_variable_names=["torque"],
    equation=power_equation,
)

turbine = BayesianNetwork([temperature_node, wind_speed_node, torque_node, power_node])

print(turbine.infer("power", (10, 20), {"temperature": 25, "wind_speed": 25}))
