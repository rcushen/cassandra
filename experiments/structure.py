from cassandra.network import Node, BayesianNetwork

from scipy.stats import norm


def torque_equation(variables: dict[str, float], parameters: dict[str, float]) -> float:
    wind_speed, temperature = variables["wind_speed"], variables["temperature"]
    torque_factor = parameters["torque_factor"]
    return torque_factor * (wind_speed**2) / (temperature)


def power_equation(variables: dict[str, float], parameters: dict[str, float]) -> float:
    torque = variables["torque"]
    power_factor = parameters["power_factor"]
    return power_factor * torque


temperature_node = Node(
    variable_name="temperature",
    domain=(-100, 100),
    parent_variable_names=[],
    system_parameter_names=[],
    marginal_pdf=norm.pdf,
    distribution_parameters={"loc": 25, "scale": 10},
)
wind_speed_node = Node(
    variable_name="wind_speed",
    domain=(0, 100),
    parent_variable_names=[],
    system_parameter_names=[],
    marginal_pdf=norm.pdf,
    distribution_parameters={"loc": 25, "scale": 5},
)
torque_node = Node(
    variable_name="torque",
    domain=(-100, 100),
    parent_variable_names=["temperature", "wind_speed"],
    system_parameter_names=["torque_factor"],
    equation=torque_equation,
)
power_node = Node(
    variable_name="power",
    domain=(0, 100),
    parent_variable_names=["torque"],
    system_parameter_names=["power_factor"],
    equation=power_equation,
)

turbine = BayesianNetwork([temperature_node, wind_speed_node, torque_node, power_node])
default_system_parameters = {
    "torque_factor": 0.5,
    "power_factor": 0.5,
}

inference = turbine.infer(
    "torque",
    (10, 12),
    {"temperature": 25, "wind_speed": 25},
    default_system_parameters,
)
print(inference)
