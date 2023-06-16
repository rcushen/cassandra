from cassandra.network import Node, BayesianNetwork

from scipy.stats import norm

def torque_equation(wind_speed: float, temperature: float) -> float:
    return (wind_speed ** 2) / (temperature)
def power_equation(torque: float) -> float:
    return torque

temperature_node = Node(
    variable_name='temperature',
    domain=(-100, 100),
    parent_variable_names=[],
    marginal_pdf=norm.pdf,
    parameters={'loc': 25, 'scale': 10}
)
wind_speed_node = Node(
    variable_name='wind_speed',
    domain=(0, 100),
    parent_variable_names=[],
    marginal_pdf=norm.pdf,
    parameters={'loc': 25, 'scale': 10}
)

torque_node = Node(
    variable_name='torque',
    domain=(-100, 100),
    parent_variable_names=['temperature', 'wind_speed'],
    equation=torque_equation
)
power_node = Node(
    variable_name='power',
    domain=(0, 100),
    parent_variable_names=['torque'],
    equation=power_equation
)

turbine = BayesianNetwork(
    [temperature_node, wind_speed_node, torque_node, power_node]
)

print(turbine.joint_pdf(
    **{'temperature': 25, 'wind_speed': 25, 'torque': 25, 'power': 25}
))

factors = turbine._get_joint_factorisation()

print(factors[0].scope)
print(factors[0].pdf(temperature=25))
# turbine.infer(
#     query_variable='power',
#     range=(0, 100),
#     evidence={'temperature': 25, 'wind_speed': 25},
# )