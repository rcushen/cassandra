from cassandra import Node, Network

import numpy as np

wind_speed_cpd = np.array([0.3, 0.4, 0.3])
wind_speed = Node(
    variable_name="wind_speed",
    parent_nodes=[],
    cpd=wind_speed_cpd
)

temperature_cpd = np.array([0.1] * 10)
temperature = Node(
    variable_name="temperature",
    parent_nodes=[],
    cpd=temperature_cpd
)

torque_cpd = np.random.rand(3, 10, 5)
torque_cpd = torque_cpd / torque_cpd.sum(axis=-1, keepdims=True)
torque = Node(
    variable_name="torque",
    parent_nodes=[wind_speed, temperature],
    cpd=torque_cpd
)

power_cpd = np.random.rand(5, 8)
power_cpd = power_cpd / power_cpd.sum(axis=-1, keepdims=True)
power = Node(
    variable_name="power",
    parent_nodes=[torque],
    cpd=power_cpd
)

turbine = Network([wind_speed, temperature, torque, power])

print("What is the probability of maximal power, given low wind speed and maximal temperature?")
inference = turbine.query({"power": 7}, {"wind_speed": 0, "temperature": 9})
print(inference)

print("What is the probability of low torque, given maximal wind speed and low temperature?")
inference = turbine.query({"torque": 0}, {"wind_speed": 2, "temperature": 0})
print(inference)
