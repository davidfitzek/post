# compare adam and qng optimizer for the barren plateau circuit.
from tqdm import tqdm
from copy import copy
import pennylane as qml
import pennylane.numpy as qnp
from pennylane import numpy as np
from pennylane import (
    AdamOptimizer,
    QNGOptimizer,
)

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# helper functions
def get_datapoint(step, exp_val, label):
    return [
        {
            "optimization steps": step,
            "cost function value": np.round(exp_val, 4),
            "label": f"{label}",
        }
    ]


def pauli_gates(p, wires, rng_key):
    if rng_key is not None:
        np.random.seed(rng_key)
    list_gate_set = []
    for _ in range(p):
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = {i: np.random.choice(gate_set) for i in wires}
        list_gate_set.append(random_gate_sequence)
    return list_gate_set


def get_circuit(list_pauli_gates, H):
    num_layers = len(list_pauli_gates)
    num_qubits = len(list_pauli_gates[0])
    wires = range(num_qubits)
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(params):
        params = np.reshape(params, (num_layers, num_qubits))
        for i in wires:
            qml.RY(np.pi / 4, wires=i)
        for layer in range(num_layers):
            for wire in wires:
                list_pauli_gates[layer][wire](params[layer][wire], wires=wire)
            qml.Barrier(wires=wires)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            qml.Barrier(wires=wires)
        return qml.expval(H)

    return circuit


if __name__ == "__main__":
    # set parameters
    p = 5
    num_qubits = 9
    steps = 150
    seed = 0
    averages = 4
    stepsize = 0.01

    list_optimizer = [
        QNGOptimizer(stepsize=stepsize),
        AdamOptimizer(stepsize=stepsize),
    ]

    # build hamiltonian
    obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
    coeffs = [1.0]
    H = qml.Hamiltonian(coeffs, obs)

    dataset = []
    for k in range(steps):
        dataset += get_datapoint(k, -1, "Ground state")

    # optimize circuit
    for i, optimizer in tqdm(enumerate(list_optimizer)):
        list_pauli_gates = pauli_gates(p=p, wires=range(num_qubits), rng_key=i)
        circuit = get_circuit(list_pauli_gates, H)
        optimizer_id = optimizer.__class__.__name__
        for j in range(averages):
            qnp.random.seed(j)
            init_params = qnp.random.random((p * num_qubits)) * 2 * np.pi
            exp_val = circuit(init_params)
            dataset += get_datapoint(0, exp_val, optimizer_id)
            params = copy(init_params)
            for k in tqdm(range(1, steps + 1)):  # skip index zero.
                params = optimizer.step(circuit, params)
                exp_val = circuit(params)
                dataset += get_datapoint(k, exp_val, optimizer_id)

    # plotting
    df = pd.DataFrame(data=dataset)
    fig, ax = plt.subplots()
    g = sns.lineplot(
        x="optimization steps", y="cost function value", hue="label", data=df, ax=ax
    )
    plt.show()
