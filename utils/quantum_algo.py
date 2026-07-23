"""Small Qiskit encoding demonstration for OrionQ zone priorities."""

from __future__ import annotations

from math import asin, sqrt
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


Zone = dict[str, Any]

MAX_QUBITS = 12
SHOTS = 512


def run_quantum_demo(
    zones: list[Zone],
    shots: int = SHOTS,
) -> tuple[list[Zone], QuantumCircuit, dict[str, int]]:
    """Encode normalized priorities as qubit probabilities and measure them."""
    selected_zones = [
        dict(zone)
        for zone in sorted(
            zones,
            key=lambda zone: zone["priority"],
            reverse=True,
        )[:MAX_QUBITS]
    ]
    circuit = QuantumCircuit(len(selected_zones))

    if not selected_zones:
        return selected_zones, circuit, {}
    if shots <= 0:
        raise ValueError("shots must be greater than zero.")

    for qubit, zone in enumerate(selected_zones):
        priority = min(1.0, max(0.0, float(zone["priority"])))
        theta = 2 * asin(sqrt(priority))
        circuit.ry(theta, qubit)

    circuit.measure_all()
    simulator = AerSimulator(seed_simulator=42)
    compiled_circuit = transpile(circuit, simulator)
    counts = simulator.run(compiled_circuit, shots=shots).result().get_counts()

    total_measurements = sum(counts.values())
    one_counts = [0] * len(selected_zones)
    for bit_string, count in counts.items():
        bits = bit_string.replace(" ", "")
        for qubit in range(len(selected_zones)):
            if bits[-1 - qubit] == "1":
                one_counts[qubit] += count

    for qubit, zone in enumerate(selected_zones):
        zone["quantum_score"] = one_counts[qubit] / total_measurements

    ranked_indices = sorted(
        range(len(selected_zones)),
        key=lambda index: (
            -selected_zones[index]["quantum_score"],
            selected_zones[index]["label"],
        ),
    )
    for rank, index in enumerate(ranked_indices, start=1):
        selected_zones[index]["quantum_rank"] = rank

    return selected_zones, circuit, dict(counts)

