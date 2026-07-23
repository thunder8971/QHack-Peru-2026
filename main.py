"""Run the complete OrionQ demonstration."""

from pathlib import Path

from utils.image_processing import assign_priority_order, process_image
from utils.quantum_algo import MAX_QUBITS, run_quantum_demo


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "misc" / "original_human.jpeg"
OUTPUT_PATH = ROOT / "misc" / "original_result.jpeg"


def main() -> None:
    print("Starting OrionQ image processing...")
    zones, _ = process_image(INPUT_PATH, OUTPUT_PATH)
    print(f"Zones found: {len(zones)}")

    quantum_zones, _, _ = run_quantum_demo(zones)
    quantum_zones = assign_priority_order(quantum_zones)
    if len(zones) > MAX_QUBITS:
        print(f"Qiskit demo limited to the top {MAX_QUBITS} zones.")

    print("Attention order:")
    for zone in quantum_zones:
        print(
            f"{zone['priority_order']}. "
            f"{zone['label']} - {zone['color']}"
        )

    print(f"Result saved to: {OUTPUT_PATH}")
    print("OrionQ finished.")


if __name__ == "__main__":
    main()

