#!/usr/bin/env python3
"""Read-only audit of saved M25 circuits. No IBM connection or submission."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qiskit import qpy


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(qc):
    """Ordered operations outside control-flow blocks, with physical wires."""
    return [
        (
            item.operation.name,
            tuple(str(p) for p in item.operation.params),
            tuple(qc.find_bit(q).index for q in item.qubits),
            tuple(qc.find_bit(c).index for c in item.clbits),
        )
        for item in qc.data
        if item.operation.name != "if_else"
    ]


def syndrome_prefix(qc):
    ops = signature(qc)
    barriers = [i for i, item in enumerate(ops) if item[0] == "barrier"]
    if len(barriers) < 3:
        raise ValueError("Expected M24 barrier structure is missing.")
    return ops[:barriers[2] + 1]


def recovery_signature(qc):
    result = {}
    for item in qc.data:
        op = item.operation
        if op.name != "if_else":
            continue

        condition_value = int(op.condition[1])
        blocks = []
        for block in op.blocks:
            instructions = []
            for inner in block.data:
                physical_qubits = tuple(
                    qc.find_bit(
                        item.qubits[block.find_bit(q).index]
                    ).index
                    for q in inner.qubits
                )
                instructions.append((
                    inner.operation.name,
                    tuple(str(p) for p in inner.operation.params),
                    physical_qubits,
                ))
            blocks.append(tuple(instructions))
        result[condition_value] = tuple(blocks)
    return result


def two_qubit_gates(qc):
    """Exclude barriers/control-flow containers; include gates in branches."""
    count = 0
    for item in qc.data:
        op = item.operation
        blocks = getattr(op, "blocks", ())
        if blocks:
            count += sum(two_qubit_gates(block) for block in blocks)
        elif op.num_qubits == 2 and op.name not in {
            "barrier", "measure", "reset", "delay"
        }:
            count += 1
    return count


def load_phase(folder, phase):
    plan = read_json(folder / f"{phase}_plan.json")
    path = folder / f"{phase}.qpy"
    if digest(path) != plan["qpy_sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    with path.open("rb") as f:
        circuits = qpy.load(f)
    if len(circuits) != len(plan["pubs"]):
        raise ValueError("Circuit/metadata count mismatch.")
    return circuits, plan["pubs"]


def audit(folder):
    calibration, cal_meta = load_phase(folder, "cal")
    tests, test_meta = load_phase(folder, "test")

    pairs = defaultdict(dict)
    for qc, row in zip(tests, test_meta):
        key = (
            row["environment"], row["block"],
            row["state"], row["injected_mask"],
        )
        if row["method"] in pairs[key]:
            raise ValueError(f"Duplicate method for {key}")
        pairs[key][row["method"]] = qc

    counts = Counter()
    branch_types = defaultdict(set)

    for key, methods in pairs.items():
        if set(methods) != {"standard", "learned"}:
            raise ValueError(f"Missing paired circuit: {key}")

        standard = methods["standard"]
        learned = methods["learned"]
        counts["test_pairs"] += 1
        counts["same_outside_recovery"] += (
            signature(standard) == signature(learned)
        )
        counts["same_two_qubit_gate_count"] += (
            two_qubit_gates(standard) == two_qubit_gates(learned)
        )

        if key[0] == "single_dominant":
            counts["single_dominant_pairs"] += 1
            counts["single_dominant_equal"] += standard == learned

        for method, qc in methods.items():
            branch_types[(key[0], method)].add(
                repr(recovery_signature(qc))
            )

    for qc, row in zip(calibration, cal_meta):
        key = (
            row["environment"], row["block"],
            "0", row["injected_mask"],
        )
        if key not in pairs:
            counts["calibration_without_matching_test_mask"] += 1
            continue

        counts["calibration_test_prefix_pairs"] += 1
        counts["same_calibration_test_prefix"] += (
            syndrome_prefix(qc)
            == syndrome_prefix(pairs[key]["standard"])
        )

    print(f"\n=== {folder.name} ===")
    print("Calibration CZ counts:",
          dict(Counter(q.count_ops().get("cz", 0) for q in calibration)))
    print("Test CZ counts:",
          dict(Counter(q.count_ops().get("cz", 0) for q in tests)))

    for name, value in sorted(counts.items()):
        print(f"{name}: {value}")

    for key, values in sorted(branch_types.items()):
        print("Recovery", key, sorted(values))

    if counts["same_outside_recovery"] != counts["test_pairs"]:
        raise AssertionError("Paired circuits differ outside recovery.")
    if counts["single_dominant_equal"] != counts["single_dominant_pairs"]:
        raise AssertionError("Single-dominant control circuits differ.")
    if (
        counts["same_calibration_test_prefix"]
        != counts["calibration_test_prefix_pairs"]
    ):
        raise AssertionError("Calibration/test prefix mismatch.")

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/confirmation_25"),
    )
    args = parser.parse_args()

    total = Counter()
    for index in range(1, 4):
        total.update(audit(args.input / f"run_{index:02d}"))

    print("\n=== TOTAL ===")
    for name, value in sorted(total.items()):
        print(f"{name}: {value}")
    print("PASS: saved-circuit comparisons.")
    print("This does not verify backend timing or native-noise break-even.")


if __name__ == "__main__":
    main()