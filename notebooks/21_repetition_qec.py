from pathlib import Path
import argparse
import csv
import json

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


STATES = ["0", "1", "+", "-", "+i", "-i"]
FAULTS = ["none", "X0", "X1", "X2", "X0X1", "Z0"]


def build(state, fault, recovery):
    data = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    out = ClassicalRegister(3, "out")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(data, anc, out, syn)

    # Prepare the input on q0.
    if state == "1":
        qc.x(data[0])
    elif state in ["+", "-", "+i", "-i"]:
        qc.h(data[0])
        if state == "-":
            qc.z(data[0])
        elif state == "+i":
            qc.s(data[0])
        elif state == "-i":
            qc.sdg(data[0])

    # Encode alpha|0> + beta|1> as alpha|000> + beta|111>.
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.barrier()

    # Controlled fault injection for mechanism verification.
    if fault == "X0X1":
        qc.x(data[0])
        qc.x(data[1])
    elif fault.startswith("X"):
        qc.x(data[int(fault[1])])
    elif fault == "Z0":
        qc.z(data[0])
    qc.barrier()

    # syn[0] = q0 XOR q1; syn[1] = q1 XOR q2.
    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])
    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])
    qc.measure(anc, syn)

    # Register integer = syn[0] + 2*syn[1].
    # Recovery receives measured syndrome, not the injected-fault label.
    if recovery:
        for syndrome, qubit in [(1, 0), (3, 1), (2, 2)]:
            with qc.if_test((syn, syndrome)):
                qc.x(data[qubit])
    qc.barrier()

    # Decode. Successful recovery leaves q1=q2=0.
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    # Measure q0 in the input state's eigenbasis.
    if state in ["+", "-"]:
        qc.h(data[0])
    elif state in ["+i", "-i"]:
        qc.sdg(data[0])
        qc.h(data[0])
    qc.measure(data, out)
    return qc


def suite():
    metadata, circuits = [], []
    for state in STATES:
        for fault in FAULTS:
            for recovery in [False, True]:
                metadata.append({
                    "state": state,
                    "fault": fault,
                    "recovery": recovery,
                })
                circuits.append(build(state, fault, recovery))
    return metadata, circuits


def score(metadata, counts):
    expected = int(metadata["state"] in ["1", "-", "-i"])
    total = sum(counts.values())
    logical_ok = 0
    full_ok = 0
    for bitstring, count in counts.items():
        value = int(bitstring.replace(" ", ""), 2)
        if (value & 1) == expected:
            logical_ok += count
        # out = q2 q1 q0; q1=q2=0 verifies return to code space
        # after decoding, in addition to the q0 eigenbasis outcome.
        if value == expected:
            full_ok += count

    return {
        **metadata,
        "shots": total,
        "logical_success": logical_ok / total,
        "decoded_full_success": full_ok / total,
    }


def save_results(output, metadata, all_counts, syndromes, ideal=False):
    rows = [score(meta, counts) for meta, counts in zip(metadata, all_counts)]

    if ideal:
        for row in rows:
            if row["recovery"] and row["fault"] in ["none", "X0", "X1", "X2"]:
                assert row["decoded_full_success"] == 1.0, row
            if (
                row["recovery"]
                and row["fault"] == "Z0"
                and row["state"] in ["+", "-", "+i", "-i"]
            ):
                assert row["logical_success"] == 0.0, row
        print("PASS: single-X recovery and Z-error negative controls.")

    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output / "counts.json").write_text(json.dumps([
        {"case": meta, "out": out, "syn": syn}
        for meta, out, syn in zip(metadata, all_counts, syndromes)
    ], indent=2))
    print(f"Saved results to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["aer", "hardware"], default="aer")
    parser.add_argument("--backend", help="IBM backend name")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--job-id", help="Retrieve a previously submitted job")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.shots < 1:
        raise ValueError("--shots must be positive.")
    if args.job_id and args.submit:
        raise ValueError("Use either --submit or --job-id.")

    output = Path(args.output or f"results/repetition_21_{args.mode}")
    metadata, circuits = suite()

    if args.job_id:
        if args.mode != "hardware":
            raise ValueError("--job-id requires --mode hardware.")
        from qiskit_ibm_runtime import QiskitRuntimeService

        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("job_id") != args.job_id:
            raise ValueError("Job ID does not match this output manifest.")
        if (output / "results.csv").exists():
            raise FileExistsError("Results already exist.")

        service = QiskitRuntimeService()
        result = service.job(args.job_id).result()
        metadata = manifest["cases"]
        if len(result) != len(metadata):
            raise ValueError("Unexpected number of job results.")

        save_results(
            output, metadata,
            [pub.data.out.get_counts() for pub in result],
            [pub.data.syn.get_counts() for pub in result],
        )
        return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    if args.mode == "aer":
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=210031
        )
        compiled = pm.run(circuits)
        result = backend.run(
            compiled, shots=args.shots, seed_simulator=210031
        ).result()

        out_counts, syn_counts = [], []
        for i in range(len(circuits)):
            out, syn = {}, {}
            for bits, count in result.get_counts(i).items():
                # Classical registers: out first (low 3 bits), syn second.
                value = int(bits.replace(" ", ""), 2)
                out_key = format(value & 7, "03b")
                syn_key = format((value >> 3) & 3, "02b")
                out[out_key] = out.get(out_key, 0) + int(count)
                syn[syn_key] = syn.get(syn_key, 0) + int(count)
            out_counts.append(out)
            syn_counts.append(syn)

        output.mkdir(parents=True, exist_ok=True)
        save_results(output, metadata, out_counts, syn_counts, ideal=True)
        return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    if args.backend:
        backend = service.backend(args.backend, use_fractional_gates=False)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=5,
            dynamic_circuits=True,
            use_fractional_gates=False,
        )

    if "if_else" not in backend.target.operation_names:
        raise RuntimeError("Selected backend does not advertise if_else.")

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=210031
    )
    compiled = pm.run(circuits)

    print(f"Backend: {backend.name}")
    print(f"Circuits: {len(compiled)}")
    print(f"Shots/circuit: {args.shots}")
    print(f"Total circuit shots: {len(compiled) * args.shots}")
    print("Hardware noise is present in addition to injected faults.")

    if not args.submit:
        print("Preparation complete. No hardware job submitted.")
        print("Run again with --backend <name above> --submit to execute.")
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": backend.name,
        "shots": args.shots,
        "cases": metadata,
        "scope": "One-round repetition-code mechanism experiment",
        "injected_faults": FAULTS,
        "note": "Not a break-even benchmark or a novel QEC code.",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))

    job = SamplerV2(mode=backend).run(compiled, shots=args.shots)
    manifest["job_id"] = job.job_id()
    path.write_text(json.dumps(manifest, indent=2))

    print(f"Submitted job: {job.job_id()}")
    print(
        "Retrieve with:\n"
        f"python notebooks/21_repetition_qec.py --mode hardware "
        f"--job-id {job.job_id()} --output {output}"
    )


if __name__ == "__main__":
    main()