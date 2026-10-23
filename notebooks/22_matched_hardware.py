from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from importlib.metadata import version
import argparse
import hashlib
import importlib.util
import json
import random

import pandas as pd
from qiskit import qpy
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mission21", HERE / "21_repetition_qec.py"
)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

SEED = 220031


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_backend(name):
    service = QiskitRuntimeService()
    backend = service.backend(name, use_fractional_gates=False)
    if "if_else" not in backend.target.operation_names:
        raise RuntimeError("Backend does not advertise if_else.")
    return service, backend


def prepare(args, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    _, backend = service_backend(args.backend)
    metadata, circuits = v.suite()

    if args.layout:
        layout = [int(x) for x in args.layout.split(",")]
    else:
        # Choose mapping using one predeclared reference circuit,
        # never from hardware test results.
        reference = v.build("+i", "none", True)
        selector = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
            seed_transpiler=SEED,
        )
        ref_compiled = selector.run(reference)
        layout = ref_compiled.layout.initial_index_layout(
            filter_ancillas=True
        )

    if len(layout) != 5 or len(set(layout)) != 5:
        raise ValueError("Layout must contain five distinct physical qubits.")
    if min(layout) < 0 or max(layout) >= backend.num_qubits:
        raise ValueError("Layout contains an invalid physical qubit.")

    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=layout,
        optimization_level=1,
        seed_transpiler=SEED,
    )
    compiled = pm.run(circuits)

    circuit_info = []
    for meta, circuit in zip(metadata, compiled):
        initial = circuit.layout.initial_index_layout(
            filter_ancillas=True
        )
        if initial != layout:
            raise AssertionError(f"Initial layout changed: {initial}")
        circuit_info.append({
            **meta,
            "initial_layout": initial,
            "final_layout": circuit.layout.final_index_layout(
                filter_ancillas=True
            ),
            "top_level_operation_counts": dict(circuit.count_ops()),
        })

    # This is submission order, not a guarantee of physical scheduling.
    rng = random.Random(SEED)
    pubs, cases = [], []
    for block in range(args.blocks):
        order = list(range(len(compiled)))
        rng.shuffle(order)
        for index in order:
            pubs.append(compiled[index].copy())
            cases.append({
                **metadata[index],
                "block": block,
                "source_circuit": index,
                "pub_index": len(cases),
            })

    output.mkdir(parents=True, exist_ok=True)
    qpy_path = output / "compiled.qpy"
    with qpy_path.open("wb") as handle:
        qpy.dump(pubs, handle)

    properties = backend.properties()
    write_json(
        output / "backend_properties.json",
        None if properties is None else properties.to_dict(),
    )
    write_json(
        output / "backend_configuration.json",
        backend.configuration().to_dict(),
    )
    write_json(output / "circuit_info.json", circuit_info)

    manifest = {
        "prepared_utc": now(),
        "backend": backend.name,
        "shots": args.shots,
        "blocks": args.blocks,
        "initial_layout": layout,
        "seed": SEED,
        "cases": cases,
        "qpy_sha256": checksum(qpy_path),
        "versions": {
            package: version(package)
            for package in ["qiskit", "qiskit-ibm-runtime"]
        },
        "primary_metric": "decoded_full_success",
        "primary_group": ["X0", "X1", "X2"],
        "limitations": [
            "Controlled injected faults; not a break-even experiment.",
            "Initial layout matched; routing may still differ.",
            "Blocks share one job and are not independent hardware sessions.",
            "Publication order is not guaranteed execution chronology.",
            "No postselection; all shots enter success rates.",
            "Q-TRACE diagnosis is not yet selecting the decoder.",
        ],
    }
    write_json(output / "manifest.json", manifest)

    print(f"Prepared on {backend.name}")
    print(f"Initial layout [q0,q1,q2,anc0,anc1]: {layout}")
    print(f"Circuits: {len(pubs)}")
    print(f"Shots per circuit: {args.shots}")
    print(f"Total shots: {len(pubs) * args.shots}")
    print("No job submitted.")


def submit(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("job_id"):
        raise RuntimeError(
            f"Already submitted: {manifest['job_id']}. Use collect."
        )
    if manifest.get("submission_started_utc"):
        raise RuntimeError(
            "A submission was previously attempted. Check IBM jobs before "
            "retrying, to avoid submitting a duplicate."
        )

    qpy_path = output / "compiled.qpy"
    if checksum(qpy_path) != manifest["qpy_sha256"]:
        raise RuntimeError("Prepared circuit file has changed.")

    _, backend = service_backend(manifest["backend"])
    with qpy_path.open("rb") as handle:
        circuits = qpy.load(handle)

    if len(circuits) != len(manifest["cases"]):
        raise RuntimeError("Circuit/manifest count mismatch.")

    manifest["submission_started_utc"] = now()
    write_json(manifest_path, manifest)

    job = SamplerV2(mode=backend).run(
        circuits, shots=manifest["shots"]
    )
    print(f"JOB ID: {job.job_id()}", flush=True)
    manifest["job_id"] = job.job_id()
    write_json(manifest_path, manifest)
    print("Submitted. Use collect after the job completes.")


def collect(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest.get("job_id"):
        raise RuntimeError("No job ID in manifest.")
    if (output / "results.csv").exists():
        raise FileExistsError("Results already collected.")

    service = QiskitRuntimeService()
    job = service.job(manifest["job_id"])
    status = str(job.status()).split(".")[-1].upper()
    print(f"Job {manifest['job_id']}: {status}")

    if status != "DONE":
        print("Results not collected. Check job status before running again.")
        return

    result = job.result()
    if len(result) != len(manifest["cases"]):
        raise RuntimeError("Unexpected number of publications.")

    rows, raw = [], []
    for meta, pub in zip(manifest["cases"], result):
        outs = pub.data.out.get_bitstrings()
        syns = pub.data.syn.get_bitstrings()
        if len(outs) != len(syns) or len(outs) != manifest["shots"]:
            raise RuntimeError("Unexpected shot count.")

        out_counts = dict(Counter(outs))
        syn_counts = dict(Counter(syns))
        joint_counts = dict(Counter(
            f"{syndrome}|{outcome}"
            for syndrome, outcome in zip(syns, outs)
        ))

        rows.append(v.score(meta, out_counts))
        raw.append({
            "case": meta,
            "out": out_counts,
            "syn": syn_counts,
            "joint_syn_out": joint_counts,
            "shot_out": outs,
            "shot_syn": syns,
        })

    write_json(output / "counts.json", raw)
    write_json(output / "job_metrics.json", job.metrics())
    write_json(output / "result_metadata.json", {
        "collected_utc": now(),
        "metadata": result.metadata,
        "pub_metadata": [pub.metadata for pub in result],
    })

    df = pd.DataFrame(rows)
    assert not df.duplicated(
        ["block", "state", "fault", "recovery"]
    ).any()
    df.to_csv(output / "results.csv", index=False)

    metrics = ["logical_success", "decoded_full_success"]
    df.groupby(["fault", "recovery"])[metrics].mean().to_csv(
        output / "summary_by_fault.csv"
    )

    # Matched descriptive differences; no claim that three blocks
    # establish reproducibility across hardware sessions.
    paired = df.pivot(
        index=["block", "state", "fault"],
        columns="recovery",
        values="decoded_full_success",
    )
    paired["delta_recovery_minus_off"] = paired[True] - paired[False]
    paired.to_csv(output / "paired_differences.csv")

    single = df[df["fault"].isin(["X0", "X1", "X2"])]
    block_summary = single.groupby(
        ["block", "recovery"]
    )[metrics].mean()
    block_summary.to_csv(output / "single_x_by_block.csv")

    print("\nSingle-X results by block:")
    print(block_summary.to_string())
    print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--layout",
        help="Optional physical indices: q0,q1,q2,anc0,anc1",
    )
    parser.add_argument("--output", default="results/matched_hardware_22")
    args = parser.parse_args()

    if args.shots < 1 or args.blocks < 1:
        raise ValueError("Shots and blocks must be positive.")

    output = Path(args.output)
    if args.action == "prepare":
        prepare(args, output)
    elif args.action == "submit":
        submit(output)
    else:
        collect(output)


if __name__ == "__main__":
    main()