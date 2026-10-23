#!/usr/bin/env python3
"""Q-TRACE M24: independently calibrated, locked hardware decoder.

Controlled injected-X benchmark, with natural hardware noise.
No postselection. Calibration cost is reported separately.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qpy
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

STATES = ["0", "1", "+", "-", "+i", "-i"]
STANDARD = [0, 1, 4, 2]  # Indexed by integer syndrome.

# Harness only: these probabilities never enter learn_decoder().
ENVIRONMENTS = {
    "single_dominant": [.70, .08, .08, .01, .10, .01, .01, .01],
    "correlated": [.65, .03, .03, .22, .02, .02, .02, .01],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    """Atomic replacement of a JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def syndrome(mask):
    b0, b1, b2 = [(mask >> i) & 1 for i in range(3)]
    return (b0 ^ b1) + 2 * (b1 ^ b2)


def prepare_state(qc, q, state):
    if state == "1":
        qc.x(q)
    elif state != "0":
        qc.h(q)
        if state == "-":
            qc.z(q)
        elif state == "+i":
            qc.s(q)
        elif state == "-i":
            qc.sdg(q)


def circuit(state, mask, decoder=None, calibration=False):
    data = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "a")
    out = ClassicalRegister(3, "out")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(data, anc, out, syn)

    prepare_state(qc, data[0], state)
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.barrier()

    for i in range(3):
        if (mask >> i) & 1:
            qc.x(data[i])
    qc.barrier()

    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])
    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])
    qc.measure(anc, syn)
    qc.barrier()

    if calibration:
        # Logical 0 is encoded as 000. Observe the data word directly.
        # No decoding and no recovery in calibration.
        qc.measure(data, out)
        return qc

    for s, recovery in enumerate(decoder):
        if recovery:
            with qc.if_test((syn, s)):
                for i in range(3):
                    if (recovery >> i) & 1:
                        qc.x(data[i])
    qc.barrier()

    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    if state in ["+", "-"]:
        qc.h(data[0])
    elif state in ["+i", "-i"]:
        qc.sdg(data[0])
        qc.h(data[0])

    qc.measure(data, out)
    return qc


def learn_decoder(joint_counts):
    """Only measured (syndrome, data word) counts enter this function.

    Choose between the two X masks compatible with each syndrome.
    A symmetric pseudocount is used; ties keep the standard decoder.
    """
    table = STANDARD.copy()
    evidence = {}

    for s in range(4):
        candidates = [m for m in range(8) if syndrome(m) == s]
        scores = {
            m: 1 + joint_counts.get((s, m), 0)
            for m in candidates
        }
        best = max(scores.values())
        winners = [m for m in candidates if scores[m] == best]
        table[s] = STANDARD[s] if STANDARD[s] in winners else winners[0]

        evidence[str(s)] = {
            "candidate_counts": {
                str(m): scores[m] - 1 for m in candidates
            },
            "incompatible_observations": sum(
                n for (observed_s, m), n in joint_counts.items()
                if observed_s == s and m not in candidates
            ),
            "chosen_mask": table[s],
        }

    return table, evidence


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()


def prepare(args, phase):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / f"{phase}_plan.json"
    qpy_path = root / f"{phase}.qpy"

    if plan_path.exists() or qpy_path.exists():
        raise RuntimeError(
            f"{phase} already prepared. Use a new --output for a new run."
        )

    if phase == "cal":
        cfg = {
            "backend": args.backend,
            "layout": [int(x) for x in args.layout.split(",")],
            "blocks": args.blocks,
            "cal_shots": args.cal_shots,
            "shots": args.shots,
            "seed": args.seed,
        }
        if len(cfg["layout"]) != 5 or len(set(cfg["layout"])) != 5:
            raise ValueError("Layout must contain five distinct qubits.")
        if min(cfg["blocks"], cfg["cal_shots"], cfg["shots"]) < 1:
            raise ValueError("Blocks and shot counts must be positive.")
        lock = None
    else:
        cal_plan = read(root / "cal_plan.json")
        cfg = cal_plan["config"]
        lock = read(root / "decoder_lock.json")
        if lock["cal_plan_sha256"] != sha(root / "cal_plan.json"):
            raise RuntimeError("Calibration plan differs from decoder lock.")
        if lock["cal_results_sha256"] != sha(root / "cal_results.json"):
            raise RuntimeError("Calibration results differ from decoder lock.")

    backend = service().backend(
        cfg["backend"], use_fractional_gates=False
    )
    if "if_else" not in backend.target.operation_names:
        raise RuntimeError("Backend does not advertise if_else support.")

    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=cfg["layout"],
        optimization_level=1,
        seed_transpiler=cfg["seed"],
    )

    rng = np.random.default_rng(
        cfg["seed"] + (0 if phase == "cal" else 100_000)
    )
    circuits, metadata = [], []

    for env, probabilities in ENVIRONMENTS.items():
        for block in range(cfg["blocks"]):
            states = ["0"] if phase == "cal" else STATES

            for state in states:
                n = cfg["cal_shots"] if phase == "cal" else cfg["shots"]
                allocation = rng.multinomial(n, probabilities)

                methods = ["calibration"] if phase == "cal" else [
                    "standard", "learned"
                ]

                # Both test methods receive the exact same mask allocation.
                for method in methods:
                    table = None
                    if method == "standard":
                        table = STANDARD
                    elif method == "learned":
                        table = lock["decoders"][env]["table"]

                    for mask, shots in enumerate(allocation):
                        if not shots:
                            continue
                        circuits.append(circuit(
                            state, mask, table, calibration=(phase == "cal")
                        ))
                        metadata.append({
                            "environment": env,
                            "block": block,
                            "state": state,
                            "method": method,
                            "injected_mask": mask,
                            "shots": int(shots),
                        })

    # Randomize PUB order. This does not assert physical execution order.
    order = rng.permutation(len(circuits))
    circuits = [circuits[int(i)] for i in order]
    metadata = [metadata[int(i)] for i in order]
    compiled = pm.run(circuits)

    for index, (qc, row) in enumerate(zip(compiled, metadata)):
        initial = list(qc.layout.initial_index_layout(filter_ancillas=True))
        final = list(qc.layout.final_index_layout(filter_ancillas=True))
        if initial != cfg["layout"]:
            raise RuntimeError(f"Unexpected initial layout: {initial}")
        row.update({
            "pub_index": index,
            "initial_layout": initial,
            "final_layout": final,
        })

    with qpy_path.open("wb") as f:
        qpy.dump(compiled, f)

    plan = {
        "phase": phase,
        "created_utc": now(),
        "config": cfg,
        "environments": ENVIRONMENTS,
        "pubs": metadata,
        "qpy_sha256": sha(qpy_path),
        "total_shots": sum(row["shots"] for row in metadata),
    }
    if phase == "test":
        plan["decoder_lock_sha256"] = sha(root / "decoder_lock.json")

    save(plan_path, plan)
    print(f"Prepared {phase}: {len(compiled)} PUBs")
    print(f"Total shots: {plan['total_shots']}")
    print("No job submitted.")


def submit(args, phase):
    from qiskit_ibm_runtime import SamplerV2

    root = args.output
    plan_path = root / f"{phase}_plan.json"
    plan = read(plan_path)
    qpy_path = root / f"{phase}.qpy"
    job_path = root / f"{phase}_job.json"

    if job_path.exists():
        raise RuntimeError(
            "Submission record already exists. Collect the existing job. "
            "If submission was interrupted, inspect IBM Jobs before retrying."
        )
    if sha(qpy_path) != plan["qpy_sha256"]:
        raise RuntimeError("QPY checksum mismatch.")
    if phase == "test":
        if sha(root / "decoder_lock.json") != plan["decoder_lock_sha256"]:
            raise RuntimeError("Decoder lock changed after test preparation.")

    with qpy_path.open("rb") as f:
        circuits = qpy.load(f)
    if len(circuits) != len(plan["pubs"]):
        raise RuntimeError("Circuit/metadata count mismatch.")

    backend = service().backend(
        plan["config"]["backend"], use_fractional_gates=False
    )
    record = {
        "started_utc": now(),
        "plan_sha256": sha(plan_path),
    }
    # Guard against accidental duplicate submission after an interruption.
    save(job_path, record)

    pubs = [
        (qc, None, row["shots"])
        for qc, row in zip(circuits, plan["pubs"])
    ]
    job = SamplerV2(mode=backend).run(pubs)
    job_id = job.job_id()
    print("JOB ID:", job_id, flush=True)
    record["job_id"] = job_id
    save(job_path, record)


def collect(args, phase):
    root = args.output
    plan_path = root / f"{phase}_plan.json"
    plan = read(plan_path)
    record = read(root / f"{phase}_job.json")
    if sha(plan_path) != record["plan_sha256"]:
        raise RuntimeError("Plan changed after submission.")
    if not record.get("job_id"):
        raise RuntimeError(
            "Submission was interrupted before saving job ID. "
            "Inspect IBM Jobs; do not blindly resubmit."
        )

    job = service().job(record["job_id"])
    status = job.status()
    status = getattr(status, "name", str(status))
    print("Job", record["job_id"], status)
    if status != "DONE":
        print("Nothing collected. Run collect again after completion.")
        return

    result_path = root / f"{phase}_results.json"
    if result_path.exists():
        records = read(result_path)
    else:
        result = job.result()
        if len(result) != len(plan["pubs"]):
            raise RuntimeError("Unexpected result count.")
        records = []

        for row, pub in zip(plan["pubs"], result):
            out = pub.data.out.get_bitstrings()
            syn = pub.data.syn.get_bitstrings()
            if len(out) != row["shots"] or len(syn) != row["shots"]:
                raise RuntimeError("Unexpected shot count.")
            joint = defaultdict(int)
            for s, o in zip(syn, out):
                joint[f"{int(s, 2)}:{int(o, 2)}"] += 1
            records.append({**row, "joint_counts": dict(joint)})
        save(result_path, records)

    if phase == "cal":
        lock_path = root / "decoder_lock.json"
        if lock_path.exists():
            print("Decoder already locked; leaving it unchanged.")
            return

        decoders = {}
        # Explicitly discard injection labels before learning.
        for env in ENVIRONMENTS:
            observed = defaultdict(int)
            for row in records:
                if row["environment"] == env:
                    for key, count in row["joint_counts"].items():
                        s, m = map(int, key.split(":"))
                        observed[(s, m)] += count
            table, evidence = learn_decoder(observed)
            decoders[env] = {"table": table, "evidence": evidence}
            print(env, "standard:", STANDARD, "learned:", table)

        save(lock_path, {
            "locked_utc": now(),
            "calibration_job_id": record["job_id"],
            "cal_plan_sha256": sha(plan_path),
            "cal_results_sha256": sha(result_path),
            "decoders": decoders,
        })
        print("Decoder locked. Next: prepare-test.")
        return

    if sha(root / "decoder_lock.json") != plan["decoder_lock_sha256"]:
        raise RuntimeError("Decoder lock changed.")

    totals = defaultdict(lambda: [0, 0, 0])
    for row in records:
        expected = int(row["state"] in ["1", "-", "-i"])
        key = (
            row["environment"], row["block"], row["method"], row["state"]
        )
        for joint, count in row["joint_counts"].items():
            _, out = map(int, joint.split(":"))
            totals[key][0] += count
            totals[key][1] += count * int((out & 1) == expected)
            totals[key][2] += count * int(out == expected)

    rows = []
    for (env, block, method, state), (n, logical, full) in totals.items():
        rows.append({
            "environment": env, "block": block,
            "method": method, "state": state, "shots": n,
            "logical_success": logical / n,
            "decoded_full_success": full / n,
        })
    write_csv(root / "by_state.csv", rows)

    summary = []
    for env in ENVIRONMENTS:
        for method in ["standard", "learned"]:
            selected = [
                r for r in rows
                if r["environment"] == env and r["method"] == method
            ]
            n = sum(r["shots"] for r in selected)
            summary.append({
                "environment": env,
                "method": method,
                "test_shots": n,
                "calibration_shots": (
                    plan["config"]["blocks"] * plan["config"]["cal_shots"]
                    if method == "learned" else 0
                ),
                "logical_success": sum(
                    r["shots"] * r["logical_success"] for r in selected
                ) / n,
                "decoded_full_success": sum(
                    r["shots"] * r["decoded_full_success"] for r in selected
                ) / n,
            })
    write_csv(root / "summary.csv", summary)

    for env in ENVIRONMENTS:
        pair = {r["method"]: r for r in summary if r["environment"] == env}
        standard = pair["standard"]["decoded_full_success"]
        learned = pair["learned"]["decoded_full_success"]
        print(
            f"{env}: standard={standard:.4%}, learned={learned:.4%}, "
            f"delta={100 * (learned - standard):+.3f} percentage points"
        )
    print("Saved by_state.csv and summary.csv.")
    print("Blocks share one job and one decoder; no independent-session CI.")


def selftest():
    """Check bit ordering, syndrome mapping and quantum recovery with Aer."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    correlated_decoder = [0, 1, 3, 2]
    circuits, expected = [], []

    # Deterministic calibration: all eight injected masks.
    for mask in range(8):
        circuits.append(circuit("0", mask, calibration=True))
        expected.append((syndrome(mask), mask))

    # Recovery: six states, eight masks, both lookup tables.
    for table in [STANDARD, correlated_decoder]:
        for state in STATES:
            for mask in range(8):
                circuits.append(circuit(state, mask, table))
                residual = mask ^ table[syndrome(mask)]
                assert residual in [0, 7]
                target = int(state in ["1", "-", "-i"])
                # Residual XXX becomes logical X after decoding.
                flips = residual == 7 and state not in ["+", "-"]
                expected.append((syndrome(mask), target ^ int(flips)))

    sim = AerSimulator()
    compiled = transpile(circuits, sim, optimization_level=0)
    result = sim.run(compiled, shots=16, seed_simulator=240031).result()

    for i, (s, out) in enumerate(expected):
        counts = result.get_counts(i)
        actual = {int(bits.replace(" ", ""), 2) for bits in counts}
        assert actual == {(s << 3) | out}, (i, actual, s, out)

    # Fixed test fixture: independent of runtime experiment configuration.
    correlated_test_probs = [
        0.65, 0.03, 0.03, 0.22,
        0.02, 0.02, 0.02, 0.01,
    ]

    fake_counts = {
        (syndrome(m), m): int(round(100 * p))
        for m, p in enumerate(correlated_test_probs)
    }
    learned, _ = learn_decoder(fake_counts)
    assert learned == correlated_decoder, learned
    print(f"PASS: {len(circuits)} ideal circuits and decoder-learning check.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "selftest",
        "prepare-cal", "submit-cal", "collect-cal",
        "prepare-test", "submit-test", "collect-test",
    ])
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--layout", default="7,8,6,17,5")
    parser.add_argument("--cal-shots", type=int, default=256)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=240031)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/learned_hardware_24"),
    )
    args = parser.parse_args()

    if args.command == "selftest":
        selftest()
        return

    action, phase = args.command.split("-")
    {"prepare": prepare, "submit": submit, "collect": collect}[action](
        args, phase
    )


if __name__ == "__main__":
    main()