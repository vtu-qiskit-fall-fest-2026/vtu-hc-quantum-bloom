#!/usr/bin/env python3
"""M31: low-calibration hardware policy pilot.

Reuses M24 circuit construction and job handling.
No calibration resampling based on policy agreement or test outcomes.
"""

import argparse
import csv
import importlib.util
import platform
from collections import Counter
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

ROOT = Path(__file__).resolve().parents[1]
M24_PATH = ROOT / "notebooks" / "24_learned_hardware.py"

spec = importlib.util.spec_from_file_location("m24", M24_PATH)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

STANDARD = [0, 1, 4, 2]
METHODS = ["standard", "learned", "cost5", "guarded95"]
# "learned" retains the M24 label; it means greedy here.


def distribution(ratio):
    # Target syndrome 1: X0 (mask 1) versus X1X2 (mask 6).
    return [
        .67, .18 * (1 - ratio), .04, .03,
        .04, .03, .18 * ratio, .01,
    ]


ENVIRONMENTS = {
    "s1_standard_favored": distribution(.25),
    "s1_weak_alternative": distribution(.55),
    "s1_strong_alternative": distribution(.75),
}
v.ENVIRONMENTS = ENVIRONMENTS


@lru_cache(maxsize=None)
def confidence(a, b):
    n = a + b + 1
    return sum(comb(n, k) for k in range(a + 1)) / (2 ** n)


def protocol(args):
    return {
        "backend": args.backend,
        "layout": args.layout,
        "seed": 310031,
        "calibration_shots_per_environment": 32,
        "test_shots_per_state_per_method": 128,
        "blocks": 1,
        "environments": ENVIRONMENTS,
        "methods": METHODS,
        "guarded_threshold": .95,
        "cost_penalty": 5,
        "primary_metric": "decoded_full_success",
        "total_planned_shots": 9312,
        "calibration_rule": "One draw; retain all environments and policies",
        "m24_sha256": v.sha(M24_PATH),
        "m31_sha256": v.sha(Path(__file__)),
    }


def check_protocol(args, create=False):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    expected = protocol(args)

    if path.exists():
        if v.read(path) != expected:
            raise RuntimeError("Protocol or source changed. Do not resume.")
    elif create:
        v.save(path, expected)
        v.save(args.output / "versions.json", {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        })
    else:
        raise FileNotFoundError("Run prepare-cal first.")


def lock_policies(args):
    root = args.output
    original_lock = v.read(root / "decoder_lock.json")
    rows = v.read(root / "cal_results.json")

    if v.sha(root / "cal_results.json") != original_lock["cal_results_sha256"]:
        raise RuntimeError("Calibration results changed.")

    tables = {}
    evidence = {}

    for env in ENVIRONMENTS:
        counts = Counter()
        for row in rows:
            if row["environment"] != env:
                continue
            if sum(row["joint_counts"].values()) != row["shots"]:
                raise RuntimeError("Calibration count mismatch.")
            for key, n in row["joint_counts"].items():
                counts[tuple(map(int, key.split(":")))] += n

        if sum(counts.values()) != 32:
            raise RuntimeError(f"Expected 32 calibration shots: {env}")

        policies = {name: STANDARD.copy() for name in METHODS}
        details = []

        # Only measured syndrome/data counts enter these decisions.
        for s, base in enumerate(STANDARD):
            alt = base ^ 7
            a, b = counts[s, alt], counts[s, base]
            probability = confidence(a, b)
            mu = (a + 1) / (a + b + 2)
            gain = max(0.0, 2 * mu * confidence(a + 1, b) - probability)
            harm = max(0.0, gain - (2 * mu - 1))

            if a > b:
                policies["learned"][s] = alt
            if probability >= .95:
                policies["guarded95"][s] = alt
            if gain > 5 * harm + 1e-12:
                policies["cost5"][s] = alt

            details.append({
                "syndrome": s,
                "base_count": b,
                "alternative_count": a,
                "alternative_probability": probability,
            })

        if policies["learned"] != original_lock["decoders"][env]["table"]:
            raise RuntimeError("Greedy decoder disagrees with M24 lock.")

        tables[env] = policies
        evidence[env] = details
        print(env, policies)

    payload = {
        "cal_results_sha256": v.sha(root / "cal_results.json"),
        "decoder_lock_sha256": v.sha(root / "decoder_lock.json"),
        "tables": tables,
        "evidence": evidence,
    }
    path = root / "policy_lock.json"
    if path.exists():
        if v.read(path) != payload:
            raise RuntimeError("Existing policy lock differs.")
    else:
        v.save(path, payload)
    print("All policies locked. No test results used.")


def verify_policies(root):
    lock = v.read(root / "policy_lock.json")
    if v.sha(root / "cal_results.json") != lock["cal_results_sha256"]:
        raise RuntimeError("Calibration changed after policy lock.")
    if v.sha(root / "decoder_lock.json") != lock["decoder_lock_sha256"]:
        raise RuntimeError("Original decoder lock changed.")
    return lock


def prepare_test(args):
    root = args.output
    plan_path = root / "test_plan.json"
    qpy_path = root / "test.qpy"
    if plan_path.exists() or qpy_path.exists():
        raise RuntimeError("Test already prepared; do not regenerate.")

    lock = verify_policies(root)
    cfg = v.read(root / "cal_plan.json")["config"]
    backend = v.service().backend(
        cfg["backend"], use_fractional_gates=False
    )
    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=cfg["layout"],
        optimization_level=1,
        seed_transpiler=cfg["seed"],
    )
    rng = np.random.default_rng(cfg["seed"] + 100_000)
    circuits, metadata = [], []

    for env, probabilities in ENVIRONMENTS.items():
        for state in v.STATES:
            allocation = rng.multinomial(cfg["shots"], probabilities)

            # Same injected-mask allocation across all four policies.
            # Equal policies are retained, not selected or discarded.
            for method in METHODS:
                table = lock["tables"][env][method]
                for mask, n in enumerate(allocation):
                    if n == 0:
                        continue
                    circuits.append(v.circuit(state, mask, table))
                    metadata.append({
                        "environment": env,
                        "block": 0,
                        "state": state,
                        "method": method,
                        "injected_mask": mask,
                        "shots": int(n),
                    })

    order = rng.permutation(len(circuits))
    circuits = [circuits[int(i)] for i in order]
    metadata = [metadata[int(i)] for i in order]
    compiled = pm.run(circuits)

    for i, (qc, row) in enumerate(zip(compiled, metadata)):
        initial = list(qc.layout.initial_index_layout(filter_ancillas=True))
        if initial != cfg["layout"]:
            raise RuntimeError("Initial layout mismatch.")
        row.update({
            "pub_index": i,
            "initial_layout": initial,
            "final_layout": list(
                qc.layout.final_index_layout(filter_ancillas=True)
            ),
        })

    with qpy_path.open("wb") as file:
        qpy.dump(compiled, file)

    v.save(plan_path, {
        "phase": "test",
        "created_utc": v.now(),
        "config": cfg,
        "environments": ENVIRONMENTS,
        "pubs": metadata,
        "qpy_sha256": v.sha(qpy_path),
        "decoder_lock_sha256": v.sha(root / "decoder_lock.json"),
        "policy_lock_sha256": v.sha(root / "policy_lock.json"),
        "total_shots": sum(row["shots"] for row in metadata),
    })
    print(f"Prepared {len(compiled)} test PUBs, 9216 test shots.")
    print("No job submitted.")


def summarize_all_policies(root):
    # M24 collect writes by_state.csv for every observed method.
    with (root / "by_state.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))

    summary = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            selected = [
                row for row in rows
                if row["environment"] == env and row["method"] == method
            ]
            if len(selected) != 6:
                raise RuntimeError("Expected all six states for every policy.")
            n = sum(int(row["shots"]) for row in selected)
            if n != 768:
                raise RuntimeError("Unexpected test shot count.")

            summary.append({
                "environment": env,
                "method": "greedy" if method == "learned" else method,
                "test_shots": n,
                # Same calibration dataset is reused by the three learners.
                # Actual experiment spends 32, not 3*32, per environment.
                "calibration_cost_if_deployed_alone": (
                    0 if method == "standard" else 32
                ),
                "logical_success": sum(
                    int(r["shots"]) * float(r["logical_success"])
                    for r in selected
                ) / n,
                "decoded_full_success": sum(
                    int(r["shots"]) * float(r["decoded_full_success"])
                    for r in selected
                ) / n,
            })

    v.write_csv(root / "policy_summary.csv", summary)
    for env in ENVIRONMENTS:
        print("\n", env)
        for row in summary:
            if row["environment"] == env:
                print(
                    f"  {row['method']:10s} "
                    f"full={row['decoded_full_success']:.3%}"
                )
    print("\nSaved policy_summary.csv for all four policies.")
    print("Single-calibration pilot; no across-calibration confidence interval.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "selftest", "prepare-cal", "submit-cal", "collect-cal",
        "prepare-test", "submit-test", "collect-test",
    ])
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--layout", default="7,8,6,17,5")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "policy_hardware_31",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()

    if args.command == "selftest":
        v.selftest()
        assert confidence(3, 0) < .95
        assert confidence(4, 0) >= .95
        assert all(np.isclose(sum(p), 1) for p in ENVIRONMENTS.values())
        print("PASS: M31 policy threshold and distributions.")
        return

    check_protocol(args, create=(args.command == "prepare-cal"))
    args.seed = 310031
    args.blocks = 1
    args.cal_shots = 32
    args.shots = 128

    if args.command == "prepare-cal":
        v.prepare(args, "cal")
    elif args.command == "submit-cal":
        v.submit(args, "cal")
    elif args.command == "collect-cal":
        v.collect(args, "cal")
        if (args.output / "decoder_lock.json").exists():
            lock_policies(args)
    elif args.command == "prepare-test":
        prepare_test(args)
    else:
        verify_policies(args.output)
        plan = v.read(args.output / "test_plan.json")
        if v.sha(args.output / "policy_lock.json") != plan["policy_lock_sha256"]:
            raise RuntimeError("Policy lock changed after preparation.")

        if args.command == "submit-test":
            v.submit(args, "test")
        else:
            v.collect(args, "test")
            if (args.output / "by_state.csv").exists():
                summarize_all_policies(args.output)


if __name__ == "__main__":
    main()