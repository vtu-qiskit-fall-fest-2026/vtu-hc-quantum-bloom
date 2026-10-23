#!/usr/bin/env python3
"""Run three pre-specified M24 replications and retain every result."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [250031, 250061, 250091]
STATES = ["0", "1", "+", "-", "+i", "-i"]
ENVS = ["single_dominant", "correlated"]

PROJECT = Path(__file__).resolve().parents[1]
M24 = PROJECT / "notebooks" / "24_learned_hardware.py"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_m24(command, folder, *extra):
    cmd = [
        sys.executable, str(M24), command,
        "--output", str(folder), *map(str, extra),
    ]
    print("\n>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)


def ensure_submitted(phase, folder):
    path = folder / f"{phase}_job.json"

    if path.exists():
        record = read_json(path)
        if not record.get("job_id"):
            raise RuntimeError(
                f"{path} exists without a job ID. "
                "Inspect IBM Jobs before doing anything else. "
                "Automatic resubmission is disabled."
            )
        print("Reusing job:", record["job_id"], flush=True)
        return

    run_m24(f"submit-{phase}", folder)


def wait_collect(phase, folder, poll, timeout):
    required = (
        ["cal_results.json", "decoder_lock.json"]
        if phase == "cal"
        else ["test_results.json", "summary.csv", "by_state.csv"]
    )
    deadline = time.monotonic() + timeout

    while not all((folder / name).exists() for name in required):
        run_m24(f"collect-{phase}", folder)

        if all((folder / name).exists() for name in required):
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{phase}: wait limit reached. "
                "Existing job is retained. Rerun this script to resume."
            )

        print(f"Waiting {poll} seconds before checking again.", flush=True)
        time.sleep(poll)


def summarize(root):
    comparisons = []
    all_states = []

    for index, seed in enumerate(SEEDS, start=1):
        folder = root / f"run_{index:02d}"
        summary = read_csv(folder / "summary.csv")
        states = read_csv(folder / "by_state.csv")
        lock = read_json(folder / "decoder_lock.json")
        cal_job = read_json(folder / "cal_job.json")["job_id"]
        test_job = read_json(folder / "test_job.json")["job_id"]

        for env in ENVS:
            methods = {
                r["method"]: r for r in summary
                if r["environment"] == env
            }
            standard = methods["standard"]
            learned = methods["learned"]

            assert int(standard["test_shots"]) == 768
            assert int(learned["test_shots"]) == 768
            assert int(learned["calibration_shots"]) == 256

            comparisons.append({
                "run": index,
                "seed": seed,
                "environment": env,
                "standard_full": float(standard["decoded_full_success"]),
                "learned_full": float(learned["decoded_full_success"]),
                "delta_full_pp": 100 * (
                    float(learned["decoded_full_success"])
                    - float(standard["decoded_full_success"])
                ),
                "delta_logical_pp": 100 * (
                    float(learned["logical_success"])
                    - float(standard["logical_success"])
                ),
                "learned_decoder": json.dumps(
                    lock["decoders"][env]["table"]
                ),
                "calibration_job_id": cal_job,
                "test_job_id": test_job,
            })

            for state in STATES:
                pair = {
                    r["method"]: r for r in states
                    if r["environment"] == env and r["state"] == state
                }
                assert set(pair) == {"standard", "learned"}
                assert all(int(r["shots"]) == 128 for r in pair.values())

                all_states.append({
                    "run": index,
                    "environment": env,
                    "state": state,
                    "standard_full": float(
                        pair["standard"]["decoded_full_success"]
                    ),
                    "learned_full": float(
                        pair["learned"]["decoded_full_success"]
                    ),
                    "delta_full_pp": 100 * (
                        float(pair["learned"]["decoded_full_success"])
                        - float(pair["standard"]["decoded_full_success"])
                    ),
                })

    write_csv(root / "comparisons.csv", comparisons)
    write_csv(root / "state_comparisons.csv", all_states)

    print("\n=== ALL THREE NEW RUNS ===")
    for env in ENVS:
        selected = [r for r in comparisons if r["environment"] == env]
        deltas = [r["delta_full_pp"] for r in selected]
        print(f"\n{env}")
        for row in selected:
            print(
                f"  Run {row['run']}: "
                f"{row['standard_full']:.2%} -> "
                f"{row['learned_full']:.2%}; "
                f"delta={row['delta_full_pp']:+.3f} pp; "
                f"decoder={row['learned_decoder']}"
            )
        print(f"  Mean delta: {sum(deltas) / len(deltas):+.3f} pp")
        print(f"  Observed range: {min(deltas):+.3f} to {max(deltas):+.3f} pp")

    print("\nNo confidence interval inferred from only three runs.")
    print("Pilot M24 is excluded from this summary.")
    print("Saved comparisons.csv and state_comparisons.csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--layout", default="7,8,6,17,5")
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=6)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results" / "confirmation_25",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    root = args.output.resolve()
    if args.summarize_only:
        summarize(root)
        return

    if args.poll < 5 or args.timeout_hours <= 0:
        parser.error("Use --poll >= 5 and a positive timeout.")
    if not M24.exists():
        raise FileNotFoundError(M24)

    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "m24_source_sha256": hashlib.sha256(M24.read_bytes()).hexdigest(),
        "backend": args.backend,
        "layout": args.layout,
        "seeds": SEEDS,
        "blocks_per_run": 1,
        "calibration_shots_per_environment": 256,
        "test_shots_per_state_per_method": 128,
        "primary_metric": "decoded_full_success",
        "primary_environment": "correlated",
        "total_planned_shots": 10752,
        "pilot_included": False,
        "stopping_rule": "Complete all three runs regardless of outcomes.",
    }
    protocol_path = root / "protocol.json"

    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise RuntimeError(
                "Protocol or M24 source changed. "
                "Restore the original configuration before resuming."
            )
    else:
        save_json(protocol_path, protocol)

    # Aer check only; this does not submit a hardware job.
    run_m24("selftest", root)

    for index, seed in enumerate(SEEDS, start=1):
        folder = root / f"run_{index:02d}"
        print(f"\n=== RUN {index}/3 — seed {seed} ===", flush=True)

        if not (folder / "cal_plan.json").exists():
            run_m24(
                "prepare-cal", folder,
                "--backend", args.backend,
                "--layout", args.layout,
                "--cal-shots", 256,
                "--shots", 128,
                "--blocks", 1,
                "--seed", seed,
            )

        ensure_submitted("cal", folder)
        wait_collect(
            "cal", folder, args.poll, args.timeout_hours * 3600
        )

        if not (folder / "test_plan.json").exists():
            run_m24("prepare-test", folder)

        ensure_submitted("test", folder)
        wait_collect(
            "test", folder, args.poll, args.timeout_hours * 3600
        )

    summarize(root)


if __name__ == "__main__":
    main()