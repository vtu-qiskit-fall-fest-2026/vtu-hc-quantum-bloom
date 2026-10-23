#!/usr/bin/env python3
"""M27: ideal-X decoder generalization and calibration sample complexity.

Uses the unchanged M24 learner.
No IBM service calls, no hardware submissions.
Test expectations are computed exactly, not sampled.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

RATIOS = [0.10, 0.30, 0.45, 0.50, 0.55, 0.70, 0.90]
BUDGETS = [32, 128, 256, 1024]
STANDARD = np.array([0, 1, 4, 2], dtype=int)
MASKS = np.arange(8)


def syndrome(mask):
    b0, b1, b2 = [(mask >> i) & 1 for i in range(3)]
    return (b0 ^ b1) + 2 * (b1 ^ b2)


SYNDROMES = np.array([syndrome(m) for m in MASKS])


def distribution(ratio):
    # Order: I, X0, X1, X0X1, X2, X0X2, X1X2, XXX.
    p = np.array([
        .65, .03, .03,
        .24 * ratio,
        .24 * (1 - ratio),
        .02, .02, .01,
    ])
    if not np.isclose(p.sum(), 1):
        raise ValueError("Invalid probability distribution.")
    return p


def load_learner(path):
    spec = importlib.util.spec_from_file_location("qtrace_m24", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.learn_decoder


def exact_pass(p, decoder):
    """Mean full decoded pass probability on the six M24 states.

    Compatible recovery leaves either I or XXX.
    XXX passes the two X-basis states, but fails the other four.
    """
    table = np.asarray(decoder, dtype=int)
    if table.shape != (4,) or np.any((table < 0) | (table > 7)):
        raise ValueError("Invalid decoder.")
    if any(syndrome(int(table[s])) != s for s in range(4)):
        raise ValueError("Recovery is incompatible with its syndrome.")

    residual = MASKS ^ table[SYNDROMES]
    correct_mask_probability = float(p[residual == 0].sum())
    return (1 + 2 * correct_mask_probability) / 3


def oracle_decoder(p):
    # Evaluator only. Never passed to the learner.
    result = []
    for s in range(4):
        candidates = MASKS[SYNDROMES == s]
        best = max(p[m] for m in candidates)
        winners = [
            int(m) for m in candidates
            if abs(p[m] - best) < 1e-12
        ]
        result.append(
            int(STANDARD[s]) if STANDARD[s] in winners else winners[0]
        )
    return result


def selftest(learn):
    # Recovery preference reverses as the distribution changes.
    for ratio, expected_s2 in [(0.1, 4), (0.5, 4), (0.9, 3)]:
        p = distribution(ratio)
        counts = np.rint(10000 * p).astype(int)
        observed = {
            (syndrome(m), m): int(counts[m]) for m in range(8)
        }
        table, _ = learn(observed)
        assert table[2] == expected_s2, (ratio, table)
        assert exact_pass(p, table) <= exact_pass(
            p, oracle_decoder(p)
        ) + 1e-12

    p = distribution(0.5)
    alternative = [0, 1, 3, 2]
    assert np.isclose(exact_pass(p, STANDARD), exact_pass(p, alternative))
    print("PASS: learner interface, tie handling and evaluation checks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=270031)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repeats = 20 if args.smoke else args.repeats
    if repeats < 2:
        parser.error("--repeats must be at least 2.")

    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "24_learned_hardware.py"
    if not source.exists():
        raise FileNotFoundError(source)

    learn = load_learner(source)
    selftest(learn)

    output = args.output or (
        root / "results" /
        ("generalization_27_smoke" if args.smoke else "generalization_27")
    )
    # Preserve previous runs instead of silently replacing them.
    output.mkdir(parents=True, exist_ok=False)

    protocol = {
        "model": "Ideal stochastic X masks; perfect gates and readout",
        "ratios": RATIOS,
        "calibration_budgets": BUDGETS,
        "calibration_repeats_per_case": repeats,
        "seed": args.seed,
        "m24_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "learner": "Unchanged M24 learn_decoder",
        "evaluation": "Exact six-state mean full decoded pass probability",
        "test_shots": 0,
        "oracle_access": "Evaluator only",
        "tie_policy": "Equal-probability choices are not errors",
        "hardware_claim": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    summary = []
    trial_fields = [
        "ratio", "calibration_shots", "repeat", "decoder",
        "standard_pass", "learned_pass", "oracle_pass",
        "delta_pp", "regret_pp", "wrong_syndrome2",
        "harmful_vs_standard",
    ]

    with (output / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=trial_fields)
        writer.writeheader()

        for ratio_index, ratio in enumerate(RATIOS):
            p = distribution(ratio)
            standard_pass = exact_pass(p, STANDARD)
            oracle_pass = exact_pass(p, oracle_decoder(p))

            for budget in BUDGETS:
                rng = np.random.default_rng(
                    np.random.SeedSequence([
                        args.seed, ratio_index, budget
                    ])
                )
                samples = rng.multinomial(budget, p, size=repeats)

                deltas, regrets, wrong, harmful = [], [], [], []
                selected_double = []

                for repeat, counts in enumerate(samples):
                    # Only observed calibration counts reach the learner.
                    observed = {
                        (syndrome(m), m): int(counts[m])
                        for m in range(8)
                    }
                    table, _ = learn(observed)
                    table = list(map(int, table))
                    learned_pass = exact_pass(p, table)

                    delta = 100 * (learned_pass - standard_pass)
                    regret = 100 * (oracle_pass - learned_pass)

                    # At ratio=.5 both choices are optimal for syndrome 2.
                    wrong_s2 = (
                        p[table[2]] < max(p[3], p[4]) - 1e-12
                    )
                    is_harmful = learned_pass < standard_pass - 1e-12

                    deltas.append(delta)
                    regrets.append(max(0.0, regret))
                    wrong.append(wrong_s2)
                    harmful.append(is_harmful)
                    selected_double.append(table[2] == 3)

                    writer.writerow({
                        "ratio": ratio,
                        "calibration_shots": budget,
                        "repeat": repeat,
                        "decoder": json.dumps(table),
                        "standard_pass": standard_pass,
                        "learned_pass": learned_pass,
                        "oracle_pass": oracle_pass,
                        "delta_pp": delta,
                        "regret_pp": max(0.0, regret),
                        "wrong_syndrome2": int(wrong_s2),
                        "harmful_vs_standard": int(is_harmful),
                    })

                summary.append({
                    "ratio": ratio,
                    "calibration_shots": budget,
                    "repeats": repeats,
                    "standard_pass": standard_pass,
                    "mean_learned_pass": standard_pass + np.mean(deltas) / 100,
                    "oracle_pass": oracle_pass,
                    "mean_delta_pp": float(np.mean(deltas)),
                    "mc_se_delta_pp": float(
                        np.std(deltas, ddof=1) / np.sqrt(repeats)
                    ),
                    "mean_regret_pp": float(np.mean(regrets)),
                    "wrong_syndrome2_rate": float(np.mean(wrong)),
                    "choose_X0X1_rate": float(np.mean(selected_double)),
                    "harmful_vs_standard_rate": float(np.mean(harmful)),
                })

                print(
                    f"ratio={ratio:.2f} cal={budget:4d} "
                    f"delta={np.mean(deltas):+.3f} pp "
                    f"wrong_s2={np.mean(wrong):.1%} "
                    f"harmful={np.mean(harmful):.1%}",
                    flush=True,
                )

    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("\nSaved:", output)
    print("Monte Carlo SE describes uncertainty in the simulated mean.")
    print("These are ideal-model results, not hardware predictions.")


if __name__ == "__main__":
    main()