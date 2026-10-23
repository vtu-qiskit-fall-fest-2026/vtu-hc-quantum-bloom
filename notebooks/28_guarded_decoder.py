#!/usr/bin/env python3
"""M28: fixed-threshold Bayesian recovery selection.

Ideal stochastic X errors, perfect gates and readout.
No IBM connection. Evaluation uses exact six-state pass probabilities.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

THRESHOLD = 0.95
BUDGETS = [32, 128, 256, 1024]
RATIOS = [0.20, 0.40, 0.60, 0.80]
STANDARD = np.array([0, 1, 4, 2], dtype=int)
ALTERNATIVE = STANDARD ^ 7


def syndrome(mask):
    b0, b1, b2 = [(mask >> i) & 1 for i in range(3)]
    return (b0 ^ b1) + 2 * (b1 ^ b2)


def load_m24(path):
    spec = importlib.util.spec_from_file_location("m24", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.learn_decoder


@lru_cache(maxsize=100000)
def alternative_confidence(n_alt, n_standard):
    """P(theta > 0.5 | counts), with prior theta ~ Beta(1, 1).

    Posterior: Beta(n_alt + 1, n_standard + 1).
    Integer-parameter identity evaluates the probability without SciPy.
    """
    n_alt, n_standard = int(n_alt), int(n_standard)
    n = n_alt + n_standard + 1
    return sum(comb(n, k) for k in range(n_alt + 1)) / (2 ** n)


def guarded_decoder(counts):
    # Learner receives only observed counts, never true probabilities.
    table = STANDARD.copy()
    confidence = []
    for s in range(4):
        base = int(STANDARD[s])
        alt = int(ALTERNATIVE[s])
        probability = alternative_confidence(
            int(counts[alt]), int(counts[base])
        )
        confidence.append(probability)
        if probability >= THRESHOLD:
            table[s] = alt
    return table, confidence


def make_distribution(target_syndrome, ratio):
    """Move the dominant ambiguous pair between syndromes 1, 2 and 3."""
    p = np.zeros(8)
    p[0], p[7] = .65, .01

    for s in [1, 2, 3]:
        if s == target_syndrome:
            p[STANDARD[s]] = .24 * (1 - ratio)
            p[ALTERNATIVE[s]] = .24 * ratio
        else:
            p[STANDARD[s]] = .03
            p[ALTERNATIVE[s]] = .02

    assert np.isclose(p.sum(), 1)
    return p


def exact_pass(p, table):
    table = np.asarray(table, dtype=int)
    assert table.shape == (4,)
    assert all(syndrome(int(table[s])) == s for s in range(4))

    # Compatible recovery leaves I or XXX.
    # XXX passes the two X-basis states and fails the other four.
    return float((1 + 2 * p[table].sum()) / 3)


def oracle_decoder(p):
    # Privileged evaluator reference, not an input to either learner.
    return np.where(
        p[ALTERNATIVE] > p[STANDARD],
        ALTERNATIVE,
        STANDARD,
    )


def selftest(m24):
    assert alternative_confidence(0, 0) == .5
    assert alternative_confidence(3, 0) == .9375
    assert alternative_confidence(4, 0) == .96875

    for s in range(4):
        counts = np.zeros(8, dtype=int)
        counts[ALTERNATIVE[s]] = 3
        table, _ = guarded_decoder(counts)
        assert table[s] == STANDARD[s]

        counts[ALTERNATIVE[s]] = 4
        table, _ = guarded_decoder(counts)
        assert table[s] == ALTERNATIVE[s]

    # Check that the count-based greedy rule matches the actual M24 learner.
    rng = np.random.default_rng(280001)
    for _ in range(100):
        counts = rng.integers(0, 20, size=8)
        observed = {
            (syndrome(m), m): int(counts[m]) for m in range(8)
        }
        actual, _ = m24(observed)
        expected = np.where(
            counts[ALTERNATIVE] > counts[STANDARD],
            ALTERNATIVE,
            STANDARD,
        )
        assert np.array_equal(actual, expected)

    print("PASS: posterior threshold and M24 learner equivalence.")


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=280071)
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

    m24 = load_m24(source)
    selftest(m24)

    output = args.output or (
        root / "results" /
        ("guarded_28_smoke" if args.smoke else "guarded_28")
    )
    output.mkdir(parents=True, exist_ok=False)

    scenarios = [
        (s, ratio, make_distribution(s, ratio))
        for s in [1, 2, 3]
        for ratio in RATIOS
    ]

    protocol = {
        "model": "Ideal stochastic X; perfect gates and readout",
        "posterior_prior": "Beta(1,1) within each syndrome pair",
        "threshold": THRESHOLD,
        "threshold_selection": "Fixed before M28 evaluation; no tuning",
        "calibration_budgets": BUDGETS,
        "repeats": repeats,
        "seed": args.seed,
        "m24_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "paired_calibration": True,
        "evaluation": "Exact six-state full decoded pass probability",
        "test_shots": 0,
        "standard_calibration_cost": 0,
        "learner_calibration_cost": "The stated budget for each learner",
        "scenarios": [
            {
                "target_syndrome": s,
                "alternative_ratio": ratio,
                "probabilities": p.tolist(),
            }
            for s, ratio, p in scenarios
        ],
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    summaries = []
    comparisons = []

    fields = [
        "target_syndrome", "ratio", "calibration_shots", "repeat",
        "calibration_counts", "greedy_decoder", "guarded_decoder",
        "confidence_by_syndrome", "standard_pass", "oracle_pass",
        "greedy_pass", "guarded_pass",
    ]

    with (output / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for scenario_index, (target, ratio, p) in enumerate(scenarios):
            standard_pass = exact_pass(p, STANDARD)
            oracle_pass = exact_pass(p, oracle_decoder(p))

            for budget in BUDGETS:
                rng = np.random.default_rng(
                    np.random.SeedSequence([
                        args.seed, scenario_index, budget
                    ])
                )
                samples = rng.multinomial(budget, p, size=repeats)

                passes = {"greedy": [], "guarded": []}
                wrong_switch = {"greedy": [], "guarded": []}
                missed_switch = {"greedy": [], "guarded": []}
                switches = {"greedy": [], "guarded": []}

                for repeat, counts in enumerate(samples):
                    observed = {
                        (syndrome(m), m): int(counts[m])
                        for m in range(8)
                    }
                    greedy, _ = m24(observed)
                    greedy = np.asarray(greedy, dtype=int)
                    guarded, confidence = guarded_decoder(counts)

                    values = {}
                    for method, table in [
                        ("greedy", greedy), ("guarded", guarded)
                    ]:
                        value = exact_pass(p, table)
                        values[method] = value
                        passes[method].append(value)

                        changed = table != STANDARD
                        beneficial = (
                            p[ALTERNATIVE] > p[STANDARD] + 1e-12
                        )
                        detrimental = (
                            p[ALTERNATIVE] < p[STANDARD] - 1e-12
                        )

                        wrong_switch[method].append(
                            bool(np.any(changed & detrimental))
                        )
                        missed_switch[method].append(
                            bool(np.any((~changed) & beneficial))
                        )
                        switches[method].append(int(changed.sum()))

                    writer.writerow({
                        "target_syndrome": target,
                        "ratio": ratio,
                        "calibration_shots": budget,
                        "repeat": repeat,
                        "calibration_counts": json.dumps(counts.tolist()),
                        "greedy_decoder": json.dumps(greedy.tolist()),
                        "guarded_decoder": json.dumps(guarded.tolist()),
                        "confidence_by_syndrome": json.dumps(confidence),
                        "standard_pass": standard_pass,
                        "oracle_pass": oracle_pass,
                        "greedy_pass": values["greedy"],
                        "guarded_pass": values["guarded"],
                    })

                for method in ["greedy", "guarded"]:
                    values = np.asarray(passes[method])
                    delta = 100 * (values - standard_pass)

                    summaries.append({
                        "target_syndrome": target,
                        "ratio": ratio,
                        "calibration_shots": budget,
                        "method": method,
                        "repeats": repeats,
                        "standard_pass": standard_pass,
                        "oracle_pass": oracle_pass,
                        "mean_pass": float(values.mean()),
                        "mean_delta_vs_standard_pp": float(delta.mean()),
                        "mc_se_delta_pp": float(
                            delta.std(ddof=1) / np.sqrt(repeats)
                        ),
                        "mean_regret_pp": float(
                            100 * (oracle_pass - values).mean()
                        ),
                        "harmful_vs_standard_rate": float(
                            (values < standard_pass - 1e-12).mean()
                        ),
                        "any_wrong_switch_rate": float(
                            np.mean(wrong_switch[method])
                        ),
                        "any_missed_beneficial_switch_rate": float(
                            np.mean(missed_switch[method])
                        ),
                        "mean_changed_syndromes": float(
                            np.mean(switches[method])
                        ),
                    })

                paired_delta = 100 * (
                    np.asarray(passes["guarded"])
                    - np.asarray(passes["greedy"])
                )
                comparisons.append({
                    "target_syndrome": target,
                    "ratio": ratio,
                    "calibration_shots": budget,
                    "mean_guarded_minus_greedy_pp": float(
                        paired_delta.mean()
                    ),
                    "paired_mc_se_pp": float(
                        paired_delta.std(ddof=1) / np.sqrt(repeats)
                    ),
                    "guarded_better_rate": float(
                        (paired_delta > 1e-10).mean()
                    ),
                    "guarded_worse_rate": float(
                        (paired_delta < -1e-10).mean()
                    ),
                })

                print(
                    f"syndrome={target} ratio={ratio:.2f} cal={budget:4d} "
                    f"guarded-greedy={paired_delta.mean():+.3f} pp",
                    flush=True,
                )

    write_csv(output / "summary.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    print("\nSaved:", output)
    print("Keep all scenarios, including those where guarded loses.")
    print("No hardware performance or global 5% error guarantee is implied.")


if __name__ == "__main__":
    main()