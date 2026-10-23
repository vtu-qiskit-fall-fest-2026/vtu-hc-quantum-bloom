#!/usr/bin/env python3
"""M29: fixed asymmetric-loss decoder comparison.

Ideal stochastic X model. No hardware calls.
Every method is evaluated under the same loss definitions.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
from qtrace.qec.decoder_policies import (
    posterior_moments,
    cost_decoder,
)

STANDARD = np.array([0, 1, 4, 2], dtype=int)
ALTERNATIVE = STANDARD ^ 7
BUDGETS = [32, 128, 256, 1024]
RATIOS = [.25, .45, .55, .75]
METHODS = ["standard", "greedy", "guarded95", "cost2", "cost5"]


def load_module(path):
    spec = importlib.util.spec_from_file_location("m28", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(target, ratio):
    # Different pair masses and ratios from M28.
    p = np.zeros(8)
    p[0], p[7] = .67, .01
    for s in [1, 2, 3]:
        if s == target:
            p[STANDARD[s]] = .18 * (1 - ratio)
            p[ALTERNATIVE[s]] = .18 * ratio
        else:
            p[STANDARD[s]] = .04
            p[ALTERNATIVE[s]] = .03
    assert np.isclose(p.sum(), 1)
    return p


def pass_rate(p, table):
    return float((1 + 2 * p[np.asarray(table)].sum()) / 3)


def loss_components(p, table):
    """Exact loss relative to the per-syndrome oracle.

    wrong: loss from switching away from a better standard action.
    missed: loss from retaining standard when alternative is better.
    Components add across syndromes; they are not net loss vs standard.
    """
    changed = np.asarray(table) != STANDARD
    advantage = (2 / 3) * (p[ALTERNATIVE] - p[STANDARD])
    wrong = float(np.maximum(-advantage, 0)[changed].sum())
    missed = float(np.maximum(advantage, 0)[~changed].sum())
    return wrong, missed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=290071)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repeats = 20 if args.smoke else args.repeats
    if repeats < 2:
        parser.error("--repeats must be at least 2.")

    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "28_guarded_decoder.py"
    m28 = load_module(source)
    if m28.THRESHOLD != .95:
        raise ValueError("This comparison requires the unchanged M28 threshold.")



    # Check symmetry, lambda=1 equivalence and nested switch decisions.
    rng_check = np.random.default_rng(290001)
    for _ in range(200):
        counts = rng_check.integers(0, 100, size=8)
        greedy = np.where(
            counts[ALTERNATIVE] > counts[STANDARD],
            ALTERNATIVE, STANDARD,
        )
        assert np.array_equal(cost_decoder(counts, 1), greedy)
        c2 = cost_decoder(counts, 2) != STANDARD
        c5 = cost_decoder(counts, 5) != STANDARD
        assert np.all(~c5 | c2)
        assert np.all(~c2 | (greedy != STANDARD))

    gain, harm = posterior_moments(0, 0)
    assert np.isclose(gain, .25) and np.isclose(harm, .25)
    print("PASS: symmetric loss, greedy equivalence and penalty ordering.")

    output = args.output or (
        root / "results" /
        ("cost_sensitive_29_smoke" if args.smoke else "cost_sensitive_29")
    )
    output.mkdir(parents=True, exist_ok=False)

    scenarios = [
        (s, ratio, distribution(s, ratio))
        for s in [1, 2, 3] for ratio in RATIOS
    ]
    protocol = {
        "model": "Ideal stochastic X; perfect gates and readout",
        "prior": "Beta(1,1) within each syndrome pair",
        "penalties": [1, 2, 5],
        "penalty_selection": "Fixed before evaluation; no tuning",
        "guarded_threshold": .95,
        "loss_definition": "penalty * wrong_switch_loss + missed_gain",
        "budgets": BUDGETS,
        "repeats": repeats,
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "bit_generator": "PCG64",
        "m28_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "m29_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_shots": 0,
        "calibration_cost": {
            "standard": 0,
            "other_methods": "Stated budget; same observed counts per repeat",
        },
        "scenarios": [
            {"target": s, "ratio": r, "probabilities": p.tolist()}
            for s, r, p in scenarios
        ],
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    summaries = []
    comparisons = []
    fields = ["target", "ratio", "budget", "repeat", "counts", "decoders"]

    with (output / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for scenario_index, (target, ratio, p) in enumerate(scenarios):
            standard_pass = pass_rate(p, STANDARD)
            oracle_pass = float(
                (1 + 2 * np.maximum(p[STANDARD], p[ALTERNATIVE]).sum()) / 3
            )

            for budget in BUDGETS:
                rng = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([
                        args.seed, scenario_index, budget
                    ])
                ))
                samples = rng.multinomial(budget, p, size=repeats)
                metrics = {method: [] for method in METHODS}

                for repeat, counts in enumerate(samples):
                    greedy = cost_decoder(counts, 1)
                    guarded, _ = m28.guarded_decoder(counts)
                    tables = {
                        "standard": STANDARD,
                        "greedy": greedy,
                        "guarded95": guarded,
                        "cost2": cost_decoder(counts, 2),
                        "cost5": cost_decoder(counts, 5),
                    }

                    for method, table in tables.items():
                        value = pass_rate(p, table)
                        wrong, missed = loss_components(p, table)

                        # Lambda=1 loss equals regret in the ideal model.
                        assert np.isclose(
                            wrong + missed, oracle_pass - value
                        )
                        metrics[method].append([value, wrong, missed])

                    writer.writerow({
                        "target": target,
                        "ratio": ratio,
                        "budget": budget,
                        "repeat": repeat,
                        "counts": json.dumps(counts.tolist()),
                        "decoders": json.dumps({
                            name: np.asarray(table).tolist()
                            for name, table in tables.items()
                        }),
                    })

                arrays = {
                    name: np.asarray(values)
                    for name, values in metrics.items()
                }
                for method, data in arrays.items():
                    values, wrong, missed = data.T
                    summaries.append({
                        "target": target,
                        "ratio": ratio,
                        "budget": budget,
                        "method": method,
                        "repeats": repeats,
                        "standard_pass": standard_pass,
                        "oracle_pass": oracle_pass,
                        "mean_pass": float(values.mean()),
                        "delta_vs_standard_pp": float(
                            100 * (values - standard_pass).mean()
                        ),
                        "regret_pp": float(100 * (wrong + missed).mean()),
                        "loss_lambda2_pp": float(
                            100 * (2 * wrong + missed).mean()
                        ),
                        "loss_lambda5_pp": float(
                            100 * (5 * wrong + missed).mean()
                        ),
                        "wrong_switch_loss_pp": float(100 * wrong.mean()),
                        "missed_gain_pp": float(100 * missed.mean()),
                        "harmful_vs_standard_rate": float(
                            (values < standard_pass - 1e-12).mean()
                        ),
                    })

                for method in ["guarded95", "cost2", "cost5"]:
                    paired = 100 * (
                        arrays[method][:, 0] - arrays["greedy"][:, 0]
                    )
                    comparisons.append({
                        "target": target,
                        "ratio": ratio,
                        "budget": budget,
                        "method": method,
                        "delta_vs_greedy_pp": float(paired.mean()),
                        "paired_mc_se_pp": float(
                            paired.std(ddof=1) / np.sqrt(repeats)
                        ),
                    })

                print(
                    f"target={target} ratio={ratio:.2f} budget={budget}: "
                    + ", ".join(
                        f"{name}={100 * data[:, 0].mean():.3f}%"
                        for name, data in arrays.items()
                    ),
                    flush=True,
                )

    write_csv(output / "summary.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    print("\nSaved:", output)
    print("Compare all methods under the SAME lambda.")
    print("Lower weighted loss does not necessarily mean higher pass rate.")


if __name__ == "__main__":
    main()